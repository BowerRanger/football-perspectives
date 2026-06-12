"""Ball stage: detection, automatic anchoring, and a piecewise-physical
3D trajectory solve.

Run flow per shot:

1. **Detect** — iterate the clip frames through the configured
   :class:`BallDetector` (WASB), bridging short gaps by appearance
   template matching, and smooth with the 2-mode IMM
   :class:`BallTracker`. Manual anchor pixels are injected as
   detections. The raw observations are persisted to a
   ``*_ball_observations.json`` sidecar.
2. **Player context** — forward-kinematics world+pixel positions of
   every player's contact joints from ``refined_poses`` (fallback
   ``hmr_world``), via :class:`PlayerContext`.
3. **Auto events** — velocity breaks on the pixel track classified as
   player touches, bounces, goal impacts or stationary spans
   (:func:`detect_events`).
4. **Auto anchors** — events plus confidently-grounded samples become
   validated :class:`BallAnchor` records
   (:func:`generate_auto_anchors`), persisted to
   ``*_ball_anchors_auto.json`` and merged with the operator's manual
   anchors — manual always wins.
5. **Solve** — merged hard-knot anchors are resolved to world positions
   (goal geometry, SMPL bone on the clicked ray, ground ray-cast) and
   become :class:`TrajectoryNode`s for the piecewise solver
   (:func:`solve_piecewise`): endpoint-exact rolling, two-knot gravity
   arcs, bounce restitution checks, split-and-retry at velocity breaks.
6. **Emit** — dense ``BallTrack`` (schema unchanged), sparse keyframes
   sidecar, and a diagnostics sidecar consumed by the quality report.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import cv2
import numpy as np

from src.pipeline.base import BaseStage
from src.schemas.ball_anchor import BallAnchor, BallAnchorSet
from src.schemas.ball_track import BallFrame, BallTrack
from src.schemas.camera_track import CameraTrack
from src.schemas.shots import ShotsManifest
from src.utils.ball_anchor_heights import (
    AIRBORNE_STATES,
    HARD_KNOT_STATES,
    airborne_bucket_range,
    state_to_height,
)
from src.utils.ball_appearance_bridge import (
    AppearanceBridge,
    AppearanceBridgeCfg,
)
from src.utils.ball_auto_anchor import (
    AutoAnchorCfg,
    auto_anchor_path,
    generate_auto_anchors,
    merge_anchors,
)
from src.utils.ball_auto_events import AutoEventCfg, detect_events
from src.utils.ball_detector import BallDetector, YOLOBallDetector
from src.utils.ball_keyframe_builder import build_ball_keyframe_set
from src.utils.ball_piecewise_solver import (
    SolverCfg,
    TrajectoryNode,
    solve_piecewise,
)
from src.utils.ball_player_context import PlayerContext
from src.utils.ball_tracker import BallTracker, TrackerStep
from src.utils.camera_projection import (
    point_to_pixel_ray_distance,
    project_point_onto_pixel_ray,
    project_world_to_image,
)
from src.utils.foot_anchor import ankle_ray_to_pitch
from src.utils.goal_geometry import GoalGeometry, resolve_goal_impact_world

logger = logging.getLogger(__name__)

# Kept as module-level names: the ray-faithfulness tests (and the C1/C4
# behaviours they pin) exercise these directly.
_project_point_onto_pixel_ray = project_point_onto_pixel_ray
_snap_world_onto_pixel_ray = project_point_onto_pixel_ray


def _build_detector(cfg: dict) -> BallDetector:
    """Construct a BallDetector from the ``ball.detector`` config key."""
    backend = str(cfg.get("detector", "yolo")).strip().lower()
    if backend == "wasb":
        from src.utils.ball_detector import WASBBallDetector  # lazy import
        wasb_cfg = cfg.get("wasb", {})
        return WASBBallDetector(
            checkpoint=wasb_cfg.get("checkpoint"),
            confidence=float(wasb_cfg.get("confidence", 0.3)),
            input_size=tuple(wasb_cfg.get("input_size", (512, 288))),
        )
    if backend == "yolo":
        return YOLOBallDetector(
            model_name=cfg.get("yolo_model", "yolov8n.pt"),
            confidence=float(cfg.get("confidence_threshold", 0.3)),
        )
    raise ValueError(f"Unknown ball.detector backend: {backend!r}")


def _load_ball_anchors(
    output_dir: Path, shot_id: str
) -> dict[int, BallAnchor]:
    """Load per-frame manual ball anchors keyed by frame index."""
    if shot_id:
        path = output_dir / "ball" / f"{shot_id}_ball_anchors.json"
    else:
        path = output_dir / "ball" / "ball_anchors.json"
    if not path.exists():
        return {}
    try:
        aset = BallAnchorSet.load(path)
    except Exception as exc:
        logger.warning("ball stage: failed to load anchors at %s: %s", path, exc)
        return {}
    return {a.frame: a for a in aset.anchors}


# Player_touch ground/air classification by surrounding anchors. The
# ball is at ground level at a touch UNLESS the touch sits between two
# airborne-implying anchors (volley between airborne anchors, header in
# a bounce chain). See the per-set notes for the asymmetric kick rule.
_PREV_AIRBORNE_STATES = frozenset({
    "airborne_low", "airborne_mid", "airborne_high",
    "header", "volley", "chest", "catch", "off_screen_flight",
    "bounce", "kick", "goal_impact",
})
_NEXT_AIRBORNE_STATES = frozenset({
    "airborne_low", "airborne_mid", "airborne_high",
    "header", "volley", "chest", "catch", "off_screen_flight",
    "bounce", "goal_impact",
})


def _classify_ground_touches(
    anchor_by_frame: dict[int, BallAnchor],
) -> set[int]:
    """Frames whose ``player_touch`` anchor is a ground-level contact."""
    sorted_frames = sorted(anchor_by_frame.keys())

    def _neighbor_implies_flight(
        idx: int, step: int, airborne_set: frozenset[str]
    ) -> bool:
        j = idx + step
        while 0 <= j < len(sorted_frames):
            anc_j = anchor_by_frame[sorted_frames[j]]
            if anc_j.state != "player_touch":
                return anc_j.state in airborne_set
            j += step
        return False

    ground_touch_frames: set[int] = set()
    for idx, fi in enumerate(sorted_frames):
        anc = anchor_by_frame[fi]
        if anc.state != "player_touch":
            continue
        prev_flight = _neighbor_implies_flight(idx, -1, _PREV_AIRBORNE_STATES)
        next_flight = _neighbor_implies_flight(idx, +1, _NEXT_AIRBORNE_STATES)
        if not (prev_flight and next_flight):
            ground_touch_frames.add(fi)
    return ground_touch_frames


def _resolve_anchor_world(
    *,
    anc: BallAnchor,
    fi: int,
    ground_touch_frames: set[int],
    player_ctx: PlayerContext,
    per_frame_K: dict[int, np.ndarray],
    per_frame_R: dict[int, np.ndarray],
    per_frame_t: dict[int, np.ndarray],
    distortion: tuple[float, float],
    ball_radius: float,
    goal_geometry: GoalGeometry,
) -> np.ndarray | None:
    """Single source of truth for resolving a hard-knot anchor to its
    world position.

    Rules:
      • ``goal_impact`` → intersect clicked-pixel ray with the goal
        element geometry; fallback (ray parallel to surface) is the
        ray-cast at the goal-impact canonical height.
      • ``player_touch`` ground-touch → clicked-pixel ray-cast at
        z = ball_radius (bone XY drifts with monocular HMR depth).
      • ``player_touch`` airborne → SMPL bone projected onto the
        clicked-pixel ray (C1: click is authoritative for lateral,
        the player provides depth). Fallback is the ray-cast at the
        player_touch default height.
      • All other hard-knot states → ray-cast at the state's canonical
        height.
    """
    if anc.image_xy is None:
        return None
    K = per_frame_K.get(fi)
    R = per_frame_R.get(fi)
    t = per_frame_t.get(fi)
    if K is None or R is None or t is None:
        return None
    uv = (float(anc.image_xy[0]), float(anc.image_xy[1]))

    if anc.state == "goal_impact" and anc.goal_element is not None:
        try:
            return np.asarray(
                resolve_goal_impact_world(
                    uv, anc.goal_element,
                    K=K, R=R, t=t,
                    distortion=distortion, geometry=goal_geometry,
                ),
                dtype=float,
            )
        except Exception as exc:
            logger.debug(
                "ball goal_impact resolver failed at frame %d (%s): %s",
                fi, anc.goal_element, exc,
            )
            # Fall through to the ray-cast fallback below.

    if anc.state == "player_touch" and fi not in ground_touch_frames:
        if anc.player_id and anc.bone:
            bone_world = player_ctx.joint_world(fi, anc.player_id, anc.bone)
            if bone_world is not None:
                return project_point_onto_pixel_ray(
                    np.asarray(bone_world, dtype=float), uv,
                    K, R, t, distortion,
                )
        # Fall through to the fallback ray-cast at z=1.0 below.

    if anc.state == "player_touch" and fi in ground_touch_frames:
        plane_z = ball_radius
    else:
        try:
            plane_z = state_to_height(anc.state)
        except ValueError:
            plane_z = ball_radius
    try:
        return np.asarray(
            ankle_ray_to_pitch(
                uv, K=K, R=R, t=t,
                plane_z=plane_z, distortion=distortion,
            ),
            dtype=float,
        )
    except Exception as exc:
        logger.debug("ball anchor projection failed at frame %d: %s", fi, exc)
        return None


def _emit_ball_keyframes(
    *,
    ball_out_path: Path,
    clip_id: str,
    fps: float,
    image_size: tuple[int, int],
    per_frame_out: list[BallFrame],
    anchor_by_frame: dict[int, BallAnchor],
    per_frame_K: dict[int, np.ndarray],
    per_frame_R: dict[int, np.ndarray],
    per_frame_t: dict[int, np.ndarray],
    distortion: tuple[float, float],
    ground_touch_frames: set[int],
) -> None:
    """Write the sparse ``*_ball_keyframes.json`` sidecar next to the dense
    track. ``world_xyz`` for each anchor is taken from the already-built
    dense ``per_frame_out`` so the two artifacts agree exactly.
    """
    world_by_frame = {
        bf.frame: bf.world_xyz
        for bf in per_frame_out
        if bf.frame in anchor_by_frame
    }
    kfset = build_ball_keyframe_set(
        clip_id=clip_id,
        fps=fps,
        image_size=image_size,
        anchor_by_frame=anchor_by_frame,
        world_by_frame=world_by_frame,
        per_frame_K=per_frame_K,
        per_frame_R=per_frame_R,
        per_frame_t=per_frame_t,
        distortion=distortion,
        ground_touch_frames=ground_touch_frames,
    )
    kf_path = ball_out_path.with_name(
        ball_out_path.name.replace("ball_track", "ball_keyframes")
    )
    kfset.save(kf_path)
    logger.debug("ball: wrote %d keyframes to %s", len(kfset.keyframes), kf_path)


def _write_observations_sidecar(
    path: Path,
    clip_id: str,
    fps: float,
    steps: list[TrackerStep],
    confidences: dict[int, float],
    sources: dict[int, str],
) -> None:
    """Persist the raw detection/tracking pass so re-solves and the
    dashboard don't need to re-run the detector."""
    payload = {
        "clip_id": clip_id,
        "fps": fps,
        "frames": [
            {
                "frame": s.frame,
                "uv": (
                    [float(s.uv[0]), float(s.uv[1])]
                    if s.uv is not None else None
                ),
                "confidence": float(confidences.get(s.frame, 0.0)),
                "p_flight": float(s.p_flight),
                "gap_fill": bool(s.is_gap_fill),
                "source": sources.get(s.frame, "none"),
            }
            for s in steps
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def _auto_event_cfg(auto_cfg: dict) -> AutoEventCfg:
    base = AutoEventCfg()
    return AutoEventCfg(
        touch_max_px=float(auto_cfg.get("touch_max_px", base.touch_max_px)),
        min_direction_change_deg=float(auto_cfg.get(
            "min_direction_change_deg", base.min_direction_change_deg)),
        min_speed_change_px=float(auto_cfg.get(
            "min_speed_change_px", base.min_speed_change_px)),
        event_window_frames=int(auto_cfg.get(
            "event_window_frames", base.event_window_frames)),
        merge_window_frames=int(auto_cfg.get(
            "merge_window_frames", base.merge_window_frames)),
        goal_line_tolerance_m=float(auto_cfg.get(
            "goal_line_tolerance_m", base.goal_line_tolerance_m)),
        goal_net_speed_drop_ratio=float(auto_cfg.get(
            "goal_net_speed_drop_ratio", base.goal_net_speed_drop_ratio)),
    )


def _auto_anchor_cfg(auto_cfg: dict, ball_radius: float) -> AutoAnchorCfg:
    base = AutoAnchorCfg()
    return AutoAnchorCfg(
        enabled=bool(auto_cfg.get("enabled", True)),
        min_event_score=float(auto_cfg.get(
            "min_event_score", base.min_event_score)),
        grounded_interval=int(auto_cfg.get(
            "grounded_interval", base.grounded_interval)),
        grounded_min_conf=float(auto_cfg.get(
            "grounded_min_conf", base.grounded_min_conf)),
        contact_max_gap_m=float(auto_cfg.get(
            "contact_max_gap_m", base.contact_max_gap_m)),
        shot_speed_px=float(auto_cfg.get(
            "shot_speed_px", base.shot_speed_px)),
        suppress_radius_frames=int(auto_cfg.get(
            "suppress_radius_frames", base.suppress_radius_frames)),
        ball_radius_m=ball_radius,
    )


def _solver_cfg(cfg: dict, ball_radius: float) -> SolverCfg:
    base = SolverCfg()
    physics = cfg.get("physics", {})
    plaus = cfg.get("plausibility", {})
    spin = cfg.get("spin", {})
    tracker = cfg.get("tracker", {})
    return SolverCfg(
        ball_radius_m=ball_radius,
        ground_z_tol_m=float(physics.get("ground_z_tol_m", base.ground_z_tol_m)),
        rolling_max_residual_px=float(physics.get(
            "rolling_max_residual_px", base.rolling_max_residual_px)),
        rolling_decel_max_m_s2=float(physics.get(
            "rolling_decel_max_m_s2", base.rolling_decel_max_m_s2)),
        flight_max_residual_px=float(cfg.get(
            "flight_max_residual_px", base.flight_max_residual_px)),
        max_splits_per_span=int(physics.get(
            "max_splits_per_span", base.max_splits_per_span)),
        min_flight_frames=int(tracker.get(
            "min_flight_frames", base.min_flight_frames)),
        restitution_min=float(physics.get(
            "restitution_min", base.restitution_min)),
        restitution_max=float(physics.get(
            "restitution_max", base.restitution_max)),
        z_max_m=float(plaus.get("z_max_m", base.z_max_m)),
        horizontal_speed_max_m_s=float(plaus.get(
            "horizontal_speed_max_m_s", base.horizontal_speed_max_m_s)),
        pitch_margin_m=float(plaus.get("pitch_margin_m", base.pitch_margin_m)),
        spin_enabled=bool(spin.get("enabled", base.spin_enabled)),
        spin_min_seconds=float(spin.get(
            "min_flight_seconds", base.spin_min_seconds)),
        spin_min_improve=float(spin.get(
            "min_residual_improvement", base.spin_min_improve)),
        spin_min_improve_hinted=float(spin.get(
            "min_residual_improvement_with_hint", base.spin_min_improve_hinted)),
        spin_max_omega_rad_s=float(spin.get(
            "max_omega_rad_s", base.spin_max_omega_rad_s)),
        drag_k_over_m=float(spin.get("drag_k_over_m", base.drag_k_over_m)),
    )


class BallStage(BaseStage):
    name = "ball"

    def __init__(
        self,
        config: dict,
        output_dir: Path,
        ball_detector: BallDetector | None = None,
        **_,
    ) -> None:
        super().__init__(config, output_dir)
        self.ball_detector = ball_detector

    def is_complete(self) -> bool:
        manifest_path = self.output_dir / "shots" / "shots_manifest.json"
        if not manifest_path.exists():
            return (self.output_dir / "ball" / "ball_track.json").exists()
        manifest = ShotsManifest.load(manifest_path)
        return all(
            (self.output_dir / "ball" / f"{shot.id}_ball_track.json").exists()
            for shot in manifest.active_shots()
        )

    def run(self) -> None:
        cfg = self.config.get("ball", {})
        detector = self.ball_detector if self.ball_detector is not None else _build_detector(cfg)

        manifest_path = self.output_dir / "shots" / "shots_manifest.json"
        if not manifest_path.exists():
            # Legacy single-shot path. Use the unprefixed file names.
            cam_path = self.output_dir / "camera" / "camera_track.json"
            ball_out = self.output_dir / "ball" / "ball_track.json"
            if not cam_path.exists():
                raise FileNotFoundError(
                    f"ball stage requires manifest at {manifest_path}; run prepare_shots first"
                )
            clip_path = self._guess_legacy_clip()
            self._run_shot("", clip_path, cam_path, ball_out, cfg, detector)
            return

        manifest = ShotsManifest.load(manifest_path)
        shot_filter = getattr(self, "shot_filter", None)
        for shot in manifest.active_shots():
            if shot_filter is not None and shot.id != shot_filter:
                continue
            cam_path = self.output_dir / "camera" / f"{shot.id}_camera_track.json"
            ball_out = self.output_dir / "ball" / f"{shot.id}_ball_track.json"
            if not cam_path.exists():
                logger.warning(
                    "ball stage skipping shot %s — no camera_track at %s",
                    shot.id, cam_path,
                )
                continue
            clip_path = self.output_dir / shot.clip_file
            if not clip_path.exists():
                logger.warning(
                    "ball stage skipping shot %s — clip missing at %s",
                    shot.id, clip_path,
                )
                continue
            self._run_shot(shot.id, clip_path, cam_path, ball_out, cfg, detector)

    def _guess_legacy_clip(self) -> Path:
        """Find a clip file under shots/ for the legacy no-manifest path."""
        shots_dir = self.output_dir / "shots"
        candidates = sorted(shots_dir.glob("*.mp4")) if shots_dir.exists() else []
        if not candidates:
            raise FileNotFoundError(
                f"ball stage: no clip files found under {shots_dir}"
            )
        return candidates[0]

    # ------------------------------------------------------------------

    def _detect_loop(
        self,
        clip_path: Path,
        cfg: dict,
        detector: BallDetector,
        anchor_by_frame: dict[int, BallAnchor],
    ) -> tuple[list[TrackerStep], dict[int, float], dict[int, str]]:
        """Per-frame detection + appearance bridging + IMM smoothing."""
        tracker_cfg = cfg.get("tracker", {})
        tracker = BallTracker(
            process_noise_grounded_px=float(tracker_cfg.get("process_noise_grounded_px", 4.0)),
            process_noise_flight_px=float(tracker_cfg.get("process_noise_flight_px", 12.0)),
            measurement_noise_px=float(tracker_cfg.get("measurement_noise_px", 2.0)),
            gating_sigma=float(tracker_cfg.get("gating_sigma", 4.0)),
            max_gap_frames=int(cfg.get("max_gap_frames", 6)),
            initial_p_flight=float(tracker_cfg.get("initial_p_flight", 0.1)),
        )
        bridge_cfg = AppearanceBridgeCfg(
            enabled=bool(cfg.get("appearance_bridge", {}).get("enabled", True)),
            max_gap_frames=int(cfg.get("appearance_bridge", {}).get("max_gap_frames", 8)),
            template_size_px=int(cfg.get("appearance_bridge", {}).get("template_size_px", 32)),
            search_radius_px=int(cfg.get("appearance_bridge", {}).get("search_radius_px", 64)),
            min_ncc=float(cfg.get("appearance_bridge", {}).get("min_ncc", 0.6)),
            template_max_age_frames=int(cfg.get("appearance_bridge", {}).get("template_max_age_frames", 30)),
            template_update_confidence=float(cfg.get("appearance_bridge", {}).get("template_update_confidence", 0.5)),
        )
        bridge = AppearanceBridge(bridge_cfg)
        consecutive_misses = 0

        steps: list[TrackerStep] = []
        raw_confidences: dict[int, float] = {}
        sources: dict[int, str] = {}
        cap = cv2.VideoCapture(str(clip_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open clip: {clip_path}")
        try:
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                anchor = anchor_by_frame.get(frame_idx)
                if anchor is not None:
                    if anchor.state == "off_screen_flight":
                        # No pixel; let the IMM predict.
                        uv: tuple[float, float] | None = None
                    else:
                        uv = (float(anchor.image_xy[0]), float(anchor.image_xy[1]))
                        raw_confidences[frame_idx] = 1.0
                        sources[frame_idx] = "anchor"
                        bridge.update_template(
                            frame=frame_idx, frame_image=frame,
                            uv=uv, confidence=1.0,
                        )
                    consecutive_misses = 0
                else:
                    det = detector.detect(frame)
                    if det is None:
                        consecutive_misses += 1
                        bridge_result = bridge.try_bridge(
                            frame=frame_idx,
                            frame_image=frame,
                            predicted_uv=(
                                (float(steps[-1].uv[0]), float(steps[-1].uv[1]))
                                if steps and steps[-1].uv is not None else None
                            ),
                            consecutive_misses=consecutive_misses,
                        )
                        if bridge_result is None:
                            uv = None
                        else:
                            uv, bridged_conf = bridge_result
                            raw_confidences[frame_idx] = bridged_conf
                            sources[frame_idx] = "bridge"
                    else:
                        consecutive_misses = 0
                        uv = (float(det[0]), float(det[1]))
                        raw_confidences[frame_idx] = float(det[2])
                        sources[frame_idx] = "detector"
                        bridge.update_template(
                            frame=frame_idx,
                            frame_image=frame,
                            uv=uv,
                            confidence=float(det[2]),
                        )
                step = tracker.update(frame_idx, uv)
                # The IMM smooths and lags the pixel track (visibly so
                # right after a kick); fits must see the raw measurement.
                # Keep the tracker's uv only where it bridges a miss.
                if uv is not None and not step.is_outlier:
                    step = TrackerStep(
                        frame=step.frame, uv=uv, p_flight=step.p_flight,
                        is_outlier=step.is_outlier,
                        is_gap_fill=step.is_gap_fill,
                    )
                steps.append(step)
                frame_idx += 1
        finally:
            cap.release()
        return steps, raw_confidences, sources

    def _run_shot(
        self,
        shot_id: str,
        clip_path: Path,
        camera_path: Path,
        ball_out_path: Path,
        cfg: dict,
        detector: BallDetector,
    ) -> None:
        camera = CameraTrack.load(camera_path)
        per_frame_K = {f.frame: np.array(f.K) for f in camera.frames}
        per_frame_R = {f.frame: np.array(f.R) for f in camera.frames}
        t_world_fallback = np.array(camera.t_world)
        per_frame_t = {
            f.frame: (np.array(f.t) if f.t is not None else t_world_fallback)
            for f in camera.frames
        }
        distortion = camera.distortion
        n_frames = max(per_frame_K) + 1 if per_frame_K else 0

        ball_radius = float(cfg.get("ball_radius_m", 0.11))
        pitch_cfg = self.config.get("pitch", {})
        goal_geometry = GoalGeometry.from_pitch_config(pitch_cfg)
        auto_cfg_dict = cfg.get("auto_anchors", {})
        event_cfg = _auto_event_cfg(auto_cfg_dict)
        anchor_cfg = _auto_anchor_cfg(auto_cfg_dict, ball_radius)
        solver_cfg = _solver_cfg(cfg, ball_radius)

        manual_by_frame = _load_ball_anchors(self.output_dir, shot_id)
        if manual_by_frame:
            logger.info(
                "ball stage: loaded %d manual anchors for shot %s",
                len(manual_by_frame), shot_id or "(legacy)",
            )

        # --- 1. Detect ------------------------------------------------
        steps, raw_confidences, sources = self._detect_loop(
            clip_path, cfg, detector, manual_by_frame,
        )
        if not steps:
            logger.warning("ball stage: clip %s contained no frames", clip_path)
            return
        n_frames = max(n_frames, steps[-1].frame + 1)
        try:
            _write_observations_sidecar(
                ball_out_path.with_name(ball_out_path.name.replace(
                    "ball_track", "ball_observations")),
                camera.clip_id, camera.fps, steps, raw_confidences, sources,
            )
        except Exception as exc:  # noqa: BLE001 — sidecar is enrichment
            logger.warning("ball: failed to write observations sidecar: %s", exc)

        # --- 2. Player context -----------------------------------------
        player_ctx = PlayerContext.load(
            self.output_dir, shot_id,
            per_frame_K=per_frame_K, per_frame_R=per_frame_R,
            per_frame_t=per_frame_t, distortion=distortion,
        )
        if not player_ctx.player_ids:
            logger.warning(
                "ball stage: no player tracks found for shot %s — automatic "
                "touch/goal detection degraded to ball-only evidence",
                shot_id or "(legacy)",
            )

        # --- 3+4. Auto events -> auto anchors ---------------------------
        events = detect_events(
            steps=steps,
            confidences=raw_confidences,
            player_ctx=player_ctx,
            per_frame_K=per_frame_K, per_frame_R=per_frame_R,
            per_frame_t=per_frame_t, distortion=distortion,
            goal_geometry=goal_geometry,
            cfg=event_cfg,
        )
        auto_by_frame: dict[int, BallAnchor] = {}
        if anchor_cfg.enabled:
            try:
                auto_anchors = generate_auto_anchors(
                    events=events, steps=steps, confidences=raw_confidences,
                    player_ctx=player_ctx,
                    per_frame_K=per_frame_K, per_frame_R=per_frame_R,
                    per_frame_t=per_frame_t, distortion=distortion,
                    fps=camera.fps, pitch_cfg=pitch_cfg, cfg=anchor_cfg,
                )
                auto_by_frame = {a.frame: a for a in auto_anchors}
                BallAnchorSet(
                    clip_id=camera.clip_id,
                    image_size=camera.image_size,
                    anchors=auto_anchors,
                ).save(auto_anchor_path(ball_out_path.parent, shot_id))
            except Exception as exc:  # noqa: BLE001 — auto anchors must never kill the stage
                logger.warning(
                    "ball stage: auto-anchor generation failed (%s) — "
                    "continuing with manual anchors only", exc,
                )
        anchor_by_frame = merge_anchors(
            manual_by_frame, auto_by_frame, anchor_cfg.suppress_radius_frames,
        )
        ground_touch_frames = _classify_ground_touches(anchor_by_frame)

        # --- 5. Resolve nodes and solve ---------------------------------
        pitch_length = float(pitch_cfg.get("length_m", 105.0))
        pitch_width = float(pitch_cfg.get("width_m", 68.0))
        # Near-horizon ray-casts blow up to hundreds of metres; a node
        # built from one would teleport the whole adjacent segment.
        node_clamp_m = max(50.0, 2.0 * max(pitch_length, pitch_width))

        def _node_world_ok(world: np.ndarray) -> bool:
            return bool(
                np.all(np.isfinite(world))
                and abs(float(world[0])) <= node_clamp_m
                and abs(float(world[1])) <= node_clamp_m
            )

        nodes: list[TrajectoryNode] = []
        contact_gaps: list[dict] = []
        for fi in sorted(anchor_by_frame):
            anc = anchor_by_frame[fi]
            if anc.state not in HARD_KNOT_STATES:
                continue
            world = _resolve_anchor_world(
                anc=anc, fi=fi,
                ground_touch_frames=ground_touch_frames,
                player_ctx=player_ctx,
                per_frame_K=per_frame_K,
                per_frame_R=per_frame_R,
                per_frame_t=per_frame_t,
                distortion=distortion,
                ball_radius=ball_radius,
                goal_geometry=goal_geometry,
            )
            if world is None:
                continue
            if not _node_world_ok(world):
                logger.warning(
                    "ball stage: anchor at frame %d resolved far off-pitch "
                    "(%.0f, %.0f) — dropping it from the solve",
                    fi, float(world[0]), float(world[1]),
                )
                continue
            is_manual = fi in manual_by_frame
            nodes.append(TrajectoryNode(
                frame=fi,
                world_xyz=(float(world[0]), float(world[1]), float(world[2])),
                state=anc.state,
                confidence=1.0 if is_manual else 0.8,
                spin=anc.spin,
                is_manual=is_manual,
            ))
            if (
                anc.state == "player_touch"
                and anc.player_id and anc.bone
                and fi in per_frame_K
            ):
                bone_world = player_ctx.joint_world(fi, anc.player_id, anc.bone)
                if bone_world is not None and anc.image_xy is not None:
                    gap = point_to_pixel_ray_distance(
                        bone_world, anc.image_xy,
                        per_frame_K[fi], per_frame_R[fi], per_frame_t[fi],
                        distortion,
                    )
                    contact_gaps.append({
                        "frame": fi,
                        "player_id": anc.player_id,
                        "bone": anc.bone,
                        "gap_m": float(gap),
                        "manual": is_manual,
                    })

        # A flight chain the clip enters or leaves mid-air has no hard
        # knot at its open end (camera cut after a cross, clip ending on
        # a shot). Pin the chain-edge airborne anchor at its bucket
        # height so the span is bracketed — coarse depth, low confidence,
        # but a determined continuous arc instead of an open guess.
        hard_frames = sorted(n.frame for n in nodes)
        airborne_chain = sorted(
            (fi, anc) for fi, anc in anchor_by_frame.items()
            if airborne_bucket_range(anc.state) is not None
            and anc.image_xy is not None
        )

        def _synth_airborne_node(fi: int, anc: BallAnchor) -> None:
            if fi not in per_frame_K:
                return
            try:
                world = np.asarray(ankle_ray_to_pitch(
                    anc.image_xy,
                    K=per_frame_K[fi], R=per_frame_R[fi], t=per_frame_t[fi],
                    plane_z=state_to_height(anc.state), distortion=distortion,
                ), dtype=float)
            except Exception:
                return
            if not _node_world_ok(world):
                return
            nodes.append(TrajectoryNode(
                frame=fi,
                world_xyz=(float(world[0]), float(world[1]), float(world[2])),
                state="airborne",
                confidence=0.5,
                is_manual=fi in manual_by_frame,
            ))

        leading = [x for x in airborne_chain
                   if not hard_frames or x[0] < hard_frames[0]]
        trailing = [x for x in airborne_chain
                    if not hard_frames or x[0] > hard_frames[-1]]
        if leading:
            _synth_airborne_node(*leading[0])
        if trailing and (not leading or trailing[-1][0] != leading[0][0]):
            _synth_airborne_node(*trailing[-1])
        nodes.sort(key=lambda n: n.frame)

        z_hints = {
            fi: bucket
            for fi, anc in anchor_by_frame.items()
            if (bucket := airborne_bucket_range(anc.state)) is not None
        }
        node_frames = {n.frame for n in nodes}
        split_hints = tuple(
            (e.frame, e.score) for e in events
            if e.kind == "velocity_break" and e.frame not in node_frames
        )

        result = solve_piecewise(
            nodes=nodes,
            steps=steps,
            confidences=raw_confidences,
            per_frame_K=per_frame_K, per_frame_R=per_frame_R,
            per_frame_t=per_frame_t, distortion=distortion,
            fps=camera.fps, n_frames=n_frames,
            pitch_length_m=float(pitch_cfg.get("length_m", 105.0)),
            pitch_width_m=float(pitch_cfg.get("width_m", 68.0)),
            split_hints=split_hints,
            z_hints=z_hints,
            manual_obs_frames={
                fi for fi, anc in manual_by_frame.items()
                if anc.image_xy is not None
            },
            cfg=solver_cfg,
        )
        world_by_frame = dict(result.world_by_frame)
        state_by_frame = dict(result.state_by_frame)

        # C4 — ray-faithfulness for airborne-bucket anchors: the clicked
        # pixel is hard lateral ground truth; keep the solved depth but
        # snap onto the clicked ray when reprojection drifts.
        ray_faithful_tol_px = float(cfg.get("ray_faithful_tolerance_px", 3.0))
        for fi, anc in anchor_by_frame.items():
            if anc.state not in AIRBORNE_STATES or anc.image_xy is None:
                continue
            if anc.state == "off_screen_flight":
                continue
            entry = world_by_frame.get(fi)
            if entry is None or fi not in per_frame_K:
                continue
            world, conf = entry
            uvp = project_world_to_image(
                per_frame_K[fi], per_frame_R[fi], per_frame_t[fi],
                distortion, np.asarray(world, dtype=float).reshape(1, 3),
            )[0]
            if float(np.linalg.norm(uvp - np.array(anc.image_xy))) <= ray_faithful_tol_px:
                continue
            snapped = project_point_onto_pixel_ray(
                np.asarray(world, dtype=float),
                (float(anc.image_xy[0]), float(anc.image_xy[1])),
                per_frame_K[fi], per_frame_R[fi], per_frame_t[fi], distortion,
            )
            world_by_frame[fi] = (snapped, conf)
            state_by_frame[fi] = "flight"

        # off_screen_flight: the operator says the ball is airborne out
        # of frame — honest state without a world position.
        for fi, anc in anchor_by_frame.items():
            if anc.state == "off_screen_flight" and fi not in world_by_frame:
                state_by_frame[fi] = "flight"

        # --- 6. Emit -----------------------------------------------------
        segment_by_frame: dict[int, int] = {}
        for seg in result.flight_segments:
            for fi in range(seg.frame_range[0], seg.frame_range[1] + 1):
                segment_by_frame[fi] = seg.id

        per_frame_out: list[BallFrame] = []
        for fi in range(n_frames):
            entry = world_by_frame.get(fi)
            state = state_by_frame.get(fi, "missing")
            if entry is not None:
                world, conf = entry
                per_frame_out.append(BallFrame(
                    frame=fi,
                    world_xyz=tuple(float(x) for x in world),
                    state=state if state != "missing" else "grounded",
                    confidence=float(conf),
                    flight_segment_id=segment_by_frame.get(fi),
                ))
            else:
                per_frame_out.append(BallFrame(
                    frame=fi,
                    world_xyz=None,
                    state="flight" if state == "flight" else "missing",
                    confidence=0.0,
                    flight_segment_id=segment_by_frame.get(fi),
                ))

        track = BallTrack(
            clip_id=camera.clip_id,
            fps=camera.fps,
            frames=tuple(per_frame_out),
            flight_segments=result.flight_segments,
        )
        track.save(ball_out_path)

        try:
            _emit_ball_keyframes(
                ball_out_path=ball_out_path,
                clip_id=camera.clip_id,
                fps=camera.fps,
                image_size=camera.image_size,
                per_frame_out=per_frame_out,
                anchor_by_frame=anchor_by_frame,
                per_frame_K=per_frame_K,
                per_frame_R=per_frame_R,
                per_frame_t=per_frame_t,
                distortion=distortion,
                ground_touch_frames=ground_touch_frames,
            )
        except Exception as exc:  # noqa: BLE001 — sidecar is enrichment, never block the stage
            logger.warning(
                "ball: failed to write keyframes sidecar for %s: %s",
                ball_out_path, exc,
            )

        for span in result.diagnostics.get("underconstrained_spans", []):
            logger.warning(
                "ball: span %d-%d could not be explained by a physical "
                "segment within the residual gate (%.1f px) — add a manual "
                "anchor inside it",
                span["start"], span["end"], span.get("residual_px") or -1.0,
            )
        diag_path = ball_out_path.with_name(
            ball_out_path.name.replace("ball_track", "ball_diag")
        )
        diag_path.write_text(json.dumps({
            "underconstrained_spans": result.diagnostics.get(
                "underconstrained_spans", []),
            "segments": result.diagnostics.get("segments", []),
            "bounces": result.diagnostics.get("bounces", []),
            "splits": result.diagnostics.get("splits", 0),
            "contact_gaps": contact_gaps,
            "events": [
                {
                    "frame": e.frame, "kind": e.kind,
                    "score": round(float(e.score), 3),
                    "player_id": e.player_id, "bone": e.bone,
                    "goal_element": e.goal_element,
                    "end_frame": e.end_frame,
                }
                for e in events
            ],
            "anchors": {
                "manual": len(manual_by_frame),
                "auto_generated": len(auto_by_frame),
                "merged": len(anchor_by_frame),
                "nodes": len(nodes),
            },
        }, indent=2))
