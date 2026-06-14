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

import dataclasses
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
from src.utils.ball_auto_events import (
    AutoEventCfg,
    detect_event_candidates,
    detect_events,
)
from src.utils.ball_detector import BallDetector, YOLOBallDetector
from src.utils.ball_keyframe_builder import build_ball_keyframe_set
from src.utils.ball_mode_search import (
    BudgetExceeded,
    _mode_search_cfg,
    solve_modes,
)
from src.utils.ball_orientation import integrate_orientation
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
from src.schemas.ball_fixes import BallFix, BallFixSet
from src.schemas.sync_map import SyncMap
from src.utils.ball_cross_replay import (
    CrossReplayCfg,
    PairFix,
    refine_pair_offset,
    triangulate_pair,
)
from src.utils.ball_second_pass import (
    SecondPassCfg,
    SecondPassDetection,
    apparent_ball_px,
    best_gated_candidate,
    corridor_predictions,
    filter_in_bounds,
    find_gap_runs,
    map_crop_candidates,
)
from src.utils.foot_anchor import ankle_ray_to_pitch
from src.utils.goal_geometry import GoalGeometry, resolve_goal_impact_world

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class _DetectArtifacts:
    """Everything produced by the detect pass for one shot.

    Carries the raw detection output, camera matrices, manual anchors, and
    the paths needed by the solve pass — so detect and solve can run in
    two separate loops.
    """

    shot_id: str
    ball_out_path: Path
    camera_clip_id: str
    camera_fps: float
    camera_image_size: tuple[int, int]
    distortion: tuple[float, float]
    n_clip: int
    n_frames: int

    # Detection output
    steps: list  # list[TrackerStep]
    raw_confidences: dict[int, float]
    sources: dict[int, str]
    detection_coverage: dict[str, float]

    # Camera matrices (per frame)
    per_frame_K: dict[int, object]
    per_frame_R: dict[int, object]
    per_frame_t: dict[int, object]

    # Manual anchors
    manual_by_frame: dict[int, object]  # dict[int, BallAnchor]


def _accepted_obs_and_cams(
    arts: "_DetectArtifacts",
) -> tuple[dict[int, tuple[tuple[float, float], float]], dict[int, tuple]]:
    """Accepted-evidence observation + camera maps for triangulation.

    Observations are frames with a non-outlier detection carrying a source
    (detector/bridge/anchor/second_pass) — the same accepted-evidence
    filter the second-pass corridor uses. Cameras are the shot's per-frame
    (K, R, t).
    """
    obs = {
        s.frame: (s.uv, arts.raw_confidences.get(s.frame, 0.0))
        for s in arts.steps
        if (s.uv is not None and s.frame in arts.sources and not s.is_outlier)
    }
    cams = {
        f: (arts.per_frame_K[f], arts.per_frame_R[f], arts.per_frame_t[f])
        for f in arts.per_frame_K
    }
    return obs, cams


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
    flight_segments: "tuple" = (),
) -> None:
    """Write the sparse ``*_ball_keyframes.json`` sidecar next to the dense
    track. ``world_xyz`` for each anchor is taken from the already-built
    dense ``per_frame_out`` so the two artifacts agree exactly.

    ``flight_segments`` lets keyframes on a spinning flight carry the fitted
    Magnus ``omega_rad_s`` so the engine can drive physically-correct curl.
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
        flight_segments=flight_segments,
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


def _build_tracker(cfg: dict, max_gap_frames: int | None = None) -> BallTracker:
    """IMM tracker from the ball config; optional max-gap override for
    corridor prediction (predictions must persist through long gaps)."""
    tracker_cfg = cfg.get("tracker", {})
    return BallTracker(
        process_noise_grounded_px=float(tracker_cfg.get("process_noise_grounded_px", 4.0)),
        process_noise_flight_px=float(tracker_cfg.get("process_noise_flight_px", 12.0)),
        measurement_noise_px=float(tracker_cfg.get("measurement_noise_px", 2.0)),
        gating_sigma=float(tracker_cfg.get("gating_sigma", 4.0)),
        max_gap_frames=(
            int(cfg.get("max_gap_frames", 6))
            if max_gap_frames is None else int(max_gap_frames)
        ),
        initial_p_flight=float(tracker_cfg.get("initial_p_flight", 0.1)),
    )


def _resmooth_observations(
    per_frame_uv: dict[int, tuple[float, float] | None],
    n_frames: int,
    cfg: dict,
) -> list[TrackerStep]:
    """Fresh IMM pass over a merged observation set (no video access).

    Applies the same raw-uv override rule as the streaming detect loop:
    fits must see raw measurements, the tracker only bridges misses.
    """
    tracker = _build_tracker(cfg)
    steps: list[TrackerStep] = []
    for f in range(n_frames):
        uv = per_frame_uv.get(f)
        step = tracker.update(f, uv)
        if uv is not None and not step.is_outlier:
            step = TrackerStep(
                frame=step.frame, uv=uv, p_flight=step.p_flight,
                is_outlier=step.is_outlier, is_gap_fill=step.is_gap_fill,
                pos_cov=step.pos_cov,
            )
        steps.append(step)
    return steps


def _second_pass_cfg(cfg: dict) -> SecondPassCfg:
    sp = cfg.get("second_pass", {})
    base = SecondPassCfg()
    return SecondPassCfg(
        enabled=bool(sp.get("enabled", base.enabled)),
        candidate_min_score=float(sp.get("candidate_min_score", base.candidate_min_score)),
        top_k=int(sp.get("top_k", base.top_k)),
        corridor_sigma=float(sp.get("corridor_sigma", base.corridor_sigma)),
        accept_min=float(sp.get("accept_min", base.accept_min)),
        zoom_min_ball_px=float(sp.get("zoom_min_ball_px", base.zoom_min_ball_px)),
        zoom_crop_px=int(sp.get("zoom_crop_px", base.zoom_crop_px)),
    )


def _cross_replay_cfg(cfg: dict) -> CrossReplayCfg:
    """Map ball.cross_replay.* to a :class:.

    Pattern: same as _second_pass_cfg in this module.
    """
    cr = cfg.get("cross_replay", {})
    base = CrossReplayCfg()
    return CrossReplayCfg(
        enabled=bool(cr.get("enabled", base.enabled)),
        min_conf=float(cr.get("min_conf", base.min_conf)),
        max_ray_miss_m=float(cr.get("max_ray_miss_m", base.max_ray_miss_m)),
        min_parallax_deg=float(cr.get("min_parallax_deg", base.min_parallax_deg)),
        offset_search_radius_frames=float(cr.get(
            "offset_search_radius_frames", base.offset_search_radius_frames)),
        offset_search_step=float(cr.get("offset_search_step", base.offset_search_step)),
        min_pairs_for_refine=int(cr.get(
            "min_pairs_for_refine", base.min_pairs_for_refine)),
        fix_weight_px_per_m=float(cr.get("fix_weight_px_per_m", base.fix_weight_px_per_m)),
    )


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
        min_break_speed_px=float(auto_cfg.get(
            "min_break_speed_px", base.min_break_speed_px)),
        bounce_min_vy_px=float(auto_cfg.get(
            "bounce_min_vy_px", base.bounce_min_vy_px)),
        stationary_max_speed_px=float(auto_cfg.get(
            "stationary_max_speed_px", base.stationary_max_speed_px)),
        stationary_min_frames=int(auto_cfg.get(
            "stationary_min_frames", base.stationary_min_frames)),
        stationary_min_conf=float(auto_cfg.get(
            "stationary_min_conf", base.stationary_min_conf)),
        goal_line_tolerance_m=float(auto_cfg.get(
            "goal_line_tolerance_m", base.goal_line_tolerance_m)),
        goal_net_speed_drop_ratio=float(auto_cfg.get(
            "goal_net_speed_drop_ratio", base.goal_net_speed_drop_ratio)),
        goal_min_direction_change_deg=float(auto_cfg.get(
            "goal_min_direction_change_deg", base.goal_min_direction_change_deg)),
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
        post_event_suppress_frames=int(auto_cfg.get(
            "post_event_suppress_frames", base.post_event_suppress_frames)),
        grounded_max_p_flight=float(auto_cfg.get(
            "grounded_max_p_flight", base.grounded_max_p_flight)),
        max_speed_m_s=float(auto_cfg.get(
            "max_speed_m_s", base.max_speed_m_s)),
        max_ground_speed_m_s=float(auto_cfg.get(
            "max_ground_speed_m_s", base.max_ground_speed_m_s)),
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
        rolling_max_speed_m_s=float(physics.get(
            "rolling_max_speed_m_s", base.rolling_max_speed_m_s)),
        max_arc_seconds=float(physics.get(
            "max_arc_seconds", base.max_arc_seconds)),
        open_span_min_apex_m=float(physics.get(
            "open_span_min_apex_m", base.open_span_min_apex_m)),
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
            arts = self._detect_shot("", clip_path, cam_path, ball_out, cfg, detector)
            if arts is not None:
                self._solve_shot(arts, cfg, fixes=None)
            return

        manifest = ShotsManifest.load(manifest_path)
        shot_filter = getattr(self, "shot_filter", None)

        # --- Pass 1: detect all shots ------------------------------------
        artifacts_by_shot: dict[str, _DetectArtifacts] = {}
        active_shots = manifest.active_shots()
        for shot in active_shots:
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
            arts = self._detect_shot(shot.id, clip_path, cam_path, ball_out, cfg, detector)
            if arts is not None:
                artifacts_by_shot[shot.id] = arts

        # --- Pass 2: triangulate per sync-group --------------------------
        fixes_by_shot, cr_summaries = self._triangulate_groups(
            artifacts_by_shot, cfg,
        )

        # --- Pass 3: solve all shots with fixes --------------------------
        for shot_id, arts in artifacts_by_shot.items():
            self._solve_shot(
                arts, cfg,
                fixes=fixes_by_shot.get(shot_id),
                cr_summary=cr_summaries.get(shot_id),
            )

    def _run_shot(
        self,
        shot_id: str,
        clip_path: Path,
        camera_path: Path,
        ball_out_path: Path,
        cfg: dict,
        detector: BallDetector,
    ) -> None:
        """Compatibility shim: detect then solve for one shot.

        Retained for callers (e.g. real-clip acceptance tests) that invoke
        the method directly.  The three-pass run() path uses
        _detect_shot / _triangulate_groups / _solve_shot instead.
        """
        arts = self._detect_shot(
            shot_id, clip_path, camera_path, ball_out_path, cfg, detector,
        )
        if arts is not None:
            self._solve_shot(arts, cfg, fixes=None)

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
        tracker = _build_tracker(cfg)
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
                    if det is not None:
                        h_img, w_img = frame.shape[:2]
                        if not (0.0 <= det[0] < w_img and 0.0 <= det[1] < h_img):
                            det = None
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
                        pos_cov=step.pos_cov,
                    )
                steps.append(step)
                frame_idx += 1
        finally:
            cap.release()
        return steps, raw_confidences, sources

    def _second_pass_loop(
        self,
        clip_path: Path,
        gap_runs: list[tuple[int, int]],
        corridors: dict[int, tuple[np.ndarray, np.ndarray]],
        per_frame_K: dict[int, np.ndarray],
        per_frame_R: dict[int, np.ndarray],
        per_frame_t: dict[int, np.ndarray],
        distortion: tuple[float, float],
        detector: BallDetector,
        sp_cfg: SecondPassCfg,
        ball_radius: float,
    ) -> list[SecondPassDetection]:
        """Revisit evidence gaps with corridor-gated candidate detection.

        Full-frame pass first (run-grouped so the detector's temporal
        buffer is primed once per run), then a zoom retry on frames where
        nothing cleared the gate and the predicted ball is small.
        """
        accepted: list[SecondPassDetection] = []
        zoom_targets: list[int] = []
        prime_offset = getattr(detector, "_frames_in", 3) - 1
        cap = cv2.VideoCapture(str(clip_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open clip: {clip_path}")
        try:
            for start, end in gap_runs:
                prime = max(0, start - prime_offset)
                cap.set(cv2.CAP_PROP_POS_FRAMES, prime)
                detector.reset()
                for f in range(prime, end + 1):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    cands = detector.detect_candidates(
                        frame, sp_cfg.candidate_min_score, sp_cfg.top_k,
                    )
                    cands = filter_in_bounds(cands, frame.shape[1], frame.shape[0])
                    if f < start or f not in corridors:
                        continue
                    mean, cov = corridors[f]
                    best = best_gated_candidate(cands, mean, cov, sp_cfg)
                    if best is not None:
                        accepted.append(SecondPassDetection(
                            frame=f, uv=best[0],
                            combined_score=best[1], used_zoom=False,
                        ))
                        continue
                    K = per_frame_K.get(f)
                    R = per_frame_R.get(f)
                    t = per_frame_t.get(f)
                    if K is None or R is None or t is None:
                        continue
                    size = apparent_ball_px(
                        K, R, t,
                        (float(mean[0]), float(mean[1])),
                        ball_radius, distortion,
                    )
                    if size is not None and size < sp_cfg.zoom_min_ball_px:
                        zoom_targets.append(f)
            for f in zoom_targets:
                mean, cov = corridors[f]
                best = self._zoom_detect(cap, f, mean, cov, detector, sp_cfg)
                if best is not None:
                    accepted.append(SecondPassDetection(
                        frame=f, uv=best[0],
                        combined_score=best[1], used_zoom=True,
                    ))
        finally:
            detector.reset()
            cap.release()
        accepted.sort(key=lambda d: d.frame)
        return accepted

    def _zoom_detect(
        self,
        cap: "cv2.VideoCapture",
        frame_idx: int,
        mean: np.ndarray,
        cov: np.ndarray,
        detector: BallDetector,
        sp_cfg: SecondPassCfg,
    ) -> tuple[tuple[float, float], float] | None:
        """Crop around the corridor and re-detect; the detector's own
        letterbox upscales the crop, magnifying a small ball."""
        half = sp_cfg.zoom_crop_px // 2
        prime_offset = getattr(detector, "_frames_in", 3) - 1
        prime = max(0, frame_idx - prime_offset)
        cap.set(cv2.CAP_PROP_POS_FRAMES, prime)
        detector.reset()
        best: tuple[tuple[float, float], float] | None = None
        for f in range(prime, frame_idx + 1):
            ret, frame = cap.read()
            if not ret:
                return None
            h, w = frame.shape[:2]
            x0 = int(np.clip(mean[0] - half, 0, max(0, w - sp_cfg.zoom_crop_px)))
            y0 = int(np.clip(mean[1] - half, 0, max(0, h - sp_cfg.zoom_crop_px)))
            crop = frame[y0:y0 + sp_cfg.zoom_crop_px, x0:x0 + sp_cfg.zoom_crop_px]
            if crop.size == 0:
                return None
            cands = detector.detect_candidates(
                crop, sp_cfg.candidate_min_score, sp_cfg.top_k,
            )
            if f == frame_idx:
                best = best_gated_candidate(
                    map_crop_candidates(cands, x0, y0), mean, cov, sp_cfg,
                )
        detector.reset()
        return best

    def _detect_shot(
        self,
        shot_id: str,
        clip_path: Path,
        camera_path: Path,
        ball_out_path: Path,
        cfg: dict,
        detector: BallDetector,
    ) -> "_DetectArtifacts | None":
        """Detect pass for one shot: camera load → manual anchors → detect
        loop → second pass → coverage → observations sidecar.

        Returns a :class:`_DetectArtifacts` with everything the solve pass
        needs, or ``None`` if the clip contained no frames.
        """
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
            return None
        n_frames = max(n_frames, steps[-1].frame + 1)

        # --- 1b. Second pass over evidence gaps -------------------------
        sp_cfg = _second_pass_cfg(cfg)
        n_clip = steps[-1].frame + 1
        outliers = {s.frame for s in steps if s.is_outlier}
        # Pass-1 raw observations ONLY (feedback-loop guard): the corridor
        # that admits a second-pass detection is never built from
        # second-pass output.
        pass1_uv: dict[int, tuple[float, float] | None] = {
            s.frame: (
                s.uv if (s.frame in sources and not s.is_outlier) else None
            )
            for s in steps
        }
        n_second_pass = 0
        n_zoom = 0
        ball_radius = float(cfg.get("ball_radius_m", 0.11))
        if sp_cfg.enabled:
            gap_runs = find_gap_runs(sources, outliers, n_clip)
            if gap_runs:
                corridors = corridor_predictions(
                    pass1_uv, n_clip,
                    tracker_factory=lambda: _build_tracker(
                        cfg, max_gap_frames=10 ** 6),
                )
                sp_dets = self._second_pass_loop(
                    clip_path, gap_runs, corridors, per_frame_K, per_frame_R,
                    per_frame_t, distortion, detector, sp_cfg, ball_radius,
                )
                if sp_dets:
                    n_second_pass = len(sp_dets)
                    n_zoom = sum(1 for d in sp_dets if d.used_zoom)
                    logger.info(
                        "ball: second pass recovered %d/%d gap frames for %s",
                        n_second_pass,
                        sum(e - s + 1 for s, e in gap_runs),
                        shot_id or "(legacy)",
                    )
                    merged_uv = dict(pass1_uv)
                    for d in sp_dets:
                        merged_uv[d.frame] = d.uv
                        raw_confidences[d.frame] = d.combined_score
                        sources[d.frame] = "second_pass"
                    steps = _resmooth_observations(merged_uv, n_clip, cfg)

        # Coverage counts ACCEPTED evidence: pass1 counts frames whose raw
        # observation survived gating (outlier-rejected frames are not
        # covered even though their source reads "detector").
        n_pass1 = sum(1 for v in pass1_uv.values() if v is not None)
        detection_coverage = {
            "pass1": n_pass1 / n_clip,
            "second_pass": n_second_pass / n_clip,
            "total": (n_pass1 + n_second_pass) / n_clip,
            # Count (not fraction): zoom-retry recoveries within second_pass.
            "zoom_recoveries": n_zoom,
        }

        try:
            _write_observations_sidecar(
                ball_out_path.with_name(ball_out_path.name.replace(
                    "ball_track", "ball_observations")),
                camera.clip_id, camera.fps, steps, raw_confidences, sources,
            )
        except Exception as exc:  # noqa: BLE001 — sidecar is enrichment
            logger.warning("ball: failed to write observations sidecar: %s", exc)

        return _DetectArtifacts(
            shot_id=shot_id,
            ball_out_path=ball_out_path,
            camera_clip_id=camera.clip_id,
            camera_fps=camera.fps,
            camera_image_size=camera.image_size,
            distortion=distortion,
            n_clip=n_clip,
            n_frames=n_frames,
            steps=steps,
            raw_confidences=raw_confidences,
            sources=sources,
            detection_coverage=detection_coverage,
            per_frame_K=per_frame_K,
            per_frame_R=per_frame_R,
            per_frame_t=per_frame_t,
            manual_by_frame=manual_by_frame,
        )

    def _triangulate_groups(
        self,
        artifacts_by_shot: dict[str, "_DetectArtifacts"],
        cfg: dict,
    ) -> tuple[dict[str, dict[int, tuple[np.ndarray, float]]], dict[str, dict]]:
        """Cross-replay triangulation pass.

        For each highlight group with >= 2 member shots present in
        ``artifacts_by_shot``: refine the saved sync offset by ray-miss
        minimisation, triangulate detection pairs into 3D fixes, and
        write per-shot :class:`BallFixSet` sidecars.

        Returns ``(fixes_by_shot, summaries_by_shot)`` where:
        - ``fixes_by_shot[shot_id]`` is ``{frame: (xyz_array, weight)}``
        - ``summaries_by_shot[shot_id]`` is the cross_replay summary dict
          (surfaced in the diag sidecar).
        """
        sync_map_path = self.output_dir / "shots" / "sync_map.json"
        if not sync_map_path.exists():
            return {}, {}

        cr_cfg = _cross_replay_cfg(cfg)
        if not cr_cfg.enabled:
            return {}, {}

        try:
            sync_map = SyncMap.load(sync_map_path)
        except Exception as exc:
            logger.warning("ball: failed to load sync_map.json: %s — skipping triangulation", exc)
            return {}, {}

        fixes_by_shot: dict[str, dict[int, tuple[np.ndarray, float]]] = {}
        summaries_by_shot: dict[str, dict] = {}
        ball_dir = self.output_dir / "ball"
        weight = cr_cfg.fix_weight_px_per_m

        for group_sync in sync_map.groups:
            # Gather the members that are both in this group AND detected.
            members_present = [
                a for a in group_sync.alignments
                if a.shot_id in artifacts_by_shot
            ]
            if len(members_present) < 2:
                continue

            # Side A = member with offset closest to 0 (the reference).
            members_present.sort(key=lambda a: abs(a.frame_offset))
            align_a = members_present[0]
            sides_b = members_present[1:]
            arts_a = artifacts_by_shot[align_a.shot_id]
            obs_a, cams_a = _accepted_obs_and_cams(arts_a)

            # Accumulate the reference shot's fixes/summary across ALL
            # partners; its sidecar is written ONCE after the partner loop
            # so a multi-partner group never clobbers the reference.
            a_fix_records: list[BallFix] = []
            a_best_miss: dict[int, float] = {}
            a_partners: dict[str, dict] = {}
            a_ray_misses: list[float] = []
            a_parallaxes: list[float] = []

            for align_b in sides_b:
                arts_b = artifacts_by_shot[align_b.shot_id]
                obs_b, cams_b = _accepted_obs_and_cams(arts_b)

                # delta_saved = offset_B - offset_A (per sync_map convention:
                # f_b - frame_offset_B == f_a - frame_offset_A
                # => f_b = f_a + (offset_B - offset_A)).
                delta_saved = float(align_b.frame_offset - align_a.frame_offset)

                refined_offset, _refine_miss, n_pairs = refine_pair_offset(
                    obs_a=obs_a, cams_a=cams_a, obs_b=obs_b, cams_b=cams_b,
                    saved_offset=delta_saved, cfg=cr_cfg,
                    distortion_a=arts_a.distortion, distortion_b=arts_b.distortion,
                )
                pair_fixes: list[PairFix] = triangulate_pair(
                    obs_a=obs_a, cams_a=cams_a, obs_b=obs_b, cams_b=cams_b,
                    offset_b_minus_a=refined_offset, cfg=cr_cfg,
                    distortion_a=arts_a.distortion, distortion_b=arts_b.distortion,
                )
                if not pair_fixes:
                    logger.info(
                        "ball: cross-replay: no inlier fixes for pair %s / %s",
                        align_a.shot_id, align_b.shot_id,
                    )
                    continue

                # Per-partner metadata. median_ray_miss is over the accepted
                # inlier fixes (what actually constrains the solve), not the
                # refine-grid median used internally for offset selection.
                inlier_miss = float(np.median([fx.ray_miss_m for fx in pair_fixes]))
                inlier_parallax = float(
                    np.median([fx.parallax_deg for fx in pair_fixes])
                )
                partner_meta = {
                    "saved_offset": delta_saved,
                    "refined_offset": refined_offset,
                    "offset_disagreement_frames": abs(refined_offset - delta_saved),
                    "n_pairs": n_pairs,
                    "n_inlier_fixes": len(pair_fixes),
                    "median_ray_miss_m": inlier_miss,
                    "median_parallax_deg": inlier_parallax,
                }
                a_partners[align_b.shot_id] = partner_meta

                # Reference-shot solver fixes: keep the lowest-ray-miss fix
                # per frame across partners; sidecar keeps every record.
                a_solver = fixes_by_shot.setdefault(align_a.shot_id, {})
                for fx in pair_fixes:
                    prev = a_best_miss.get(fx.frame_a)
                    if prev is None or fx.ray_miss_m < prev:
                        a_best_miss[fx.frame_a] = fx.ray_miss_m
                        a_solver[fx.frame_a] = (np.array(fx.xyz), weight)
                    a_fix_records.append(BallFix(
                        frame=fx.frame_a, xyz=fx.xyz,
                        ray_miss_m=fx.ray_miss_m, parallax_deg=fx.parallax_deg,
                        partner_shot=align_b.shot_id, partner_frame=fx.frame_b,
                    ))
                    a_ray_misses.append(fx.ray_miss_m)
                    a_parallaxes.append(fx.parallax_deg)

                # Partner shot has a single partner (the reference): write its
                # fixes + summary + sidecar now.
                b_solver = fixes_by_shot.setdefault(align_b.shot_id, {})
                for fx in pair_fixes:
                    b_solver[fx.frame_b] = (np.array(fx.xyz), weight)
                b_summary = {
                    "partner_shots": [align_a.shot_id],
                    **partner_meta,
                    "partners": {align_a.shot_id: partner_meta},
                }
                summaries_by_shot[align_b.shot_id] = b_summary
                ball_dir.mkdir(parents=True, exist_ok=True)
                BallFixSet(
                    clip_id=arts_b.camera_clip_id,
                    group_id=group_sync.group_id,
                    cross_replay=b_summary,
                    fixes=tuple(
                        BallFix(
                            frame=fx.frame_b, xyz=fx.xyz,
                            ray_miss_m=fx.ray_miss_m, parallax_deg=fx.parallax_deg,
                            partner_shot=align_a.shot_id, partner_frame=fx.frame_a,
                        )
                        for fx in pair_fixes
                    ),
                ).save(ball_dir / f"{align_b.shot_id}_ball_fixes.json")
                logger.info(
                    "ball: cross-replay %s / %s: %d inlier fixes, "
                    "refined offset %.2f (saved %.2f), median miss %.3f m",
                    align_a.shot_id, align_b.shot_id,
                    len(pair_fixes), refined_offset, delta_saved, inlier_miss,
                )

            if not a_partners:
                continue  # no partner produced fixes for the reference

            # Aggregate the reference shot's summary. Offsets are inherently
            # per-partner, so top-level scalars report the dominant pairing
            # (most inlier fixes) and the WORST disagreement (the review cue);
            # full per-partner detail lives in `partners`.
            dominant = max(
                a_partners.values(), key=lambda p: p["n_inlier_fixes"],
            )
            a_summary = {
                "partner_shots": sorted(a_partners.keys()),
                "saved_offset": dominant["saved_offset"],
                "refined_offset": dominant["refined_offset"],
                "offset_disagreement_frames": max(
                    p["offset_disagreement_frames"] for p in a_partners.values()
                ),
                "n_pairs": sum(p["n_pairs"] for p in a_partners.values()),
                "n_inlier_fixes": sum(
                    p["n_inlier_fixes"] for p in a_partners.values()
                ),
                "median_ray_miss_m": float(np.median(a_ray_misses)),
                "median_parallax_deg": float(np.median(a_parallaxes)),
                "partners": a_partners,
            }
            summaries_by_shot[align_a.shot_id] = a_summary
            ball_dir.mkdir(parents=True, exist_ok=True)
            BallFixSet(
                clip_id=arts_a.camera_clip_id,
                group_id=group_sync.group_id,
                cross_replay=a_summary,
                fixes=tuple(a_fix_records),
            ).save(ball_dir / f"{align_a.shot_id}_ball_fixes.json")

        return fixes_by_shot, summaries_by_shot

    def _solve_shot(
        self,
        artifacts: "_DetectArtifacts",
        cfg: dict,
        fixes: "dict[int, tuple[np.ndarray, float]] | None",
        cr_summary: "dict | None" = None,
    ) -> None:
        """Solve pass for one shot: player context → events → anchors →
        nodes → solve_piecewise (with optional world fixes) → outputs → diag.

        ``fixes`` is ``{frame: (xyz_array, weight)}`` from the triangulation
        pass; ``None`` or empty means the pre-1.5 single-shot path.
        ``cr_summary`` is this run's cross_replay summary for the shot (None
        when no triangulation produced fixes); it is written to the diag
        directly so the diag always reflects the current run.
        """
        shot_id = artifacts.shot_id
        ball_out_path = artifacts.ball_out_path
        per_frame_K = artifacts.per_frame_K
        per_frame_R = artifacts.per_frame_R
        per_frame_t = artifacts.per_frame_t
        distortion = artifacts.distortion
        steps = artifacts.steps
        raw_confidences = artifacts.raw_confidences
        sources = artifacts.sources
        manual_by_frame = artifacts.manual_by_frame
        n_frames = artifacts.n_frames

        ball_radius = float(cfg.get("ball_radius_m", 0.11))
        pitch_cfg = self.config.get("pitch", {})
        goal_geometry = GoalGeometry.from_pitch_config(pitch_cfg)
        auto_cfg_dict = cfg.get("auto_anchors", {})
        event_cfg = _auto_event_cfg(auto_cfg_dict)
        anchor_cfg = _auto_anchor_cfg(auto_cfg_dict, ball_radius)
        solver_cfg = _solver_cfg(cfg, ball_radius)

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
                    fps=artifacts.camera_fps, pitch_cfg=pitch_cfg, cfg=anchor_cfg,
                    sources=sources,
                )
                auto_by_frame = {a.frame: a for a in auto_anchors}
                BallAnchorSet(
                    clip_id=artifacts.camera_clip_id,
                    image_size=artifacts.camera_image_size,
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

        # The solver seam (Phase 2, Task 8). Both paths receive the EXACT same
        # kwargs — factored into one dict so the global path and its whole-shot
        # piecewise fallback can never drift apart (F8: the fallback must yield
        # the identical piecewise result, not a degraded beam result).
        solve_kwargs = dict(
            nodes=nodes,
            steps=steps,
            confidences=raw_confidences,
            per_frame_K=per_frame_K, per_frame_R=per_frame_R,
            per_frame_t=per_frame_t, distortion=distortion,
            fps=artifacts.camera_fps, n_frames=n_frames,
            pitch_length_m=float(pitch_cfg.get("length_m", 105.0)),
            pitch_width_m=float(pitch_cfg.get("width_m", 68.0)),
            split_hints=split_hints,
            z_hints=z_hints,
            manual_obs_frames={
                fi for fi, anc in manual_by_frame.items()
                if anc.image_xy is not None
            },
            cfg=solver_cfg,
            world_fixes=fixes or None,
        )

        solver_name = str(cfg.get("solver", "piecewise"))
        mode_search_fallback = False
        if solver_name == "global":
            # The beam re-segments the timeline, so it wants the RICHER
            # permissive candidate set (soft-NMS, two events per window),
            # not the greedy-merged `events` the auto-anchor path uses —
            # otherwise the beam can't split where greedy merging hid a
            # breakpoint (the whole point of Phase 2).
            event_candidates = detect_event_candidates(
                steps=steps,
                confidences=raw_confidences,
                player_ctx=player_ctx,
                per_frame_K=per_frame_K, per_frame_R=per_frame_R,
                per_frame_t=per_frame_t, distortion=distortion,
                goal_geometry=goal_geometry,
                cfg=event_cfg,
                profile="permissive",
            )
            try:
                result = solve_modes(
                    **solve_kwargs,
                    mode_search_cfg=_mode_search_cfg(cfg),
                    player_ctx=player_ctx,
                    events=event_candidates,
                )
            except BudgetExceeded as exc:
                logger.warning(
                    "ball stage: global mode-search exceeded its fit budget "
                    "(%s) — falling back to whole-shot piecewise solve", exc,
                )
                mode_search_fallback = True
                result = solve_piecewise(**solve_kwargs)
            except Exception as exc:  # noqa: BLE001 — any global-solve failure falls back
                logger.warning(
                    "ball stage: global mode-search failed (%s) — falling "
                    "back to whole-shot piecewise solve", exc,
                )
                mode_search_fallback = True
                result = solve_piecewise(**solve_kwargs)
        else:
            result = solve_piecewise(**solve_kwargs)
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

        # Integrate a per-frame unit quaternion over the dense track so the
        # exported ball visibly rotates/curls. ω comes from physics (flight
        # Magnus spin / rolling-consistent ground spin), never a measured
        # ball-spin observation; see ball_orientation.integrate_orientation.
        quat_by_frame = integrate_orientation(
            per_frame_out, result.flight_segments,
            artifacts.camera_fps, ball_radius,
        )
        per_frame_out = [
            dataclasses.replace(bf, quat_wxyz=quat_by_frame.get(bf.frame))
            for bf in per_frame_out
        ]

        track = BallTrack(
            clip_id=artifacts.camera_clip_id,
            fps=artifacts.camera_fps,
            frames=tuple(per_frame_out),
            flight_segments=result.flight_segments,
        )
        track.save(ball_out_path)

        try:
            _emit_ball_keyframes(
                ball_out_path=ball_out_path,
                clip_id=artifacts.camera_clip_id,
                fps=artifacts.camera_fps,
                image_size=artifacts.camera_image_size,
                per_frame_out=per_frame_out,
                anchor_by_frame=anchor_by_frame,
                per_frame_K=per_frame_K,
                per_frame_R=per_frame_R,
                per_frame_t=per_frame_t,
                distortion=distortion,
                ground_touch_frames=ground_touch_frames,
                flight_segments=result.flight_segments,
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

        # cross_replay summary reflects THIS run's triangulation (None for
        # single-shot / disabled / no-partner). A stale fixes sidecar from a
        # prior run is removed when this run produced no fixes, so the
        # on-disk fixes never outlive the solve that consumed them.
        fixes_path = ball_out_path.with_name(
            ball_out_path.name.replace("ball_track", "ball_fixes")
        )
        if cr_summary is None and fixes_path.exists():
            try:
                fixes_path.unlink()
            except OSError:
                pass

        diag_path = ball_out_path.with_name(
            ball_out_path.name.replace("ball_track", "ball_diag")
        )
        diag: dict = {
            "solver": solver_name,
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
            "detection_coverage": artifacts.detection_coverage,
            "cross_replay": cr_summary,
        }
        if solver_name == "global":
            # Surface the beam search-effort metrics + the fallback flag so the
            # quality report / operator can see whether the global solve
            # completed or fell back to whole-shot piecewise (F8).
            diag["mode_search_fallback"] = mode_search_fallback
            diag["mode_search"] = result.diagnostics.get("mode_search", {})
        diag_path.write_text(json.dumps(diag, indent=2))
