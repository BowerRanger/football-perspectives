"""Ball stage: per-frame detection, IMM smoothing, ground projection,
and 3D trajectory reconstruction.

The stage owns the entire ball pipeline.  It reads only the clip video
files (via the shots manifest) and the camera track from earlier
stages; it does **not** read any ball detections from
``output/tracks/``.

Run flow per shot:

1. **Detect** — iterate the clip frames and ask the configured
   :class:`BallDetector` for ``(u, v, confidence)`` per frame.
2. **Smooth** — feed the per-frame detections through
   :class:`BallTracker`, a 2-mode IMM Kalman filter. Output includes a
   per-frame mode posterior ``p_flight`` and bounded gap-fill.
3. **Reconstruct 3D position** — ground-project each smoothed pixel to
   the world frame at ``z = ball_radius_m`` via
   :func:`src.utils.foot_anchor.ankle_ray_to_pitch`.
4. **Flight fit** — run-length encode frames where ``p_flight >= 0.5``;
   for each run run a parabola fit and (when the segment is long
   enough) a Magnus refinement. Accept Magnus only if it improves the
   pixel residual by ``ball.spin.min_residual_improvement`` and
   ``|ω|`` is within ``ball.spin.max_omega_rad_s``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import cv2
import numpy as np

from src.pipeline.base import BaseStage
from src.schemas.ball_track import BallFrame, BallTrack, FlightSegment
from src.schemas.camera_track import CameraTrack
from src.schemas.shots import ShotsManifest
from src.schemas.ball_anchor import BallAnchor, BallAnchorSet
from src.utils.ball_anchor_heights import (
    AIRBORNE_STATES,
    EVENT_STATES,
    HARD_KNOT_STATES,
    airborne_bucket_range,
    state_to_height,
)
from src.utils.ball_detector import BallDetector, YOLOBallDetector
from src.utils.ball_tracker import BallTracker, TrackerStep
from src.utils.ball_plausibility import (
    GroundPromotionCfg,
    GroundedRun,
    PitchDims,
    PlausibilityCfg,
    find_implausible_grounded_runs,
    is_plausible_trajectory,
)
from src.utils.bundle_adjust import (
    _integrate_magnus_positions,
    fit_magnus_trajectory,
    fit_parabola_to_image_observations,
)
from src.utils.ball_appearance_bridge import (
    AppearanceBridge,
    AppearanceBridgeCfg,
)
from src.utils.ball_kick_anchor import KickAnchorCfg, find_kick_anchor
from src.utils.foot_anchor import ankle_ray_to_pitch


logger = logging.getLogger(__name__)


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


def _demote_run_to_missing(
    per_frame_world: dict[int, tuple[np.ndarray, float]],
    a: int,
    b: int,
) -> None:
    """Drop world positions for frames [a, b] so they emit state='missing'."""
    for fi in range(a, b + 1):
        per_frame_world.pop(fi, None)


def _load_foot_uvs_for_shot(
    output_dir: Path, shot_id: str
) -> dict[int, list[tuple[float, float]]]:
    """Aggregate ankle pixel positions across all players for a shot.

    Reads ``output/hmr_world/<shot>__<player>_kp2d.json`` files (COCO-17
    keypoints; indices 15 = left_ankle, 16 = right_ankle). Returns a dict
    keyed by frame index with a list of ankle pixel positions, ignoring
    any with confidence below 0.3.
    """
    hmr_dir = output_dir / "hmr_world"
    if not hmr_dir.exists():
        return {}
    if shot_id:
        pattern = f"{shot_id}__*_kp2d.json"
    else:
        pattern = "*_kp2d.json"
    feet_by_frame: dict[int, list[tuple[float, float]]] = {}
    for path in hmr_dir.glob(pattern):
        try:
            payload = json.loads(path.read_text())
        except Exception:
            continue
        for entry in payload.get("frames", []):
            fi = int(entry.get("frame", -1))
            if fi < 0:
                continue
            kps = entry.get("keypoints", [])
            for idx in (15, 16):
                if idx >= len(kps):
                    continue
                kp = kps[idx]
                if len(kp) < 3 or kp[2] < 0.3:
                    continue
                feet_by_frame.setdefault(fi, []).append((float(kp[0]), float(kp[1])))
    return feet_by_frame


def _load_ball_anchors(
    output_dir: Path, shot_id: str
) -> dict[int, BallAnchor]:
    """Load per-frame ball anchors keyed by frame index.

    Returns an empty dict when no anchor file exists.
    """
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
            for shot in manifest.shots
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
        for shot in manifest.shots:
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
        tracker_cfg = cfg.get("tracker", {})
        spin_cfg = cfg.get("spin", {})
        max_residual = float(cfg.get("flight_max_residual_px", 5.0))
        plaus_cfg = PlausibilityCfg(
            z_max_m=float(cfg.get("plausibility", {}).get("z_max_m", 50.0)),
            horizontal_speed_max_m_s=float(cfg.get("plausibility", {}).get("horizontal_speed_max_m_s", 40.0)),
            pitch_margin_m=float(cfg.get("plausibility", {}).get("pitch_margin_m", 5.0)),
        )
        pitch_cfg = self.config.get("pitch", {})
        pitch_dims = PitchDims(
            length_m=float(pitch_cfg.get("length_m", 105.0)),
            width_m=float(pitch_cfg.get("width_m", 68.0)),
        )

        tracker = BallTracker(
            process_noise_grounded_px=float(tracker_cfg.get("process_noise_grounded_px", 4.0)),
            process_noise_flight_px=float(tracker_cfg.get("process_noise_flight_px", 12.0)),
            measurement_noise_px=float(tracker_cfg.get("measurement_noise_px", 2.0)),
            gating_sigma=float(tracker_cfg.get("gating_sigma", 4.0)),
            max_gap_frames=int(cfg.get("max_gap_frames", 6)),
            initial_p_flight=float(tracker_cfg.get("initial_p_flight", 0.1)),
        )

        feet_pixel_by_frame = _load_foot_uvs_for_shot(self.output_dir, shot_id)
        anchor_by_frame = _load_ball_anchors(self.output_dir, shot_id)
        if anchor_by_frame:
            logger.info(
                "ball stage: loaded %d anchors for shot %s",
                len(anchor_by_frame), shot_id or "(legacy)",
            )
        forced_flight: set[int] = {
            fi for fi, a in anchor_by_frame.items()
            if a.state in AIRBORNE_STATES
        }
        # Raw anchor pixels keyed by frame for exact world-position override
        # after the tracker loop. Off_screen_flight anchors have no pixel and
        # are excluded here.
        anchor_pixels: dict[int, tuple[float, float]] = {
            fi: (float(a.image_xy[0]), float(a.image_xy[1]))
            for fi, a in anchor_by_frame.items()
            if a.image_xy is not None
        }
        kick_cfg = KickAnchorCfg(
            enabled=bool(cfg.get("kick_anchor", {}).get("enabled", True))
                    and bool(feet_pixel_by_frame),
            max_pixel_distance_px=float(cfg.get("kick_anchor", {}).get("max_pixel_distance_px", 30.0)),
            lookahead_frames=int(cfg.get("kick_anchor", {}).get("lookahead_frames", 4)),
            min_pixel_acceleration_px_per_frame=float(cfg.get("kick_anchor", {}).get("min_pixel_acceleration_px_per_frame", 6.0)),
            foot_anchor_z_m=float(cfg.get("kick_anchor", {}).get("foot_anchor_z_m", 0.11)),
        )
        if not feet_pixel_by_frame and cfg.get("kick_anchor", {}).get("enabled", True):
            logger.warning(
                "ball stage: kick_anchor enabled but no kp2d sidecars found under %s",
                self.output_dir / "hmr_world",
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
                        # No pixel; let the IMM predict, but record the
                        # forced flight marker for the flight-run pass below.
                        uv: tuple[float, float] | None = None
                    else:
                        uv = (float(anchor.image_xy[0]), float(anchor.image_xy[1]))
                        raw_confidences[frame_idx] = 1.0
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
                    else:
                        consecutive_misses = 0
                        uv = (float(det[0]), float(det[1]))
                        raw_confidences[frame_idx] = float(det[2])
                        bridge.update_template(
                            frame=frame_idx,
                            frame_image=frame,
                            uv=uv,
                            confidence=float(det[2]),
                        )
                step = tracker.update(frame_idx, uv)
                steps.append(step)
                frame_idx += 1
        finally:
            cap.release()

        if frame_idx == 0:
            logger.warning("ball stage: clip %s contained no frames", clip_path)
            return

        n_frames = max(n_frames, frame_idx)

        # 3D ground projection of every smoothed step.
        per_frame_world: dict[int, tuple[np.ndarray, float]] = {}
        for step in steps:
            if step.uv is None:
                continue
            fi = step.frame
            if fi not in per_frame_K:
                continue
            try:
                world = ankle_ray_to_pitch(
                    step.uv,
                    K=per_frame_K[fi],
                    R=per_frame_R[fi],
                    t=per_frame_t[fi],
                    plane_z=ball_radius,
                    distortion=distortion,
                )
            except Exception as exc:
                logger.debug("ball ground projection failed at frame %d: %s", fi, exc)
                continue
            base_conf = raw_confidences.get(fi, 0.5)
            # Gap-filled frames have no direct detection — discount.
            conf = base_conf * (0.3 if step.is_gap_fill else 1.0)
            per_frame_world[fi] = (world, conf)

        # Override world position for anchored frames: project the exact
        # anchor pixel rather than the Kalman-smoothed tracker output so
        # that the user-supplied position is reproduced exactly.
        # Use the anchor STATE's height plane — airborne_high at z=0.11
        # would project the airborne ball's pixel onto the ground, which
        # lands far past the ball's true XY.
        for fi, uv_anchor in anchor_pixels.items():
            if fi not in per_frame_K:
                continue
            anchor_state = anchor_by_frame[fi].state
            try:
                plane_z = state_to_height(anchor_state)
            except ValueError:
                plane_z = ball_radius
            try:
                world = ankle_ray_to_pitch(
                    uv_anchor,
                    K=per_frame_K[fi],
                    R=per_frame_R[fi],
                    t=per_frame_t[fi],
                    plane_z=plane_z,
                    distortion=distortion,
                )
            except Exception as exc:
                logger.debug("ball anchor projection failed at frame %d: %s", fi, exc)
                continue
            per_frame_world[fi] = (world, 1.0)

        # Flight segmentation by IMM mode posterior.
        min_flight = int(tracker_cfg.get("min_flight_frames", 6))
        max_flight = int(tracker_cfg.get("max_flight_frames", 90))
        flight_runs = self._flight_runs(steps, min_flight, max_flight)

        # Layer 5 — event-splitting: kick/catch/bounce anchors split flight runs.
        # Semantics: a cut at the run start (cut == a_run) keeps the event
        # frame as the start of the remaining segment (e.g. kick starts a new
        # arc here); a cut strictly inside the run excludes the event frame
        # from both sub-runs (e.g. bounce frame is ground contact, not in air).
        if anchor_by_frame:
            event_frames = sorted(
                fi for fi, a_ev in anchor_by_frame.items()
                if a_ev.state in EVENT_STATES
            )
            if event_frames:
                split_runs: list[tuple[int, int]] = []
                for (a_run, b_run) in flight_runs:
                    cuts = [fi for fi in event_frames if a_run <= fi <= b_run]
                    if not cuts:
                        split_runs.append((a_run, b_run))
                        continue
                    prev = a_run
                    for cut in cuts:
                        if cut - 1 >= prev:
                            split_runs.append((prev, cut - 1))
                        # When the cut is at the very start of the run there is
                        # no pre-segment; keep the event frame as the new start
                        # so that kick anchors remain in their flight segment.
                        prev = cut if cut == a_run else cut + 1
                    if prev <= b_run:
                        split_runs.append((prev, b_run))
                flight_runs = split_runs

        flight_segments: list[FlightSegment] = []
        flight_membership: dict[int, int] = {}
        spin_enabled = bool(spin_cfg.get("enabled", True))
        spin_min_seconds = float(spin_cfg.get("min_flight_seconds", 0.5))
        spin_min_improve = float(spin_cfg.get("min_residual_improvement", 0.2))
        spin_max_omega = float(spin_cfg.get("max_omega_rad_s", 200.0))
        drag = float(spin_cfg.get("drag_k_over_m", 0.005))
        g = -9.81
        g_vec = np.array([0.0, 0.0, g])

        for sid, (a, b) in enumerate(flight_runs):
            obs_pairs = [
                (fi, steps[fi].uv)
                for fi in range(a, b + 1)
                if steps[fi].uv is not None and fi in per_frame_K
            ]
            if len(obs_pairs) < min_flight:
                continue
            obs = [(o[0], (float(o[1][0]), float(o[1][1]))) for o in obs_pairs]
            Ks_seg = [per_frame_K[o[0]] for o in obs]
            Rs_seg = [per_frame_R[o[0]] for o in obs]
            ts_seg = [per_frame_t[o[0]] for o in obs]

            ball_uvs_seg = {fi: uv for fi, uv in obs}
            anchor_world: np.ndarray | None = None
            if kick_cfg.enabled:
                # Pick the nearest foot per frame in the segment seed region.
                seed_feet: dict[int, tuple[float, float]] = {}
                for fi in range(a, min(a + kick_cfg.lookahead_frames + 1, b + 1)):
                    feet = feet_pixel_by_frame.get(fi, [])
                    if not feet or fi not in ball_uvs_seg:
                        continue
                    bu, bv = ball_uvs_seg[fi]
                    nearest = min(feet, key=lambda f: (f[0] - bu) ** 2 + (f[1] - bv) ** 2)
                    seed_feet[fi] = nearest
                if a in per_frame_K:
                    anchor_world = find_kick_anchor(
                        segment_start_frame=a,
                        ball_uvs=ball_uvs_seg,
                        foot_uvs_by_frame=seed_feet,
                        K=per_frame_K[a],
                        R=per_frame_R[a],
                        t=per_frame_t[a],
                        cfg=kick_cfg,
                        distortion=distortion,
                    )

            # Layer 5 — hard knots from anchored frames within this segment.
            knot_frames_arg: dict[int, np.ndarray] = {}
            for fi in range(a, b + 1):
                anc = anchor_by_frame.get(fi)
                if anc is None or anc.state not in HARD_KNOT_STATES:
                    continue
                if anc.image_xy is None:
                    continue
                if fi not in per_frame_K:
                    continue
                z = state_to_height(anc.state)
                world_at_anchor = ankle_ray_to_pitch(
                    anc.image_xy,
                    K=per_frame_K[fi], R=per_frame_R[fi], t=per_frame_t[fi],
                    plane_z=z, distortion=distortion,
                )
                knot_frames_arg[fi - a] = np.asarray(world_at_anchor, dtype=float)

            # If the seed frame is a hard knot AND Layer 3 didn't set
            # anchor_world, promote frame 0 to p0_fixed.
            if 0 in knot_frames_arg and anchor_world is None:
                anchor_world = knot_frames_arg.pop(0)

            try:
                p0, v0, parab_resid = fit_parabola_to_image_observations(
                    obs, Ks=Ks_seg, Rs=Rs_seg, t_world=ts_seg,
                    fps=camera.fps, distortion=distortion,
                    p0_fixed=anchor_world,
                    knot_frames=knot_frames_arg or None,
                )
            except Exception as exc:
                logger.debug("parabola fit failed on segment %d: %s", sid, exc)
                continue
            if parab_resid > max_residual:
                continue
            segment_duration_s = (b - a) / camera.fps
            if not is_plausible_trajectory(
                p0, v0, omega=None,
                duration_s=segment_duration_s, fps=camera.fps,
                cfg=plaus_cfg, pitch=pitch_dims,
            ):
                logger.info(
                    "ball seg %d (%d-%d): parabola failed plausibility, dropping",
                    sid, a, b,
                )
                continue

            spin_axis: list[float] | None = None
            spin_omega: float | None = None
            spin_confidence: float | None = None
            effective_p0, effective_v0 = p0, v0
            effective_resid = parab_resid
            omega_world: np.ndarray | None = None

            duration_s = (b - a) / camera.fps
            if spin_enabled and duration_s >= spin_min_seconds:
                try:
                    mp0, mv0, momega, magnus_resid = fit_magnus_trajectory(
                        obs,
                        Ks=Ks_seg, Rs=Rs_seg, t_world=ts_seg,
                        fps=camera.fps,
                        drag_k_over_m=drag,
                        p0_seed=p0, v0_seed=v0,
                        p0_fixed=anchor_world,
                    )
                except Exception as exc:
                    logger.debug("magnus fit failed on segment %d: %s", sid, exc)
                else:
                    omega_mag = float(np.linalg.norm(momega))
                    improvement = (
                        (parab_resid - magnus_resid) / parab_resid
                        if parab_resid > 0 else 0.0
                    )
                    magnus_plausible = is_plausible_trajectory(
                        mp0, mv0, omega=momega,
                        duration_s=duration_s, fps=camera.fps,
                        cfg=plaus_cfg, pitch=pitch_dims,
                    )
                    if (
                        omega_mag > 0
                        and omega_mag <= spin_max_omega
                        and improvement >= spin_min_improve
                        and magnus_plausible
                    ):
                        spin_axis = list((momega / omega_mag).astype(float))
                        spin_omega = omega_mag
                        # 0.2 improvement on a 0.5s segment → ~0.4 confidence;
                        # 0.5 improvement on 1.0s → ~1.0.
                        duration_factor = min(1.0, duration_s / 1.0)
                        spin_confidence = float(min(1.0, (improvement / 0.5) * duration_factor))
                        effective_p0, effective_v0 = mp0, mv0
                        effective_resid = magnus_resid
                        omega_world = momega

            # Replace per-frame world_xyz inside the flight with the fitted
            # trajectory evaluation. Preserves original BallStage behaviour
            # and gives clean curves through the gltf export.
            for fi in range(a, b + 1):
                if fi not in per_frame_K:
                    continue
                flight_membership[fi] = sid
                dt = (fi - a) / camera.fps
                if omega_world is not None:
                    positions = _integrate_magnus_positions(
                        effective_p0,
                        effective_v0,
                        omega_world,
                        g_vec,
                        drag,
                        np.array([0.0, dt]),
                    )
                    pos = positions[-1]
                else:
                    pos = effective_p0 + effective_v0 * dt + 0.5 * g_vec * dt ** 2
                prev_conf = per_frame_world.get(fi, (None, 0.5))[1]
                per_frame_world[fi] = (pos, prev_conf)

            flight_segments.append(
                FlightSegment(
                    id=sid,
                    frame_range=(a, b),
                    parabola={
                        "p0": [float(x) for x in effective_p0],
                        "v0": [float(x) for x in effective_v0],
                        "g": g,
                        "spin_axis_world": spin_axis,
                        "spin_omega_rad_s": spin_omega,
                        "spin_confidence": spin_confidence,
                    },
                    fit_residual_px=effective_resid,
                )
            )

        promote_cfg = GroundPromotionCfg(
            enabled=bool(cfg.get("flight_promotion", {}).get("enabled", True)),
            min_run_frames=int(cfg.get("flight_promotion", {}).get("min_run_frames", 6)),
            off_pitch_margin_m=float(cfg.get("flight_promotion", {}).get("off_pitch_margin_m", 5.0)),
            max_ground_speed_m_s=float(cfg.get("flight_promotion", {}).get("max_ground_speed_m_s", 35.0)),
        )

        # Provisional state map matching what would be emitted below.
        provisional_state: dict[int, str] = {}
        for fi in range(n_frames):
            if fi in per_frame_world:
                provisional_state[fi] = "flight" if fi in flight_membership else "grounded"
            else:
                provisional_state[fi] = "missing"

        runs_to_promote = find_implausible_grounded_runs(
            per_frame_xyz=per_frame_world,
            per_frame_state=provisional_state,
            fps=camera.fps,
            cfg=promote_cfg,
            pitch=pitch_dims,
        )

        next_segment_id = (max(flight_membership.values()) + 1) if flight_membership else 0
        min_flight_frames_for_refit = int(tracker_cfg.get("min_flight_frames", 6))
        for run in runs_to_promote:
            obs_pairs = [
                (fi, steps[fi].uv) for fi in range(run.start, run.end + 1)
                if 0 <= fi < len(steps) and steps[fi].uv is not None and fi in per_frame_K
            ]
            if len(obs_pairs) < min_flight_frames_for_refit:
                continue
            obs = [(o[0], (float(o[1][0]), float(o[1][1]))) for o in obs_pairs]
            Ks_seg = [per_frame_K[o[0]] for o in obs]
            Rs_seg = [per_frame_R[o[0]] for o in obs]
            ts_seg = [per_frame_t[o[0]] for o in obs]
            try:
                p0, v0, parab_resid = fit_parabola_to_image_observations(
                    obs, Ks=Ks_seg, Rs=Rs_seg, t_world=ts_seg,
                    fps=camera.fps, distortion=distortion,
                )
            except Exception as exc:
                # Refit failure means the data isn't actually a clean
                # flight arc — the original ground projection (noisy but
                # bounded) is a better fallback than nothing.
                logger.debug("promotion refit failed at run %d-%d: %s — leaving as grounded",
                             run.start, run.end, exc)
                continue
            seg_duration = (run.end - run.start) / camera.fps
            if not is_plausible_trajectory(
                p0, v0, omega=None,
                duration_s=seg_duration, fps=camera.fps,
                cfg=plaus_cfg, pitch=pitch_dims,
            ):
                logger.info(
                    "ball: promotion refit for run %d-%d failed plausibility; "
                    "leaving as grounded",
                    run.start, run.end,
                )
                continue

            sid_new = next_segment_id
            next_segment_id += 1
            for fi in range(run.start, run.end + 1):
                if fi not in per_frame_K:
                    continue
                dt = (fi - run.start) / camera.fps
                pos = p0 + v0 * dt + 0.5 * g_vec * dt ** 2
                prev_conf = per_frame_world.get(fi, (None, 0.5))[1]
                per_frame_world[fi] = (pos, prev_conf)
                flight_membership[fi] = sid_new
            flight_segments.append(
                FlightSegment(
                    id=sid_new,
                    frame_range=(run.start, run.end),
                    parabola={
                        "p0": [float(x) for x in p0],
                        "v0": [float(x) for x in v0],
                        "g": g,
                        "spin_axis_world": None,
                        "spin_omega_rad_s": None,
                        "spin_confidence": None,
                    },
                    fit_residual_px=parab_resid,
                )
            )

        # Layer 5: forced-flight frames from airborne_* / off_screen_flight
        # anchors. We do NOT create FlightSegment entries here — the user
        # marked the frame airborne but we have no parabola data to fit
        # (single-frame runs are not real flights). Instead the BallFrame
        # assembly below uses the `forced_flight` set directly to classify
        # those frames as state="flight". Avoids polluting the segments
        # table with zero-parabola placeholders.

        # Layer 5: linear interpolation between consecutive grounded
        # anchors. When two grounded anchors are separated by frames
        # with NO anchors of any kind in between, the world position at
        # every intermediate frame is linearly interpolated from the
        # two anchor ray-casts. This overrides WASB-driven positions
        # which are noisy and frequently project off-pitch.
        if anchor_by_frame:
            grounded_anchor_frames = sorted(
                fi for fi, a in anchor_by_frame.items()
                if a.state == "grounded" and a.image_xy is not None
            )
            for i in range(len(grounded_anchor_frames) - 1):
                fa = grounded_anchor_frames[i]
                fb = grounded_anchor_frames[i + 1]
                if fb - fa <= 1:
                    continue
                # Skip if any other anchor (of any kind) lies strictly
                # between fa and fb — those would interrupt the smooth
                # grounded path.
                if any(fa < fi < fb for fi in anchor_by_frame.keys() if fi != fa and fi != fb):
                    continue
                wa = per_frame_world.get(fa)
                wb = per_frame_world.get(fb)
                if wa is None or wb is None:
                    continue
                pa, _ = wa
                pb, _ = wb
                span = fb - fa
                for fi in range(fa + 1, fb):
                    t = (fi - fa) / span
                    pos = pa * (1.0 - t) + pb * t
                    # Confidence reflects "interpolated", high but not 1.0.
                    per_frame_world[fi] = (pos, 0.9)
                    # Force out of any flight membership the WASB path
                    # might have assigned (rare but possible).
                    flight_membership.pop(fi, None)

        # Layer 5 Phase 2: parabola fit through maximal non-grounded
        # anchor spans. For each contiguous run of non-grounded anchors
        # (no grounded anchor between them), gather all anchor pixels
        # as observations and all anchor-state heights as soft knots,
        # then fit a parabola. If the fit is plausible, fill the span's
        # unanchored frames with the parabola's per-frame evaluation
        # and add a real FlightSegment. The linear-interp pass below
        # serves as the fallback when the parabola fit fails or the
        # span has too few pixels.
        parabola_handled_spans: list[tuple[int, int]] = []
        if anchor_by_frame:
            ordered_for_spans = sorted(anchor_by_frame.items(), key=lambda kv: kv[0])
            _NON_GROUNDED = AIRBORNE_STATES | EVENT_STATES
            spans: list[list[tuple[int, BallAnchor]]] = []
            # Span boundary rules:
            #   grounded  → close current span (state change).
            #   kick      → close current AND start a new span (kick is
            #               the start of a fresh flight).
            #   bounce    → close current including the bounce (flight
            #   catch       ends here; subsequent flight needs a fresh kick).
            #   header    → continues current (mid-flight deflection).
            #   airborne_*, off_screen_flight → continues current.
            current_span: list[tuple[int, BallAnchor]] = []
            for fi, anc in ordered_for_spans:
                if anc.state == "grounded":
                    if len(current_span) >= 2:
                        spans.append(current_span)
                    current_span = []
                elif anc.state == "kick":
                    if len(current_span) >= 2:
                        spans.append(current_span)
                    current_span = [(fi, anc)]
                elif anc.state in ("bounce", "catch"):
                    current_span.append((fi, anc))
                    if len(current_span) >= 2:
                        spans.append(current_span)
                    current_span = []
                elif anc.state in _NON_GROUNDED:
                    current_span.append((fi, anc))
                else:
                    if len(current_span) >= 2:
                        spans.append(current_span)
                    current_span = []
            if len(current_span) >= 2:
                spans.append(current_span)

            for span in spans:
                fa_span = span[0][0]
                fb_span = span[-1][0]
                # Build obs from every anchor pixel in the span. Only
                # HARD_KNOT_STATES contribute to knot_frames — airborne
                # bucket heights (1/6/15 m) are too coarse to pin Z,
                # and using them as soft constraints would pull the fit
                # toward the wrong height when the user picks the wrong
                # bucket. The pixel observations alone constrain XY
                # along each camera ray; gravity + hard knots constrain
                # Z.
                obs_p2: list[tuple[int, tuple[float, float]]] = []
                Ks_p2: list[np.ndarray] = []
                Rs_p2: list[np.ndarray] = []
                ts_p2: list[np.ndarray] = []
                knots: dict[int, np.ndarray] = {}
                z_ranges: dict[int, tuple[float, float]] = {}
                p0_pin: np.ndarray | None = None
                for fi, anc in span:
                    if anc.image_xy is None or fi not in per_frame_K:
                        continue
                    obs_p2.append((fi, (float(anc.image_xy[0]), float(anc.image_xy[1]))))
                    Ks_p2.append(per_frame_K[fi])
                    Rs_p2.append(per_frame_R[fi])
                    ts_p2.append(per_frame_t[fi])
                    rel = fi - fa_span
                    if anc.state in HARD_KNOT_STATES:
                        try:
                            world_at_anchor = ankle_ray_to_pitch(
                                anc.image_xy,
                                K=per_frame_K[fi], R=per_frame_R[fi], t=per_frame_t[fi],
                                plane_z=state_to_height(anc.state),
                                distortion=distortion,
                            )
                        except Exception:
                            continue
                        knots[rel] = np.asarray(world_at_anchor, dtype=float)
                    else:
                        # airborne_low/mid/high → one-sided Z hinge that
                        # forces z into the bucket range but lets the
                        # pixel obs determine the exact value inside it.
                        bucket = airborne_bucket_range(anc.state)
                        if bucket is not None:
                            z_ranges[rel] = bucket
                # Need at least 3 obs to make a parabola fit meaningful.
                if len(obs_p2) < 3:
                    continue
                # If the span's start frame is a hard-knot anchor (kick),
                # promote it to p0_fixed so the segment origin is pinned.
                if 0 in knots and span[0][1].state in HARD_KNOT_STATES:
                    p0_pin = knots.pop(0)
                try:
                    p2_p0, p2_v0, p2_resid = fit_parabola_to_image_observations(
                        obs_p2, Ks=Ks_p2, Rs=Rs_p2, t_world=ts_p2,
                        fps=camera.fps, distortion=distortion,
                        p0_fixed=p0_pin, knot_frames=knots or None,
                        z_range_frames=z_ranges or None,
                    )
                except Exception as exc:
                    logger.debug(
                        "Phase 2 parabola fit failed for span %d-%d: %s",
                        fa_span, fb_span, exc,
                    )
                    continue
                duration_s = (fb_span - fa_span) / camera.fps
                if duration_s <= 0 or not is_plausible_trajectory(
                    p2_p0, p2_v0, omega=None,
                    duration_s=duration_s, fps=camera.fps,
                    cfg=plaus_cfg, pitch=pitch_dims,
                ):
                    logger.info(
                        "Phase 2 parabola fit for span %d-%d failed plausibility — "
                        "falling back to linear interp",
                        fa_span, fb_span,
                    )
                    continue
                # Success: drop any pre-existing flight segments inside
                # this span (the parabola we just fit is the authoritative
                # answer) and emit a single new segment.
                surviving: list[FlightSegment] = []
                for seg in flight_segments:
                    seg_a, seg_b = seg.frame_range
                    if seg_a >= fa_span and seg_b <= fb_span:
                        for fi in range(seg_a, seg_b + 1):
                            if flight_membership.get(fi) == seg.id:
                                flight_membership.pop(fi, None)
                        continue
                    surviving.append(seg)
                flight_segments[:] = surviving

                sid_new = (max(flight_membership.values()) + 1) if flight_membership else 0
                # Avoid colliding with any segment IDs still in the list.
                existing_ids = {s.id for s in flight_segments}
                while sid_new in existing_ids:
                    sid_new += 1
                g_vec = np.array([0.0, 0.0, -9.81])
                # Inside a successful Phase 2 span the parabola is the
                # authoritative answer for every frame — including the
                # anchored ones, whose pixels constrained the fit. The
                # per-anchor ray-cast at bucket heights would otherwise
                # land 10s of metres off the truth for coarse buckets.
                for fi in range(fa_span, fb_span + 1):
                    forced_flight.add(fi)
                    flight_membership[fi] = sid_new
                    dt_k = (fi - fa_span) / camera.fps
                    pos = (
                        (p0_pin if p0_pin is not None else p2_p0)
                        + p2_v0 * dt_k + 0.5 * (dt_k ** 2) * g_vec
                    )
                    per_frame_world[fi] = (pos, 0.92)
                flight_segments.append(FlightSegment(
                    id=sid_new,
                    frame_range=(fa_span, fb_span),
                    parabola={
                        "p0": [float(x) for x in (p0_pin if p0_pin is not None else p2_p0)],
                        "v0": [float(x) for x in p2_v0],
                        "g": -9.81,
                        "spin_axis_world": None,
                        "spin_omega_rad_s": None,
                        "spin_confidence": None,
                    },
                    fit_residual_px=p2_resid,
                ))
                parabola_handled_spans.append((fa_span, fb_span))

        # Layer 5 Phase 1: flight-span LINEAR interpolation — fallback
        # when Phase 2 didn't handle the span (fit raised, plausibility
        # failed, or fewer than 3 anchor pixels). When consecutive
        # anchors are BOTH non-grounded with no grounded anchor between
        # them, linearly interpolate world XYZ between the two anchor
        # ray-casts (each at its own state height) and mark every
        # intermediate frame as flight.
        def _in_handled_span(fi: int) -> bool:
            for sa, sb in parabola_handled_spans:
                if sa <= fi <= sb:
                    return True
            return False

        if anchor_by_frame:
            ordered = sorted(anchor_by_frame.items(), key=lambda kv: kv[0])
            _NON_GROUNDED = AIRBORNE_STATES | EVENT_STATES
            for (fa, anc_a), (fb, anc_b) in zip(ordered, ordered[1:]):
                if anc_a.state == "grounded" or anc_b.state == "grounded":
                    continue
                if anc_a.state not in _NON_GROUNDED or anc_b.state not in _NON_GROUNDED:
                    continue
                if fb - fa <= 1:
                    continue
                if _in_handled_span(fa) and _in_handled_span(fb):
                    continue
                for fi in range(fa + 1, fb):
                    forced_flight.add(fi)
                    flight_membership.pop(fi, None)
                wa = per_frame_world.get(fa)
                wb = per_frame_world.get(fb)
                if wa is None or wb is None:
                    continue
                pa, _ = wa
                pb, _ = wb
                span_len = fb - fa
                for fi in range(fa + 1, fb):
                    t = (fi - fa) / span_len
                    pos = pa * (1.0 - t) + pb * t
                    per_frame_world[fi] = (pos, 0.85)

        per_frame_out: list[BallFrame] = []
        for fi in range(n_frames):
            in_flight = fi in flight_membership or fi in forced_flight
            if fi in per_frame_world:
                world, conf = per_frame_world[fi]
                state = "flight" if in_flight else "grounded"
                per_frame_out.append(
                    BallFrame(
                        frame=fi,
                        world_xyz=tuple(float(x) for x in world),
                        state=state,
                        confidence=float(conf),
                        flight_segment_id=flight_membership.get(fi),
                    )
                )
            else:
                if in_flight:
                    per_frame_out.append(
                        BallFrame(
                            frame=fi,
                            world_xyz=None,
                            state="flight",
                            confidence=0.0,
                            flight_segment_id=flight_membership.get(fi),
                        )
                    )
                else:
                    per_frame_out.append(
                        BallFrame(
                            frame=fi,
                            world_xyz=None,
                            state="missing",
                            confidence=0.0,
                        )
                    )

        track = BallTrack(
            clip_id=camera.clip_id,
            fps=camera.fps,
            frames=tuple(per_frame_out),
            flight_segments=tuple(flight_segments),
        )
        track.save(ball_out_path)

    @staticmethod
    def _flight_runs(
        steps: list[TrackerStep], min_flight: int, max_flight: int
    ) -> list[tuple[int, int]]:
        """Run-length encode frames with ``p_flight >= 0.5``.

        Returns ``(start_frame, end_frame)`` pairs, both inclusive.
        Runs shorter than ``min_flight`` or longer than ``max_flight``
        are dropped (long runs are likely tracker confusion, not real
        flights).
        """
        runs: list[tuple[int, int]] = []
        start: int | None = None
        for step in steps:
            in_flight = step.p_flight >= 0.5 and step.uv is not None
            if in_flight and start is None:
                start = step.frame
            elif not in_flight and start is not None:
                end = step.frame - 1
                if min_flight <= (end - start + 1) <= max_flight:
                    runs.append((start, end))
                start = None
        if start is not None and steps:
            end = steps[-1].frame
            if min_flight <= (end - start + 1) <= max_flight:
                runs.append((start, end))
        return runs
