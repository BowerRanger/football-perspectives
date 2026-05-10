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

import logging
from pathlib import Path

import cv2
import numpy as np

from src.pipeline.base import BaseStage
from src.schemas.ball_track import BallFrame, BallTrack, FlightSegment
from src.schemas.camera_track import CameraTrack
from src.schemas.shots import ShotsManifest
from src.utils.ball_detector import BallDetector, YOLOBallDetector
from src.utils.ball_tracker import BallTracker, TrackerStep
from src.utils.bundle_adjust import (
    _integrate_magnus_positions,
    fit_magnus_trajectory,
    fit_parabola_to_image_observations,
)
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

        tracker = BallTracker(
            process_noise_grounded_px=float(tracker_cfg.get("process_noise_grounded_px", 4.0)),
            process_noise_flight_px=float(tracker_cfg.get("process_noise_flight_px", 12.0)),
            measurement_noise_px=float(tracker_cfg.get("measurement_noise_px", 2.0)),
            gating_sigma=float(tracker_cfg.get("gating_sigma", 4.0)),
            max_gap_frames=int(cfg.get("max_gap_frames", 6)),
        )

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
                det = detector.detect(frame)
                if det is None:
                    uv: tuple[float, float] | None = None
                else:
                    uv = (float(det[0]), float(det[1]))
                    raw_confidences[frame_idx] = float(det[2])
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

        # Flight segmentation by IMM mode posterior.
        min_flight = int(tracker_cfg.get("min_flight_frames", 6))
        max_flight = int(tracker_cfg.get("max_flight_frames", 90))
        flight_runs = self._flight_runs(steps, min_flight, max_flight)

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

            try:
                p0, v0, parab_resid = fit_parabola_to_image_observations(
                    obs, Ks=Ks_seg, Rs=Rs_seg, t_world=ts_seg,
                    fps=camera.fps, distortion=distortion,
                )
            except Exception as exc:
                logger.debug("parabola fit failed on segment %d: %s", sid, exc)
                continue
            if parab_resid > max_residual:
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
                    )
                except Exception as exc:
                    logger.debug("magnus fit failed on segment %d: %s", sid, exc)
                else:
                    omega_mag = float(np.linalg.norm(momega))
                    improvement = (
                        (parab_resid - magnus_resid) / parab_resid
                        if parab_resid > 0 else 0.0
                    )
                    if (
                        omega_mag > 0
                        and omega_mag <= spin_max_omega
                        and improvement >= spin_min_improve
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

        per_frame_out: list[BallFrame] = []
        for fi in range(n_frames):
            if fi in per_frame_world:
                world, conf = per_frame_world[fi]
                state = "flight" if fi in flight_membership else "grounded"
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
