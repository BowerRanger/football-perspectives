"""Ball stage: ground projection + parabolic flight reconstruction.

Reads:
    - ``output/camera/camera_track.json`` (per-frame K, R; clip-shared t_world)
    - ``output/tracks/ball_track.json`` (raw 2D ball detections; ``frames``
      list with ``frame``, ``bbox_centre``, ``confidence``).

Writes:
    - ``output/ball/ball_track.json`` (BallTrack JSON: per-frame world_xyz,
      state in {grounded, flight, occluded, missing}, plus FlightSegment
      entries for each accepted parabolic fit).

Algorithm:
    1.  For every detected frame, project the bbox centre to the ground
        plane at ``z = ball_radius`` (default 0.11 m).
    2.  Compute pixel velocity vs the previous detected frame.  Runs of
        frames with velocity >= ``flight_px_velocity`` and length in
        ``[min_flight_frames, max_flight_frames]`` are flight candidates.
    3.  For each candidate run an LM parabola fit; if the per-pixel
        residual < ``flight_max_residual_px``, replace the per-frame
        ground projection with the parabola evaluation and tag the
        frames as ``state="flight"``.
    4.  Frames in the camera span without a detection emit
        ``state="missing"``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.pipeline.base import BaseStage
from src.schemas.ball_track import BallFrame, BallTrack, FlightSegment
from src.schemas.camera_track import CameraTrack
from src.utils.bundle_adjust import fit_parabola_to_image_observations
from src.utils.foot_anchor import ankle_ray_to_pitch


class BallStage(BaseStage):
    name = "ball"

    def is_complete(self) -> bool:
        from src.schemas.shots import ShotsManifest
        manifest_path = self.output_dir / "shots" / "shots_manifest.json"
        if not manifest_path.exists():
            return (self.output_dir / "ball" / "ball_track.json").exists()
        manifest = ShotsManifest.load(manifest_path)
        return all(
            (self.output_dir / "ball" / f"{shot.id}_ball_track.json").exists()
            for shot in manifest.shots
        )

    def run(self) -> None:
        from src.schemas.shots import ShotsManifest
        cfg = self.config.get("ball", {})
        manifest_path = self.output_dir / "shots" / "shots_manifest.json"
        if not manifest_path.exists():
            # Legacy single-shot path. If both legacy files exist, run
            # against them; otherwise fail with the same message as before.
            cam_path = self.output_dir / "camera" / "camera_track.json"
            ball_in = self.output_dir / "tracks" / "ball_track.json"
            ball_out = self.output_dir / "ball" / "ball_track.json"
            if cam_path.exists() and ball_in.exists():
                self._run_shot("", cam_path, ball_in, ball_out, cfg)
                return
            raise FileNotFoundError(
                f"ball stage requires manifest at {manifest_path}; run prepare_shots first"
            )
        manifest = ShotsManifest.load(manifest_path)
        shot_filter = getattr(self, "shot_filter", None)
        for shot in manifest.shots:
            if shot_filter is not None and shot.id != shot_filter:
                continue
            cam_path = self.output_dir / "camera" / f"{shot.id}_camera_track.json"
            ball_in = self.output_dir / "tracks" / f"{shot.id}_ball_track.json"
            ball_out = self.output_dir / "ball" / f"{shot.id}_ball_track.json"
            if not cam_path.exists():
                logger = __import__("logging").getLogger(__name__)
                logger.warning(
                    "ball stage skipping shot %s — no camera_track at %s",
                    shot.id, cam_path,
                )
                continue
            if not ball_in.exists():
                logger = __import__("logging").getLogger(__name__)
                logger.warning(
                    "ball stage skipping shot %s — no raw ball detections at %s",
                    shot.id, ball_in,
                )
                continue
            self._run_shot(shot.id, cam_path, ball_in, ball_out, cfg)

    def _run_shot(
        self,
        shot_id: str,
        camera_path: Path,
        ball_in_path: Path,
        ball_out_path: Path,
        cfg: dict,
    ) -> None:
        """Single-shot ball reconstruction. Body is the original run()'s
        logic with input/output paths parameterised on shot_id."""
        camera = CameraTrack.load(camera_path)
        per_frame_K = {f.frame: np.array(f.K) for f in camera.frames}
        per_frame_R = {f.frame: np.array(f.R) for f in camera.frames}
        # Per-frame t when present (current camera stage always writes it).
        # Fall back to clip-shared t_world for legacy tracks.
        t_world_fallback = np.array(camera.t_world)
        per_frame_t = {
            f.frame: (np.array(f.t) if f.t is not None else t_world_fallback)
            for f in camera.frames
        }
        distortion = camera.distortion

        with ball_in_path.open() as fh:
            ball_input = json.load(fh)

        ball_radius = float(cfg.get("ball_radius_m", 0.11))
        flight_velocity = float(cfg.get("flight_px_velocity", 25.0))
        min_flight = int(cfg.get("min_flight_frames", 4))
        max_flight = int(cfg.get("max_flight_frames", 60))
        max_residual = float(cfg.get("flight_max_residual_px", 5.0))

        observations = sorted(ball_input["frames"], key=lambda f: f["frame"])
        n_frames = max(f.frame for f in camera.frames) + 1
        provisional: dict[int, tuple[tuple[float, float], np.ndarray, float]] = {}
        prev_uv: tuple[float, float] | None = None
        velocities: dict[int, float] = {}
        for f in observations:
            fi = int(f["frame"])
            uv = tuple(f["bbox_centre"])
            if fi not in per_frame_K:
                continue
            world = ankle_ray_to_pitch(
                uv,
                K=per_frame_K[fi],
                R=per_frame_R[fi],
                t=per_frame_t[fi],
                plane_z=ball_radius,
                distortion=distortion,
            )
            provisional[fi] = (uv, world, float(f.get("confidence", 0.5)))
            if prev_uv is not None:
                velocities[fi] = float(
                    np.linalg.norm(np.array(uv) - np.array(prev_uv))
                )
            prev_uv = uv

        # Flight segmentation by pixel velocity.
        candidate: list[int] = []
        segments: list[tuple[int, int]] = []
        for fi in sorted(velocities):
            if velocities[fi] >= flight_velocity:
                candidate.append(fi)
            else:
                if min_flight <= len(candidate) <= max_flight:
                    segments.append((candidate[0], candidate[-1]))
                candidate = []
        if min_flight <= len(candidate) <= max_flight:
            segments.append((candidate[0], candidate[-1]))

        # Fit each segment via LM; accept if residual gate passes.
        flight_outs: list[FlightSegment] = []
        flight_membership: dict[int, int] = {}
        for sid, (a, b) in enumerate(segments):
            obs = [
                (fi, provisional[fi][0])
                for fi in range(a, b + 1)
                if fi in provisional
            ]
            if len(obs) < min_flight:
                continue
            p0, v0, residual = fit_parabola_to_image_observations(
                obs,
                Ks=[per_frame_K.get(o[0], np.eye(3)) for o in obs],
                Rs=[per_frame_R.get(o[0], np.eye(3)) for o in obs],
                t_world=[per_frame_t.get(o[0], t_world_fallback) for o in obs],
                fps=camera.fps,
                distortion=distortion,
            )
            if residual > max_residual:
                continue
            for fi, _ in obs:
                flight_membership[fi] = sid
                dt = (fi - a) / camera.fps
                world = (
                    p0
                    + v0 * dt
                    + 0.5 * np.array([0.0, 0.0, -9.81]) * dt ** 2
                )
                provisional[fi] = (provisional[fi][0], world, provisional[fi][2])
            flight_outs.append(
                FlightSegment(
                    id=sid,
                    frame_range=(a, b),
                    parabola={"p0": list(p0), "v0": list(v0), "g": -9.81},
                    fit_residual_px=residual,
                )
            )

        # Assemble per-frame output across the camera span.
        per_frame: list[BallFrame] = []
        for fi in range(n_frames):
            if fi in provisional:
                _, world, conf = provisional[fi]
                state = "flight" if fi in flight_membership else "grounded"
                per_frame.append(
                    BallFrame(
                        frame=fi,
                        world_xyz=tuple(float(x) for x in world),
                        state=state,
                        confidence=conf,
                        flight_segment_id=flight_membership.get(fi),
                    )
                )
            else:
                per_frame.append(
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
            frames=tuple(per_frame),
            flight_segments=tuple(flight_outs),
        )
        track.save(ball_out_path)
