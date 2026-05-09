"""Aggregate per-stage diagnostics into ``output/quality_report.json``.

Called at the end of :func:`src.pipeline.runner.run_pipeline` after all
stages have completed.  Each section (camera, hmr_world, ball) is
independent — missing inputs simply omit that section from the report.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.schemas.anchor import AnchorSet
from src.schemas.ball_track import BallTrack
from src.schemas.camera_track import CameraTrack


def write_quality_report(output_dir: Path) -> None:
    """Aggregate diagnostics from camera/, hmr_world/, ball/ into a single JSON."""
    report: dict = {}

    cam_path = output_dir / "camera" / "camera_track.json"
    anchors_path = output_dir / "camera" / "anchors.json"
    if cam_path.exists() and anchors_path.exists():
        cam = CameraTrack.load(cam_path)
        anchors = AnchorSet.load(anchors_path)
        confs = np.array([f.confidence for f in cam.frames])
        low_mask = confs < 0.6
        ranges: list[list[int]] = []
        i = 0
        while i < len(low_mask):
            if low_mask[i]:
                j = i
                while j < len(low_mask) and low_mask[j]:
                    j += 1
                ranges.append([cam.frames[i].frame, cam.frames[j - 1].frame])
                i = j
            else:
                i += 1

        # Reprojection residual per landmark for each anchor frame, averaged.
        # Uses per-frame t when available (post Phase 1) and applies the
        # clip-shared distortion (post Phase 2) so the metric reflects the
        # actual model the camera stage solved, not a linearised approximation.
        from src.utils.camera_projection import project_world_to_image

        anchor_residuals: list[float] = []
        camera_by_frame: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        t_world_fallback = np.array(cam.t_world)
        for f in cam.frames:
            t_f = np.array(f.t) if f.t is not None else t_world_fallback
            camera_by_frame[f.frame] = (np.array(f.K), np.array(f.R), t_f)
        distortion = cam.distortion
        for anchor in anchors.anchors:
            if anchor.frame not in camera_by_frame:
                continue
            K, R, t_f = camera_by_frame[anchor.frame]
            world = np.array(
                [lm.world_xyz for lm in anchor.landmarks], dtype=np.float64,
            )
            if len(world) == 0:
                continue
            obs = np.array(
                [lm.image_xy for lm in anchor.landmarks], dtype=np.float64,
            )
            try:
                proj = project_world_to_image(K, R, t_f, distortion, world)
            except Exception:
                continue
            for i in range(len(world)):
                anchor_residuals.append(
                    float(np.linalg.norm(obs[i] - proj[i]))
                )

        # Static-camera contract: with cam.camera_centre set, every frame's
        # -R^T @ t must equal that centre. Surface the worst drift so a
        # regression in the inter-anchor interpolation can't slip past us.
        if cam.camera_centre is not None:
            C = np.asarray(cam.camera_centre, dtype=float)
            drifts: list[float] = []
            for f in cam.frames:
                if f.t is None:
                    continue
                R = np.asarray(f.R, dtype=float)
                t = np.asarray(f.t, dtype=float)
                drifts.append(float(np.linalg.norm(-R.T @ t - C)))
            body_drift_max_m: float | None = max(drifts) if drifts else 0.0
        else:
            body_drift_max_m = None

        report["camera"] = {
            "anchor_count": len(anchors.anchors),
            "low_confidence_frame_count": int(low_mask.sum()),
            "low_confidence_frame_ranges": ranges,
            "mean_anchor_residual_px": (
                float(np.mean(anchor_residuals)) if anchor_residuals else 0.0
            ),
            "body_drift_max_m": body_drift_max_m,
            "distortion": list(cam.distortion),
        }

    hmr_dir = output_dir / "hmr_world"
    if hmr_dir.exists():
        npz_files = sorted(hmr_dir.glob("*_smpl_world.npz"))
        per_player_conf: list[float] = []
        low_players: list[str] = []
        for p in npz_files:
            z = np.load(p)
            mc = float(z["confidence"].mean()) if z["confidence"].size else 0.0
            per_player_conf.append(mc)
            if mc < 0.5:
                low_players.append(str(z["player_id"]))
        report["hmr_world"] = {
            "tracked_players": len(npz_files),
            "mean_per_player_confidence": (
                float(np.mean(per_player_conf)) if per_player_conf else 0.0
            ),
            "low_confidence_players": low_players,
        }

    ball_path = output_dir / "ball" / "ball_track.json"
    if ball_path.exists():
        ball = BallTrack.load(ball_path)
        states = [f.state for f in ball.frames]
        residuals = [s.fit_residual_px for s in ball.flight_segments]
        report["ball"] = {
            "grounded_frames": states.count("grounded"),
            "flight_segments": len(ball.flight_segments),
            "missing_frames": states.count("missing"),
            "mean_flight_fit_residual_px": (
                float(np.mean(residuals)) if residuals else 0.0
            ),
        }

    refined_summary_path = output_dir / "refined_poses" / "refined_poses_summary.json"
    if refined_summary_path.exists():
        report["refined_poses"] = json.loads(refined_summary_path.read_text())

    out = output_dir / "quality_report.json"
    out.write_text(json.dumps(report, indent=2))
