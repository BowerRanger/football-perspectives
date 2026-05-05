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
        anchor_residuals: list[float] = []
        camera_by_frame = {
            f.frame: (np.array(f.K), np.array(f.R)) for f in cam.frames
        }
        t_w = np.array(cam.t_world)
        for anchor in anchors.anchors:
            if anchor.frame not in camera_by_frame:
                continue
            K, R = camera_by_frame[anchor.frame]
            for lm in anchor.landmarks:
                cam_pt = R @ np.array(lm.world_xyz) + t_w
                if cam_pt[2] <= 0:
                    continue
                proj = (K @ cam_pt)[:2] / cam_pt[2]
                anchor_residuals.append(
                    float(np.linalg.norm(np.array(lm.image_xy) - proj))
                )

        report["camera"] = {
            "anchor_count": len(anchors.anchors),
            "low_confidence_frame_count": int(low_mask.sum()),
            "low_confidence_frame_ranges": ranges,
            "mean_anchor_residual_px": (
                float(np.mean(anchor_residuals)) if anchor_residuals else 0.0
            ),
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

    out = output_dir / "quality_report.json"
    out.write_text(json.dumps(report, indent=2))
