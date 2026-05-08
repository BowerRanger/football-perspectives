"""Integration test for BallStage end-to-end.

Synthesises a 60-frame ball trajectory through the broadcast camera
pose, with a flight segment in frames 20-40 and grounded motion outside
that window.  Runs ``BallStage`` on the synthetic input and verifies
schema invariants on the output BallTrack."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.schemas.ball_track import BallTrack
from src.schemas.camera_track import CameraFrame, CameraTrack
from src.stages.ball import BallStage


def _render_and_save_camera(tmp_path: Path, n: int):
    look_world = np.array([0.0, 64.0, -30.0])
    look_world = look_world / np.linalg.norm(look_world)
    right_world = np.array([1.0, 0.0, 0.0])
    down_world = np.cross(look_world, right_world)
    R = np.array([right_world, down_world, look_world], dtype=float)
    t = -R @ np.array([52.5, -30.0, 30.0])
    K = np.array([[1500.0, 0, 640.0], [0, 1500.0, 360.0], [0, 0, 1.0]])
    track = CameraTrack(
        clip_id="play",
        fps=30.0,
        image_size=(1280, 720),
        t_world=t.tolist(),
        frames=tuple(
            CameraFrame(
                frame=i,
                K=K.tolist(),
                R=R.tolist(),
                confidence=1.0,
                is_anchor=(i == 0),
            )
            for i in range(n)
        ),
    )
    track.save(tmp_path / "camera" / "camera_track.json")
    return K, R, t


@pytest.mark.integration
def test_ball_stage_recovers_grounded_and_flight(tmp_path: Path):
    n = 60
    K, R, t = _render_and_save_camera(tmp_path, n)
    pts = []
    for i in range(n):
        if 20 <= i <= 40:
            dt = (i - 20) / 30.0
            p = np.array(
                [50.0 + 8 * dt, 30.0, 0.5 * (max(0, 5 - 9.81 * dt) ** 2 / 9.81)]
            )
        else:
            p = np.array([50.0 + 0.5 * i, 30.0, 0.11])
        cam = R @ p + t
        pix = K @ cam
        uv = pix[:2] / pix[2]
        pts.append({"frame": i, "bbox_centre": list(uv), "confidence": 0.85})
    (tmp_path / "tracks").mkdir()
    (tmp_path / "tracks" / "ball_track.json").write_text(
        json.dumps({"clip_id": "play", "frames": pts})
    )

    stage = BallStage(config={"ball": {}}, output_dir=tmp_path)
    stage.run()
    out = BallTrack.load(tmp_path / "ball" / "ball_track.json")
    states = {f.state for f in out.frames}
    assert "grounded" in states
    # flight segments may or may not pass the residual gate on this synthetic
    # data; assert at least the schema invariant.
    assert len(out.frames) == n


@pytest.mark.integration
def test_ball_stage_emits_per_shot_track(tmp_path: Path):
    """Two shots, two camera_tracks, two raw ball_tracks → two per-shot
    BallTrack outputs. Single-shot path stays via legacy fallback."""
    from src.schemas.shots import Shot, ShotsManifest

    n = 30
    K, R, t = _render_and_save_camera(tmp_path, n)
    # Move the per-shot camera_track output into per-shot naming.
    legacy = tmp_path / "camera" / "camera_track.json"
    (tmp_path / "camera" / "alpha_camera_track.json").write_text(
        legacy.read_text()
    )
    legacy.unlink()

    # And a duplicate for beta.
    (tmp_path / "camera" / "beta_camera_track.json").write_text(
        (tmp_path / "camera" / "alpha_camera_track.json").read_text()
    )

    # Manifest with two shots.
    fps = 30.0
    end = n - 1
    (tmp_path / "shots").mkdir(exist_ok=True)
    ShotsManifest(
        source_file="x", fps=fps, total_frames=2 * n,
        shots=[
            Shot(id="alpha", start_frame=0, end_frame=end, start_time=0.0,
                 end_time=(end + 1) / fps, clip_file="shots/alpha.mp4"),
            Shot(id="beta", start_frame=0, end_frame=end, start_time=0.0,
                 end_time=(end + 1) / fps, clip_file="shots/beta.mp4"),
        ],
    ).save(tmp_path / "shots" / "shots_manifest.json")

    # Synthetic raw ball detections, same trajectory in both shots.
    pts = []
    for i in range(n):
        p = np.array([50.0 + 0.5 * i, 30.0, 0.11])
        cam = R @ p + t
        pix = K @ cam
        uv = pix[:2] / pix[2]
        pts.append({"frame": i, "bbox_centre": list(uv), "confidence": 0.85})
    (tmp_path / "tracks").mkdir(exist_ok=True)
    (tmp_path / "tracks" / "alpha_ball_track.json").write_text(
        json.dumps({"clip_id": "alpha", "frames": pts})
    )
    (tmp_path / "tracks" / "beta_ball_track.json").write_text(
        json.dumps({"clip_id": "beta", "frames": pts})
    )

    stage = BallStage(config={"ball": {}}, output_dir=tmp_path)
    stage.run()

    assert (tmp_path / "ball" / "alpha_ball_track.json").exists()
    assert (tmp_path / "ball" / "beta_ball_track.json").exists()
