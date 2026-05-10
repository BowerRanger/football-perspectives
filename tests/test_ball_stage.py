"""Integration tests for BallStage end-to-end.

Synthesises camera tracks + clip files + FakeBallDetector output;
verifies the four-step pipeline (detect → IMM smooth → ground project →
flight fit) produces the expected BallTrack."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from src.schemas.ball_track import BallTrack
from src.schemas.camera_track import CameraFrame, CameraTrack
from src.schemas.shots import Shot, ShotsManifest
from src.stages.ball import BallStage
from src.utils.ball_detector import FakeBallDetector


def _camera_pose() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    look = np.array([0.0, 64.0, -30.0])
    look /= np.linalg.norm(look)
    right = np.array([1.0, 0.0, 0.0])
    down = np.cross(look, right)
    R = np.array([right, down, look], dtype=float)
    t = -R @ np.array([52.5, -30.0, 30.0])
    K = np.array([[1500.0, 0, 640.0], [0, 1500.0, 360.0], [0, 0, 1.0]])
    return K, R, t


def _save_camera_track(
    path: Path,
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    n: int,
    clip_id: str = "play",
    fps: float = 30.0,
) -> None:
    track = CameraTrack(
        clip_id=clip_id,
        fps=fps,
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
    track.save(path)


def _write_blank_clip(path: Path, n: int, fps: float = 30.0) -> None:
    """The BallDetector is faked in tests, so the frame contents don't matter —
    we just need the VideoCapture to return ``n`` frames."""
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (320, 240)
    )
    for _ in range(n):
        writer.write(np.full((240, 320, 3), [50, 200, 50], dtype=np.uint8))
    writer.release()


def _project(p: np.ndarray, K: np.ndarray, R: np.ndarray, t: np.ndarray) -> tuple[float, float]:
    cam = R @ p + t
    pix = K @ cam
    return float(pix[0] / pix[2]), float(pix[1] / pix[2])


@pytest.mark.integration
def test_ball_stage_recovers_grounded_and_flight(tmp_path: Path):
    n = 60
    fps = 30.0
    K, R, t = _camera_pose()
    _save_camera_track(tmp_path / "camera" / "camera_track.json", K, R, t, n, fps=fps)
    _write_blank_clip(tmp_path / "shots" / "play.mp4", n, fps=fps)

    detections: list[tuple[float, float, float] | None] = []
    for i in range(n):
        if 20 <= i <= 40:
            dt = (i - 20) / fps
            p = np.array([50.0 + 8 * dt, 30.0, 0.5 * (max(0, 5 - 9.81 * dt) ** 2 / 9.81)])
        else:
            p = np.array([50.0 + 0.5 * i, 30.0, 0.11])
        u, v = _project(p, K, R, t)
        detections.append((u, v, 0.9))

    stage = BallStage(
        config={"ball": {"detector": "fake"}},
        output_dir=tmp_path,
        ball_detector=FakeBallDetector(detections),
    )
    stage.run()

    out = BallTrack.load(tmp_path / "ball" / "ball_track.json")
    assert len(out.frames) == n
    states = {f.state for f in out.frames}
    assert "grounded" in states
    # FlightSegment.parabola must always include the spin keys (None when
    # the Magnus refinement did not engage).
    for seg in out.flight_segments:
        for key in ("spin_axis_world", "spin_omega_rad_s", "spin_confidence"):
            assert key in seg.parabola


@pytest.mark.integration
def test_ball_stage_marks_missing_after_long_gap(tmp_path: Path):
    n = 40
    fps = 30.0
    K, R, t = _camera_pose()
    _save_camera_track(tmp_path / "camera" / "camera_track.json", K, R, t, n, fps=fps)
    _write_blank_clip(tmp_path / "shots" / "play.mp4", n, fps=fps)

    detections: list[tuple[float, float, float] | None] = []
    for i in range(n):
        if 10 <= i <= 25:
            # Long missed-detection gap — far longer than max_gap_frames.
            detections.append(None)
        else:
            p = np.array([50.0 + 0.5 * i, 30.0, 0.11])
            u, v = _project(p, K, R, t)
            detections.append((u, v, 0.9))

    stage = BallStage(
        config={"ball": {"detector": "fake", "max_gap_frames": 3}},
        output_dir=tmp_path,
        ball_detector=FakeBallDetector(detections),
    )
    stage.run()

    out = BallTrack.load(tmp_path / "ball" / "ball_track.json")
    # First few frames inside the gap are gap-filled; later frames in the
    # gap should be state="missing" after max_gap_frames expires.
    assert out.frames[20].state == "missing"
    assert out.frames[20].world_xyz is None


@pytest.mark.integration
def test_ball_stage_emits_per_shot_track(tmp_path: Path):
    n = 30
    fps = 30.0
    K, R, t = _camera_pose()
    _save_camera_track(
        tmp_path / "camera" / "alpha_camera_track.json", K, R, t, n,
        clip_id="alpha", fps=fps,
    )
    _save_camera_track(
        tmp_path / "camera" / "beta_camera_track.json", K, R, t, n,
        clip_id="beta", fps=fps,
    )
    _write_blank_clip(tmp_path / "shots" / "alpha.mp4", n, fps=fps)
    _write_blank_clip(tmp_path / "shots" / "beta.mp4", n, fps=fps)

    end = n - 1
    ShotsManifest(
        source_file="x", fps=fps, total_frames=2 * n,
        shots=[
            Shot(id="alpha", start_frame=0, end_frame=end, start_time=0.0,
                 end_time=(end + 1) / fps, clip_file="shots/alpha.mp4"),
            Shot(id="beta", start_frame=0, end_frame=end, start_time=0.0,
                 end_time=(end + 1) / fps, clip_file="shots/beta.mp4"),
        ],
    ).save(tmp_path / "shots" / "shots_manifest.json")

    detections: list[tuple[float, float, float] | None] = []
    for i in range(n):
        p = np.array([50.0 + 0.5 * i, 30.0, 0.11])
        u, v = _project(p, K, R, t)
        detections.append((u, v, 0.9))
    # FakeBallDetector cycles, so the same list serves both shots.

    stage = BallStage(
        config={"ball": {"detector": "fake"}},
        output_dir=tmp_path,
        ball_detector=FakeBallDetector(detections),
    )
    stage.run()

    assert (tmp_path / "ball" / "alpha_ball_track.json").exists()
    assert (tmp_path / "ball" / "beta_ball_track.json").exists()
