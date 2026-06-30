"""Stage-level test: BallStage attaches per-frame orientation quats.

After the dense ``BallFrame`` list is built, ``_solve_shot`` integrates a
per-frame unit quaternion (T2's ``integrate_orientation``) and saves it on
each frame.  Flight/grounded frames must carry a unit-norm ``quat_wxyz``;
missing frames carry ``None`` (mirroring ``world_xyz``).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from src.schemas.ball_track import BallTrack
from src.schemas.camera_track import CameraFrame, CameraTrack
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
    path: Path, K: np.ndarray, R: np.ndarray, t: np.ndarray, n: int,
    fps: float = 30.0,
) -> None:
    CameraTrack(
        clip_id="play",
        fps=fps,
        image_size=(1280, 720),
        t_world=t.tolist(),
        frames=tuple(
            CameraFrame(frame=i, K=K.tolist(), R=R.tolist(),
                        confidence=1.0, is_anchor=(i == 0))
            for i in range(n)
        ),
    ).save(path)


def _write_blank_clip(path: Path, n: int, fps: float = 30.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (1280, 720)
    )
    for _ in range(n):
        writer.write(np.full((720, 1280, 3), [50, 200, 50], dtype=np.uint8))
    writer.release()


def _project(p: np.ndarray, K: np.ndarray, R: np.ndarray, t: np.ndarray):
    cam = R @ p + t
    pix = K @ cam
    return float(pix[0] / pix[2]), float(pix[1] / pix[2])


@pytest.mark.integration
def test_ball_stage_attaches_unit_quaternions(tmp_path: Path):
    n = 60
    fps = 30.0
    K, R, t = _camera_pose()
    _save_camera_track(tmp_path / "camera" / "camera_track.json", K, R, t, n, fps=fps)
    _write_blank_clip(tmp_path / "shots" / "play.mp4", n, fps=fps)

    # Rolling ball across the centre of the pitch.
    detections: list[tuple[float, float, float] | None] = []
    for i in range(n):
        p = np.array([30.0 + 0.2 * i, 34.0, 0.11])
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

    quat_frames = [f for f in out.frames if f.world_xyz is not None]
    assert quat_frames, "expected world-bearing frames to carry quaternions"
    for f in quat_frames:
        assert f.quat_wxyz is not None, f"frame {f.frame} missing quat"
        norm = float(np.linalg.norm(np.asarray(f.quat_wxyz, dtype=float)))
        assert abs(norm - 1.0) < 1e-5, f"frame {f.frame} quat not unit: {norm}"

    # Missing frames mirror world_xyz: no quaternion.
    for f in out.frames:
        if f.world_xyz is None:
            assert f.quat_wxyz is None

    # A rolling ball must actually rotate: the first and last quats differ.
    first = np.asarray(quat_frames[0].quat_wxyz, dtype=float)
    last = np.asarray(quat_frames[-1].quat_wxyz, dtype=float)
    assert float(np.linalg.norm(first - last)) > 1e-3
