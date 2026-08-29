"""Integration tests: ball.detection_cache wired into BallStage.run().

Uses a clip with DISTINCT pixel content per frame (unlike test_ball_stage's
shared solid-color fixture) — the cache keys on frame-content hash, so a
fixture where every frame is byte-identical would collide across frame
indices and defeat the point of these tests.
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


def _save_camera_track(path: Path, K, R, t, n: int, fps: float = 30.0) -> None:
    track = CameraTrack(
        clip_id="play", fps=fps, image_size=(1280, 720),
        t_world=t.tolist(),
        frames=tuple(
            CameraFrame(frame=i, K=K.tolist(), R=R.tolist(),
                       confidence=1.0, is_anchor=(i == 0))
            for i in range(n)
        ),
    )
    track.save(path)


def _write_unique_frames_clip(path: Path, n: int, fps: float = 30.0) -> None:
    """Every frame gets distinct pixel content so the cache's
    frame-content hash cannot collide across frame indices within a run
    (the shared _write_blank_clip fixture used elsewhere fills every
    frame identically, which is fine when the detector is faked by call
    order but would defeat a content-hash cache)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (1280, 720)
    )
    for i in range(n):
        color = [(i * 3) % 256, (i * 7 + 40) % 256, (i * 13 + 80) % 256]
        writer.write(np.full((720, 1280, 3), color, dtype=np.uint8))
    writer.release()


def _project(p: np.ndarray, K, R, t) -> tuple[float, float]:
    cam = R @ p + t
    pix = K @ cam
    return float(pix[0] / pix[2]), float(pix[1] / pix[2])


_SIDECAR_NAMES = (
    "ball_track.json",
    "ball_observations.json",
    "ball_anchors_auto.json",
    "ball_keyframes.json",
)


@pytest.mark.integration
def test_cache_replays_stale_detections_bit_identical_across_runs(tmp_path: Path):
    """The strong version of the bit-identical requirement: run 2 uses a
    detector that would produce a COMPLETELY DIFFERENT track (all misses)
    if actually invoked. If the four sidecars still match run 1 exactly,
    that proves the cache — not coincidence — produced the replay."""
    n = 40
    fps = 30.0
    K, R, t = _camera_pose()
    out = tmp_path
    _save_camera_track(out / "camera" / "camera_track.json", K, R, t, n, fps=fps)
    _write_unique_frames_clip(out / "shots" / "play.mp4", n, fps=fps)

    rolling: list[tuple[float, float, float] | None] = []
    for i in range(n):
        p = np.array([30.0 + 0.2 * i, 34.0, 0.11])
        u, v = _project(p, K, R, t)
        rolling.append((u, v, 0.9))
    all_misses: list[tuple[float, float, float] | None] = [None] * n

    cfg = {"ball": {"detector": "fake",
                    "detection_cache": {"enabled": True}}}

    stage1 = BallStage(config=cfg, output_dir=out,
                       ball_detector=FakeBallDetector(list(rolling)))
    stage1.run()

    ball_dir = out / "ball"
    cache_file = ball_dir / "detection_cache.json"
    assert cache_file.exists(), "cache file should be written by run 1"

    snapshot = {name: (ball_dir / name).read_bytes() for name in _SIDECAR_NAMES}
    out1 = BallTrack.load(ball_dir / "ball_track.json")
    assert "grounded" in {f.state for f in out1.frames}

    # Run 2: a detector scripted to see NOTHING. If the cache weren't
    # intercepting detect() calls, this would produce a track with no
    # grounded detections at all.
    stage2 = BallStage(config=cfg, output_dir=out,
                       ball_detector=FakeBallDetector(list(all_misses)))
    stage2.run()

    for name in _SIDECAR_NAMES:
        after = (ball_dir / name).read_bytes()
        assert after == snapshot[name], (
            f"{name} differs between cache-hit reruns — cache did not "
            "fully intercept the second run's detector calls"
        )


@pytest.mark.integration
def test_cache_disabled_leaves_no_cache_file_and_detector_runs_fresh(tmp_path: Path):
    """Default (enabled: false, or key absent entirely) must be a no-op:
    no cache file appears, and a differently-scripted second detector
    genuinely changes the output — first-run behaviour is unaffected by
    this feature existing."""
    n = 40
    fps = 30.0
    K, R, t = _camera_pose()
    out = tmp_path
    _save_camera_track(out / "camera" / "camera_track.json", K, R, t, n, fps=fps)
    _write_unique_frames_clip(out / "shots" / "play.mp4", n, fps=fps)

    rolling: list[tuple[float, float, float] | None] = []
    for i in range(n):
        p = np.array([30.0 + 0.2 * i, 34.0, 0.11])
        u, v = _project(p, K, R, t)
        rolling.append((u, v, 0.9))

    cfg = {"ball": {"detector": "fake"}}  # no detection_cache key at all

    stage1 = BallStage(config=cfg, output_dir=out,
                       ball_detector=FakeBallDetector(list(rolling)))
    stage1.run()

    ball_dir = out / "ball"
    assert not (ball_dir / "detection_cache.json").exists()

    out1 = BallTrack.load(ball_dir / "ball_track.json")
    assert "grounded" in {f.state for f in out1.frames}

    all_misses: list[tuple[float, float, float] | None] = [None] * n
    stage2 = BallStage(config=cfg, output_dir=out,
                       ball_detector=FakeBallDetector(list(all_misses)))
    stage2.run()
    assert not (ball_dir / "detection_cache.json").exists()

    out2 = BallTrack.load(ball_dir / "ball_track.json")
    # With no cache, the all-misses detector genuinely changes the
    # output: nothing is ever detected, so nothing anchors a grounded
    # run (unlike run 1, which sees the same rolling detections live).
    assert "grounded" not in {f.state for f in out2.frames}
