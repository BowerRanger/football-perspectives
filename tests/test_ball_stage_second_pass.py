"""Second-pass integration: re-smoothing and the BallStage end-to-end run."""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.schemas.ball_track import BallTrack
from src.schemas.camera_track import CameraFrame, CameraTrack
from src.stages.ball import _build_tracker, _resmooth_observations
from src.utils.ball_detector import FakeBallDetector


# ---------------------------------------------------------------------------
# Module-private helpers (copied verbatim from tests/test_ball_stage.py;
# do NOT import across test modules to keep the two suites independent).
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# ScriptedDetector: drives second-pass candidate lists via a FIFO queue.
# ---------------------------------------------------------------------------

class ScriptedDetector(FakeBallDetector):
    """Pass-1 detections by call order; second-pass candidate lists are
    served FIFO across detect_candidates calls (the second pass visits
    frames in a deterministic order: prime frames first, then the gap)."""

    def __init__(self, detections, second_pass_cands):
        super().__init__(detections)
        self._sp = deque(second_pass_cands)

    def detect_candidates(self, frame, min_score, top_k=5):
        if not self._sp:
            return []
        cands = self._sp.popleft()
        kept = [c for c in cands if c[2] >= min_score]
        kept.sort(key=lambda c: -c[2])
        return kept[:top_k]


# ---------------------------------------------------------------------------
# Existing unit tests (unchanged).
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_resmooth_keeps_raw_uv_and_fills_gaps():
    n = 30
    uv = {f: (100.0 + 5.0 * f, 400.0) for f in range(n)}
    uv[10] = None
    uv[11] = None
    steps = _resmooth_observations(uv, n, cfg={})
    assert len(steps) == n
    # Raw observations pass through exactly (raw-uv override rule).
    assert steps[5].uv == (125.0, 400.0)
    # Short gap is IMM-filled near the constant-velocity line.
    assert steps[10].uv is not None
    assert abs(steps[10].uv[0] - 150.0) < 5.0
    assert steps[10].is_gap_fill


@pytest.mark.unit
def test_build_tracker_honours_max_gap_override():
    tracker = _build_tracker({}, max_gap_frames=10 ** 6)
    for i in range(5):
        tracker.update(i, (100.0 + i, 400.0))
    last = None
    for i in range(5, 105):
        last = tracker.update(i, None)
    assert last.uv is not None  # would be None with the default max_gap


# ---------------------------------------------------------------------------
# Carried-over unit test (Task 6 review): outlier frames must not pass raw.
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_resmooth_does_not_override_outlier_frames():
    """A wild measurement the IMM gates as an outlier must NOT pass through
    raw — the blended prediction wins."""
    n = 20
    uv = {f: (100.0 + 5.0 * f, 400.0) for f in range(n)}
    uv[15] = (5000.0, 4000.0)  # far outside any gate
    steps = _resmooth_observations(uv, n, cfg={})
    assert steps[15].is_outlier
    assert steps[15].uv != (5000.0, 4000.0)


# ---------------------------------------------------------------------------
# Integration tests for the second-pass loop in BallStage._run_shot.
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_second_pass_fills_gap_and_never_anchors(tmp_path: Path):
    from src.stages.ball import BallStage

    n = 60
    fps = 30.0
    K, R, t = _camera_pose()
    _save_camera_track(tmp_path / "camera" / "camera_track.json", K, R, t, n, fps=fps)
    _write_blank_clip(tmp_path / "shots" / "play.mp4", n, fps=fps)

    truth = {i: np.array([30.0 + 0.2 * i, 34.0, 0.11]) for i in range(n)}
    uv_truth = {i: _project(truth[i], K, R, t) for i in range(n)}

    # Pass 1: detections everywhere except frames 20-24.
    detections = []
    for i in range(n):
        if 20 <= i <= 24:
            detections.append(None)
        else:
            detections.append((uv_truth[i][0], uv_truth[i][1], 0.9))

    # Second pass visits frames 18..24 (prime offset 2, then the gap).
    # Prime frames return nothing; gap frames offer the true ball at a
    # weak score plus a strong decoy far outside the corridor.
    sp_cands = [[], []]
    for i in range(20, 25):
        sp_cands.append([
            (uv_truth[i][0] + 1.0, uv_truth[i][1] - 1.0, 0.55),
            (uv_truth[i][0] + 400.0, uv_truth[i][1] + 200.0, 0.95),  # decoy
        ])

    stage = BallStage(
        config={"ball": {
            "detector": "fake",
            # appearance bridge would gap-fill 20-24 itself; isolate the
            # second pass by disabling it.
            "appearance_bridge": {"enabled": False},
            "second_pass": {"enabled": True, "zoom_min_ball_px": 0.0},
            "auto_anchors": {"enabled": True, "grounded_interval": 8},
        }},
        output_dir=tmp_path,
        ball_detector=ScriptedDetector(detections, sp_cands),
    )
    stage.run()

    obs = json.loads((tmp_path / "ball" / "ball_observations.json").read_text())
    by_frame = {f["frame"]: f for f in obs["frames"]}
    for i in range(20, 25):
        assert by_frame[i]["source"] == "second_pass"
        assert abs(by_frame[i]["uv"][0] - uv_truth[i][0]) < 3.0  # decoy rejected
        assert by_frame[i]["confidence"] > 0.0

    diag = json.loads((tmp_path / "ball" / "ball_diag.json").read_text())
    cov = diag["detection_coverage"]
    assert cov["second_pass"] > 0.0
    assert cov["total"] == pytest.approx(cov["pass1"] + cov["second_pass"])
    assert cov["total"] > cov["pass1"]

    # Second-pass frames never become anchors.
    anchors_path = tmp_path / "ball" / "ball_anchors_auto.json"
    if anchors_path.exists():
        anchors = json.loads(anchors_path.read_text())
        frames = [a["frame"] for a in anchors.get("anchors", [])]
        assert not any(20 <= f <= 24 for f in frames)

    # Track is continuous through the gap (no missing state inside it).
    track = BallTrack.load(tmp_path / "ball" / "ball_track.json")
    states = {f.frame: f.state for f in track.frames}
    assert all(states[i] != "missing" for i in range(20, 25))


@pytest.mark.integration
def test_second_pass_disabled_is_noop(tmp_path: Path):
    from src.stages.ball import BallStage

    n = 30
    fps = 30.0
    K, R, t = _camera_pose()
    _save_camera_track(tmp_path / "camera" / "camera_track.json", K, R, t, n, fps=fps)
    _write_blank_clip(tmp_path / "shots" / "play.mp4", n, fps=fps)
    detections = []
    for i in range(n):
        p = np.array([30.0 + 0.2 * i, 34.0, 0.11])
        u, v = _project(p, K, R, t)
        detections.append((u, v, 0.9))

    stage = BallStage(
        config={"ball": {"detector": "fake", "second_pass": {"enabled": False}}},
        output_dir=tmp_path,
        ball_detector=FakeBallDetector(detections),
    )
    stage.run()
    diag = json.loads((tmp_path / "ball" / "ball_diag.json").read_text())
    assert diag["detection_coverage"]["second_pass"] == 0.0
