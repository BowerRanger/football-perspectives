"""Strike-window integration: BallStage._detect_shot end-to-end with a
redetect-capable scripted fake detector.

Module-private helpers are copied (not imported) from
tests/test_ball_stage_second_pass.py per that file's own convention — the
two suites stay independent.

Test strategy: the scripted detector always answers with a candidate
placed EXACTLY at the geometric centre of whatever crop it was given
(upscaled-local coordinate ``(crop_px//2*upscale, crop_px//2*upscale)``).
Since the crop is built by centring ``crop_px`` on the corridor
prediction, this candidate maps back through the upscale+translate
round-trip to EXACTLY that corridor centre, whatever it is — so the test
verifies the mechanical wiring (config parsing, cv2 crop/upscale, the
coordinate round-trip, gating, source tagging, budget/coverage counters,
the no-anchor invariant, SUPPORTS_REDETECT gating) without needing to
hand-predict the tracker/IMM's exact corridor math. Real detection
QUALITY (does upscaling actually recover a genuinely blurred ball) is a
separate question, answered by the gberch f343 smoke test, not this file.
"""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.schemas.camera_track import CameraFrame, CameraTrack
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
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (1280, 720)
    )
    for _ in range(n):
        writer.write(np.full((720, 1280, 3), [50, 200, 50], dtype=np.uint8))
    writer.release()


def _project(p: np.ndarray, K: np.ndarray, R: np.ndarray, t: np.ndarray) -> tuple[float, float]:
    cam = R @ p + t
    pix = K @ cam
    return float(pix[0] / pix[2]), float(pix[1] / pix[2])


class RedetectScriptedDetector(FakeBallDetector):
    """SUPPORTS_REDETECT=True (WASB/YOLO-like): pass-1 detections cycle by
    call order (via FakeBallDetector.detect); detect_candidates() serves
    a fixed candidate FIFO — by default the same "crop centre" candidate
    forever, so it never runs dry regardless of window sizing."""

    SUPPORTS_REDETECT = True
    _frames_in = 3

    def __init__(self, detections, crop_center_candidate=None):
        super().__init__(detections)
        self._crop_center_candidate = crop_center_candidate
        self.candidate_calls = 0

    def detect_candidates(self, frame, min_score, top_k=5):
        self.candidate_calls += 1
        if self._crop_center_candidate is None:
            return []
        u, v, s = self._crop_center_candidate
        if s < min_score:
            return []
        return [(u, v, s)]


@pytest.mark.integration
def test_strike_window_recovers_evidence_gap_at_a_fast_break(tmp_path: Path):
    """A genuine detection gap straddles a hard velocity break (slow roll
    -> a materially faster launch, e.g. Δv well above min_dspeed_px). With
    second_pass and foot_guided both disabled, any recovered evidence in
    the gap must come from the strike-window pass.

    ``auto_anchors`` is disabled here so the no-anchor-minting check is
    unambiguous: a genuine break is exactly the scenario where strike-
    window evidence is INTENDED to (legitimately) support downstream
    event/anchor minting once auto_anchors runs — that is the whole point
    of registering "strike_window" in event_evidence_sources — so with
    auto_anchors on, a real anchor CAN legitimately land inside the
    recovered span. Disabling it isolates the narrower invariant this test
    actually guards: the strike-window PASS ITSELF never constructs an
    anchor record (same as second_pass/foot_guided)."""
    from src.stages.ball import BallStage

    n = 60
    fps = 30.0
    K, R, t = _camera_pose()
    _save_camera_track(tmp_path / "camera" / "camera_track.json", K, R, t, n, fps=fps)
    _write_blank_clip(tmp_path / "shots" / "play.mp4", n, fps=fps)

    # World x(f): slow roll (0.05 m/frame) to f=20, then a break to a
    # faster launch (0.3 m/frame) — ~6x the pre-break pixel speed.
    def world_x(f: int) -> float:
        if f <= 20:
            return 30.0 + 0.05 * f
        return world_x(20) + 0.3 * (f - 20)

    truth = {f: np.array([world_x(f), 34.0, 0.11]) for f in range(n)}
    uv_truth = {f: _project(truth[f], K, R, t) for f in range(n)}

    # Gap straddling the break: pass-1 sees nothing on [21, 28].
    gap = range(21, 29)
    detections = [
        None if i in gap else (uv_truth[i][0], uv_truth[i][1], 0.9)
        for i in range(n)
    ]

    crop_px, upscale = 160, 2.0  # StrikeWindowCfg defaults
    center_local = (crop_px // 2) * upscale  # maps back to the corridor centre exactly
    detector = RedetectScriptedDetector(
        detections, crop_center_candidate=(center_local, center_local, 0.8),
    )

    stage = BallStage(
        config={"ball": {
            "detector": "fake",
            "appearance_bridge": {"enabled": False},
            "second_pass": {
                "enabled": False,
                "strike_window": {
                    "enabled": True,
                    "min_dspeed_px": 1.0,  # generous: any real break trips it
                    "window_radius_frames": 8,
                    "max_windows_per_shot": 3,
                    "accept_min": 0.1,
                },
            },
            "foot_guided": {"enabled": False},
            "auto_anchors": {"enabled": False},
        }},
        output_dir=tmp_path,
        ball_detector=detector,
    )
    stage.run()

    obs = json.loads((tmp_path / "ball" / "ball_observations.json").read_text())
    by_frame = {f["frame"]: f for f in obs["frames"]}
    sw_frames = [f for f, r in by_frame.items() if r["source"] == "strike_window"]

    assert sw_frames, "strike-window pass never fired"
    # It must have found SOME evidence inside (or immediately around) the
    # gap it was meant to address.
    assert any(21 <= f <= 28 for f in sw_frames)
    for f in sw_frames:
        assert by_frame[f]["confidence"] > 0.0
        assert by_frame[f]["uv"] is not None

    diag = json.loads((tmp_path / "ball" / "ball_diag.json").read_text())
    assert diag["detection_coverage"]["strike_window"] == len(sw_frames)

    # Never mints auto-anchors directly (same invariant as second_pass):
    # with auto_anchors disabled, the strike-window pass itself must not
    # have produced an anchors sidecar.
    anchors_path = tmp_path / "ball" / "ball_anchors_auto.json"
    if anchors_path.exists():
        anchors = json.loads(anchors_path.read_text())
        assert not anchors.get("anchors")


@pytest.mark.integration
def test_strike_window_disabled_is_noop(tmp_path: Path):
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
        config={"ball": {
            "detector": "fake",
            "second_pass": {"enabled": False, "strike_window": {"enabled": False}},
            "foot_guided": {"enabled": False},
        }},
        output_dir=tmp_path,
        ball_detector=RedetectScriptedDetector(detections),
    )
    stage.run()
    diag = json.loads((tmp_path / "ball" / "ball_diag.json").read_text())
    assert diag["detection_coverage"]["strike_window"] == 0


@pytest.mark.integration
def test_strike_window_skipped_for_non_redetect_detector(tmp_path: Path):
    """FakeBallDetector (SUPPORTS_REDETECT=False, the scripted-cycle
    default used everywhere else in the ball test suite) must never be
    re-queried by the strike-window pass — desyncing its cycle would
    silently corrupt every other test relying on it."""
    from src.stages.ball import BallStage

    n = 40
    fps = 30.0
    K, R, t = _camera_pose()
    _save_camera_track(tmp_path / "camera" / "camera_track.json", K, R, t, n, fps=fps)
    _write_blank_clip(tmp_path / "shots" / "play.mp4", n, fps=fps)
    detections = []
    for i in range(n):
        speed = 0.2 if i < 20 else 3.0
        p = np.array([30.0 + speed * i, 34.0, 0.11])
        u, v = _project(p, K, R, t)
        detections.append((u, v, 0.9))

    stage = BallStage(
        config={"ball": {
            "detector": "fake",
            "second_pass": {"enabled": False, "strike_window": {"enabled": True}},
            "foot_guided": {"enabled": False},
        }},
        output_dir=tmp_path,
        ball_detector=FakeBallDetector(detections),
    )
    stage.run()  # must not raise / desync
    diag = json.loads((tmp_path / "ball" / "ball_diag.json").read_text())
    assert diag["detection_coverage"]["strike_window"] == 0


@pytest.mark.integration
def test_strike_window_respects_max_windows_per_shot_budget(tmp_path: Path):
    """Four independent hard breaks, budget capped to 2 windows: the
    number of detect_candidates() calls (and thus real cost) must stay
    bounded, not scale with the number of breaks."""
    from src.stages.ball import BallStage

    n = 220
    fps = 30.0
    K, R, t = _camera_pose()
    _save_camera_track(tmp_path / "camera" / "camera_track.json", K, R, t, n, fps=fps)
    _write_blank_clip(tmp_path / "shots" / "play.mp4", n, fps=fps)

    def _piecewise_x(n: int, segments: list[tuple[int, float]]) -> dict[int, float]:
        """``segments``: (start_frame, m/frame) pairs; the slope in force
        from each start_frame (inclusive) until the next one."""
        xs: dict[int, float] = {}
        x = 30.0
        slope = 0.0
        seg_idx = 0
        for f in range(n):
            if seg_idx < len(segments) and f == segments[seg_idx][0]:
                slope = segments[seg_idx][1]
                seg_idx += 1
            xs[f] = x
            x += slope
        return xs

    world_xs = _piecewise_x(n, [
        (0, 0.05), (30, 0.3), (40, 0.05),
        (90, 0.3), (100, 0.05),
        (150, 0.3), (160, 0.05),
        (200, 0.3),
    ])
    uv_truth = {
        f: _project(np.array([world_xs[f], 34.0, 0.11]), K, R, t) for f in range(n)
    }
    detections = [(uv_truth[i][0], uv_truth[i][1], 0.9) for i in range(n)]

    detector = RedetectScriptedDetector(detections, crop_center_candidate=(160.0, 160.0, 0.8))

    stage = BallStage(
        config={"ball": {
            "detector": "fake",
            "appearance_bridge": {"enabled": False},
            "second_pass": {
                "enabled": False,
                "strike_window": {
                    "enabled": True,
                    "min_dspeed_px": 1.0,
                    "window_radius_frames": 6,
                    "max_windows_per_shot": 2,
                    "max_crops_per_window": 13,
                    "accept_min": 0.1,
                },
            },
            "foot_guided": {"enabled": False},
        }},
        output_dir=tmp_path,
        ball_detector=detector,
    )
    stage.run()

    # <= 2 windows * (13 crops + 2 priming frames) each.
    assert detector.candidate_calls <= 2 * (13 + 2)
