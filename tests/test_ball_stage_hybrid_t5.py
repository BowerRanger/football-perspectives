"""Unit tests for T5's ball.py hybrid-trajectory wiring helpers:
``_hybrid_fix_knots`` (cross-replay fixes -> gated Knots,
ball.hybrid.fixes.*), ``_hybrid_cue_corroboration`` (event-cue fusion ->
CueEvidence, ball.hybrid.cues.*, corroboration-only), and
``_hybrid_flight_segments`` (hybrid flight-span diagnostics ->
FlightSegment, carrying spin keys through to
ball_orientation.integrate_orientation). Plus a stage-level smoke test
confirming BallTrack.flight_segments comes from the hybrid layer (not
the reference solver) when trajectory=hybrid.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from src.schemas.ball_anchor import BallAnchor, BallAnchorSet
from src.schemas.ball_track import BallTrack
from src.schemas.camera_track import CameraFrame, CameraTrack
from src.stages.ball import (
    BallStage,
    _hybrid_cue_corroboration,
    _hybrid_fix_knots,
    _hybrid_flight_segments,
)
from src.utils.ball_detector import FakeBallDetector


def _pinhole():
    K = np.array([[1500.0, 0, 640.0], [0, 1500.0, 360.0], [0, 0, 1.0]])
    R = np.eye(3)
    C = np.array([0.0, -30.0, 20.0])
    t = -R @ C
    return K, R, t


def _project(p, K, R, t):
    cam = R @ p + t
    pix = K @ cam
    return float(pix[0] / pix[2]), float(pix[1] / pix[2])


@dataclass
class _Fx:
    frame: int
    xyz: tuple[float, float, float]


# ---------------------------------------------------------------------------
# _hybrid_fix_knots
# ---------------------------------------------------------------------------

def test_hybrid_fix_knots_disabled_returns_nothing():
    K, R, t = _pinhole()
    arts = SimpleNamespace(per_frame_K={0: K}, per_frame_R={0: R}, per_frame_t={0: t},
                            distortion=(0.0, 0.0))
    fixes = {0: (np.array([1.0, 2.0, 0.11]), 30.0)}
    knots, dropped = _hybrid_fix_knots(
        fixes=fixes, manual_by_frame={}, artifacts=arts,
        fixes_cfg={"enabled": False})
    assert knots == []
    assert dropped == []


def test_hybrid_fix_knots_enabled_builds_depth_hard_knots():
    K, R, t = _pinhole()
    arts = SimpleNamespace(per_frame_K={0: K}, per_frame_R={0: R}, per_frame_t={0: t},
                            distortion=(0.0, 0.0))
    xyz = (10.0, 20.0, 0.5)
    fixes = {0: (np.array(xyz), 30.0)}
    knots, dropped = _hybrid_fix_knots(
        fixes=fixes, manual_by_frame={}, artifacts=arts,
        fixes_cfg={"enabled": True, "tol_px": 3.0})
    assert dropped == []
    assert len(knots) == 1
    k = knots[0]
    assert k.frame == 0
    assert k.source == "fix"
    assert k.depth_hard is True
    assert np.allclose(k.xyz, xyz)


def test_hybrid_fix_knots_drops_fix_conflicting_with_manual_anchor():
    """Operator-wins: a fix that reprojects far from a manual anchor's
    own click at the same frame is dropped, not silently trusted."""
    K, R, t = _pinhole()
    arts = SimpleNamespace(per_frame_K={0: K}, per_frame_R={0: R}, per_frame_t={0: t},
                            distortion=(0.0, 0.0))
    true_xy = _project(np.array([10.0, 20.0, 0.11]), K, R, t)
    manual = {0: BallAnchor(frame=0, image_xy=true_xy, state="grounded")}
    # A fix far off the manual click's ray (different world point).
    bad_fix_xyz = (10.0, 20.0, 25.0)
    fixes = {0: (np.array(bad_fix_xyz), 30.0)}
    knots, dropped = _hybrid_fix_knots(
        fixes=fixes, manual_by_frame=manual, artifacts=arts,
        fixes_cfg={"enabled": True, "tol_px": 3.0})
    assert knots == []
    assert len(dropped) == 1
    assert dropped[0]["reason"] == "manual_anchor_conflict"


# ---------------------------------------------------------------------------
# _hybrid_cue_corroboration
# ---------------------------------------------------------------------------

def test_hybrid_cue_corroboration_disabled_by_default(tmp_path: Path):
    result = _hybrid_cue_corroboration(
        cues_cfg={"enabled": False}, output_dir=tmp_path, shot_id="s1",
        clip_path=tmp_path / "shots" / "s1.mp4", fps=30.0,
        contact_frames=[10, 20], auto_event_frames=[10],
    )
    assert result == ()


def test_hybrid_cue_corroboration_never_raises_on_missing_clip(tmp_path: Path):
    """Best-effort: a missing/unreadable clip must not raise, even with
    cues enabled."""
    result = _hybrid_cue_corroboration(
        cues_cfg={"enabled": True}, output_dir=tmp_path, shot_id="s1",
        clip_path=tmp_path / "shots" / "does_not_exist.mp4", fps=30.0,
        contact_frames=[10, 20], auto_event_frames=[10],
    )
    assert result == ()


def test_hybrid_cue_corroboration_empty_contact_frames_short_circuits(tmp_path: Path):
    result = _hybrid_cue_corroboration(
        cues_cfg={"enabled": True}, output_dir=tmp_path, shot_id="s1",
        clip_path=tmp_path / "shots" / "s1.mp4", fps=30.0,
        contact_frames=[], auto_event_frames=[],
    )
    assert result == ()


# ---------------------------------------------------------------------------
# _hybrid_flight_segments
# ---------------------------------------------------------------------------

def test_hybrid_flight_segments_builds_parabola_without_spin():
    diag = {"spans": [
        {"span": (10, 40), "model": "flight", "p0": (0.0, 0.0, 0.11),
         "v0": (10.0, 0.0, 8.0), "g": -9.81, "max_residual_px": 2.5},
        {"span": (0, 10), "model": "roll"},  # non-flight, must be skipped
    ]}
    segs = _hybrid_flight_segments(diag)
    assert len(segs) == 1
    seg = segs[0]
    assert seg.id == 0
    assert seg.frame_range == (10, 40)
    assert seg.parabola["p0"] == [0.0, 0.0, 0.11]
    assert seg.parabola["v0"] == [10.0, 0.0, 8.0]
    assert seg.parabola["g"] == -9.81
    assert seg.parabola["spin_axis_world"] is None
    assert seg.parabola["spin_omega_rad_s"] is None
    assert seg.fit_residual_px == pytest.approx(2.5)


def test_hybrid_flight_segments_carries_spin_keys_ball_orientation_reads():
    from src.utils.ball_orientation import _flight_omega

    omega_world = (0.0, 0.0, 12.0)
    diag = {"spans": [
        {"span": (5, 25), "model": "flight", "p0": (1.0, 2.0, 0.11),
         "v0": (9.0, 1.0, 7.0), "g": -9.81, "max_residual_px": 1.1,
         "omega_world": omega_world, "rad_s": 12.0},
    ]}
    segs = _hybrid_flight_segments(diag)
    assert len(segs) == 1
    seg = segs[0]
    assert seg.parabola["spin_axis_world"] == pytest.approx([0.0, 0.0, 1.0])
    assert seg.parabola["spin_omega_rad_s"] == pytest.approx(12.0)
    # The exact field ball_orientation._flight_omega reads to rotate the
    # exported ball -- confirms the wiring is end-to-end correct, not
    # just shaped correctly.
    omega = _flight_omega(seg)
    assert np.allclose(omega, np.array(omega_world))


def test_hybrid_flight_segments_skips_spans_missing_p0v0():
    """A probe span from ball_hybrid_gating's residual gate (or any
    span dict that predates the p0/v0 addition) must be skipped, not
    crash FlightSegment construction."""
    diag = {"spans": [{"span": (0, 5), "model": "flight"}]}
    assert _hybrid_flight_segments(diag) == ()


# ---------------------------------------------------------------------------
# Stage-level smoke: hybrid's own flight segments (with spin, when
# enabled) replace the reference solver's in BallTrack.flight_segments.
# ---------------------------------------------------------------------------

def _save_camera_track(path, K, R, t, n, clip_id="play", fps=30.0):
    track = CameraTrack(
        clip_id=clip_id, fps=fps, image_size=(1280, 720), t_world=t.tolist(),
        frames=tuple(CameraFrame(frame=i, K=K.tolist(), R=R.tolist(),
                                  confidence=1.0, is_anchor=(i == 0))
                     for i in range(n)),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    track.save(path)


def _write_blank_clip(path, n, fps=30.0):
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (1280, 720))
    for _ in range(n):
        writer.write(np.full((720, 1280, 3), [50, 200, 50], dtype=np.uint8))
    writer.release()


def _camera_pose_looking_at_pitch():
    """Same convention as test_ball_stage.py's _camera_pose(): broadcast
    camera actually looking at the pitch (identity R in _pinhole() above
    looks straight up -- fine for the unit tests, wrong for anything
    that must in-bounds-project a real trajectory through the stage)."""
    look = np.array([0.0, 64.0, -30.0])
    look /= np.linalg.norm(look)
    right = np.array([1.0, 0.0, 0.0])
    down = np.cross(look, right)
    R = np.array([right, down, look], dtype=float)
    t = -R @ np.array([52.5, -30.0, 30.0])
    K = np.array([[1500.0, 0, 640.0], [0, 1500.0, 360.0], [0, 0, 1.0]])
    return K, R, t


@pytest.mark.integration
def test_hybrid_trajectory_flight_segments_replace_reference_ones(tmp_path: Path):
    n, fps = 90, 30.0
    K, R, t = _camera_pose_looking_at_pitch()
    _save_camera_track(tmp_path / "camera" / "camera_track.json", K, R, t, n, fps=fps)
    _write_blank_clip(tmp_path / "shots" / "play.mp4", n, fps=fps)

    p_a = np.array([40.0, 34.0, 0.11])
    v0 = np.array([6.0, 3.0, 8.0])
    g = np.array([0.0, 0.0, -9.81])
    frame_a, frame_b = 10, 55
    duration_s = (frame_b - frame_a) / fps
    p_b = p_a + v0 * duration_s + 0.5 * g * duration_s ** 2
    p_b[2] = 0.11

    detections = []
    for i in range(n):
        if frame_a <= i <= frame_b:
            t_s = (i - frame_a) / fps
            w = p_a + v0 * t_s + 0.5 * g * t_s ** 2
        elif i < frame_a:
            w = p_a
        else:
            w = p_b
        u, v = _project(w, K, R, t)
        detections.append((u, v, 0.9))

    anchors = BallAnchorSet(
        clip_id="play", image_size=(1280, 720),
        anchors=(
            BallAnchor(frame=frame_a, image_xy=_project(p_a, K, R, t), state="kick"),
            BallAnchor(frame=frame_b, image_xy=_project(p_b, K, R, t), state="bounce"),
        ),
    )
    (tmp_path / "ball").mkdir(parents=True, exist_ok=True)
    anchors.save(tmp_path / "ball" / "ball_anchors.json")

    stage = BallStage(
        config={"ball": {"detector": "fake", "trajectory": "hybrid"}},
        output_dir=tmp_path, ball_detector=FakeBallDetector(detections),
    )
    stage.run()

    track = BallTrack.load(tmp_path / "ball" / "ball_track.json")
    diag = json.loads((tmp_path / "ball" / "ball_diag.json").read_text())
    hybrid_flight_spans = [s for s in diag["hybrid_trajectory"]["spans"]
                            if s.get("model") == "flight"]
    assert hybrid_flight_spans, "expected the hybrid layer to fit a flight span"
    assert len(track.flight_segments) == len(hybrid_flight_spans)
    for seg, span in zip(track.flight_segments, hybrid_flight_spans):
        assert seg.frame_range == tuple(span["span"])
        assert seg.parabola["p0"] == list(span["p0"])
