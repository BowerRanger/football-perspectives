"""eyes:<PID> rig — a player's smoothed eye-line aimed at the ball."""
from __future__ import annotations

import numpy as np
import pytest

from src.schemas.ball_track import BallFrame, BallTrack
from src.utils.virtual_cameras import RigConfig, build_eyes_track
from tests.test_virtual_cameras import _straight_standing_track


def _ball(frames_xyz: dict[int, tuple[float, float, float]]) -> BallTrack:
    return BallTrack(
        clip_id="shot_01", fps=30.0,
        frames=tuple(BallFrame(frame=f, world_xyz=xyz, state="grounded", confidence=1.0)
                     for f, xyz in sorted(frames_xyz.items())),
        flight_segments=(),
    )


def _centre_and_axis(frame) -> tuple[np.ndarray, np.ndarray]:
    R = np.array(frame.R)
    return -R.T @ np.array(frame.t), R[2]


@pytest.mark.unit
def test_eyes_track_sits_at_eye_height_and_aims_at_ball() -> None:
    track = _straight_standing_track(5)
    ball = _ball({f: (14.0, 25.0, 0.11) for f in range(5)})
    cam = build_eyes_track(track, ball, RigConfig(), (1080, 1920), 30.0, "P001_eyes")
    assert len(cam.frames) == 5
    centre, axis = _centre_and_axis(cam.frames[2])
    assert 1.5 < centre[2] < 2.0          # above the head joint (eye line)
    expect = np.array([14.0, 25.0, 0.11]) - centre
    np.testing.assert_allclose(axis, expect / np.linalg.norm(expect), atol=1e-6)


@pytest.mark.unit
def test_eyes_track_smooths_a_ball_jump() -> None:
    """A one-frame ball spike is averaged out of the aim (smooth_frames=5)."""
    track = _straight_standing_track(5)
    xyz = {f: (14.0, 25.0, 0.11) for f in range(5)}
    xyz[2] = (14.0, 35.0, 0.11)
    raw_cfg = RigConfig(eyes_smooth_frames=1)
    smooth_cfg = RigConfig(eyes_smooth_frames=5)
    _, raw_axis = _centre_and_axis(build_eyes_track(
        track, _ball(xyz), raw_cfg, (1080, 1920), 30.0, "c").frames[2])
    _, smooth_axis = _centre_and_axis(build_eyes_track(
        track, _ball(xyz), smooth_cfg, (1080, 1920), 30.0, "c").frames[2])
    centre, steady_axis = _centre_and_axis(build_eyes_track(
        track, _ball({f: (14.0, 25.0, 0.11) for f in range(5)}), smooth_cfg,
        (1080, 1920), 30.0, "c").frames[2])
    assert np.dot(smooth_axis, steady_axis) > np.dot(raw_axis, steady_axis)


@pytest.mark.unit
def test_eyes_track_without_ball_looks_ahead() -> None:
    track = _straight_standing_track(3)
    cam = build_eyes_track(track, None, RigConfig(), (1080, 1920), 30.0, "c")
    _, axis = _centre_and_axis(cam.frames[0])
    assert abs(axis[2]) < 1e-6            # level gaze along the facing direction
