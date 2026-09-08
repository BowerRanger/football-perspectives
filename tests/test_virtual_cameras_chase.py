from __future__ import annotations

import numpy as np
import pytest

from src.schemas.ball_track import BallFrame, BallTrack
from src.schemas.smpl_world import SmplWorldTrack
from src.utils import virtual_cameras as vcam


def _static_track(pid, x, y, n=20):
    return SmplWorldTrack(
        player_id=pid,
        frames=np.arange(n),
        betas=np.zeros(10),
        thetas=np.zeros((n, 24, 3)),
        root_R=np.tile(np.eye(3), (n, 1, 1)),
        root_t=np.tile(np.array([x, y, 0.9]), (n, 1)),
        confidence=np.ones(n),
    )


def _moving_ball(n, x0=10.0, vx=1.0, y=34.0, z=0.11):
    return BallTrack(
        clip_id="clip", fps=25.0, flight_segments=(),
        frames=tuple(
            BallFrame(frame=i, world_xyz=(x0 + vx * i, y, z), state="grounded", confidence=1.0)
            for i in range(n)
        ),
    )


@pytest.mark.unit
def test_chase_frame_count_matches_span():
    tracks = [_static_track("P001", 50.0, 34.0, n=20)]
    ball = _moving_ball(20)
    out = vcam.build_chase_track(tracks, ball, vcam.RigConfig(), (1920, 1080), 25.0, "clip")
    assert len(out.frames) == 20


@pytest.mark.unit
def test_chase_trails_behind_moving_ball_and_looks_at_it():
    n = 30
    tracks = [_static_track("P001", 50.0, 34.0, n)]
    ball = _moving_ball(n, x0=10.0, vx=2.0)  # fast enough to clear chase_min_speed_m_s
    cfg = vcam.RigConfig(chase_back_m=5.0, chase_height_m=2.0, chase_smooth_frames=3)
    out = vcam.build_chase_track(tracks, ball, cfg, (1920, 1080), 25.0, "clip")

    mid = out.frames[15]
    R, t = np.asarray(mid.R), np.asarray(mid.t)
    C = -R.T @ t
    ball_pos = np.array([10.0 + 2.0 * 15, 34.0, 0.11])
    # Camera trails behind the ball along its direction of travel (+x):
    # camera x should be behind (less than) the ball's x.
    assert C[0] < ball_pos[0]
    # chase_height_m is added atop the ball's own height (near ground),
    # not an absolute world height like drone/goal/orbit/dolly.
    assert C[2] == pytest.approx(ball_pos[2] + cfg.chase_height_m, abs=0.05)
    # Looks at the ball.
    fwd = R[2]
    to_target = ball_pos - C
    assert np.dot(fwd, to_target / np.linalg.norm(to_target)) > 0.99
    # High confidence while locked onto real ball motion.
    assert mid.confidence == pytest.approx(1.0)


@pytest.mark.unit
def test_chase_falls_back_to_centroid_when_ball_missing():
    n = 10
    tracks = [_static_track("P001", 42.0, 20.0, n)]
    out_no_ball = vcam.build_chase_track(tracks, None, vcam.RigConfig(), (1920, 1080), 25.0, "clip")
    assert len(out_no_ball.frames) == n
    fr = out_no_ball.frames[5]
    R, t = np.asarray(fr.R), np.asarray(fr.t)
    C = -R.T @ t
    fwd = R[2]
    to_centroid = np.array([42.0, 20.0, 0.9]) - C
    assert np.dot(fwd, to_centroid / np.linalg.norm(to_centroid)) > 0.99
    # Fallback frames carry reduced confidence.
    assert fr.confidence == pytest.approx(0.5)


@pytest.mark.unit
def test_chase_falls_back_to_centroid_when_ball_static():
    n = 15
    tracks = [_static_track("P001", 42.0, 20.0, n)]
    static_ball = BallTrack(
        clip_id="clip", fps=25.0, flight_segments=(),
        frames=tuple(
            BallFrame(frame=i, world_xyz=(60.0, 34.0, 0.11), state="grounded", confidence=1.0)
            for i in range(n)
        ),
    )
    cfg = vcam.RigConfig(chase_min_speed_m_s=0.5)
    out = vcam.build_chase_track(tracks, static_ball, cfg, (1920, 1080), 25.0, "clip")
    fr = out.frames[7]
    R, t = np.asarray(fr.R), np.asarray(fr.t)
    C = -R.T @ t
    fwd = R[2]
    # Fallback target is the shared "action centroid" (same primitive as
    # build_drone_track) — mean of the player root and the static ball,
    # NOT the raw ball position (that's the whole point of the test:
    # a static ball must not be chased).
    centroid = np.array([(42.0 + 60.0) / 2.0, (20.0 + 34.0) / 2.0, (0.9 + 0.11) / 2.0])
    to_centroid = centroid - C
    assert np.dot(fwd, to_centroid / np.linalg.norm(to_centroid)) > 0.99
    to_ball = np.array([60.0, 34.0, 0.11]) - C
    assert not np.allclose(fwd, to_ball / np.linalg.norm(to_ball), atol=1e-2)


@pytest.mark.unit
def test_chase_degenerate_config_does_not_raise_on_coincident_center_target():
    """chase_back_m=0 and chase_height_m=0 with a static ball collapses
    center and target to the same point — must be guarded, not raise."""
    n = 5
    tracks = [_static_track("P001", 50.0, 34.0, n)]
    static_ball = BallTrack(
        clip_id="clip", fps=25.0, flight_segments=(),
        frames=tuple(
            BallFrame(frame=i, world_xyz=(50.0, 34.0, 0.9), state="grounded", confidence=1.0)
            for i in range(n)
        ),
    )
    cfg = vcam.RigConfig(chase_back_m=0.0, chase_height_m=0.0)
    out = vcam.build_chase_track(tracks, static_ball, cfg, (1920, 1080), 25.0, "clip")
    assert len(out.frames) == n  # no exception raised


@pytest.mark.unit
def test_chase_empty_tracks_returns_empty_camera():
    out = vcam.build_chase_track([], None, vcam.RigConfig(), (1920, 1080), 25.0, "clip")
    assert out.frames == ()
