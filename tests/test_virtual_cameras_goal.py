from __future__ import annotations

import numpy as np
import pytest

from src.schemas.smpl_world import SmplWorldTrack
from src.utils import virtual_cameras as vcam
from src.utils.pitch import PITCH_LENGTH, PITCH_WIDTH


def _static_track(pid, x, y, n=10):
    return SmplWorldTrack(
        player_id=pid,
        frames=np.arange(n),
        betas=np.zeros(10),
        thetas=np.zeros((n, 24, 3)),
        root_R=np.tile(np.eye(3), (n, 1, 1)),
        root_t=np.tile(np.array([x, y, 0.9]), (n, 1)),
        confidence=np.ones(n),
    )


# ── build_goal_track ────────────────────────────────────────────────

@pytest.mark.unit
def test_goal_left_sits_behind_left_goal_line():
    tracks = [_static_track("P001", 40.0, 34.0)]
    cfg = vcam.RigConfig()
    track = vcam.build_goal_track("left", tracks, None, cfg, (1920, 1080), 25.0, "clip")
    assert len(track.frames) == 10
    R, t = np.asarray(track.frames[0].R), np.asarray(track.frames[0].t)
    C = -R.T @ t
    assert C[0] == pytest.approx(-cfg.goal_back_m, abs=1e-6)  # behind x=0
    assert C[1] == pytest.approx(PITCH_WIDTH / 2.0, abs=1e-6)
    assert C[2] == pytest.approx(cfg.goal_height_m, abs=1e-6)
    # Looks down-pitch toward the centroid.
    fwd = R[2]
    to_target = np.array([40.0, 34.0, 0.9]) - C
    assert np.dot(fwd, to_target / np.linalg.norm(to_target)) > 0.999


@pytest.mark.unit
def test_goal_right_sits_behind_right_goal_line():
    tracks = [_static_track("P001", 60.0, 34.0)]
    cfg = vcam.RigConfig()
    track = vcam.build_goal_track("right", tracks, None, cfg, (1920, 1080), 25.0, "clip")
    R, t = np.asarray(track.frames[0].R), np.asarray(track.frames[0].t)
    C = -R.T @ t
    assert C[0] == pytest.approx(PITCH_LENGTH + cfg.goal_back_m, abs=1e-6)
    assert C[1] == pytest.approx(PITCH_WIDTH / 2.0, abs=1e-6)


@pytest.mark.unit
def test_goal_invalid_side_raises():
    tracks = [_static_track("P001", 40.0, 34.0)]
    with pytest.raises(ValueError):
        vcam.build_goal_track("center", tracks, None, vcam.RigConfig(), (1920, 1080), 25.0, "clip")


@pytest.mark.unit
def test_goal_pans_with_smoothed_centroid():
    """The camera stays fixed but pans: two different centroid positions
    across the clip must produce two different look directions."""
    n = 20
    track = _static_track("P001", 20.0, 34.0, n)
    track.root_t[10:, 0] = 80.0  # centroid jumps partway through
    cfg = vcam.RigConfig(drone_smooth_frames=3)
    out = vcam.build_goal_track("left", [track], None, cfg, (1920, 1080), 25.0, "clip")
    R0 = np.asarray(out.frames[0].R)
    R_last = np.asarray(out.frames[-1].R)
    assert not np.allclose(R0[2], R_last[2], atol=1e-3)
    # Camera centre itself never moves.
    t0 = np.asarray(out.frames[0].t)
    t_last = np.asarray(out.frames[-1].t)
    C0 = -R0.T @ t0
    C_last = -R_last.T @ t_last
    np.testing.assert_allclose(C0, C_last, atol=1e-9)


@pytest.mark.unit
def test_goal_empty_tracks_returns_empty_camera():
    out = vcam.build_goal_track("left", [], None, vcam.RigConfig(), (1920, 1080), 25.0, "clip")
    assert out.frames == ()


# ── build_goalline_track ────────────────────────────────────────────

@pytest.mark.unit
def test_goalline_left_sits_on_goal_line_near_post():
    tracks = [_static_track("P001", 40.0, 34.0)]
    cfg = vcam.RigConfig()
    track = vcam.build_goalline_track("left", tracks, None, cfg, (1920, 1080), 25.0, "clip")
    assert len(track.frames) == 10
    R, t = np.asarray(track.frames[0].R), np.asarray(track.frames[0].t)
    C = -R.T @ t
    assert C[0] == pytest.approx(0.0, abs=1e-6)  # exactly on the goal line
    assert C[2] == pytest.approx(cfg.goalline_height_m, abs=1e-6)
    # Near post is y = 34 - 3.66 = 30.34; offset inward (+y) toward centre.
    assert 30.34 < C[1] < PITCH_WIDTH / 2.0


@pytest.mark.unit
def test_goalline_right_sits_on_goal_line():
    tracks = [_static_track("P001", 60.0, 34.0)]
    cfg = vcam.RigConfig()
    track = vcam.build_goalline_track("right", tracks, None, cfg, (1920, 1080), 25.0, "clip")
    R, t = np.asarray(track.frames[0].R), np.asarray(track.frames[0].t)
    C = -R.T @ t
    assert C[0] == pytest.approx(PITCH_LENGTH, abs=1e-6)


@pytest.mark.unit
def test_goalline_invalid_side_raises():
    with pytest.raises(ValueError):
        vcam.build_goalline_track(
            "up", [_static_track("P001", 40.0, 34.0)], None, vcam.RigConfig(),
            (1920, 1080), 25.0, "clip")


@pytest.mark.unit
def test_goalline_looks_at_target():
    tracks = [_static_track("P001", 50.0, 34.0)]
    cfg = vcam.RigConfig()
    track = vcam.build_goalline_track("left", tracks, None, cfg, (1920, 1080), 25.0, "clip")
    R, t = np.asarray(track.frames[0].R), np.asarray(track.frames[0].t)
    C = -R.T @ t
    fwd = R[2]
    to_target = np.array([50.0, 34.0, 0.9]) - C
    assert np.dot(fwd, to_target / np.linalg.norm(to_target)) > 0.999
