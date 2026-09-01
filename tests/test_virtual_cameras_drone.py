from __future__ import annotations

import numpy as np
import pytest

from src.schemas.smpl_world import SmplWorldTrack
from src.utils import virtual_cameras as vcam


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


@pytest.mark.unit
def test_drone_hovers_behind_and_above_centroid():
    tracks = [_static_track("P001", 40.0, 30.0), _static_track("P002", 60.0, 30.0)]
    cfg = vcam.RigConfig()
    track = vcam.build_drone_track(tracks, None, cfg, (1920, 1080), 25.0, "clip")
    assert len(track.frames) == 10
    fr = track.frames[0]
    R, t = np.asarray(fr.R), np.asarray(fr.t)
    C = -R.T @ t                       # camera centre in world
    assert C[2] == pytest.approx(cfg.drone_height_m, abs=1e-6)
    assert C[0] == pytest.approx(50.0, abs=1e-6)          # centroid x
    assert C[1] == pytest.approx(30.0 - cfg.drone_back_m, abs=1e-6)
    # looks at the centroid: forward (row 2 of R) points from C to centroid
    fwd = R[2]
    to_target = np.array([50.0, 30.0, 0.9]) - C
    assert np.dot(fwd, to_target / np.linalg.norm(to_target)) > 0.999


@pytest.mark.unit
def test_drone_smooths_jittery_centroid():
    n = 50
    zig = _static_track("P001", 50.0, 30.0, n)
    zig.root_t[::2, 0] += 5.0          # 5 m x-jitter every other frame
    cfg = vcam.RigConfig(drone_smooth_frames=25)
    track = vcam.build_drone_track([zig], None, cfg, (1920, 1080), 25.0, "clip")
    centres = np.array([-(np.asarray(f.R)).T @ np.asarray(f.t)
                        for f in track.frames])
    dx = np.abs(np.diff(centres[:, 0]))
    assert dx.max() < 1.0              # jitter absorbed by the moving average


@pytest.mark.unit
def test_drone_includes_ball_in_centroid():
    tracks = [_static_track("P001", 40.0, 30.0)]
    # Create frame objects with frame and world_xyz attributes
    # to match what _ball_xyz_by_frame expects
    class BallFrame:
        def __init__(self, frame_idx, xyz):
            self.frame = frame_idx
            self.world_xyz = xyz

    ball = type("BT", (), {"frames": [
        BallFrame(i, [80.0, 30.0, 0.11]) for i in range(10)]})()
    cfg = vcam.RigConfig()
    with_ball = vcam.build_drone_track(tracks, ball, cfg, (1920, 1080), 25.0, "c")
    without = vcam.build_drone_track(tracks, None, cfg, (1920, 1080), 25.0, "c")
    cx_with = -(np.asarray(with_ball.frames[0].R)).T @ np.asarray(with_ball.frames[0].t)
    cx_without = -(np.asarray(without.frames[0].R)).T @ np.asarray(without.frames[0].t)
    assert cx_with[0] > cx_without[0]  # ball at x=80 pulls the view right
