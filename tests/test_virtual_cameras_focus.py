"""Shot-planning knobs: rig ``focus`` (centroid/ball/player) and the
orbit sweep window."""
from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from src.schemas.ball_track import BallFrame, BallTrack
from src.utils.virtual_cameras import RigConfig, build_goal_track, build_orbit_track
from tests.test_virtual_cameras import _straight_standing_track


def _player(pid: str, xy: tuple[float, float], n: int = 11):
    t = _straight_standing_track(n)
    root_t = t.root_t.copy()
    root_t[:, 0], root_t[:, 1] = xy
    return dataclasses.replace(t, player_id=pid, root_t=root_t)


def _ball(n: int, xyz=(30.0, 40.0, 0.11)) -> BallTrack:
    return BallTrack(clip_id="s", fps=30.0, flight_segments=(), frames=tuple(
        BallFrame(frame=f, world_xyz=xyz, state="grounded", confidence=1.0)
        for f in range(n)))


def _aim_point_xy(frame) -> np.ndarray:
    """Where the optical axis crosses the horizontal plane through the
    camera's target height is awkward; instead return the axis direction
    projected on the ground (unit)."""
    R = np.array(frame.R)
    d = R[2][:2]
    return d / np.linalg.norm(d)


@pytest.mark.unit
@pytest.mark.parametrize("focus, expect_xy", [
    ("ball", (30.0, 40.0)),
    ("P002", (5.0, 30.0)),
])
def test_goal_track_focus_targets(focus, expect_xy):
    tracks = [_player("P001", (60.0, 10.0)), _player("P002", (5.0, 30.0))]
    cfg = RigConfig(focus=focus)
    cam = build_goal_track("left", tracks, _ball(11), cfg, (1080, 1920), 30.0, "c")
    R = np.array(cam.frames[5].R)
    centre = -R.T @ np.array(cam.frames[5].t)
    want = np.array(expect_xy) - centre[:2]
    np.testing.assert_allclose(_aim_point_xy(cam.frames[5]), want / np.linalg.norm(want),
                               atol=1e-6)


@pytest.mark.unit
def test_focus_unknown_player_raises():
    with pytest.raises(ValueError, match="P999"):
        build_goal_track("left", [_player("P001", (60.0, 10.0))], None,
                         RigConfig(focus="P999"), (1080, 1920), 30.0, "c")


@pytest.mark.unit
def test_orbit_window_confines_sweep():
    tracks = [_player("P001", (50.0, 30.0))]
    cfg = RigConfig(focus="P001", orbit_sweep_deg=90.0, orbit_start_frame=4,
                    orbit_end_frame=6)
    cam = build_orbit_track(tracks, None, cfg, (1080, 1920), 30.0, "c")

    def azimuth(fr):
        R = np.array(fr.R)
        c = -R.T @ np.array(fr.t)
        return math.degrees(math.atan2(c[0] - 50.0, -(c[1] - 30.0)))

    angles = [azimuth(f) for f in cam.frames]
    assert angles[0] == pytest.approx(-45.0, abs=1e-6)   # held before window
    assert angles[4] == pytest.approx(-45.0, abs=1e-6)
    assert angles[5] == pytest.approx(0.0, abs=1e-6)
    assert angles[6] == pytest.approx(45.0, abs=1e-6)
    assert angles[10] == pytest.approx(45.0, abs=1e-6)  # held after window
