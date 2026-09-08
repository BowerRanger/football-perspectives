from __future__ import annotations

import math

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
def test_orbit_frame_count_matches_span():
    tracks = [_static_track("P001", 50.0, 34.0, n=17)]
    out = vcam.build_orbit_track(tracks, None, vcam.RigConfig(), (1920, 1080), 25.0, "clip")
    assert len(out.frames) == 17


@pytest.mark.unit
def test_orbit_stays_at_fixed_radius_and_height_from_pivot():
    tracks = [_static_track("P001", 50.0, 34.0)]
    cfg = vcam.RigConfig(orbit_radius_m=12.0, orbit_height_m=5.0)
    out = vcam.build_orbit_track(tracks, None, cfg, (1920, 1080), 25.0, "clip")
    pivot = np.array([50.0, 34.0, 0.9])
    for fr in out.frames:
        R, t = np.asarray(fr.R), np.asarray(fr.t)
        C = -R.T @ t
        assert C[2] == pytest.approx(cfg.orbit_height_m, abs=1e-6)
        horiz_dist = np.linalg.norm((C - pivot)[:2])
        assert horiz_dist == pytest.approx(cfg.orbit_radius_m, abs=1e-6)


@pytest.mark.unit
def test_orbit_always_looks_at_pivot():
    tracks = [_static_track("P001", 50.0, 34.0)]
    out = vcam.build_orbit_track(tracks, None, vcam.RigConfig(), (1920, 1080), 25.0, "clip")
    pivot = np.array([50.0, 34.0, 0.9])
    for fr in out.frames:
        R, t = np.asarray(fr.R), np.asarray(fr.t)
        C = -R.T @ t
        fwd = R[2]
        to_target = pivot - C
        assert np.dot(fwd, to_target / np.linalg.norm(to_target)) > 0.999


@pytest.mark.unit
def test_orbit_sweeps_full_configured_arc():
    """Azimuth at the first and last frame should differ by ~orbit_sweep_deg."""
    n = 30
    tracks = [_static_track("P001", 50.0, 34.0, n)]
    cfg = vcam.RigConfig(orbit_sweep_deg=90.0, orbit_radius_m=10.0)
    out = vcam.build_orbit_track(tracks, None, cfg, (1920, 1080), 25.0, "clip")
    pivot = np.array([50.0, 34.0])

    def azimuth(fr):
        R, t = np.asarray(fr.R), np.asarray(fr.t)
        C = -R.T @ t
        rel = C[:2] - pivot
        return math.degrees(math.atan2(rel[0], -rel[1]))

    az0 = azimuth(out.frames[0])
    az_last = azimuth(out.frames[-1])
    assert abs(az_last - az0) == pytest.approx(cfg.orbit_sweep_deg, abs=1.0)


@pytest.mark.unit
def test_orbit_smooths_jittery_pivot():
    n = 50
    zig = _static_track("P001", 50.0, 34.0, n)
    zig.root_t[::2, 0] += 5.0
    cfg = vcam.RigConfig(drone_smooth_frames=25, orbit_sweep_deg=0.0)
    out = vcam.build_orbit_track([zig], None, cfg, (1920, 1080), 25.0, "clip")
    centres = np.array([-(np.asarray(f.R)).T @ np.asarray(f.t) for f in out.frames])
    dx = np.abs(np.diff(centres[:, 0]))
    assert dx.max() < 1.0


@pytest.mark.unit
def test_orbit_empty_tracks_returns_empty_camera():
    out = vcam.build_orbit_track([], None, vcam.RigConfig(), (1920, 1080), 25.0, "clip")
    assert out.frames == ()
