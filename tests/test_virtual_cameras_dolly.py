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
def test_dolly_frame_count_matches_span():
    tracks = [_static_track("P001", 30.0, 34.0, n=12)]
    out = vcam.build_dolly_track(tracks, None, vcam.RigConfig(), (1920, 1080), 25.0, "clip")
    assert len(out.frames) == 12


@pytest.mark.unit
def test_dolly_fixed_y_low_height_tracks_centroid_x():
    tracks = [_static_track("P001", 30.0, 34.0)]
    cfg = vcam.RigConfig(dolly_y_m=-3.0, dolly_height_m=1.0)
    out = vcam.build_dolly_track(tracks, None, cfg, (1920, 1080), 25.0, "clip")
    R, t = np.asarray(out.frames[0].R), np.asarray(out.frames[0].t)
    C = -R.T @ t
    assert C[0] == pytest.approx(30.0, abs=1e-6)  # tracks centroid x
    assert C[1] == pytest.approx(cfg.dolly_y_m, abs=1e-6)  # fixed near touchline
    assert C[2] == pytest.approx(cfg.dolly_height_m, abs=1e-6)  # low


@pytest.mark.unit
def test_dolly_x_follows_centroid_as_it_moves():
    n = 20
    track = _static_track("P001", 10.0, 34.0, n)
    track.root_t[10:, 0] = 90.0
    cfg = vcam.RigConfig(drone_smooth_frames=1)  # no smoothing lag for this check
    out = vcam.build_dolly_track([track], None, cfg, (1920, 1080), 25.0, "clip")
    R0, t0 = np.asarray(out.frames[0].R), np.asarray(out.frames[0].t)
    R19, t19 = np.asarray(out.frames[19].R), np.asarray(out.frames[19].t)
    C0 = -R0.T @ t0
    C19 = -R19.T @ t19
    assert C0[0] == pytest.approx(10.0, abs=1e-6)
    assert C19[0] == pytest.approx(90.0, abs=1e-6)
    # y stays fixed regardless of centroid movement.
    assert C0[1] == pytest.approx(cfg.dolly_y_m, abs=1e-6)
    assert C19[1] == pytest.approx(cfg.dolly_y_m, abs=1e-6)


@pytest.mark.unit
def test_dolly_looks_at_centroid():
    tracks = [_static_track("P001", 55.0, 20.0)]
    out = vcam.build_dolly_track(tracks, None, vcam.RigConfig(), (1920, 1080), 25.0, "clip")
    R, t = np.asarray(out.frames[0].R), np.asarray(out.frames[0].t)
    C = -R.T @ t
    fwd = R[2]
    to_target = np.array([55.0, 20.0, 0.9]) - C
    assert np.dot(fwd, to_target / np.linalg.norm(to_target)) > 0.999


@pytest.mark.unit
def test_dolly_empty_tracks_returns_empty_camera():
    out = vcam.build_dolly_track([], None, vcam.RigConfig(), (1920, 1080), 25.0, "clip")
    assert out.frames == ()
