"""Replay playback-speed + offset estimation from players' pitch positions."""

from __future__ import annotations

import numpy as np
import pytest

from src.utils import replay_speed as rs


def _players(n_players: int, n_frames: int, seed: int = 0) -> np.ndarray:
    """(n_frames, n_players, 2) smooth pitch trajectories at 25 fps."""
    rng = np.random.default_rng(seed)
    pos = np.column_stack([rng.uniform(20, 85, n_players), rng.uniform(8, 60, n_players)])
    vel = rng.normal(0, 2.5, (n_players, 2))
    out = np.empty((n_frames, n_players, 2))
    for f in range(n_frames):
        vel += rng.normal(0, 0.6, vel.shape) * 0.2
        vel = np.clip(vel, -7, 7)
        pos = pos + vel / 25.0
        out[f] = pos
    return out


def _sample(traj: np.ndarray, t: float) -> np.ndarray:
    i = int(np.floor(t))
    a = t - i
    i = max(0, min(i, len(traj) - 2))
    return (1 - a) * traj[i] + a * traj[i + 1]


def _views(traj, rate, offset, n_rep, *, seed=1, noise=0.25, focus_r=18.0, rate2=None):
    rng = np.random.default_rng(seed)
    live = {f: traj[f] + rng.normal(0, noise, traj[f].shape) for f in range(len(traj))}
    focus = traj[int(offset + rate * n_rep / 2)].mean(axis=0)
    replay = {}
    t = float(offset)
    for j in range(n_rep):
        r = rate if (rate2 is None or j < n_rep // 2) else rate2
        p = _sample(traj, t)
        keep = np.linalg.norm(p - focus, axis=1) < focus_r
        pts = p[keep] + rng.normal(0, noise, (int(keep.sum()), 2))
        if len(pts):
            replay[j] = pts
        t += r
    return live, replay


@pytest.mark.parametrize("rate,offset", [(1.0, 60.0), (0.34, 85.0), (0.5, 40.0)])
def test_recovers_rate_and_offset(rate, offset):
    traj = _players(22, 320)
    live, replay = _views(traj, rate, offset, n_rep=int(150 / max(rate, 0.4)))
    est = rs.estimate_speed(live, replay)
    assert est is not None
    assert est.rate == pytest.approx(rate, rel=0.03)
    assert est.offset == pytest.approx(offset, abs=2.0)
    assert est.cost_m < 1.0
    assert est.confidence > 0.6
    assert not est.ramp


def test_flags_a_speed_ramp():
    traj = _players(22, 320, seed=3)
    live, replay = _views(traj, 0.27, 60.0, n_rep=220, rate2=0.42, seed=4)
    est = rs.estimate_speed(live, replay)
    assert est is not None and est.ramp
    assert est.rate_first == pytest.approx(0.27, rel=0.1)
    assert est.rate_second == pytest.approx(0.42, rel=0.1)


def test_unrelated_replay_has_low_confidence():
    live, _ = _views(_players(22, 320, seed=5), 1.0, 50.0, n_rep=10)
    _, replay = _views(_players(22, 320, seed=9), 1.0, 50.0, n_rep=120, seed=7)
    est = rs.estimate_speed(live, replay)
    assert est is None or est.confidence < 0.4


def test_too_little_overlap_returns_none():
    assert rs.estimate_speed({0: np.zeros((3, 2))}, {0: np.zeros((2, 2))}) is None


def test_feet_on_pitch_projects_box_bottoms():
    # camera 20 m up at the halfway line, looking at the pitch centre
    centre, target = np.array([52.5, -30.0, 20.0]), np.array([52.5, 34.0, 0.0])
    fwd = (target - centre) / np.linalg.norm(target - centre)
    right = np.cross(fwd, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    R = np.stack([right, down, fwd])
    t = -R @ centre
    K = np.array([[1500.0, 0, 960], [0, 1500.0, 540], [0, 0, 1]])
    p = np.array([40.0, 30.0, 0.0])
    uvw = K @ (R @ p + t)
    u, v = uvw[:2] / uvw[2]
    tracks = {"tracks": [{"class_name": "player", "frames": [
        {"frame": 3, "bbox": [u - 10, v - 60, u + 10, v], "interpolated": False}]}]}
    pts = rs.feet_on_pitch(tracks, lambda f: (K, R, t, (0.0, 0.0)))
    assert set(pts) == {3}
    assert np.allclose(pts[3][0], p[:2], atol=1e-6)
