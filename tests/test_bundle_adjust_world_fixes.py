"""world_fixes: 3D soft constraints in the LM flight fitters."""

from __future__ import annotations

import numpy as np
import pytest

from src.utils.bundle_adjust import (
    fit_magnus_trajectory,
    fit_parabola_to_image_observations,
)
from tests.fixtures.ball_synthetic import broadcast_camera, project_track

FPS = 25.0
G = np.array([0.0, 0.0, -9.81])


def _arc(n, p0, v0):
    return {f: p0 + v0 * (f / FPS) + 0.5 * G * (f / FPS) ** 2
            for f in range(n)}


def _fit_inputs(worlds, K, R, t, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    pix = project_track(worlds, K, R, t)
    obs = []
    for f, (u, v) in sorted(pix.items()):
        du, dv = rng.normal(0.0, noise, 2) if noise else (0.0, 0.0)
        obs.append((f, (u + du, v + dv)))
    n = len(obs)
    return obs, [K] * n, [R] * n, [t] * n


@pytest.mark.unit
def test_parabola_unchanged_without_fixes():
    K, R, t = broadcast_camera()
    worlds = _arc(30, np.array([40.0, 30.0, 0.11]), np.array([8.0, 4.0, 9.0]))
    obs, Ks, Rs, ts = _fit_inputs(worlds, K, R, t)
    a = fit_parabola_to_image_observations(obs, Ks=Ks, Rs=Rs, t_world=ts, fps=FPS)
    b = fit_parabola_to_image_observations(
        obs, Ks=Ks, Rs=Rs, t_world=ts, fps=FPS, world_fixes=[])
    assert np.allclose(a[0], b[0]) and np.allclose(a[1], b[1])
    assert a[2] == pytest.approx(b[2])


@pytest.mark.unit
def test_parabola_fixes_pull_depth_to_truth():
    """Sparse noisy monocular obs leave depth soft; two exact 3D fixes
    must pull the recovered trajectory onto the truth."""
    K, R, t = broadcast_camera()
    p0, v0 = np.array([40.0, 30.0, 0.11]), np.array([8.0, 4.0, 9.0])
    worlds = _arc(30, p0, v0)
    sparse = {f: worlds[f] for f in range(0, 30, 6)}  # 5 obs only
    obs, Ks, Rs, ts = _fit_inputs(sparse, K, R, t, noise=2.0)
    fixes = [(8, np.asarray(worlds[8]), 30.0), (20, np.asarray(worlds[20]), 30.0)]
    p0n, v0n, _ = fit_parabola_to_image_observations(
        obs, Ks=Ks, Rs=Rs, t_world=ts, fps=FPS)
    p0f, v0f, _ = fit_parabola_to_image_observations(
        obs, Ks=Ks, Rs=Rs, t_world=ts, fps=FPS, world_fixes=fixes)
    err_n = np.linalg.norm(p0n - p0) + np.linalg.norm(v0n - v0)
    err_f = np.linalg.norm(p0f - p0) + np.linalg.norm(v0f - v0)
    assert err_f <= err_n + 1e-9
    assert np.linalg.norm(p0f - p0) < 0.5


@pytest.mark.unit
def test_magnus_accepts_fixes_and_respects_weight():
    K, R, t = broadcast_camera()
    p0, v0 = np.array([40.0, 30.0, 0.11]), np.array([8.0, 4.0, 9.0])
    worlds = _arc(30, p0, v0)
    obs, Ks, Rs, ts = _fit_inputs(worlds, K, R, t, noise=1.0)
    wrong_fix = [(15, np.asarray(worlds[15]) + np.array([0.0, 5.0, 0.0]), 1e-6)]
    # Without any fix, get base error.
    p0_base, _, _, _ = fit_magnus_trajectory(
        obs, Ks=Ks, Rs=Rs, t_world=ts, fps=FPS,
        omega_mag_bound=10.0)
    p0w, v0w, _, _ = fit_magnus_trajectory(
        obs, Ks=Ks, Rs=Rs, t_world=ts, fps=FPS,
        omega_mag_bound=10.0, world_fixes=wrong_fix)
    # Near-zero weight: the wrong fix must not drag the solution away
    # from the base fit (error ≤ base + small tolerance).
    base_err = np.linalg.norm(p0_base - p0)
    assert np.linalg.norm(p0w - p0) <= base_err + 0.01
