"""Recover a known parabolic ball trajectory from per-frame image
observations, using the per-frame camera-track refinement (K_t, R_t,
t_world) as the projection model."""

from __future__ import annotations

import numpy as np
import pytest

from src.utils.bundle_adjust import fit_parabola_to_image_observations


@pytest.mark.unit
def test_recover_known_parabola():
    K = np.array([[1500.0, 0, 960], [0, 1500.0, 540], [0, 0, 1]])
    # Broadcast pose -- camera at (52.5, -30, 30) looking at pitch centre.
    look_world = np.array([0.0, 64.0, -30.0])
    look_world = look_world / np.linalg.norm(look_world)
    right_world = np.array([1.0, 0.0, 0.0])
    down_world = np.cross(look_world, right_world)
    R = np.array([right_world, down_world, look_world], dtype=float)
    t = -R @ np.array([52.5, -30.0, 30.0])

    p0 = np.array([30.0, 40.0, 0.5])
    v0 = np.array([12.0, -8.0, 9.0])
    g = np.array([0.0, 0.0, -9.81])
    fps = 30.0
    n = 30
    pts = np.array([p0 + v0 * (i / fps) + 0.5 * g * (i / fps) ** 2 for i in range(n)])
    obs = []
    for fi, p in enumerate(pts):
        cam = R @ p + t
        pix = K @ cam
        obs.append((fi, tuple(pix[:2] / pix[2])))

    Ks = [K] * n
    Rs = [R] * n
    p0_hat, v0_hat, residual = fit_parabola_to_image_observations(
        obs, Ks=Ks, Rs=Rs, t_world=t, fps=fps,
    )
    assert np.linalg.norm(p0_hat - p0) < 0.5
    assert np.linalg.norm(v0_hat - v0) < 0.5
    assert residual < 1.0
