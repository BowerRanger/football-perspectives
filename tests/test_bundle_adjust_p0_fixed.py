"""Tests for the p0_fixed kwarg on parabola/Magnus fits."""

from __future__ import annotations

import numpy as np
import pytest

from src.utils.bundle_adjust import fit_parabola_to_image_observations


def _camera() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    K = np.array([[1500.0, 0, 640.0], [0, 1500.0, 360.0], [0, 0, 1.0]])
    R = np.eye(3)
    t = np.array([0.0, 0.0, 0.0])
    return K, R, t


def _synthesise_observations(
    p0: np.ndarray, v0: np.ndarray, K, R, t, n: int, fps: float = 30.0
):
    g_vec = np.array([0.0, 0.0, -9.81])
    obs = []
    for i in range(n):
        dt = i / fps
        pt = p0 + v0 * dt + 0.5 * g_vec * dt ** 2
        cam = R @ pt + t
        u = float((K @ cam)[0] / (K @ cam)[2])
        v = float((K @ cam)[1] / (K @ cam)[2])
        obs.append((i, (u, v)))
    return obs


def test_p0_fixed_none_matches_existing_behaviour():
    K, R, t = _camera()
    # Pick a benign aerial scenario well inside the camera frustum.
    p0_true = np.array([0.0, 5.0, 0.11])
    v0_true = np.array([3.0, 0.5, 12.0])
    obs = _synthesise_observations(p0_true, v0_true, K, R, t, n=15)

    p0_a, v0_a, resid_a = fit_parabola_to_image_observations(
        obs,
        Ks=[K] * len(obs), Rs=[R] * len(obs), t_world=t,
        fps=30.0,
    )
    p0_b, v0_b, resid_b = fit_parabola_to_image_observations(
        obs,
        Ks=[K] * len(obs), Rs=[R] * len(obs), t_world=t,
        fps=30.0,
        p0_fixed=None,
    )
    assert resid_a == pytest.approx(resid_b)
    assert np.allclose(p0_a, p0_b)
    assert np.allclose(v0_a, v0_b)


def test_p0_fixed_pins_p0_exactly():
    K, R, t = _camera()
    p0_true = np.array([0.0, 5.0, 0.11])
    v0_true = np.array([3.0, 0.5, 12.0])
    obs = _synthesise_observations(p0_true, v0_true, K, R, t, n=15)

    p0_anchored, v0_anchored, resid = fit_parabola_to_image_observations(
        obs,
        Ks=[K] * len(obs), Rs=[R] * len(obs), t_world=t,
        fps=30.0,
        p0_fixed=p0_true,
    )
    assert np.allclose(p0_anchored, p0_true)
    assert np.allclose(v0_anchored, v0_true, atol=0.1)
    assert resid < 0.5


def test_p0_fixed_with_noisy_observations_recovers_v0():
    rng = np.random.default_rng(7)
    K, R, t = _camera()
    p0_true = np.array([0.0, 5.0, 0.11])
    v0_true = np.array([3.0, 0.5, 12.0])
    obs = _synthesise_observations(p0_true, v0_true, K, R, t, n=15)
    noisy = [(fi, (uv[0] + rng.normal(0, 0.5), uv[1] + rng.normal(0, 0.5))) for fi, uv in obs]

    _, v0_recovered, resid = fit_parabola_to_image_observations(
        noisy,
        Ks=[K] * len(noisy), Rs=[R] * len(noisy), t_world=t,
        fps=30.0,
        p0_fixed=p0_true,
    )
    assert np.linalg.norm(v0_recovered - v0_true) < 1.0
    assert resid < 2.0
