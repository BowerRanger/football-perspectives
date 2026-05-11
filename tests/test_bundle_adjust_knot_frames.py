"""Tests for the knot_frames kwarg on fit_parabola_to_image_observations."""

from __future__ import annotations

import numpy as np
import pytest

from src.utils.bundle_adjust import fit_parabola_to_image_observations


def _camera():
    K = np.array([[1500.0, 0, 640.0], [0, 1500.0, 360.0], [0, 0, 1.0]])
    R = np.eye(3)
    t = np.array([0.0, 0.0, 0.0])
    return K, R, t


def _synthesise_observations(p0, v0, K, R, t, n: int, fps: float = 30.0):
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


def test_no_knot_frames_matches_baseline():
    """When knot_frames is None, results must equal the existing fit."""
    K, R, t = _camera()
    p0 = np.array([0.0, 5.0, 0.11])
    v0 = np.array([3.0, 0.5, 12.0])
    obs = _synthesise_observations(p0, v0, K, R, t, n=15)

    a = fit_parabola_to_image_observations(
        obs, Ks=[K] * len(obs), Rs=[R] * len(obs), t_world=t, fps=30.0,
    )
    b = fit_parabola_to_image_observations(
        obs, Ks=[K] * len(obs), Rs=[R] * len(obs), t_world=t, fps=30.0,
        knot_frames=None,
    )
    assert np.allclose(a[0], b[0])
    assert np.allclose(a[1], b[1])
    assert a[2] == pytest.approx(b[2])


def test_single_knot_pulls_fit_through_known_point():
    """Anchoring a knot at the apex pulls the noisy fit toward truth."""
    rng = np.random.default_rng(13)
    K, R, t = _camera()
    p0_true = np.array([0.0, 5.0, 0.11])
    v0_true = np.array([3.0, 0.5, 12.0])
    obs = _synthesise_observations(p0_true, v0_true, K, R, t, n=20)
    # Heavy noise so the unconstrained fit drifts.
    noisy = [(fi, (uv[0] + rng.normal(0, 3.0), uv[1] + rng.normal(0, 3.0))) for fi, uv in obs]

    # Compute true apex world position for the knot.
    apex_idx = 12
    apex_world = p0_true + v0_true * (apex_idx / 30.0) + 0.5 * np.array([0, 0, -9.81]) * (apex_idx / 30.0) ** 2

    p0_a, v0_a, resid_a = fit_parabola_to_image_observations(
        noisy, Ks=[K] * len(noisy), Rs=[R] * len(noisy), t_world=t, fps=30.0,
    )
    p0_b, v0_b, resid_b = fit_parabola_to_image_observations(
        noisy, Ks=[K] * len(noisy), Rs=[R] * len(noisy), t_world=t, fps=30.0,
        knot_frames={apex_idx: apex_world},
    )

    def parab_at(p0, v0, fi):
        dt = fi / 30.0
        return p0 + v0 * dt + 0.5 * np.array([0, 0, -9.81]) * dt ** 2

    apex_pred_a = parab_at(p0_a, v0_a, apex_idx)
    apex_pred_b = parab_at(p0_b, v0_b, apex_idx)

    err_a = float(np.linalg.norm(apex_pred_a - apex_world))
    err_b = float(np.linalg.norm(apex_pred_b - apex_world))
    assert err_b < err_a, f"knotted fit should land closer to truth at apex (err_b={err_b:.3f} vs err_a={err_a:.3f})"
    assert err_b < 0.5, f"knotted apex error should be sub-half-metre, got {err_b:.3f}"


def test_two_knots_pin_endpoints():
    """A start-frame knot + end-frame knot should constrain a short arc."""
    K, R, t = _camera()
    p0_true = np.array([0.0, 5.0, 0.11])
    v0_true = np.array([3.0, 0.5, 12.0])
    obs = _synthesise_observations(p0_true, v0_true, K, R, t, n=10)
    apex_world_start = p0_true.copy()
    apex_world_end = p0_true + v0_true * (9 / 30.0) + 0.5 * np.array([0, 0, -9.81]) * (9 / 30.0) ** 2

    p0_fit, v0_fit, resid = fit_parabola_to_image_observations(
        obs, Ks=[K] * len(obs), Rs=[R] * len(obs), t_world=t, fps=30.0,
        knot_frames={0: apex_world_start, 9: apex_world_end},
    )

    def parab_at(p0, v0, fi):
        dt = fi / 30.0
        return p0 + v0 * dt + 0.5 * np.array([0, 0, -9.81]) * dt ** 2

    assert np.linalg.norm(parab_at(p0_fit, v0_fit, 0) - apex_world_start) < 0.5
    assert np.linalg.norm(parab_at(p0_fit, v0_fit, 9) - apex_world_end) < 0.5


def test_knot_frames_with_p0_fixed():
    """knot_frames and p0_fixed compose: p0_fixed pins the start
    exactly, additional knots act as soft constraints elsewhere."""
    K, R, t = _camera()
    p0_true = np.array([0.0, 5.0, 0.11])
    v0_true = np.array([3.0, 0.5, 12.0])
    obs = _synthesise_observations(p0_true, v0_true, K, R, t, n=15)
    apex_idx = 10
    apex_world = p0_true + v0_true * (apex_idx / 30.0) + 0.5 * np.array([0, 0, -9.81]) * (apex_idx / 30.0) ** 2

    p0_fit, v0_fit, resid = fit_parabola_to_image_observations(
        obs, Ks=[K] * len(obs), Rs=[R] * len(obs), t_world=t, fps=30.0,
        p0_fixed=p0_true, knot_frames={apex_idx: apex_world},
    )
    assert np.allclose(p0_fit, p0_true)
    assert np.linalg.norm(v0_fit - v0_true) < 0.5


def test_z_range_frames_enforces_bucket_when_pixels_underdetermine_depth():
    """A monocular flight observed from far away has near-parallel
    camera rays — pixel obs alone can't pin Z. A z_range_frames hinge
    forces the fit's z into the requested bucket at each constrained
    frame."""
    K, R, t = _camera()
    # True flight: low arc with apex at z=4 m.
    p0_true = np.array([0.0, 5.0, 0.11])
    v0_true = np.array([2.0, 0.0, 8.0])
    obs = _synthesise_observations(p0_true, v0_true, K, R, t, n=15)

    # Without any z constraints, the fit is well-determined here because
    # the camera is at the origin and rays diverge. But we can verify
    # the hinge mechanic: ask the fit to put z >= 10 m at frame 5.
    # The fit should comply.
    p0, v0, _ = fit_parabola_to_image_observations(
        obs, Ks=[K] * len(obs), Rs=[R] * len(obs), t_world=t, fps=30.0,
        z_range_frames={5: (10.0, 25.0)},
    )
    dt = 5 / 30.0
    z_at_5 = p0[2] + v0[2] * dt + 0.5 * (dt ** 2) * (-9.81)
    assert z_at_5 >= 9.5, f"hinge should force z >= 10 m at frame 5, got {z_at_5:.2f}"


def test_z_range_frames_silent_when_z_inside_bucket():
    """If the unconstrained fit already lies inside the bucket, the
    hinge contributes zero residual and the fit is unchanged."""
    K, R, t = _camera()
    p0_true = np.array([0.0, 5.0, 0.11])
    v0_true = np.array([2.0, 0.0, 8.0])
    obs = _synthesise_observations(p0_true, v0_true, K, R, t, n=15)
    # True z at frame 5 ≈ 0.11 + 8*(5/30) - 0.5*9.81*(5/30)² = 1.31 m.
    # Bucket [0, 2] should not pull at all.
    p0_a, v0_a, resid_a = fit_parabola_to_image_observations(
        obs, Ks=[K] * len(obs), Rs=[R] * len(obs), t_world=t, fps=30.0,
    )
    p0_b, v0_b, resid_b = fit_parabola_to_image_observations(
        obs, Ks=[K] * len(obs), Rs=[R] * len(obs), t_world=t, fps=30.0,
        z_range_frames={5: (0.0, 2.0)},
    )
    assert np.allclose(p0_a, p0_b, atol=1e-3)
    assert np.allclose(v0_a, v0_b, atol=1e-3)
