"""Tests for src/utils/ball_hybrid_physics.py.

Ported from prototypes/ball_hybrid_poc/tests/test_hybrid.py's physics
group (see that file's docstring) plus a direct bit-for-bit cross-check
against the PoC module it was ported from, per CONTRACT.md's requirement
that the production physics match the prototype's outputs exactly (or to
<=1e-9) on the same inputs.
"""

from __future__ import annotations

import numpy as np
import pytest

from prototypes.ball_hybrid_poc import hybrid_physics as poc_physics
from src.utils import ball_hybrid_physics as physics

BALL_R = physics.BALL_RADIUS_M


# ---------------------------------------------------------------------------
# Ported behavioural tests (prototypes/ball_hybrid_poc/tests/test_hybrid.py)
# ---------------------------------------------------------------------------

def test_shoot_arc_recovers_gravity_only_analytically():
    p_a = np.array([0.0, 0.0, 0.11])
    p_b = np.array([10.0, 1.0, 0.11])
    v0 = physics.shoot_arc(p_a, 0.0, p_b, 1.0, cd=0.0)
    p_end = physics.simulate(p_a, v0, [1.0], cd=0.0)[0]
    assert np.linalg.norm(p_end - p_b) < 1e-6


def test_shoot_arc_recovers_drag_arc_from_two_noisy_knots():
    rng = np.random.default_rng(7)
    true_cd = 0.30
    p0 = np.array([0.0, 0.0, 1.0])
    v0_true = np.array([18.0, 4.0, 9.0])
    T = 1.8
    times = np.linspace(0.0, T, 13)
    true_traj = physics.simulate(p0, v0_true, times, cd=true_cd)

    knot_noise = 0.03  # 3 cm
    p_a = true_traj[0] + rng.normal(scale=knot_noise, size=3)
    p_b = true_traj[-1] + rng.normal(scale=knot_noise, size=3)

    v0_hat = physics.shoot_arc(p_a, 0.0, p_b, T, cd=true_cd)
    recon = physics.simulate(p_a, v0_hat, times, cd=true_cd)

    err = np.linalg.norm(recon - true_traj, axis=1)
    assert np.all(err < 0.20), f"max error {err.max():.3f} m exceeds 20cm"


def test_fit_roll_segment_endpoint_exact():
    roll = physics.fit_roll_segment(
        (0.0, 0.0), (10.0, 2.0), duration_s=3.0,
        obs=[(1.0, np.array([3.2, 0.7])), (2.0, np.array([6.9, 1.4]))])
    p0 = roll.eval([0.0], z=BALL_R)[0]
    pT = roll.eval([3.0], z=BALL_R)[0]
    assert np.allclose(p0[:2], [0.0, 0.0], atol=1e-9)
    assert np.allclose(pT[:2], [10.0, 2.0], atol=1e-9)
    assert p0[2] == pytest.approx(BALL_R)


def test_fit_roll_segment_clamps_to_friction_bound():
    roll = physics.fit_roll_segment(
        (0.0, 0.0), (1.0, 0.0), duration_s=0.1,
        obs=[(0.05, np.array([50.0, 0.0]))], mu_max=0.9, g=9.81)
    accel_mag = float(np.linalg.norm(roll.accel_xy))
    assert accel_mag <= 0.9 * 9.81 + 1e-6


def test_bounce_velocity_flips_vertical_scales_by_restitution():
    v_in = np.array([5.0, 0.0, -8.0])
    v_out = physics.bounce_velocity(v_in, restitution_e=0.6)
    assert v_out[2] == pytest.approx(0.6 * 8.0)
    assert v_out[0] == pytest.approx(5.0)


def test_hermite_blend_passes_through_endpoints():
    p0 = np.array([0.0, 0.0, 0.0])
    p1 = np.array([1.0, 2.0, 0.0])
    m0 = np.array([1.0, 0.0, 0.0])
    m1 = np.array([1.0, 0.0, 0.0])
    out = physics.hermite_blend(p0, m0, p1, m1, [0.0, 1.0])
    assert np.allclose(out[0], p0)
    assert np.allclose(out[1], p1)


# ---------------------------------------------------------------------------
# Bit-for-bit parity with the PoC module (CONTRACT.md porting requirement)
# ---------------------------------------------------------------------------

def test_simulate_matches_poc_bit_for_bit():
    p0 = np.array([1.0, -2.0, 0.5])
    v0 = np.array([10.0, 5.0, 7.0])
    times = np.linspace(-0.5, 2.0, 26)
    omega = np.array([0.0, 0.0, 20.0])
    a = physics.simulate(p0, v0, times, cd=0.28, omega=omega,
                          magnus_coeff=physics.DEFAULT_MAGNUS_COEFF)
    b = poc_physics.simulate(p0, v0, times, cd=0.28, omega=omega,
                              magnus_coeff=poc_physics.DEFAULT_MAGNUS_COEFF)
    assert np.array_equal(a, b)


def test_shoot_arc_matches_poc_bit_for_bit():
    p_a = np.array([0.0, 0.0, 0.11])
    p_b = np.array([12.0, -3.0, 0.11])
    a = physics.shoot_arc(p_a, 0.0, p_b, 1.4, cd=0.3)
    b = poc_physics.shoot_arc(p_a, 0.0, p_b, 1.4, cd=0.3)
    assert np.allclose(a, b, atol=1e-9, rtol=0)


def test_fit_roll_segment_matches_poc_bit_for_bit():
    obs = [(1.0, np.array([3.2, 0.7]), 1.0), (2.0, np.array([6.9, 1.4]), 2.0)]
    a = physics.fit_roll_segment((0.0, 0.0), (10.0, 2.0), 3.0, obs)
    b = poc_physics.fit_roll_segment((0.0, 0.0), (10.0, 2.0), 3.0, obs)
    assert a.accel_xy == pytest.approx(b.accel_xy, abs=1e-9)
    assert a.a_xy == b.a_xy
    assert a.b_xy == b.b_xy


def test_constants_match_poc():
    assert physics.BALL_RADIUS_M == poc_physics.BALL_RADIUS_M
    assert physics.BALL_MASS_KG == poc_physics.BALL_MASS_KG
    assert physics.CD_DEFAULT == poc_physics.CD_DEFAULT
    assert physics.CD_BOUNDS == poc_physics.CD_BOUNDS
    assert physics.DEFAULT_MAGNUS_COEFF == poc_physics.DEFAULT_MAGNUS_COEFF
