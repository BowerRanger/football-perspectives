"""Tests for ``src/utils/ball_bench_truth.py``: the independent physics
simulator (unit tests) plus an import-independence grep test."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

import src.utils.ball_bench_truth as bt

REPO_ROOT = Path(__file__).resolve().parents[1]
_MODULE_PATH = REPO_ROOT / "src" / "utils" / "ball_bench_truth.py"

# --------------------------------------------------------------------------
# Physics unit tests
# --------------------------------------------------------------------------


def test_drag_deceleration_30ms():
    """A 30 m/s ball with Cd 0.25 decelerates 12 +/- 1.5 m/s^2 initially."""
    v = np.array([30.0, 0.0, 0.0])
    a = bt.drag_accel(v, bt.DragParams(cd_const=0.25, crisis=False))
    decel = float(np.linalg.norm(a))
    assert 10.5 <= decel <= 13.5, decel


def test_drag_crisis_transition_monotonic_and_bounded():
    p = bt.DragParams(crisis=True, cd_low=0.45, cd_high=0.20,
                      v_low=10.0, v_high=20.0)
    assert bt.drag_coefficient(5.0, p) == pytest.approx(0.45)
    assert bt.drag_coefficient(25.0, p) == pytest.approx(0.20)
    mid = bt.drag_coefficient(15.0, p)
    assert 0.20 < mid < 0.45


def test_bounce_loses_energy():
    """Restitution < 1 must reduce kinetic energy. Also checks the ball
    doesn't reverse its horizontal direction on a routine bounce."""
    vel = np.array([4.0, 1.0, -8.0])
    omega = np.array([0.0, 5.0, 0.0])  # mild topspin
    ke_before = 0.5 * bt.BALL_MASS_KG * float(np.dot(vel, vel))
    vel2, omega2 = bt.apply_bounce(vel, omega, bt.BounceParams())
    ke_after = 0.5 * bt.BALL_MASS_KG * float(np.dot(vel2, vel2))
    assert ke_after < ke_before
    assert vel2[2] > 0, "ball must bounce upward after a downward approach"
    assert np.all(np.isfinite(omega2))


def test_bounce_restitution_controls_energy_loss():
    vel = np.array([2.0, 0.0, -6.0])
    omega = np.zeros(3)
    v_bouncy, _ = bt.apply_bounce(vel, omega, bt.BounceParams(e_n=0.75))
    v_dead, _ = bt.apply_bounce(vel, omega, bt.BounceParams(e_n=0.60))
    assert abs(v_bouncy[2]) > abs(v_dead[2])


def test_roll_distance_matches_solved_initial_speed():
    p = bt.RollParams()
    for target_dist in (0.5, 5.0, 20.0):
        for duration in (0.3, 1.0, 3.0):
            v0 = bt.solve_roll_initial_speed(target_dist, duration, p)
            got = bt.roll_distance_at(v0, duration, p)
            assert got == pytest.approx(target_dist, abs=1e-3), (
                target_dist, duration, v0, got)


def test_roll_distance_zero_target_gives_zero_speed():
    assert bt.solve_roll_initial_speed(0.0, 1.0, bt.RollParams()) == 0.0


def test_simulate_flight_no_drag_no_spin_matches_projectile():
    """Sanity: with drag/lift forced to (near) zero, the arc should match
    a textbook projectile within numerical tolerance."""
    drag = bt.DragParams(cd_const=1e-9, crisis=False)
    spin = bt.SpinParams(omega0=np.zeros(3))
    v0 = np.array([10.0, 0.0, 8.0])
    duration = 1.0
    res = bt.simulate_flight(np.zeros(3), v0, duration, drag, spin)
    pos, vel = res.state_at(duration)
    expected_x = v0[0] * duration
    expected_z = v0[2] * duration - 0.5 * bt.G * duration ** 2
    assert pos[0] == pytest.approx(expected_x, abs=0.05)
    assert pos[2] == pytest.approx(expected_z, abs=0.05)
    assert vel[2] == pytest.approx(v0[2] - bt.G * duration, abs=0.05)


# --------------------------------------------------------------------------
# Independence: no ball_physics / ball_piecewise_solver / ball_hybrid_* /
# BallStage import anywhere in this module (see module docstring — the
# builder section is allowed to use ball_eval/goal_geometry, which are
# grading/GT primitives, not solvers).
# --------------------------------------------------------------------------

_FORBIDDEN_IMPORT_SUBSTRINGS = (
    "ball_physics", "ball_piecewise_solver", "ball_hybrid", "bundle_adjust",
    "stages.ball", "stages.camera", "line_camera_refine", "anchor_solver",
)


def test_ball_bench_truth_has_no_physics_or_solver_imports():
    tree = ast.parse(_MODULE_PATH.read_text())
    modules = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                modules.append(node.module)
    for mod in modules:
        for bad in _FORBIDDEN_IMPORT_SUBSTRINGS:
            assert bad not in mod, (
                f"ball_bench_truth.py imported {mod!r} containing "
                f"forbidden substring {bad!r} — this module must not "
                "depend on the pipeline's own ball physics/solver family")
