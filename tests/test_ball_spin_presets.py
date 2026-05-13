"""Unit tests for ``src.utils.ball_spin_presets``: preset → omega_seed
mapping consumed by the ball stage Magnus refinement.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.utils.ball_spin_presets import (
    SPIN_ENABLED_STATES,
    SPIN_ENABLED_TOUCH_TYPES,
    VALID_SPIN_PRESETS,
    VALID_TOUCH_TYPES,
    omega_seed_from_preset,
)


def test_valid_presets_and_enabled_states():
    assert VALID_SPIN_PRESETS == frozenset({
        "none", "instep_curl_right", "instep_curl_left",
        "outside_curl_right", "outside_curl_left",
        "topspin", "backspin", "knuckle",
    })
    assert SPIN_ENABLED_STATES == frozenset({"player_touch"})
    assert VALID_TOUCH_TYPES == frozenset({"shot", "volley"})
    assert SPIN_ENABLED_TOUCH_TYPES == frozenset({"shot", "volley"})


@pytest.mark.parametrize("preset", ["none", "knuckle", None])
def test_zero_seed_for_explicit_no_spin(preset):
    """'none', 'knuckle', and missing presets all seed at zero — the
    LM stays at the parabola unless data argues otherwise."""
    seed = omega_seed_from_preset(preset, v0=np.array([10.0, 0.0, 5.0]))
    np.testing.assert_array_equal(seed, np.zeros(3))


def test_curl_presets_are_vertical_axis():
    v0 = np.array([10.0, 0.0, 5.0])
    right = omega_seed_from_preset("instep_curl_right", v0=v0)
    left = omega_seed_from_preset("instep_curl_left", v0=v0)
    # ±z, opposite signs.
    assert right[0] == 0.0 and right[1] == 0.0 and right[2] > 0
    assert left[0] == 0.0 and left[1] == 0.0 and left[2] < 0
    assert np.isclose(right[2], -left[2])
    # Outside-of-foot curl is the mirror of inside-of-foot.
    out_right = omega_seed_from_preset("outside_curl_right", v0=v0)
    out_left = omega_seed_from_preset("outside_curl_left", v0=v0)
    np.testing.assert_allclose(out_right, left)
    np.testing.assert_allclose(out_left, right)


def test_topspin_produces_downward_magnus_force():
    """For a ball moving along +x at the kick, top-spin axis should be
    horizontal and perpendicular to v0 such that ω × v has a -z
    component (downward force → dipping trajectory)."""
    v0 = np.array([10.0, 0.0, 5.0])
    omega = omega_seed_from_preset("topspin", v0=v0)
    # Horizontal axis (no z).
    assert omega[2] == pytest.approx(0.0)
    # Perpendicular to v0 in the horizontal plane.
    vh = np.array([v0[0], v0[1], 0.0])
    assert abs(float(np.dot(omega, vh))) < 1e-6
    # ω × v has negative z.
    cross = np.cross(omega, v0)
    assert cross[2] < 0


def test_backspin_produces_upward_magnus_force():
    v0 = np.array([10.0, 0.0, 5.0])
    omega = omega_seed_from_preset("backspin", v0=v0)
    cross = np.cross(omega, v0)
    assert cross[2] > 0


def test_topspin_with_no_v0_returns_zero():
    """Top/back-spin presets need v0 to orient the axis; without it the
    helper returns zeros so the LM falls back to its default behaviour."""
    seed = omega_seed_from_preset("topspin", v0=None)
    np.testing.assert_array_equal(seed, np.zeros(3))


def test_topspin_with_zero_horizontal_velocity_returns_zero():
    """A pure vertical kick has no horizontal velocity to orient the
    top-spin axis against — return zeros rather than dividing by 0."""
    seed = omega_seed_from_preset("topspin", v0=np.array([0.0, 0.0, 5.0]))
    np.testing.assert_array_equal(seed, np.zeros(3))


def test_unknown_preset_raises():
    with pytest.raises(ValueError, match="unknown spin preset"):
        omega_seed_from_preset("flutter", v0=np.array([1.0, 0.0, 0.0]))


def test_curl_magnitudes_are_in_realistic_range():
    """Curl presets seed in the football-realistic 5–25 rad/s range —
    the actual value is internal but worth pinning so casual tweaks
    don't silently push it into noise / non-physical territory."""
    seed = omega_seed_from_preset(
        "instep_curl_right", v0=np.array([10.0, 0.0, 5.0]),
    )
    mag = float(np.linalg.norm(seed))
    assert 5.0 <= mag <= 25.0


def test_topspin_magnitude_in_realistic_range():
    seed = omega_seed_from_preset(
        "topspin", v0=np.array([20.0, 0.0, 10.0]),
    )
    mag = float(np.linalg.norm(seed))
    assert 20.0 <= mag <= 100.0
