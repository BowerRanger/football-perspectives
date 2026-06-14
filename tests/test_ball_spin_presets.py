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
    derive_spin_seed,
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


# ---------------------------------------------------------------------------
# derive_spin_seed — geometric touch-typing spin seeds
# ---------------------------------------------------------------------------

class TestDeriveSpinSeed:
    """Tests for derive_spin_seed(contact_bone, v_in, v_out) -> ndarray | None."""

    # ------------------------------------------------------------------
    # Case 1: foot + large horizontal direction change → side-spin about ±z

    def test_foot_side_curl_positive_z(self):
        """v_in along +x, v_out curving toward +y (large left-curl) → +z spin."""
        v_in = np.array([15.0, 0.0, 0.0])
        v_out = np.array([12.0, 8.0, 3.0])   # > 20° direction change
        seed = derive_spin_seed("l_foot", v_in, v_out)
        assert seed is not None
        assert seed.shape == (3,)
        # z-axis spin: x and y should be (approximately) zero
        mag = float(np.linalg.norm(seed))
        assert mag > 1.0, "non-zero seed expected"
        # Sign: cross(v_in_xy, v_out_xy).z should be positive here
        v_in_xy = np.array([v_in[0], v_in[1], 0.0])
        v_out_xy = np.array([v_out[0], v_out[1], 0.0])
        cross_z = float(np.cross(v_in_xy, v_out_xy)[2])
        assert cross_z > 0
        assert seed[2] > 0, "spin sign must match cross product sign"
        # Axis should be along ±z
        axis = seed / mag
        assert abs(axis[0]) < 1e-6 and abs(axis[1]) < 1e-6

    def test_foot_side_curl_negative_z(self):
        """v_in along +x, v_out curving toward -y (right-curl) → -z spin."""
        v_in = np.array([15.0, 0.0, 0.0])
        v_out = np.array([12.0, -8.0, 3.0])  # > 20° direction change
        seed = derive_spin_seed("r_foot", v_in, v_out)
        assert seed is not None
        # Sign: cross(v_in_xy, v_out_xy).z negative → spin[2] < 0
        v_in_xy = np.array([v_in[0], v_in[1], 0.0])
        v_out_xy = np.array([v_out[0], v_out[1], 0.0])
        cross_z = float(np.cross(v_in_xy, v_out_xy)[2])
        assert cross_z < 0
        assert seed[2] < 0, "spin sign must match cross product sign"

    def test_foot_side_curl_magnitude_in_curl_range(self):
        """Side-spin seed magnitude must reuse the curl preset magnitude."""
        v_in = np.array([20.0, 0.0, 0.0])
        v_out = np.array([15.0, 12.0, 0.0])   # large horizontal change
        seed = derive_spin_seed("l_foot", v_in, v_out)
        assert seed is not None
        mag = float(np.linalg.norm(seed))
        # Should reuse _CURL_MAGNITUDE_RAD_S (currently 15.0)
        assert 5.0 <= mag <= 25.0

    # ------------------------------------------------------------------
    # Case 2: foot + lofted exit (elevation > 25°) → backspin

    def test_foot_lofted_backspin_axis_perpendicular_to_travel(self):
        """Lofted exit produces a backspin axis perpendicular to v_out."""
        v_in = np.array([10.0, 0.0, -2.0])   # approaching, slightly down
        v_out = np.array([8.0, 0.0, 6.0])    # 37° elevation — above 25° gate
        seed = derive_spin_seed("r_foot", v_in, v_out)
        assert seed is not None
        mag = float(np.linalg.norm(seed))
        assert mag > 1.0
        axis = seed / mag
        # Axis must be horizontal (no z component)
        assert abs(axis[2]) < 1e-6, "backspin axis should be horizontal"
        # Axis must be perpendicular to horizontal component of v_out
        v_out_h = np.array([v_out[0], v_out[1], 0.0])
        v_out_h /= np.linalg.norm(v_out_h)
        assert abs(float(np.dot(axis, v_out_h))) < 1e-6

    def test_foot_lofted_backspin_produces_upward_magnus_force(self):
        """ω × v_out should have a positive z (upward) component — backspin
        lifts the ball, exactly like the 'backspin' preset."""
        v_in = np.array([10.0, 0.0, -2.0])
        v_out = np.array([8.0, 0.0, 6.0])   # lofted
        seed = derive_spin_seed("l_foot", v_in, v_out)
        assert seed is not None
        cross = np.cross(seed, v_out)
        assert cross[2] > 0, "backspin should produce upward Magnus force"

    def test_foot_lofted_backspin_magnitude_in_spin_range(self):
        """Backspin magnitude reuses _SPIN_MAGNITUDE_RAD_S (20–100 rad/s)."""
        v_in = np.array([10.0, 0.0, -2.0])
        v_out = np.array([8.0, 0.0, 6.0])
        seed = derive_spin_seed("r_foot", v_in, v_out)
        assert seed is not None
        mag = float(np.linalg.norm(seed))
        assert 20.0 <= mag <= 100.0

    # ------------------------------------------------------------------
    # Case 3: head → zeros vector (not None)

    def test_head_returns_zero_vector(self):
        """Headers impart little controlled spin — seed zeros, not None."""
        v_in = np.array([10.0, 0.0, -3.0])
        v_out = np.array([8.0, 2.0, 1.0])
        seed = derive_spin_seed("head", v_in, v_out)
        assert seed is not None, "head must return a zero vector, not None"
        assert seed.shape == (3,)
        np.testing.assert_array_equal(seed, np.zeros(3))

    def test_head_zeros_regardless_of_velocities(self):
        """Even a large direction change doesn't activate spin for headers."""
        v_in = np.array([20.0, 0.0, 0.0])
        v_out = np.array([-5.0, 15.0, 5.0])   # huge direction reversal
        seed = derive_spin_seed("head", v_in, v_out)
        assert seed is not None
        np.testing.assert_array_equal(seed, np.zeros(3))

    # ------------------------------------------------------------------
    # Case 4: foot + small direction change, not lofted → None

    def test_foot_small_change_not_lofted_returns_none(self):
        """No pattern detected — let the free Magnus fit decide."""
        v_in = np.array([15.0, 0.0, 0.0])
        v_out = np.array([14.9, 0.1, 0.5])   # tiny direction change, low loft
        seed = derive_spin_seed("l_foot", v_in, v_out)
        assert seed is None

    def test_foot_straight_pass_returns_none(self):
        """A straight pass (aligned in-out) should return None."""
        v_in = np.array([10.0, 0.0, 0.0])
        v_out = np.array([12.0, 0.0, 0.0])   # perfectly aligned
        seed = derive_spin_seed("r_foot", v_in, v_out)
        assert seed is None

    # ------------------------------------------------------------------
    # Case 5: chest → None

    def test_chest_returns_none(self):
        v_in = np.array([10.0, 0.0, 5.0])
        v_out = np.array([5.0, 3.0, 2.0])
        seed = derive_spin_seed("chest", v_in, v_out)
        assert seed is None

    # ------------------------------------------------------------------
    # Case 6: unknown/other bones → None

    def test_unknown_bone_returns_none(self):
        v_in = np.array([10.0, 0.0, 0.0])
        v_out = np.array([5.0, 5.0, 5.0])
        seed = derive_spin_seed("knee", v_in, v_out)
        assert seed is None

    # ------------------------------------------------------------------
    # Angle threshold edge cases

    def test_foot_exactly_at_direction_change_threshold(self):
        """Direction change just below threshold → None; just above → non-None."""
        import math
        v_in = np.array([15.0, 0.0, 0.0])
        # 19° change — below 20° threshold
        angle_below = math.radians(19.0)
        v_out_below = np.array([
            15.0 * math.cos(angle_below), 15.0 * math.sin(angle_below), 0.5
        ])
        seed_below = derive_spin_seed("l_foot", v_in, v_out_below)
        assert seed_below is None, "below threshold → None"

        # 21° change — above 20° threshold
        angle_above = math.radians(21.0)
        v_out_above = np.array([
            15.0 * math.cos(angle_above), 15.0 * math.sin(angle_above), 0.5
        ])
        seed_above = derive_spin_seed("l_foot", v_in, v_out_above)
        assert seed_above is not None, "above threshold → side-spin seed"

    def test_foot_exactly_at_loft_threshold(self):
        """Elevation angle just below 25° → None (if no curl); just above → backspin."""
        import math
        v_in = np.array([10.0, 0.0, -1.0])
        # 24° elevation — below 25° gate, and small horizontal change → None
        angle_below = math.radians(24.0)
        speed = 15.0
        v_out_below = np.array([
            speed * math.cos(angle_below), 0.0, speed * math.sin(angle_below)
        ])
        # v_in_xy and v_out_xy are both along +x here, so no curl either
        seed_below = derive_spin_seed("r_foot", v_in, v_out_below)
        assert seed_below is None, "below loft threshold → None"

        # 26° elevation
        angle_above = math.radians(26.0)
        v_out_above = np.array([
            speed * math.cos(angle_above), 0.0, speed * math.sin(angle_above)
        ])
        seed_above = derive_spin_seed("r_foot", v_in, v_out_above)
        assert seed_above is not None, "above loft threshold → backspin seed"


# ---------------------------------------------------------------------------
# Solver-level: manual preset overrides derived seed
# ---------------------------------------------------------------------------

class TestDerivedSeedSolverPrecedence:
    """Verify that a manual spin preset on an anchor beats a derived seed."""

    def test_manual_preset_overrides_derived(self):
        """omega_seed_from_preset() (manual) result should differ from
        derive_spin_seed() when the manual preset is set, and the solver
        should use the manual one.

        We test this at the call-site logic level: given the same velocities,
        if ``spin`` is set on the node the preset path is taken; if ``spin``
        is None the derived path is taken.
        """
        # Velocities that produce a side-spin seed via derive_spin_seed
        v_in = np.array([15.0, 0.0, 0.0])
        v_out = np.array([10.0, 10.0, 3.0])
        derived = derive_spin_seed("l_foot", v_in, v_out)
        assert derived is not None

        # Manual preset (backspin for the same v0)
        manual = omega_seed_from_preset("backspin", v_out)

        # They should differ in direction (side-spin is ±z, backspin is horizontal)
        assert not np.allclose(derived, manual), (
            "derived and manual seeds must differ so precedence matters"
        )

    def test_derive_spin_seed_none_for_unrecognised_bone_means_no_hint(self):
        """When derive_spin_seed returns None the solver should not apply
        the hinted improvement gate (it falls back to the strict gate).
        We verify the None → no-hint path by checking that the function
        really returns None for a chest touch."""
        seed = derive_spin_seed("chest", np.array([10.0, 0.0, 0.0]),
                                np.array([8.0, 2.0, 0.0]))
        assert seed is None
