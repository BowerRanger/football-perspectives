"""Tests for the BallAnchorState → height table and helpers."""

from __future__ import annotations

import pytest

from src.utils.ball_anchor_heights import (
    AIRBORNE_BUCKETS,
    AIRBORNE_STATES,
    EVENT_STATES,
    HARD_KNOT_STATES,
    airborne_bucket_range,
    state_to_height,
)


def test_airborne_bucket_ranges():
    assert airborne_bucket_range("airborne_low") == (0.0, 2.0)
    assert airborne_bucket_range("airborne_mid") == (2.0, 10.0)
    assert airborne_bucket_range("airborne_high") == (10.0, 25.0)


def test_non_airborne_states_have_no_bucket_range():
    assert airborne_bucket_range("grounded") is None
    assert airborne_bucket_range("kick") is None
    assert airborne_bucket_range("catch") is None
    assert airborne_bucket_range("bounce") is None
    assert airborne_bucket_range("header") is None
    assert airborne_bucket_range("off_screen_flight") is None


def test_bucket_midpoints_match_state_heights():
    # Sanity: the existing state-height table should sit inside each
    # airborne bucket's range.
    for state, (z_min, z_max) in AIRBORNE_BUCKETS.items():
        h = state_to_height(state)
        assert z_min <= h <= z_max, (
            f"state_to_height({state!r})={h} is outside bucket [{z_min}, {z_max}]"
        )


def test_grounded_height():
    assert state_to_height("grounded") == 0.11


def test_airborne_bucket_heights():
    assert state_to_height("airborne_low") == 1.0
    assert state_to_height("airborne_mid") == 6.0
    assert state_to_height("airborne_high") == 15.0


def test_event_heights():
    assert state_to_height("kick") == 0.11
    assert state_to_height("bounce") == 0.11
    assert state_to_height("catch") == 1.5
    assert state_to_height("header") == 2.5
    assert state_to_height("volley") == 1.0
    assert state_to_height("chest") == 1.3


def test_off_screen_flight_has_no_height():
    with pytest.raises(ValueError):
        state_to_height("off_screen_flight")


def test_unknown_state_raises():
    with pytest.raises(ValueError):
        state_to_height("nonsense")


def test_hard_knot_states():
    assert "grounded" in HARD_KNOT_STATES
    assert "kick" in HARD_KNOT_STATES
    assert "catch" in HARD_KNOT_STATES
    assert "bounce" in HARD_KNOT_STATES
    # Body-part contact events pin the trajectory at the contact height
    # (head 2.5 m, volley 1.0 m, chest 1.3 m). Heights vary by ~1 m
    # per event but that's much less than the multi-metre depth slop
    # we'd see treating them as pixel-only obs.
    assert "header" in HARD_KNOT_STATES
    assert "volley" in HARD_KNOT_STATES
    assert "chest" in HARD_KNOT_STATES
    # Airborne buckets are coarse so do NOT pin world position exactly.
    assert "airborne_low" not in HARD_KNOT_STATES
    assert "airborne_mid" not in HARD_KNOT_STATES
    assert "airborne_high" not in HARD_KNOT_STATES
    # Off-screen has no pixel so cannot be a knot.
    assert "off_screen_flight" not in HARD_KNOT_STATES


def test_airborne_state_classification():
    assert "airborne_low" in AIRBORNE_STATES
    assert "airborne_mid" in AIRBORNE_STATES
    assert "airborne_high" in AIRBORNE_STATES
    assert "header" in AIRBORNE_STATES
    assert "volley" in AIRBORNE_STATES
    assert "chest" in AIRBORNE_STATES
    assert "off_screen_flight" in AIRBORNE_STATES
    assert "grounded" not in AIRBORNE_STATES
    assert "kick" not in AIRBORNE_STATES


def test_event_states():
    assert EVENT_STATES == frozenset({
        "kick", "catch", "bounce", "header", "volley", "chest",
        "player_touch", "goal_impact",
    })


def test_goal_impact_classification():
    """goal_impact pins the trajectory at the geometric contact point
    (hard knot), forces the IMM into the flight branch (airborne), and
    splits flight runs (event). It is NOT a ground-level state."""
    from src.utils.ball_anchor_heights import GROUND_LEVEL_STATES

    assert "goal_impact" in HARD_KNOT_STATES
    assert "goal_impact" in AIRBORNE_STATES
    assert "goal_impact" in EVENT_STATES
    assert "goal_impact" not in GROUND_LEVEL_STATES


def test_goal_impact_fallback_height():
    """The fallback height for goal_impact is the crossbar — matches
    FIFA goal height. Used only when the geometry resolver fails."""
    assert state_to_height("goal_impact") == 2.44


def test_valid_goal_elements():
    from src.utils.ball_anchor_heights import VALID_GOAL_ELEMENTS

    assert VALID_GOAL_ELEMENTS == frozenset({
        "post", "crossbar", "back_net", "side_net",
    })
