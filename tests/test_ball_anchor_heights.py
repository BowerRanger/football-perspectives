"""Tests for the BallAnchorState → height table and helpers."""

from __future__ import annotations

import pytest

from src.utils.ball_anchor_heights import (
    AIRBORNE_STATES,
    EVENT_STATES,
    HARD_KNOT_STATES,
    state_to_height,
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
    # Airborne buckets are coarse so do NOT pin world position exactly.
    assert "airborne_low" not in HARD_KNOT_STATES
    assert "airborne_mid" not in HARD_KNOT_STATES
    assert "airborne_high" not in HARD_KNOT_STATES
    # Header height varies too much to pin exactly.
    assert "header" not in HARD_KNOT_STATES
    # Off-screen has no pixel so cannot be a knot.
    assert "off_screen_flight" not in HARD_KNOT_STATES


def test_airborne_state_classification():
    assert "airborne_low" in AIRBORNE_STATES
    assert "airborne_mid" in AIRBORNE_STATES
    assert "airborne_high" in AIRBORNE_STATES
    assert "header" in AIRBORNE_STATES
    assert "off_screen_flight" in AIRBORNE_STATES
    assert "grounded" not in AIRBORNE_STATES
    assert "kick" not in AIRBORNE_STATES


def test_event_states():
    assert EVENT_STATES == frozenset({"kick", "catch", "bounce", "header"})
