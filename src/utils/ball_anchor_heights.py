"""State→height lookup and classification sets for ball anchors.

Single source of truth: every consumer (Layer 5, server preview,
editor JS-via-API) reads from these constants so coarse bucket
adjustments only happen here.
"""

from __future__ import annotations


_STATE_HEIGHT_M: dict[str, float] = {
    "grounded":      0.11,
    "airborne_low":  1.0,
    "airborne_mid":  6.0,
    "airborne_high": 15.0,
    "kick":          0.11,
    "catch":         1.5,
    "bounce":        0.11,
    "header":        2.5,
    "volley":        1.0,
    "chest":         1.3,
}

# States whose pixel + height pin the trajectory exactly. The body-part
# contact events (header, volley, chest) are included so they can act
# as the boundary of a flight sub-span — the parabola before the contact
# must land at the body part, and the parabola after must start from it.
# Each midpoint produces ≤1 m of vertical error against typical play —
# smaller than the multi-metre depth ambiguity from pixel-only obs.
HARD_KNOT_STATES: frozenset[str] = frozenset({
    "grounded", "kick", "catch", "bounce", "header", "volley", "chest",
})

# States that force the IMM into the flight branch for that frame and
# extend / create a flight segment. Body-part contacts (header, volley,
# chest) are treated as airborne even though they're contact events —
# the ball IS in the air when the body part touches it.
AIRBORNE_STATES: frozenset[str] = frozenset({
    "airborne_low", "airborne_mid", "airborne_high",
    "header", "volley", "chest", "off_screen_flight",
})

# States that mark a flight boundary (split flight runs at this frame).
EVENT_STATES: frozenset[str] = frozenset({
    "kick", "catch", "bounce", "header", "volley", "chest",
})

# States where the ball is physically at ground level (z = 0.11 m).
# Used by the Layer 5 grounded-anchor linear-interp pass to fill the
# world XY of unanchored frames between e.g. a grounded marker and the
# next kick — both endpoints are at the ground so XY interpolates
# smoothly along the pitch. catch (1.5 m) is excluded because the ball
# is in a player's hands; airborne/header/volley/chest are excluded
# because the ball is in the air.
GROUND_LEVEL_STATES: frozenset[str] = frozenset({
    "grounded", "kick", "bounce",
})

# Z-range buckets for the airborne tags. Each entry is (z_min_m, z_max_m).
# Used by the Phase 2 parabola fit as a one-sided hinge constraint: zero
# penalty when the fitted z is inside the bucket, growing penalty outside.
# Hard-knot states are omitted because their world position is pinned via
# knot_frames at the exact state height.
AIRBORNE_BUCKETS: dict[str, tuple[float, float]] = {
    "airborne_low":  (0.0, 2.0),
    "airborne_mid":  (2.0, 10.0),
    "airborne_high": (10.0, 25.0),
}


def state_to_height(state: str) -> float:
    """Return the assumed ball height in metres for ``state``.

    Raises ValueError for ``off_screen_flight`` (no world position) and
    for unknown states.
    """
    if state == "off_screen_flight":
        raise ValueError("off_screen_flight has no associated height")
    if state not in _STATE_HEIGHT_M:
        raise ValueError(f"unknown ball anchor state: {state!r}")
    return _STATE_HEIGHT_M[state]


def airborne_bucket_range(state: str) -> tuple[float, float] | None:
    """Return ``(z_min, z_max)`` metres for an airborne anchor state,
    or ``None`` for non-airborne states. Used as a one-sided hinge
    constraint in the Phase 2 parabola fit.
    """
    return AIRBORNE_BUCKETS.get(state)
