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
}

# States whose pixel + height should be enforced exactly by the fit.
# `header` height varies a lot (1.8 m standing to 3+ m jumping) so it
# is intentionally NOT a hard knot — the soft-bucket midpoint is used
# as a hint, not a pin.
HARD_KNOT_STATES: frozenset[str] = frozenset({
    "grounded", "kick", "catch", "bounce",
})

# States that force the IMM into the flight branch for that frame and
# extend / create a flight segment. `header` is treated as airborne
# even though it's a contact event — the ball IS in the air when the
# head touches it.
AIRBORNE_STATES: frozenset[str] = frozenset({
    "airborne_low", "airborne_mid", "airborne_high",
    "header", "off_screen_flight",
})

# States that mark a flight boundary (split flight runs at this frame).
EVENT_STATES: frozenset[str] = frozenset({
    "kick", "catch", "bounce", "header",
})


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
