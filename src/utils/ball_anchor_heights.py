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
    # player_touch has no fixed height — the ball stage runs SMPL FK on
    # the named bone at the anchor frame and uses that as the exact
    # world position. This entry is a fallback used only when the
    # SmplWorldTrack is unavailable.
    "player_touch":  1.0,
    # goal_impact has no fixed height — the ball stage intersects the
    # pixel ray with the goal-element geometry (post/crossbar/back_net/
    # side_net) at the anchor frame and uses that as the exact world
    # position. This entry is a fallback used only when the geometry
    # resolver fails (e.g. ray parallel to the surface). 2.44 matches
    # FIFA crossbar height so the fallback plane sits at the goal mouth.
    "goal_impact":   2.44,
}

# States whose pixel + height pin the trajectory exactly. The body-part
# contact events (header, volley, chest) are included so they can act
# as the boundary of a flight sub-span — the parabola before the contact
# must land at the body part, and the parabola after must start from it.
# Each midpoint produces ≤1 m of vertical error against typical play —
# smaller than the multi-metre depth ambiguity from pixel-only obs.
HARD_KNOT_STATES: frozenset[str] = frozenset({
    "grounded", "kick", "catch", "bounce", "header", "volley", "chest",
    "player_touch", "goal_impact",
})

# States that force the IMM into the flight branch for that frame and
# extend / create a flight segment. Body-part contacts and player_touch
# are treated as airborne even though they're contact events — the
# ball IS in the air when the body part touches it. goal_impact is
# always mid-flight (the ball cannot strike the goal frame at rest).
AIRBORNE_STATES: frozenset[str] = frozenset({
    "airborne_low", "airborne_mid", "airborne_high",
    "header", "volley", "chest", "player_touch", "goal_impact",
    "off_screen_flight",
})

# States that mark a flight boundary (split flight runs at this frame).
EVENT_STATES: frozenset[str] = frozenset({
    "kick", "catch", "bounce", "header", "volley", "chest", "player_touch",
    "goal_impact",
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


# Bone tags exposed in the Player-Touch UI, mapped to SMPL joint
# indices (see ``src/utils/smpl_skeleton.py`` SMPL_JOINT_NAMES).
# 'chest' uses spine2 (index 6), the upper-back joint that sits at
# typical chest height in the canonical SMPL rest pose.
BONE_TO_SMPL_INDEX: dict[str, int] = {
    "l_foot":     10,  # left_foot
    "r_foot":     11,  # right_foot
    "l_knee":     4,   # left_knee
    "r_knee":     5,   # right_knee
    "chest":      6,   # spine2 (upper torso)
    "head":       15,  # head
    "l_shoulder": 16,  # left_shoulder
    "r_shoulder": 17,  # right_shoulder
    "l_hand":     22,  # left_hand
    "r_hand":     23,  # right_hand
}

VALID_BONES: frozenset[str] = frozenset(BONE_TO_SMPL_INDEX)


# Goal-element tags exposed in the Goal-Impact UI. Each names a primitive
# of the goal frame or net used by ``src.utils.goal_geometry`` to
# intersect the camera pixel ray with a known 3D surface or line:
#  - post: vertical line (one of the two posts at the relevant goal)
#  - crossbar: horizontal line between the posts at z = goal_height_m
#  - back_net: vertical plane goal_depth_m behind the goal line
#  - side_net: vertical plane through one post, extending back to back_net
VALID_GOAL_ELEMENTS: frozenset[str] = frozenset({
    "post", "crossbar", "back_net", "side_net",
})
