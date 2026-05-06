"""FIFA-standard pitch landmark catalogue (105 x 68 m).

Coordinate system: x along nearside touchline (0 = near-left corner,
105 = near-right corner), y from near (0) to far (68), z up. Goal
crossbars at z = 2.44 m, corner flag tops at z = 1.5 m.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PitchLandmark:
    name: str
    world_xyz: tuple[float, float, float]


# FIFA-standard goal width = 7.32 m, centred on the goal line midpoint.
_GOAL_HALF_W = 7.32 / 2
_PITCH_LEN = 105.0
_PITCH_WID = 68.0
_GOAL_HEIGHT = 2.44
_FLAG_HEIGHT = 1.5
# Penalty arc radius is 9.15 m centred on the penalty spot (11 m from goal
# line). The arc intersects the 18-yard box edge at y = 34 ± _D_DY.
_D_DY = (9.15**2 - 5.5**2) ** 0.5  # ≈ 7.3125 m
# Penalty arc intersection with the centre line (y=34): the arc crosses
# y=34 at x = penalty_spot_x ± 9.15. The "D" side (outside the 18-yd box)
# is at penalty_spot_x + 9.15 = 11 + 9.15 = 20.15 m.
_D_CENTRE_X = 11.0 + 9.15  # 20.15 m
# Centre line × centre circle intersections: circle radius 9.15 m about
# (52.5, 34) crosses y=34 at x = 52.5 ± 9.15.
_CC_HALF_X = 9.15

# Mow-grid landmarks within the 18-yard box: each mow square is 5.5 m on
# a side (the 18-yd box depth divided by 3 — see also `_D_DY` above for
# context). Anchored so the centre line at y=34 is a stripe boundary;
# the "near" mow line is 3 squares (= one box-depth, 16.5 m, conventional
# "18 yards") below centre, and the "far" mow line 3 squares above. These
# extend across the full pitch length, so each crossing of a painted
# feature (goal line, 18-yd box front, halfway line) gives a known-world
# point landmark even when no painted intersection is in shot. Per
# user-observed groundskeeping convention, this grid is consistent across
# top-league pitches inside the 18-yard box; midfield-zone stripe count
# varies per stadium.
_MOW_SQUARE_M = 16.5 / 3  # 5.5 m
_NEAR_MOW_Y = 34.0 - 3 * _MOW_SQUARE_M  # 17.5 m
_FAR_MOW_Y = 34.0 + 3 * _MOW_SQUARE_M   # 50.5 m

_LANDMARKS: tuple[PitchLandmark, ...] = (
    # Corners (z=0)
    PitchLandmark("near_left_corner",  (0.0,        0.0,        0.0)),
    PitchLandmark("near_right_corner", (_PITCH_LEN, 0.0,        0.0)),
    PitchLandmark("far_left_corner",   (0.0,        _PITCH_WID, 0.0)),
    PitchLandmark("far_right_corner",  (_PITCH_LEN, _PITCH_WID, 0.0)),
    # Halfway line on each touchline
    PitchLandmark("halfway_near", (_PITCH_LEN / 2, 0.0,        0.0)),
    PitchLandmark("halfway_far",  (_PITCH_LEN / 2, _PITCH_WID, 0.0)),
    # Centre circle centre + cardinal points (radius 9.15 m)
    PitchLandmark("centre_spot",        (52.5, 34.0,             0.0)),
    PitchLandmark("centre_circle_near", (52.5, 34.0 - 9.15,       0.0)),
    PitchLandmark("centre_circle_far",  (52.5, 34.0 + 9.15,       0.0)),
    # 18-yard box — inner corners (16.5 m from goal line)
    PitchLandmark("left_18yd_near",  (16.5,             34.0 - 20.16, 0.0)),
    PitchLandmark("left_18yd_far",   (16.5,             34.0 + 20.16, 0.0)),
    PitchLandmark("right_18yd_near", (_PITCH_LEN - 16.5, 34.0 - 20.16, 0.0)),
    PitchLandmark("right_18yd_far",  (_PITCH_LEN - 16.5, 34.0 + 20.16, 0.0)),
    # 18-yard box — outer corners (where the box meets the goal line)
    PitchLandmark("left_18yd_goal_near",  (0.0,        34.0 - 20.16, 0.0)),
    PitchLandmark("left_18yd_goal_far",   (0.0,        34.0 + 20.16, 0.0)),
    PitchLandmark("right_18yd_goal_near", (_PITCH_LEN, 34.0 - 20.16, 0.0)),
    PitchLandmark("right_18yd_goal_far",  (_PITCH_LEN, 34.0 + 20.16, 0.0)),
    # 18-yard box — penalty arc D intersections (where the arc meets the box edge)
    PitchLandmark("left_18yd_d_near",  (16.5,             34.0 - _D_DY, 0.0)),
    PitchLandmark("left_18yd_d_far",   (16.5,             34.0 + _D_DY, 0.0)),
    PitchLandmark("right_18yd_d_near", (_PITCH_LEN - 16.5, 34.0 - _D_DY, 0.0)),
    PitchLandmark("right_18yd_d_far",  (_PITCH_LEN - 16.5, 34.0 + _D_DY, 0.0)),
    # 6-yard box — inner corners (5.5 m from goal line)
    PitchLandmark("left_6yd_near",  (5.5,              34.0 - 9.16,  0.0)),
    PitchLandmark("left_6yd_far",   (5.5,              34.0 + 9.16,  0.0)),
    PitchLandmark("right_6yd_near", (_PITCH_LEN - 5.5,  34.0 - 9.16,  0.0)),
    PitchLandmark("right_6yd_far",  (_PITCH_LEN - 5.5,  34.0 + 9.16,  0.0)),
    # 6-yard box — outer corners (where the box meets the goal line)
    PitchLandmark("left_6yd_goal_near",  (0.0,        34.0 - 9.16, 0.0)),
    PitchLandmark("left_6yd_goal_far",   (0.0,        34.0 + 9.16, 0.0)),
    PitchLandmark("right_6yd_goal_near", (_PITCH_LEN, 34.0 - 9.16, 0.0)),
    PitchLandmark("right_6yd_goal_far",  (_PITCH_LEN, 34.0 + 9.16, 0.0)),
    # Penalty spots (11 m from goal line, on goal centreline)
    PitchLandmark("left_penalty_spot",  (11.0,              34.0, 0.0)),
    PitchLandmark("right_penalty_spot", (_PITCH_LEN - 11.0, 34.0, 0.0)),
    # Goal post bases (z = 0)
    PitchLandmark("left_goal_left_post_base",   (0.0,        34.0 - _GOAL_HALF_W, 0.0)),
    PitchLandmark("left_goal_right_post_base",  (0.0,        34.0 + _GOAL_HALF_W, 0.0)),
    PitchLandmark("right_goal_left_post_base",  (_PITCH_LEN, 34.0 - _GOAL_HALF_W, 0.0)),
    PitchLandmark("right_goal_right_post_base", (_PITCH_LEN, 34.0 + _GOAL_HALF_W, 0.0)),
    # Goal crossbar endpoints (z = 2.44)
    PitchLandmark("left_goal_crossbar_left",   (0.0,        34.0 - _GOAL_HALF_W, _GOAL_HEIGHT)),
    PitchLandmark("left_goal_crossbar_right",  (0.0,        34.0 + _GOAL_HALF_W, _GOAL_HEIGHT)),
    PitchLandmark("right_goal_crossbar_left",  (_PITCH_LEN, 34.0 - _GOAL_HALF_W, _GOAL_HEIGHT)),
    PitchLandmark("right_goal_crossbar_right", (_PITCH_LEN, 34.0 + _GOAL_HALF_W, _GOAL_HEIGHT)),
    # Corner flag tops (z = 1.5)
    PitchLandmark("near_left_corner_flag_top",  (0.0,        0.0,        _FLAG_HEIGHT)),
    PitchLandmark("near_right_corner_flag_top", (_PITCH_LEN, 0.0,        _FLAG_HEIGHT)),
    PitchLandmark("far_left_corner_flag_top",   (0.0,        _PITCH_WID, _FLAG_HEIGHT)),
    PitchLandmark("far_right_corner_flag_top",  (_PITCH_LEN, _PITCH_WID, _FLAG_HEIGHT)),

    # ── Mow-grid intersections (z = 0) ──────────────────────────────────
    # Centre line (y=34) × painted features. The y=34 mow-stripe boundary
    # passes through both penalty spots and the centre spot, so any of the
    # crossings below are unambiguous reference points the user can pick
    # even when the painted intersection itself is out of frame.
    PitchLandmark("centre_line_left_goal_intersect",   (0.0,                        34.0, 0.0)),
    PitchLandmark("centre_line_right_goal_intersect",  (_PITCH_LEN,                 34.0, 0.0)),
    PitchLandmark("centre_line_left_18yd_intersect",   (16.5,                       34.0, 0.0)),
    PitchLandmark("centre_line_right_18yd_intersect",  (_PITCH_LEN - 16.5,          34.0, 0.0)),
    PitchLandmark("centre_line_left_D_intersect",      (_D_CENTRE_X,                34.0, 0.0)),
    PitchLandmark("centre_line_right_D_intersect",     (_PITCH_LEN - _D_CENTRE_X,   34.0, 0.0)),
    PitchLandmark("centre_line_centre_circle_left",    (52.5 - _CC_HALF_X,          34.0, 0.0)),
    PitchLandmark("centre_line_centre_circle_right",   (52.5 + _CC_HALF_X,          34.0, 0.0)),

    # Near mow line (y=17.5, three 5.5 m squares below the centre line)
    # × goal line / 18-yd front / halfway. Useful when the near touchline
    # is out of shot but the mow boundary three squares "below" the
    # penalty spots IS visible.
    PitchLandmark("near_mow_line_left_goal_intersect",   (0.0,                _NEAR_MOW_Y, 0.0)),
    PitchLandmark("near_mow_line_right_goal_intersect",  (_PITCH_LEN,         _NEAR_MOW_Y, 0.0)),
    PitchLandmark("near_mow_line_left_18yd_intersect",   (16.5,               _NEAR_MOW_Y, 0.0)),
    PitchLandmark("near_mow_line_right_18yd_intersect",  (_PITCH_LEN - 16.5,  _NEAR_MOW_Y, 0.0)),
    PitchLandmark("near_mow_line_halfway_intersect",     (52.5,               _NEAR_MOW_Y, 0.0)),

    # Far mow line (y=50.5, three squares above the centre line) — mirror
    # of the near mow line.
    PitchLandmark("far_mow_line_left_goal_intersect",    (0.0,                _FAR_MOW_Y, 0.0)),
    PitchLandmark("far_mow_line_right_goal_intersect",   (_PITCH_LEN,         _FAR_MOW_Y, 0.0)),
    PitchLandmark("far_mow_line_left_18yd_intersect",    (16.5,               _FAR_MOW_Y, 0.0)),
    PitchLandmark("far_mow_line_right_18yd_intersect",   (_PITCH_LEN - 16.5,  _FAR_MOW_Y, 0.0)),
    PitchLandmark("far_mow_line_halfway_intersect",      (52.5,               _FAR_MOW_Y, 0.0)),
)

LANDMARK_CATALOGUE: dict[str, PitchLandmark] = {lm.name: lm for lm in _LANDMARKS}


def get_landmark(name: str) -> PitchLandmark:
    if name not in LANDMARK_CATALOGUE:
        raise KeyError(f"Unknown pitch landmark: {name!r}")
    return LANDMARK_CATALOGUE[name]


def has_non_coplanar(landmarks) -> bool:
    """True iff the landmark set spans more than one z-plane.

    Accepts any iterable of objects with a ``world_xyz`` attribute (or
    3-element sequence). This lets the same helper guard against
    coplanar inputs whether the caller passes ``PitchLandmark`` or
    ``LandmarkObservation`` instances.
    """
    z_values: set[float] = set()
    for lm in landmarks:
        xyz = getattr(lm, "world_xyz", lm)
        z_values.add(round(float(xyz[2]), 6))
    return len(z_values) >= 2
