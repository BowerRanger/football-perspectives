"""FIFA-standard pitch line catalogue for manual line-correspondence
annotation in the camera-calibration anchor editor.

Each entry has either a fully-known ``world_segment`` (two endpoints in
pitch coordinates) OR a ``world_direction`` only (a 3D unit-vector
direction the line is parallel to, with unknown world position):

- **Position-known segments**: touchlines, goal lines, halfway, 18yd / 6yd
  box edges, goal-frame posts and crossbars, perimeter LED-board edges.
- **Direction-only ("vanishing-point") lines**: ``vertical_separator`` for
  any vertical seam between adjacent LED-board panels — direction (0,0,1).
  The user clicks two points along ANY visible vertical seam; the solver
  uses it to constrain the world-z vanishing point. Critical when goal
  posts and corner flag tops aren't visible (most thin pre-action shots).

Used by:
- ``src/web/server.py``'s ``GET /pitch_lines`` endpoint (palette feed).
- ``src/utils/anchor_solver.py``'s joint-BA residuals (the user's drawn
  ``image_segment`` should lie on the projection of the corresponding
  ``world_segment``).

Coordinate system matches ``src/utils/pitch_landmarks.py``: x along the
nearside touchline (0..105), y from near (0) to far (68), z up.

Advertising-board dimensions are *nominal* values (Premier League–ish
defaults: 1.0 m board height, 2 m offset from touchline, 4 m offset from
goal line). Residuals on ad-board-anchored frames may be looser than fully
known geometry if the stadium uses different dimensions; per-stadium
overrides are tracked in ``docs/FEATURE_IDEAS.md``.
"""

from __future__ import annotations


_PITCH_LEN = 105.0
_PITCH_WID = 68.0
_GOAL_HEIGHT = 2.44
_GOAL_HALF_W = 7.32 / 2  # 3.66
# Goal-post centres: y = 34 ± _GOAL_HALF_W = 30.34 / 37.66
_GP_NEAR = 34.0 - _GOAL_HALF_W
_GP_FAR = 34.0 + _GOAL_HALF_W

# 18-yard box: front edge at 16.5 m from goal line, sides at 34 ± 20.16 m
_18_DEPTH = 16.5
_18_HALF = 20.16
_18_NEAR = 34.0 - _18_HALF  # 13.84
_18_FAR = 34.0 + _18_HALF   # 54.16

# 6-yard box: front edge at 5.5 m from goal line, sides at 34 ± 9.16 m
_6_DEPTH = 5.5
_6_HALF = 9.16
_6_NEAR = 34.0 - _6_HALF    # 24.84
_6_FAR = 34.0 + _6_HALF     # 43.16

# Advertising boards (perimeter LED ribbons) — nominal Premier League–ish
# defaults. See module docstring for caveats and the future per-stadium
# override note in docs/FEATURE_IDEAS.md.
_AD_BOARD_HEIGHT = 1.0          # m — typical LED ribbon top edge
_AD_TOUCHLINE_OFFSET = 2.0      # m — boards beyond near (y=-2) / far (y=70)
_AD_GOALLINE_OFFSET = 4.0       # m — boards beyond left (x=-4) / right (x=109)
_AD_NEAR_Y = -_AD_TOUCHLINE_OFFSET                  # -2.0
_AD_FAR_Y = _PITCH_WID + _AD_TOUCHLINE_OFFSET       # 70.0
_AD_LEFT_X = -_AD_GOALLINE_OFFSET                   # -4.0
_AD_RIGHT_X = _PITCH_LEN + _AD_GOALLINE_OFFSET      # 109.0


# Each value is ((x1, y1, z1), (x2, y2, z2)) — endpoints of the world line.
LINE_CATALOGUE: dict[
    str, tuple[tuple[float, float, float], tuple[float, float, float]]
] = {
    # Pitch perimeter
    "near_touchline":         ((0.0,        0.0,    0.0), (_PITCH_LEN, 0.0,        0.0)),
    "far_touchline":          ((0.0,        _PITCH_WID, 0.0), (_PITCH_LEN, _PITCH_WID, 0.0)),
    "left_goal_line":         ((0.0,        0.0,    0.0), (0.0,        _PITCH_WID, 0.0)),
    "right_goal_line":        ((_PITCH_LEN, 0.0,    0.0), (_PITCH_LEN, _PITCH_WID, 0.0)),
    "halfway_line":           ((52.5,       0.0,    0.0), (52.5,       _PITCH_WID, 0.0)),

    # Left penalty area (18-yard box)
    "left_18yd_front":        ((_18_DEPTH,  _18_NEAR, 0.0), (_18_DEPTH,  _18_FAR, 0.0)),
    "left_18yd_top":          ((0.0,        _18_NEAR, 0.0), (_18_DEPTH,  _18_NEAR, 0.0)),
    "left_18yd_bottom":       ((0.0,        _18_FAR,  0.0), (_18_DEPTH,  _18_FAR,  0.0)),

    # Right penalty area (18-yard box)
    "right_18yd_front":       ((_PITCH_LEN - _18_DEPTH, _18_NEAR, 0.0), (_PITCH_LEN - _18_DEPTH, _18_FAR, 0.0)),
    "right_18yd_top":         ((_PITCH_LEN, _18_NEAR, 0.0), (_PITCH_LEN - _18_DEPTH, _18_NEAR, 0.0)),
    "right_18yd_bottom":      ((_PITCH_LEN, _18_FAR,  0.0), (_PITCH_LEN - _18_DEPTH, _18_FAR,  0.0)),

    # Left goal area (6-yard box)
    "left_6yd_front":         ((_6_DEPTH,   _6_NEAR, 0.0), (_6_DEPTH,   _6_FAR, 0.0)),
    "left_6yd_top":           ((0.0,        _6_NEAR, 0.0), (_6_DEPTH,   _6_NEAR, 0.0)),
    "left_6yd_bottom":        ((0.0,        _6_FAR,  0.0), (_6_DEPTH,   _6_FAR,  0.0)),

    # Right goal area (6-yard box)
    "right_6yd_front":        ((_PITCH_LEN - _6_DEPTH, _6_NEAR, 0.0), (_PITCH_LEN - _6_DEPTH, _6_FAR, 0.0)),
    "right_6yd_top":          ((_PITCH_LEN, _6_NEAR, 0.0), (_PITCH_LEN - _6_DEPTH, _6_NEAR, 0.0)),
    "right_6yd_bottom":       ((_PITCH_LEN, _6_FAR,  0.0), (_PITCH_LEN - _6_DEPTH, _6_FAR,  0.0)),

    # Goal frame — vertical posts (great for breaking coplanarity)
    "left_goal_left_post":    ((0.0,        _GP_NEAR, 0.0), (0.0,        _GP_NEAR, _GOAL_HEIGHT)),
    "left_goal_right_post":   ((0.0,        _GP_FAR,  0.0), (0.0,        _GP_FAR,  _GOAL_HEIGHT)),
    "right_goal_left_post":   ((_PITCH_LEN, _GP_NEAR, 0.0), (_PITCH_LEN, _GP_NEAR, _GOAL_HEIGHT)),
    "right_goal_right_post":  ((_PITCH_LEN, _GP_FAR,  0.0), (_PITCH_LEN, _GP_FAR,  _GOAL_HEIGHT)),

    # Goal frame — horizontal crossbars (z = 2.44)
    "left_goal_crossbar":     ((0.0,        _GP_NEAR, _GOAL_HEIGHT), (0.0,        _GP_FAR, _GOAL_HEIGHT)),
    "right_goal_crossbar":    ((_PITCH_LEN, _GP_NEAR, _GOAL_HEIGHT), (_PITCH_LEN, _GP_FAR, _GOAL_HEIGHT)),

    # Advertising boards (perimeter LED ribbons) — give a vertical depth cue
    # (z=0 base + z=1 top, parallel) on frames where goal frames / corner
    # flags aren't clearly visible. Endpoints span the typical visible
    # extent; the solver only uses the line direction so endpoints don't
    # need to match the user's clicks exactly.
    # Near touchline-parallel boards
    "near_advertising_board_base": ((0.0,        _AD_NEAR_Y, 0.0),
                                    (_PITCH_LEN, _AD_NEAR_Y, 0.0)),
    "near_advertising_board_top":  ((0.0,        _AD_NEAR_Y, _AD_BOARD_HEIGHT),
                                    (_PITCH_LEN, _AD_NEAR_Y, _AD_BOARD_HEIGHT)),
    # Far touchline-parallel boards
    "far_advertising_board_base":  ((0.0,        _AD_FAR_Y, 0.0),
                                    (_PITCH_LEN, _AD_FAR_Y, 0.0)),
    "far_advertising_board_top":   ((0.0,        _AD_FAR_Y, _AD_BOARD_HEIGHT),
                                    (_PITCH_LEN, _AD_FAR_Y, _AD_BOARD_HEIGHT)),
    # Behind-left-goal boards (parallel to left goal line; spans 18-yd width)
    "behind_left_goal_ad_board_base": ((_AD_LEFT_X, _18_NEAR, 0.0),
                                       (_AD_LEFT_X, _18_FAR,  0.0)),
    "behind_left_goal_ad_board_top":  ((_AD_LEFT_X, _18_NEAR, _AD_BOARD_HEIGHT),
                                       (_AD_LEFT_X, _18_FAR,  _AD_BOARD_HEIGHT)),
    # Behind-right-goal boards
    "behind_right_goal_ad_board_base": ((_AD_RIGHT_X, _18_NEAR, 0.0),
                                        (_AD_RIGHT_X, _18_FAR,  0.0)),
    "behind_right_goal_ad_board_top":  ((_AD_RIGHT_X, _18_NEAR, _AD_BOARD_HEIGHT),
                                        (_AD_RIGHT_X, _18_FAR,  _AD_BOARD_HEIGHT)),
}


def get_line(name: str) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    if name not in LINE_CATALOGUE:
        raise KeyError(f"Unknown pitch line: {name!r}")
    return LINE_CATALOGUE[name]


# Direction-only lines (vanishing-point constraints). The user's clicked
# image segment must point at the projection of the world direction —
# world position of the line is unknown and unconstrained. Useful when the
# user can spot a *direction* (e.g. a vertical seam between LED-board
# panels) but not its absolute world location.
VANISHING_LINE_CATALOGUE: dict[str, tuple[float, float, float]] = {
    # The vertical (z-axis) seam between adjacent LED-board panels — visible
    # in nearly any frame where the boards are. Adds VP3 (world +z) to
    # frames that don't show goal posts or corner flags.
    "vertical_separator":          (0.0, 0.0, 1.0),
    # Optional: mowing-stripe lines. Mowing stripes alternate parallel to
    # the touchlines (world +x) — direction (1,0,0). Often visible when
    # turf is freshly cut and the user can't make out painted lines.
    "mowing_stripe_along_touchline": (1.0, 0.0, 0.0),
    # Mowing stripes perpendicular to touchlines (world +y) — direction
    # (0,1,0). Same idea, perpendicular pattern.
    "mowing_stripe_along_goalline":  (0.0, 1.0, 0.0),
}


def get_vanishing_line(name: str) -> tuple[float, float, float]:
    if name not in VANISHING_LINE_CATALOGUE:
        raise KeyError(f"Unknown vanishing line: {name!r}")
    return VANISHING_LINE_CATALOGUE[name]
