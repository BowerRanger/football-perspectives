"""Canonical FIFA pitch line geometry as 3D polylines.

Each line is a list of (x, y, z) world points in pitch coordinates
(metres) sampled densely enough that linear interpolation gives a
visually-smooth curve when projected through a perspective camera.
The polylines are intended for calibration-debug overlays — project
each polyline through the calibration's (K, R, t) and stroke as a
poly-line on the source frame.

All points lie on z = 0 unless they are part of a goal frame, where
crossbars sit at z = 2.44 m and posts go from z = 0 to z = 2.44 m.
"""

from __future__ import annotations

import numpy as np

from src.utils.pitch import (
    PITCH_LENGTH,
    PITCH_WIDTH,
    _CIRCLE_R,
    _GOAL_HALF,
    _GOAL_HEIGHT,
    _LEFT_D_DY,
    _RIGHT_D_DY,
)

# 18-yard box (penalty area)
_18Y_INSET = 16.5
_18Y_HALF = (54.16 - 13.84) / 2.0
# 6-yard box (goal area)
_6Y_INSET = 5.5
_6Y_HALF = (43.16 - 24.84) / 2.0
# Penalty spot insets
_PEN_X_LEFT = 11.0
_PEN_X_RIGHT = PITCH_LENGTH - _PEN_X_LEFT


def _arc(centre: tuple[float, float], radius: float, t0: float, t1: float, n: int) -> np.ndarray:
    """Sample ``n`` points along an arc of ``radius`` around ``centre``.

    Angles ``t0``..``t1`` are in radians.  Returns ``(n, 3)`` with z=0.
    """
    ts = np.linspace(t0, t1, n)
    xs = centre[0] + radius * np.cos(ts)
    ys = centre[1] + radius * np.sin(ts)
    return np.stack([xs, ys, np.zeros_like(xs)], axis=1)


def _segment(p0: tuple[float, float, float], p1: tuple[float, float, float], n: int = 16) -> np.ndarray:
    """Sample ``n`` points along the segment from ``p0`` to ``p1``."""
    return np.linspace(np.array(p0, dtype=np.float64), np.array(p1, dtype=np.float64), n)


def pitch_polylines() -> list[np.ndarray]:
    """Return all pitch markings as a list of ``(N, 3)`` polylines.

    Polylines (in order):
      0  near touchline
      1  far touchline
      2  left goal line
      3  right goal line
      4  halfway line
      5  centre circle
      6  left 18-yard box (3-segment U)
      7  right 18-yard box (3-segment U)
      8  left 6-yard box (3-segment U)
      9  right 6-yard box (3-segment U)
      10 left penalty arc
      11 right penalty arc
      12 left goal frame (post → bar → post)
      13 right goal frame (post → bar → post)
    """
    lines: list[np.ndarray] = []

    # Touchlines
    lines.append(_segment((0.0, 0.0, 0.0), (PITCH_LENGTH, 0.0, 0.0), n=64))
    lines.append(_segment((0.0, PITCH_WIDTH, 0.0), (PITCH_LENGTH, PITCH_WIDTH, 0.0), n=64))
    # Goal lines
    lines.append(_segment((0.0, 0.0, 0.0), (0.0, PITCH_WIDTH, 0.0), n=48))
    lines.append(_segment((PITCH_LENGTH, 0.0, 0.0), (PITCH_LENGTH, PITCH_WIDTH, 0.0), n=48))
    # Halfway line
    lines.append(
        _segment((PITCH_LENGTH / 2, 0.0, 0.0), (PITCH_LENGTH / 2, PITCH_WIDTH, 0.0), n=48)
    )
    # Centre circle
    lines.append(_arc((PITCH_LENGTH / 2, PITCH_WIDTH / 2), _CIRCLE_R, 0.0, 2 * np.pi, n=72))

    # Left 18-yard box: from (0, 13.84) → (16.5, 13.84) → (16.5, 54.16) → (0, 54.16)
    lines.append(
        np.concatenate([
            _segment((0.0, 13.84, 0.0), (_18Y_INSET, 13.84, 0.0), n=24),
            _segment((_18Y_INSET, 13.84, 0.0), (_18Y_INSET, 54.16, 0.0), n=24),
            _segment((_18Y_INSET, 54.16, 0.0), (0.0, 54.16, 0.0), n=24),
        ])
    )
    # Right 18-yard box (mirror)
    lines.append(
        np.concatenate([
            _segment((PITCH_LENGTH, 13.84, 0.0), (PITCH_LENGTH - _18Y_INSET, 13.84, 0.0), n=24),
            _segment((PITCH_LENGTH - _18Y_INSET, 13.84, 0.0), (PITCH_LENGTH - _18Y_INSET, 54.16, 0.0), n=24),
            _segment((PITCH_LENGTH - _18Y_INSET, 54.16, 0.0), (PITCH_LENGTH, 54.16, 0.0), n=24),
        ])
    )
    # Left 6-yard box: (0, 24.84) → (5.5, 24.84) → (5.5, 43.16) → (0, 43.16)
    lines.append(
        np.concatenate([
            _segment((0.0, 24.84, 0.0), (_6Y_INSET, 24.84, 0.0), n=12),
            _segment((_6Y_INSET, 24.84, 0.0), (_6Y_INSET, 43.16, 0.0), n=12),
            _segment((_6Y_INSET, 43.16, 0.0), (0.0, 43.16, 0.0), n=12),
        ])
    )
    # Right 6-yard box
    lines.append(
        np.concatenate([
            _segment((PITCH_LENGTH, 24.84, 0.0), (PITCH_LENGTH - _6Y_INSET, 24.84, 0.0), n=12),
            _segment((PITCH_LENGTH - _6Y_INSET, 24.84, 0.0), (PITCH_LENGTH - _6Y_INSET, 43.16, 0.0), n=12),
            _segment((PITCH_LENGTH - _6Y_INSET, 43.16, 0.0), (PITCH_LENGTH, 43.16, 0.0), n=12),
        ])
    )

    # Penalty arcs — only the portion outside the 18-yard box.
    # Left arc: centre at (11, 34), radius 9.15, the visible portion goes
    # from y = 34-_LEFT_D_DY (top of box edge) up around the front to
    # y = 34+_LEFT_D_DY.  Compute the angle range so the endpoints sit
    # exactly on the 18-yard box edge x = _18Y_INSET.
    dx_left = _18Y_INSET - _PEN_X_LEFT  # 5.5
    angle_at_box = np.arccos(dx_left / _CIRCLE_R)  # radians from +x axis
    lines.append(
        _arc((_PEN_X_LEFT, PITCH_WIDTH / 2), _CIRCLE_R, -angle_at_box, angle_at_box, n=24),
    )
    # Right penalty arc — mirror around x = 105.  The visible arc opens
    # toward the centre (negative x direction from the spot), so sweep
    # from π - angle_at_box round to π + angle_at_box.
    lines.append(
        _arc(
            (_PEN_X_RIGHT, PITCH_WIDTH / 2),
            _CIRCLE_R,
            np.pi - angle_at_box,
            np.pi + angle_at_box,
            n=24,
        )
    )

    # Left goal frame: near-side post (z 0→2.44) → crossbar (post→post) → far-side post
    lines.append(
        np.concatenate([
            _segment((0.0, PITCH_WIDTH / 2 - _GOAL_HALF, 0.0),
                     (0.0, PITCH_WIDTH / 2 - _GOAL_HALF, _GOAL_HEIGHT), n=8),
            _segment((0.0, PITCH_WIDTH / 2 - _GOAL_HALF, _GOAL_HEIGHT),
                     (0.0, PITCH_WIDTH / 2 + _GOAL_HALF, _GOAL_HEIGHT), n=12),
            _segment((0.0, PITCH_WIDTH / 2 + _GOAL_HALF, _GOAL_HEIGHT),
                     (0.0, PITCH_WIDTH / 2 + _GOAL_HALF, 0.0), n=8),
        ])
    )
    # Right goal frame
    lines.append(
        np.concatenate([
            _segment((PITCH_LENGTH, PITCH_WIDTH / 2 - _GOAL_HALF, 0.0),
                     (PITCH_LENGTH, PITCH_WIDTH / 2 - _GOAL_HALF, _GOAL_HEIGHT), n=8),
            _segment((PITCH_LENGTH, PITCH_WIDTH / 2 - _GOAL_HALF, _GOAL_HEIGHT),
                     (PITCH_LENGTH, PITCH_WIDTH / 2 + _GOAL_HALF, _GOAL_HEIGHT), n=12),
            _segment((PITCH_LENGTH, PITCH_WIDTH / 2 + _GOAL_HALF, _GOAL_HEIGHT),
                     (PITCH_LENGTH, PITCH_WIDTH / 2 + _GOAL_HALF, 0.0), n=8),
        ])
    )

    return lines
