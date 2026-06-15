"""Robust direction-change segmentation (Phase B2/B3).

Replaces the fragile local velocity-break detector (``_raw_break_candidates``)
with a *global* piecewise-linear fit of the (sparse, noisy) pixel track. The
ball's path is approximated by straight segments; a sharp **corner** between
two segments is a direction change — a touch, bounce or impact candidate. A
smooth flight parabola produces only gradual bends (no single corner clears
the angle gate), so it isn't over-segmented; a kick/bounce produces a sharp
corner that does.

Deterministic bottom-up merging (Keogh): start from fine segments, repeatedly
merge the lowest-residual adjacent pair until the cheapest merge exceeds a
residual budget. No randomness — same input, same breakpoints.

Output is the existing ``_Break`` shape so the downstream classify cascade
(touch/bounce/goal) and NMS consume it unchanged. See
docs/superpowers/specs/2026-06-15-ball-detection-direction-changes-design.md §5.2.
"""

from __future__ import annotations

import math

import numpy as np

from src.utils.ball_auto_events import AutoEventCfg, _Break


def _fit_line(pts: list[tuple[float, float, float]]) -> tuple[np.ndarray, float]:
    """Fit x(t), y(t) linear over points ``(t, x, y)``; return
    ``(velocity=(vx, vy) px/frame, max_residual_px)``."""
    t = np.array([p[0] for p in pts], float)
    x = np.array([p[1] for p in pts], float)
    y = np.array([p[2] for p in pts], float)
    if len(pts) == 1:
        return np.zeros(2), 0.0
    A = np.vstack([t, np.ones_like(t)]).T
    (ax, bx), _, _, _ = np.linalg.lstsq(A, x, rcond=None)
    (ay, by), _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    rx = x - (ax * t + bx)
    ry = y - (ay * t + by)
    resid = float(np.max(np.hypot(rx, ry))) if len(pts) > 2 else 0.0
    return np.array([ax, ay]), resid


def _breaks_from_segments(
    pts: list[tuple[float, float, float]],
    bounds: list[tuple[int, int]],
    cfg: AutoEventCfg,
) -> list[_Break]:
    breaks: list[_Break] = []
    for j in range(len(bounds) - 1):
        v_b, _ = _fit_line(pts[bounds[j][0]: bounds[j][1] + 1])
        v_a, _ = _fit_line(pts[bounds[j + 1][0]: bounds[j + 1][1] + 1])
        sb = float(np.linalg.norm(v_b))
        sa = float(np.linalg.norm(v_a))
        dspeed = abs(sa - sb)
        if min(sb, sa) >= cfg.min_break_speed_px:
            cosang = float(np.dot(v_b, v_a) / (sb * sa))
            dir_change = math.degrees(math.acos(max(-1.0, min(1.0, cosang))))
        else:
            dir_change = 0.0
        if dir_change < cfg.min_direction_change_deg and dspeed < cfg.min_speed_change_px:
            continue
        e = bounds[j][1]
        corner = int(round((pts[e][0] + pts[e + 1][0]) / 2.0))
        strength = min(
            1.0,
            0.5 * (dir_change / 90.0)
            + 0.5 * (dspeed / (3.0 * cfg.min_speed_change_px)),
        )
        breaks.append(_Break(
            frame=corner, strength=strength, dir_change_deg=dir_change,
            dspeed_px=dspeed, speed_before=sb, speed_after=sa,
            vy_before=float(v_b[1]), vy_after=float(v_a[1]),
        ))
    return breaks


def segment_track(
    uvs: dict[int, tuple[float, float]],
    *,
    cfg: AutoEventCfg | None = None,
    max_residual_px: float = 6.0,
) -> list[_Break]:
    """Direction-change ``_Break``s from a sparse pixel track ``{frame: (u, v)}``."""
    cfg = cfg or AutoEventCfg()
    pts = [(float(f), float(uv[0]), float(uv[1]))
           for f, uv in sorted(uvs.items()) if uv is not None]
    if len(pts) < 4:
        return []

    # Initial fine segments: adjacent pairs of point indices.
    bounds: list[tuple[int, int]] = []
    i = 0
    while i < len(pts) - 1:
        bounds.append((i, i + 1))
        i += 2
    if bounds[-1][1] != len(pts) - 1:
        bounds[-1] = (bounds[-1][0], len(pts) - 1)

    # Bottom-up merge by lowest residual until the cheapest exceeds the budget.
    while len(bounds) > 1:
        costs = [
            _fit_line(pts[bounds[k][0]: bounds[k + 1][1] + 1])[1]
            for k in range(len(bounds) - 1)
        ]
        j = int(np.argmin(costs))
        if costs[j] > max_residual_px:
            break
        bounds[j] = (bounds[j][0], bounds[j + 1][1])
        del bounds[j + 1]

    return _breaks_from_segments(pts, bounds, cfg)


__all__ = ["segment_track"]
