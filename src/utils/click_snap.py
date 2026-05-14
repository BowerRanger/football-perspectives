"""Sub-pixel click refinement (snap-to-feature) for the anchor editor.

Given a user click on a broadcast frame, search a small window for nearby
painted-line features and refine the click to sub-pixel accuracy.

Two refinement modes:

- ``line_intersection``: search for the two painted lines whose intersection
  this click is meant to mark (e.g. corner of the 18-yard box). Returns the
  intersection point with sub-pixel accuracy.

- ``line_endpoint``: search for the painted line this click is meant to
  mark an endpoint of. Returns the click projected onto the detected line.

A heuristic mode ``auto`` picks intersection if two roughly-perpendicular
lines are found, else single-line projection if only one line is found,
else returns the click unchanged.

All routines work in a small local image patch (default 60×60 px) for
fast inference. Algorithm reuses the bright-ridge convolution / parabola-
peak math from ``line_detector.py``; the difference is that here we don't
have a bootstrap world line — we scan multiple orientations to find
whatever painted features pass through the window.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, NamedTuple

import cv2
import numpy as np

from src.utils.line_detector import _parabolic_subpixel


SnapMode = Literal["line_intersection", "line_endpoint", "auto", "off"]


class SnappedPoint(NamedTuple):
    xy: tuple[float, float]
    snapped: bool
    """``True`` if a feature was found and used; ``False`` if the original
    click was returned unchanged."""
    mode_used: SnapMode
    confidence: float
    """0..1 — fraction of feature search angles that produced consistent
    line detections. Reported to the UI so users can override low-
    confidence snaps."""


@dataclass(frozen=True)
class SnapConfig:
    window_px: int = 60
    """Half-width of the local search window. The user's click must land
    within ``window_px`` of the painted feature for the snap to succeed."""

    n_angles: int = 18
    """How many cross-section orientations to try when scanning for a
    line. 18 angles = 10° steps over 180° — covers any line orientation
    with ≤5° error before subpixel refinement converges."""

    cross_section_half_width: int = 12
    """For each orientation, how many pixels either side of the click to
    sample for the bright-ridge response."""

    min_ridge_response: float = 8.0
    """Minimum bright-ridge response (paint-vs-flank contrast in grey
    levels) for a line to be accepted at a given orientation."""

    angle_smoothing_window: int = 3
    """When picking the best angle, smooth the per-angle response over
    this window so noise doesn't bias the choice."""

    perpendicular_tolerance_deg: float = 25.0
    """Two lines are treated as "perpendicular enough" for intersection
    snapping if their orientation differs by 90° ± this many degrees.
    25° tolerance is loose enough to accept the 6-yard box edges which
    project at oblique angles on broadcast shots."""


def _ridge_response_at_angle(
    gray: np.ndarray,
    centre: tuple[float, float],
    angle_rad: float,
    cfg: SnapConfig,
) -> tuple[float, float] | None:
    """Sample a cross-section perpendicular to ``angle_rad`` through
    ``centre`` and return (ridge_response, sub_pixel_offset) at the peak.

    ``angle_rad`` is the LINE's orientation (its tangent direction). The
    cross-section is taken along the normal ``(-sin, cos)``. Sub-pixel
    offset is signed pixels along that normal from ``centre`` to the
    detected line centreline.

    Returns ``None`` if the cross-section can't be sampled (off-image)
    or no ridge peak rises above ``min_ridge_response``.
    """
    h, w = gray.shape
    normal = np.array([-np.sin(angle_rad), np.cos(angle_rad)])
    n = cfg.cross_section_half_width
    offsets = np.arange(-n, n + 1)
    xs = centre[0] + offsets * normal[0]
    ys = centre[1] + offsets * normal[1]
    valid = (xs >= 0) & (xs < w - 1) & (ys >= 0) & (ys < h - 1)
    if valid.sum() < 11:
        return None
    xi = xs[valid]; yi = ys[valid]
    x0 = np.floor(xi).astype(np.int32); y0 = np.floor(yi).astype(np.int32)
    fx = xi - x0; fy = yi - y0
    g = (gray[y0, x0] * (1 - fx) * (1 - fy)
         + gray[y0, x0 + 1] * fx * (1 - fy)
         + gray[y0 + 1, x0] * (1 - fx) * fy
         + gray[y0 + 1, x0 + 1] * fx * fy).astype(np.float32)
    L = len(g)
    if L < 11:
        return None
    paint_w = 3; flank_w = 8
    kernel = np.zeros(paint_w + 2 * flank_w, dtype=np.float32)
    kernel[flank_w:flank_w + paint_w] = +1.0 / paint_w
    kernel[:flank_w] = -0.5 / flank_w
    kernel[-flank_w:] = -0.5 / flank_w
    resp = np.convolve(g, kernel[::-1], mode="same")
    edge = flank_w + 1
    resp[:edge] = -np.inf
    resp[-edge:] = -np.inf
    peak = int(np.argmax(resp))
    if not np.isfinite(resp[peak]) or resp[peak] < cfg.min_ridge_response:
        return None
    if peak <= 0 or peak >= L - 1:
        return None
    sub = _parabolic_subpixel(resp[peak - 1], resp[peak], resp[peak + 1])
    if not np.isfinite(sub):
        sub = 0.0
    valid_offsets = offsets[valid].astype(np.float64)
    centre_idx = peak + sub
    if centre_idx < 0 or centre_idx > L - 1:
        return None
    base = int(np.floor(centre_idx))
    frac = centre_idx - base
    real_off = (1 - frac) * valid_offsets[base] + frac * valid_offsets[
        min(base + 1, L - 1)
    ]
    return float(resp[peak]), float(real_off)


def _find_lines_through_click(
    gray: np.ndarray,
    click_xy: tuple[float, float],
    cfg: SnapConfig,
) -> list[tuple[float, float, float]]:
    """Find painted lines passing near the click by scanning orientations.

    Returns a list of ``(angle_rad, sub_pixel_offset, ridge_response)``
    tuples — one per accepted orientation, sorted by descending
    response. Adjacent orientations are merged so we report at most one
    line per cluster.
    """
    angles = np.linspace(0.0, np.pi, cfg.n_angles, endpoint=False)
    responses: list[tuple[float, float, float] | None] = []
    for a in angles:
        result = _ridge_response_at_angle(gray, click_xy, a, cfg)
        responses.append((a, result[1], result[0]) if result else None)

    # Smooth angular response (with wrap) to suppress spurious peaks
    resp_only = np.array([r[2] if r else 0.0 for r in responses])
    smoothed = np.zeros_like(resp_only)
    w = cfg.angle_smoothing_window
    for i in range(cfg.n_angles):
        idxs = [(i + k - w // 2) % cfg.n_angles for k in range(w)]
        smoothed[i] = np.mean([resp_only[j] for j in idxs])

    # Find local maxima above min_ridge_response
    accepted: list[tuple[float, float, float]] = []
    for i in range(cfg.n_angles):
        prev_i = (i - 1) % cfg.n_angles
        next_i = (i + 1) % cfg.n_angles
        if smoothed[i] < cfg.min_ridge_response: continue
        if smoothed[i] < smoothed[prev_i] or smoothed[i] < smoothed[next_i]: continue
        if responses[i] is None: continue
        accepted.append(responses[i])

    accepted.sort(key=lambda r: -r[2])
    return accepted


def _line_at_click(
    click_xy: tuple[float, float],
    angle_rad: float,
    sub_offset: float,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return two points on the detected line — used downstream to
    intersect lines or project the click onto a single line."""
    normal = np.array([-np.sin(angle_rad), np.cos(angle_rad)])
    direction = np.array([np.cos(angle_rad), np.sin(angle_rad)])
    pt_on_line = np.array(click_xy) + sub_offset * normal
    a = pt_on_line - 10 * direction
    b = pt_on_line + 10 * direction
    return (float(a[0]), float(a[1])), (float(b[0]), float(b[1]))


def _intersect(
    a1: tuple[float, float], a2: tuple[float, float],
    b1: tuple[float, float], b2: tuple[float, float],
) -> tuple[float, float] | None:
    """Intersection of two infinite lines (a1-a2) and (b1-b2).
    Returns ``None`` if parallel."""
    x1, y1 = a1; x2, y2 = a2; x3, y3 = b1; x4, y4 = b2
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-9:
        return None
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    return float(x1 + t * (x2 - x1)), float(y1 + t * (y2 - y1))


def snap_click(
    frame_bgr: np.ndarray,
    click_xy: tuple[float, float],
    mode: SnapMode = "auto",
    cfg: SnapConfig | None = None,
) -> SnappedPoint:
    """Refine the user's click to sub-pixel accuracy.

    Crops a ``window_px``-sized patch around the click and runs angle-
    scanning bright-ridge detection. Behaviour depends on ``mode`` and
    on how many lines are found:

    | mode               | 0 lines           | 1 line                | ≥2 lines (perp.)    |
    | ------------------ | ----------------- | --------------------- | ------------------- |
    | ``off``            | click unchanged   | click unchanged       | click unchanged     |
    | ``line_endpoint``  | click unchanged   | project on line       | project on best line|
    | ``line_intersection``| click unchanged | click unchanged       | intersection        |
    | ``auto``           | click unchanged   | project on line       | intersection        |

    Returns a :class:`SnappedPoint` with ``snapped=False`` when no
    refinement was applied.
    """
    if cfg is None:
        cfg = SnapConfig()
    if mode == "off":
        return SnappedPoint(click_xy, snapped=False, mode_used="off", confidence=0.0)

    h, w = frame_bgr.shape[:2]
    cx, cy = click_xy
    half = cfg.window_px
    x0 = max(0, int(round(cx - half))); x1 = min(w, int(round(cx + half + 1)))
    y0 = max(0, int(round(cy - half))); y1 = min(h, int(round(cy + half + 1)))
    if x1 - x0 < 11 or y1 - y0 < 11:
        return SnappedPoint(click_xy, snapped=False, mode_used=mode, confidence=0.0)
    crop = frame_bgr[y0:y1, x0:x1]
    gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
    click_local = (cx - x0, cy - y0)
    accepted = _find_lines_through_click(gray_crop, click_local, cfg)
    confidence = min(1.0, len(accepted) / 2.0)
    if not accepted:
        return SnappedPoint(click_xy, snapped=False, mode_used=mode, confidence=0.0)

    if mode == "line_endpoint" or (
        mode == "auto" and len(accepted) == 1
    ):
        a, off, _r = accepted[0]
        line_a, line_b = _line_at_click(click_local, a, off)
        # Project the local click onto the line
        d = np.array(line_b) - np.array(line_a)
        d /= np.linalg.norm(d)
        proj_local = np.array(line_a) + np.dot(np.array(click_local) - np.array(line_a), d) * d
        return SnappedPoint(
            (float(proj_local[0] + x0), float(proj_local[1] + y0)),
            snapped=True, mode_used="line_endpoint", confidence=confidence,
        )

    if mode in ("line_intersection", "auto") and len(accepted) >= 2:
        a1, off1, _ = accepted[0]
        # Find a second line roughly perpendicular to the first
        for a2, off2, _ in accepted[1:]:
            d_ang = abs(((a2 - a1) * 180 / np.pi + 90) % 180 - 90)
            if d_ang >= 90 - cfg.perpendicular_tolerance_deg:
                line1_a, line1_b = _line_at_click(click_local, a1, off1)
                line2_a, line2_b = _line_at_click(click_local, a2, off2)
                inter = _intersect(line1_a, line1_b, line2_a, line2_b)
                if inter is None:
                    return SnappedPoint(click_xy, snapped=False, mode_used=mode, confidence=0.0)
                return SnappedPoint(
                    (float(inter[0] + x0), float(inter[1] + y0)),
                    snapped=True, mode_used="line_intersection", confidence=confidence,
                )

    return SnappedPoint(click_xy, snapped=False, mode_used=mode, confidence=0.0)
