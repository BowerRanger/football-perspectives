"""Detect long straight lines on the pitch surface for calibration cues.

The broadcast camera sees three categories of useful straight lines:

1. **Painted pitch markings** — touchlines, halfway, penalty boxes,
   etc.  PnLCalib already detects these via its line head.  We don't
   re-detect them here.

2. **Mowing stripes** — alternating dark/light bands of grass left by
   the groundskeeper's mower.  Almost every modern broadcast pitch has
   them.  They are parallel to either the touchlines or the goal lines
   (depends on stadium convention) and provide a *much* denser set of
   parallel lines than the painted markings.

3. **Advertising boards** — physical/LED panels surrounding the pitch
   at z ≈ 0.9 m.  Their top and bottom edges are straight 3D lines
   parallel to the touchlines.

This module detects (2) and (3) and returns enough information for a
calibration refinement step to compute or check the pitch-plane
vanishing points.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DetectedLine:
    """A 2D line segment in image space.

    ``(x1, y1)`` and ``(x2, y2)`` are the segment endpoints in pixels.
    ``length`` is the segment length and ``angle`` is the line
    orientation in radians (0..π, measured from +x axis).
    """

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def length(self) -> float:
        return float(np.hypot(self.x2 - self.x1, self.y2 - self.y1))

    @property
    def angle(self) -> float:
        a = float(np.arctan2(self.y2 - self.y1, self.x2 - self.x1))
        # Map to [0, pi)
        if a < 0:
            a += np.pi
        if a >= np.pi:
            a -= np.pi
        return a

    def midpoint(self) -> np.ndarray:
        return np.array([(self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0],
                        dtype=np.float64)

    def direction(self) -> np.ndarray:
        d = np.array([self.x2 - self.x1, self.y2 - self.y1], dtype=np.float64)
        n = np.linalg.norm(d)
        return d / n if n > 1e-9 else d


def _pitch_mask(frame_bgr: np.ndarray) -> np.ndarray:
    """Boolean mask of pitch (green) pixels.

    Uses a generous HSV green range that catches both healthy grass
    and the darker mowing stripes.  False positives in the crowd are
    typically eliminated by morphological filtering downstream.
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    mask = (h >= 30) & (h <= 90) & (s >= 30) & (v >= 30)
    return mask.astype(np.uint8) * 255


def _largest_pitch_region(mask: np.ndarray) -> np.ndarray:
    """Keep only the largest connected green blob — drops crowd false positives."""
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n_labels <= 1:
        return mask
    # label 0 is background — pick the largest non-background label
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest = int(np.argmax(areas)) + 1
    out = np.where(labels == largest, 255, 0).astype(np.uint8)
    # Smooth the boundary so the bottom rim of the field doesn't get jagged
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    return out


def detect_pitch_lines(
    frame_bgr: np.ndarray,
    *,
    min_length_frac: float = 0.1,
    max_lines: int = 200,
) -> tuple[list[DetectedLine], np.ndarray]:
    """Return long straight lines visible on the pitch surface.

    Args:
        frame_bgr: source frame (H, W, 3) BGR.
        min_length_frac: minimum line length as a fraction of frame
            diagonal — short detections get dropped.
        max_lines: cap on the returned line count (longest first).

    Returns:
        ``(lines, pitch_mask)``.  ``lines`` is a list of
        :class:`DetectedLine`, sorted by length descending.
        ``pitch_mask`` is the binary mask used to restrict detection
        — useful for downstream board detection (the off-pitch
        region above the mask is where boards live).
    """
    h, w = frame_bgr.shape[:2]
    diag = float(np.hypot(w, h))
    min_len = min_length_frac * diag

    mask = _largest_pitch_region(_pitch_mask(frame_bgr))
    # Restrict edge detection to the pitch interior only
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    pitch_only = cv2.bitwise_and(gray, gray, mask=mask)
    edges = cv2.Canny(pitch_only, 40, 120)

    raw = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 360,
        threshold=80,
        minLineLength=int(min_len),
        maxLineGap=int(min_len * 0.4),
    )
    if raw is None:
        return [], mask

    lines: list[DetectedLine] = []
    for seg in raw[:, 0, :]:
        x1, y1, x2, y2 = (float(v) for v in seg)
        lines.append(DetectedLine(x1, y1, x2, y2))
    lines.sort(key=lambda ln: -ln.length)
    return lines[:max_lines], mask


def detect_board_lines(
    frame_bgr: np.ndarray,
    pitch_mask: np.ndarray,
    *,
    band_height_frac: float = 0.15,
    min_length_frac: float = 0.15,
    max_lines: int = 30,
) -> list[DetectedLine]:
    """Detect long horizontal-ish lines just above the pitch.

    Advertising boards form a strip of bright/saturated content right
    above the painted touchlines.  Their top and bottom edges are
    straight 3D lines parallel to the touchlines.  In image space they
    project to *near-horizontal* lines because the camera is roughly
    perpendicular to them.

    Heuristic: take a band of height ``band_height_frac * frame_h``
    immediately above the top of the pitch mask, detect long lines
    whose absolute angle is within ±25° of horizontal, return sorted
    by length.

    Returns:
        List of :class:`DetectedLine` in image coordinates.  Empty
        when the pitch mask doesn't expose a clear top edge or no
        horizontal lines are found.
    """
    h, w = frame_bgr.shape[:2]
    diag = float(np.hypot(w, h))
    min_len = min_length_frac * diag

    # Find the top of the pitch per column — the highest y where mask is set
    mask_bool = pitch_mask > 0
    cols_with_pitch = np.where(mask_bool.any(axis=0))[0]
    if cols_with_pitch.size < w * 0.3:
        return []
    top_per_col = np.full(w, -1, dtype=np.int32)
    # vectorised top-pitch-y per column
    for c in cols_with_pitch:
        col = mask_bool[:, c]
        top = int(np.argmax(col))
        if col[top]:
            top_per_col[c] = top
    valid = top_per_col >= 0
    if not np.any(valid):
        return []
    median_top = int(np.median(top_per_col[valid]))
    band_top = max(0, median_top - int(band_height_frac * h))
    band_bot = max(band_top + 1, median_top - 4)  # stop just before the touchline

    # Crop the band and run Canny + HoughLinesP
    band = frame_bgr[band_top:band_bot, :, :]
    if band.shape[0] < 5:
        return []
    gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    raw = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi / 360, threshold=60,
        minLineLength=int(min_len * 0.7), maxLineGap=int(min_len * 0.3),
    )
    if raw is None:
        return []

    out: list[DetectedLine] = []
    for seg in raw[:, 0, :]:
        x1, y1, x2, y2 = (float(v) for v in seg)
        ln = DetectedLine(x1, y1 + band_top, x2, y2 + band_top)
        if ln.length < min_len:
            continue
        # Restrict to near-horizontal (within 25° of horizontal)
        a = ln.angle
        if a > np.pi / 2:
            a = np.pi - a
        if a < np.deg2rad(25):
            out.append(ln)
    out.sort(key=lambda ln: -ln.length)
    return out[:max_lines]
