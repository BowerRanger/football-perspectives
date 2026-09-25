"""Motion-blur streak event cue: for each real detection, segment the
ball blob in a small crop, PCA it to get streak length + orientation and
an implied image speed, and flag frames where consecutive detections'
orientation changes sharply -- a proxy for a sudden direction change
(kick/header/bounce) between two detector-only frames.

Exposure assumption (documented, not measured per clip): broadcast
cameras commonly run close to a 180-degree shutter, i.e. exposure time is
about half the frame interval. ``implied_speed_px_s`` bakes this in as
``_SHUTTER_FRACTION``; treat implied speeds as directional, not
calibrated, unless a clip's actual shutter angle is known.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np

from src.utils.ball_cue_types import CueEvidence

_SHUTTER_FRACTION = 0.5


@dataclass(frozen=True)
class BlobShape:
    length_px: float
    angle_deg: float  # major-axis orientation in [0, 180) -- no direction sense
    area_px: float
    centroid_px: tuple[float, float]


def analyze_blob(crop_gray: np.ndarray, *, min_area_px: float = 3.0) -> BlobShape | None:
    """Otsu-threshold ``crop_gray`` and PCA the largest connected
    component's pixel coordinates to recover streak length + orientation.
    Returns ``None`` when no component clears ``min_area_px`` (empty crop,
    no blob segmented, or too small to trust the PCA)."""
    if crop_gray.size == 0:
        return None
    blurred = cv2.GaussianBlur(crop_gray, (3, 3), 0)
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    n, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return None
    areas = stats[1:, cv2.CC_STAT_AREA]
    best_label = int(np.argmax(areas)) + 1
    area = float(areas[best_label - 1])
    if area < min_area_px:
        return None
    ys, xs = np.nonzero(labels == best_label)
    pts = np.column_stack([xs, ys]).astype(np.float64)
    mean = pts.mean(axis=0)
    centered = pts - mean
    cov = (centered.T @ centered) / max(1, len(pts) - 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    major = eigvecs[:, order[0]]
    proj = centered @ major
    length = float(proj.max() - proj.min())
    angle = float(np.degrees(np.arctan2(major[1], major[0])) % 180.0)
    return BlobShape(length_px=length, angle_deg=angle, area_px=area,
                      centroid_px=(float(mean[0]), float(mean[1])))


def implied_speed_px_s(
    length_px: float, fps: float, *, shutter_fraction: float = _SHUTTER_FRACTION,
) -> float:
    """Streak length / assumed exposure time -> implied image speed in
    px/s. Exposure time = ``shutter_fraction / fps``."""
    exposure_s = shutter_fraction / fps
    if exposure_s <= 0:
        return 0.0
    return length_px / exposure_s


def angle_delta_deg(a: float, b: float) -> float:
    """Smallest difference between two axis orientations, mod 180 (a
    streak's major axis has no direction sense: 5deg and 175deg are
    10deg apart, not 170)."""
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


def crop_around(
    frame_gray: np.ndarray, uv: tuple[float, float], radius: int,
) -> np.ndarray:
    h, w = frame_gray.shape[:2]
    cu, cv_ = int(round(uv[0])), int(round(uv[1]))
    x0, x1 = max(0, cu - radius), min(w, cu + radius + 1)
    y0, y1 = max(0, cv_ - radius), min(h, cv_ + radius + 1)
    return frame_gray[y0:y1, x0:x1]


def compute_blur_cues(
    detections: list[tuple[int, tuple[float, float]]],
    frame_lookup: Callable[[int], np.ndarray | None],
    *,
    fps: float,
    crop_radius: int = 20,
    min_streak_px: float = 4.0,
    angle_change_deg: float = 25.0,
    max_frame_gap: int = 3,
) -> list[CueEvidence]:
    """``detections`` is ``(frame, uv)`` pairs (typically the real
    detector's dense observation track). ``frame_lookup(frame)`` returns
    a grayscale frame or ``None`` when unavailable (e.g. not decoded /
    not cached by the caller). Emits a ``CueEvidence`` at the later frame
    of any consecutive streaky-detection pair (gap <= ``max_frame_gap``)
    whose blob orientation changes by >= ``angle_change_deg``.
    """
    shapes: list[tuple[int, BlobShape]] = []
    for frame, uv in sorted(detections, key=lambda d: d[0]):
        gray = frame_lookup(frame)
        if gray is None:
            continue
        crop = crop_around(gray, uv, crop_radius)
        shape = analyze_blob(crop)
        if shape is None or shape.length_px < min_streak_px:
            continue
        shapes.append((frame, shape))

    events: list[CueEvidence] = []
    for (f_prev, s_prev), (f_cur, s_cur) in zip(shapes, shapes[1:]):
        if f_cur - f_prev > max_frame_gap:
            continue
        delta = angle_delta_deg(s_prev.angle_deg, s_cur.angle_deg)
        if delta < angle_change_deg:
            continue
        conf = float(np.clip(delta / 90.0, 0.0, 1.0))
        events.append(CueEvidence(
            frame=f_cur, kind="contact", cue="blur_direction_change", conf=conf,
            xyz=None, uv=s_cur.centroid_px,
        ))
    return events


__all__ = [
    "BlobShape", "analyze_blob", "implied_speed_px_s", "angle_delta_deg",
    "crop_around", "compute_blur_cues",
]
