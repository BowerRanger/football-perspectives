"""Robust centre-circle ellipse detection for wide-field / midfield frames.

The strip-search :func:`circle_detector.detect_circle` samples a ridge per point
around the *projected* circle. On zoomed-out broadcast shots the painted ring is
thin and crossed by the halfway line, so per-point search is noisy (it picks up
the crossing line, mowing stripes and grass highlights) — see the investigation
in docs. This detector instead finds the ring as a whole:

  1. white-on-pitch Canny edges in a band around the projected circle,
  2. reject points lying on long straight lines (the halfway line crossing it),
  3. RANSAC-fit an ellipse (guided to be near the projected one) + refine on
     inliers.

It returns the fitted image ellipse (``(cx, cy)``, ``(major, minor)``, angle in
degrees — OpenCV ``fitEllipse`` convention) plus an inlier-coverage confidence.
The static-camera solve consumes it as an ellipse-distance constraint, which —
unlike straight box lines — pins radial distortion + principal point because the
ring spans the image. (A circle is rotationally symmetric, so the constraint is
"projected catalogue circle lies ON the detected ellipse", not point-to-point.)
"""

from __future__ import annotations

from typing import NamedTuple

import cv2
import numpy as np

from src.utils.camera_projection import project_world_to_image
from src.utils.circle_detector import CENTRE_CIRCLE_CENTRE, CENTRE_CIRCLE_RADIUS
from src.utils.line_detector import DetectorConfig, _prepare_frame

Ellipse = tuple[tuple[float, float], tuple[float, float], float]


class EllipseDetection(NamedTuple):
    ellipse: Ellipse           # ((cx, cy), (major_axis, minor_axis), angle_deg)
    confidence: float          # RANSAC inliers / band edge points
    n_inliers: int


def _ellipse_distance(pts: np.ndarray, e: Ellipse) -> np.ndarray:
    """Approx algebraic point-to-ellipse distance in pixels (|r-1| * min_semi)."""
    (ecx, ecy), (major, minor), ang = e
    th = np.radians(ang)
    ct, st = np.cos(th), np.sin(th)
    dx = pts[:, 0] - ecx
    dy = pts[:, 1] - ecy
    xr = (dx * ct + dy * st) / max(major / 2, 1e-6)
    yr = (-dx * st + dy * ct) / max(minor / 2, 1e-6)
    return np.abs(np.hypot(xr, yr) - 1.0) * (min(major, minor) / 2.0)


def _ransac_ellipse(
    P: np.ndarray, expected: Ellipse, *, iters: int, tol: float, seed: int,
) -> Ellipse | None:
    rng = np.random.RandomState(seed)
    (ecx, ecy), (eMA, ema), _ = expected
    best_mask: np.ndarray | None = None
    best_n = 0
    for _ in range(iters):
        idx = rng.choice(len(P), 6, replace=False)
        try:
            e = cv2.fitEllipse(P[idx].astype(np.float32))
        except cv2.error:
            continue
        (cx, cy), (MA, ma), _ = e
        if np.hypot(cx - ecx, cy - ecy) > 0.3 * max(eMA, ema):
            continue
        if not (0.5 < max(MA, ma) / max(max(eMA, ema), 1.0) < 1.8) or min(MA, ma) < 2:
            continue
        mask = _ellipse_distance(P, e) < tol
        if mask.sum() > best_n:
            best_n, best_mask = int(mask.sum()), mask
    if best_mask is None or best_n < 12:
        return None
    e = cv2.fitEllipse(P[best_mask].astype(np.float32))
    for _ in range(3):
        mask = _ellipse_distance(P, e) < tol
        if mask.sum() < 12:
            break
        e = cv2.fitEllipse(P[mask].astype(np.float32))
    return e


def detect_circle_ellipse(
    frame_bgr: np.ndarray,
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    distortion: tuple[float, float],
    cfg: DetectorConfig,
    *,
    centre: tuple[float, float, float] = CENTRE_CIRCLE_CENTRE,
    radius: float = CENTRE_CIRCLE_RADIUS,
    n_samples: int = 180,
    band_px: float = 30.0,
    ransac_iters: int = 300,
    ransac_tol_px: float = 3.0,
    seed: int = 0,
) -> EllipseDetection | None:
    """Detect the painted centre circle as a clean image ellipse near where it
    projects under ``(K, R, t)``. Returns ``None`` when too little of the ring
    is in view or no reliable ellipse is found.
    """
    cx, cy, cz = centre
    th = np.linspace(0.0, 2.0 * np.pi, n_samples, endpoint=False)
    world = np.stack(
        [cx + radius * np.cos(th), cy + radius * np.sin(th),
         np.full(n_samples, cz)], axis=1)
    in_front = world @ np.asarray(R)[2] + np.asarray(t)[2] > 0.1
    if in_front.sum() < 20:
        return None
    proj = project_world_to_image(K, R, t, distortion, world[in_front])
    h, w = frame_bgr.shape[:2]
    proj = proj[np.isfinite(proj).all(axis=1)]
    proj = proj[(proj[:, 0] >= 0) & (proj[:, 0] < w)
                & (proj[:, 1] >= 0) & (proj[:, 1] < h)]
    if len(proj) < 20:
        return None
    expected = cv2.fitEllipse(proj.astype(np.float32))

    pad = band_px + 8
    bx0, by0 = max(0, int(proj[:, 0].min() - pad)), max(0, int(proj[:, 1].min() - pad))
    bx1, by1 = min(w, int(proj[:, 0].max() + pad)), min(h, int(proj[:, 1].max() + pad))
    gray, green = _prepare_frame(frame_bgr, cfg)
    pitch = cv2.dilate(green, np.ones((21, 21), np.uint8))
    edges = cv2.Canny(gray.astype(np.uint8), 40, 120)
    sub = edges[by0:by1, bx0:bx1]
    subp = pitch[by0:by1, bx0:bx1]
    ys, xs = np.where((sub > 0) & (subp > 0))
    if len(xs) < 20:
        return None
    P = np.c_[xs + bx0, ys + by0].astype(np.float64)
    n_band = (_ellipse_distance(P, expected) < band_px)
    P = P[n_band]
    if len(P) < 20:
        return None

    # Reject the halfway line (and any long straight line) crossing the ring.
    mask = np.zeros((by1 - by0, bx1 - bx0), np.uint8)
    mask[(P[:, 1] - by0).astype(int).clip(0, by1 - by0 - 1),
         (P[:, 0] - bx0).astype(int).clip(0, bx1 - bx0 - 1)] = 255
    hlines = cv2.HoughLinesP(mask, 1, np.pi / 180, 40, minLineLength=60, maxLineGap=10)
    if hlines is not None:
        keep = np.ones(len(P), bool)
        for ln in hlines[:, 0]:
            a = np.array([ln[0] + bx0, ln[1] + by0], float)
            b = np.array([ln[2] + bx0, ln[3] + by0], float)
            d = b - a
            L = np.linalg.norm(d)
            if L < 1:
                continue
            nrm = np.array([-d[1], d[0]]) / L
            perp = np.abs((P - a) @ nrm)
            along = ((P - a) @ d) / L ** 2
            keep &= ~((perp < 5) & (along > -0.05) & (along < 1.05))
        P = P[keep]
    if len(P) < 15:
        return None

    e = _ransac_ellipse(P, expected, iters=ransac_iters, tol=ransac_tol_px, seed=seed)
    if e is None:
        return None
    inl = P[_ellipse_distance(P, e) < ransac_tol_px]
    n_in = len(inl)
    # confidence = angular coverage of the ring (fraction of 36 sectors with an
    # inlier) — robust to a partial arc, unlike a raw edge-pixel count.
    angs = np.arctan2(inl[:, 1] - e[0][1], inl[:, 0] - e[0][0])
    conf = float((np.histogram(angs, bins=36, range=(-np.pi, np.pi))[0] > 0).mean())
    if conf < 0.5 or n_in < 12:
        return None
    return EllipseDetection(
        ellipse=((float(e[0][0]), float(e[0][1])),
                 (float(e[1][0]), float(e[1][1])), float(e[2])),
        confidence=float(conf), n_inliers=n_in)
