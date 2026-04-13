"""Vanishing-point based calibration refinement.

Given two orthogonal vanishing points on the pitch plane (one for the
touchline direction, one for the goal-line direction) and a known
camera world position, we can recover the camera intrinsics (focal
length) and rotation analytically — no iterative solver needed.

The single-view metrology technique used here is standard (see
Hartley & Zisserman §8.6).  Inputs:

- Two pixel-space vanishing points ``v1``, ``v2`` known to be the
  projections of two orthogonal world directions.
- Principal point ``(cx, cy)`` — assumed to be the image centre.
- Camera world position ``C`` — kept fixed from the static-camera
  fuser.

Outputs:

- Focal length ``f``.
- 3×3 rotation matrix ``R`` mapping world → camera.
- Translation ``t = -R @ C``.

We also expose helper routines for:

- Computing the vanishing point of a set of 2D line segments via
  least-squares (each segment contributes one constraint to the
  intersection point).
- Picking the dominant vanishing-point direction from a cluster of
  detected lines.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from src.utils.pitch_line_detector import DetectedLine

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VPCalibrationResult:
    """Calibration recovered from a pair of orthogonal vanishing points."""

    K: np.ndarray            # (3, 3) intrinsic
    R: np.ndarray            # (3, 3) world→camera rotation
    rvec: np.ndarray         # (3,) Rodrigues
    tvec: np.ndarray         # (3,) translation
    focal_length: float      # pixels
    vp_touchline: np.ndarray   # (2,) pixel coords of touchline VP
    vp_goalline: np.ndarray    # (2,) pixel coords of goal-line VP
    n_inliers_touchline: int
    n_inliers_goalline: int


def _line_to_homogeneous(line: DetectedLine) -> np.ndarray:
    """Return the 3-vector representation of the line in pixel coords.

    ``ax + by + c = 0`` form, normalised so ``a^2 + b^2 = 1``.
    """
    p1 = np.array([line.x1, line.y1, 1.0], dtype=np.float64)
    p2 = np.array([line.x2, line.y2, 1.0], dtype=np.float64)
    h = np.cross(p1, p2)
    n = float(np.hypot(h[0], h[1]))
    if n < 1e-9:
        return h
    return h / n


def vanishing_point_from_lines(
    lines: list[DetectedLine],
    *,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Least-squares vanishing point of a set of image-space lines.

    Each line contributes one equation ``a*x + b*y + c = 0``; the
    vanishing point is the (homogeneous) point that minimises the
    weighted sum of squared algebraic distances.

    Args:
        lines: list of :class:`DetectedLine` (length ≥ 2).
        weights: optional weight per line.  If ``None``, segment
            length is used.

    Returns:
        ``(2,)`` pixel coordinates of the vanishing point.  Returns
        a vector of NaN if the system is degenerate (all lines parallel
        in image space — VP at infinity along that direction).
    """
    if len(lines) < 2:
        return np.array([np.nan, np.nan], dtype=np.float64)
    if weights is None:
        weights = np.array([ln.length for ln in lines], dtype=np.float64)
    rows: list[np.ndarray] = []
    for ln, w in zip(lines, weights):
        h = _line_to_homogeneous(ln)
        rows.append(float(w) * h)
    A = np.array(rows, dtype=np.float64)
    _, _, Vt = np.linalg.svd(A)
    vp_h = Vt[-1]
    if abs(vp_h[2]) < 1e-9:
        # VP at infinity in image space → return direction as a far point
        return np.array([vp_h[0] * 1e6, vp_h[1] * 1e6], dtype=np.float64)
    return np.array([vp_h[0] / vp_h[2], vp_h[1] / vp_h[2]], dtype=np.float64)


def _angle_diff(a: float, b: float) -> float:
    """Smallest angular difference between two directions in [0, π)."""
    d = abs(a - b) % np.pi
    return float(min(d, np.pi - d))


def cluster_lines_by_orientation(
    lines: list[DetectedLine],
    *,
    n_clusters: int = 2,
    angle_tol_deg: float = 8.0,
) -> list[list[DetectedLine]]:
    """Group lines into clusters by image-space orientation.

    Greedy 1-D agglomerative clustering: pick the longest unassigned
    line, gather all lines whose orientation is within ``angle_tol_deg``,
    repeat until ``n_clusters`` clusters are accumulated or no lines
    remain.

    Returns clusters sorted by total weight (sum of segment lengths)
    descending.
    """
    remaining = sorted(lines, key=lambda ln: -ln.length)
    clusters: list[list[DetectedLine]] = []
    tol_rad = np.deg2rad(angle_tol_deg)

    while remaining and len(clusters) < n_clusters:
        seed = remaining.pop(0)
        cluster = [seed]
        leftover: list[DetectedLine] = []
        for ln in remaining:
            if _angle_diff(ln.angle, seed.angle) <= tol_rad:
                cluster.append(ln)
            else:
                leftover.append(ln)
        clusters.append(cluster)
        remaining = leftover
    clusters.sort(key=lambda c: -sum(ln.length for ln in c))
    return clusters


def calibration_from_vanishing_points(
    vp_touchline: np.ndarray,
    vp_goalline: np.ndarray,
    image_size: tuple[int, int],
    camera_position_world: np.ndarray,
    touchline_direction_world: np.ndarray = np.array([1.0, 0.0, 0.0]),
    goalline_direction_world: np.ndarray = np.array([0.0, 1.0, 0.0]),
) -> VPCalibrationResult | None:
    """Compute (K, R, t) from two orthogonal pixel-space vanishing points.

    Args:
        vp_touchline: pixel coords of the VP for lines parallel to
            world ``touchline_direction_world``.
        vp_goalline: pixel coords of the VP for lines parallel to
            world ``goalline_direction_world``.
        image_size: ``(width, height)``.  Principal point assumed at
            the centre.
        camera_position_world: known camera world position ``C``.
        touchline_direction_world: world direction of the first line
            family.  Default is ``[1, 0, 0]`` (touchlines run along
            the pitch length).
        goalline_direction_world: world direction of the second line
            family.  Default is ``[0, 1, 0]``.

    Returns:
        :class:`VPCalibrationResult` or ``None`` when the focal length
        comes out negative / imaginary (the two VPs are not orthogonal
        with respect to the assumed principal point).
    """
    width, height = image_size
    cx = width / 2.0
    cy = height / 2.0

    v1 = np.asarray(vp_touchline, dtype=np.float64)
    v2 = np.asarray(vp_goalline, dtype=np.float64)
    if not (np.all(np.isfinite(v1)) and np.all(np.isfinite(v2))):
        return None

    # Single-view metrology focal length:
    #   f^2 = -(v1x - cx) * (v2x - cx) - (v1y - cy) * (v2y - cy)
    f_sq = -((v1[0] - cx) * (v2[0] - cx) + (v1[1] - cy) * (v2[1] - cy))
    if f_sq <= 0:
        logger.debug("VPs not orthogonal w.r.t. centre — f^2 = %.1f", f_sq)
        return None
    f = float(np.sqrt(f_sq))

    K = np.array([[f, 0.0, cx],
                  [0.0, f, cy],
                  [0.0, 0.0, 1.0]], dtype=np.float64)

    # Camera-frame direction vectors of the two VP world directions:
    #   d1_cam = K^-1 @ [v1x, v1y, 1]
    #   d2_cam = K^-1 @ [v2x, v2y, 1]
    K_inv = np.linalg.inv(K)
    d1_cam = K_inv @ np.array([v1[0], v1[1], 1.0])
    d2_cam = K_inv @ np.array([v2[0], v2[1], 1.0])
    d1_cam /= np.linalg.norm(d1_cam)
    d2_cam /= np.linalg.norm(d2_cam)

    # World direction unit vectors
    d1_world = np.asarray(touchline_direction_world, dtype=np.float64)
    d2_world = np.asarray(goalline_direction_world, dtype=np.float64)
    d1_world /= np.linalg.norm(d1_world)
    d2_world /= np.linalg.norm(d2_world)

    # The rotation R (world → camera) maps d1_world → d1_cam and
    # d2_world → d2_cam.  Build orthonormal world & camera bases and
    # compose R = B_cam @ B_world^T.
    d3_world = np.cross(d1_world, d2_world)
    d3_cam = np.cross(d1_cam, d2_cam)
    # Re-orthogonalise d2 in case the VPs aren't perfectly orthogonal.
    d2_world = np.cross(d3_world, d1_world)
    d2_cam = np.cross(d3_cam, d1_cam)
    d2_world /= np.linalg.norm(d2_world)
    d2_cam /= np.linalg.norm(d2_cam)
    d3_world /= np.linalg.norm(d3_world)
    d3_cam /= np.linalg.norm(d3_cam)

    B_world = np.column_stack([d1_world, d2_world, d3_world])
    B_cam = np.column_stack([d1_cam, d2_cam, d3_cam])
    R = B_cam @ B_world.T

    # Ensure right-handed (det +1).  If we picked the wrong sign for
    # one of the VPs the determinant will be -1 — flip d3 to fix it.
    if np.linalg.det(R) < 0:
        d3_cam = -d3_cam
        B_cam = np.column_stack([d1_cam, d2_cam, d3_cam])
        R = B_cam @ B_world.T

    # Translation: known camera position
    C = np.asarray(camera_position_world, dtype=np.float64).reshape(3)
    t = -R @ C

    import cv2
    rvec, _ = cv2.Rodrigues(R)
    return VPCalibrationResult(
        K=K,
        R=R,
        rvec=rvec.reshape(3),
        tvec=t,
        focal_length=f,
        vp_touchline=v1,
        vp_goalline=v2,
        n_inliers_touchline=0,
        n_inliers_goalline=0,
    )
