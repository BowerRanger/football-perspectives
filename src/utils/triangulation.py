"""Triangulation math for multi-view 3D reconstruction.

All functions are pure numpy — no side effects or file I/O.
"""

import numpy as np
import cv2
from scipy.signal import savgol_filter

# COCO skeleton bone edges (pairs of joint indices)
COCO_BONES: list[tuple[int, int]] = [
    (0, 1), (0, 2), (1, 3), (2, 4),        # head
    (5, 6),                                   # shoulders
    (5, 7), (7, 9), (6, 8), (8, 10),        # arms
    (5, 11), (6, 12), (11, 12),              # torso
    (11, 13), (13, 15), (12, 14), (14, 16),  # legs
]

# Ankle joint indices in COCO 17
_LEFT_ANKLE = 15
_RIGHT_ANKLE = 16


def weighted_dlt(
    projections: list[np.ndarray],
    points_2d: list[np.ndarray],
    weights: list[float],
) -> np.ndarray:
    """Triangulate a single 3D point from N ≥ 2 views using weighted DLT.

    Args:
        projections: list of 3×4 projection matrices.
        points_2d: list of (u, v) pixel coordinates.
        weights: per-view confidence weights.

    Returns:
        (3,) array — 3D point in world coordinates.
    """
    rows: list[np.ndarray] = []
    for (u, v), P, w in zip(points_2d, projections, weights):
        rows.append(w * (u * P[2] - P[0]))
        rows.append(w * (v * P[2] - P[1]))
    A = np.array(rows, dtype=np.float64)
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    return (X[:3] / X[3]).astype(np.float32)


def compute_reprojection_errors(
    pt_3d: np.ndarray,
    projections: list[np.ndarray],
    points_2d: list[np.ndarray],
) -> np.ndarray:
    """Per-view reprojection error (pixels) for a single 3D point.

    Returns:
        (N,) array of pixel distances.
    """
    pt_h = np.append(pt_3d, 1.0).astype(np.float64)
    errors = np.empty(len(projections), dtype=np.float32)
    for i, (P, uv) in enumerate(zip(projections, points_2d)):
        projected = P @ pt_h
        projected = projected[:2] / projected[2]
        errors[i] = float(np.linalg.norm(projected - uv))
    return errors


def ransac_triangulate(
    projections: list[np.ndarray],
    points_2d: list[np.ndarray],
    weights: list[float],
    threshold: float = 15.0,
    min_inliers: int = 2,
) -> tuple[np.ndarray, float, int]:
    """RANSAC wrapper around weighted DLT.

    Tries all 2-view combinations, triangulates, scores by inlier count,
    then re-triangulates with all inliers.

    Returns:
        (point_3d, mean_reproj_error, n_inlier_views)
    """
    n = len(projections)
    if n < 2:
        return np.full(3, np.nan, dtype=np.float32), float("inf"), 0

    if n == 2:
        pt = weighted_dlt(projections, points_2d, weights)
        errs = compute_reprojection_errors(pt, projections, points_2d)
        mean_err = float(np.mean(errs))
        # With only two views we can't meaningfully RANSAC, but we MUST
        # still reject solutions whose reprojection error exceeds the
        # threshold.  A wildly-wrong calibration (e.g. extrapolated
        # rotation) produces a mathematically valid DLT solution with
        # huge reprojection error; if we accept it the preview fills
        # with off-pitch garbage.
        if mean_err > threshold:
            return np.full(3, np.nan, dtype=np.float32), float("inf"), 0
        return pt, mean_err, 2

    best_inliers: list[int] = []
    best_pt = np.full(3, np.nan, dtype=np.float32)

    for i in range(n):
        for j in range(i + 1, n):
            pt = weighted_dlt(
                [projections[i], projections[j]],
                [points_2d[i], points_2d[j]],
                [weights[i], weights[j]],
            )
            errs = compute_reprojection_errors(pt, projections, points_2d)
            inliers = [k for k in range(n) if errs[k] < threshold]
            if len(inliers) > len(best_inliers):
                best_inliers = inliers
                best_pt = pt

    if len(best_inliers) < min_inliers:
        return np.full(3, np.nan, dtype=np.float32), float("inf"), 0

    # Re-triangulate with all inliers
    inlier_P = [projections[k] for k in best_inliers]
    inlier_uv = [points_2d[k] for k in best_inliers]
    inlier_w = [weights[k] for k in best_inliers]
    pt = weighted_dlt(inlier_P, inlier_uv, inlier_w)
    errs = compute_reprojection_errors(pt, inlier_P, inlier_uv)
    return pt, float(np.mean(errs)), len(best_inliers)


def temporal_smooth_savgol(
    positions: np.ndarray,
    window: int = 7,
    order: int = 3,
) -> np.ndarray:
    """Savitzky-Golay smoothing per joint trajectory.

    Handles NaN gaps by linear interpolation before filtering, then
    re-masks original NaN positions.

    Args:
        positions: (N, 17, 3) array, may contain NaN.
        window: filter window length (must be odd).
        order: polynomial order.

    Returns:
        (N, 17, 3) smoothed positions.
    """
    n_frames, n_joints, _ = positions.shape
    if n_frames < window:
        return positions.copy()

    result = positions.copy()
    for j in range(n_joints):
        for d in range(3):
            traj = positions[:, j, d].copy()
            valid = ~np.isnan(traj)
            if valid.sum() < window:
                continue
            # Interpolate NaN gaps
            if not valid.all():
                xp = np.where(valid)[0]
                fp = traj[valid]
                traj = np.interp(np.arange(n_frames), xp, fp)
            traj = savgol_filter(traj, window, order)
            # Re-mask original NaN positions
            traj[~valid] = np.nan
            result[:, j, d] = traj
    return result


def enforce_bone_lengths(
    positions: np.ndarray,
    tolerance: float = 0.2,
) -> np.ndarray:
    """Clamp bone lengths to within tolerance of per-player median.

    For each bone, compute the median length across all valid frames.
    For frames where a bone deviates > tolerance fraction from median,
    scale the child joint toward/away from the parent to match the median.

    Args:
        positions: (N, 17, 3) array.
        tolerance: max fractional deviation from median (e.g. 0.2 = 20%).

    Returns:
        (N, 17, 3) corrected positions.
    """
    result = positions.copy()
    for parent, child in COCO_BONES:
        bone_vecs = result[:, child] - result[:, parent]
        lengths = np.linalg.norm(bone_vecs, axis=1)
        valid = ~np.isnan(lengths) & (lengths > 0)
        if valid.sum() < 3:
            continue
        median_len = float(np.median(lengths[valid]))
        if median_len < 1e-6:
            continue
        for f in range(len(result)):
            if not valid[f]:
                continue
            ratio = lengths[f] / median_len
            if abs(ratio - 1.0) > tolerance:
                direction = bone_vecs[f] / lengths[f]
                result[f, child] = result[f, parent] + direction * median_len
    return result


def snap_feet_to_ground(
    positions: np.ndarray,
    velocity_threshold: float = 0.1,
    ground_z: float = 0.0,
) -> np.ndarray:
    """Snap ankle joints to ground plane when foot velocity is near zero.

    Args:
        positions: (N, 17, 3) array.
        velocity_threshold: m/frame threshold below which feet are snapped.
        ground_z: z-coordinate of the ground plane.

    Returns:
        (N, 17, 3) corrected positions.
    """
    result = positions.copy()
    for ankle_idx in [_LEFT_ANKLE, _RIGHT_ANKLE]:
        traj = result[:, ankle_idx, :]  # (N, 3)
        # Compute per-frame velocity (central difference)
        velocity = np.full(len(traj), np.nan)
        for f in range(1, len(traj) - 1):
            if np.any(np.isnan(traj[f - 1])) or np.any(np.isnan(traj[f + 1])):
                continue
            velocity[f] = np.linalg.norm(traj[f + 1] - traj[f - 1]) / 2.0
        # Snap to ground when velocity is low
        for f in range(len(traj)):
            if np.isnan(velocity[f]):
                continue
            if velocity[f] < velocity_threshold:
                result[f, ankle_idx, 2] = ground_z
    return result
