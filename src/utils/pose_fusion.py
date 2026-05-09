"""Math primitives for cross-shot SMPL fusion.

Used by ``src.stages.refined_poses``. All functions are pure — no I/O,
no logging — and operate on numpy arrays. SO(3) operations follow the
chordal (Frobenius) cost so the weighted mean is the SVD-projection
of the Euclidean weighted mean back onto the rotation group.
"""

from __future__ import annotations

import numpy as np


def so3_chordal_mean(rotations: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted chordal mean of a stack of SO(3) matrices.

    Minimises ``sum_v w_v ||R - R_v||_F^2`` over R in SO(3) via
    SVD-projection of the weighted Euclidean mean.

    Args:
        rotations: ``(V, 3, 3)`` proper rotation matrices.
        weights:   ``(V,)`` non-negative weights with sum > 0.

    Returns:
        The mean rotation as a ``(3, 3)`` proper rotation matrix.

    Raises:
        ValueError: if shapes mismatch or weights sum to zero.
    """
    if rotations.ndim != 3 or rotations.shape[1:] != (3, 3):
        raise ValueError(f"rotations must be (V, 3, 3); got {rotations.shape}")
    if weights.shape != (rotations.shape[0],):
        raise ValueError(
            f"weights must be ({rotations.shape[0]},); got {weights.shape}"
        )
    w_sum = float(weights.sum())
    if w_sum <= 0.0:
        raise ValueError("weights sum to zero")
    weighted_mean = (weights[:, None, None] * rotations).sum(axis=0) / w_sum
    U, _, Vt = np.linalg.svd(weighted_mean)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = U @ Vt
    return R


def so3_geodesic_distance(R1: np.ndarray, R2: np.ndarray) -> float:
    """Angular distance in radians between two SO(3) matrices."""
    cos_theta = (np.trace(R1.T @ R2) - 1.0) / 2.0
    cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
    return float(np.arccos(cos_theta))
