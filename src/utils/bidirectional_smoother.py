"""Forward/backward propagation fusion for camera tracking."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def smooth_between_anchors(
    Ks_fwd: list[np.ndarray],
    Rs_fwd: list[np.ndarray],
    Ks_bwd: list[np.ndarray],
    Rs_bwd: list[np.ndarray],
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Per-frame K, R fusion of forward (from prior anchor) and backward
    (from next anchor) propagation outputs. Endpoints exactly match
    their respective anchors by construction.

    The two input lists must have the same length and represent the
    inclusive frame range [anchor_a, anchor_b].
    """
    if len(Ks_fwd) != len(Ks_bwd) or len(Rs_fwd) != len(Rs_bwd):
        raise ValueError("forward and backward sequences must have equal length")
    n = len(Ks_fwd)
    if n < 2:
        raise ValueError(f"need >=2 frames between anchors, got {n}")

    Ks_out: list[np.ndarray] = []
    Rs_out: list[np.ndarray] = []
    for i in range(n):
        w_fwd = (n - 1 - i) / (n - 1)  # 1.0 at anchor_a, 0.0 at anchor_b
        K = w_fwd * Ks_fwd[i] + (1 - w_fwd) * Ks_bwd[i]
        rots = Rotation.from_matrix([Rs_fwd[i], Rs_bwd[i]])
        slerp = Slerp([0.0, 1.0], rots)
        R = slerp([1 - w_fwd]).as_matrix()[0]
        Ks_out.append(K)
        Rs_out.append(R)
    return Ks_out, Rs_out
