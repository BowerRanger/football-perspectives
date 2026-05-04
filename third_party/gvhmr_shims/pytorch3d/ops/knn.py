"""Pure-PyTorch KNN shim for pytorch3d.ops.knn used by GVHMR."""

import torch
from collections import namedtuple

KNNResult = namedtuple("KNNResult", ["dists", "idx", "knn"])


def knn_points(p1, p2, K=1, return_nn=False):
    """Brute-force K-nearest-neighbors (no CUDA kernel needed).

    Args:
        p1: (B, N, D) query points.
        p2: (B, M, D) reference points.
        K: number of nearest neighbors.
        return_nn: if True, also return the neighbor coordinates.

    Returns:
        (dists, idx, knn) where dists is (B, N, K) squared distances,
        idx is (B, N, K) indices into p2, knn is (B, N, K, D) or None.
    """
    # (B, N, M) pairwise squared distances
    diff = p1.unsqueeze(2) - p2.unsqueeze(1)  # (B, N, M, D)
    dists_all = (diff * diff).sum(dim=-1)      # (B, N, M)

    dists, idx = dists_all.topk(K, dim=-1, largest=False)  # (B, N, K)

    knn_pts = None
    if return_nn:
        # Gather neighbor points
        idx_expanded = idx.unsqueeze(-1).expand(-1, -1, -1, p2.shape[-1])
        knn_pts = p2.unsqueeze(1).expand(-1, p1.shape[1], -1, -1).gather(2, idx_expanded)

    return dists, idx, knn_pts
