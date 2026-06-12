"""Pure candidate-blob extraction from ball-detector heatmaps.

Torch- and cv2-free so the candidate logic is unit-testable in the
light venv. 8-connected labelling matches the cv2.connectedComponents
behaviour previously used in wasb_ball_detector.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

_EIGHT_CONNECTED = np.ones((3, 3), dtype=int)


def heatmap_candidates(
    hm: np.ndarray,
    min_score: float,
    top_k: int,
) -> list[tuple[float, float, float]]:
    """Top-k heatmap blobs as ``(x, y, peak)`` in heatmap pixel coords.

    Blobs are connected components of ``hm >= min_score``, ordered by
    descending sum-of-weights (mass); ``(x, y)`` is the heatmap-weighted
    centroid and ``peak`` the blob's max heatmap value.
    """
    mask = hm >= min_score
    if not mask.any():
        return []
    labels, n_labels = ndimage.label(mask, structure=_EIGHT_CONNECTED)
    blobs: list[tuple[float, float, float, float]] = []
    for label in range(1, n_labels + 1):
        ys, xs = np.nonzero(labels == label)
        ws = hm[ys, xs]
        mass = float(ws.sum())
        x = float((xs * ws).sum() / mass)
        y = float((ys * ws).sum() / mass)
        blobs.append((mass, x, y, float(ws.max())))
    blobs.sort(key=lambda b: -b[0])
    return [(x, y, peak) for _, x, y, peak in blobs[:top_k]]
