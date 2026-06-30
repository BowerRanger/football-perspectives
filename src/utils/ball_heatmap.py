"""Pure candidate-blob extraction from ball-detector heatmaps.

Torch- and cv2-free so the candidate logic is unit-testable in the
light venv. 8-connected labelling matches the cv2.connectedComponents
behaviour previously used in wasb_ball_detector.

Threshold convention: pixels with ``hm >= min_score`` enter the mask.
The boundary is inclusive; strict ``>`` is not used because exact float
equality on sigmoid heatmaps is measure-zero in practice.  This is a
deliberate, documented difference from the old cv2 code which used ``>``.
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

    Blobs are connected components of ``hm >= min_score`` (inclusive
    boundary — see module docstring), ordered by descending sum-of-weights
    (mass); ``(x, y)`` is the heatmap-weighted centroid and ``peak`` the
    blob's max heatmap value.  NaN pixels in *hm* are treated as
    sub-threshold and never enter the mask.

    Blobs with zero or negative mass (e.g. components consisting entirely
    of zero-valued pixels admitted when ``min_score <= 0``) are silently
    skipped so they never produce NaN centroids.
    """
    # NaN-safe mask: NaN >= min_score is False, so NaN pixels are excluded.
    mask = hm >= min_score
    if not mask.any():
        return []

    labels, n_labels = ndimage.label(mask, structure=_EIGHT_CONNECTED)
    label_ids = np.arange(1, n_labels + 1)

    # Vectorised aggregation over all labels at once — O(pixels) rather than
    # O(pixels × labels).
    masses = ndimage.sum_labels(hm, labels, label_ids)

    # Guard: skip zero-mass blobs (all-zero weight pixels) to avoid NaN
    # centroids from division by zero.
    valid = masses > 0
    if not valid.any():
        return []

    valid_ids = label_ids[valid]
    valid_masses = masses[valid]

    rows, cols = np.indices(hm.shape)
    xs_sum = ndimage.sum_labels(hm * cols, labels, valid_ids)
    ys_sum = ndimage.sum_labels(hm * rows, labels, valid_ids)
    peaks = ndimage.maximum(hm, labels, valid_ids)

    cx = xs_sum / valid_masses
    cy = ys_sum / valid_masses

    # Descending mass; stable sort preserves lowest-label-first tie-break.
    order = np.argsort(-valid_masses, kind="stable")

    blobs = [
        (float(cx[i]), float(cy[i]), float(peaks[i]))
        for i in order[:top_k]
    ]
    return blobs
