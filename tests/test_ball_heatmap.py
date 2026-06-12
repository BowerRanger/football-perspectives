"""heatmap_candidates: top-k blob extraction from a detector heatmap."""

from __future__ import annotations

import numpy as np
import pytest

from src.utils.ball_heatmap import heatmap_candidates


def _hm_with_blobs(*blobs: tuple[int, int, float, int]) -> np.ndarray:
    """Blobs as (y, x, peak, radius) square stamps on a 72x128 heatmap."""
    hm = np.zeros((72, 128), dtype=np.float32)
    for y, x, peak, r in blobs:
        hm[y - r:y + r + 1, x - r:x + r + 1] = peak
    return hm


@pytest.mark.unit
def test_orders_by_blob_mass_and_returns_peak():
    hm = _hm_with_blobs((20, 30, 0.9, 1), (50, 100, 0.6, 3))
    # Blob 2 has lower peak but far more mass (7x7 @ 0.6 vs 3x3 @ 0.9).
    cands = heatmap_candidates(hm, min_score=0.3, top_k=5)
    assert len(cands) == 2
    (x0, y0, p0), (x1, y1, p1) = cands
    assert (round(x0), round(y0), p0) == (100, 50, pytest.approx(0.6))
    assert (round(x1), round(y1), p1) == (30, 20, pytest.approx(0.9))


@pytest.mark.unit
def test_min_score_filters_and_top_k_truncates():
    hm = _hm_with_blobs((20, 30, 0.9, 1), (50, 100, 0.2, 3), (60, 10, 0.5, 2))
    assert len(heatmap_candidates(hm, min_score=0.3, top_k=5)) == 2
    assert len(heatmap_candidates(hm, min_score=0.3, top_k=1)) == 1
    assert heatmap_candidates(hm, min_score=0.95, top_k=5) == []


@pytest.mark.unit
def test_diagonal_pixels_are_one_blob():
    """8-connectivity parity with cv2.connectedComponents."""
    hm = np.zeros((10, 10), dtype=np.float32)
    hm[2, 2] = 0.8
    hm[3, 3] = 0.8  # diagonal neighbour
    assert len(heatmap_candidates(hm, min_score=0.5, top_k=5)) == 1
