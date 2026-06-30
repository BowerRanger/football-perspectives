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


@pytest.mark.unit
def test_single_pixel_blob_exact_coords():
    """A single active pixel returns its exact (x, y) and peak."""
    hm = np.zeros((20, 30), dtype=np.float32)
    hm[7, 13] = 0.75
    cands = heatmap_candidates(hm, min_score=0.5, top_k=5)
    assert len(cands) == 1
    x, y, peak = cands[0]
    assert x == pytest.approx(13.0)
    assert y == pytest.approx(7.0)
    assert peak == pytest.approx(0.75)


@pytest.mark.unit
def test_float32_input_produces_finite_centroids():
    """float32 heatmap must not produce NaN or Inf centroids."""
    rng = np.random.default_rng(42)
    hm = rng.random((72, 128)).astype(np.float32)
    cands = heatmap_candidates(hm, min_score=0.5, top_k=10)
    for x, y, peak in cands:
        assert np.isfinite(x), f"x={x} is not finite"
        assert np.isfinite(y), f"y={y} is not finite"
        assert np.isfinite(peak), f"peak={peak} is not finite"


@pytest.mark.unit
def test_nan_pixels_excluded():
    """NaN pixels must not bleed into output; the valid blob survives."""
    hm = np.full((20, 30), np.nan, dtype=np.float32)
    # One valid blob at (y=10, x=15, peak=0.8, r=1)
    hm[9:12, 14:17] = 0.8
    cands = heatmap_candidates(hm, min_score=0.5, top_k=5)
    assert len(cands) == 1
    x, y, peak = cands[0]
    assert np.isfinite(x) and np.isfinite(y) and np.isfinite(peak)
    assert x == pytest.approx(15.0)
    assert y == pytest.approx(10.0)


@pytest.mark.unit
def test_zero_mass_no_nan_candidates():
    """Zero-mass blobs must be silently skipped — no NaN candidates (Fix 2).

    A heatmap where some pixels are below zero (excluded by mask) and some
    are exactly zero (included when min_score <= 0) creates blobs whose
    sum-of-weights is 0.  The old code divides by that mass → NaN.
    """
    # Background is -0.1 (excluded); a zero-valued patch is a separate,
    # isolated blob from the 0.5 pixel, so its mass = 0.
    hm = np.full((10, 10), -0.1, dtype=np.float32)
    hm[3, 3] = 0.5   # real blob
    hm[7, 7] = 0.0   # zero-weight pixel — its own connected component
    hm[7, 8] = 0.0
    cands = heatmap_candidates(hm, min_score=0.0, top_k=20)
    for x, y, peak in cands:
        assert np.isfinite(x), "NaN x leaked from zero-mass blob"
        assert np.isfinite(y), "NaN y leaked from zero-mass blob"


@pytest.mark.unit
def test_inclusive_min_score_boundary():
    """A pixel whose value equals min_score exactly is included (Fix 3)."""
    hm = np.zeros((10, 10), dtype=np.float32)
    threshold = 0.4
    hm[5, 5] = threshold  # exactly at threshold
    cands = heatmap_candidates(hm, min_score=threshold, top_k=5)
    assert len(cands) == 1
    x, y, _ = cands[0]
    assert x == pytest.approx(5.0)
    assert y == pytest.approx(5.0)
