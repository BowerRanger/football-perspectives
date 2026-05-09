"""Unit tests for src.utils.pose_fusion (math primitives only)."""

from __future__ import annotations

import numpy as np
import pytest

from src.utils.pose_fusion import (
    robust_translation_fuse,
    so3_chordal_mean,
    so3_geodesic_distance,
)


def _rotation_matrix_z(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


@pytest.mark.unit
def test_so3_chordal_mean_returns_input_for_single_view() -> None:
    R = _rotation_matrix_z(np.pi / 4)
    result = so3_chordal_mean(R[None, :, :], np.array([1.0]))
    np.testing.assert_allclose(result, R, atol=1e-9)


@pytest.mark.unit
def test_so3_chordal_mean_identity_for_two_equal_rotations() -> None:
    R = _rotation_matrix_z(np.pi / 6)
    result = so3_chordal_mean(np.stack([R, R]), np.array([1.0, 1.0]))
    np.testing.assert_allclose(result, R, atol=1e-9)


@pytest.mark.unit
def test_so3_chordal_mean_skews_toward_heavy_view() -> None:
    R1 = np.eye(3)
    R2 = _rotation_matrix_z(np.pi / 2)
    # Weight 1.0 on R1, 0.0 on R2 → result is R1.
    result = so3_chordal_mean(np.stack([R1, R2]), np.array([1.0, 0.0]))
    np.testing.assert_allclose(result, R1, atol=1e-9)


@pytest.mark.unit
def test_so3_chordal_mean_returns_proper_rotation() -> None:
    rng = np.random.default_rng(0)
    Rs = np.stack([_rotation_matrix_z(a) for a in rng.uniform(-1, 1, 5)])
    result = so3_chordal_mean(Rs, np.ones(5))
    assert np.linalg.det(result) > 0
    np.testing.assert_allclose(result.T @ result, np.eye(3), atol=1e-9)


@pytest.mark.unit
def test_so3_chordal_mean_rejects_zero_weights() -> None:
    R = np.eye(3)
    with pytest.raises(ValueError):
        so3_chordal_mean(R[None, :, :], np.array([0.0]))


@pytest.mark.unit
def test_so3_geodesic_distance_known_angles() -> None:
    R1 = np.eye(3)
    R2 = _rotation_matrix_z(np.pi / 2)
    assert abs(so3_geodesic_distance(R1, R2) - np.pi / 2) < 1e-9
    assert so3_geodesic_distance(R1, R1) < 1e-9


@pytest.mark.unit
def test_so3_geodesic_distance_symmetric() -> None:
    R1 = _rotation_matrix_z(0.3)
    R2 = _rotation_matrix_z(1.1)
    d12 = so3_geodesic_distance(R1, R2)
    d21 = so3_geodesic_distance(R2, R1)
    assert abs(d12 - d21) < 1e-12


@pytest.mark.unit
def test_robust_translation_fuse_two_views_passthrough() -> None:
    positions = np.array([[1.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    weights = np.array([1.0, 1.0])
    fused, kept = robust_translation_fuse(positions, weights, k_sigma=3.0)
    np.testing.assert_allclose(fused, [2.0, 0.0, 0.0])
    assert kept.tolist() == [True, True]


@pytest.mark.unit
def test_robust_translation_fuse_drops_far_outlier() -> None:
    positions = np.array(
        [[1.0, 0.0, 0.0], [1.05, 0.0, 0.0], [11.0, 0.0, 0.0]]
    )
    weights = np.ones(3)
    fused, kept = robust_translation_fuse(positions, weights, k_sigma=3.0)
    assert kept.tolist() == [True, True, False]
    np.testing.assert_allclose(fused, [1.025, 0.0, 0.0])


@pytest.mark.unit
def test_robust_translation_fuse_weighted_after_drop() -> None:
    positions = np.array(
        [[1.0, 0.0, 0.0], [3.0, 0.0, 0.0], [50.0, 0.0, 0.0]]
    )
    weights = np.array([1.0, 3.0, 1.0])
    fused, kept = robust_translation_fuse(positions, weights, k_sigma=3.0)
    assert kept.tolist() == [True, True, False]
    np.testing.assert_allclose(fused, [2.5, 0.0, 0.0])


@pytest.mark.unit
def test_robust_translation_fuse_no_drops_when_clustered() -> None:
    positions = np.array(
        [[1.0, 0.0, 0.0], [1.05, 0.0, 0.0], [0.95, 0.0, 0.0]]
    )
    weights = np.ones(3)
    fused, kept = robust_translation_fuse(positions, weights, k_sigma=3.0)
    assert kept.tolist() == [True, True, True]
    np.testing.assert_allclose(fused, [1.0, 0.0, 0.0])


@pytest.mark.unit
def test_robust_translation_fuse_zero_weight_view_excluded_from_mean() -> None:
    positions = np.array([[1.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    weights = np.array([1.0, 0.0])
    fused, kept = robust_translation_fuse(positions, weights, k_sigma=3.0)
    np.testing.assert_allclose(fused, [1.0, 0.0, 0.0])
    assert kept.tolist() == [True, True]


@pytest.mark.unit
def test_robust_translation_fuse_zero_total_weight_returns_zero() -> None:
    positions = np.array([[1.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    weights = np.zeros(2)
    fused, kept = robust_translation_fuse(positions, weights, k_sigma=3.0)
    np.testing.assert_allclose(fused, [0.0, 0.0, 0.0])
    assert kept.tolist() == [False, False]
