"""Unit tests for src.utils.pose_fusion (math primitives only)."""

from __future__ import annotations

import numpy as np
import pytest

from src.utils.pose_fusion import so3_chordal_mean, so3_geodesic_distance


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
