import numpy as np
import pytest

from src.utils.feature_propagator import (
    PropagatorResult,
    decompose_homography_to_R_zoom,
    propagate_one_frame,
)


@pytest.mark.unit
def test_decompose_zero_motion_homography_returns_identity():
    K = np.array([[1500.0, 0, 960], [0, 1500.0, 540], [0, 0, 1]])
    H = np.eye(3)
    dR, zoom = decompose_homography_to_R_zoom(H, K)
    assert np.allclose(dR, np.eye(3), atol=1e-6)
    assert abs(zoom - 1.0) < 1e-6


@pytest.mark.unit
def test_decompose_recovers_known_pan():
    K = np.array([[1500.0, 0, 960], [0, 1500.0, 540], [0, 0, 1]])
    angle = np.deg2rad(2.0)
    dR_true = np.array(
        [[np.cos(angle), 0, np.sin(angle)],
         [0, 1, 0],
         [-np.sin(angle), 0, np.cos(angle)]],
    )
    H = K @ dR_true @ np.linalg.inv(K)
    dR, zoom = decompose_homography_to_R_zoom(H, K)
    assert np.allclose(dR, dR_true, atol=1e-3)
    assert abs(zoom - 1.0) < 1e-3


@pytest.mark.unit
def test_decompose_recovers_known_zoom():
    K = np.array([[1500.0, 0, 960], [0, 1500.0, 540], [0, 0, 1]])
    K_next = np.array([[1650.0, 0, 960], [0, 1650.0, 540], [0, 0, 1]])
    H = K_next @ np.linalg.inv(K)
    dR, zoom = decompose_homography_to_R_zoom(H, K)
    assert np.allclose(dR, np.eye(3), atol=1e-3)
    assert abs(zoom - 1.1) < 1e-2
