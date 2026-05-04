import numpy as np
import pytest

from src.utils.bidirectional_smoother import smooth_between_anchors


def _slerp_naive(R0, R1, t):
    """Helper to compose a known truth trajectory."""
    from scipy.spatial.transform import Rotation, Slerp
    rots = Rotation.from_matrix([R0, R1])
    return Slerp([0, 1], rots)([t]).as_matrix()[0]


@pytest.mark.unit
def test_smoother_matches_anchors_at_endpoints():
    K_anchor_a = np.array([[1500.0, 0, 960], [0, 1500.0, 540], [0, 0, 1]])
    K_anchor_b = np.array([[1700.0, 0, 960], [0, 1700.0, 540], [0, 0, 1]])
    R_anchor_a = np.eye(3)
    R_anchor_b = np.array(
        [[0.99619, 0, 0.08716],
         [0, 1, 0],
         [-0.08716, 0, 0.99619]],
        dtype=float,
    )

    # Forward propagator returns clean linear interpolation.
    Ks_fwd = [K_anchor_a + (K_anchor_b - K_anchor_a) * (i / 10) for i in range(11)]
    Rs_fwd = [_slerp_naive(R_anchor_a, R_anchor_b, i / 10) for i in range(11)]
    Ks_bwd = [K_anchor_a + (K_anchor_b - K_anchor_a) * (i / 10) for i in range(11)]
    Rs_bwd = [_slerp_naive(R_anchor_a, R_anchor_b, i / 10) for i in range(11)]

    Ks_smooth, Rs_smooth = smooth_between_anchors(Ks_fwd, Rs_fwd, Ks_bwd, Rs_bwd)
    assert np.allclose(Ks_smooth[0], K_anchor_a)
    assert np.allclose(Ks_smooth[-1], K_anchor_b)
    assert np.allclose(Rs_smooth[0], R_anchor_a, atol=1e-6)
    assert np.allclose(Rs_smooth[-1], R_anchor_b, atol=1e-6)


@pytest.mark.unit
def test_smoother_bounds_drift_to_half():
    # Forward drifts +1° per step; backward is exact.
    K = np.array([[1500.0, 0, 960], [0, 1500.0, 540], [0, 0, 1]])
    n = 11
    Rs_truth = [_slerp_naive(np.eye(3), _yaw(np.deg2rad(10)), i / 10) for i in range(n)]
    Rs_fwd = [r @ _yaw(np.deg2rad(0.1 * i)) for i, r in enumerate(Rs_truth)]  # drifting
    Rs_bwd = list(Rs_truth)  # exact
    Ks_fwd = [K] * n
    Ks_bwd = [K] * n

    Ks_smooth, Rs_smooth = smooth_between_anchors(Ks_fwd, Rs_fwd, Ks_bwd, Rs_bwd)

    # At midpoint the smoothed estimate should be much closer to truth than fwd alone
    mid = n // 2
    err_fwd = np.linalg.norm(Rs_fwd[mid] - Rs_truth[mid], ord="fro")
    err_smooth = np.linalg.norm(Rs_smooth[mid] - Rs_truth[mid], ord="fro")
    assert err_smooth < 0.6 * err_fwd


def _yaw(angle: float) -> np.ndarray:
    return np.array(
        [[np.cos(angle), 0, np.sin(angle)],
         [0, 1, 0],
         [-np.sin(angle), 0, np.cos(angle)]],
    )
