import numpy as np
import pytest

from src.utils.neural_calibrator import convert_pnlcalib_to_ours


def test_convert_identity_rotation_centres_to_corner_origin():
    R = np.eye(3)
    C_pnl = np.array([0.0, 0.0, 0.0])
    rvec, tvec, C_ours = convert_pnlcalib_to_ours(R, C_pnl)
    assert C_ours == pytest.approx([52.5, 34.0, 0.0])


def test_convert_flips_y_and_z():
    R = np.eye(3)
    C_pnl = np.array([10.0, 5.0, -15.0])
    _, _, C_ours = convert_pnlcalib_to_ours(R, C_pnl)
    assert C_ours == pytest.approx([62.5, 29.0, 15.0])


def test_convert_preserves_projection_consistency():
    rng = np.random.default_rng(0)
    R = np.linalg.qr(rng.standard_normal((3, 3)))[0]
    C_pnl = np.array([3.0, -2.0, -20.0])
    rvec, tvec, C_ours = convert_pnlcalib_to_ours(R, C_pnl)
    import cv2
    R_ours, _ = cv2.Rodrigues(rvec)
    assert tvec == pytest.approx((-R_ours @ C_ours))
