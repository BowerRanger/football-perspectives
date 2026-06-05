import numpy as np
from scipy.spatial.transform import Rotation

from src.utils.temporal_smoothing import quat_savgol


def _max_jump_deg(Rs):
    out = []
    for a, b in zip(Rs[:-1], Rs[1:]):
        c = (np.trace(a.T @ b) - 1) / 2
        out.append(np.degrees(np.arccos(max(-1.0, min(1.0, c)))))
    return max(out)


def test_quat_savgol_reduces_single_frame_spike():
    angles = np.linspace(0, 2, 30)  # slow pan about z
    Rs = np.stack([Rotation.from_euler("z", a, degrees=True).as_matrix() for a in angles])
    Rs[10] = Rotation.from_euler("z", angles[10] + 5.0, degrees=True).as_matrix()
    out = quat_savgol(Rs, window=9, order=2)
    assert _max_jump_deg(out) < 0.5 * _max_jump_deg(Rs)


def test_quat_savgol_actually_changes_rotations():
    """Guards against the slerp_window no-op bug it replaced."""
    rng = np.random.default_rng(0)
    Rs = np.stack([Rotation.from_rotvec(0.02 * rng.standard_normal(3)).as_matrix() for _ in range(20)])
    out = quat_savgol(Rs, window=7, order=2)
    assert not np.allclose(out, Rs)


def test_quat_savgol_noop_for_tiny_window():
    Rs = np.stack([np.eye(3)] * 5)
    assert np.allclose(quat_savgol(Rs, window=2), Rs)
