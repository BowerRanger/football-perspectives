import numpy as np
import pytest

from src.utils.temporal_smoothing import savgol_axis, slerp_window, ground_snap_z


@pytest.mark.unit
def test_savgol_does_not_blow_up_on_short_input():
    x = np.linspace(0, 1, 5).reshape(-1, 1)
    out = savgol_axis(x, window=11, order=2)
    assert out.shape == x.shape


@pytest.mark.unit
def test_slerp_window_passes_through_short_sequence():
    Rs = np.tile(np.eye(3), (2, 1, 1))
    out = slerp_window(Rs, window=5)
    assert np.allclose(out, Rs)


@pytest.mark.unit
def test_ground_snap_pulls_low_velocity_z_toward_zero():
    z = np.array([0.5, 0.5, 0.5, 0.5])
    out = ground_snap_z(z)
    assert (out < z).all()


def test_pin_and_smooth_preserves_pinned_and_relaxes_gaps():
    import numpy as np
    from scipy.spatial.transform import Rotation
    from src.utils.temporal_smoothing import pin_and_smooth_quat
    # smooth pan with a 2-frame interp gap whose middle frame is a bad guess
    angles = [0.3 * i for i in range(30)]
    angles[15] += 4.0  # bad interp guess (seam spike)
    Rs = np.stack([Rotation.from_euler("y", a, degrees=True).as_matrix() for a in angles])
    solved = [i != 15 for i in range(30)]  # frame 15 is the interp gap
    out = pin_and_smooth_quat(Rs, solved, window=9, iters=4)
    # pinned frames untouched
    for i in range(30):
        if solved[i]:
            assert np.allclose(out[i], Rs[i], atol=1e-9)
    # the gap frame got relaxed toward its neighbours (spike removed)
    assert not np.allclose(out[15], Rs[15], atol=1e-3)

    def jit(R):
        c = [(np.trace(R[i].T @ R[i + 1]) - 1) / 2 for i in range(len(R) - 1)]
        return max(np.degrees(np.arccos(np.clip(c, -1, 1))))
    assert jit(out) < jit(Rs) - 1.0
