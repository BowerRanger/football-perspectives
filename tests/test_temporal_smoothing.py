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
