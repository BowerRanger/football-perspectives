import numpy as np
import pytest

from src.utils.smpl_pitch_transform import (
    SMPL_TO_PITCH_STATIC,
    smpl_root_in_pitch_frame,
)


def _yaw(angle: float) -> np.ndarray:
    return np.array(
        [[np.cos(angle), 0, np.sin(angle)],
         [0, 1, 0],
         [-np.sin(angle), 0, np.cos(angle)]],
    )


@pytest.mark.unit
def test_walking_forward_camera_tilted_down_keeps_pitch_up_axis_aligned():
    """Regression pin (see decisions D8): with no camera rotation
    (R_world_to_cam = I), the SMPL canonical up axis (+y) must map to the
    pitch canonical up axis (+z). This pins the static transform that
    fixes the historical upside-down avatar bug."""
    R_world_to_cam = np.eye(3)
    # In the camera frame the avatar root is upright (identity).
    root_R_cam = np.eye(3)
    R_world = smpl_root_in_pitch_frame(root_R_cam, R_world_to_cam)
    # SMPL canonical +y (up) should map to pitch +z (up).
    avatar_up_local = np.array([0, 1, 0])
    avatar_up_world = R_world @ avatar_up_local
    assert avatar_up_world[2] > 0.9


@pytest.mark.unit
def test_static_transform_is_constant():
    assert SMPL_TO_PITCH_STATIC.shape == (3, 3)
    # Determinant 1 (orthogonal, right-handed).
    assert abs(np.linalg.det(SMPL_TO_PITCH_STATIC) - 1.0) < 1e-6
    # Orthogonal: R^T R == I.
    rt_r = SMPL_TO_PITCH_STATIC.T @ SMPL_TO_PITCH_STATIC
    assert np.allclose(rt_r, np.eye(3), atol=1e-9)


@pytest.mark.unit
def test_pitch_up_axis_recovered_under_yawed_camera():
    """Even when the camera is yawed about the pitch's vertical axis
    (a pure pan), an upright SMPL avatar should remain upright in pitch
    frame. This guards against accidental ordering of the composition."""
    # Pure yaw of 30 degrees about world-z, applied as world->camera.
    # World->camera yaw matrix about z:
    a = np.deg2rad(30.0)
    R_world_to_cam = np.array(
        [[np.cos(a), np.sin(a), 0],
         [-np.sin(a), np.cos(a), 0],
         [0, 0, 1]],
        dtype=float,
    )
    root_R_cam = np.eye(3)
    R_world = smpl_root_in_pitch_frame(root_R_cam, R_world_to_cam)
    avatar_up_world = R_world @ np.array([0, 1, 0])
    # Yaw about z should not tilt the up axis off pitch +z.
    assert avatar_up_world[2] > 0.9
