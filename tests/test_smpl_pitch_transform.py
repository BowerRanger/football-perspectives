import numpy as np
import pytest

from src.utils.smpl_pitch_transform import (
    GVHMR_TO_OPENCV_CAM,
    SMPL_TO_PITCH_STATIC,
    smpl_root_in_pitch_frame,
)


# A physically-realistic OpenCV extrinsic for a broadcast camera that
# sits behind the near touchline looking horizontally at the pitch:
#   world +x (along touchline)         -> camera +x (right in image)
#   world +y (across pitch, far side)  -> camera +z (forward into image)
#   world +z (up)                      -> camera -y (up in image)
# This is the mapping every camera_track.json frame approximates; tests
# that pass ``R_w2c = I`` are unphysical because they imply a z-up
# camera frame, which OpenCV cameras never use.
R_W2C_BROADCAST = np.array(
    [[1, 0,  0],
     [0, 0, -1],
     [0, 1,  0]],
    dtype=float,
)


def _yaw_about_world_z(angle: float) -> np.ndarray:
    """World-frame rotation about +z by ``angle`` rad (a camera pan).

    Composed onto an OpenCV ``R_w2c`` on the **right** with
    ``R_w2c_new = R_w2c_initial @ yaw.T`` — i.e. the camera body has
    rotated by ``yaw`` in world coordinates while the world stays
    still. (Multiplying on the left is a *camera-frame* rotation,
    which is a roll, not a pan, and was the bug in the previous test.)
    """
    c, s = np.cos(angle), np.sin(angle)
    return np.array(
        [[ c, -s, 0],
         [ s,  c, 0],
         [ 0,  0, 1]],
        dtype=float,
    )


@pytest.mark.unit
def test_upright_avatar_stays_upright_under_broadcast_camera():
    """For an upright player observed by a typical OpenCV broadcast
    camera, GVHMR's ``root_R_cam`` is identity (body axes aligned with
    GVHMR's y-up camera axes). The full chain must map SMPL canonical
    +y (body's "up") to pitch +z."""
    R_world_to_cam = R_W2C_BROADCAST
    root_R_cam = np.eye(3)
    R_world = smpl_root_in_pitch_frame(root_R_cam, R_world_to_cam)
    avatar_up_world = R_world @ np.array([0.0, 1.0, 0.0])
    assert avatar_up_world[2] > 0.99, avatar_up_world


@pytest.mark.unit
def test_static_transform_is_constant():
    assert GVHMR_TO_OPENCV_CAM.shape == (3, 3)
    # Determinant 1 (orthogonal, right-handed).
    assert abs(np.linalg.det(GVHMR_TO_OPENCV_CAM) - 1.0) < 1e-6
    # Orthogonal: R^T R == I.
    rt_r = GVHMR_TO_OPENCV_CAM.T @ GVHMR_TO_OPENCV_CAM
    assert np.allclose(rt_r, np.eye(3), atol=1e-9)
    # Backwards-compat alias still resolves to the same object.
    assert SMPL_TO_PITCH_STATIC is GVHMR_TO_OPENCV_CAM


@pytest.mark.unit
def test_pitch_up_axis_recovered_under_panned_camera():
    """Adding a pan (camera-body yaw about world +z) to the broadcast
    camera must not tilt the recovered avatar up axis. Guards against
    accidental ordering errors in the composition."""
    yaw = _yaw_about_world_z(np.deg2rad(30.0))
    R_world_to_cam = R_W2C_BROADCAST @ yaw.T
    root_R_cam = np.eye(3)
    R_world = smpl_root_in_pitch_frame(root_R_cam, R_world_to_cam)
    avatar_up_world = R_world @ np.array([0.0, 1.0, 0.0])
    assert avatar_up_world[2] > 0.99, avatar_up_world


@pytest.mark.unit
def test_foot_below_root_in_pitch():
    """Sanity: with body upright in pitch frame, applying R_world to
    the SMPL-local foot offset (0, -0.95, 0) should give a 0.95 m drop
    along pitch -z. Pins the relationship that ``hmr_world.py`` relies
    on for foot-anchored translation."""
    R_world = smpl_root_in_pitch_frame(np.eye(3), R_W2C_BROADCAST)
    foot_in_root = np.array([0.0, -0.95, 0.0])  # SMPL canonical y-up
    foot_offset_world = R_world @ foot_in_root
    np.testing.assert_allclose(foot_offset_world, [0.0, 0.0, -0.95], atol=1e-9)
