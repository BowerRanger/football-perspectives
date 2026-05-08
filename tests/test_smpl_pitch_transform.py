"""Tests for the SMPL canonical → pitch-world rotation chain.

Convention reminders:
- Pitch world is z-up.
- OpenCV camera is y-down, z-into-scene.
- ``root_R_cam`` (from GVHMR's ``smpl_params_incam.global_orient``) maps
  a SMPL canonical (y-up) body vector directly into the OpenCV camera
  frame; the chain to pitch world is then just ``R_w2c.T @ root_R_cam``.
- For an upright body observed by a forward-looking camera, the
  body-up axis in the camera frame is camera ``-y`` (because OpenCV
  is y-down), so ``root_R_cam`` for an upright body is a 180° flip
  about world horizontal — concretely ``X_180`` here.
"""

import numpy as np
import pytest

from src.utils.smpl_pitch_transform import smpl_root_in_pitch_frame


# A physically-realistic OpenCV extrinsic for a broadcast camera that
# sits behind the near touchline looking horizontally at the pitch:
#   world +x (along touchline)         -> camera +x (right in image)
#   world +y (across pitch, far side)  -> camera +z (forward into image)
#   world +z (up)                      -> camera -y (up in image)
R_W2C_BROADCAST = np.array(
    [[1, 0,  0],
     [0, 0, -1],
     [0, 1,  0]],
    dtype=float,
)

# 180° rotation about +x — this is what ``root_R_cam`` looks like for
# an upright body under the OpenCV (y-down) convention. Maps canonical
# +y (body up) → camera -y (image up).
X_180 = np.array(
    [[1,  0,  0],
     [0, -1,  0],
     [0,  0, -1]],
    dtype=float,
)


def _yaw_about_world_z(angle: float) -> np.ndarray:
    """World-frame rotation about +z by ``angle`` rad (a camera pan)."""
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
    camera, ``root_R_cam`` is X_180 (canonical y-up → camera y-down).
    The chain must then map SMPL canonical +y (body's "up") to pitch +z."""
    R_world = smpl_root_in_pitch_frame(X_180, R_W2C_BROADCAST)
    avatar_up_world = R_world @ np.array([0.0, 1.0, 0.0])
    assert avatar_up_world[2] > 0.99, avatar_up_world


@pytest.mark.unit
def test_pitch_up_axis_recovered_under_panned_camera():
    """Adding a pan (camera-body yaw about world +z) to the broadcast
    camera must not tilt the recovered avatar up axis. Guards against
    accidental ordering errors in the composition."""
    yaw = _yaw_about_world_z(np.deg2rad(30.0))
    R_world_to_cam = R_W2C_BROADCAST @ yaw.T
    R_world = smpl_root_in_pitch_frame(X_180, R_world_to_cam)
    avatar_up_world = R_world @ np.array([0.0, 1.0, 0.0])
    assert avatar_up_world[2] > 0.99, avatar_up_world


@pytest.mark.unit
def test_foot_below_root_in_pitch():
    """Sanity: with body upright in pitch frame, applying R_world to
    the SMPL-local foot offset (0, -0.95, 0) should give a 0.95 m drop
    along pitch -z. Pins the relationship that ``hmr_world.py`` relies
    on for foot-anchored translation."""
    R_world = smpl_root_in_pitch_frame(X_180, R_W2C_BROADCAST)
    foot_in_root = np.array([0.0, -0.95, 0.0])  # SMPL canonical y-up
    foot_offset_world = R_world @ foot_in_root
    np.testing.assert_allclose(foot_offset_world, [0.0, 0.0, -0.95], atol=1e-9)


@pytest.mark.unit
def test_simplified_chain_is_just_R_w2c_T():
    """The chain has dropped the historical X_180 conjugation; the
    function is now a thin wrapper around ``R_w2c.T @ root_R_cam``."""
    R_w2c = R_W2C_BROADCAST
    R_arbitrary = X_180  # any 3x3 rotation will do
    expected = R_w2c.T @ R_arbitrary
    actual = smpl_root_in_pitch_frame(R_arbitrary, R_w2c)
    np.testing.assert_allclose(actual, expected, atol=1e-12)
