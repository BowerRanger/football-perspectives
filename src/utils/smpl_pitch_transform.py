"""GVHMR camera-frame SMPL root → pitch-world rotation.

GVHMR predicts ``root_R_cam`` in a **y-up** camera frame (matching its
internal SMPL canonical convention: x-right, y-up, z-toward-camera).
The pipeline's ``R_world_to_cam`` is in the standard **y-down OpenCV**
convention (x-right, y-down, z-into-scene).

To go from a SMPL canonical (y-up) body vector all the way to a pitch
z-up world vector we need to bridge the two camera conventions. They
differ by a 180° rotation around the camera +x axis — the matrix
``GVHMR_TO_OPENCV_CAM`` below. The full chain is:

    V_world  =  R_w2c.T  @  GVHMR_TO_OPENCV_CAM  @  root_R_cam  @  V_smpl

(i) ``root_R_cam`` rotates the SMPL canonical body vector into GVHMR's
    y-up camera frame.
(ii) ``GVHMR_TO_OPENCV_CAM`` re-expresses that vector in OpenCV y-down
     camera axes.
(iii) ``R_w2c.T`` carries it from camera to z-up world.

Note: the previous version of this module used the matrix
[[1,0,0],[0,0,-1],[0,1,0]] in slot (ii), which is a 90° rotation rather
than the 180° flip the conventions actually need. That misalignment
left an upright player rendering ~90° tipped over in pitch frame and
the foot-anchor projecting the pelvis below the pitch.
"""

from __future__ import annotations

import numpy as np


# Re-express a vector from GVHMR's y-up camera frame in OpenCV y-down
# camera axes. The two cameras share the +x axis (right) but their y
# and z axes are flipped (y-up vs y-down, z-toward-camera vs
# z-into-scene). A 180° rotation around the +x axis maps between them.
GVHMR_TO_OPENCV_CAM: np.ndarray = np.array(
    [[1,  0,  0],
     [0, -1,  0],
     [0,  0, -1]],
    dtype=float,
)

# Backwards-compatible alias. The old name was misleading — this matrix
# is not a SMPL-to-pitch axis remapping, it's a camera-frame y-up↔y-down
# bridge. Kept for any external imports.
SMPL_TO_PITCH_STATIC = GVHMR_TO_OPENCV_CAM


def smpl_root_in_pitch_frame(
    root_R_cam: np.ndarray,        # 3x3, GVHMR root rotation (y-up cam frame)
    R_world_to_cam: np.ndarray,    # 3x3, OpenCV extrinsic (world -> y-down cam)
) -> np.ndarray:
    """Lift GVHMR's camera-frame SMPL root rotation into pitch-world.

    See module docstring for the conventions. Returns a 3×3 rotation
    that maps a SMPL canonical (y-up) body vector to a z-up pitch-world
    vector — i.e. ``R @ (0,1,0)`` is the body's "up" direction in
    pitch metres for an upright player.
    """
    return R_world_to_cam.T @ GVHMR_TO_OPENCV_CAM @ root_R_cam
