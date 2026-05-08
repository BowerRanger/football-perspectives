"""GVHMR per-frame SMPL root → pitch-world rotation.

GVHMR's ``smpl_params_incam`` exposes a ``global_orient`` whose rotation
matrix maps a SMPL canonical (y-up) body vector directly to the OpenCV
camera frame (y-down, z-into-scene). The GVHMR demo confirms this — see
``third_party/gvhmr/tools/demo/demo.py:215``, where the demo feeds
``smpl_params_incam`` into SMPL and renders the resulting verts directly
in the camera view with no intermediate frame conversion.

To lift that into the pipeline's pitch z-up world we just transpose
``R_w2c``:

    V_world  =  R_w2c.T  @  root_R_cam  @  V_smpl

(i) ``root_R_cam`` rotates the SMPL canonical body vector into OpenCV
    camera coords.
(ii) ``R_w2c.T`` carries that vector from camera to pitch z-up world.

History note: an earlier version of this module bridged ``root_R_cam``
through a 180° rotation around +x ("GVHMR_TO_OPENCV_CAM") on the
assumption that ``global_orient`` was in GVHMR's y-up cam frame, but
that source was actually ``smpl_params_global`` (in the gravity-view
``ay`` world frame), not the camera frame. The bridge was both wrong-
matrix and wrong-source. Switching to ``smpl_params_incam`` makes the
chain a clean ``R_w2c.T @ root_R_cam`` with no implicit handedness flip.
"""

from __future__ import annotations

import numpy as np


def smpl_root_in_pitch_frame(
    root_R_cam: np.ndarray,        # 3x3, GVHMR root rotation (OpenCV cam frame)
    R_world_to_cam: np.ndarray,    # 3x3, OpenCV extrinsic (world -> y-down cam)
) -> np.ndarray:
    """Lift GVHMR's camera-frame SMPL root rotation into pitch-world.

    See module docstring for the conventions. Returns a 3×3 rotation
    that maps a SMPL canonical (y-up) body vector to a z-up pitch-world
    vector — i.e. ``R @ (0, 1, 0)`` is the body's "up" direction in
    pitch metres for an upright player.
    """
    return R_world_to_cam.T @ root_R_cam
