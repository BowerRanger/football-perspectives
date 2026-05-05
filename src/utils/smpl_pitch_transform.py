"""SMPL-world to pitch-world coordinate transform.

GVHMR's internal world: y-up, character facing -z (canonical SMPL).
Pitch world: z-up, x along nearside touchline, y toward far side.

The static transform aligns axes:
- SMPL +y (up) -> pitch +z
- SMPL +x (right) -> pitch +x
- SMPL -z (forward) -> pitch +y
"""

from __future__ import annotations

import numpy as np


SMPL_TO_PITCH_STATIC: np.ndarray = np.array(
    [[1, 0,  0],
     [0, 0, -1],
     [0, 1,  0]],
    dtype=float,
)


def smpl_root_in_pitch_frame(
    root_R_cam: np.ndarray,        # 3x3, root rotation in camera frame
    R_world_to_cam: np.ndarray,    # 3x3, OpenCV extrinsic (world -> camera)
) -> np.ndarray:
    """Express a SMPL root rotation given in the camera frame in pitch
    coordinates.

    The composition is:
        cam-frame root -> world-frame root via R_world_to_cam.T (camera -> world)
        SMPL canonical -> pitch via SMPL_TO_PITCH_STATIC

    Apply both to express the SMPL root rotation in the pitch frame.
    """
    return R_world_to_cam.T @ SMPL_TO_PITCH_STATIC @ root_R_cam
