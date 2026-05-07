"""Foot-anchored translation: compute pitch-frame root position from
ankle keypoint + camera (K, R, t).

Used to lock SMPL avatars to the ground plane: instead of trusting
GVHMR's monocular depth estimate (noisy, drifts), we project the visible
ankle pixel back to the pitch plane via the calibrated camera and place
the SMPL root such that the foot lands exactly there.
"""

from __future__ import annotations

import numpy as np

from src.utils.camera_projection import undistort_pixel


def ankle_ray_to_pitch(
    uv: np.ndarray | tuple[float, float],
    *,
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    plane_z: float = 0.05,
    distortion: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """Cast a ray from the camera centre through pixel (u, v) and intersect
    with the plane z = plane_z. Returns world-frame xyz.

    When the camera has non-zero radial ``distortion``, the input pixel is
    undistorted first so the linear ``K^-1`` back-projection lands on the
    correct ray. Skipping this step on a distorted lens biases foot
    positions by tens of centimetres at the far touchline.
    """
    uv = np.asarray(uv, dtype=float)
    if distortion != (0.0, 0.0):
        uv = undistort_pixel(uv, K, distortion)
    # Camera centre in world frame: C = -R^T t.
    C = -R.T @ t
    # Ray direction in world frame: d = R^T K^-1 (u, v, 1).
    pixel_h = np.array([uv[0], uv[1], 1.0])
    d_cam = np.linalg.inv(K) @ pixel_h
    d_world = R.T @ d_cam
    if abs(d_world[2]) < 1e-9:
        raise ValueError("ray parallel to ground plane")
    s = (plane_z - C[2]) / d_world[2]
    return C + s * d_world


def anchor_translation(
    foot_world: np.ndarray,
    foot_in_root: np.ndarray,
    R_root_world: np.ndarray,
) -> np.ndarray:
    """Given the world-frame foot position and the foot offset relative
    to the root in the root frame, return the world-frame root position.

    foot_world = root_t + R_root_world @ foot_in_root
    """
    return foot_world - R_root_world @ foot_in_root
