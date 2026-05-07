"""Centralised world<->image projection with optional radial distortion.

Single source of truth for ``(K, R, t, distortion) → image`` math across the
codebase. Avoids subtle bugs from a dozen call sites each reimplementing the
same pinhole math, only some of which honour distortion.

The distortion model is OpenCV's 2-coefficient radial:
``x' = x · (1 + k1·r² + k2·r⁴)`` where ``(x, y) = ((u-cx)/fx, (v-cy)/fy)``
and ``r² = x² + y²``. Tangential ``(p1, p2)`` is omitted — broadcast lenses
are dominantly radial and adding tangential terms risks overfit on a sparse
anchor set.
"""

from __future__ import annotations

import cv2
import numpy as np


def project_world_to_image(
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    distortion: tuple[float, float],
    world_points: np.ndarray,
) -> np.ndarray:
    """Project world-frame 3D points to image-plane 2D pixels.

    Args:
        K: 3x3 intrinsic matrix.
        R: 3x3 world->camera rotation.
        t: (3,) world->camera translation.
        distortion: (k1, k2) radial distortion. Use (0, 0) for pinhole.
        world_points: (N, 3) array of world-frame points.

    Returns:
        (N, 2) image-plane projections (u, v) in pixels.
    """
    pts = np.asarray(world_points, dtype=np.float64).reshape(-1, 1, 3)
    rvec, _ = cv2.Rodrigues(np.asarray(R, dtype=np.float64))
    tvec = np.asarray(t, dtype=np.float64).reshape(3, 1)
    k1, k2 = distortion
    dist = np.array([k1, k2, 0.0, 0.0, 0.0], dtype=np.float64)
    out, _ = cv2.projectPoints(pts, rvec, tvec, K.astype(np.float64), dist)
    return out.reshape(-1, 2)


def undistort_pixel(
    pixel_uv: tuple[float, float] | np.ndarray,
    K: np.ndarray,
    distortion: tuple[float, float],
) -> np.ndarray:
    """Map a distorted pixel to the linear-pinhole equivalent.

    Used by downstream stages (foot anchoring, ball ground-projection) that
    back-project a 2D pixel into a 3D ray and need to undo the lens
    distortion before applying the inverse pinhole.

    Returns a (2,) numpy array (u_lin, v_lin).
    """
    uv = np.asarray(pixel_uv, dtype=np.float64).reshape(1, 1, 2)
    k1, k2 = distortion
    dist = np.array([k1, k2, 0.0, 0.0, 0.0], dtype=np.float64)
    # ``P=K`` returns coordinates in pixel space rather than the normalised
    # camera frame — the natural form for downstream callers that go on to
    # apply the inverse pinhole.
    undist = cv2.undistortPoints(
        uv, K.astype(np.float64), dist, P=K.astype(np.float64),
    )
    return undist.reshape(2)
