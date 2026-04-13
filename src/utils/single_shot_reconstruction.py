"""Single-shot 3D reconstruction via foot-grounding + vertical body axis.

Given 2D COCO pose keypoints and a known camera calibration for a single
view, reconstruct 3D world-space joint positions under the following
assumptions:

1. **Foot on pitch plane**: the player's foot is on the ground (``z=0``
   pitch plane).  This gives a unique 3D foot position from the 2D foot
   pixel by back-projecting through the camera.

2. **Vertical body axis**: the player's body stands vertically above the
   foot position.  Every joint ``(x_j, y_j, z_j)`` lies on the line
   ``(x_foot, y_foot, z)`` for some ``z``.  This turns the depth-ambiguous
   single-view 3D problem into a 1D search per joint: find the ``z_j``
   that projects to the observed 2D pixel.

Works well for running, walking, and standing players.  Produces
biased heights during jumps, tackles, and diving saves — those frames
can be refined later via SMPL fitting's shape/pose prior.

This is the fallback path used when multi-view triangulation isn't
available (only one calibrated shot has the player at a given frame).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

from src.schemas.poses import Keypoint
from src.utils.camera import project_to_pitch

logger = logging.getLogger(__name__)

_N_COCO_JOINTS = 17

# COCO keypoint indices — match the canonical order in
# ``src/schemas/poses.py::COCO_KEYPOINT_NAMES``.
_NOSE = 0
_LEFT_SHOULDER = 5
_RIGHT_SHOULDER = 6
_LEFT_HIP = 11
_RIGHT_HIP = 12
_LEFT_ANKLE = 15
_RIGHT_ANKLE = 16

# Anatomical height of the ankle keypoint above the ground plane (metres).
# The COCO ankle keypoint tracks the ankle *joint*, not the sole of the foot,
# so projecting its pixel onto z=0 introduces a systematic offset in the
# direction of the camera ray.  We back-project onto this plane instead.
_ANKLE_HEIGHT = 0.08


@dataclass(frozen=True)
class SingleShotResult:
    """3D joint positions + per-joint confidence for one frame."""

    positions: np.ndarray    # (17, 3) world coords (x, y, z); NaN for unsolved joints
    confidences: np.ndarray  # (17,) per-joint confidence, 0..1
    foot_position: np.ndarray  # (3,) the grounding foot world position


def reconstruct_player(
    keypoints: list[Keypoint],
    K: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    min_foot_confidence: float = 0.3,
    min_joint_confidence: float = 0.2,
) -> SingleShotResult | None:
    """Reconstruct 3D joint positions for a single player at one frame.

    Args:
        keypoints: COCO 17 keypoints in pixel coordinates with confidences.
        K, rvec, tvec: camera calibration (intrinsic + extrinsic).
        min_foot_confidence: reject the reconstruction if both ankles have
            confidence below this threshold.
        min_joint_confidence: joints with confidence below this threshold
            are left as NaN in the output (don't contribute to the 3D pose).

    Returns:
        :class:`SingleShotResult` or ``None`` when both ankles are
        below ``min_foot_confidence`` (no reliable grounding point).
    """
    if len(keypoints) != _N_COCO_JOINTS:
        logger.debug(
            "reconstruct_player: expected %d keypoints, got %d",
            _N_COCO_JOINTS, len(keypoints),
        )
        return None

    K = np.asarray(K, dtype=np.float64)
    rvec = np.asarray(rvec, dtype=np.float64).reshape(3)
    tvec = np.asarray(tvec, dtype=np.float64).reshape(3)

    foot_world = _compute_foot_world_position(
        keypoints, K, rvec, tvec, min_foot_confidence,
    )
    if foot_world is None:
        return None

    positions = np.full((_N_COCO_JOINTS, 3), np.nan, dtype=np.float32)
    confidences = np.zeros(_N_COCO_JOINTS, dtype=np.float32)

    # Pre-compute the camera extrinsic projection matrix rows for the
    # 1D solve.  The camera matrix is [R | t] — rows 0/1/2 give the
    # numerator and denominator of the perspective divide.
    R, _ = cv2.Rodrigues(rvec)
    Rt = np.hstack([R, tvec.reshape(3, 1)])  # (3, 4)

    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])

    for j, kp in enumerate(keypoints):
        conf = float(kp.conf)
        if conf < min_joint_confidence:
            continue

        z_j = _solve_joint_z(
            u=float(kp.x), v=float(kp.y),
            x_foot=float(foot_world[0]), y_foot=float(foot_world[1]),
            Rt=Rt, fx=fx, fy=fy, cx=cx, cy=cy,
        )
        if z_j is None:
            continue
        # Clamp z to a sane human range [0, 2.5 m] to reject numerical
        # blow-ups from near-degenerate projections.
        if z_j < -0.3 or z_j > 2.8:
            continue
        positions[j] = [foot_world[0], foot_world[1], max(0.0, float(z_j))]
        confidences[j] = conf

    return SingleShotResult(
        positions=positions,
        confidences=confidences,
        foot_position=np.array(foot_world, dtype=np.float64),
    )


def _compute_foot_world_position(
    keypoints: list[Keypoint],
    K: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    min_foot_confidence: float,
) -> np.ndarray | None:
    """Project the player's foot midpoint onto the pitch plane.

    Uses the midpoint of the two ankles when both are confident; falls
    back to whichever ankle is above ``min_foot_confidence``.  Returns
    world coordinates ``(x, y, 0)`` or ``None`` if neither ankle is
    reliable.
    """
    left = keypoints[_LEFT_ANKLE]
    right = keypoints[_RIGHT_ANKLE]
    left_ok = float(left.conf) >= min_foot_confidence
    right_ok = float(right.conf) >= min_foot_confidence

    if not left_ok and not right_ok:
        return None

    if left_ok and right_ok:
        u = (float(left.x) + float(right.x)) / 2.0
        v = (float(left.y) + float(right.y)) / 2.0
    elif left_ok:
        u = float(left.x)
        v = float(left.y)
    else:
        u = float(right.x)
        v = float(right.y)

    # Back-project the ankle pixel onto the horizontal plane at
    # ``z = _ANKLE_HEIGHT``.  project_to_pitch only supports z=0, so we
    # do the plane-ray intersection manually.
    pixel = np.array([u, v], dtype=np.float64)
    try:
        xy = _backproject_pixel_to_plane(
            pixel,
            K.astype(np.float64),
            np.asarray(rvec, dtype=np.float64),
            np.asarray(tvec, dtype=np.float64),
            plane_z=_ANKLE_HEIGHT,
        )
    except np.linalg.LinAlgError:
        return None
    return np.array([float(xy[0]), float(xy[1]), _ANKLE_HEIGHT], dtype=np.float64)


def _backproject_pixel_to_plane(
    pixel: np.ndarray,
    K: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    plane_z: float,
) -> np.ndarray:
    """Intersect the camera ray through ``pixel`` with the plane ``z = plane_z``.

    Returns the ``(x, y)`` world coordinates at that plane, or raises
    ``numpy.linalg.LinAlgError`` if the ray is parallel to the plane.
    """
    R, _ = cv2.Rodrigues(rvec)
    tvec = tvec.reshape(3)

    # Camera world position C = -R^T @ t
    cam_world = -R.T @ tvec

    # Ray direction in world coords
    pixel_h = np.array([pixel[0], pixel[1], 1.0], dtype=np.float64)
    direction_camera = np.linalg.inv(K) @ pixel_h
    direction_world = R.T @ direction_camera

    # Plane intersection: C + s * d hits plane_z at s = (plane_z - C_z) / d_z
    dz = float(direction_world[2])
    if abs(dz) < 1e-9:
        raise np.linalg.LinAlgError("Camera ray parallel to plane")
    s = (float(plane_z) - float(cam_world[2])) / dz
    xy = cam_world[:2] + s * direction_world[:2]
    return xy


def _solve_joint_z(
    *,
    u: float,
    v: float,
    x_foot: float,
    y_foot: float,
    Rt: np.ndarray,
    fx: float, fy: float, cx: float, cy: float,
) -> float | None:
    """Find the z that projects ``(x_foot, y_foot, z)`` onto the pixel ``(u, v)``.

    The projection is ``pixel = K @ (R @ world + t)``.  With ``(x, y)``
    fixed at the foot position, only ``z`` is unknown.  Both ``u`` and
    ``v`` give a linear equation in ``z``; we solve each and average.
    Returns ``None`` if the equations are degenerate (e.g., camera-ray
    parallel to the vertical axis).
    """
    r1 = Rt[0]
    r2 = Rt[1]
    r3 = Rt[2]

    world_base = np.array([x_foot, y_foot, 0.0, 1.0], dtype=np.float64)
    A0 = float(r1 @ world_base)
    B0 = float(r2 @ world_base)
    C0 = float(r3 @ world_base)

    # z coefficients along each row (contribution of the z term to each
    # of the three [R | t] @ [x, y, z, 1] components).
    A1 = float(r1[2])
    B1 = float(r2[2])
    C1 = float(r3[2])

    # From u equation: fx * (A0 + A1*z) + cx * (C0 + C1*z) = u * (C0 + C1*z)
    #                  fx*A1*z + cx*C1*z - u*C1*z = u*C0 - fx*A0 - cx*C0
    #                  z * (fx*A1 + (cx - u)*C1) = (u - cx)*C0 - fx*A0
    denom_u = fx * A1 + (cx - u) * C1
    numer_u = (u - cx) * C0 - fx * A0
    # From v equation: analogous with fy, cy, B
    denom_v = fy * B1 + (cy - v) * C1
    numer_v = (v - cy) * C0 - fy * B0

    # Accept whichever equations are non-degenerate; average if both.
    z_vals: list[float] = []
    if abs(denom_u) > 1e-6:
        z_vals.append(numer_u / denom_u)
    if abs(denom_v) > 1e-6:
        z_vals.append(numer_v / denom_v)
    if not z_vals:
        return None
    return float(np.mean(z_vals))
