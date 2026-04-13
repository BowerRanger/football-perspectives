"""Single-shot 3D reconstruction via foot-grounding + canonical body proportions.

Given 2D COCO pose keypoints and a known camera calibration for a single
view, reconstruct 3D world-space joint positions under the following
assumptions:

1. **Foot on pitch plane**: the player's foot is on the ground (the
   ``z = _ANKLE_HEIGHT`` plane).  Back-projecting the foot pixel through
   the camera gives a unique world-space foot position.

2. **Canonical joint heights**: each COCO joint sits at a fixed world
   height equal to ``body_height * relative_height[j]``, where the
   relative heights come from average human anatomy and ``body_height``
   defaults to 1.8 m.  Back-projecting each joint's pixel onto its own
   horizontal plane gives a per-joint ``(x, y)`` from a single linear
   solve — so limbs spread horizontally instead of collapsing onto a
   single vertical line.

3. **Plausibility clamp**: the back-projected ``(x, y)`` is rejected if
   it lands more than ``_MAX_HORIZONTAL_OFFSET`` metres from the foot
   anchor.  This catches numerical blow-ups (camera ray nearly
   tangential to the height plane) and joints whose true z is wildly
   different from the canonical estimate.

Works well for running, walking, and standing players — natural
horizontal spread on arms, legs, and head.  Produces biased positions
during jumps, raised arms, tackles, and diving saves; those frames
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

# Default body height used to scale canonical relative heights.
_DEFAULT_BODY_HEIGHT_M = 1.8

# Maximum horizontal distance (metres) a joint's back-projected (x, y)
# can sit from the foot anchor before we reject it as implausible.
# Limbs naturally extend ~0.8 m from the body axis; 1.5 m gives headroom
# for fully outstretched arms while still catching numerical blow-ups.
_MAX_HORIZONTAL_OFFSET_M = 1.5

# Canonical world height of each COCO 17 joint, expressed as a fraction
# of total body height (foot at 0.0, top of head at ~1.0).  Source:
# typical adult anatomy averaged across published anthropometric tables.
# Used to choose the back-projection plane for each joint when the
# camera only sees the player from one viewpoint.
_COCO_RELATIVE_HEIGHTS = (
    0.94,  # 0  nose
    0.96,  # 1  left_eye
    0.96,  # 2  right_eye
    0.93,  # 3  left_ear
    0.93,  # 4  right_ear
    0.82,  # 5  left_shoulder
    0.82,  # 6  right_shoulder
    0.62,  # 7  left_elbow
    0.62,  # 8  right_elbow
    0.45,  # 9  left_wrist
    0.45,  # 10 right_wrist
    0.53,  # 11 left_hip
    0.53,  # 12 right_hip
    0.28,  # 13 left_knee
    0.28,  # 14 right_knee
    0.05,  # 15 left_ankle
    0.05,  # 16 right_ankle
)


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
    # Plausibility: the foot anchor must land within a generous bounding
    # box around the FIFA pitch.  Camera rays nearly tangent to the
    # ground plane back-project to wildly off-pitch positions, and
    # we don't want those polluting the bird's-eye view.
    if not (-15.0 <= float(foot_world[0]) <= 120.0):
        return None
    if not (-15.0 <= float(foot_world[1]) <= 83.0):
        return None

    positions = np.full((_N_COCO_JOINTS, 3), np.nan, dtype=np.float32)
    confidences = np.zeros(_N_COCO_JOINTS, dtype=np.float32)

    foot_xy = np.array([foot_world[0], foot_world[1]], dtype=np.float64)
    max_offset_sq = _MAX_HORIZONTAL_OFFSET_M ** 2

    for j, kp in enumerate(keypoints):
        conf = float(kp.conf)
        if conf < min_joint_confidence:
            continue

        target_z = _DEFAULT_BODY_HEIGHT_M * _COCO_RELATIVE_HEIGHTS[j]
        try:
            xy = _backproject_pixel_to_plane(
                np.array([float(kp.x), float(kp.y)], dtype=np.float64),
                K, rvec, tvec,
                plane_z=target_z,
            )
        except np.linalg.LinAlgError:
            continue

        # Plausibility: the joint must sit within a human-sized cylinder
        # around the foot anchor.  Joints whose true z is far from the
        # canonical estimate (raised arms, jumps) end up with a large
        # horizontal back-projection error along the camera ray and get
        # filtered here.
        offset_sq = float((xy[0] - foot_xy[0]) ** 2 + (xy[1] - foot_xy[1]) ** 2)
        if offset_sq > max_offset_sq:
            continue

        positions[j] = [float(xy[0]), float(xy[1]), float(target_z)]
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


