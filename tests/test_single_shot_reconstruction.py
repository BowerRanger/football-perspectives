"""Unit tests for src/utils/single_shot_reconstruction.py."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.schemas.poses import Keypoint
from src.utils.single_shot_reconstruction import (
    _COCO_RELATIVE_HEIGHTS,
    _DEFAULT_BODY_HEIGHT_M,
    _N_COCO_JOINTS,
    reconstruct_player,
)


def _broadcast_camera() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Synthetic broadcast-style camera looking down at the pitch.

    Camera ~25 m above the near touchline, pointed at pitch centre.
    """
    K = np.array([[1500.0, 0.0, 960.0],
                  [0.0, 1500.0, 540.0],
                  [0.0, 0.0, 1.0]], dtype=np.float64)
    # Camera world position: (52.5, -20, 25), looking at (52.5, 34, 0).
    cam_pos = np.array([52.5, -20.0, 25.0], dtype=np.float64)
    target = np.array([52.5, 34.0, 0.0], dtype=np.float64)
    forward = target - cam_pos
    forward /= np.linalg.norm(forward)
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    # Camera frame: x=right, y=-up, z=forward (OpenCV convention).
    R = np.stack([right, -up, forward], axis=0)
    rvec, _ = cv2.Rodrigues(R)
    tvec = -R @ cam_pos
    return K, rvec.reshape(3), tvec.reshape(3)


def _project_world_to_pixel(
    pt: np.ndarray, K: np.ndarray, rvec: np.ndarray, tvec: np.ndarray,
) -> tuple[float, float]:
    R, _ = cv2.Rodrigues(rvec)
    cam = R @ pt + tvec
    px = K @ cam
    return float(px[0] / px[2]), float(px[1] / px[2])


def _make_upright_player(
    foot_xy: tuple[float, float],
    body_height: float = _DEFAULT_BODY_HEIGHT_M,
    shoulder_half_width: float = 0.22,
) -> list[np.ndarray]:
    """Build a synthetic upright-player skeleton in world coordinates.

    Returns a length-17 list of (3,) world-space points whose heights
    match ``_COCO_RELATIVE_HEIGHTS * body_height`` and whose horizontal
    spread is anatomically plausible.
    """
    fx, fy = foot_xy
    pts: list[np.ndarray] = [None] * _N_COCO_JOINTS  # type: ignore[list-item]

    def at(z_frac: float, dx: float = 0.0, dy: float = 0.0) -> np.ndarray:
        return np.array([fx + dx, fy + dy, body_height * z_frac], dtype=np.float64)

    pts[0] = at(_COCO_RELATIVE_HEIGHTS[0])                      # nose
    pts[1] = at(_COCO_RELATIVE_HEIGHTS[1], dx=-0.03)             # l eye
    pts[2] = at(_COCO_RELATIVE_HEIGHTS[2], dx=+0.03)             # r eye
    pts[3] = at(_COCO_RELATIVE_HEIGHTS[3], dx=-0.07)             # l ear
    pts[4] = at(_COCO_RELATIVE_HEIGHTS[4], dx=+0.07)             # r ear
    pts[5] = at(_COCO_RELATIVE_HEIGHTS[5], dx=-shoulder_half_width)
    pts[6] = at(_COCO_RELATIVE_HEIGHTS[6], dx=+shoulder_half_width)
    pts[7] = at(_COCO_RELATIVE_HEIGHTS[7], dx=-shoulder_half_width)
    pts[8] = at(_COCO_RELATIVE_HEIGHTS[8], dx=+shoulder_half_width)
    pts[9] = at(_COCO_RELATIVE_HEIGHTS[9], dx=-shoulder_half_width)
    pts[10] = at(_COCO_RELATIVE_HEIGHTS[10], dx=+shoulder_half_width)
    pts[11] = at(_COCO_RELATIVE_HEIGHTS[11], dx=-0.10)           # l hip
    pts[12] = at(_COCO_RELATIVE_HEIGHTS[12], dx=+0.10)           # r hip
    pts[13] = at(_COCO_RELATIVE_HEIGHTS[13], dx=-0.10)           # l knee
    pts[14] = at(_COCO_RELATIVE_HEIGHTS[14], dx=+0.10)           # r knee
    pts[15] = at(_COCO_RELATIVE_HEIGHTS[15], dx=-0.10)           # l ankle
    pts[16] = at(_COCO_RELATIVE_HEIGHTS[16], dx=+0.10)           # r ankle
    return pts


def _project_skeleton(
    world_pts: list[np.ndarray],
    K: np.ndarray, rvec: np.ndarray, tvec: np.ndarray,
    conf: float = 0.9,
) -> list[Keypoint]:
    out: list[Keypoint] = []
    for i, pt in enumerate(world_pts):
        u, v = _project_world_to_pixel(pt, K, rvec, tvec)
        out.append(Keypoint(name=f"j{i}", x=u, y=v, conf=conf))
    return out


class TestReconstructPlayer:
    def test_upright_synthetic_player_recovers_horizontal_spread(self):
        K, rvec, tvec = _broadcast_camera()
        foot_xy = (52.5, 34.0)  # pitch centre
        world_pts = _make_upright_player(foot_xy)
        keypoints = _project_skeleton(world_pts, K, rvec, tvec)

        result = reconstruct_player(keypoints, K, rvec, tvec)
        assert result is not None

        # Joints should NOT all share the same (x, y) — the bug we're
        # fixing collapsed every joint to the foot anchor.
        xs = result.positions[:, 0]
        ys = result.positions[:, 1]
        assert np.nanstd(xs) > 0.05, "joints have no horizontal x-spread"
        # In y the synthetic skeleton has no spread by construction;
        # x-spread alone proves the per-joint back-projection is active.

        # Shoulders should sit ~0.22 m apart in x (the configured width).
        l_shoulder = result.positions[5]
        r_shoulder = result.positions[6]
        shoulder_dx = abs(l_shoulder[0] - r_shoulder[0])
        assert 0.30 < shoulder_dx < 0.55, (
            f"recovered shoulder width {shoulder_dx:.3f} m not near 0.44"
        )

        # Joint heights should match canonical values within a few cm.
        for j in range(_N_COCO_JOINTS):
            expected_z = _DEFAULT_BODY_HEIGHT_M * _COCO_RELATIVE_HEIGHTS[j]
            assert abs(result.positions[j, 2] - expected_z) < 0.01

    def test_returns_none_when_both_ankles_low_confidence(self):
        K, rvec, tvec = _broadcast_camera()
        kps = [Keypoint(name=f"j{i}", x=960.0, y=540.0, conf=0.9)
               for i in range(_N_COCO_JOINTS)]
        kps[15] = Keypoint(name="l_ankle", x=960.0, y=540.0, conf=0.0)
        kps[16] = Keypoint(name="r_ankle", x=960.0, y=540.0, conf=0.0)
        result = reconstruct_player(kps, K, rvec, tvec)
        assert result is None

    def test_low_confidence_joint_left_nan(self):
        K, rvec, tvec = _broadcast_camera()
        world_pts = _make_upright_player((52.5, 34.0))
        keypoints = _project_skeleton(world_pts, K, rvec, tvec)
        # Drop confidence on the nose
        keypoints[0] = Keypoint(name="nose", x=keypoints[0].x, y=keypoints[0].y, conf=0.0)
        result = reconstruct_player(keypoints, K, rvec, tvec, min_joint_confidence=0.2)
        assert result is not None
        assert np.all(np.isnan(result.positions[0]))
        # Other joints still recovered
        assert not np.any(np.isnan(result.positions[5]))
