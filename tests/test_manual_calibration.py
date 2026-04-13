"""Unit tests for src/utils/manual_calibration.py."""

from __future__ import annotations

import cv2
import numpy as np

from src.utils.manual_calibration import solve_from_annotations
from src.utils.pitch import FIFA_LANDMARKS


def _build_synthetic_camera(
    cam_pos: tuple[float, float, float] = (52.5, -25.0, 28.0),
    target: tuple[float, float, float] = (52.5, 34.0, 0.0),
    fx: float = 3500.0,
    image_size: tuple[int, int] = (1920, 1080),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Synthetic pinhole camera looking at the pitch."""
    cx = image_size[0] / 2.0
    cy = image_size[1] / 2.0
    K = np.array([[fx, 0.0, cx], [0.0, fx, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    cam = np.array(cam_pos, dtype=np.float64)
    tgt = np.array(target, dtype=np.float64)
    forward = tgt - cam
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    R = np.stack([right, -up, forward], axis=0)
    rvec, _ = cv2.Rodrigues(R)
    tvec = -R @ cam
    return K, rvec.reshape(3), tvec.reshape(3)


def _project(world_pt: np.ndarray, K, rvec, tvec) -> tuple[float, float]:
    proj, _ = cv2.projectPoints(
        world_pt.reshape(1, 3).astype(np.float64),
        rvec, tvec, K, None,
    )
    return float(proj[0, 0, 0]), float(proj[0, 0, 1])


class TestSolveFromAnnotations:
    def test_recovers_synthetic_camera_with_8_landmarks(self):
        K, rvec, tvec = _build_synthetic_camera()
        # Pick 8 well-distributed FIFA landmarks
        landmark_names = [
            "corner_near_left", "corner_near_right",
            "corner_far_left", "corner_far_right",
            "halfway_near", "halfway_far",
            "left_18yard_near_right", "right_18yard_near_left",
        ]
        annotations = {}
        for name in landmark_names:
            world = FIFA_LANDMARKS[name]
            annotations[name] = list(_project(world, K, rvec, tvec))

        result = solve_from_annotations(
            annotations, image_size=(1920, 1080), fx_init=3000.0,
        )
        assert result is not None
        assert result.n_points == 8
        assert result.mean_reprojection_error_px < 1.0
        # fx should converge near the true 3500 (8 points, full mode)
        K_recovered = np.array(result.camera_frame.intrinsic_matrix)
        assert abs(K_recovered[0, 0] - 3500.0) < 50.0
        assert result.mode == "full"

    def test_pose_only_with_4_landmarks(self):
        # 4 points → can't refine fx; stays at fx_init
        K, rvec, tvec = _build_synthetic_camera()
        landmark_names = [
            "corner_near_left", "corner_near_right",
            "corner_far_left", "corner_far_right",
        ]
        annotations = {}
        for name in landmark_names:
            world = FIFA_LANDMARKS[name]
            annotations[name] = list(_project(world, K, rvec, tvec))
        result = solve_from_annotations(
            annotations, image_size=(1920, 1080), fx_init=3500.0,
        )
        assert result is not None
        assert result.n_points == 4
        # Pose-only mode keeps fx fixed at the seed, so any meaningful
        # error metric here is dominated by the fact that fx is correct
        # (we built the synthetic data with fx=3500 = fx_init).
        assert result.mode == "pose_only"
        assert result.mean_reprojection_error_px < 1.0

    def test_returns_none_with_too_few_landmarks(self):
        result = solve_from_annotations(
            {"corner_near_left": [100, 200]},
            image_size=(1920, 1080),
        )
        assert result is None

    def test_skips_unknown_landmark_names(self):
        K, rvec, tvec = _build_synthetic_camera()
        annotations = {}
        # 4 valid + 1 garbage
        for name in ["corner_near_left", "corner_near_right",
                     "corner_far_left", "corner_far_right"]:
            world = FIFA_LANDMARKS[name]
            annotations[name] = list(_project(world, K, rvec, tvec))
        annotations["bogus_landmark"] = [500, 500]
        result = solve_from_annotations(
            annotations, image_size=(1920, 1080), fx_init=3500.0,
        )
        assert result is not None
        assert result.n_points == 4

    def test_skips_malformed_pixel_values(self):
        K, rvec, tvec = _build_synthetic_camera()
        annotations = {}
        for name in ["corner_near_left", "corner_near_right",
                     "corner_far_left", "corner_far_right"]:
            world = FIFA_LANDMARKS[name]
            annotations[name] = list(_project(world, K, rvec, tvec))
        annotations["halfway_near"] = "not a coordinate"
        annotations["halfway_far"] = [100]  # wrong length
        result = solve_from_annotations(
            annotations, image_size=(1920, 1080), fx_init=3500.0,
        )
        assert result is not None
        assert result.n_points == 4
