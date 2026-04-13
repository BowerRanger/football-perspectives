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


class TestFixedPositionSolve:
    def test_clustered_3_landmark_recovery(self):
        """3 landmarks all in one corner of the pitch must recover
        the pose accurately when the camera position is known."""
        K, rvec, tvec = _build_synthetic_camera(fx=3500.0)
        cam_pos = -cv2.Rodrigues(rvec)[0].T @ tvec
        # 3 clustered landmarks all near the left 18-yard box
        names = [
            "left_18yard_far_right", "left_6yard_far_right",
            "left_penalty_spot",
        ]
        annotations = {}
        for name in names:
            world = FIFA_LANDMARKS[name]
            annotations[name] = list(_project(world, K, rvec, tvec))
        result = solve_from_annotations(
            annotations, image_size=(1920, 1080),
            fx_init=3000.0,  # deliberately wrong — test fx refinement
            camera_position_world=cam_pos,
        )
        assert result is not None, "3-point fixed solve should succeed"
        assert result.mode == "fixed_position"
        assert result.n_points == 3
        # Rotation should be recovered to within ~0.5° on noise-free input
        recovered_rvec = np.array(result.camera_frame.rotation_vector)
        delta_angle = float(np.linalg.norm(recovered_rvec - rvec))
        assert delta_angle < np.deg2rad(0.5), f"delta_angle={np.rad2deg(delta_angle):.2f}°"
        # fx should converge to truth (3500) from the wrong seed (3000)
        recovered_fx = float(np.array(result.camera_frame.intrinsic_matrix)[0, 0])
        assert abs(recovered_fx - 3500.0) < 35.0, f"fx={recovered_fx}"

    def test_fixed_position_beats_free_on_clustered_landmarks(self):
        """With clustered annotations, the fixed-position path must
        project *distant* unclicked landmarks much closer to their
        true pixel positions than the free-position path."""
        K, rvec, tvec = _build_synthetic_camera(fx=3500.0)
        cam_pos = -cv2.Rodrigues(rvec)[0].T @ tvec
        names = [
            "left_18yard_far_right", "left_6yard_far_right",
            "left_penalty_spot", "left_18yard_d_far",
            "left_goal_far_post_base",
        ]
        annotations = {}
        for name in names:
            world = FIFA_LANDMARKS[name]
            annotations[name] = list(_project(world, K, rvec, tvec))

        # Free-position solve
        free = solve_from_annotations(
            annotations, image_size=(1920, 1080), fx_init=3500.0,
        )
        assert free is not None
        assert free.mode in ("pose_only", "full")

        # Fixed-position solve
        fixed = solve_from_annotations(
            annotations, image_size=(1920, 1080), fx_init=3500.0,
            camera_position_world=cam_pos,
        )
        assert fixed is not None
        assert fixed.mode == "fixed_position"

        # Project a distant landmark (far right corner) through each
        # recovered calibration and measure the error vs truth.
        distant = np.asarray(FIFA_LANDMARKS["corner_far_right"], dtype=np.float64)
        truth_px = np.array(_project(distant, K, rvec, tvec))

        def recovered_px(cf_result):
            K_r = np.array(cf_result.camera_frame.intrinsic_matrix, dtype=np.float64)
            rvec_r = np.asarray(cf_result.camera_frame.rotation_vector, dtype=np.float64)
            tvec_r = np.asarray(cf_result.camera_frame.translation_vector, dtype=np.float64)
            proj, _ = cv2.projectPoints(
                distant.reshape(1, 3), rvec_r, tvec_r, K_r, None,
            )
            return proj.reshape(2)

        free_err = float(np.linalg.norm(recovered_px(free) - truth_px))
        fixed_err = float(np.linalg.norm(recovered_px(fixed) - truth_px))
        # Fixed should be *at least* as good as free on this test.
        # In practice the free path overfits the cluster and the
        # fixed path stays near-zero.
        assert fixed_err <= free_err + 1e-6, (
            f"fixed_err={fixed_err:.2f}px must be ≤ free_err={free_err:.2f}px"
        )
        # And fixed should be essentially sub-pixel
        assert fixed_err < 2.0, f"fixed_err={fixed_err:.2f}px"

    def test_3_landmarks_rejected_without_known_position(self):
        """3 landmarks alone (no camera position) must return None —
        the free-position path still needs ≥4."""
        K, rvec, tvec = _build_synthetic_camera()
        annotations = {}
        for name in ["corner_near_left", "corner_near_right", "corner_far_left"]:
            world = FIFA_LANDMARKS[name]
            annotations[name] = list(_project(world, K, rvec, tvec))
        result = solve_from_annotations(
            annotations, image_size=(1920, 1080), fx_init=3500.0,
        )
        assert result is None

    def test_fewer_than_3_rejected_even_with_position(self):
        K, rvec, tvec = _build_synthetic_camera()
        cam_pos = -cv2.Rodrigues(rvec)[0].T @ tvec
        annotations = {}
        for name in ["corner_near_left", "corner_near_right"]:
            world = FIFA_LANDMARKS[name]
            annotations[name] = list(_project(world, K, rvec, tvec))
        result = solve_from_annotations(
            annotations, image_size=(1920, 1080),
            camera_position_world=cam_pos,
        )
        assert result is None
