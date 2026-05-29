from __future__ import annotations

import numpy as np

from src.utils.virtual_cameras import intrinsics_from_fov, look_at_view


def test_intrinsics_from_fov_centres_principal_point() -> None:
    K = np.array(intrinsics_from_fov(90.0, (1920, 1080)))
    # 90° horizontal FOV over 1920 px → fx = 960.
    assert np.isclose(K[0, 0], 960.0)
    np.testing.assert_allclose([K[0, 2], K[1, 2]], [960.0, 540.0])
    assert K[2, 2] == 1.0


def test_look_at_view_is_proper_rotation_and_centres_camera() -> None:
    center = np.array([0.0, -5.0, 1.7])
    target = np.array([0.0, 0.0, 0.0])
    R, t = look_at_view(center, target)

    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-9)
    assert np.isclose(np.linalg.det(R), 1.0)
    # C = -R^T t recovers the requested centre.
    np.testing.assert_allclose(-R.T @ t, center, atol=1e-9)


def test_look_at_view_optical_axis_points_at_target() -> None:
    center = np.array([0.0, -5.0, 0.0])
    target = np.array([0.0, 0.0, 0.0])
    R, _ = look_at_view(center, target)
    # Camera +Z (row 2 of R) is the optical ray; should point center→target (+y).
    np.testing.assert_allclose(R[2], [0.0, 1.0, 0.0], atol=1e-9)


def test_look_at_view_handles_target_along_world_up() -> None:
    center = np.array([0.0, 0.0, 0.0])
    target = np.array([0.0, 0.0, 5.0])
    R, _ = look_at_view(center, target)
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-9)
    assert np.isclose(np.linalg.det(R), 1.0)
