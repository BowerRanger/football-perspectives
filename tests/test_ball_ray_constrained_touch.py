"""C1: airborne player_touch must lie on the clicked-pixel ray."""
from __future__ import annotations

import numpy as np

from src.stages.ball import _project_point_onto_pixel_ray
from src.utils.camera_projection import project_world_to_image


def _cam():
    K = np.array([[1500.0, 0, 640.0], [0, 1500.0, 360.0], [0, 0, 1.0]])
    R = np.eye(3)
    t = np.array([0.0, 0.0, 60.0])  # camera 60 m back along -z looking +z
    return K, R, t


def test_projects_off_ray_bone_onto_clicked_ray():
    K, R, t = _cam()
    # A bone off to the side of where the user clicked.
    clicked_uv = (640.0, 300.0)
    bone = np.array([1.5, 2.0, 5.0])  # arbitrary off-ray 3D point
    out = _project_point_onto_pixel_ray(bone, clicked_uv, K, R, t, (0.0, 0.0))
    # Result must reproject to the clicked pixel (lateral -> 0).
    uvp = project_world_to_image(K, R, t, (0.0, 0.0), out.reshape(1, 3))[0]
    assert np.linalg.norm(uvp - np.array(clicked_uv)) < 0.5


def test_preserves_along_ray_depth_of_point():
    K, R, t = _cam()
    clicked_uv = (700.0, 380.0)
    bone = np.array([0.7, 1.0, 8.0])
    out = _project_point_onto_pixel_ray(bone, clicked_uv, K, R, t, (0.0, 0.0))
    C = -R.T @ t
    d = R.T @ (np.linalg.inv(K) @ np.array([clicked_uv[0], clicked_uv[1], 1.0]))
    d = d / np.linalg.norm(d)
    expected_depth = float(np.dot(bone - C, d))
    assert abs(float(np.dot(out - C, d)) - expected_depth) < 1e-6
