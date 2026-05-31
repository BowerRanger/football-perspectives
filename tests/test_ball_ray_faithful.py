"""C4: snap an off-ray anchored frame onto the clicked ray, keep depth."""
from __future__ import annotations

import numpy as np

from src.stages.ball import _snap_world_onto_pixel_ray
from src.utils.camera_projection import project_world_to_image


def _cam():
    K = np.array([[1500.0, 0, 640.0], [0, 1500.0, 360.0], [0, 0, 1.0]])
    R = np.eye(3)
    t = np.array([0.0, 0.0, 50.0])
    return K, R, t


def test_off_ray_point_snaps_onto_ray():
    K, R, t = _cam()
    clicked = (700.0, 300.0)
    world = np.array([2.0, -1.0, 6.0])  # off the clicked ray
    snapped = _snap_world_onto_pixel_ray(world, clicked, K, R, t, (0.0, 0.0))
    uvp = project_world_to_image(K, R, t, (0.0, 0.0), snapped.reshape(1, 3))[0]
    assert np.linalg.norm(uvp - np.array(clicked)) < 0.5


def test_on_ray_point_unchanged():
    K, R, t = _cam()
    clicked = (640.0, 360.0)
    C = -R.T @ t
    d = R.T @ (np.linalg.inv(K) @ np.array([clicked[0], clicked[1], 1.0]))
    d = d / np.linalg.norm(d)
    world = C + 7.0 * d  # already on the ray
    snapped = _snap_world_onto_pixel_ray(world, clicked, K, R, t, (0.0, 0.0))
    assert np.linalg.norm(snapped - world) < 1e-6
