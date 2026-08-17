"""Unit tests for src/utils/ball_eval.py — the sub-20cm campaign harness.

All tests use a synthetic pinhole camera; no clip data required.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.schemas.ball_anchor import BallAnchor

pytestmark = pytest.mark.unit


def _cam():
    """Simple pinhole: camera at (0,-20,10), looking at the pitch origin."""
    K = np.array([[1500.0, 0, 960], [0, 1500.0, 540], [0, 0, 1.0]])
    fwd = np.array([0.0, 20.0, -10.0])
    fwd /= np.linalg.norm(fwd)
    up = np.array([0.0, 0.0, 1.0])
    right = np.cross(fwd, up)
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    R = np.stack([right, down, fwd])  # world→cam rows
    C = np.array([0.0, -20.0, 10.0])
    t = -R @ C
    return K, R, t


def test_pixel_ray_roundtrip():
    from src.utils.ball_eval import pixel_ray, point_ray_distance
    from src.utils.camera_projection import project_world_to_image

    K, R, t = _cam()
    P = np.array([2.0, 5.0, 0.11])
    uv = project_world_to_image(K, R, t, (0.0, 0.0), P.reshape(1, 3))[0]
    C, d = pixel_ray(uv, K, R, t)
    perp, along = point_ray_distance(P, C, d)
    assert perp < 1e-6
    assert along > 0


def test_ray_plane_z_recovers_ground_point():
    from src.utils.ball_eval import pixel_ray, ray_plane_z
    from src.utils.camera_projection import project_world_to_image

    K, R, t = _cam()
    P = np.array([-3.0, 12.0, 0.11])
    uv = project_world_to_image(K, R, t, (0.0, 0.0), P.reshape(1, 3))[0]
    C, d = pixel_ray(uv, K, R, t)
    X = ray_plane_z(C, d, 0.11)
    assert X is not None
    assert np.allclose(X, P, atol=1e-6)


def test_ray_plane_z_none_when_parallel():
    from src.utils.ball_eval import ray_plane_z

    X = ray_plane_z(np.array([0.0, 0.0, 5.0]), np.array([0.0, 1.0, 0.0]), 0.11)
    assert X is None


def test_anchor_gt_world_ground_exact():
    from src.utils.ball_eval import anchor_gt_world
    from src.utils.camera_projection import project_world_to_image

    K, R, t = _cam()
    P = np.array([4.0, 8.0, 0.11])
    uv = project_world_to_image(K, R, t, (0.0, 0.0), P.reshape(1, 3))[0]
    anc = BallAnchor(frame=5, state="grounded",
                     image_xy=(float(uv[0]), float(uv[1])))
    gt, kind = anchor_gt_world(anc, K, R, t, (0.0, 0.0), ball_radius=0.11)
    assert kind == "ground_exact"
    assert np.allclose(gt, P, atol=1e-6)


def test_anchor_gt_world_joint_depth_projects_joint_onto_ray():
    from src.utils.ball_eval import (anchor_gt_world, pixel_ray,
                                     point_ray_distance)
    from src.utils.camera_projection import project_world_to_image

    K, R, t = _cam()
    true_ball = np.array([1.0, 6.0, 0.3])
    uv = project_world_to_image(K, R, t, (0.0, 0.0),
                                true_ball.reshape(1, 3))[0]
    joint = true_ball + np.array([0.05, 0.4, 0.05])  # FK drift off-ray
    anc = BallAnchor(frame=5, state="player_touch",
                     image_xy=(float(uv[0]), float(uv[1])),
                     player_id="P001", bone="r_foot")
    gt, kind = anchor_gt_world(anc, K, R, t, (0.0, 0.0), ball_radius=0.11,
                               joint_world=tuple(joint))
    assert kind == "joint_depth"
    C, d = pixel_ray(uv, K, R, t)
    perp, _ = point_ray_distance(np.asarray(gt), C, d)
    assert perp < 1e-9  # GT lies on the clicked ray
    assert np.linalg.norm(gt - true_ball) < 0.45  # depth ≈ joint depth


def test_anchor_gt_world_airborne_is_ray_only_and_no_pixel_is_none():
    from src.utils.ball_eval import anchor_gt_world

    K, R, t = _cam()
    anc = BallAnchor(frame=5, state="airborne_low", image_xy=(900.0, 400.0))
    gt, kind = anchor_gt_world(anc, K, R, t, (0.0, 0.0), ball_radius=0.11)
    assert (gt, kind) == (None, "ray_only")
    anc2 = BallAnchor(frame=6, state="off_screen_flight", image_xy=None)
    assert anchor_gt_world(anc2, K, R, t, (0.0, 0.0),
                           ball_radius=0.11) == (None, "none")
