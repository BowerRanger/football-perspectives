"""Unit tests for ``src.utils.goal_geometry``: pixel-ray resolution
of goal-impact ball anchors onto known goal-frame primitives.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.utils.goal_geometry import (
    GoalGeometry,
    resolve_goal_impact_world,
)


# Camera looking down the pitch toward the near goal (x=0). Centre at
# world (-10, 34, 1.22). Forward = +x. Up = +z. Right = +y. This means
# the goal mouth fills roughly the centre of the image and a pixel
# offset from cx maps cleanly to a y offset in pitch metres.
_K = np.array([[1500.0, 0.0, 640.0],
               [0.0, 1500.0, 360.0],
               [0.0, 0.0, 1.0]])
_R = np.array([[0.0, 1.0, 0.0],
               [0.0, 0.0, -1.0],
               [1.0, 0.0, 0.0]])
_C = np.array([-10.0, 34.0, 1.22])
_t = -_R @ _C  # world->camera translation
_DISTORTION = (0.0, 0.0)


def _geometry() -> GoalGeometry:
    return GoalGeometry.from_pitch_config({
        "length_m": 105.0,
        "width_m": 68.0,
        "goal_height_m": 2.44,
        "goal_width_m": 7.32,
        "goal_depth_m": 1.5,
    })


def _project_world_to_pixel(world: np.ndarray) -> tuple[float, float]:
    """Reproject a world point through (_R, _t, _K) for synthetic test pixels."""
    cam = _R @ world + _t
    u = cam[0] * 1500.0 / cam[2] + 640.0
    v = cam[1] * 1500.0 / cam[2] + 360.0
    return float(u), float(v)


def test_geometry_from_pitch_config_uses_defaults_for_missing_goal_dims():
    g = GoalGeometry.from_pitch_config({
        "length_m": 105.0,
        "width_m": 68.0,
        "goal_height_m": 2.44,
    })
    assert g.goal_line_x_near == 0.0
    assert g.goal_line_x_far == 105.0
    assert g.post_y_left == pytest.approx(34.0 - 7.32 / 2)
    assert g.post_y_right == pytest.approx(34.0 + 7.32 / 2)
    assert g.crossbar_z == 2.44
    assert g.net_depth == 1.5


def test_resolve_crossbar_returns_point_on_crossbar():
    geom = _geometry()
    target = np.array([0.0, 34.0, geom.crossbar_z])
    uv = _project_world_to_pixel(target)
    world = resolve_goal_impact_world(
        uv, "crossbar",
        K=_K, R=_R, t=_t, distortion=_DISTORTION, geometry=geom,
    )
    np.testing.assert_allclose(world, target, atol=0.05)


def test_resolve_post_returns_point_on_left_post():
    geom = _geometry()
    target = np.array([0.0, geom.post_y_left, 1.0])
    uv = _project_world_to_pixel(target)
    world = resolve_goal_impact_world(
        uv, "post",
        K=_K, R=_R, t=_t, distortion=_DISTORTION, geometry=geom,
    )
    np.testing.assert_allclose(world, target, atol=0.05)


def test_resolve_post_returns_point_on_right_post():
    geom = _geometry()
    target = np.array([0.0, geom.post_y_right, 1.5])
    uv = _project_world_to_pixel(target)
    world = resolve_goal_impact_world(
        uv, "post",
        K=_K, R=_R, t=_t, distortion=_DISTORTION, geometry=geom,
    )
    np.testing.assert_allclose(world, target, atol=0.05)


def test_resolve_back_net_returns_point_behind_near_goal():
    geom = _geometry()
    target = np.array([-geom.net_depth, 34.0, 1.22])
    uv = _project_world_to_pixel(target)
    world = resolve_goal_impact_world(
        uv, "back_net",
        K=_K, R=_R, t=_t, distortion=_DISTORTION, geometry=geom,
    )
    np.testing.assert_allclose(world, target, atol=0.05)


def test_resolve_side_net_returns_point_on_left_side():
    geom = _geometry()
    target = np.array([-0.5, geom.post_y_left, 1.22])
    uv = _project_world_to_pixel(target)
    world = resolve_goal_impact_world(
        uv, "side_net",
        K=_K, R=_R, t=_t, distortion=_DISTORTION, geometry=geom,
    )
    np.testing.assert_allclose(world, target, atol=0.05)


def test_resolve_picks_near_goal_when_camera_faces_near_goal():
    """The two goals compete on equal footing. With the camera looking
    toward x=0, the near goal's crossbar must be picked over the far
    goal (which is at x=105 and behind the ray at this geometry)."""
    geom = _geometry()
    target_near = np.array([0.0, 34.0, geom.crossbar_z])
    uv = _project_world_to_pixel(target_near)
    world = resolve_goal_impact_world(
        uv, "crossbar",
        K=_K, R=_R, t=_t, distortion=_DISTORTION, geometry=geom,
    )
    assert world[0] == pytest.approx(0.0, abs=1e-6)


def test_resolve_picks_far_goal_when_camera_faces_far_goal():
    """Same pixel resolution math should pick the far goal when the
    camera faces +x toward x=105."""
    # Flip the camera to the other end-line: same orientation
    # (looking +x) but starting at x=120 (still beyond the far
    # goal line at 105). Then ray points away from far goal — instead
    # use a camera at x=-10 looking +x and check picks near goal already
    # covered; for the far-goal case put camera at x=115 facing -x.
    R_back = np.array([[0.0, -1.0, 0.0],
                       [0.0, 0.0, -1.0],
                       [-1.0, 0.0, 0.0]])
    C_back = np.array([115.0, 34.0, 1.22])
    t_back = -R_back @ C_back

    geom = _geometry()
    target_far = np.array([105.0, 34.0, geom.crossbar_z])
    cam = R_back @ target_far + t_back
    uv = (
        float(cam[0] * 1500.0 / cam[2] + 640.0),
        float(cam[1] * 1500.0 / cam[2] + 360.0),
    )
    world = resolve_goal_impact_world(
        uv, "crossbar",
        K=_K, R=R_back, t=t_back, distortion=_DISTORTION, geometry=geom,
    )
    assert world[0] == pytest.approx(105.0, abs=1e-3)


def test_resolve_raises_when_no_intersection():
    """A pixel that points at the sky (no goal element in front of the
    camera) raises ValueError so the caller can fall back to the
    generic ray-to-plane path."""
    geom = _geometry()
    # Pixel high above the goal — image_xy with v << 0 maps to a ray
    # going strongly upward, missing every post/crossbar/net extent.
    world_above = np.array([0.0, 34.0, 50.0])
    uv = _project_world_to_pixel(world_above)
    # Pick "side_net": needs ray to cross the y_post plane within
    # the x_range [-1.5, 0]. The ray here has d_y ≈ 0 so the side_net
    # planes get rejected as parallel.
    with pytest.raises(ValueError):
        resolve_goal_impact_world(
            uv, "side_net",
            K=_K, R=_R, t=_t, distortion=_DISTORTION, geometry=geom,
        )


def test_resolve_unknown_element_raises():
    geom = _geometry()
    with pytest.raises(ValueError, match="unknown goal_element"):
        resolve_goal_impact_world(
            (640.0, 360.0), "bar",
            K=_K, R=_R, t=_t, distortion=_DISTORTION, geometry=geom,
        )
