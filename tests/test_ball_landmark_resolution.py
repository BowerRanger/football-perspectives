"""Grounded anchors with a landmark snap to the feature in BOTH anchor
resolution paths (piecewise _resolve_anchor_world, events
_resolve_waypoint_world)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from src.schemas.ball_anchor import BallAnchor
from src.utils.pitch_landmarks import LANDMARK_CATALOGUE

BALL_R = 0.11
NAME = "left_goal_left_post_base"


def _camera_pose():
    look = np.array([0.0, 64.0, -30.0])
    look /= np.linalg.norm(look)
    right = np.array([1.0, 0.0, 0.0])
    down = np.cross(look, right)
    R = np.array([right, down, look], dtype=float)
    t = -R @ np.array([52.5, -30.0, 30.0])
    K = np.array([[1500.0, 0, 640.0], [0, 1500.0, 360.0], [0, 0, 1.0]])
    return K, R, t


def _anchor() -> BallAnchor:
    return BallAnchor(frame=5, image_xy=(400.0, 400.0), state="grounded",
                      landmark=NAME)


def test_piecewise_path_snaps_grounded_landmark():
    from src.stages.ball import _resolve_anchor_world
    from src.utils.goal_geometry import GoalGeometry

    K, R, t = _camera_pose()
    lm = LANDMARK_CATALOGUE[NAME]
    world = _resolve_anchor_world(
        anc=_anchor(), fi=5, ground_touch_frames=set(),
        # grounded+landmark path must not touch the player context
        player_ctx=SimpleNamespace(),
        per_frame_K={5: K}, per_frame_R={5: R}, per_frame_t={5: t},
        distortion=(0.0, 0.0), ball_radius=BALL_R,
        goal_geometry=GoalGeometry.from_pitch_config(
            {"length_m": 105.0, "width_m": 68.0, "goal_height_m": 2.44,
             "goal_width_m": 7.32, "goal_depth_m": 1.5}),
    )
    assert world is not None
    assert world[0] == pytest.approx(lm.world_xyz[0])
    assert world[1] == pytest.approx(lm.world_xyz[1])
    assert world[2] == pytest.approx(BALL_R)


def test_events_path_snaps_grounded_landmark():
    from src.utils.ball_event_resolver import _resolve_waypoint_world

    K, R, t = _camera_pose()
    lm = LANDMARK_CATALOGUE[NAME]
    world = _resolve_waypoint_world(
        _anchor(), 5, K, R, t, (0.0, 0.0), BALL_R, None)
    assert world is not None
    assert world[0] == pytest.approx(lm.world_xyz[0])
    assert world[1] == pytest.approx(lm.world_xyz[1])
    assert world[2] == pytest.approx(BALL_R)


def test_grounded_without_landmark_unchanged_ray_cast():
    from src.utils.ball_event_resolver import _resolve_waypoint_world
    from src.utils.foot_anchor import ankle_ray_to_pitch
    from src.utils.ball_anchor_heights import state_to_height

    K, R, t = _camera_pose()
    anc = BallAnchor(frame=5, image_xy=(400.0, 400.0), state="grounded")
    world = _resolve_waypoint_world(anc, 5, K, R, t, (0.0, 0.0), BALL_R, None)
    expected = ankle_ray_to_pitch(
        (400.0, 400.0), K=K, R=R, t=t,
        plane_z=state_to_height("grounded"), distortion=(0.0, 0.0))
    assert world == pytest.approx(expected)
