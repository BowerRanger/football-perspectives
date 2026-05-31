import numpy as np

from src.schemas.ball_anchor import BallAnchor
from src.utils.ball_keyframe_builder import build_ball_keyframe_set


def _ident_cam():
    K = np.array([[1000.0, 0.0, 960.0], [0.0, 1000.0, 540.0], [0.0, 0.0, 1.0]])
    R = np.eye(3)
    # Camera 10 m up looking down -z so pixels map onto the ground.
    t = np.array([0.0, 0.0, 10.0])
    return K, R, t


def test_grounded_anchor_yields_ground_depth_source_no_ray():
    K, R, t = _ident_cam()
    anchors = {5: BallAnchor(frame=5, image_xy=(960.0, 540.0), state="grounded")}
    world = {5: (3.0, 4.0, 0.11)}
    kfset = build_ball_keyframe_set(
        clip_id="c", fps=25.0, image_size=(1920, 1080),
        anchor_by_frame=anchors, world_by_frame=world,
        per_frame_K={5: K}, per_frame_R={5: R}, per_frame_t={5: t},
        distortion=(0.0, 0.0),
    )
    assert len(kfset.keyframes) == 1
    kf = kfset.keyframes[0]
    assert kf.frame == 5
    assert kf.state == "grounded"
    assert kf.depth_source == "ground"
    assert kf.world_xyz == (3.0, 4.0, 0.11)
    assert kf.ray is None  # rays only for airborne


def test_airborne_anchor_carries_ray_and_ray_physics_source():
    K, R, t = _ident_cam()
    anchors = {7: BallAnchor(frame=7, image_xy=(960.0, 540.0), state="airborne_high")}
    world = {7: (0.0, 0.0, 4.0)}
    kfset = build_ball_keyframe_set(
        clip_id="c", fps=25.0, image_size=(1920, 1080),
        anchor_by_frame=anchors, world_by_frame=world,
        per_frame_K={7: K}, per_frame_R={7: R}, per_frame_t={7: t},
        distortion=(0.0, 0.0),
    )
    kf = kfset.keyframes[0]
    assert kf.depth_source == "ray_physics"
    assert kf.ray is not None
    origin, direction = kf.ray
    # Fixture geometry (R=I, t=(0,0,10)): camera centre C = -R^T t = (0,0,-10).
    assert np.allclose(origin, (0.0, 0.0, -10.0))
    # Pixel at the principal point (960,540): inv(K)@[u,v,1] = [0,0,1],
    # so the unit direction is +z.
    assert np.allclose(direction, (0.0, 0.0, 1.0))


def test_player_touch_carries_player_bone_and_player_bone_source():
    K, R, t = _ident_cam()
    anchors = {
        3: BallAnchor(
            frame=3, image_xy=(900.0, 500.0), state="player_touch",
            player_id="P002", bone="head",
        )
    }
    world = {3: (1.0, 1.0, 1.8)}
    kfset = build_ball_keyframe_set(
        clip_id="c", fps=25.0, image_size=(1920, 1080),
        anchor_by_frame=anchors, world_by_frame=world,
        per_frame_K={3: K}, per_frame_R={3: R}, per_frame_t={3: t},
        distortion=(0.0, 0.0),
        ground_touch_frames=set(),
    )
    kf = kfset.keyframes[0]
    assert kf.depth_source == "player_bone"
    assert kf.player_id == "P002"
    assert kf.bone == "head"


def test_goal_impact_depth_source():
    K, R, t = _ident_cam()
    anchors = {
        10: BallAnchor(
            frame=10, image_xy=(960.0, 540.0), state="goal_impact",
            goal_element="crossbar",
        )
    }
    world = {10: (104.5, 34.0, 2.44)}
    kfset = build_ball_keyframe_set(
        clip_id="c", fps=25.0, image_size=(1920, 1080),
        anchor_by_frame=anchors, world_by_frame=world,
        per_frame_K={10: K}, per_frame_R={10: R}, per_frame_t={10: t},
        distortion=(0.0, 0.0),
    )
    kf = kfset.keyframes[0]
    assert kf.depth_source == "goal_geometry"
    assert kf.goal_element == "crossbar"


def test_player_touch_ground_touch_frame_is_ground():
    K, R, t = _ident_cam()
    anchors = {
        3: BallAnchor(
            frame=3, image_xy=(900.0, 500.0), state="player_touch",
            player_id="P001", bone="left_ankle",
        )
    }
    world = {3: (10.0, 5.0, 0.11)}
    kfset = build_ball_keyframe_set(
        clip_id="c", fps=25.0, image_size=(1920, 1080),
        anchor_by_frame=anchors, world_by_frame=world,
        per_frame_K={3: K}, per_frame_R={3: R}, per_frame_t={3: t},
        distortion=(0.0, 0.0),
        ground_touch_frames={3},
    )
    kf = kfset.keyframes[0]
    assert kf.depth_source == "ground"


def test_off_screen_flight_has_no_ray_no_world():
    K, R, t = _ident_cam()
    anchors = {
        15: BallAnchor(frame=15, image_xy=None, state="off_screen_flight")
    }
    # world_by_frame does NOT contain frame 15 — world is None
    world: dict = {}
    kfset = build_ball_keyframe_set(
        clip_id="c", fps=25.0, image_size=(1920, 1080),
        anchor_by_frame=anchors, world_by_frame=world,
        per_frame_K={15: K}, per_frame_R={15: R}, per_frame_t={15: t},
        distortion=(0.0, 0.0),
    )
    kf = kfset.keyframes[0]
    assert kf.world_xyz is None
    assert kf.ray is None
    assert kf.image_xy is None
    assert kf.depth_source == "ground"


def test_keyframes_sorted_by_frame():
    K, R, t = _ident_cam()
    anchors = {
        9: BallAnchor(frame=9, image_xy=(1.0, 1.0), state="grounded"),
        2: BallAnchor(frame=2, image_xy=(1.0, 1.0), state="kick"),
    }
    world = {9: (0.0, 0.0, 0.1), 2: (0.0, 0.0, 0.1)}
    kfset = build_ball_keyframe_set(
        clip_id="c", fps=25.0, image_size=(1920, 1080),
        anchor_by_frame=anchors, world_by_frame=world,
        per_frame_K={2: K, 9: K}, per_frame_R={2: R, 9: R},
        per_frame_t={2: t, 9: t}, distortion=(0.0, 0.0),
    )
    assert [kf.frame for kf in kfset.keyframes] == [2, 9]
