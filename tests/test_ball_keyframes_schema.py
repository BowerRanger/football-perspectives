import json
from pathlib import Path

import pytest

from src.schemas.ball_keyframes import BallKeyframe, BallKeyframeSet


def _grounded() -> BallKeyframe:
    return BallKeyframe(
        frame=10,
        state="grounded",
        world_xyz=(12.0, 4.0, 0.11),
        image_xy=(800.0, 600.0),
        depth_source="ground",
        confidence=1.0,
    )


def _airborne() -> BallKeyframe:
    return BallKeyframe(
        frame=20,
        state="airborne_high",
        world_xyz=(18.3, 9.2, 4.1),
        image_xy=(900.0, 300.0),
        ray=((0.0, 0.0, 15.0), (0.0, 0.0, -1.0)),
        depth_source="ray_physics",
        confidence=0.8,
    )


def test_round_trip_preserves_all_fields(tmp_path: Path):
    src = BallKeyframeSet(
        clip_id="clipA",
        fps=25.0,
        image_size=(1920, 1080),
        keyframes=(_grounded(), _airborne()),
    )
    path = tmp_path / "ball_keyframes.json"
    src.save(path)
    back = BallKeyframeSet.load(path)
    assert back == src


def test_player_touch_requires_player_and_bone(tmp_path: Path):
    path = tmp_path / "kf.json"
    BallKeyframeSet(
        clip_id="c", fps=25.0, image_size=(1920, 1080),
        keyframes=(
            BallKeyframe(
                frame=1, state="player_touch", world_xyz=(1.0, 2.0, 1.0),
                image_xy=(10.0, 10.0), depth_source="player_bone",
                player_id="P001", bone="right_foot", confidence=1.0,
            ),
        ),
    ).save(path)
    # Missing player_id must fail validation on load.
    raw = json.loads(path.read_text())
    del raw["keyframes"][0]["player_id"]
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="player_id is required"):
        BallKeyframeSet.load(path)


def test_unknown_state_rejected(tmp_path: Path):
    path = tmp_path / "kf.json"
    path.write_text(
        '{"clip_id":"c","fps":25.0,"image_size":[1920,1080],'
        '"keyframes":[{"frame":1,"state":"banana","depth_source":"ground"}]}'
    )
    with pytest.raises(ValueError, match="unknown ball keyframe state"):
        BallKeyframeSet.load(path)


def test_player_touch_missing_bone_raises(tmp_path: Path):
    path = tmp_path / "kf.json"
    BallKeyframeSet(
        clip_id="c", fps=25.0, image_size=(1920, 1080),
        keyframes=(
            BallKeyframe(
                frame=1, state="player_touch", world_xyz=(1.0, 2.0, 1.0),
                image_xy=(10.0, 10.0), depth_source="player_bone",
                player_id="P001", bone="right_foot", confidence=1.0,
            ),
        ),
    ).save(path)
    raw = json.loads(path.read_text())
    del raw["keyframes"][0]["bone"]
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="bone is required"):
        BallKeyframeSet.load(path)


def test_unknown_depth_source_rejected(tmp_path: Path):
    path = tmp_path / "kf.json"
    path.write_text(
        '{"clip_id":"c","fps":25.0,"image_size":[1920,1080],'
        '"keyframes":[{"frame":1,"state":"grounded","depth_source":"banana",'
        '"world_xyz":[1.0,2.0,0.0],"image_xy":[800.0,600.0]}]}'
    )
    with pytest.raises(ValueError, match="depth_source"):
        BallKeyframeSet.load(path)


def test_goal_impact_missing_goal_element_raises(tmp_path: Path):
    path = tmp_path / "kf.json"
    path.write_text(
        '{"clip_id":"c","fps":25.0,"image_size":[1920,1080],'
        '"keyframes":[{"frame":5,"state":"goal_impact","depth_source":"goal_geometry",'
        '"world_xyz":[105.0,34.0,1.2],"image_xy":[960.0,540.0]}]}'
    )
    with pytest.raises(ValueError, match="goal_element is required"):
        BallKeyframeSet.load(path)


def test_off_screen_flight_with_world_xyz_raises(tmp_path: Path):
    path = tmp_path / "kf.json"
    path.write_text(
        '{"clip_id":"c","fps":25.0,"image_size":[1920,1080],'
        '"keyframes":[{"frame":7,"state":"off_screen_flight","depth_source":"ray_physics",'
        '"world_xyz":[1,2,3],"image_xy":null}]}'
    )
    with pytest.raises(ValueError, match="off_screen_flight must have null"):
        BallKeyframeSet.load(path)


def test_off_screen_flight_clean_round_trips(tmp_path: Path):
    path = tmp_path / "kf.json"
    path.write_text(
        '{"clip_id":"c","fps":25.0,"image_size":[1920,1080],'
        '"keyframes":[{"frame":7,"state":"off_screen_flight","depth_source":"ray_physics",'
        '"world_xyz":null,"image_xy":null}]}'
    )
    result = BallKeyframeSet.load(path)
    assert len(result.keyframes) == 1
    kf = result.keyframes[0]
    assert kf.state == "off_screen_flight"
    assert kf.world_xyz is None
    assert kf.image_xy is None
