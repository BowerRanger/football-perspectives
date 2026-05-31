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
        ray=((0.0, 0.0, 15.0), (0.1, 0.2, -0.97)),
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
    import json
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
