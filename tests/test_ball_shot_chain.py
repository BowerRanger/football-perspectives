"""Shot-chain proposal pairing + validation warnings."""

from __future__ import annotations

import pytest

from src.schemas.ball_keyframes import BallKeyframe, BallKeyframeSet
from src.utils.ball_auto_events import BallEvent
from src.utils.ball_shot_chain import (
    ShotChainCfg,
    chain_warnings,
    propose_shot_chains,
)

CFG = ShotChainCfg()


def _kfset(kfs: list[BallKeyframe]) -> BallKeyframeSet:
    return BallKeyframeSet(
        clip_id="play", fps=30.0, image_size=(1280, 720),
        keyframes=tuple(kfs), segments=(),
    )


def _kf(frame: int, world) -> BallKeyframe:
    return BallKeyframe(
        frame=frame, state="grounded", depth_source="ground",
        world_xyz=world,
    )


def test_propose_pairs_last_touch_before_impact():
    events = [
        BallEvent(frame=10, kind="touch", score=0.8, player_id="P1", bone="r_foot"),
        BallEvent(frame=30, kind="touch", score=0.6, player_id="P2", bone="l_foot"),
        BallEvent(frame=60, kind="goal_impact", score=0.9, goal_element="back_net"),
    ]
    assert propose_shot_chains(events, CFG) == ((30, 60),)


def test_propose_respects_window_and_disabled():
    events = [
        BallEvent(frame=1, kind="touch", score=0.8, player_id="P1", bone="r_foot"),
        BallEvent(frame=200, kind="goal_impact", score=0.9, goal_element="post"),
    ]
    assert propose_shot_chains(events, CFG) == ()  # 199 > 75 frame window
    assert propose_shot_chains(
        [BallEvent(frame=10, kind="touch", score=0.8, player_id="P1", bone="r_foot"),
         BallEvent(frame=40, kind="goal_impact", score=0.9, goal_element="post")],
        ShotChainCfg(enabled=False),
    ) == ()


def test_propose_one_chain_per_impact_multiple_impacts():
    events = [
        BallEvent(frame=10, kind="touch", score=0.8, player_id="P1", bone="r_foot"),
        BallEvent(frame=40, kind="goal_impact", score=0.9, goal_element="crossbar"),
        BallEvent(frame=55, kind="goal_impact", score=0.7, goal_element="back_net"),
    ]
    assert propose_shot_chains(events, CFG) == ((10, 40), (10, 55))


def test_warnings_ok_chain_is_empty():
    # 30 frames at 30 fps between two knots 20 m apart -> 20 m/s: in band.
    kfs = _kfset([_kf(10, (10.0, 34.0, 0.11)), _kf(40, (30.0, 34.0, 0.11))])
    assert chain_warnings([10, 40], kfs, 30.0, CFG) == []


def test_warnings_flag_launch_speed_out_of_band():
    # 2 m in 1 s -> 2 m/s: below the 8 m/s floor (a mis-clicked frame).
    kfs = _kfset([_kf(10, (10.0, 34.0, 0.11)), _kf(40, (12.0, 34.0, 0.11))])
    warns = chain_warnings([10, 40], kfs, 30.0, CFG)
    assert len(warns) == 1
    assert warns[0]["kind"] == "launch_speed"
    assert warns[0]["frames"] == [10, 40]
    assert warns[0]["speed_m_s"] == pytest.approx(2.0)


def test_warnings_flag_missing_and_unresolved():
    kfs = _kfset([_kf(10, (10.0, 34.0, 0.11)), _kf(40, None)])
    warns = chain_warnings([10, 40, 99], kfs, 30.0, CFG)
    kinds = {w["kind"] for w in warns}
    assert "unresolved_world" in kinds   # frame 40 has no world
    assert "missing_keyframe" in kinds   # frame 99 has no keyframe


def test_warnings_none_keyframes_degrades():
    warns = chain_warnings([10, 40], None, 30.0, CFG)
    assert [w["kind"] for w in warns] == ["missing_keyframe"]


def test_cfg_from_default_yaml_keys():
    import yaml
    from pathlib import Path
    cfg = yaml.safe_load(
        Path("config/default.yaml").read_text())["ball"]["shot_chain"]
    assert cfg["enabled"] is True
    assert cfg["pair_window_frames"] == 75
    assert cfg["launch_speed_warn_min_m_s"] == 8.0
    assert cfg["launch_speed_warn_max_m_s"] == 45.0
