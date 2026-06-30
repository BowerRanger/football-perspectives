"""Carry/possession span detection from resolved keyframes."""

from __future__ import annotations

from src.schemas.ball_keyframes import BallKeyframe
from src.utils.ball_possession import detect_carry_spans


def _touch(frame, world, player="P1", bone="r_foot"):
    return BallKeyframe(
        frame=frame, state="player_touch", depth_source="player_bone",
        world_xyz=world, image_xy=(10.0, 20.0), player_id=player, bone=bone,
    )


def test_close_same_player_touches_form_a_carry_span():
    kfs = (
        _touch(0, (0.0, 0.0, 0.2)),
        _touch(8, (1.0, 0.0, 0.2)),
        _touch(15, (2.0, 0.0, 0.2)),
    )
    spans = detect_carry_spans(kfs, max_gap_frames=15, max_step_m=3.0)
    assert (0, 8) in spans
    assert (8, 15) in spans


def test_far_apart_touches_are_not_a_carry():
    kfs = (
        _touch(0, (0.0, 0.0, 0.2)),
        _touch(8, (25.0, 0.0, 0.2)),  # 25 m apart = a pass, not a dribble
    )
    assert detect_carry_spans(kfs, max_gap_frames=15, max_step_m=3.0) == []


def test_different_players_are_not_a_carry():
    kfs = (
        _touch(0, (0.0, 0.0, 0.2), player="P1"),
        _touch(6, (1.0, 0.0, 0.2), player="P2"),
    )
    assert detect_carry_spans(kfs) == []


def test_large_frame_gap_breaks_carry():
    kfs = (
        _touch(0, (0.0, 0.0, 0.2)),
        _touch(40, (1.0, 0.0, 0.2)),  # 40-frame gap = ball left and returned
    )
    assert detect_carry_spans(kfs, max_gap_frames=15) == []
