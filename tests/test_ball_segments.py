"""Tests for deriving interpolation segments from ordered keyframes."""

from __future__ import annotations

from src.schemas.ball_keyframes import BallKeyframe
from src.utils.ball_segments import derive_segments


def _kf(frame, world, state="grounded", depth="ground", **kw):
    return BallKeyframe(
        frame=frame, state=state, depth_source=depth,
        world_xyz=world,
        image_xy=(10.0, 20.0) if world is not None else None,
        **kw,
    )


def test_shot_touch_to_bounce_is_ballistic():
    kfs = (
        _kf(0, (0.0, 0.0, 0.3), state="player_touch", depth="player_bone",
            player_id="P1", bone="r_foot", touch_type="shot"),
        _kf(20, (15.0, 0.0, 0.11), state="bounce"),
    )
    segs = derive_segments(kfs, n_frames=21, fps=25.0)
    ball = [s for s in segs if s.kind == "ballistic"]
    assert len(ball) == 1
    assert ball[0].start_frame == 0
    assert ball[0].end_frame == 20


def test_grounded_to_grounded_is_roll():
    kfs = (_kf(0, (0.0, 0.0, 0.11)), _kf(10, (5.0, 0.0, 0.11)))
    segs = derive_segments(kfs, n_frames=11, fps=25.0)
    between = [s for s in segs if s.start_frame == 0 and s.end_frame == 10]
    assert len(between) == 1
    assert between[0].kind == "roll"


def test_off_screen_endpoint_is_free_flight():
    kfs = (
        _kf(0, (0.0, 0.0, 0.11), state="kick"),
        BallKeyframe(frame=10, state="off_screen_flight",
                     depth_source="ray_physics", world_xyz=None, image_xy=None),
    )
    segs = derive_segments(kfs, n_frames=11, fps=25.0)
    between = [s for s in segs if s.start_frame == 0 and s.end_frame == 10]
    assert between[0].kind == "free_flight"


def test_open_start_and_end_emit_boundary_holds():
    kfs = (_kf(5, (1.0, 1.0, 0.11)), _kf(8, (2.0, 1.0, 0.11)))
    segs = derive_segments(kfs, n_frames=12, fps=25.0)
    starts = {(s.start_frame, s.end_frame): s for s in segs}
    # open start: frame 0 -> first keyframe
    assert (0, 5) in starts and starts[(0, 5)].kind == "rest"
    assert starts[(0, 5)].hints.get("inferred") is True
    # open end: last keyframe -> last clip frame
    assert (8, 11) in starts and starts[(8, 11)].kind == "rest"
    assert starts[(8, 11)].hints.get("open_ended") is True


def test_carry_span_classified_as_carry():
    kfs = (
        _kf(0, (0.0, 0.0, 0.2), state="player_touch", depth="player_bone",
            player_id="P1", bone="r_foot"),
        _kf(8, (1.0, 0.0, 0.2), state="player_touch", depth="player_bone",
            player_id="P1", bone="r_foot"),
    )
    segs = derive_segments(kfs, n_frames=9, fps=25.0, carry_spans=[(0, 8)])
    between = [s for s in segs if s.start_frame == 0 and s.end_frame == 8][0]
    assert between.kind == "carry"
    assert between.hints.get("player_id") == "P1"


def test_empty_keyframes_yields_no_segments():
    assert derive_segments((), n_frames=10, fps=25.0) == ()


def test_single_keyframe_yields_only_boundary_holds():
    segs = derive_segments((_kf(4, (0.0, 0.0, 0.11)),), n_frames=10, fps=25.0)
    # No between-pair segment, but boundary holds on each side.
    spans = {(s.start_frame, s.end_frame) for s in segs}
    assert (0, 4) in spans
    assert (4, 9) in spans
