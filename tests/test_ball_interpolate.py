"""Tests for the pure §10 reference interpolator: a sparse BallKeyframeSet
(+ derived segments) rendered into a dense BallTrack."""

from __future__ import annotations

from src.schemas.ball_keyframes import BallKeyframe, BallKeyframeSet, BallSegment
from src.utils.ball_interpolate import interpolate_events


def _kf(frame, world, state="grounded", depth="ground", **kw):
    return BallKeyframe(
        frame=frame, state=state, depth_source=depth,
        world_xyz=world,
        image_xy=(10.0, 20.0) if world is not None else None,
        **kw,
    )


def test_roll_segment_linear_midpoint():
    ks = BallKeyframeSet(
        clip_id="c", fps=10.0, image_size=(100, 100),
        keyframes=(
            _kf(0, (0.0, 0.0, 0.11)),
            _kf(10, (10.0, 4.0, 0.11)),
        ),
        segments=(BallSegment(0, 10, "roll", {}),),
    )
    track = interpolate_events(ks)
    assert len(track.frames) == 11
    f5 = track.frames[5]
    assert abs(f5.world_xyz[0] - 5.0) < 1e-6
    assert abs(f5.world_xyz[1] - 2.0) < 1e-6
    assert abs(f5.world_xyz[2] - 0.11) < 1e-6


def test_ballistic_segment_arcs_above_endpoints():
    ks = BallKeyframeSet(
        clip_id="c", fps=10.0, image_size=(100, 100),
        keyframes=(
            _kf(0, (0.0, 0.0, 0.11), state="kick"),
            _kf(10, (10.0, 0.0, 0.11), state="grounded"),
        ),
        segments=(BallSegment(0, 10, "ballistic", {"gravity": -9.81}),),
    )
    track = interpolate_events(ks)
    # endpoints exact
    assert abs(track.frames[0].world_xyz[2] - 0.11) < 1e-6
    assert abs(track.frames[10].world_xyz[2] - 0.11) < 1e-6
    # apex above the two equal-height endpoints
    assert track.frames[5].world_xyz[2] > 0.11 + 0.1
    assert track.frames[5].state == "flight"


def test_free_flight_with_unknown_endpoint_leaves_gap_none():
    ks = BallKeyframeSet(
        clip_id="c", fps=10.0, image_size=(100, 100),
        keyframes=(
            _kf(0, (0.0, 0.0, 0.11)),
            BallKeyframe(frame=10, state="off_screen_flight",
                         depth_source="ray_physics",
                         world_xyz=None, image_xy=None),
        ),
        segments=(BallSegment(0, 10, "free_flight", {}),),
    )
    track = interpolate_events(ks)
    assert track.frames[5].world_xyz is None
    assert track.frames[5].state == "flight"


def test_carry_segment_uses_linear_interpolation_in_phase_1():
    """Characterisation test: a ``carry`` span interpolates LINEARLY in
    Phase 1 (the §10 player-foot-path target is a Phase-3 follow-up). This
    pins the current behaviour so the Phase-3 change is a deliberate, visible
    edit rather than a silent one."""
    ks = BallKeyframeSet(
        clip_id="c", fps=10.0, image_size=(100, 100),
        keyframes=(
            _kf(0, (0.0, 0.0, 0.2), state="player_touch", depth="player_bone"),
            _kf(10, (10.0, 4.0, 0.2), state="player_touch", depth="player_bone"),
        ),
        segments=(BallSegment(0, 10, "carry", {"player_id": "P1"}),),
    )
    track = interpolate_events(ks)
    f5 = track.frames[5]
    assert abs(f5.world_xyz[0] - 5.0) < 1e-6
    assert abs(f5.world_xyz[1] - 2.0) < 1e-6


def test_frames_are_contiguous_and_span_clip():
    ks = BallKeyframeSet(
        clip_id="c", fps=10.0, image_size=(100, 100),
        keyframes=(_kf(0, (0.0, 0.0, 0.11)), _kf(6, (6.0, 0.0, 0.11))),
        segments=(BallSegment(0, 6, "roll", {}),),
    )
    track = interpolate_events(ks, n_frames=12)
    assert len(track.frames) == 12
    assert [f.frame for f in track.frames] == list(range(12))
