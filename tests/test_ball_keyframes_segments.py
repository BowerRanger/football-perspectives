"""Tests for sparse-interpolation extensions to the keyframe schema:
BallKeyframe.end_frame and BallKeyframeSet.segments (BallSegment)."""

from __future__ import annotations

import json
from pathlib import Path

from src.schemas.ball_keyframes import BallKeyframe, BallKeyframeSet, BallSegment


def _kf(frame: int, **kw) -> BallKeyframe:
    return BallKeyframe(
        frame=frame,
        state=kw.pop("state", "grounded"),
        depth_source=kw.pop("depth_source", "ground"),
        world_xyz=kw.pop("world_xyz", (1.0, 2.0, 0.11)),
        image_xy=kw.pop("image_xy", (10.0, 20.0)),
        **kw,
    )


def test_keyframe_end_frame_round_trips(tmp_path: Path):
    src = BallKeyframeSet(
        clip_id="c", fps=25.0, image_size=(1920, 1080),
        keyframes=(_kf(5, end_frame=12),),
    )
    p = tmp_path / "kf.json"
    src.save(p)
    back = BallKeyframeSet.load(p)
    assert back.keyframes[0].end_frame == 12


def test_segments_round_trip(tmp_path: Path):
    seg = BallSegment(
        start_frame=5, end_frame=20, kind="ballistic",
        hints={"gravity": -9.81, "open_ended": False},
    )
    src = BallKeyframeSet(
        clip_id="c", fps=25.0, image_size=(1920, 1080),
        keyframes=(_kf(5), _kf(20)),
        segments=(seg,),
    )
    p = tmp_path / "kf.json"
    src.save(p)
    back = BallKeyframeSet.load(p)
    assert len(back.segments) == 1
    assert back.segments[0].kind == "ballistic"
    assert back.segments[0].start_frame == 5
    assert back.segments[0].end_frame == 20
    assert back.segments[0].hints["gravity"] == -9.81


def test_legacy_keyframes_load_with_empty_segments(tmp_path: Path):
    """A keyframes file written before segments existed loads with an
    empty segments tuple and end_frame None."""
    p = tmp_path / "legacy.json"
    p.write_text(json.dumps({
        "clip_id": "c", "fps": 25.0, "image_size": [100, 100],
        "keyframes": [{
            "frame": 0, "state": "grounded", "depth_source": "ground",
            "world_xyz": [1, 2, 0.11], "image_xy": [10, 20],
        }],
    }))
    back = BallKeyframeSet.load(p)
    assert back.segments == ()
    assert back.keyframes[0].end_frame is None
