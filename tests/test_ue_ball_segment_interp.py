"""Parity: the UE-side segment evaluator must produce the same per-frame
positions as the pipeline reference interpolator for identical keyframes +
segments. Guards against the two interpolators drifting apart (spec §17).

The UE module lives outside this repo (the UE editor Python tree). It is
written to be Unreal-free and dependency-free, so we import it by path.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from src.schemas.ball_keyframes import BallKeyframe, BallKeyframeSet, BallSegment
from src.utils.ball_interpolate import interpolate_events

_UE_MODULE = Path(
    "/Users/joebower/workplace/FootballPerspectives 5.8/Content/Python/"
    "football_perspectives/ball_segment_interp.py"
)


def _load_ue_module():
    if not _UE_MODULE.exists():
        pytest.skip(f"UE module not present at {_UE_MODULE}")
    spec = importlib.util.spec_from_file_location("ue_ball_segment_interp", _UE_MODULE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _kf(frame, world, state="grounded"):
    return BallKeyframe(
        frame=frame, state=state, depth_source="ground",
        world_xyz=world, image_xy=(10.0, 20.0),
    )


def _scenario():
    """A kick -> arc -> bounce -> roll-to-rest sequence exercising
    ballistic, roll and rest segments."""
    keyframes = (
        _kf(0, (0.0, 0.0, 0.11), state="kick"),
        _kf(15, (8.0, 1.0, 0.11), state="bounce"),
        _kf(30, (12.0, 1.0, 0.11), state="grounded"),
    )
    segments = (
        BallSegment(0, 15, "ballistic", {"gravity": -9.81}),
        BallSegment(15, 30, "roll", {}),
    )
    return keyframes, segments


def test_ue_evaluator_matches_pipeline_interpolator():
    ue = _load_ue_module()
    keyframes, segments = _scenario()
    fps = 30.0

    ks = BallKeyframeSet(
        clip_id="c", fps=fps, image_size=(1280, 720),
        keyframes=keyframes, segments=segments,
    )
    track = interpolate_events(ks, n_frames=31)
    pipe = {f.frame: f.world_xyz for f in track.frames if f.world_xyz is not None}

    ue_world = ue.interpolate_segments(
        [(k.frame, k.world_xyz) for k in keyframes],
        [(s.start_frame, s.end_frame, s.kind, s.hints) for s in segments],
        fps=fps,
    )

    assert set(ue_world) == set(pipe)
    for fr in pipe:
        a, b = ue_world[fr], pipe[fr]
        for x, y in zip(a, b):
            assert abs(x - y) < 1e-6, f"frame {fr}: UE {a} vs pipeline {b}"
