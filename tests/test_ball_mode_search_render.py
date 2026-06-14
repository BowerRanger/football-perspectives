"""Task 7 TDD: ``solve_modes`` entry point + render to ``SolveResult``.

Covers the required scenarios:
1. ``solve_modes`` on a synthetic multi-segment scene returns a ``SolveResult``;
   ``state_by_frame`` covers ALL frames in range(n_frames); every state value is
   a valid ``schemas.ball_track`` State literal (assert membership).
2. FLIGHT frames carry a non-null ``flight_segment_id``; POSSESSED frames are
   ``"grounded"`` (never ``"flight"``).
3. An out-of-view span > N frames appears in BOTH ``out_of_view_spans`` and
   ``underconstrained_spans`` diagnostics (F14).
4. ``diagnostics["mode_search"]`` is present with the required keys.

The Task-0 golden test (``tests/test_solver_commit_extraction.py``) and the
quality-report test are exercised separately; this file is the render contract.
"""
from __future__ import annotations

import typing

import numpy as np

from src.schemas.ball_track import FlightSegment as _BallFlightSegment
from src.schemas.ball_track import State as _BallState
from src.utils.ball_auto_events import BallEvent
from src.utils.ball_mode_search import ModeSearchCfg, solve_modes
from src.utils.ball_piecewise_solver import SolveResult, SolverCfg
from src.utils.ball_player_context import JointSample, PlayerContext
from tests.fixtures.ball_synthetic import (
    FPS,
    ballistic_worlds,
    broadcast_camera,
    per_frame_cams,
    project_track,
    rolling_worlds,
    steps_from_pixels,
)

# The four valid per-frame state literals (assert renderer output membership).
_VALID_STATES = set(typing.get_args(_BallState))
assert _VALID_STATES == {"grounded", "flight", "occluded", "missing"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _kwargs(worlds, n_frames, *, pixels_override=None, p_flight=0.0,
            cfg=None):
    K, R, t = broadcast_camera()
    pixels = pixels_override if pixels_override is not None else \
        project_track(worlds, K, R, t)
    steps = steps_from_pixels(pixels, n_frames, p_flight=p_flight)
    per_K, per_R, per_t = per_frame_cams(n_frames)
    return dict(
        nodes=(),
        steps=steps,
        confidences={i: 1.0 for i in range(n_frames)},
        per_frame_K=per_K,
        per_frame_R=per_R,
        per_frame_t=per_t,
        distortion=(0.0, 0.0),
        fps=FPS,
        n_frames=n_frames,
        pitch_length_m=105.0,
        pitch_width_m=68.0,
        split_hints=(),
        z_hints=None,
        manual_obs_frames=None,
        world_fixes=None,
        cfg=cfg or SolverCfg(),
    )


def _multi_segment_scene(n=90):
    """flight (with bounce) → roll, the canonical multi-segment scene."""
    roll_start = 65
    arc = ballistic_worlds((20.0, 30.0, 0.5), (9.0, 1.0, 8.0), roll_start,
                           restitution=0.7)
    worlds = {f: arc[f] for f in range(roll_start)}
    last_xy = worlds[roll_start - 1][:2]
    roll = rolling_worlds(tuple(last_xy), (0.4, 0.05), n - roll_start,
                          start_frame=roll_start)
    worlds.update(roll)
    return worlds, n


# ---------------------------------------------------------------------------
# Test 1 — SolveResult shape + full-frame state coverage + valid literals
# ---------------------------------------------------------------------------

def test_solve_modes_returns_solveresult_full_frame_valid_states():
    worlds, n = _multi_segment_scene()
    res = solve_modes(**_kwargs(worlds, n))

    assert isinstance(res, SolveResult)
    # state_by_frame covers EVERY frame in range(n_frames).
    assert set(res.state_by_frame) == set(range(n))
    # Every state value is a valid BallFrame.State literal.
    for f, st in res.state_by_frame.items():
        assert st in _VALID_STATES, f"frame {f} has invalid state {st!r}"
    # flight_segments are real FlightSegment objects.
    for seg in res.flight_segments:
        assert isinstance(seg, _BallFlightSegment)
    # diagnostics carries the segments list (same shape as piecewise).
    assert "segments" in res.diagnostics
    assert isinstance(res.diagnostics["segments"], list)


# ---------------------------------------------------------------------------
# Test 2 — flight frames carry a non-null id; possessed frames are grounded
# ---------------------------------------------------------------------------

def test_flight_frames_have_segment_id_and_no_flight_state_for_possessed():
    worlds, n = _multi_segment_scene()
    res = solve_modes(**_kwargs(worlds, n))

    flight_frames = [f for f, st in res.state_by_frame.items() if st == "flight"]
    assert flight_frames, "scene must produce at least one flight frame"

    # Build the frame -> segment-id map the renderer must produce.
    seg_id_by_frame = res.diagnostics.get("flight_segment_id_by_frame", {})
    for f in flight_frames:
        sid = seg_id_by_frame.get(f)
        assert sid is not None, (
            f"flight frame {f} has no flight_segment_id"
        )
        assert any(s.id == sid for s in res.flight_segments)


def test_possessed_frames_are_grounded_not_flight():
    """A span where the ball rides a player's foot must render as grounded."""
    n = 40
    K, R, t = broadcast_camera()
    # Foot world the ball tracks for the whole clip (a possession dribble).
    foot_world_by_frame = {
        f: np.array([30.0 + 0.2 * f, 34.0, 0.11]) for f in range(n)
    }
    pixels = {
        f: project_track({f: foot_world_by_frame[f]}, K, R, t)[f]
        for f in range(n)
    }
    samples_by_frame: dict[int, tuple[JointSample, ...]] = {}
    for f in range(n):
        w = foot_world_by_frame[f]
        uv = pixels[f]
        samples_by_frame[f] = tuple(
            JointSample(player_id="P1", bone=bone,
                        world_xyz=(float(w[0]), float(w[1]), float(w[2])),
                        uv=uv, confidence=0.9)
            for bone in ("l_foot", "r_foot")
        )
    ctx = PlayerContext(samples_by_frame=samples_by_frame, player_ids=("P1",))

    kwargs = _kwargs(foot_world_by_frame, n, pixels_override=pixels)
    res = solve_modes(player_ctx=ctx, **kwargs)

    # No frame may be "flight" if the winning segment over it is possessed.
    seg_diag = res.diagnostics["segments"]
    possessed_ranges = [
        (s["start"], s["end"]) for s in seg_diag if s["kind"] == "possessed"
    ]
    for (fa, fb) in possessed_ranges:
        for f in range(fa, fb + 1):
            assert res.state_by_frame[f] != "flight", (
                f"possessed frame {f} rendered as flight"
            )
            assert res.state_by_frame[f] == "grounded"


# ---------------------------------------------------------------------------
# Test 3 — out-of-view span > N surfaces in both diag lists
# ---------------------------------------------------------------------------

def test_out_of_view_span_in_both_diag_lists():
    """A long gap with no usable detections renders OUT_OF_VIEW and surfaces in
    out_of_view_spans AND underconstrained_spans."""
    n = 60
    # First 20 frames: a clean roll (detections). Frames 20..45: no detections
    # at all (ball off-camera). Frames 46..59: another roll (detections).
    roll_a = rolling_worlds((20.0, 30.0), (0.4, 0.0), 20, start_frame=0)
    roll_b = rolling_worlds((40.0, 34.0), (0.4, 0.0), 14, start_frame=46)
    worlds = {**roll_a, **roll_b}
    K, R, t = broadcast_camera()
    pixels: dict[int, tuple[float, float] | None] = {}
    proj_a = project_track(roll_a, K, R, t)
    proj_b = project_track(roll_b, K, R, t)
    for f in range(n):
        if f in proj_a:
            pixels[f] = proj_a[f]
        elif f in proj_b:
            pixels[f] = proj_b[f]
        else:
            pixels[f] = None  # genuinely missing — out of view

    # A detection drop surfaces velocity-break events at the gap edges (the
    # real pipeline produces these from the pixel track / detector coverage);
    # they bracket the gap so the beam can isolate it as OUT_OF_VIEW.
    events = (
        BallEvent(frame=19, kind="velocity_break", score=0.5),
        BallEvent(frame=46, kind="velocity_break", score=0.5),
    )
    kwargs = _kwargs(worlds, n, pixels_override=pixels)
    res = solve_modes(events=events, **kwargs)

    oov = res.diagnostics.get("out_of_view_spans", [])
    assert oov, "expected at least one out_of_view span"
    # The big gap (20..45) must be present as a span > N frames.
    big = [s for s in oov if s["end"] - s["start"] + 1 > 3]
    assert big, f"expected a long OOV span, got {oov}"
    # Those long spans are ALSO in underconstrained_spans (F14).
    uc = res.diagnostics.get("underconstrained_spans", [])
    for span in big:
        assert any(
            u["start"] == span["start"] and u["end"] == span["end"]
            for u in uc
        ), f"OOV span {span} not surfaced in underconstrained_spans {uc}"
    # OOV INTERIOR frames stay "missing" (F12/F13). The shared boundary frames
    # belong to the neighbouring grounded segments (a segment writes its own
    # endpoints), so only the strict interior is guaranteed missing.
    for span in big:
        for f in range(span["start"] + 1, span["end"]):
            assert res.state_by_frame[f] == "missing", (
                f"OOV interior frame {f} should be missing, got "
                f"{res.state_by_frame[f]!r}"
            )


# ---------------------------------------------------------------------------
# Test 4 — mode_search diagnostics present with required keys
# ---------------------------------------------------------------------------

def test_mode_search_diag_present_with_keys():
    worlds, n = _multi_segment_scene()
    res = solve_modes(**_kwargs(worlds, n))

    ms = res.diagnostics.get("mode_search")
    assert ms is not None, "diagnostics must carry mode_search"
    for key in (
        "hypotheses_explored", "beam_width", "winning_cost",
        "runner_up_cost", "fit_calls", "budget_hit",
    ):
        assert key in ms, f"mode_search missing key {key!r}"
    assert ms["budget_hit"] is False
    assert ms["beam_width"] == ModeSearchCfg().beam_width
    assert ms["hypotheses_explored"] > 0
    assert ms["fit_calls"] > 0
