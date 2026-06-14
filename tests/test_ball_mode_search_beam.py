"""Task 6 TDD: the left-to-right beam search loop (``run_beam``).

Covers the five required scenarios:
1. Multi-impact: flight→bounce→flight→roll → winner segments the bounces.
2. Manual-anchor forcing: a manual breakpoint mid-flight is never spanned.
3. Spurious breaks: ~10 closely-spaced low-score breaks over one clean arc
   collapse into ≤2 segments via skip/absorb.
4. Determinism: two identical runs → identical winning partition.
5. runner_up partition differs from the winner (or is None).

Breakpoints are built by hand here — full breakpoint construction is Task 7.
"""

from __future__ import annotations

import numpy as np

from src.utils.ball_mode_search import (
    Breakpoint,
    Mode,
    _SegmentSolver,
    run_beam,
)
from src.utils.ball_piecewise_solver import SolverCfg
from src.utils.ball_mode_search import ModeSearchCfg
from tests.fixtures.ball_synthetic import (
    FPS,
    ballistic_worlds,
    broadcast_camera,
    per_frame_cams,
    project_track,
    rolling_worlds,
    steps_from_pixels,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _solver_kwargs(worlds, n_frames, *, p_flight=0.0, cfg=None):
    K, R, t = broadcast_camera()
    pixels = project_track(worlds, K, R, t)
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
    ), pixels


def _uv_lookup(pixels):
    def ball_uv_at(frame):
        return pixels.get(int(frame))
    return ball_uv_at


def _bp(frame, *, kind="touch", event_score=0.0, is_manual=False):
    return Breakpoint(frame=frame, kind=kind, event_score=event_score,
                      is_manual=is_manual)


def _boundary(frame):
    return Breakpoint(frame=frame, kind="boundary", event_score=0.0,
                      is_manual=False)


def _partition(hyp):
    """A hashable signature of a hypothesis partition: frame ranges + modes."""
    return tuple((s.fa, s.fb, int(s.mode)) for s in hyp.segments)


# ---------------------------------------------------------------------------
# Test 1 — multi-impact: flight → bounce → flight → roll
# ---------------------------------------------------------------------------

def test_multi_impact_resolves_segments_at_bounces():
    """A flight that bounces, flies again, then rolls.  Event breakpoints sit
    at the two bounce frames; the winner must place segment boundaries there
    and label flight before / after the bounce."""
    n = 70
    # Phase A: ballistic with a bounce ~frame 24 (restitution arc).
    arc = ballistic_worlds((20.0, 30.0, 0.5), (9.0, 1.0, 7.5), 50,
                           restitution=0.6)
    # Find the bounce frame (local z minimum / restitution kick).
    zs = np.array([arc[f][2] for f in range(50)])
    # bounce ~ where z dips toward the floor; pick a clear value.
    bounce1 = 22
    # Phase B (roll) after the ball settles: append a clean ground roll.
    worlds = {f: arc[f] for f in range(40)}
    last_xy = worlds[39][:2]
    roll = rolling_worlds(tuple(last_xy), (0.4, 0.05), n - 40, start_frame=40)
    worlds.update(roll)

    kwargs, pixels = _solver_kwargs(worlds, n)
    seg = _SegmentSolver(**kwargs)
    ball_uv_at = _uv_lookup(pixels)

    breakpoints = [
        _boundary(0),
        _bp(bounce1, kind="bounce", event_score=0.9),
        _bp(40, kind="ground", event_score=0.85),
        _boundary(n - 1),
    ]
    res = run_beam(seg, breakpoints, n, ModeSearchCfg(), ball_uv_at=ball_uv_at)

    win = res.winner
    assert win is not None
    # At least 3 segments resolved.
    assert len(win.segments) >= 3, (
        f"expected >=3 segments, got {len(win.segments)}: "
        f"{[(s.fa, s.fb, s.mode.name) for s in win.segments]}"
    )
    # The bounce frame and the ground-contact frame are segment boundaries.
    boundary_frames = {s.fa for s in win.segments} | {s.fb for s in win.segments}
    assert bounce1 in boundary_frames
    assert 40 in boundary_frames
    # The last segment over the clean roll is ROLLING.
    assert win.segments[-1].mode is Mode.ROLLING
    # There is at least one FLIGHT segment in the airborne portion.
    assert any(s.mode is Mode.FLIGHT for s in win.segments)


# ---------------------------------------------------------------------------
# Test 2 — manual-anchor forcing: never spanned
# ---------------------------------------------------------------------------

def test_manual_anchor_is_never_spanned():
    """A manual breakpoint mid-flight must be a hard segment boundary —
    no winning segment's [fa,fb] may strictly contain it."""
    n = 50
    worlds = ballistic_worlds((20.0, 30.0, 0.5), (9.0, 1.0, 8.0), n)
    kwargs, pixels = _solver_kwargs(worlds, n)
    seg = _SegmentSolver(**kwargs)
    ball_uv_at = _uv_lookup(pixels)

    manual_frame = 25
    breakpoints = [
        _boundary(0),
        # A few spurious low-score breaks the beam would otherwise absorb.
        _bp(12, event_score=0.1),
        _bp(manual_frame, kind="manual", event_score=0.0, is_manual=True),
        _bp(38, event_score=0.1),
        _boundary(n - 1),
    ]
    res = run_beam(seg, breakpoints, n, ModeSearchCfg(), ball_uv_at=ball_uv_at)

    win = res.winner
    assert win is not None
    for s in win.segments:
        assert not (s.fa < manual_frame < s.fb), (
            f"segment {s.fa}-{s.fb} strictly spans the manual anchor "
            f"at {manual_frame}"
        )
    # The manual frame must actually appear as a boundary.
    boundary_frames = {s.fa for s in win.segments} | {s.fb for s in win.segments}
    assert manual_frame in boundary_frames


# ---------------------------------------------------------------------------
# Test 3 — spurious breaks collapse via skip/absorb
# ---------------------------------------------------------------------------

def test_spurious_breaks_collapse_into_few_segments():
    """~10 closely-spaced low-score velocity-break breakpoints over one clean
    gravity arc should be absorbed into ≤2 segments."""
    n = 60
    worlds = ballistic_worlds((18.0, 28.0, 0.5), (9.0, 1.5, 8.5), n)
    kwargs, pixels = _solver_kwargs(worlds, n)
    seg = _SegmentSolver(**kwargs)
    ball_uv_at = _uv_lookup(pixels)

    # 10 spurious low-score breaks spread across the arc.
    spurious = [
        _bp(f, kind="velocity_break", event_score=0.05)
        for f in range(8, 52, 4)
    ]
    breakpoints = [_boundary(0), *spurious, _boundary(n - 1)]

    # With max_skip large enough to bridge all the spurious breaks, the clean
    # arc collapses to a single FLIGHT segment (≤2) — the skip/absorb merge.
    res = run_beam(seg, breakpoints, n, ModeSearchCfg(max_skip=len(spurious)),
                   ball_uv_at=ball_uv_at)
    win = res.winner
    assert win is not None
    assert len(win.segments) <= 2, (
        f"clean arc should collapse to <=2 segments, got {len(win.segments)}: "
        f"{[(s.fa, s.fb, s.mode.name) for s in win.segments]}"
    )
    assert any(s.mode is Mode.FLIGHT for s in win.segments)

    # With the DEFAULT (bounded) max_skip=3, branching is still bounded: the
    # 10 spurious breaks cannot explode into 10 segments — each span absorbs
    # up to 3 breaks, so the winner stays at a handful of segments (≤4), not
    # one per break.  This is the bounded-branching guarantee.
    res_bounded = run_beam(seg, breakpoints, n, ModeSearchCfg(),
                           ball_uv_at=_uv_lookup(pixels))
    assert len(res_bounded.winner.segments) <= 4, (
        f"bounded max_skip should keep segment count small, got "
        f"{len(res_bounded.winner.segments)}"
    )


# ---------------------------------------------------------------------------
# Test 4 — determinism
# ---------------------------------------------------------------------------

def test_determinism_identical_runs():
    n = 60
    arc = ballistic_worlds((20.0, 30.0, 0.5), (9.0, 1.0, 7.5), 40,
                           restitution=0.6)
    worlds = {f: arc[f] for f in range(40)}
    roll = rolling_worlds(tuple(worlds[39][:2]), (0.4, 0.05), n - 40,
                          start_frame=40)
    worlds.update(roll)

    breakpoints = [
        _boundary(0),
        _bp(22, kind="bounce", event_score=0.9),
        _bp(40, kind="ground", event_score=0.85),
        _boundary(n - 1),
    ]

    kwargs1, pixels1 = _solver_kwargs(worlds, n)
    seg1 = _SegmentSolver(**kwargs1)
    res1 = run_beam(seg1, breakpoints, n, ModeSearchCfg(),
                    ball_uv_at=_uv_lookup(pixels1))

    kwargs2, pixels2 = _solver_kwargs(worlds, n)
    seg2 = _SegmentSolver(**kwargs2)
    res2 = run_beam(seg2, breakpoints, n, ModeSearchCfg(),
                    ball_uv_at=_uv_lookup(pixels2))

    assert _partition(res1.winner) == _partition(res2.winner)


# ---------------------------------------------------------------------------
# Test 5 — runner_up partition differs from winner (or is None)
# ---------------------------------------------------------------------------

def test_runner_up_partition_distinct():
    n = 50
    arc = ballistic_worlds((20.0, 30.0, 0.5), (9.0, 1.0, 7.5), 30,
                           restitution=0.6)
    worlds = {f: arc[f] for f in range(30)}
    roll = rolling_worlds(tuple(worlds[29][:2]), (0.4, 0.05), n - 30,
                          start_frame=30)
    worlds.update(roll)

    breakpoints = [
        _boundary(0),
        _bp(14, kind="bounce", event_score=0.7),
        _bp(30, kind="ground", event_score=0.7),
        _boundary(n - 1),
    ]
    kwargs, pixels = _solver_kwargs(worlds, n)
    seg = _SegmentSolver(**kwargs)
    res = run_beam(seg, breakpoints, n, ModeSearchCfg(),
                   ball_uv_at=_uv_lookup(pixels))

    if res.runner_up is not None:
        assert _partition(res.runner_up) != _partition(res.winner), (
            "runner_up must be partition-distinct from the winner (F11)"
        )


# ---------------------------------------------------------------------------
# Test 6 — BudgetExceeded propagates out of run_beam
# ---------------------------------------------------------------------------

def test_budget_exceeded_propagates():
    import pytest
    from src.utils.ball_mode_search import BudgetExceeded

    n = 40
    worlds = ballistic_worlds((20.0, 30.0, 0.5), (9.0, 1.0, 8.0), n)
    kwargs, pixels = _solver_kwargs(worlds, n)
    seg = _SegmentSolver(**kwargs, mode_cfg=ModeSearchCfg(max_segment_fit_calls=1))

    breakpoints = [
        _boundary(0),
        _bp(20, event_score=0.5),
        _boundary(n - 1),
    ]
    with pytest.raises(BudgetExceeded):
        run_beam(seg, breakpoints, n, ModeSearchCfg(max_segment_fit_calls=1),
                 ball_uv_at=_uv_lookup(pixels))


# ---------------------------------------------------------------------------
# Test 7 — result accounting fields present
# ---------------------------------------------------------------------------

def test_beam_result_accounting():
    n = 40
    worlds = ballistic_worlds((20.0, 30.0, 0.5), (9.0, 1.0, 8.0), n)
    kwargs, pixels = _solver_kwargs(worlds, n)
    seg = _SegmentSolver(**kwargs)
    breakpoints = [_boundary(0), _bp(20, event_score=0.5), _boundary(n - 1)]
    res = run_beam(seg, breakpoints, n, ModeSearchCfg(),
                   ball_uv_at=_uv_lookup(pixels))
    assert res.hypotheses_explored > 0
    assert res.fit_calls > 0
    assert res.winner is not None
    # Winner must span the whole clip: first seg starts at 0, last ends at n-1.
    assert res.winner.segments[0].fa == 0
    assert res.winner.segments[-1].fb == n - 1


# ---------------------------------------------------------------------------
# Test 8 — max_skip config present
# ---------------------------------------------------------------------------

def test_max_skip_in_cfg():
    assert ModeSearchCfg().max_skip == 3
    assert ModeSearchCfg(max_skip=5).max_skip == 5
