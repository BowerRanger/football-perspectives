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
    n = 90
    roll_start = 65
    # Phase A: ballistic with a GENUINE high-energy ground bounce mid-flight.
    # The launch vz (8.0) + restitution 0.7 makes the ball rise to ~3.6 m, land
    # and bounce, then fly a second ~1.6 m arc — two physically-distinct gravity
    # arcs joined at the floor impact, BOTH tall/long enough for the
    # endpoint-free FLIGHT fitter to recover an unambiguous parabola.  (A
    # shallow bounce is monocular-ambiguous: the free-floating fitter cannot
    # resolve the depth of a <0.5 m arc, so the correct labelling would be
    # un-winnable for fitter reasons unrelated to the scoring fix under test.)
    arc = ballistic_worlds((20.0, 30.0, 0.5), (9.0, 1.0, 8.0), roll_start,
                           restitution=0.7)
    # The first ground impact is the floor-contact local z-minimum: the frame
    # whose z is below both neighbours (the restitution kick follows it).
    zs = np.array([arc[f][2] for f in range(roll_start)])
    bounce1 = int(
        next(f for f in range(2, roll_start - 1)
             if zs[f] < zs[f - 1] and zs[f] < zs[f + 1])
    )
    assert 30 <= bounce1 <= 55, f"unexpected bounce frame {bounce1}"
    # Phase B (roll) after the ball settles: append a clean ground roll.
    worlds = {f: arc[f] for f in range(roll_start)}
    last_xy = worlds[roll_start - 1][:2]
    roll = rolling_worlds(tuple(last_xy), (0.4, 0.05), n - roll_start,
                          start_frame=roll_start)
    worlds.update(roll)

    kwargs, pixels = _solver_kwargs(worlds, n)
    seg = _SegmentSolver(**kwargs)
    ball_uv_at = _uv_lookup(pixels)

    breakpoints = [
        _boundary(0),
        _bp(bounce1, kind="bounce", event_score=0.9),
        _bp(roll_start, kind="ground", event_score=0.85),
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
    assert roll_start in boundary_frames
    # The last segment over the clean roll is ROLLING.
    assert win.segments[-1].mode is Mode.ROLLING
    # There is at least one FLIGHT segment in the airborne portion.
    assert any(s.mode is Mode.FLIGHT for s in win.segments)

    # --- C5: physics-aware scoring assertions (reproduce + fix C1) ---------
    # (a) The post-impact airborne arc must be MODELLED as FLIGHT, not gapped:
    #     the segment that STARTS at the bounce frame is FLIGHT.
    seg_at_bounce = next(
        (s for s in win.segments if s.fa == bounce1), None
    )
    assert seg_at_bounce is not None, (
        f"no segment starts at the bounce frame {bounce1}: "
        f"{[(s.fa, s.fb, s.mode.name) for s in win.segments]}"
    )
    assert seg_at_bounce.mode is Mode.FLIGHT, (
        f"post-impact arc starting at {bounce1} must be FLIGHT, got "
        f"{seg_at_bounce.mode.name}: "
        f"{[(s.fa, s.fb, s.mode.name) for s in win.segments]}"
    )

    # (b) NO winning segment may be OUT_OF_VIEW over a span that contains
    #     >=3 confident ball observations — gapping out a span full of
    #     detections is the C1 defect.
    def _n_confident_obs(fa: int, fb: int) -> int:
        return sum(
            1 for f in range(fa, fb + 1)
            if ball_uv_at(f) is not None
        )

    for s in win.segments:
        if s.mode is Mode.OUT_OF_VIEW:
            n_obs = _n_confident_obs(s.fa, s.fb)
            assert n_obs < 3, (
                f"OUT_OF_VIEW segment {s.fa}-{s.fb} spans {n_obs} confident "
                f"ball observations (>=3) — the beam gapped a span full of "
                f"detections (C1): "
                f"{[(s.fa, s.fb, s.mode.name) for s in win.segments]}"
            )


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


# ---------------------------------------------------------------------------
# C2 — structural determinism via partition-signature tie-break
# ---------------------------------------------------------------------------

def _make_hyp(segments, cost):
    """Build a Hypothesis from a list of (fa, fb, mode, player_id) tuples."""
    from src.utils.ball_mode_search import Hypothesis, Segment
    segs = tuple(
        Segment(fa=fa, fb=fb, mode=mode, worlds={}, residual_px=0.0,
                underconstrained=False, kind="x", player_id=pid,
                boundary_vel=(None, None))
        for (fa, fb, mode, pid) in segments
    )
    last = segs[-1]
    return Hypothesis(
        last_bp_idx=last.fb, cur_mode=last.mode, segments=segs,
        cost=cost, end_velocity=None, player_id=last.player_id,
    )


def test_quant_key_is_total_order_on_tied_cost():
    """Two DIFFERENT partitions that tie on (cost, last_frame, mode_int) must
    still order deterministically by the partition signature (C2) — never by
    list append order."""
    from src.utils.ball_mode_search import _quant_key

    # Same cost, same last_frame (10), same cur_mode (FLIGHT), but the FIRST
    # segment's mode differs → different partitions that would otherwise tie.
    h1 = _make_hyp([(0, 5, Mode.FLIGHT, None), (5, 10, Mode.FLIGHT, None)], 12.0)
    h2 = _make_hyp([(0, 5, Mode.ROLLING, None), (5, 10, Mode.FLIGHT, None)], 12.0)

    k1, k2 = _quant_key(h1), _quant_key(h2)
    # The first three fields tie; the signature field must break it.
    assert k1[:3] == k2[:3]
    assert k1 != k2, "partition-signature tie-break field must distinguish them"
    # Ordering is reproducible regardless of insertion order.
    assert sorted([h1, h2], key=_quant_key) == sorted([h2, h1], key=_quant_key)


def test_tied_partitions_resolve_to_same_winner_regardless_of_bp_order():
    """End-to-end determinism: building the SAME breakpoints with the segment
    solver fresh each time yields an identical winning partition, and the
    winner is reproducible (the structural tie-break removes append-order
    dependence even when several partitions tie on cost)."""
    n = 50
    # A clean single arc with several zero-score breakpoints: many partitions
    # (absorbing different subsets of breaks) tie on cost.
    worlds = ballistic_worlds((20.0, 30.0, 0.5), (9.0, 1.0, 8.0), n)

    breakpoints = [
        _boundary(0),
        _bp(15, event_score=0.0),
        _bp(25, event_score=0.0),
        _bp(35, event_score=0.0),
        _boundary(n - 1),
    ]

    winners = []
    for _ in range(5):
        kwargs, pixels = _solver_kwargs(worlds, n)
        seg = _SegmentSolver(**kwargs)
        res = run_beam(seg, breakpoints, n, ModeSearchCfg(max_skip=4),
                       ball_uv_at=_uv_lookup(pixels))
        winners.append(_partition(res.winner))

    # Every run must produce exactly the same winning partition.
    assert len(set(winners)) == 1, (
        f"non-deterministic winner across runs: {set(winners)}"
    )


# ---------------------------------------------------------------------------
# C4 — strictly-increasing breakpoint guard
# ---------------------------------------------------------------------------

def test_duplicate_frame_breakpoints_rejected():
    import pytest
    n = 30
    worlds = ballistic_worlds((20.0, 30.0, 0.5), (9.0, 1.0, 8.0), n)
    kwargs, pixels = _solver_kwargs(worlds, n)
    seg = _SegmentSolver(**kwargs)
    # Duplicate frame 15.
    breakpoints = [_boundary(0), _bp(15), _bp(15), _boundary(n - 1)]
    with pytest.raises(ValueError, match="strictly increasing"):
        run_beam(seg, breakpoints, n, ModeSearchCfg(),
                 ball_uv_at=_uv_lookup(pixels))


def test_out_of_order_breakpoints_rejected():
    import pytest
    n = 30
    worlds = ballistic_worlds((20.0, 30.0, 0.5), (9.0, 1.0, 8.0), n)
    kwargs, pixels = _solver_kwargs(worlds, n)
    seg = _SegmentSolver(**kwargs)
    # Frame 20 before frame 10 — out of order.
    breakpoints = [_boundary(0), _bp(20), _bp(10), _boundary(n - 1)]
    with pytest.raises(ValueError, match="strictly increasing"):
        run_beam(seg, breakpoints, n, ModeSearchCfg(),
                 ball_uv_at=_uv_lookup(pixels))
