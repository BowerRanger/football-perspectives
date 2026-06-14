"""Regression TDD: resolved anchor worlds are threaded into the beam.

The global mode-search solver regressed coverage on well-anchored clips
because ``run_beam`` fit every FLIGHT segment endpoint-free — even across
resolved anchor nodes — discarding the anchor DEPTH constraints that make
piecewise reach 0.97 on origi01's 60 manual anchors.  Endpoint-free flight
over a depth-ambiguous span comes out underconstrained / high-residual, so
OUT_OF_VIEW (cost ∝ observed frames) wins → missing frames where piecewise
solved flight.

These tests pin the fix:

1. ``run_beam`` WITH ``node_world_by_frame`` populated at both ends of a
   depth-ambiguous arc → the winning span is FLIGHT (not OUT_OF_VIEW) with
   worlds matching truth tightly.  WITHOUT node worlds (endpoint-free) the
   same arc is worse — wrong-depth worlds or OUT_OF_VIEW chosen.
2. ``solve_modes`` with anchor nodes bracketing a flight span → the rendered
   SolveResult labels that span ``"flight"`` with accurate worlds (not
   missing / occluded).
"""

from __future__ import annotations

import numpy as np

from src.utils.ball_mode_search import (
    Breakpoint,
    Mode,
    ModeSearchCfg,
    _SegmentSolver,
    run_beam,
    solve_modes,
)
from src.utils.ball_piecewise_solver import SolverCfg, TrajectoryNode
from tests.fixtures.ball_synthetic import (
    FPS,
    ballistic_worlds,
    broadcast_camera,
    per_frame_cams,
    project_track,
    steps_from_pixels,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _solver_kwargs(worlds, n_frames, *, nodes=(), p_flight=0.0, cfg=None,
                   pixel_noise=0.0, seed=0):
    K, R, t = broadcast_camera()
    pixels = project_track(worlds, K, R, t)
    if pixel_noise:
        # Pixel noise on a short, shallow arc surfaces the monocular depth
        # degeneracy: the endpoint-free parabola fit drifts to an absurd depth
        # (the regression mechanism), while anchor knots pin it to truth.
        rng = np.random.default_rng(seed)
        pixels = {
            f: (uv[0] + rng.normal(0.0, pixel_noise),
                uv[1] + rng.normal(0.0, pixel_noise))
            for f, uv in pixels.items()
        }
    steps = steps_from_pixels(pixels, n_frames, p_flight=p_flight)
    per_K, per_R, per_t = per_frame_cams(n_frames)
    return dict(
        nodes=nodes,
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


def _boundary(frame):
    return Breakpoint(frame=frame, kind="boundary", event_score=0.0,
                      is_manual=False)


def _depth_ambiguous_arc(n=10):
    """A short, shallow gravity arc whose monocular pixel track is
    depth-ambiguous under realistic pixel noise.

    A gentle launch (apex <~0.5 m) over only ~10 frames leaves the arc's depth
    weakly observed: with a few pixels of detection noise the endpoint-free
    parabola fit drifts to an absurd depth (the production regression), while an
    anchor knot at each end pins it to truth.  The arc stays above ground so the
    plausibility guard does not pre-reject the TRUE worlds.
    """
    p0 = (30.0, 34.0, 0.3)
    v0 = (5.0, 1.0, 3.0)   # gentle launch, apex ~0.7 m → depth-ambiguous
    return ballistic_worlds(p0, v0, n), n


# Detection-noise sigma (px) and RNG seed for the depth-ambiguity scenario.
_PIXEL_NOISE_PX = 3.0
_NOISE_SEED = 0


# ---------------------------------------------------------------------------
# Test 1 — beam level: anchor knots make the flight segment win with truth
# ---------------------------------------------------------------------------

def test_run_beam_uses_anchor_worlds_to_keep_anchored_flight():
    worlds, n = _depth_ambiguous_arc()
    fa, fb = 0, n - 1
    kwargs, pixels = _solver_kwargs(
        worlds, n, pixel_noise=_PIXEL_NOISE_PX, seed=_NOISE_SEED,
    )

    bps = [_boundary(fa), _boundary(fb)]
    cfg = ModeSearchCfg()

    true_a = np.asarray(worlds[fa], float)
    true_b = np.asarray(worlds[fb], float)
    node_worlds = {fa: true_a, fb: true_b}

    # --- WITH resolved anchor worlds at both ends ---
    seg_anch = _SegmentSolver(**kwargs)
    res_anch = run_beam(
        seg_anch, bps, n, cfg,
        ball_uv_at=_uv_lookup(pixels),
        node_world_by_frame=node_worlds,
    )
    win_anch = res_anch.winner
    # One segment spanning the whole arc, labelled FLIGHT (NOT out_of_view).
    modes_anch = [s.mode for s in win_anch.segments]
    assert Mode.FLIGHT in modes_anch, (
        f"anchored beam should keep FLIGHT; got {[m.name for m in modes_anch]}"
    )
    assert Mode.OUT_OF_VIEW not in modes_anch, (
        "anchored, depth-resolved flight must not collapse to OUT_OF_VIEW"
    )
    flight_seg = next(s for s in win_anch.segments if s.mode is Mode.FLIGHT)
    assert flight_seg.fa == fa and flight_seg.fb == fb
    # The pinned endpoints land on the TRUE worlds (the anchor knots).
    assert np.linalg.norm(flight_seg.worlds[fa] - true_a) < 0.25
    assert np.linalg.norm(flight_seg.worlds[fb] - true_b) < 0.25
    # ... and the whole arc is recovered accurately (the anchor depth keeps the
    # fit on the true arc despite the pixel noise that derails the free fit).
    for f in (n // 4, n // 2, 3 * n // 4):
        assert np.linalg.norm(flight_seg.worlds[f] - worlds[f]) < 1.0
    assert not flight_seg.underconstrained

    # --- WITHOUT anchor worlds (endpoint-free) on the same arc ---
    seg_free = _SegmentSolver(**kwargs)
    res_free = run_beam(
        seg_free, bps, n, cfg,
        ball_uv_at=_uv_lookup(pixels),
        # no node_world_by_frame → endpoint-free
    )
    win_free = res_free.winner
    free_modes = [s.mode for s in win_free.segments]
    free_flight = next(
        (s for s in win_free.segments if s.mode is Mode.FLIGHT), None
    )

    # The endpoint-free result must be strictly worse on this depth-ambiguous
    # arc: either it gave up to OUT_OF_VIEW, OR a flight was kept but its worlds
    # are markedly less accurate than the anchored fit (the depth is wrong).
    if free_flight is None:
        assert Mode.OUT_OF_VIEW in free_modes, (
            "endpoint-free beam should fall back to OUT_OF_VIEW when it cannot "
            f"resolve the arc; got {[m.name for m in free_modes]}"
        )
    else:
        anch_err = max(
            np.linalg.norm(flight_seg.worlds[f] - worlds[f])
            for f in range(fa, fb + 1)
        )
        free_err = max(
            np.linalg.norm(free_flight.worlds[f] - worlds[f])
            for f in range(fa, fb + 1)
        )
        assert free_err > anch_err + 0.25, (
            "endpoint-free flight should be less accurate than the "
            f"anchor-pinned flight (free={free_err:.3f}, anch={anch_err:.3f})"
        )


# ---------------------------------------------------------------------------
# Test 2 — solve_modes level: anchor nodes bracket a flight span
# ---------------------------------------------------------------------------

def test_solve_modes_anchor_nodes_render_flight_with_accurate_worlds():
    worlds, n = _depth_ambiguous_arc()
    fa, fb = 0, n - 1
    true_a = worlds[fa]
    true_b = worlds[fb]

    nodes = (
        TrajectoryNode(
            frame=fa, world_xyz=tuple(float(x) for x in true_a),
            state="flight", confidence=1.0, is_manual=True,
        ),
        TrajectoryNode(
            frame=fb, world_xyz=tuple(float(x) for x in true_b),
            state="flight", confidence=1.0, is_manual=True,
        ),
    )
    kwargs, _ = _solver_kwargs(
        worlds, n, nodes=nodes,
        pixel_noise=_PIXEL_NOISE_PX, seed=_NOISE_SEED,
    )

    res = solve_modes(**kwargs)

    # state_by_frame covers every frame.
    assert set(res.state_by_frame) == set(range(n))

    # The bracketed span renders as flight (not missing / occluded) at the
    # interior — the anchored depth keeps the FLIGHT explanation alive.
    interior = list(range(fa + 1, fb))
    flight_frames = [f for f in interior if res.state_by_frame[f] == "flight"]
    assert len(flight_frames) >= len(interior) * 0.8, (
        "anchor-bracketed flight span should mostly render as flight; "
        f"got states {[res.state_by_frame[f] for f in interior]}"
    )

    # Recovered worlds are accurate against the true arc.
    for f in (n // 4, n // 2, 3 * n // 4):
        w, _conf = res.world_by_frame[f]
        assert np.linalg.norm(np.asarray(w, float) - worlds[f]) < 1.0, (
            f"frame {f} world off truth: {w} vs {worlds[f]}"
        )
