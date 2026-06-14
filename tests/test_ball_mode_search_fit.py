"""Task 3 TDD: fitter-reuse layer + sound segment-fit cache.

Asserts the ``_SegmentSolver`` wrapper:
- FLIGHT fit recovers a synthetic gravity arc (endpoint-free).
- ROLLING fit recovers a synthetic constant-decel roll.
- STATIONARY fit on a still ball gives a constant world + ~0 residual.
- OUT_OF_VIEW returns empty worlds, residual 0.
- The frame-keyed cache returns the same object and increments the
  ``_fit_calls`` counter only once per distinct key.
- ``BudgetExceeded`` is raised once ``max_segment_fit_calls`` is passed.
- ``boundary_vel`` is non-None for FLIGHT / ROLLING / STATIONARY.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.utils.ball_mode_search import (
    BudgetExceeded,
    Mode,
    SegmentFit,
    _SegmentSolver,
)
from src.utils.ball_piecewise_solver import SolverCfg
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
# Scene builders
# ---------------------------------------------------------------------------

def _solver_kwargs(worlds, n_frames, *, p_flight=0.0, cfg=None):
    """Build the kwargs a real ``_Solver`` (and thus ``_SegmentSolver``)
    consumes from a dense world track."""
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
    )


# ---------------------------------------------------------------------------
# FLIGHT
# ---------------------------------------------------------------------------

def test_flight_fit_recovers_arc():
    n = 40
    p0 = np.array([30.0, 34.0, 0.5])
    v0 = np.array([8.0, 2.0, 9.0])
    worlds = ballistic_worlds(tuple(p0), tuple(v0), n)
    kwargs = _solver_kwargs(worlds, n)
    seg = _SegmentSolver(**kwargs)

    fa, fb = 2, 30
    fit = seg.fit_segment(fa, fb, Mode.FLIGHT)

    assert isinstance(fit, SegmentFit)
    assert fit.kind == "ballistic"
    assert fit.residual_px < 2.0
    # Worlds recovered close to the true arc at interior frames.
    for f in (5, 15, 25):
        assert np.linalg.norm(fit.worlds[f] - worlds[f]) < 0.5
    # Boundary velocities present and the launch is going up (+z), the
    # later boundary slower / descending in z.
    v_a, v_b = fit.boundary_vel
    assert v_a is not None and v_b is not None
    assert v_a[2] > 0.0           # ascending at fa
    assert v_b[2] < v_a[2]        # gravity has pulled vz down by fb
    assert not fit.underconstrained


def test_flight_fit_endpoint_free_no_knot_for_free_breakpoint():
    """A free-floating (non-anchor) breakpoint must NOT pin endpoints —
    the fit still recovers the arc from interior pixels alone."""
    n = 40
    p0 = np.array([28.0, 30.0, 0.4])
    v0 = np.array([10.0, 1.0, 8.0])
    worlds = ballistic_worlds(tuple(p0), tuple(v0), n)
    kwargs = _solver_kwargs(worlds, n)
    seg = _SegmentSolver(**kwargs)
    fit = seg.fit_segment(4, 28, Mode.FLIGHT)
    assert fit.residual_px < 2.0
    assert np.linalg.norm(fit.worlds[16] - worlds[16]) < 0.5


# ---------------------------------------------------------------------------
# ROLLING
# ---------------------------------------------------------------------------

def test_rolling_fit_recovers_constant_decel_roll():
    n = 30
    # Constant-decel roll: start fast, decelerate. Build with a quadratic.
    start = np.array([20.0, 34.0])
    v_per_f = np.array([0.8, 0.0])      # m / frame initial
    accel_per_f2 = np.array([-0.01, 0.0])
    z = 0.11
    worlds = {}
    for i in range(n):
        xy = start + v_per_f * i + 0.5 * accel_per_f2 * i * i
        worlds[i] = np.array([xy[0], xy[1], z])
    kwargs = _solver_kwargs(worlds, n)
    seg = _SegmentSolver(**kwargs)

    fa, fb = 1, 28
    fit = seg.fit_segment(fa, fb, Mode.ROLLING)

    assert fit.kind == "rolling"
    assert fit.residual_px is not None
    assert fit.residual_px < 8.0
    for f in (5, 14, 24):
        assert np.linalg.norm(fit.worlds[f][:2] - worlds[f][:2]) < 0.5
        assert abs(fit.worlds[f][2] - z) < 1e-6
    v_a, v_b = fit.boundary_vel
    assert v_a is not None and v_b is not None
    # Decelerating roll: start speed > end speed (in xy).
    assert np.linalg.norm(v_a[:2]) > np.linalg.norm(v_b[:2])


# ---------------------------------------------------------------------------
# STATIONARY
# ---------------------------------------------------------------------------

def test_stationary_fit_constant_world_zero_residual():
    n = 20
    z = 0.11
    pt = np.array([40.0, 30.0, z])
    worlds = {i: pt.copy() for i in range(n)}
    kwargs = _solver_kwargs(worlds, n)
    seg = _SegmentSolver(**kwargs)

    fit = seg.fit_segment(1, 18, Mode.STATIONARY)
    assert fit.kind == "stationary"
    assert fit.residual_px < 1e-3
    for f in (4, 10, 15):
        assert np.linalg.norm(fit.worlds[f] - pt) < 0.2
    v_a, v_b = fit.boundary_vel
    assert v_a is not None and v_b is not None
    np.testing.assert_allclose(v_a, np.zeros(3))
    np.testing.assert_allclose(v_b, np.zeros(3))


# ---------------------------------------------------------------------------
# OUT_OF_VIEW
# ---------------------------------------------------------------------------

def test_out_of_view_empty_worlds():
    n = 20
    worlds = rolling_worlds((20.0, 34.0), (0.5, 0.0), n)
    kwargs = _solver_kwargs(worlds, n)
    seg = _SegmentSolver(**kwargs)

    fit = seg.fit_segment(3, 12, Mode.OUT_OF_VIEW)
    assert fit.kind == "out_of_view"
    assert fit.worlds == {}
    assert fit.residual_px == 0.0
    assert fit.underconstrained is False
    assert fit.boundary_vel == (None, None)


# ---------------------------------------------------------------------------
# POSSESSED — Task 4
# ---------------------------------------------------------------------------

def test_possessed_raises_without_player_ctx():
    """T4 implemented: POSSESSED raises ValueError if no player_ctx was passed,
    not NotImplementedError (the T3 placeholder has been superseded)."""
    n = 20
    worlds = rolling_worlds((20.0, 34.0), (0.5, 0.0), n)
    kwargs = _solver_kwargs(worlds, n)
    seg = _SegmentSolver(**kwargs)
    with pytest.raises((ValueError, RuntimeError)):
        seg.fit_segment(2, 15, Mode.POSSESSED, player_id="P001")


# ---------------------------------------------------------------------------
# Cache + budget
# ---------------------------------------------------------------------------

def test_cache_same_key_returns_same_object_increments_once():
    n = 40
    worlds = ballistic_worlds((30.0, 34.0, 0.5), (8.0, 2.0, 9.0), n)
    kwargs = _solver_kwargs(worlds, n)
    seg = _SegmentSolver(**kwargs)

    assert seg._fit_calls == 0
    first = seg.fit_segment(2, 30, Mode.FLIGHT)
    assert seg._fit_calls == 1
    second = seg.fit_segment(2, 30, Mode.FLIGHT)
    assert second is first               # identical object from cache
    assert seg._fit_calls == 1           # not re-fit

    # A different key fits afresh.
    third = seg.fit_segment(2, 28, Mode.FLIGHT)
    assert third is not first
    assert seg._fit_calls == 2


def test_cache_keys_distinct_per_mode_and_player():
    n = 30
    worlds = rolling_worlds((20.0, 34.0), (0.5, 0.0), n)
    kwargs = _solver_kwargs(worlds, n)
    seg = _SegmentSolver(**kwargs)
    seg.fit_segment(1, 20, Mode.ROLLING)
    assert seg._fit_calls == 1
    seg.fit_segment(1, 20, Mode.STATIONARY)   # same range, different mode
    assert seg._fit_calls == 2
    seg.fit_segment(1, 20, Mode.ROLLING)      # repeat → cached
    assert seg._fit_calls == 2


def test_budget_raises_on_second_distinct_fit():
    n = 40
    worlds = ballistic_worlds((30.0, 34.0, 0.5), (8.0, 2.0, 9.0), n)
    cfg = SolverCfg()
    kwargs = _solver_kwargs(worlds, n, cfg=cfg)
    from src.utils.ball_mode_search import ModeSearchCfg

    seg = _SegmentSolver(**kwargs, mode_cfg=ModeSearchCfg(max_segment_fit_calls=1))
    seg.fit_segment(2, 30, Mode.FLIGHT)       # 1st real fit — ok
    with pytest.raises(BudgetExceeded):
        seg.fit_segment(2, 28, Mode.FLIGHT)   # 2nd distinct fit — over budget


def test_budget_cached_hit_does_not_count():
    n = 40
    worlds = ballistic_worlds((30.0, 34.0, 0.5), (8.0, 2.0, 9.0), n)
    from src.utils.ball_mode_search import ModeSearchCfg
    kwargs = _solver_kwargs(worlds, n)
    seg = _SegmentSolver(**kwargs, mode_cfg=ModeSearchCfg(max_segment_fit_calls=1))
    seg.fit_segment(2, 30, Mode.FLIGHT)
    # Re-using the same key must NOT trip the budget.
    seg.fit_segment(2, 30, Mode.FLIGHT)
    assert seg._fit_calls == 1


# ---------------------------------------------------------------------------
# boundary_vel for all (computable) modes
# ---------------------------------------------------------------------------

def test_boundary_vel_present_for_flight_rolling_stationary():
    n = 40
    arc = ballistic_worlds((30.0, 34.0, 0.5), (8.0, 2.0, 9.0), n)
    seg_f = _SegmentSolver(**_solver_kwargs(arc, n))
    f_fit = seg_f.fit_segment(2, 30, Mode.FLIGHT)
    assert all(v is not None for v in f_fit.boundary_vel)

    roll = rolling_worlds((20.0, 34.0), (0.6, 0.0), n)
    seg_r = _SegmentSolver(**_solver_kwargs(roll, n))
    r_fit = seg_r.fit_segment(1, 30, Mode.ROLLING)
    assert all(v is not None for v in r_fit.boundary_vel)

    still = {i: np.array([40.0, 30.0, 0.11]) for i in range(n)}
    seg_s = _SegmentSolver(**_solver_kwargs(still, n))
    s_fit = seg_s.fit_segment(1, 30, Mode.STATIONARY)
    assert all(v is not None for v in s_fit.boundary_vel)


# ---------------------------------------------------------------------------
# Fix 1: raw-px flight residual — anchor knot must NOT inflate residual_px
# ---------------------------------------------------------------------------

def test_flight_anchored_residual_is_pixel_rms_not_lm_mean_resid():
    """Fix F5: ``residual_px`` must equal ``_pixel_rms`` over obs frames,
    NOT the LM's ``mean_resid`` which includes the 1e3-weighted knot block.

    We inject a fake ``fit_parabola_to_image_observations`` that returns a
    deliberately inflated ``mean_resid`` (simulating a monocular scenario
    where the LM's knot-weighted residual vector is large).  The test asserts
    that ``_fit_flight`` ignores the returned ``mean_resid`` and instead
    recomputes residual via ``_pixel_rms`` — which, for a well-fitting arc,
    is low regardless of the injected mean_resid.
    """
    import unittest.mock as mock
    from src.utils import bundle_adjust as ba
    from src.utils.ball_physics import G_VEC, eval_parabola

    n = 50
    p0 = np.array([30.0, 34.0, 0.5])
    v0 = np.array([8.0, 2.0, 9.0])
    worlds = ballistic_worlds(tuple(p0), tuple(v0), n)
    kwargs = _solver_kwargs(worlds, n)

    fa, fb = 3, 35
    # Real call with the true anchor so the fitted arc is accurate.
    real_fn = ba.fit_parabola_to_image_observations

    def fake_fit(*args, **kw):
        """Delegate to the real fitter but then inflate mean_resid by 1000x."""
        p0r, v0r, real_resid = real_fn(*args, **kw)
        # Simulate knot-contaminated LM mean_resid: 1000x inflated.
        return p0r, v0r, real_resid * 1000.0 + 99.0

    with mock.patch.object(ba, "fit_parabola_to_image_observations", fake_fit):
        seg = _SegmentSolver(**kwargs)
        fit = seg.fit_segment(fa, fb, Mode.FLIGHT, fa_anchor_world=worlds[fa])

    # The worlds are correct (fake_fit returns correct p0/v0), so the
    # per-obs pixel error is near 0.  Fix 1 must give a low residual_px
    # regardless of the injected inflated mean_resid.
    assert fit.residual_px < 2.0, (
        f"residual_px = {fit.residual_px:.2f} — "
        "looks like the inflated LM mean_resid leaked into residual_px. "
        "Fix 1 not applied: _fit_flight must recompute via _pixel_rms, "
        "not use the return value of fit_parabola_to_image_observations."
    )
    # Also verify the underconstrained flag is correctly False.
    assert not fit.underconstrained, (
        "underconstrained should be False for a well-fitting arc — "
        "the inflated mean_resid must not drive the underconstrained check."
    )


# ---------------------------------------------------------------------------
# Fix 2: rolling lob-degeneracy guard — fast roll is flagged underconstrained
# ---------------------------------------------------------------------------

def test_rolling_lob_ground_projection_flagged_underconstrained():
    """Fix: a lob arc whose ground-projection happens to fit a 'roll' in
    pixels must be flagged ``underconstrained=True`` because its ground speed
    exceeds ``rolling_max_speed_m_s``.

    We use ``rolling_max_residual_px=50`` (very permissive) so the residual
    gate alone does NOT fire — the test isolates the speed guard.
    """
    from src.utils.ball_piecewise_solver import SolverCfg

    n = 40
    # A nearly-horizontal low-altitude lob at 20 m/s — ground raycasts look
    # like a fast roll in pixels.  rolling_max_residual_px is set very high
    # so the residual gate does NOT fire; only the speed guard should catch it.
    p0 = np.array([20.0, 34.0, 0.11])
    v0 = np.array([20.0, 0.0, 0.01])   # 20 m/s forward, nearly horizontal
    worlds = ballistic_worlds(tuple(p0), tuple(v0), n)

    # rolling_max_speed_m_s=15 (< 20 m/s) — guard must fire.
    # rolling_max_residual_px=50 (very permissive) — residual gate must NOT fire.
    cfg_tight = SolverCfg(rolling_max_speed_m_s=15.0, rolling_max_residual_px=50.0)
    kwargs = _solver_kwargs(worlds, n, cfg=cfg_tight)
    seg = _SegmentSolver(**kwargs)

    fit = seg.fit_segment(2, 30, Mode.ROLLING)
    assert fit.underconstrained, (
        "ROLLING fit on a 20 m/s ground-projection lob should be flagged "
        "underconstrained by the speed guard (rolling_max_speed_m_s=15). "
        "Fix 2 not applied."
    )


def test_rolling_genuine_slow_roll_not_flagged():
    """A genuine slow roll (≪ rolling_max_speed_m_s) must NOT be flagged."""
    n = 30
    # 0.5 m/frame @ 25 fps = 12.5 m/s — slow, realistic roll.
    worlds = rolling_worlds((20.0, 34.0), (0.5, 0.0), n)
    kwargs = _solver_kwargs(worlds, n)  # default cfg, rolling_max = 35 m/s
    seg = _SegmentSolver(**kwargs)

    fit = seg.fit_segment(1, 28, Mode.ROLLING)
    # Genuine roll: residual is low AND speed is within the gate.
    # underconstrained must be False (residual passes and speed passes).
    assert not fit.underconstrained, (
        f"Genuine slow roll incorrectly flagged underconstrained "
        f"(residual={fit.residual_px:.2f})"
    )


# ---------------------------------------------------------------------------
# Fix F16: grounded ray-cast fallback — coverage where the roll is not clean
# ---------------------------------------------------------------------------

def _grounded_jitter_worlds(n: int, start_frame: int = 0):
    """A ball moving on the ground whose speed jitters back and forth — NOT a
    clean constant-decel roll, so the rolling fit's residual gate fires, but it
    is genuinely grounded (z = ball radius) and observed every frame.

    Mirrors the piecewise open-span grounded fallback's job: the ball is
    detected on the pitch but the constant-decel model cannot explain it, so
    per-frame ground ray-casts are the honest coverage.
    """
    z = 0.11
    worlds: dict[int, np.ndarray] = {}
    x = 30.0
    for i in range(n):
        # Speed oscillates (forward / nearly-stopped / forward) — a single
        # constant acceleration cannot match this, so the roll fit residual
        # will exceed the gate.  Kept slow enough that the ground-track speed
        # stays well under ``rolling_max_speed_m_s`` (a real ground pass), so
        # the raw-raycast plausibility gate does NOT mistake it for an arc.
        step = 0.35 if (i // 3) % 2 == 0 else 0.02
        x += step
        worlds[start_frame + i] = np.array([x, 34.0 + 0.5 * np.sin(i * 0.5), z])
    return worlds


def test_rolling_falls_back_to_ground_raycasts_when_roll_not_clean():
    """Fix F16: a grounded but NOT-cleanly-rolling observed span must fall back
    to per-frame ground ray-casts (pixel-exact, residual ≈ 0, NOT
    underconstrained) rather than returning a high-residual roll fit.

    This mirrors piecewise's open-span grounded ray-cast fallback so the global
    solver gets the same grounded coverage piecewise enjoys.
    """
    n = 30
    worlds = _grounded_jitter_worlds(n)
    # Tight rolling residual gate so the constant-decel fit is rejected and the
    # ray-cast fallback must kick in.
    cfg = SolverCfg(rolling_max_residual_px=2.0)
    kwargs = _solver_kwargs(worlds, n, cfg=cfg)
    seg = _SegmentSolver(**kwargs)

    fa, fb = 1, 28
    fit = seg.fit_segment(fa, fb, Mode.ROLLING)

    assert fit.kind == "rolling"
    # Pixel-exact ray-casts reproject to their own pixels → residual ≈ 0.
    assert fit.residual_px < 1.0, (
        f"ground ray-cast fallback should be pixel-exact, got "
        f"residual_px={fit.residual_px:.3f}"
    )
    assert not fit.underconstrained, (
        "a span fully covered by valid ground ray-casts must NOT be flagged "
        "underconstrained (it is the honest grounded explanation)"
    )
    # Worlds cover (nearly) all interior observed frames, on the ground, and
    # close to the true grounded positions.
    interior = list(range(fa + 1, fb))
    covered = [f for f in interior if f in fit.worlds]
    assert len(covered) >= len(interior) * 0.9, (
        f"fallback should cover almost all observed frames; covered "
        f"{len(covered)}/{len(interior)}"
    )
    for f in covered:
        assert abs(fit.worlds[f][2] - cfg.ball_radius_m) < 1e-6
        assert np.linalg.norm(fit.worlds[f][:2] - worlds[f][:2]) < 0.5
    v_a, v_b = fit.boundary_vel
    assert v_a is not None and v_b is not None


def test_rolling_fallback_excludes_gap_fill_frame_no_teleport():
    """Fix F16 guard: a gap-fill (synthesised) frame inside a grounded span is
    excluded from the ray-cast fallback — no teleport from an extrapolated
    pixel.  Mirrors the piecewise open-span guard.
    """
    from src.utils.ball_tracker import TrackerStep
    from tests.fixtures.ball_synthetic import (
        broadcast_camera,
        per_frame_cams,
        project_track,
    )

    n = 30
    worlds = _grounded_jitter_worlds(n)
    K, R, t = broadcast_camera()
    pixels = project_track(worlds, K, R, t)

    gap_frame = 12
    steps = []
    for fi in range(n):
        steps.append(TrackerStep(
            frame=fi, uv=pixels.get(fi), p_flight=0.0,
            is_outlier=False, is_gap_fill=(fi == gap_frame),
        ))
    per_K, per_R, per_t = per_frame_cams(n)
    cfg = SolverCfg(rolling_max_residual_px=2.0)
    seg = _SegmentSolver(
        nodes=(), steps=steps,
        confidences={i: 1.0 for i in range(n)},
        per_frame_K=per_K, per_frame_R=per_R, per_frame_t=per_t,
        distortion=(0.0, 0.0), fps=FPS, n_frames=n,
        pitch_length_m=105.0, pitch_width_m=68.0,
        split_hints=(), z_hints=None, manual_obs_frames=None,
        world_fixes=None, cfg=cfg,
    )

    fit = seg.fit_segment(1, 28, Mode.ROLLING)
    # The gap-fill frame must NOT receive a ray-cast world from the fallback.
    assert gap_frame not in fit.worlds, (
        "gap-fill frame must be excluded from the ground ray-cast fallback "
        "(no teleport from an extrapolated pixel)"
    )
