"""Tests for src/utils/ball_hybrid_trajectory.py.

Exercises the module through its real public interface
(``HybridShotCtx``/``Knot``), mirroring
prototypes/ball_hybrid_poc/tests/test_hybrid.py's synthetic-scenario
coverage but against the production module's restructured API
(``resolve_knots`` returns ``Knot`` objects; the top-level entry point is
``finalize_track``, which takes an already-resolved knot set — auto-event
gating is ``ball_hybrid_gating``'s job, tested separately).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np
import pytest

from src.utils.ball_eval import point_ray_distance
from src.utils.ball_hybrid_physics import BALL_RADIUS_M, simulate
from src.utils.ball_hybrid_trajectory import (
    DEFAULT_CFG,
    build_trajectory,
    finalize_track,
    full_cfg,
    is_sharp_knot,
    resolve_knots,
    run_trajectory,
)
from src.utils.ball_hybrid_types import HybridShotCtx

BALL_R = BALL_RADIUS_M


# ---------------------------------------------------------------------------
# Test fixtures: a real HybridShotCtx (no fake needed — it's a plain
# dataclass over per-frame K/R/t dicts) + lightweight anchor/observation
# stand-ins matching the shapes resolve_knots/finalize_track expect.
# ---------------------------------------------------------------------------

@dataclass
class _FakeAnchor:
    frame: int
    image_xy: tuple[float, float] | None
    state: str
    player_id: str | None = None
    bone: str | None = None
    goal_element: str | None = None


@dataclass
class _FakeFix:
    frame: int
    xyz: tuple[float, float, float]


@dataclass
class _FakeObs:
    frame: int
    uv: tuple[float, float]
    conf: float
    source: str = "detector"


def _pinhole_K(fx=1000.0, fy=1000.0, cx=960.0, cy=540.0):
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])


def _make_ctx(n_frames=90, fps=30.0) -> HybridShotCtx:
    K = _pinhole_K()
    R = np.array([
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0],
        [1.0, 0.0, 0.0],
    ])
    C = np.array([-25.0, 5.0, 8.0])
    t = -R @ C
    frames = range(n_frames)
    per_K = {f: K for f in frames}
    per_R = {f: R for f in frames}
    per_t = {f: t for f in frames}
    return HybridShotCtx(clip_id="synthtest", fps=fps, image_size=(1920, 1080),
                          per_frame_K=per_K, per_frame_R=per_R, per_frame_t=per_t,
                          distortion=(0.0, 0.0))


def _find_landing_time(p0, v0, cd, t_max=5.0, n=4000):
    times = np.linspace(0.0, t_max, n)
    traj = simulate(p0, v0, times, cd=cd)
    z = traj[:, 2]
    below = np.where(z <= BALL_R)[0]
    below = below[below > 0]
    if len(below) == 0:
        raise RuntimeError("synthetic scenario: ball never lands within t_max")
    i = int(below[0])
    t0, t1 = times[i - 1], times[i]
    z0, z1 = z[i - 1], z[i]
    frac = (BALL_R - z0) / (z1 - z0)
    return float(t0 + frac * (t1 - t0))


def _synthetic_drag_kick_scenario(fps=30.0, cd=0.30):
    ctx = _make_ctx(fps=fps)
    p_a = np.array([0.0, 0.0, BALL_R])
    v0_true = np.array([13.0, 0.0, 8.5])
    duration_s = _find_landing_time(p_a, v0_true, cd)
    frame_a = 10
    frame_b = frame_a + int(round(duration_s * fps))
    duration_s = (frame_b - frame_a) / fps
    p_b = simulate(p_a, v0_true, [duration_s], cd=cd)[0]
    p_b[2] = BALL_R

    anchors = [
        _FakeAnchor(frame=frame_a, image_xy=tuple(float(x) for x in ctx.project(frame_a, p_a)),
                    state="kick", player_id="P001", bone="right_foot"),
        _FakeAnchor(frame=frame_b, image_xy=tuple(float(x) for x in ctx.project(frame_b, p_b)),
                    state="bounce"),
    ]

    rng = np.random.default_rng(11)
    obs = []
    for frac in np.linspace(0.15, 0.85, 8):
        t_s = frac * duration_s
        frame = int(round(frame_a + t_s * fps))
        p_true = simulate(p_a, v0_true, [t_s], cd=cd)[0]
        uv = ctx.project(frame, p_true) + rng.normal(scale=1.0, size=2)
        obs.append(_FakeObs(frame=frame, uv=(float(uv[0]), float(uv[1])), conf=0.9))

    return ctx, anchors, obs, p_a, p_b, v0_true, cd, frame_a, frame_b, duration_s


def _run(ctx, observations, anchors, fixes=(), cfg=None, player_context=None):
    cfg = full_cfg(cfg)
    hard, ray = resolve_knots(ctx, anchors, fixes, player_context=player_context)
    frames, diag = finalize_track(ctx, hard, ray, observations, cfg)
    return frames, diag, hard, ray


# ---------------------------------------------------------------------------
# resolve_knots
# ---------------------------------------------------------------------------

def test_resolve_knots_splits_hard_vs_ray():
    ctx, anchors, obs, *_ = _synthetic_drag_kick_scenario()
    anchors = anchors + [_FakeAnchor(frame=50, image_xy=(500.0, 500.0), state="airborne_mid")]
    hard, rays = resolve_knots(ctx, anchors)
    assert len(hard) == 2
    assert {k.kind for k in hard} == {"kick", "bounce"}
    assert all(k.depth_hard for k in hard)
    assert len(rays) == 1
    assert rays[0].kind == "airborne_mid"
    assert rays[0].depth_hard is False


def test_resolve_knots_non_event_grounded_anchor_is_hard_but_not_sharp():
    ctx = _make_ctx()
    p_mid_ground = np.array([9.0, 1.0, BALL_R])
    uv = tuple(float(x) for x in ctx.project(20, p_mid_ground))
    anchor = _FakeAnchor(frame=20, image_xy=uv, state="grounded")
    hard, rays = resolve_knots(ctx, [anchor])
    assert rays == []
    assert len(hard) == 1
    assert hard[0].kind == "grounded"
    assert hard[0].depth_hard is True
    assert not is_sharp_knot(hard[0])


def test_resolve_knots_goal_impact_uses_goal_geometry():
    ctx = _make_ctx()
    target = np.array([0.0, 34.0, 2.44])
    uv = tuple(float(x) for x in ctx.project(20, target))
    anchor = _FakeAnchor(frame=20, image_xy=uv, state="goal_impact", goal_element="crossbar")
    hard, rays = resolve_knots(ctx, [anchor])
    assert rays == []
    assert len(hard) == 1
    assert np.linalg.norm(np.array(hard[0].xyz) - target) < 1e-3
    assert hard[0].depth_hard is True


def test_resolve_knots_catch_uses_hand_joint_like_touch():
    ctx = _make_ctx()
    joint = np.array([5.0, 1.0, 1.8])
    uv = tuple(float(x) for x in ctx.project(30, joint))
    anchor = _FakeAnchor(frame=30, image_xy=uv, state="catch", player_id="KEEPER", bone="right_hand")
    pc = SimpleNamespace(joint_world=lambda frame, pid, bone: joint)
    hard, rays = resolve_knots(ctx, [anchor], player_context=pc)
    assert rays == []
    assert len(hard) == 1
    C, d = ctx.ray(30, uv)
    dist_to_joint = float(np.linalg.norm(joint - C))
    dist_to_resolved = float(np.linalg.norm(np.array(hard[0].xyz) - C))
    assert dist_to_resolved == pytest.approx(dist_to_joint - BALL_R, abs=1e-6)


def test_resolve_knots_fix_is_hard_depth_hard_source_fix():
    ctx = _make_ctx()
    fix = _FakeFix(frame=40, xyz=(10.0, 20.0, 3.0))
    hard, rays = resolve_knots(ctx, [], fixes=[fix])
    assert len(hard) == 1
    assert hard[0].source == "fix"
    assert hard[0].depth_hard is True
    assert is_sharp_knot(hard[0])


# ---------------------------------------------------------------------------
# finalize_track end-to-end
# ---------------------------------------------------------------------------

def test_manual_anchors_never_move():
    ctx, anchors, obs, p_a, p_b, v0_true, cd, frame_a, frame_b, _T = _synthetic_drag_kick_scenario()
    frames, diag, *_ = _run(ctx, obs, anchors, cfg={"cd": cd, "fit_cd": True})
    assert frames[frame_a]["mode"] == "anchor"
    assert frames[frame_b]["mode"] == "anchor"
    assert np.allclose(frames[frame_a]["xyz"], p_a, atol=1e-6)
    assert np.allclose(frames[frame_b]["xyz"], p_b, atol=1e-6)


def test_z_never_below_radius():
    ctx, anchors, obs, *_ = _synthetic_drag_kick_scenario()
    frames, *_ = _run(ctx, obs, anchors, cfg={"cd": 0.30})
    for f in frames.values():
        assert f["xyz"][2] >= BALL_RADIUS_M - 1e-9


def test_drag_beats_no_drag_mid_flight():
    ctx, anchors, obs, p_a, p_b, v0_true, cd, frame_a, frame_b, T = (
        _synthetic_drag_kick_scenario(cd=0.30))
    mid_t = T / 2.0
    mid_frame = int(round(frame_a + mid_t * ctx.fps))
    true_mid = simulate(p_a, v0_true, [mid_t], cd=cd)[0]

    frames_drag, *_ = _run(ctx, obs, anchors, cfg={"cd": 0.30, "fit_cd": False})
    frames_nodrag, *_ = _run(ctx, obs, anchors, cfg={"cd": 0.0, "fit_cd": False})

    err_drag = np.linalg.norm(np.array(frames_drag[mid_frame]["xyz"]) - true_mid)
    err_nodrag = np.linalg.norm(np.array(frames_nodrag[mid_frame]["xyz"]) - true_mid)
    assert err_drag < err_nodrag
    assert err_drag < 0.20


def test_faithful_mode_near_confident_observations():
    ctx, anchors, obs, *_ = _synthetic_drag_kick_scenario()
    frames, *_ = _run(ctx, obs, anchors, cfg={"cd": 0.30, "fit_cd": True})
    obs_frames = {o.frame for o in obs}
    faithful_near_obs = [f for f in obs_frames
                          if f in frames and frames[f]["mode"] == "faithful"]
    assert faithful_near_obs


def test_empty_inputs_returns_empty_track():
    ctx = _make_ctx()
    frames, diag, *_ = _run(ctx, [], [])
    assert frames == {}


def test_flight_span_state_is_flight_and_roll_span_state_is_grounded():
    ctx, anchors, obs, p_a, p_b, v0_true, cd, frame_a, frame_b, T = (
        _synthetic_drag_kick_scenario())
    frames, *_ = _run(ctx, obs, anchors, cfg={"cd": cd, "fit_cd": False})
    mid_frame = (frame_a + frame_b) // 2
    assert frames[mid_frame]["state"] == "flight"


def test_open_end_tail_flight_beats_holding_the_knot():
    ctx = _make_ctx()
    cd = 0.0
    p_a = np.array([0.0, 0.0, BALL_R])
    v0_true = np.array([12.0, 1.0, 7.0])
    frame_a = 5
    fps = ctx.fps

    rng = np.random.default_rng(5)
    obs = []
    for frac in np.linspace(0.1, 0.6, 6):
        t_s = frac * 1.0
        frame = int(round(frame_a + t_s * fps))
        p_true = simulate(p_a, v0_true, [t_s], cd=cd)[0]
        uv = ctx.project(frame, p_true) + rng.normal(scale=0.5, size=2)
        obs.append(_FakeObs(frame=frame, uv=(float(uv[0]), float(uv[1])), conf=0.9))

    anchors = [_FakeAnchor(frame=frame_a,
                            image_xy=tuple(float(x) for x in ctx.project(frame_a, p_a)),
                            state="kick")]
    frames, *_ = _run(ctx, obs, anchors, cfg={"cd": 0.0, "fit_cd": False})

    check_t = 0.35
    check_frame = int(round(frame_a + check_t * fps))
    true_p = simulate(p_a, v0_true, [check_t], cd=cd)[0]
    assert check_frame in frames
    got = np.array(frames[check_frame]["xyz"])
    held_err = float(np.linalg.norm(p_a - true_p))
    fit_err = float(np.linalg.norm(got - true_p))
    assert fit_err < held_err
    assert fit_err < 0.5


# ---------------------------------------------------------------------------
# Operator-input-always-wins invariant
# ---------------------------------------------------------------------------

def test_manual_anchor_reprojects_within_ray_faithful_tolerance():
    """Every manual (sharp) anchor's output must reproject within
    ball.ray_faithful_tolerance_px (config/default.yaml) of its click."""
    ray_faithful_tol_px = 3.0  # config/default.yaml: ball.ray_faithful_tolerance_px
    ctx, anchors, obs, p_a, p_b, *_ = _synthetic_drag_kick_scenario()
    frames, *_ = _run(ctx, obs, anchors)
    for a in anchors:
        got = np.array(frames[a.frame]["xyz"])
        uv = ctx.project(a.frame, got)
        err_px = float(np.hypot(uv[0] - a.image_xy[0], uv[1] - a.image_xy[1]))
        assert err_px <= ray_faithful_tol_px


def test_no_auto_knot_survives_within_2_frames_of_manual():
    """Covered structurally: resolve_knots's ``by_frame``/``ray_by_frame``
    dicts key on ``frame`` and manual anchors are always resolved first by
    the caller into ``hard_knots``/``ray_knots`` BEFORE any auto candidate
    is considered by ``ball_hybrid_gating`` (which drops a candidate
    within ``min_frame_gap_from_manual`` frames) — see
    test_ball_hybrid_gating.py's dedicated near-manual test for the
    end-to-end check of that gate."""
    ctx, anchors, obs, *_ = _synthetic_drag_kick_scenario()
    hard, ray = resolve_knots(ctx, anchors)
    manual_frames = {k.frame for k in hard} | {k.frame for k in ray}
    assert manual_frames == {a.frame for a in anchors}


# ---------------------------------------------------------------------------
# The origi01 held-out fix: airborne ray evidence must not pull a flight
# span's Cd fit (hence depth) as hard as ground-constrained evidence.
# ---------------------------------------------------------------------------

def test_airborne_ray_evidence_weight_is_much_smaller_than_grounded():
    assert DEFAULT_CFG["airborne_ray_weight"] < DEFAULT_CFG["grounded_ray_weight"] / 5.0


def test_inconsistent_airborne_ray_does_not_blow_up_flight_depth():
    """A single mildly-inconsistent airborne ray anchor inside a flight
    span must not meaningfully distort the arc's DEPTH at nearby frames —
    the regression this guards against: a depth-only error is invisible
    to 2-D reprojection, so heavily-weighted ray evidence could warp Cd
    (and hence the whole arc's depth) while still reprojecting perfectly
    at its own frame."""
    ctx, anchors, obs, p_a, p_b, v0_true, cd, frame_a, frame_b, T = (
        _synthetic_drag_kick_scenario(cd=0.30))
    mid_frame = (frame_a + frame_b) // 2
    t_s = (mid_frame - frame_a) / ctx.fps
    p_true_mid = simulate(p_a, v0_true, [t_s], cd=cd)[0]

    # A ray anchor whose CLICK is slightly off the true ray direction —
    # plausible click noise on a genuinely airborne point, never resolves
    # to ground_exact/joint_depth (no joint context), so it becomes a
    # ray-only knot.
    noisy_uv = tuple(float(x) for x in (
        ctx.project(mid_frame, p_true_mid) + np.array([6.0, -4.0])))
    ray_anchor = _FakeAnchor(frame=mid_frame, image_xy=noisy_uv, state="airborne_mid")

    baseline_frames, *_ = _run(ctx, obs, anchors, cfg={"cd": cd, "fit_cd": True})
    with_ray_frames, *_ = _run(ctx, obs, anchors + [ray_anchor],
                                cfg={"cd": cd, "fit_cd": True})

    probe_frame = mid_frame - 5
    base_err = np.linalg.norm(
        np.array(baseline_frames[probe_frame]["xyz"])
        - simulate(p_a, v0_true, [(probe_frame - frame_a) / ctx.fps], cd=cd)[0])
    with_ray_err = np.linalg.norm(
        np.array(with_ray_frames[probe_frame]["xyz"])
        - simulate(p_a, v0_true, [(probe_frame - frame_a) / ctx.fps], cd=cd)[0])
    # The ray anchor may still nudge the fit a little (it IS evidence),
    # but must not cause an order-of-magnitude depth blowup nearby.
    assert with_ray_err < base_err + 0.5


# ---------------------------------------------------------------------------
# run_trajectory: the single orchestrating entry point (resolve -> gate ->
# finalize) other callers (ball.py's wiring, bench/hold-out eval scripts)
# should use instead of re-deriving the sequence themselves.
# ---------------------------------------------------------------------------

def test_run_trajectory_matches_manual_resolve_gate_finalize_sequence():
    ctx, anchors, obs, p_a, p_b, v0_true, cd, frame_a, frame_b, _T = (
        _synthetic_drag_kick_scenario())
    frames, diag = run_trajectory(ctx, obs, anchors, cfg={"cd": cd, "fit_cd": True})
    assert frames[frame_a]["mode"] == "anchor"
    assert np.allclose(frames[frame_a]["xyz"], p_a, atol=1e-6)
    assert "gate" in diag
    assert diag["gate"]["n_candidates"] == 0


def test_run_trajectory_folds_in_auto_anchors_via_the_gate():
    ctx, anchors, obs, p_a, p_b, v0_true, cd, frame_a, frame_b, _T = (
        _synthetic_drag_kick_scenario())
    mid_frame = (frame_a + frame_b) // 2
    if mid_frame - frame_a <= 2 or frame_b - mid_frame <= 2:
        mid_frame = frame_a + max(3, (frame_b - frame_a) // 2)
    t_s = (mid_frame - frame_a) / ctx.fps
    p_mid = simulate(p_a, v0_true, [t_s], cd=cd)[0]
    uv_mid = tuple(float(x) for x in ctx.project(mid_frame, p_mid))
    C, d_hat = ctx.ray(mid_frame, uv_mid)
    _, along = point_ray_distance(p_mid, C, d_hat)
    joint = C + (along + BALL_R) * d_hat
    pc = SimpleNamespace(joint_world=lambda frame, pid, bone: joint)

    @dataclass
    class _ScoredAnchor(_FakeAnchor):
        score: float = 0.9

    candidate = _ScoredAnchor(frame=mid_frame, image_xy=uv_mid, state="player_touch",
                               player_id="P099", bone="right_foot")
    frames, diag = run_trajectory(ctx, obs, anchors, auto_anchors=[candidate],
                                   cfg={"cd": cd, "fit_cd": False},
                                   player_context=pc)
    assert diag["gate"]["n_candidates"] == 1
    assert diag["gate"]["n_accepted_hard"] == 1
    assert frames[mid_frame]["mode"] == "anchor"
    assert np.allclose(frames[mid_frame]["xyz"], p_mid, atol=1e-3)


# ---------------------------------------------------------------------------
# IC-E's ball_hybrid_spin.fit_span_spin wiring (solve_span's flight branch,
# gated behind cfg["spin"]["enabled"], default off).
# ---------------------------------------------------------------------------

def test_spin_disabled_by_default_no_omega_in_span_diag():
    ctx, anchors, obs, *_ = _synthetic_drag_kick_scenario()
    frames, diag, *_ = _run(ctx, obs, anchors, cfg={"cd": 0.30, "fit_cd": False})
    for span in diag["spans"]:
        assert "omega_world" not in span
        assert "rad_s" not in span


def test_spin_enabled_fits_omega_on_a_genuinely_spun_trajectory():
    ctx = _make_ctx()
    cd = 0.25
    omega_true = np.array([0.0, 0.0, 25.0])  # pure sidespin, well inside bounds
    p_a = np.array([0.0, 0.0, BALL_R])
    v0_true = np.array([14.0, 0.0, 9.0])
    frame_a = 5
    fps = ctx.fps

    duration_s = _find_landing_time(p_a, v0_true, cd)
    frame_b = frame_a + int(round(duration_s * fps))
    duration_s = (frame_b - frame_a) / fps
    p_b = simulate(p_a, v0_true, [duration_s], cd=cd, omega=omega_true)[0]
    p_b[2] = BALL_R

    anchors = [
        _FakeAnchor(frame=frame_a, image_xy=tuple(float(x) for x in ctx.project(frame_a, p_a)),
                    state="kick", player_id="P001", bone="right_foot"),
        _FakeAnchor(frame=frame_b, image_xy=tuple(float(x) for x in ctx.project(frame_b, p_b)),
                    state="bounce"),
    ]
    obs = []
    for frac in np.linspace(0.1, 0.9, 16):
        t_s = frac * duration_s
        frame = int(round(frame_a + t_s * fps))
        if frame in (frame_a, frame_b):
            continue
        p_true = simulate(p_a, v0_true, [t_s], cd=cd, omega=omega_true)[0]
        uv = ctx.project(frame, p_true)
        obs.append(_FakeObs(frame=frame, uv=(float(uv[0]), float(uv[1])), conf=0.95))

    spin_cfg = {
        "enabled": True, "min_obs": 6, "min_delta_bic": 1.0, "min_resid_gain": 0.02,
    }
    frames, diag, *_ = _run(ctx, obs, anchors,
                             cfg={"cd": cd, "fit_cd": False, "spin": spin_cfg,
                                  # Keep the spin-perturbed points as robust-gate
                                  # INLIERS (rather than discarding most of them
                                  # as outliers against the drag-only arc) so
                                  # fit_span_spin sees the full evidence set.
                                  "inlier_px": 30.0})
    flight_spans = [s for s in diag["spans"] if s["model"] == "flight"]
    assert flight_spans, "expected at least one flight span"
    assert any("omega_world" in s for s in flight_spans), (
        "expected the spin fit to be accepted on a strongly, cleanly spun "
        f"synthetic arc; spans={flight_spans}")
    spun = next(s for s in flight_spans if "omega_world" in s)
    assert spun["rad_s"] > 0.0
    # Knots must still be exact even with spin applied.
    assert np.allclose(frames[frame_a]["xyz"], p_a, atol=1e-6)
    assert np.allclose(frames[frame_b]["xyz"], p_b, atol=1e-6)
