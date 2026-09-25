"""Tests for src/utils/ball_hybrid_gating.py.

Reuses tests/test_ball_hybrid_trajectory.py's synthetic drag-kick
scenario fixtures (same shapes) to exercise the four independent gates:
kind whitelist, confidence floor (+ cue corroboration relaxing it),
evidence consistency, and residual improvement.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.utils.ball_eval import point_ray_distance
from src.utils.ball_hybrid_gating import DEFAULT_GATING_CFG, gate_auto_events
from src.utils.ball_hybrid_physics import BALL_RADIUS_M, simulate
from src.utils.ball_hybrid_trajectory import resolve_knots
from src.utils.ball_hybrid_types import CueEvidence

BALL_R = BALL_RADIUS_M


def _pinhole_K(fx=1000.0, fy=1000.0, cx=960.0, cy=540.0):
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])


def _make_test_ctx(n_frames=90, fps=30.0):
    from src.utils.ball_hybrid_types import HybridShotCtx
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


@dataclass
class _Anchor:
    frame: int
    image_xy: tuple
    state: str
    player_id: str | None = None
    bone: str | None = None
    goal_element: str | None = None


@dataclass
class _Obs:
    frame: int
    uv: tuple
    conf: float
    source: str = "detector"


def _find_landing_time(p0, v0, cd, t_max=5.0, n=4000):
    times = np.linspace(0.0, t_max, n)
    traj = simulate(p0, v0, times, cd=cd)
    z = traj[:, 2]
    below = np.where(z <= BALL_R)[0]
    below = below[below > 0]
    i = int(below[0])
    t0, t1 = times[i - 1], times[i]
    z0, z1 = z[i - 1], z[i]
    frac = (BALL_R - z0) / (z1 - z0)
    return float(t0 + frac * (t1 - t0))


def _scenario(fps=30.0, cd=0.30):
    ctx = _make_test_ctx(fps=fps)
    p_a = np.array([0.0, 0.0, BALL_R])
    v0_true = np.array([13.0, 0.0, 8.5])
    duration_s = _find_landing_time(p_a, v0_true, cd)
    frame_a = 10
    frame_b = frame_a + int(round(duration_s * fps))
    duration_s = (frame_b - frame_a) / fps
    p_b = simulate(p_a, v0_true, [duration_s], cd=cd)[0]
    p_b[2] = BALL_R

    anchors = [
        _Anchor(frame=frame_a, image_xy=tuple(float(x) for x in ctx.project(frame_a, p_a)),
                state="kick", player_id="P001", bone="right_foot"),
        _Anchor(frame=frame_b, image_xy=tuple(float(x) for x in ctx.project(frame_b, p_b)),
                state="bounce"),
    ]

    rng = np.random.default_rng(11)
    obs = []
    for frac in np.linspace(0.15, 0.85, 8):
        t_s = frac * duration_s
        frame = int(round(frame_a + t_s * fps))
        p_true = simulate(p_a, v0_true, [t_s], cd=cd)[0]
        uv = ctx.project(frame, p_true) + rng.normal(scale=1.0, size=2)
        obs.append(_Obs(frame=frame, uv=(float(uv[0]), float(uv[1])), conf=0.9))

    return ctx, anchors, obs, p_a, p_b, v0_true, cd, frame_a, frame_b, duration_s


def _mid_frame(frame_a, frame_b, gap=2):
    mid = (frame_a + frame_b) // 2
    if mid - frame_a <= gap or frame_b - mid <= gap:
        mid = frame_a + max(gap + 1, (frame_b - frame_a) // 2)
    return mid


# ---------------------------------------------------------------------------
# Kind whitelist
# ---------------------------------------------------------------------------

def test_non_event_state_never_becomes_a_candidate():
    ctx, anchors, obs, p_a, p_b, v0_true, cd, frame_a, frame_b, _T = _scenario()
    hard, ray = resolve_knots(ctx, anchors)
    mid_frame = _mid_frame(frame_a, frame_b)
    t_s = (mid_frame - frame_a) / ctx.fps
    p_mid = simulate(p_a, v0_true, [t_s], cd=cd)[0]
    uv_mid = tuple(float(x) for x in ctx.project(mid_frame, p_mid))
    candidate = _Anchor(frame=mid_frame, image_xy=uv_mid, state="airborne_mid")

    result = gate_auto_events(ctx, hard, ray, [candidate], obs,
                               trajectory_cfg={"cd": cd, "fit_cd": False})
    assert result.n_candidates == 1
    assert result.n_rejected_kind == 1
    assert result.accepted_hard == ()
    assert result.accepted_ray == ()


# ---------------------------------------------------------------------------
# Confidence floor (+ corroboration relaxing it)
# ---------------------------------------------------------------------------

def test_low_confidence_candidate_rejected_without_corroboration():
    ctx, anchors, obs, p_a, p_b, v0_true, cd, frame_a, frame_b, _T = _scenario()
    hard, ray = resolve_knots(ctx, anchors)
    mid_frame = _mid_frame(frame_a, frame_b)
    t_s = (mid_frame - frame_a) / ctx.fps
    p_mid = simulate(p_a, v0_true, [t_s], cd=cd)[0]
    uv_mid = tuple(float(x) for x in ctx.project(mid_frame, p_mid))
    candidate = _Anchor(frame=mid_frame, image_xy=uv_mid, state="bounce")
    # score defaults to 0.0 via _candidate_conf when the fake has no
    # score/conf attribute -- below the default floor of 0.5.
    result = gate_auto_events(ctx, hard, ray, [candidate], obs,
                               trajectory_cfg={"cd": cd, "fit_cd": False})
    assert result.n_rejected_confidence == 1
    assert result.accepted_hard == ()


def test_real_ball_anchor_confidence_field_is_read_correctly():
    """Regression test (found via a real origi01 fold0 bench run, 2026-09-25):
    a real ``src.schemas.ball_anchor.BallAnchor`` -- what
    ``generate_auto_anchors`` actually mints, and what ``ball.py``'s
    ``_run_hybrid_trajectory`` passes as ``auto_anchors`` -- carries its
    detector score as ``.confidence``, NOT ``.score``/``.conf``. An
    earlier version of ``_candidate_conf`` only checked ``.score``/
    ``.conf`` (matching this test file's own fixture convention, which
    happened to use ``.score``), silently reading 0.0 confidence for
    EVERY real auto-anchor candidate -- on origi01 fold0 this rejected
    all 54 auto-anchor candidates on confidence (0 accepted), starving
    the hybrid trajectory of every real touch/bounce knot the reference
    solver uses. A confidently-scored real BallAnchor must clear the
    default floor and be considered a genuine candidate."""
    from src.schemas.ball_anchor import BallAnchor

    ctx, anchors, obs, p_a, p_b, v0_true, cd, frame_a, frame_b, _T = _scenario()
    hard, ray = resolve_knots(ctx, anchors)
    mid_frame = _mid_frame(frame_a, frame_b)
    t_s = (mid_frame - frame_a) / ctx.fps
    p_mid = simulate(p_a, v0_true, [t_s], cd=cd)[0]
    uv_mid = tuple(float(x) for x in ctx.project(mid_frame, p_mid))
    # A real BallAnchor, confidently scored (0.85 clears the 0.5 default
    # floor by a wide margin) -- must NOT be rejected on confidence.
    candidate = BallAnchor(frame=mid_frame, image_xy=uv_mid, state="bounce",
                            confidence=0.85)
    result = gate_auto_events(ctx, hard, ray, [candidate], obs,
                               trajectory_cfg={"cd": cd, "fit_cd": False})
    assert result.n_rejected_confidence == 0


def test_corroborated_low_confidence_candidate_passes_the_relaxed_floor():
    ctx, anchors, obs, p_a, p_b, v0_true, cd, frame_a, frame_b, _T = _scenario()
    hard, ray = resolve_knots(ctx, anchors)
    mid_frame = _mid_frame(frame_a, frame_b)
    t_s = (mid_frame - frame_a) / ctx.fps
    p_mid = simulate(p_a, v0_true, [t_s], cd=cd)[0]
    uv_mid = tuple(float(x) for x in ctx.project(mid_frame, p_mid))
    # player_touch + a joint one radius further along the same sight-line
    # resolves EXACTLY to p_mid (same construction as the "consistent"
    # test below) so this test isolates the confidence-floor gate from
    # the consistency/residual gates.
    C, d_hat = ctx.ray(mid_frame, uv_mid)
    _, along = point_ray_distance(p_mid, C, d_hat)
    joint = C + (along + BALL_R) * d_hat

    class _PC:
        def joint_world(self, frame, pid, bone):
            return joint

    @dataclass
    class _ScoredAnchor(_Anchor):
        score: float = 0.35  # above corroborated_confidence_floor(0.3), below 0.5

    candidate = _ScoredAnchor(frame=mid_frame, image_xy=uv_mid, state="player_touch",
                               player_id="P099", bone="right_foot")
    cue = CueEvidence(frame=mid_frame, kind="player_touch", cue="audio_impact", conf=0.8)

    no_corrob = gate_auto_events(ctx, hard, ray, [candidate], obs,
                                  trajectory_cfg={"cd": cd, "fit_cd": False},
                                  player_context=_PC())
    with_corrob = gate_auto_events(ctx, hard, ray, [candidate], obs,
                                    trajectory_cfg={"cd": cd, "fit_cd": False},
                                    player_context=_PC(), corroboration=[cue])
    assert no_corrob.n_rejected_confidence == 1
    assert with_corrob.n_rejected_confidence == 0
    assert len(with_corrob.accepted_hard) == 1


# ---------------------------------------------------------------------------
# Near-manual drop (operator input always wins)
# ---------------------------------------------------------------------------

def test_candidate_near_manual_anchor_is_dropped():
    ctx, anchors, obs, p_a, p_b, v0_true, cd, frame_a, frame_b, _T = _scenario()
    hard, ray = resolve_knots(ctx, anchors)
    near_frame = frame_a + 1  # within default gap of 2
    p_near = simulate(p_a, v0_true, [1.0 / ctx.fps], cd=cd)[0]

    @dataclass
    class _ScoredAnchor(_Anchor):
        score: float = 0.9

    candidate = _ScoredAnchor(
        frame=near_frame,
        image_xy=tuple(float(x) for x in ctx.project(near_frame, p_near)),
        state="bounce")
    result = gate_auto_events(ctx, hard, ray, [candidate], obs,
                               trajectory_cfg={"cd": cd, "fit_cd": False})
    assert result.n_rejected_near_manual == 1
    assert result.accepted_hard == ()


# ---------------------------------------------------------------------------
# Evidence consistency + residual improvement
# ---------------------------------------------------------------------------

def test_consistent_candidate_that_improves_the_span_is_accepted():
    ctx, anchors, obs, p_a, p_b, v0_true, cd, frame_a, frame_b, _T = _scenario()
    hard, ray = resolve_knots(ctx, anchors)
    mid_frame = _mid_frame(frame_a, frame_b)
    t_s = (mid_frame - frame_a) / ctx.fps
    p_mid = simulate(p_a, v0_true, [t_s], cd=cd)[0]
    uv_mid = tuple(float(x) for x in ctx.project(mid_frame, p_mid))
    C, d_hat = ctx.ray(mid_frame, uv_mid)
    _, along = point_ray_distance(p_mid, C, d_hat)
    joint = C + (along + BALL_R) * d_hat

    class _PC:
        def joint_world(self, frame, pid, bone):
            return joint

    @dataclass
    class _ScoredAnchor(_Anchor):
        score: float = 0.9

    candidate = _ScoredAnchor(frame=mid_frame, image_xy=uv_mid, state="player_touch",
                               player_id="P099", bone="right_foot")
    result = gate_auto_events(ctx, hard, ray, [candidate], obs,
                               trajectory_cfg={"cd": cd, "fit_cd": False},
                               player_context=_PC())
    assert result.n_candidates == 1
    assert len(result.accepted_hard) == 1
    assert result.n_rejected_residual == 0
    assert np.allclose(result.accepted_hard[0].xyz, p_mid, atol=1e-3)


def test_inconsistent_candidate_rejected_by_consistency_or_residual_gate():
    ctx, anchors, obs, p_a, p_b, v0_true, cd, frame_a, frame_b, _T = _scenario()
    hard, ray = resolve_knots(ctx, anchors)
    mid_frame = _mid_frame(frame_a, frame_b)
    t_s = (mid_frame - frame_a) / ctx.fps
    p_mid = simulate(p_a, v0_true, [t_s], cd=cd)[0]
    assert p_mid[2] > 1.0, "test assumes the true arc is well off the ground here"
    uv_mid = tuple(float(x) for x in ctx.project(mid_frame, p_mid))

    @dataclass
    class _ScoredAnchor(_Anchor):
        score: float = 0.9

    # A "bounce" click resolves onto the GROUND plane, far from the true
    # (airborne) arc -- consistent with real detector observations near
    # it is impossible, so it should fail consistency, residual, or the
    # implausible-launch-speed gate (a "kick" -> ground "bounce" pairing
    # this close together implies a speed solve_span's own flight model
    # can't reach, so it falls back to roll -- which the gate treats as
    # disqualifying, not a free pass; see ball_hybrid_gating.py's module
    # docstring, gate 5).
    candidate = _ScoredAnchor(frame=mid_frame, image_xy=uv_mid, state="bounce")
    result = gate_auto_events(ctx, hard, ray, [candidate], obs,
                               trajectory_cfg={"cd": cd, "fit_cd": False})
    assert result.accepted_hard == ()
    assert (result.n_rejected_consistency + result.n_rejected_residual
            + result.n_rejected_implausible_velocity) == 1
