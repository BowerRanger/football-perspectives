"""Auto-event acceptance gate for the ball-hybrid trajectory layer.

Decides which of the ball stage's OWN auto-generated events (kinematic
touches, velocity-break bounces, auto goal impacts — same
``BallAnchorSet`` schema as manual anchors, method evidence not truth)
get folded into the hybrid trajectory as extra knots. Manual anchors
always win (CLAUDE.md's "operator input always wins" invariant): an
auto candidate is never even considered within
``min_frame_gap_from_manual`` frames of a manual knot/ray, and this
module never mutates or overwrites the manual knot list — it only
proposes ADDITIONS to it.

Four independent gates, all must pass (except the residual-improvement
gate, which only applies to a candidate that resolves to a hard 3-D
knot — a ray-only candidate has no span-boundary role to test):

1. **Kind whitelist** — only a real EVENT state (``player_touch``,
   ``kick``, ``bounce``, ``goal_impact``, ``catch``, ``header``,
   ``volley``, ``chest``) is even a candidate; a synthetic
   ``grounded``/``airborne_*`` auto entry is a re-interpolation of
   evidence already in the fit and carries no new information (ported
   from the PoC's identical rule).
2. **Confidence floor** — the candidate's own event score must clear
   ``confidence_floor``, UNLESS at least one ``CueEvidence`` (IC-D's
   independent corroboration — audio/shadow/crowd/etc. cues, not yet
   wired as of this module's introduction, hence the empty default)
   agrees on the same frame/kind with its own confidence, in which case
   the lower ``corroborated_confidence_floor`` applies instead.
3. **Evidence consistency** — the candidate's proposed 3-D point must
   reproject close (median ``consistency_max_px``) to real, confident
   detector observations within ``±consistency_window_frames`` of it —
   a candidate whose position doesn't line up with what was actually
   seen nearby is more likely a misattributed/false event than a real
   one.
4. **Residual-improvement test** (hard candidates only) — inserting the
   candidate as an internal span-boundary knot must not make the
   bracketing span's own worst-evidence reprojection residual
   meaningfully worse. ``residual_improve_frac`` (default ``0.0``,
   "must not make the span any worse") reproduces the PoC's validated
   gate exactly: the allowed ceiling is
   ``max(base_worst, trajectory_cfg["inlier_px"])`` — the ordinary
   inlier tolerance is always an allowed cushion, since re-fitting Cd on
   two shorter half-spans (each with fewer points) can nudge the
   worst-point residual up or down by a pixel or two purely from noise,
   even when the candidate lands exactly on an already-good fit (the
   common case for a real, well-detected touch). Set a positive
   ``residual_improve_frac`` for a stricter, opt-in policy: the gate
   becomes ``new_worst <= base_worst * (1 - residual_improve_frac)``,
   with no inlier-tolerance cushion — the candidate must demonstrably
   help, not just avoid hurting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from src.utils.ball_anchor_heights import EVENT_STATES
from src.utils.ball_hybrid_trajectory import (
    anchor_kind,
    resolve_knots,
    solve_span,
)
from src.utils.ball_hybrid_types import CueEvidence, HybridShotCtx, Knot

DEFAULT_GATING_CFG: dict[str, Any] = {
    "kind_whitelist": tuple(sorted(EVENT_STATES)),
    "confidence_floor": 0.5,
    "corroborated_confidence_floor": 0.3,
    "corroboration_window_frames": 3,
    "consistency_window_frames": 5,
    "consistency_max_px": 20.0,
    "residual_improve_frac": 0.0,
    "min_frame_gap_from_manual": 2,
}


def gating_cfg(cfg: Mapping[str, Any] | None = None) -> dict[str, Any]:
    out = dict(DEFAULT_GATING_CFG)
    if cfg:
        out.update(cfg)
    return out


@dataclass(frozen=True)
class GateResult:
    accepted_hard: tuple[Knot, ...]
    accepted_ray: tuple[Knot, ...]
    n_candidates: int
    n_rejected_kind: int
    n_rejected_confidence: int
    n_rejected_near_manual: int
    n_rejected_consistency: int
    n_rejected_residual: int
    diagnostics: dict = field(default_factory=dict)


def _candidate_conf(a: Any) -> float:
    if isinstance(a, Mapping):
        return float(a.get("score", a.get("conf", 0.0)) or 0.0)
    return float(getattr(a, "score", getattr(a, "conf", 0.0)) or 0.0)


def _candidate_frame(a: Any) -> int:
    return int(a["frame"]) if isinstance(a, Mapping) else int(a.frame)


def _best_corroboration(
    frame: int, kind: str, corroboration: Sequence[CueEvidence], window: int,
) -> float:
    """Highest-confidence cue that agrees with ``kind`` within
    ``window`` frames of ``frame``; 0.0 when none corroborates."""
    best = 0.0
    for c in corroboration:
        if c.kind != kind or abs(int(c.frame) - frame) > window:
            continue
        best = max(best, float(c.conf))
    return best


def _reproj_px(ctx: HybridShotCtx, frame: int, xyz, uv: tuple[float, float]) -> float:
    proj = ctx.project(frame, xyz)
    return float(np.hypot(float(proj[0]) - uv[0], float(proj[1]) - uv[1]))


def _consistency_px(
    ctx: HybridShotCtx, knot: Knot, observations: Sequence[Any],
    window: int, conf_min: float,
) -> float | None:
    """Median reprojection (px) of ``knot.xyz`` into each nearby
    confident observation's OWN frame camera, vs. that observation's
    detected pixel. ``None`` when no confident observation falls in the
    window (consistency simply can't be checked — treated as passing;
    it's the CONFIDENCE and RESIDUAL gates that carry the weight for a
    candidate with no nearby real detections)."""
    errs = []
    for o in observations:
        if o.conf < conf_min or abs(o.frame - knot.frame) > window:
            continue
        if not ctx.has_frame(o.frame):
            continue
        errs.append(_reproj_px(ctx, o.frame, np.asarray(knot.xyz), o.uv))
    if not errs:
        return None
    return float(np.median(errs))


def gate_auto_events(
    ctx: HybridShotCtx,
    hard_knots: Sequence[Knot],
    ray_knots: Sequence[Knot],
    auto_anchors: Sequence[Any],
    observations: Sequence[Any],
    cfg: Mapping[str, Any] | None = None,
    trajectory_cfg: Mapping[str, Any] | None = None,
    *,
    player_context: Any = None,
    corroboration: Sequence[CueEvidence] = (),
) -> GateResult:
    """Decide which of ``auto_anchors`` to fold into the trajectory as
    extra knots on top of ``hard_knots``/``ray_knots`` (manual + fixes).
    Never mutates its inputs — returns the ADDITIONS only; the caller
    merges them (``hard_knots + result.accepted_hard`` etc.) before
    calling ``ball_hybrid_trajectory.build_trajectory``.

    ``trajectory_cfg`` is ``ball_hybrid_trajectory``'s own cfg (needed to
    run the residual-improvement probe with the same span-fit settings
    the real build will use); defaults to
    ``ball_hybrid_trajectory.DEFAULT_CFG`` when omitted.
    """
    from src.utils.ball_hybrid_trajectory import full_cfg as _traj_full_cfg

    g = gating_cfg(cfg)
    tcfg = _traj_full_cfg(trajectory_cfg)
    whitelist = frozenset(g["kind_whitelist"])
    gap = g["min_frame_gap_from_manual"]

    manual_frames = [k.frame for k in hard_knots] + [r.frame for r in ray_knots]

    def _near_manual(f: int) -> bool:
        return any(abs(f - mf) <= gap for mf in manual_frames)

    n_candidates = len(auto_anchors)
    n_rej_kind = n_rej_conf = n_rej_near = 0
    surviving: list[Any] = []
    for a in auto_anchors:
        kind = anchor_kind(a)
        if kind not in whitelist:
            n_rej_kind += 1
            continue
        frame = _candidate_frame(a)
        if _near_manual(frame):
            n_rej_near += 1
            continue
        conf = _candidate_conf(a)
        corrob = _best_corroboration(frame, kind, corroboration,
                                      g["corroboration_window_frames"])
        floor = (g["corroborated_confidence_floor"] if corrob > 0.0
                 else g["confidence_floor"])
        if conf < floor:
            n_rej_conf += 1
            continue
        surviving.append(a)

    auto_hard, auto_ray = resolve_knots(
        ctx, surviving, fixes=(), source="auto", player_context=player_context)

    # Ray-only candidates: no span-boundary role to test, so they only
    # need the gates already applied above (kind/near-manual/confidence).
    # Consistency is still worth checking (cheap, and catches a wildly
    # mislabelled click).
    n_rej_consistency = 0
    accepted_ray: list[Knot] = []
    for r in auto_ray:
        c = _consistency_px(ctx, r, observations, g["consistency_window_frames"],
                             tcfg["faithful_conf_min"])
        if c is not None and c > g["consistency_max_px"]:
            n_rej_consistency += 1
            continue
        accepted_ray.append(r)

    # Hard candidates: consistency + residual-improvement, evaluated
    # against the bracketing span in the KNOT LIST AS IT GROWS (so two
    # candidates in the same span are each judged against the current
    # state, same order-dependent approach as the PoC).
    knots = list(hard_knots)
    combined_rays = list(ray_knots) + accepted_ray
    accepted_hard: list[Knot] = []
    n_rej_residual = 0
    probe_cfg = dict(tcfg)
    probe_cfg["max_splits_per_span"] = 0  # cheap probe, no recursive splitting

    for cand in sorted(auto_hard, key=lambda k: k.frame):
        c = _consistency_px(ctx, cand, observations, g["consistency_window_frames"],
                             tcfg["faithful_conf_min"])
        if c is not None and c > g["consistency_max_px"]:
            n_rej_consistency += 1
            continue

        knots.sort(key=lambda k: k.frame)
        idx = None
        for i in range(len(knots) - 1):
            if knots[i].frame < cand.frame < knots[i + 1].frame:
                idx = i
                break
        if idx is None:
            if cand.frame not in {k.frame for k in knots}:
                knots.append(cand)
                accepted_hard.append(cand)
            continue

        a_knot, b_knot = knots[idx], knots[idx + 1]
        _, base_info = solve_span(ctx, a_knot, b_knot, observations,
                                   combined_rays, probe_cfg, 0, [])
        base_worst = max((i.get("max_residual_px") or 0.0) for i in base_info)

        _, info1 = solve_span(ctx, a_knot, cand, observations, combined_rays,
                               probe_cfg, 0, [])
        _, info2 = solve_span(ctx, cand, b_knot, observations, combined_rays,
                               probe_cfg, 0, [])
        new_worst = max([(i.get("max_residual_px") or 0.0)
                          for i in (info1 + info2)], default=0.0)

        frac = g["residual_improve_frac"]
        if frac <= 0.0:
            # Default ("must not make the span any worse"): same slack
            # the PoC's validated gate used — a candidate that lands
            # exactly on an already-good fit can still nudge the worst-
            # point residual up or down by a pixel or two purely from
            # refitting Cd on fewer points per half-span, so the ordinary
            # inlier tolerance (tcfg["inlier_px"]) is always an allowed
            # ceiling, not just base_worst itself.
            allowed = max(base_worst, tcfg["inlier_px"])
        else:
            # Stricter, opt-in mode: the candidate must demonstrably
            # IMPROVE the span's fit by at least this fraction — no
            # inlier-tolerance cushion.
            allowed = base_worst * (1.0 - frac)
        if new_worst <= allowed:
            knots.append(cand)
            accepted_hard.append(cand)
        else:
            n_rej_residual += 1

    return GateResult(
        accepted_hard=tuple(sorted(accepted_hard, key=lambda k: k.frame)),
        accepted_ray=tuple(sorted(accepted_ray, key=lambda k: k.frame)),
        n_candidates=n_candidates,
        n_rejected_kind=n_rej_kind,
        n_rejected_confidence=n_rej_conf,
        n_rejected_near_manual=n_rej_near,
        n_rejected_consistency=n_rej_consistency,
        n_rejected_residual=n_rej_residual,
        diagnostics={
            "n_accepted_hard": len(accepted_hard),
            "n_accepted_ray": len(accepted_ray),
        },
    )
