"""The hybrid ball extractor (Task B / SPIKE core method).

``run_hybrid(ctx, observations, anchors, fixes=(), auto_anchors=(), cfg=None) -> Track``

Pipeline (see CONTRACT.md and the task brief):

  a. Resolve hard 3-D knots (manual anchors via the same semantics as
     ``src.utils.ball_eval.anchor_gt_world`` — ground_exact/joint_depth —
     PLUS ``goal_impact`` resolved via goal-frame geometry
     (``src.utils.goal_geometry.resolve_goal_impact_world``, post/
     crossbar/net intersection) and ``catch`` resolved via the same
     "joint one ball-radius back along the sight-line" convention
     ``anchor_gt_world`` uses for ``player_touch`` (a keeper's hand
     joint), plus cross-replay fixes) and hard RAY constraints (states
     that still can't be resolved to a 3-D point — airborne_*, or a
     goal_impact/catch that missed geometry/had no joint — lateral-exact,
     depth free); sort by frame.
  a2. Auto-event knots: ``auto_anchors`` (the CURRENT ball stage's own
     auto-generated events — kinematic touches, velocity-break bounces,
     auto goal impacts) are resolved the same way and folded in as
     extra, SOFT-ish knots on top of the manual event layer — this PoC's
     hybrid is a new TRAJECTORY layer over the existing EVENT layer, not
     a replacement for it. Manual anchors always win: an auto knot/ray
     within ``auto_anchor_min_frame_gap`` frames of any manual knot/ray
     is dropped. A surviving auto knot is accepted only if inserting it
     doesn't make its bracketing span's own worst-evidence-residual gate
     any worse (``_integrate_auto_knots``); accepted/rejected counts are
     in ``diagnostics["auto_anchors"]``.
  b. Robust evidence: real detector observations are graded per-span
     against that span's own physics fit, ITERATED (fit -> gate outliers
     by reprojection residual -> refit) up to ``robust_gate_max_iters``
     times or until the inlier set stabilises; low-confidence/high-
     residual detections never move the knot-exact fit.
  c. Per span between consecutive hard knots: pick roll (both ends
     ground-level, no launch state) or flight (gravity + drag, optional
     Cd fit bounded to ``cd_bounds``, both endpoints always hit exactly
     via boundary-value shooting); split-and-retry (up to
     ``max_splits_per_span``) at the worst-residual evidence frame when
     a single arc can't explain the span (treated as an internal bounce
     knot, recursed).
  d. Physics track P_phys(frame) for every frame in [first knot/evidence,
     last knot/evidence]. The clip head/tail (before the first knot or
     after the last) gets a genuine free-end fit anchored at that one
     knot — a ground-roll (linear ray-fit velocity) or drag-flight
     (LM-fit v0 against reprojection) model — covering exactly out to
     where the evidence in that direction ends (``_fit_open_end``);
     falls back to holding the knot only when there's too little
     evidence to fit (``min_evid_for_open_end_fit``).
  e. Hybrid blend: delta = faithful_point - P_phys at confident inlier
     evidence (and exactly at hard ray anchors), smoothed by
     ``blend.blend_deltas`` and added back; final = P_phys + delta_s,
     with knot/ray-anchor frames re-snapped exactly afterwards.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.optimize import minimize_scalar

from src.utils.ball_anchor_heights import GROUND_LEVEL_STATES
from src.utils.ball_eval import anchor_gt_world, point_ray_distance, ray_plane_z
from src.utils.goal_geometry import GoalGeometry, resolve_goal_impact_world

from .blend import blend_deltas
from .hybrid_physics import (
    BALL_RADIUS_M,
    CD_BOUNDS,
    CD_DEFAULT,
    DEFAULT_MAGNUS_COEFF,
    fit_roll_segment,
    shoot_arc,
    simulate,
)
from .types import Observation, Track, TrackFrame

GROUND_EXACT_STATES = frozenset(GROUND_LEVEL_STATES) | {"bounce"}
_LAUNCH_STATES = frozenset({
    "kick", "header", "volley", "chest", "goal_impact", "player_touch",
})

# Plausibility envelope for the free-end (open-head/open-tail) fit: an
# unconstrained one-sided LM fit (one hard knot, no second endpoint) has a
# real monocular depth/speed ambiguity, especially over a longer span, and
# CAN converge to a technically-low-pixel-residual but physically absurd
# solution (observed on real evidence: a fitted v0 sending the ball to
# y=143m on a 68m-wide pitch). Interior spans don't need this — both ends
# are hard 3-D knots there, so shoot_arc is a boundary-value solve, not a
# free fit, and can't run away like this. Same order of magnitude as
# ball_piecewise_solver.SolverCfg's z_max_m/horizontal_speed_max_m_s.
_OPEN_END_PITCH_MARGIN_M = 15.0
_OPEN_END_PITCH_LENGTH_M = 105.0
_OPEN_END_PITCH_WIDTH_M = 68.0
_OPEN_END_MAX_HEIGHT_M = 50.0
_OPEN_END_MAX_LAUNCH_SPEED_M_S = 45.0

DEFAULT_CFG: dict[str, Any] = {
    "cd": CD_DEFAULT,               # 0.0 = gravity only (drag ablation)
    "fit_cd": True,
    "cd_bounds": CD_BOUNDS,
    "magnus": False,                # bounded Magnus refinement (best-effort)
    "magnus_coeff": DEFAULT_MAGNUS_COEFF,
    "blend_halflife_frames": 5.0,
    "faithful_conf_min": 0.5,
    "inlier_px": 15.0,
    "roll_mu_max": 0.9,
    "max_splits_per_span": 3,
    # Deliberately modest: high enough to saturate confidence exactly at
    # a ray anchor's own frame (which gets force-snapped to "anchor" mode
    # regardless) without inflating "faithful" confidence for several
    # frames around it purely from the anchor's pull rather than actual
    # detector evidence (a wide window there previously mislabelled
    # anchor-dominated frames as "faithful" and made the CLI's faithful-
    # frame-vs-observation reprojection stat misleading — see report).
    "ray_anchor_weight": 1.2,
    "min_obs_for_cd_fit": 5,
    "split_residual_factor": 2.5,
    # iteration 2 additions
    "robust_gate_max_iters": 3,
    "auto_anchor_min_frame_gap": 2,  # drop an auto anchor within this many
                                     # frames of any manual knot/ray anchor
    "min_evid_for_open_end_fit": 2,
}

# Default FIFA pitch dims (matches CLAUDE.md / run_all.py's "pitch" block):
# 105m x 68m, 2.44m crossbar, 7.32m goal width, 1.5m net depth.
_GOAL_GEOMETRY = GoalGeometry.from_pitch_config({})


# ---------------------------------------------------------------------------
# Knot resolution
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _Knot:
    frame: int
    xyz: np.ndarray
    state: str
    kind: str
    is_manual: bool = True


@dataclass(frozen=True)
class _RayAnchor:
    frame: int
    C: np.ndarray
    d_hat: np.ndarray
    uv: tuple[float, float]
    state: str


def _anchor_attrs(a: Any) -> tuple[int, tuple[float, float] | None, str,
                                    str | None, str | None, str | None]:
    if isinstance(a, Mapping):
        frame = int(a["frame"])
        raw_xy = a.get("image_xy")
        image_xy = (float(raw_xy[0]), float(raw_xy[1])) if raw_xy is not None else None
        state = str(a["state"])
        player_id = a.get("player_id")
        bone = a.get("bone")
        goal_element = a.get("goal_element")
    else:
        frame = int(a.frame)
        image_xy = a.image_xy
        state = a.state
        player_id = a.player_id
        bone = a.bone
        goal_element = getattr(a, "goal_element", None)
    return frame, image_xy, state, player_id, bone, goal_element


def _resolve_joint_depth(
    C: np.ndarray, d_hat: np.ndarray, joint_world: Any, ball_radius: float,
) -> tuple[np.ndarray | None, str]:
    """Same "ball sits one radius in front of the contacting limb along
    the sight-line" convention as ``ball_eval.anchor_gt_world``'s
    ``player_touch`` branch, factored out so ``catch`` (keeper hand joint
    on the click ray) can reuse it — ``anchor_gt_world`` itself only
    special-cases the literal string ``"player_touch"``."""
    _, along = point_ray_distance(np.asarray(joint_world, dtype=float), C, d_hat)
    if along > ball_radius:
        P = C + (along - ball_radius) * d_hat
        if P[2] >= ball_radius:
            return P, "joint_depth"
    X = ray_plane_z(C, d_hat, ball_radius)
    if X is not None:
        return X, "ground_exact"
    return None, "ray_only"


def resolve_knots(
    ctx: Any,
    anchors: Sequence[Any],
    fixes: Sequence[Any] = (),
) -> tuple[list[_Knot], list[_RayAnchor]]:
    """Split ``anchors`` (+ ``fixes``) into hard 3-D knots and hard ray
    constraints, exactly mirroring ``ball_eval.anchor_gt_world``'s own
    classification (``ground_exact``/``joint_depth`` -> hard knot,
    ``ray_only``/``none`` -> ray constraint or dropped)."""
    by_frame: dict[int, _Knot] = {}
    ray_anchors: list[_RayAnchor] = []

    for a in anchors:
        frame, image_xy, state, player_id, bone, goal_element = _anchor_attrs(a)
        if image_xy is None or frame not in ctx.per_frame_K:
            continue
        K = ctx.per_frame_K[frame]
        R = ctx.per_frame_R[frame]
        t = ctx.per_frame_t[frame]

        # goal_impact -> goal geometry (post/crossbar/net intersection),
        # a genuine hard 3-D knot. anchor_gt_world doesn't resolve this
        # (it only special-cases ground_exact/player_touch), so it's
        # handled here first; falls through to the generic path (-> ray
        # constraint) if there's no goal_element or the ray misses.
        if state == "goal_impact" and goal_element:
            try:
                xyz = resolve_goal_impact_world(
                    image_xy, goal_element, K=K, R=R, t=t,
                    distortion=ctx.distortion, geometry=_GOAL_GEOMETRY,
                )
                knot = _Knot(frame=frame, xyz=np.asarray(xyz, dtype=float),
                              state=state, kind="goal_geometry", is_manual=True)
                if frame not in by_frame:
                    by_frame[frame] = knot
                continue
            except ValueError:
                pass  # no geometric hit; fall through to ray-only below

        joint_world = None
        if state in ("player_touch", "catch") and player_id and bone:
            pc = ctx.player_context()
            joint_world = pc.joint_world(frame, player_id, bone)

        # catch -> keeper hand joint on the click ray, same "ball sits one
        # radius in front of the contact along the sight-line" convention
        # as player_touch. anchor_gt_world only special-cases the literal
        # string "player_touch", so this is resolved separately.
        if state == "catch" and joint_world is not None:
            C, d_hat = ctx.ray(frame, image_xy)
            xyz, kind = _resolve_joint_depth(C, d_hat, joint_world, BALL_RADIUS_M)
            if xyz is not None and kind in ("ground_exact", "joint_depth"):
                knot = _Knot(frame=frame, xyz=np.asarray(xyz, dtype=float),
                              state=state, kind=kind, is_manual=True)
                if frame not in by_frame:
                    by_frame[frame] = knot
                continue

        view = SimpleNamespace(image_xy=image_xy, state=state)
        xyz, kind = anchor_gt_world(
            view, K, R, t, ctx.distortion,
            ball_radius=BALL_RADIUS_M, joint_world=joint_world,
        )
        if kind in ("ground_exact", "joint_depth") and xyz is not None:
            knot = _Knot(frame=frame, xyz=np.asarray(xyz, dtype=float),
                          state=state, kind=kind, is_manual=True)
            if frame not in by_frame:
                by_frame[frame] = knot
        else:
            C, d_hat = ctx.ray(frame, image_xy)
            ray_anchors.append(_RayAnchor(frame=frame, C=C, d_hat=d_hat,
                                           uv=image_xy, state=state))

    for fx in fixes:
        frame = int(fx.frame)
        by_frame[frame] = _Knot(frame=frame, xyz=np.asarray(fx.xyz, dtype=float),
                                 state="fix", kind="fix", is_manual=True)

    hard_knots = sorted(by_frame.values(), key=lambda k: k.frame)
    ray_anchors.sort(key=lambda r: r.frame)
    return hard_knots, ray_anchors


# ---------------------------------------------------------------------------
# Per-span evidence + fitting helpers
# ---------------------------------------------------------------------------

def _reproj_px(ctx: Any, frame: int, xyz: np.ndarray, uv: tuple[float, float]) -> float:
    proj = ctx.project(frame, xyz)
    return float(np.hypot(float(proj[0]) - uv[0], float(proj[1]) - uv[1]))


def _span_evidence(
    observations: Sequence[Observation],
    ray_anchors: Sequence[_RayAnchor],
    a_frame: int,
    b_frame: int,
    ray_weight: float,
) -> list[tuple[int, tuple[float, float], float]]:
    evid = [(o.frame, o.uv, float(o.conf))
            for o in observations if a_frame < o.frame < b_frame]
    evid += [(r.frame, r.uv, ray_weight)
              for r in ray_anchors if a_frame < r.frame < b_frame]
    return evid


def _is_ground_knot(knot: "_Knot") -> bool:
    """True when ``knot`` sits at ground level AND isn't a launch event.

    A ``kick`` knot's own position is at the ball radius (ground level)
    but it launches the ball INTO the air, so the span/span-end that
    opens from it must be treated as flight, not roll — the plain height
    check alone would wrongly call it ground-level. Shared by
    ``_choose_model`` (interior spans) and ``_fit_open_end`` (clip
    head/tail free-end spans) so the two can't drift apart.
    """
    if knot.state in _LAUNCH_STATES:
        return False
    return knot.state in GROUND_EXACT_STATES or knot.xyz[2] <= BALL_RADIUS_M + 0.05


def _choose_model(a_knot: _Knot, b_knot: _Knot,
                   span_rays: Sequence[_RayAnchor]) -> str:
    if any(r.state.startswith("airborne") for r in span_rays):
        return "flight"
    if _is_ground_knot(a_knot) and _is_ground_knot(b_knot):
        return "roll"
    return "flight"


def _fit_roll_iterative(a_xy, b_xy, duration_s: float,
                         ground_obs: list[tuple[float, np.ndarray]],
                         cfg: Mapping[str, Any]):
    """Endpoint-exact roll fit with the same iterative fit->gate->refit
    robust-gating idea as the flight branch: drop ground observations
    whose residual from the current fit is a clear outlier (> 3x the
    fit's own median residual, floored at 0.5m so a tight, well-behaved
    fit isn't destabilised by refitting on near-nothing), refit, repeat
    up to ``robust_gate_max_iters`` times."""
    active = list(ground_obs)
    roll = fit_roll_segment(a_xy, b_xy, duration_s, active,
                             mu_max=cfg["roll_mu_max"])
    for _ in range(max(0, cfg["robust_gate_max_iters"] - 1)):
        if not active:
            break
        resid = [float(np.linalg.norm(roll.eval([t_s], z=0.0)[0][:2] - xy))
                  for t_s, xy in active]
        thresh = max(0.5, 3.0 * float(np.median(resid)))
        new_active = [ob for ob, r in zip(active, resid) if r <= thresh]
        if len(new_active) == len(active):
            break
        active = new_active
        roll = fit_roll_segment(a_xy, b_xy, duration_s, active,
                                 mu_max=cfg["roll_mu_max"])
    return roll


def _fit_open_end(
    ctx: Any,
    knot: _Knot,
    evidence: list[tuple[int, tuple[float, float], float]],
    direction: int,
    cfg: Mapping[str, Any],
) -> dict[int, np.ndarray]:
    """Free-end fit for a clip head/tail span: one hard knot, no second
    endpoint. ``direction`` is -1 for a head span (evidence frames <
    ``knot.frame``) or +1 for a tail span (frames > ``knot.frame``).
    Returns ``{frame: xyz}`` for every frame strictly between the knot
    and the farthest evidence frame in that direction, inclusive of that
    farthest evidence frame — i.e. the fitted span "ends where evidence
    ends" exactly (there is nothing further to extrapolate: this
    extractor's dense track is only ever built out to the last available
    knot/evidence frame in each direction, never past it). Returns ``{}``
    (caller falls back to holding the knot) when there isn't enough
    evidence to fit anything meaningfully."""
    if len(evidence) < cfg["min_evid_for_open_end_fit"]:
        return {}
    far_frame = (min(f for f, _, _ in evidence) if direction < 0
                 else max(f for f, _, _ in evidence))
    ground = _is_ground_knot(knot)
    frames = (list(range(far_frame, knot.frame)) if direction < 0
              else list(range(knot.frame + 1, far_frame + 1)))
    if not frames:
        return {}

    if ground:
        z_level = float(knot.xyz[2])
        num = np.zeros(2)
        den = 0.0
        for frame, uv, conf in evidence:
            C, d = ctx.ray(frame, uv)
            dz = float(d[2])
            if abs(dz) < 1e-9:
                continue
            s = (z_level - float(C[2])) / dz
            if s <= 0:
                continue
            P = C + s * d
            t_s = (frame - knot.frame) / ctx.fps
            if abs(t_s) < 1e-6:
                continue
            num += conf * t_s * (P[:2] - knot.xyz[:2])
            den += conf * t_s * t_s
        if den <= 1e-9:
            return {}
        v0_xy = num / den
        if float(np.linalg.norm(v0_xy)) > _OPEN_END_MAX_LAUNCH_SPEED_M_S:
            return {}
        out = {}
        for f in frames:
            t_s = (f - knot.frame) / ctx.fps
            xy = knot.xyz[:2] + v0_xy * t_s
            out[f] = np.array([xy[0], xy[1], z_level])
        return out if _open_end_plausible(out.values()) else {}

    cd = cfg["cd"]

    def _residual(v0):
        res = []
        for frame, uv, conf in evidence:
            t_s = (frame - knot.frame) / ctx.fps
            p = simulate(knot.xyz, v0, [t_s], cd=cd,
                         magnus_coeff=cfg["magnus_coeff"])[0]
            proj = ctx.project(frame, p)
            w = float(conf) ** 0.5
            res.append(w * (float(proj[0]) - uv[0]))
            res.append(w * (float(proj[1]) - uv[1]))
        return res

    try:
        from scipy.optimize import least_squares
        # Bounded (trf, not the unconstrained lm) on v0's components: an
        # unconstrained free-v0 fit has a real monocular depth/speed
        # ambiguity and can converge to a technically-low-residual but
        # physically absurd launch speed (see _OPEN_END_* constants).
        b = _OPEN_END_MAX_LAUNCH_SPEED_M_S
        sol = least_squares(_residual, np.zeros(3), method="trf",
                             bounds=([-b, -b, -b], [b, b, b]), max_nfev=200)
        v0 = sol.x
    except Exception:  # noqa: BLE001 — best-effort fit; caller falls back
        return {}
    times = [(f - knot.frame) / ctx.fps for f in frames]
    positions = simulate(knot.xyz, v0, times, cd=cd,
                          magnus_coeff=cfg["magnus_coeff"])
    if not _open_end_plausible(positions):
        return {}
    return {f: positions[i] for i, f in enumerate(frames)}


def _open_end_plausible(positions) -> bool:
    """Sanity bound on a free-end fit's resulting positions: even a
    speed-bounded fit can still curve somewhere absurd over a long span
    (drag/gravity is nonlinear), so this is a second, independent check
    directly on the output rather than just the launch velocity."""
    lo_x, hi_x = -_OPEN_END_PITCH_MARGIN_M, _OPEN_END_PITCH_LENGTH_M + _OPEN_END_PITCH_MARGIN_M
    lo_y, hi_y = -_OPEN_END_PITCH_MARGIN_M, _OPEN_END_PITCH_WIDTH_M + _OPEN_END_PITCH_MARGIN_M
    for p in positions:
        if not (lo_x <= p[0] <= hi_x and lo_y <= p[1] <= hi_y
                and BALL_RADIUS_M - 1e-6 <= p[2] <= _OPEN_END_MAX_HEIGHT_M):
            return False
    return True


def _solve_span(
    ctx: Any,
    a_knot: _Knot,
    b_knot: _Knot,
    observations: Sequence[Observation],
    ray_anchors: Sequence[_RayAnchor],
    cfg: Mapping[str, Any],
    splits_used: int,
    internal_knot_frames: list[int],
) -> tuple[dict[int, np.ndarray], list[dict]]:
    duration_frames = b_knot.frame - a_knot.frame
    if duration_frames <= 0:
        return {}, []
    duration_s = duration_frames / ctx.fps

    span_rays = [r for r in ray_anchors if a_knot.frame < r.frame < b_knot.frame]
    model = _choose_model(a_knot, b_knot, span_rays)

    if model == "roll":
        z_level = 0.5 * (float(a_knot.xyz[2]) + float(b_knot.xyz[2]))
        ground_obs: list[tuple[float, np.ndarray]] = []
        for o in observations:
            if not (a_knot.frame < o.frame < b_knot.frame):
                continue
            C, d = ctx.ray(o.frame, o.uv)
            dz = float(d[2])
            if abs(dz) < 1e-9:
                continue
            s = (z_level - float(C[2])) / dz
            if s <= 0:
                continue
            P = C + s * d
            ground_obs.append(((o.frame - a_knot.frame) / ctx.fps, P[:2]))
        roll = _fit_roll_iterative(a_knot.xyz[:2], b_knot.xyz[:2], duration_s,
                                    ground_obs, cfg)
        frames = list(range(a_knot.frame, b_knot.frame + 1))
        times = [(f - a_knot.frame) / ctx.fps for f in frames]
        positions = roll.eval(times, z_level)
        pts = {f: positions[i] for i, f in enumerate(frames)}

        # Reprojection residual against the SAME evidence definition the
        # flight branch uses (real observations + ray anchors, not just
        # the ground_obs used to fit the roll) -- without this, a roll
        # subspan reports no residual at all, which silently blinds the
        # auto-knot accept/reject gate (_integrate_auto_knots) and the
        # split-and-retry trigger to a badly-fitting roll.
        roll_evid = _span_evidence(observations, ray_anchors, a_knot.frame,
                                    b_knot.frame, cfg["ray_anchor_weight"])
        worst_roll: tuple[int, float] | None = None
        for frame, uv, _w in roll_evid:
            err = _reproj_px(ctx, frame, pts[frame], uv)
            if worst_roll is None or err > worst_roll[1]:
                worst_roll = (frame, err)

        info = {"span": (a_knot.frame, b_knot.frame), "model": "roll",
                "n_obs": len(ground_obs),
                "max_residual_px": worst_roll[1] if worst_roll else None}
        return pts, [info]

    # --- flight (iterative robust gating: fit -> gate outliers by
    # reprojection residual -> refit, up to robust_gate_max_iters times or
    # until the inlier set stabilises) --------------------------------
    evid_all = _span_evidence(observations, ray_anchors, a_knot.frame, b_knot.frame,
                               cfg["ray_anchor_weight"])
    cd = cfg["cd"]
    active_evid = list(evid_all)
    v0 = shoot_arc(a_knot.xyz, 0.0, b_knot.xyz, duration_s, cd=cd,
                    magnus_coeff=cfg["magnus_coeff"])
    pts: dict[int, np.ndarray] = {}

    for _iteration in range(max(1, cfg["robust_gate_max_iters"])):
        if cfg["fit_cd"] and len(active_evid) >= cfg["min_obs_for_cd_fit"]:
            def _cost(cd_val: float, _evid=active_evid, _a=a_knot.xyz, _b=b_knot.xyz,
                       _T=duration_s, _a_frame=a_knot.frame) -> float:
                v0_trial = shoot_arc(_a, 0.0, _b, _T, cd=cd_val)
                total = 0.0
                for frame, uv, w in _evid:
                    t_s = (frame - _a_frame) / ctx.fps
                    p = simulate(_a, v0_trial, [t_s], cd=cd_val)[0]
                    total += w * _reproj_px(ctx, frame, p, uv) ** 2
                return total

            lo, hi = cfg["cd_bounds"]
            if lo < hi:
                res = minimize_scalar(_cost, bounds=(lo, hi), method="bounded",
                                       options={"xatol": 1e-3, "maxiter": 15})
                cd = float(res.x)

        v0 = shoot_arc(a_knot.xyz, 0.0, b_knot.xyz, duration_s, cd=cd,
                        magnus_coeff=cfg["magnus_coeff"])
        frames = list(range(a_knot.frame, b_knot.frame + 1))
        times = [(f - a_knot.frame) / ctx.fps for f in frames]
        positions = simulate(a_knot.xyz, v0, times, cd=cd, magnus_coeff=cfg["magnus_coeff"])
        pts = {f: positions[i] for i, f in enumerate(frames)}

        new_active = [(frame, uv, w) for frame, uv, w in evid_all
                      if _reproj_px(ctx, frame, pts[frame], uv) <= cfg["inlier_px"]]
        if not new_active:
            break  # keep the last valid fit rather than refit on nothing
        if {f for f, _, _ in new_active} == {f for f, _, _ in active_evid}:
            break  # stable
        active_evid = new_active

    worst: tuple[int, tuple[float, float], float] | None = None
    for frame, uv, _w in evid_all:
        err = _reproj_px(ctx, frame, pts[frame], uv)
        if worst is None or err > worst[2]:
            worst = (frame, uv, err)

    info = {"span": (a_knot.frame, b_knot.frame), "model": "flight", "cd": cd,
            "n_obs": len(evid_all), "n_inliers": len(active_evid),
            "max_residual_px": worst[2] if worst else None}

    if (worst is not None
            and worst[2] > cfg["inlier_px"] * cfg["split_residual_factor"]
            and splits_used < cfg["max_splits_per_span"]):
        split_frame, split_uv, _err = worst
        C, d = ctx.ray(split_frame, split_uv)
        arc_pt = pts[split_frame]
        _, along = point_ray_distance(arc_pt, C, d)
        faithful = C + max(along, 0.0) * d
        z = max(float(faithful[2]), BALL_RADIUS_M)
        split_xyz = np.array([faithful[0], faithful[1], z])
        split_knot = _Knot(frame=split_frame, xyz=split_xyz, state="bounce",
                            kind="internal", is_manual=False)
        internal_knot_frames.append(split_frame)
        pts1, info1 = _solve_span(ctx, a_knot, split_knot, observations, ray_anchors,
                                   cfg, splits_used + 1, internal_knot_frames)
        pts2, info2 = _solve_span(ctx, split_knot, b_knot, observations, ray_anchors,
                                   cfg, splits_used + 1, internal_knot_frames)
        merged = {**pts1, **pts2}
        return merged, [*info1, *info2]

    return pts, [info]


def _build_physics_track(
    ctx: Any,
    hard_knots: Sequence[_Knot],
    ray_anchors: Sequence[_RayAnchor],
    observations: Sequence[Observation],
    cfg: Mapping[str, Any],
) -> tuple[dict[int, np.ndarray], list[dict], list[int]]:
    pts: dict[int, np.ndarray] = {}
    diagnostics: list[dict] = []
    internal_knot_frames: list[int] = []

    for a_knot, b_knot in zip(hard_knots, hard_knots[1:]):
        span_pts, span_info = _solve_span(ctx, a_knot, b_knot, observations,
                                           ray_anchors, cfg, 0, internal_knot_frames)
        pts.update(span_pts)
        diagnostics.extend(span_info)

    if len(hard_knots) == 1:
        pts[hard_knots[0].frame] = hard_knots[0].xyz

    if hard_knots:
        all_frames = sorted(set(list(pts)
                                 + [o.frame for o in observations]
                                 + [r.frame for r in ray_anchors]))
        first_frame = hard_knots[0].frame
        last_frame = hard_knots[-1].frame
        head_candidates = [f for f in all_frames if f < first_frame]
        tail_candidates = [f for f in all_frames if f > last_frame]
        if head_candidates:
            head_start = min(head_candidates)
            head_evid = _span_evidence(observations, ray_anchors,
                                        head_start - 1, first_frame,
                                        cfg["ray_anchor_weight"])
            fitted = _fit_open_end(ctx, hard_knots[0], head_evid, -1, cfg)
            for f in range(head_start, first_frame):
                pts[f] = fitted.get(f, hard_knots[0].xyz)
            diagnostics.append({
                "span": (head_start, first_frame),
                "model": "open_head" if fitted else "hold_head",
                "n_obs": len(head_evid),
            })
        if tail_candidates:
            tail_end = max(tail_candidates)
            tail_evid = _span_evidence(observations, ray_anchors,
                                        last_frame, tail_end + 1,
                                        cfg["ray_anchor_weight"])
            fitted = _fit_open_end(ctx, hard_knots[-1], tail_evid, 1, cfg)
            for f in range(last_frame + 1, tail_end + 1):
                pts[f] = fitted.get(f, hard_knots[-1].xyz)
            diagnostics.append({
                "span": (last_frame, tail_end),
                "model": "open_tail" if fitted else "hold_tail",
                "n_obs": len(tail_evid),
            })

    return pts, diagnostics, internal_knot_frames


def _delta_evidence(
    ctx: Any,
    pts: Mapping[int, np.ndarray],
    observations: Sequence[Observation],
    ray_anchors: Sequence[_RayAnchor],
    cfg: Mapping[str, Any],
) -> dict[int, tuple[tuple[float, float, float], float]]:
    evidence: dict[int, tuple[tuple[float, float, float], float]] = {}
    for o in observations:
        if o.conf < cfg["faithful_conf_min"]:
            continue
        p_phys = pts.get(o.frame)
        if p_phys is None:
            continue
        err_px = _reproj_px(ctx, o.frame, p_phys, o.uv)
        if err_px > cfg["inlier_px"]:
            continue
        C, d = ctx.ray(o.frame, o.uv)
        _, along = point_ray_distance(p_phys, C, d)
        faithful = C + max(along, 0.0) * d
        delta = faithful - p_phys
        evidence[o.frame] = (tuple(float(x) for x in delta), float(o.conf))
    for r in ray_anchors:
        p_phys = pts.get(r.frame)
        if p_phys is None:
            continue
        _, along = point_ray_distance(p_phys, r.C, r.d_hat)
        faithful = r.C + max(along, 0.0) * r.d_hat
        delta = faithful - p_phys
        evidence[r.frame] = (tuple(float(x) for x in delta), float(cfg["ray_anchor_weight"]))
    return evidence


# ---------------------------------------------------------------------------
# Auto-event (current-stage) knot integration
# ---------------------------------------------------------------------------

def _integrate_auto_knots(
    ctx: Any,
    hard_knots: list[_Knot],
    ray_anchors: list[_RayAnchor],
    auto_hard: list[_Knot],
    auto_rays: list[_RayAnchor],
    observations: Sequence[Observation],
    cfg: Mapping[str, Any],
) -> tuple[list[_Knot], list[_Knot], int, int, list[_RayAnchor]]:
    """Fold the current stage's auto-event anchors (kinematic touches,
    velocity-break bounces, auto goal impacts) in as extra, SOFT-ish
    knots — manual anchors always win, and an accepted auto knot must
    not make its bracketing span's own evidence fit worse.

    Returns ``(knots, accepted, n_rejected, n_dropped_near_manual,
    auto_rays_kept)``. ``knots`` is the full, sorted hard-knot list
    (manual + accepted auto) ready for ``_build_physics_track``.
    """
    gap = cfg["auto_anchor_min_frame_gap"]
    manual_frames = [k.frame for k in hard_knots] + [r.frame for r in ray_anchors]

    def _near_manual(f: int) -> bool:
        return any(abs(f - mf) <= gap for mf in manual_frames)

    candidates = sorted((k for k in auto_hard if not _near_manual(k.frame)),
                         key=lambda k: k.frame)
    n_dropped_near_manual = len(auto_hard) - len(candidates)

    cand_frames = {k.frame for k in candidates}
    auto_rays_kept = [r for r in auto_rays
                       if not _near_manual(r.frame)
                       and not any(abs(r.frame - cf) <= gap for cf in cand_frames)]

    # The ordinary inlier/outlier boundary, not the more lenient
    # split-retry threshold (inlier_px * split_residual_factor): that
    # looser bar answers "is this span worth attempting to explain with
    # an internal bounce", not "should I trust this auto knot" -- a
    # candidate that pushes evidence residual past the normal inlier
    # tolerance shouldn't be folded in just because a bounce-worthy span
    # would also have tolerated it.
    gate_px = cfg["inlier_px"]
    probe_cfg = dict(cfg)
    probe_cfg["max_splits_per_span"] = 0  # cheap probe, no recursive splitting
    combined_rays = list(ray_anchors) + auto_rays_kept

    knots = list(hard_knots)
    accepted: list[_Knot] = []
    rejected = 0

    for cand in candidates:
        knots.sort(key=lambda k: k.frame)
        idx = None
        for i in range(len(knots) - 1):
            if knots[i].frame < cand.frame < knots[i + 1].frame:
                idx = i
                break
        if idx is None:
            # Outside the current knot bracket (e.g. before the first or
            # after the last knot) -- nothing to gate against; accept it
            # rather than silently drop a legitimate edge event. Also
            # covers an (unlikely) exact-frame collision with an existing
            # knot, which is simply skipped.
            if cand.frame not in {k.frame for k in knots}:
                knots.append(cand)
                accepted.append(cand)
            continue

        a_knot, b_knot = knots[idx], knots[idx + 1]
        _, base_info = _solve_span(ctx, a_knot, b_knot, observations,
                                    combined_rays, probe_cfg, 0, [])
        base_worst = max((i.get("max_residual_px") or 0.0) for i in base_info)

        _, info1 = _solve_span(ctx, a_knot, cand, observations, combined_rays,
                                probe_cfg, 0, [])
        _, info2 = _solve_span(ctx, cand, b_knot, observations, combined_rays,
                                probe_cfg, 0, [])
        new_worst = max([(i.get("max_residual_px") or 0.0)
                          for i in (info1 + info2)], default=0.0)

        if new_worst <= max(base_worst, gate_px):
            knots.append(cand)
            accepted.append(cand)
        else:
            rejected += 1

    knots.sort(key=lambda k: k.frame)
    return knots, accepted, rejected, n_dropped_near_manual, auto_rays_kept


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _run_hybrid_full(
    ctx: Any,
    observations: Sequence[Observation],
    anchors: Sequence[Any],
    fixes: Sequence[Any] = (),
    auto_anchors: Sequence[Any] = (),
    *,
    cfg: Mapping[str, Any] | None = None,
) -> tuple[Track, dict]:
    full_cfg = dict(DEFAULT_CFG)
    if cfg:
        full_cfg.update(cfg)

    hard_knots, ray_anchors = resolve_knots(ctx, anchors, fixes)
    obs_sorted = sorted(observations, key=lambda o: o.frame)

    auto_diag = {"n_candidates": 0, "n_accepted": 0, "n_rejected": 0,
                 "n_dropped_near_manual": 0, "n_auto_rays_added": 0}
    if auto_anchors:
        auto_hard, auto_rays = resolve_knots(ctx, auto_anchors, fixes=())
        hard_knots, accepted, rejected, n_dropped, extra_rays = _integrate_auto_knots(
            ctx, hard_knots, ray_anchors, auto_hard, auto_rays, obs_sorted, full_cfg)
        ray_anchors = sorted(list(ray_anchors) + extra_rays, key=lambda r: r.frame)
        auto_diag = {
            "n_candidates": len(auto_hard),
            "n_accepted": len(accepted),
            "n_rejected": rejected,
            "n_dropped_near_manual": n_dropped,
            "n_auto_rays_added": len(extra_rays),
        }

    pts, span_diag, internal_knot_frames = _build_physics_track(
        ctx, hard_knots, ray_anchors, obs_sorted, full_cfg)

    if not pts:
        empty = Track(clip_id=ctx.clip_id, method="hybrid", frames=())
        return empty, {"spans": [], "n_knots": 0, "n_ray_anchors": 0,
                        "auto_anchors": auto_diag}

    evidence = _delta_evidence(ctx, pts, obs_sorted, ray_anchors, full_cfg)
    event_frames = ([k.frame for k in hard_knots] + internal_knot_frames)
    frames_sorted = sorted(pts)
    blended = blend_deltas(frames_sorted, evidence,
                            halflife_frames=full_cfg["blend_halflife_frames"],
                            event_frames=event_frames)

    knot_by_frame = {k.frame: k for k in hard_knots}
    ray_by_frame = {r.frame: r for r in ray_anchors}

    out_frames = []
    mode_counts = {"anchor": 0, "faithful": 0, "simulated": 0}
    for f in frames_sorted:
        base = np.asarray(pts[f], dtype=float)
        delta, conf = blended.get(f, ((0.0, 0.0, 0.0), 0.0))
        final = base + np.asarray(delta, dtype=float)
        mode = "faithful" if conf >= 0.5 else "simulated"
        out_conf: float | None = conf

        if f in knot_by_frame:
            final = np.array(knot_by_frame[f].xyz, dtype=float)
            mode, out_conf = "anchor", 1.0
        elif f in ray_by_frame:
            r = ray_by_frame[f]
            _, along = point_ray_distance(final, r.C, r.d_hat)
            final = r.C + max(along, 0.0) * r.d_hat
            mode, out_conf = "anchor", 1.0

        final[2] = max(float(final[2]), BALL_RADIUS_M)
        mode_counts[mode] = mode_counts.get(mode, 0) + 1
        out_frames.append(TrackFrame(
            frame=f, xyz=(float(final[0]), float(final[1]), float(final[2])),
            mode=mode, conf=out_conf))

    track = Track(clip_id=ctx.clip_id, method="hybrid", frames=tuple(out_frames))
    diagnostics = {
        "spans": span_diag,
        "n_knots": len(hard_knots),
        "n_ray_anchors": len(ray_anchors),
        "n_internal_bounces": len(internal_knot_frames),
        "mode_counts": mode_counts,
        "auto_anchors": auto_diag,
    }
    return track, diagnostics


def run_hybrid(
    ctx: Any,
    observations: Sequence[Observation],
    anchors: Sequence[Any],
    fixes: Sequence[Any] = (),
    auto_anchors: Sequence[Any] = (),
    *,
    cfg: Mapping[str, Any] | None = None,
) -> Track:
    """The hybrid ball extractor. See module docstring for the pipeline.

    ``auto_anchors``: the current stage's auto-event anchors (kinematic
    touches, velocity-break bounces, auto goal impacts — same
    ``BallAnchorSet`` schema as ``anchors``, method evidence not truth).
    Folded in as extra, softly-gated knots (see
    ``_integrate_auto_knots``); manual ``anchors`` always win.
    """
    track, _diag = _run_hybrid_full(ctx, observations, anchors, fixes,
                                     auto_anchors, cfg=cfg)
    return track


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_clip_report(clip_id: str, ctx: Any, track: Track, diagnostics: dict,
                        observations: Sequence[Observation], runtime_s: float) -> None:
    # Score "faithful" frames only against observations that were actually
    # eligible to become evidence (conf >= faithful_conf_min) — a frame
    # can be labelled "faithful" from a nearby confident detection while
    # its own frame's observation is a low-confidence outlier that was
    # correctly rejected as evidence; scoring against that rejected point
    # would conflate "matches trusted evidence" with "matches raw noise".
    conf_min = DEFAULT_CFG["faithful_conf_min"]
    by_frame_obs = {}
    for o in observations:
        if o.conf < conf_min:
            continue
        by_frame_obs.setdefault(o.frame, o)

    faithful_errs = []
    for tf in track.frames:
        if tf.mode != "faithful" or tf.xyz is None:
            continue
        o = by_frame_obs.get(tf.frame)
        if o is None:
            continue
        faithful_errs.append(_reproj_px(ctx, tf.frame, np.asarray(tf.xyz), o.uv))

    n = len(track.frames)
    mode_counts = diagnostics.get("mode_counts", {})
    model_counts: dict[str, int] = {}
    for s in diagnostics.get("spans", []):
        model_counts[s["model"]] = model_counts.get(s["model"], 0) + 1

    print(f"\n=== {clip_id} ===")
    print(f"frames: {n}  knots: {diagnostics['n_knots']}  "
          f"ray_anchors: {diagnostics['n_ray_anchors']}  "
          f"internal_bounces: {diagnostics['n_internal_bounces']}")
    if faithful_errs:
        arr = np.array(faithful_errs)
        print(f"faithful-frame reprojection px: "
              f"median={np.median(arr):.2f}  p95={np.percentile(arr, 95):.2f}  "
              f"(n={len(arr)})")
    else:
        print("faithful-frame reprojection px: n/a (no faithful frames with evidence)")
    print("mode split: " + ", ".join(
        f"{k}={100.0 * v / n:.1f}%" for k, v in sorted(mode_counts.items())))
    print("segment models: " + ", ".join(
        f"{k}={v}" for k, v in sorted(model_counts.items())))
    print(f"runtime: {runtime_s:.1f}s")


def main(argv: Sequence[str] | None = None) -> None:
    import argparse
    import time
    from pathlib import Path

    from .ctx import CLIPS, load_clip
    from .types import save_json

    parser = argparse.ArgumentParser(description="Run the hybrid ball extractor on real clips.")
    parser.add_argument("--clips", default=",".join(sorted(CLIPS)),
                         help="comma-separated clip ids")
    parser.add_argument("--no-drag", action="store_true",
                         help="ablation: cd=0, fit_cd=False (gravity-only)")
    parser.add_argument("--output-root", default=None,
                         help="override M/output-ball-poc (defaults to ctx.M)")
    args = parser.parse_args(argv)

    from . import ctx as ctx_mod
    output_root = Path(args.output_root) if args.output_root else Path(ctx_mod.M) / "output-ball-poc"

    cfg = {"cd": 0.0, "fit_cd": False} if args.no_drag else None

    for clip_id in args.clips.split(","):
        clip_id = clip_id.strip()
        if not clip_id:
            continue
        t0 = time.perf_counter()
        clip_ctx = load_clip(clip_id)
        anchors = list(clip_ctx.anchors.anchors)
        fixes = list(clip_ctx.fixes)
        observations = list(clip_ctx.observations)
        track, diagnostics = _run_hybrid_full(clip_ctx, observations, anchors, fixes, cfg=cfg)
        runtime_s = time.perf_counter() - t0

        out_path = output_root / clip_id / "track_hybrid_real_full.json"
        save_json(out_path, track)
        _print_clip_report(clip_id, clip_ctx, track, diagnostics, observations, runtime_s)
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
