"""The ball-hybrid trajectory layer: knot resolution + per-span physics fit
+ local smoothing + broadcast/physics delta blend.

Ported and restructured from ``prototypes/ball_hybrid_poc/hybrid.py`` (see
that module's docstring for the original design history and
``prototypes/ball_hybrid_poc/CONTRACT.md`` for the PoC's measurement
contract). This module owns the trajectory MECHANICS; the auto-event
acceptance policy (kind whitelist, confidence floor, evidence-consistency
gate, residual-improvement gate, cue corroboration) lives in
``ball_hybrid_gating.py``, which calls back into this module's
``solve_span``/``resolve_knots`` to probe candidate knots.

## Knot taxonomy (unchanged from the PoC)

Every anchor/fix that resolves to a genuine 3-D point (ground plane,
player/keeper joint ray-intersection, or goal geometry) is a hard
SPAN-BOUNDARY knot (``Knot.depth_hard=True``). A state that can't be
pinned to 3-D without extra context (``airborne_low/mid/high``,
``off_screen_flight``, or an EVENT anchor that failed 3-D resolution)
becomes a ``Knot.depth_hard=False`` "ray knot": lateral-exact (its pixel
ray is ground truth), depth free.

Hard knots split further into SHARP (``is_sharp_knot``: a real EVENT
state, a cross-replay fix, or a data-discovered internal split — a
genuine velocity break belongs here, exact re-snap) and non-sharp
(everything else, almost always a "grounded" click — exact position for
FITTING, but its velocity kink is smoothed locally over a small window
by ``smooth_non_event_knot_windows`` rather than left as a hard break).

## The origi01 held-out fix (2026-09, T1c)

The PoC's ``hybrid_events`` real-footage held-out eval regressed sharply
on origi01 (airborne_mid/high held-out frames 218/224/234/287/297: 3-D
error spiking to ~1.70 m vs. the reference stage's 0.36 m). Root cause:
inside a FLIGHT span, ray-only (``depth_hard=False``) evidence — by
construction always depth-ambiguous, that's *why* it wasn't resolved to
a hard knot — was being weighted as heavily as a manual click
(``anchor_fit_weight=20``) in the span's ``shoot_arc`` drag-coefficient
(Cd) fit and its robust-gating residual. Cd sets the arc's curvature
along its ENTIRE length, not just at the ray's own frame, so an
inconsistent or merely noisy ray anchor could warp the arc's mid-span
DEPTH substantially even though the fit's 2-D reprojection (the only
thing being minimised) stayed near-perfect — reprojection error cannot
see a depth-only error, because by definition a ray anchor's "faithful"
point sits ON its ray at any depth.

Fix: this module now weights ray-only evidence differently depending on
whether its depth is externally pinned by the fit itself.
- In a ROLL span, ray evidence is first projected onto the span's own
  KNOWN ground level (``z_level``, interpolated from the two hard
  endpoints) — that assumption externally supplies the depth, so the
  projected point is genuinely trustworthy; it keeps the historical
  weight (``grounded_ray_weight``, default 20.0, matching the PoC's
  ``anchor_fit_weight``).
- In a FLIGHT span (Cd-fit, robust gating, and the open-end free flight
  fit), a ray knot's depth is NOT pinned by anything — it is exactly the
  depth-ambiguous case the module docstring above describes — so it now
  gets a much smaller ``airborne_ray_weight`` (default 2.0: comparable
  to one solid detector observation, not a click-grade hard constraint).
  It can still nudge the fit's LATERAL shape and participate in outlier
  gating, but can no longer single-handedly dictate the arc's depth.

This is the production analogue of ``ball_hybrid_types.Knot.depth_hard``:
a ray knot's ``xyz`` field always carries a best-effort placeholder
position (for diagnostics/display), but nothing in this module ever
treats that placeholder depth as authoritative — depth for a ray knot
is always either (a) externally supplied by a ground-plane assumption
(roll spans) or (b) left to the physics fit with only weak lateral
pull (flight spans), never re-snapped from the anchor itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import minimize_scalar

from src.utils.ball_anchor_heights import EVENT_STATES, GROUND_LEVEL_STATES
from src.utils.ball_eval import anchor_gt_world, point_ray_distance, ray_plane_z
from src.utils.ball_hybrid_physics import (
    BALL_RADIUS_M,
    CD_BOUNDS,
    CD_DEFAULT,
    DEFAULT_MAGNUS_COEFF,
    G,
    fit_roll_segment,
    hermite_blend,
    shoot_arc,
    simulate,
)
from src.utils.ball_hybrid_blend import blend_deltas, clamp_delta_rate
from src.utils.ball_hybrid_spin import DEFAULT_BOUNDS as DEFAULT_SPIN_BOUNDS
from src.utils.ball_hybrid_spin import fit_span_spin
from src.utils.ball_hybrid_types import HybridShotCtx, Knot
from src.utils.goal_geometry import GoalGeometry, resolve_goal_impact_world

GROUND_EXACT_STATES = frozenset(GROUND_LEVEL_STATES) | {"bounce"}
_LAUNCH_STATES = frozenset({"kick", "header", "volley", "chest", "goal_impact"})
_AMBIGUOUS_GROUND_STATES = frozenset({"player_touch", "catch"})

# Plausibility envelope for the free-end (open-head/open-tail) fit — see
# ball_hybrid.hybrid.py's identical constants for the rationale (an
# unconstrained one-sided LM fit has a real monocular depth/speed
# ambiguity and can otherwise converge to a technically-low-residual but
# physically absurd solution).
_OPEN_END_PITCH_MARGIN_M = 15.0
_OPEN_END_PITCH_LENGTH_M = 105.0
_OPEN_END_PITCH_WIDTH_M = 68.0
_OPEN_END_MAX_HEIGHT_M = 50.0
_OPEN_END_MAX_LAUNCH_SPEED_M_S = 45.0
# Public alias: ball_hybrid_gating.py's auto-knot acceptance gate reuses
# this SAME bound (not a new arbitrary number) to reject an interior
# flight span whose implied two-knot launch speed is physically absurd
# -- see that module's plausibility check for why "both ends are hard
# knots, so shoot_arc can't run away" (this constant's original
# rationale, true for a genuinely-correct pair of knots) breaks down
# when one of the knots itself is a bad auto-generated candidate.
MAX_LAUNCH_SPEED_M_S = _OPEN_END_MAX_LAUNCH_SPEED_M_S

DEFAULT_CFG: dict[str, Any] = {
    "cd": CD_DEFAULT,
    "fit_cd": True,
    "cd_bounds": CD_BOUNDS,
    "magnus": False,
    "magnus_coeff": DEFAULT_MAGNUS_COEFF,
    "blend_halflife_frames": 5.0,
    "faithful_conf_min": 0.5,
    "inlier_px": 15.0,
    "roll_mu_max": 0.9,
    "max_splits_per_span": 3,
    "ray_anchor_weight": 1.2,          # delta-blend pull weight for a ray knot
    "min_obs_for_cd_fit": 5,
    "split_residual_factor": 2.5,
    "robust_gate_max_iters": 3,
    "min_evid_for_open_end_fit": 2,
    # Ray-evidence SPAN-FIT weight, split by whether the span externally
    # pins depth (roll: ground plane) or not (flight: depth-ambiguous —
    # see module docstring "the origi01 held-out fix").
    "grounded_ray_weight": 20.0,
    "airborne_ray_weight": 2.0,
    "anchor_not_honoured_px": 4.0,
    "anchor_delta_max_m": 1.5,
    "hermite_k_min": 3,
    "hermite_k_max": 5,
    "hermite_k_ref_frames": 4,
    "hermite_k_ref_speed_m_s": 2.0,
    "blend_max_delta_step_frac": 0.5,
    "blend_max_delta_step_floor_m_s": 1.0,
    # An interior flight span whose fitted launch speed exceeds this
    # falls back to the roll model instead (see solve_span's flight
    # branch docstring comment) — reuses the SAME bound the free-end fit
    # already enforces (_open_end_plausible), not a new arbitrary number.
    "max_launch_speed_m_s": MAX_LAUNCH_SPEED_M_S,
    # Bounded 2-dof Magnus (spin) refinement per flight span (IC-E,
    # ball_hybrid_spin.fit_span_spin) — see solve_span's flight branch.
    # Default OFF until the benchmark decision (~0.3-1.5s/span cost).
    "spin": {
        "enabled": False,
        "bounds": DEFAULT_SPIN_BOUNDS,
        "min_obs": 8,
        "min_delta_bic": 6.0,
        "min_resid_gain": 0.10,
        # Bounded Magnus on SHOT spans only (design D6.3): a flight span
        # whose start knot is a shot/volley touch (touch_type shot|volley
        # or a spin preset on the anchor) gets the spin refinement even
        # though global ``enabled`` stays off. The curl is capped so the
        # peak Magnus acceleration stays <= shot_max_accel_m_s2 (~10 m/s^2,
        # i.e. what the gberch instep curl measured at 8). Code default off
        # (bare configs / bench unchanged); config/default.yaml turns it on.
        "shot_spans": False,
        "shot_max_accel_m_s2": 10.0,
        "shot_start_frames": (),
    },
    # D6 goal-mouth constraint + detector direction gate (run_trajectory).
    "goal": {"enabled": True},
    "direction_gate": {"enabled": True},
}

_GOAL_GEOMETRY = GoalGeometry.from_pitch_config({})


def full_cfg(cfg: Mapping[str, Any] | None = None) -> dict[str, Any]:
    out = dict(DEFAULT_CFG)
    if cfg:
        out.update(cfg)
    return out


# ---------------------------------------------------------------------------
# Knot resolution
# ---------------------------------------------------------------------------

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


def anchor_kind(a: Any) -> str:
    """The ``state``/kind string for any anchor-like object (mapping or
    dataclass) — used by the gating module's kind whitelist."""
    return _anchor_attrs(a)[2]


def _resolve_joint_depth(
    C: np.ndarray, d_hat: np.ndarray, joint_world: Any, ball_radius: float,
) -> tuple[np.ndarray | None, str]:
    _, along = point_ray_distance(np.asarray(joint_world, dtype=float), C, d_hat)
    if along > ball_radius:
        P = C + (along - ball_radius) * d_hat
        if P[2] >= ball_radius:
            return P, "joint_depth"
    X = ray_plane_z(C, d_hat, ball_radius)
    if X is not None:
        return X, "ground_exact"
    return None, "ray_only"


def _ray_placeholder_xyz(C: np.ndarray, d_hat: np.ndarray) -> tuple[float, float, float]:
    """Best-effort DISPLAY-ONLY position for a depth-free ray knot: the
    ground-plane intersection when the ray actually reaches z=BALL_RADIUS_M
    going forward, else a point a nominal 20 m along the ray. Never used
    by the fit for depth (see module docstring) — only for diagnostics
    and as the Knot dataclass's required ``xyz`` field."""
    X = ray_plane_z(C, d_hat, BALL_RADIUS_M)
    if X is not None:
        return (float(X[0]), float(X[1]), float(X[2]))
    P = C + 20.0 * d_hat
    return (float(P[0]), float(P[1]), float(max(P[2], BALL_RADIUS_M)))


def resolve_knots(
    ctx: HybridShotCtx,
    anchors: Sequence[Any],
    fixes: Sequence[Any] = (),
    *,
    source: str = "manual",
    player_context: Any = None,
) -> tuple[list[Knot], list[Knot]]:
    """Split ``anchors`` (+ ``fixes``) into hard 3-D span-boundary knots
    and ray-only (depth-free) knots. ``player_context`` is a
    ``PlayerContext``-like object (``.joint_world(frame, player_id,
    bone)``) needed to resolve ``player_touch``/``catch`` depth; when
    ``None``, those states fall back to the ray-only branch. ``source``
    is stamped on every returned ``Knot`` (``"manual"``, ``"auto"``, or
    ``"fix"`` — fixes are always stamped ``"fix"`` regardless of this
    argument)."""
    by_frame: dict[int, Knot] = {}
    ray_by_frame: dict[int, Knot] = {}

    for a in anchors:
        frame, image_xy, state, player_id, bone, goal_element = _anchor_attrs(a)
        if image_xy is None or not ctx.has_frame(frame):
            continue

        if state == "goal_impact" and goal_element:
            try:
                xyz = resolve_goal_impact_world(
                    image_xy, goal_element,
                    K=ctx.per_frame_K[frame], R=ctx.per_frame_R[frame],
                    t=ctx.per_frame_t[frame], distortion=ctx.distortion,
                    geometry=_GOAL_GEOMETRY,
                )
                # ``mouth`` marks the ball crossing the line, not a contact:
                # a smooth hard knot ("line_cross"), not a velocity break.
                by_frame.setdefault(frame, Knot(
                    frame=frame, xyz=tuple(float(x) for x in xyz),
                    kind="line_cross" if goal_element == "mouth" else state,
                    depth_hard=True, source=source, uv=image_xy))
                continue
            except ValueError:
                pass

        joint_world = None
        if state in ("player_touch", "catch") and player_id and bone and player_context is not None:
            joint_world = player_context.joint_world(frame, player_id, bone)

        if state == "catch" and joint_world is not None:
            C, d_hat = ctx.ray(frame, image_xy)
            xyz, kind = _resolve_joint_depth(C, d_hat, joint_world, BALL_RADIUS_M)
            if xyz is not None and kind in ("ground_exact", "joint_depth"):
                by_frame.setdefault(frame, Knot(
                    frame=frame, xyz=tuple(float(x) for x in xyz), kind=state,
                    depth_hard=True, source=source, uv=image_xy))
                continue

        view = SimpleNamespace(image_xy=image_xy, state=state)
        xyz, kind = anchor_gt_world(
            view, ctx.per_frame_K[frame], ctx.per_frame_R[frame],
            ctx.per_frame_t[frame], ctx.distortion,
            ball_radius=BALL_RADIUS_M, joint_world=joint_world,
        )
        if kind in ("ground_exact", "joint_depth") and xyz is not None:
            by_frame.setdefault(frame, Knot(
                frame=frame, xyz=tuple(float(x) for x in xyz), kind=state,
                depth_hard=True, source=source, uv=image_xy))
        else:
            C, d_hat = ctx.ray(frame, image_xy)
            ray_by_frame.setdefault(frame, Knot(
                frame=frame, xyz=_ray_placeholder_xyz(C, d_hat), kind=state,
                depth_hard=False, source=source, uv=image_xy))

    for fx in fixes:
        frame = int(fx.frame)
        by_frame[frame] = Knot(
            frame=frame, xyz=tuple(float(x) for x in fx.xyz), kind="fix",
            depth_hard=True, source="fix")

    hard_knots = sorted(by_frame.values(), key=lambda k: k.frame)
    ray_knots = sorted(ray_by_frame.values(), key=lambda k: k.frame)
    return hard_knots, ray_knots


def is_sharp_knot(knot: Knot) -> bool:
    """True when a real velocity break belongs at this knot: an EVENT
    state, a cross-replay fix, or a data-discovered internal split."""
    return knot.kind in EVENT_STATES or knot.kind == "fix" or knot.kind == "internal_split"


def _is_ground_knot(knot: Knot) -> bool:
    if knot.kind in _LAUNCH_STATES:
        return False
    return knot.kind in GROUND_EXACT_STATES or knot.xyz[2] <= BALL_RADIUS_M + 0.05


# ---------------------------------------------------------------------------
# Per-span evidence + fitting
# ---------------------------------------------------------------------------

def _reproj_px(ctx: HybridShotCtx, frame: int, xyz: np.ndarray,
               uv: tuple[float, float]) -> float:
    proj = ctx.project(frame, xyz)
    return float(np.hypot(float(proj[0]) - uv[0], float(proj[1]) - uv[1]))


def _span_evidence(
    observations: Sequence[Any],
    ray_knots: Sequence[Knot],
    a_frame: int,
    b_frame: int,
    ray_weight: float,
) -> list[tuple[int, tuple[float, float], float]]:
    evid = [(o.frame, o.uv, float(o.conf))
            for o in observations if a_frame < o.frame < b_frame]
    evid += [(r.frame, r.uv, ray_weight)
             for r in ray_knots if a_frame < r.frame < b_frame and r.uv is not None]
    return evid


def _choose_model(ctx: HybridShotCtx, a_knot: Knot, b_knot: Knot,
                   observations: Sequence[Any], ray_knots: Sequence[Knot],
                   cfg: Mapping[str, Any]) -> str:
    span_rays = [r for r in ray_knots if a_knot.frame < r.frame < b_knot.frame]
    if any(r.kind.startswith("airborne") for r in span_rays):
        return "flight"
    if not (_is_ground_knot(a_knot) and _is_ground_knot(b_knot)):
        return "flight"
    if not (a_knot.kind in _AMBIGUOUS_GROUND_STATES
            or b_knot.kind in _AMBIGUOUS_GROUND_STATES):
        return "roll"

    duration_s = (b_knot.frame - a_knot.frame) / ctx.fps
    evid = _span_evidence(observations, ray_knots, a_knot.frame, b_knot.frame,
                           cfg["grounded_ray_weight"])
    if not evid:
        return "roll"
    z_level = 0.5 * (float(a_knot.xyz[2]) + float(b_knot.xyz[2]))

    ground_obs = []
    for frame, uv, w in evid:
        C, d = ctx.ray(frame, uv)
        dz = float(d[2])
        if abs(dz) < 1e-9:
            continue
        s = (z_level - float(C[2])) / dz
        if s <= 0:
            continue
        ground_obs.append(((frame - a_knot.frame) / ctx.fps, (C + s * d)[:2], w))
    roll = fit_roll_segment(a_knot.xyz[:2], b_knot.xyz[:2], duration_s,
                             ground_obs, mu_max=cfg["roll_mu_max"])
    roll_worst = 0.0
    for frame, uv, _w in evid:
        t_s = (frame - a_knot.frame) / ctx.fps
        p = roll.eval([t_s], z_level)[0]
        roll_worst = max(roll_worst, _reproj_px(ctx, frame, p, uv))

    v0 = shoot_arc(a_knot.xyz, 0.0, b_knot.xyz, duration_s, cd=cfg["cd"])
    flight_worst = 0.0
    for frame, uv, _w in evid:
        t_s = (frame - a_knot.frame) / ctx.fps
        p = simulate(a_knot.xyz, v0, [t_s], cd=cfg["cd"])[0]
        flight_worst = max(flight_worst, _reproj_px(ctx, frame, p, uv))
    return "roll" if roll_worst <= flight_worst else "flight"


def _fit_roll_iterative(a_xy, b_xy, duration_s: float,
                         ground_obs: list[tuple[float, np.ndarray, float]],
                         cfg: Mapping[str, Any]):
    active = list(ground_obs)
    roll = fit_roll_segment(a_xy, b_xy, duration_s, active, mu_max=cfg["roll_mu_max"])
    for _ in range(max(0, cfg["robust_gate_max_iters"] - 1)):
        if not active:
            break
        resid = [float(np.linalg.norm(roll.eval([t_s], z=0.0)[0][:2] - xy))
                  for t_s, xy, _w in active]
        thresh = max(0.5, 3.0 * float(np.median(resid)))
        new_active = [ob for ob, r in zip(active, resid) if r <= thresh]
        if len(new_active) == len(active):
            break
        active = new_active
        roll = fit_roll_segment(a_xy, b_xy, duration_s, active, mu_max=cfg["roll_mu_max"])
    return roll


def _fit_open_end(ctx: HybridShotCtx, knot: Knot,
                   evidence: list[tuple[int, tuple[float, float], float]],
                   direction: int, cfg: Mapping[str, Any]) -> dict[int, np.ndarray]:
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
            num += conf * t_s * (P[:2] - np.asarray(knot.xyz[:2]))
            den += conf * t_s * t_s
        if den <= 1e-9:
            return {}
        v0_xy = num / den
        if float(np.linalg.norm(v0_xy)) > _OPEN_END_MAX_LAUNCH_SPEED_M_S:
            return {}
        out = {}
        for f in frames:
            t_s = (f - knot.frame) / ctx.fps
            xy = np.asarray(knot.xyz[:2]) + v0_xy * t_s
            out[f] = np.array([xy[0], xy[1], z_level])
        return out if _open_end_plausible(out.values()) else {}

    cd = cfg["cd"]

    def _residual(v0):
        res = []
        for frame, uv, conf in evidence:
            t_s = (frame - knot.frame) / ctx.fps
            p = simulate(np.asarray(knot.xyz), v0, [t_s], cd=cd,
                         magnus_coeff=cfg["magnus_coeff"])[0]
            proj = ctx.project(frame, p)
            w = float(conf) ** 0.5
            res.append(w * (float(proj[0]) - uv[0]))
            res.append(w * (float(proj[1]) - uv[1]))
        return res

    try:
        from scipy.optimize import least_squares
        b = _OPEN_END_MAX_LAUNCH_SPEED_M_S
        sol = least_squares(_residual, np.zeros(3), method="trf",
                             bounds=([-b, -b, -b], [b, b, b]), max_nfev=200)
        v0 = sol.x
    except Exception:  # noqa: BLE001 — best-effort fit; caller falls back
        return {}
    times = [(f - knot.frame) / ctx.fps for f in frames]
    positions = simulate(np.asarray(knot.xyz), v0, times, cd=cd,
                          magnus_coeff=cfg["magnus_coeff"])
    if not _open_end_plausible(positions):
        return {}
    return {f: positions[i] for i, f in enumerate(frames)}


def _open_end_plausible(positions) -> bool:
    lo_x, hi_x = -_OPEN_END_PITCH_MARGIN_M, _OPEN_END_PITCH_LENGTH_M + _OPEN_END_PITCH_MARGIN_M
    lo_y, hi_y = -_OPEN_END_PITCH_MARGIN_M, _OPEN_END_PITCH_WIDTH_M + _OPEN_END_PITCH_MARGIN_M
    for p in positions:
        if not (lo_x <= p[0] <= hi_x and lo_y <= p[1] <= hi_y
                and BALL_RADIUS_M - 1e-6 <= p[2] <= _OPEN_END_MAX_HEIGHT_M):
            return False
    return True


def _shot_spin_bounds(
    bounds: tuple[float, float], a_xyz, b_xyz, duration_s: float, cd: float,
    magnus_coeff: float, max_accel_m_s2: float,
) -> tuple[float, float]:
    """Clamp the spin box so the peak Magnus acceleration k*|omega x v|
    stays <= ``max_accel_m_s2`` at the span's launch speed (a hard physical
    envelope on a shot's curl, tighter than the global 10 rev/s box)."""
    v0 = shoot_arc(a_xyz, 0.0, b_xyz, duration_s, cd=cd, magnus_coeff=magnus_coeff)
    speed = float(np.linalg.norm(v0))
    if speed < 1e-6 or magnus_coeff <= 0.0:
        return bounds
    cap = max_accel_m_s2 / (magnus_coeff * speed)
    hi = min(abs(bounds[1]), cap)
    return (-hi, hi)


def solve_span(
    ctx: HybridShotCtx,
    a_knot: Knot,
    b_knot: Knot,
    observations: Sequence[Any],
    ray_knots: Sequence[Knot],
    cfg: Mapping[str, Any],
    splits_used: int = 0,
    internal_knot_frames: list[int] | None = None,
) -> tuple[dict[int, np.ndarray], list[dict]]:
    """Fit one span between two adjacent hard knots: roll or flight
    (drag + optional Cd fit), with iterative robust gating and
    split-and-retry at the worst-residual evidence frame. Public so
    ``ball_hybrid_gating.py`` can probe a candidate auto-knot's effect on
    a span's own residual without duplicating this logic."""
    if internal_knot_frames is None:
        internal_knot_frames = []
    duration_frames = b_knot.frame - a_knot.frame
    if duration_frames <= 0:
        return {}, []
    duration_s = duration_frames / ctx.fps

    model = _choose_model(ctx, a_knot, b_knot, observations, ray_knots, cfg)

    def _roll_branch(fallback: bool = False) -> tuple[dict[int, np.ndarray], list[dict]]:
        z_level = 0.5 * (float(a_knot.xyz[2]) + float(b_knot.xyz[2]))

        def _ground_project(frame: int, uv: tuple[float, float]) -> np.ndarray | None:
            C, d = ctx.ray(frame, uv)
            dz = float(d[2])
            if abs(dz) < 1e-9:
                return None
            s = (z_level - float(C[2])) / dz
            if s <= 0:
                return None
            return C + s * d

        ground_obs: list[tuple[float, np.ndarray, float]] = []
        for o in observations:
            if not (a_knot.frame < o.frame < b_knot.frame):
                continue
            P = _ground_project(o.frame, o.uv)
            if P is not None:
                ground_obs.append(((o.frame - a_knot.frame) / ctx.fps, P[:2], float(o.conf)))
        for r in ray_knots:
            if not (a_knot.frame < r.frame < b_knot.frame) or r.uv is None:
                continue
            P = _ground_project(r.frame, r.uv)
            if P is not None:
                ground_obs.append(((r.frame - a_knot.frame) / ctx.fps, P[:2],
                                    float(cfg["grounded_ray_weight"])))

        roll = _fit_roll_iterative(a_knot.xyz[:2], b_knot.xyz[:2], duration_s,
                                    ground_obs, cfg)
        frames = list(range(a_knot.frame, b_knot.frame + 1))
        times = [(f - a_knot.frame) / ctx.fps for f in frames]
        positions = roll.eval(times, z_level)
        pts = {f: positions[i] for i, f in enumerate(frames)}

        roll_evid = _span_evidence(observations, ray_knots, a_knot.frame,
                                    b_knot.frame, cfg["grounded_ray_weight"])
        worst_roll: tuple[int, tuple[float, float], float] | None = None
        for frame, uv, _w in roll_evid:
            err = _reproj_px(ctx, frame, pts[frame], uv)
            if worst_roll is None or err > worst_roll[2]:
                worst_roll = (frame, uv, err)

        info = {"span": (a_knot.frame, b_knot.frame), "model": "roll",
                "n_obs": len(ground_obs),
                "max_residual_px": worst_roll[2] if worst_roll else None}
        if fallback:
            # This span was originally classified "flight" by
            # _choose_model but fell back to roll because the flight fit
            # implied an unreachable launch speed — ball_hybrid_gating's
            # acceptance gate treats this as a hard reject regardless of
            # how low the roll fit's OWN residual looks: a roll model
            # can always find SOME low-residual straight-ish line between
            # two points, which would otherwise silently launder a
            # candidate whose true (flight) fit correctly showed it
            # didn't belong here.
            info["fallback_from_flight"] = True

        if (worst_roll is not None
                and worst_roll[2] > cfg["inlier_px"] * cfg["split_residual_factor"]
                and splits_used < cfg["max_splits_per_span"]):
            split_frame, split_uv, _err = worst_roll
            split_xy = _ground_project(split_frame, split_uv)
            if split_xy is not None:
                split_xyz = (float(split_xy[0]), float(split_xy[1]), float(z_level))
                split_knot = Knot(frame=split_frame, xyz=split_xyz, kind="internal_split",
                                   depth_hard=True, source="auto")
                internal_knot_frames.append(split_frame)
                pts1, info1 = solve_span(ctx, a_knot, split_knot, observations,
                                          ray_knots, cfg, splits_used + 1,
                                          internal_knot_frames)
                pts2, info2 = solve_span(ctx, split_knot, b_knot, observations,
                                          ray_knots, cfg, splits_used + 1,
                                          internal_knot_frames)
                return {**pts1, **pts2}, [*info1, *info2]

        return pts, [info]

    if model == "roll":
        return _roll_branch()

    # --- flight ---------------------------------------------------------
    evid_all = _span_evidence(observations, ray_knots, a_knot.frame, b_knot.frame,
                               cfg["airborne_ray_weight"])
    cd = cfg["cd"]
    active_evid = list(evid_all)
    a_xyz, b_xyz = np.asarray(a_knot.xyz), np.asarray(b_knot.xyz)
    v0 = shoot_arc(a_xyz, 0.0, b_xyz, duration_s, cd=cd, magnus_coeff=cfg["magnus_coeff"])
    pts: dict[int, np.ndarray] = {}

    for _iteration in range(max(1, cfg["robust_gate_max_iters"])):
        if cfg["fit_cd"] and len(active_evid) >= cfg["min_obs_for_cd_fit"]:
            def _cost(cd_val: float, _evid=active_evid, _a=a_xyz, _b=b_xyz,
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

        v0 = shoot_arc(a_xyz, 0.0, b_xyz, duration_s, cd=cd, magnus_coeff=cfg["magnus_coeff"])
        frames = list(range(a_knot.frame, b_knot.frame + 1))
        times = [(f - a_knot.frame) / ctx.fps for f in frames]
        positions = simulate(a_xyz, v0, times, cd=cd, magnus_coeff=cfg["magnus_coeff"])
        pts = {f: positions[i] for i, f in enumerate(frames)}

        new_active = [(frame, uv, w) for frame, uv, w in evid_all
                      if _reproj_px(ctx, frame, pts[frame], uv) <= cfg["inlier_px"]]
        if not new_active:
            break
        if {f for f, _, _ in new_active} == {f for f, _, _ in active_evid}:
            break
        active_evid = new_active

    # Physical-plausibility fallback: an interior span's shoot_arc is a
    # boundary-value solve between two HARD knots, so (unlike the free-
    # end fit) it was assumed safe from ever running away — that holds
    # for two genuinely-correct knots, but breaks down when either one
    # is a data-driven point (an accepted auto knot, or a knot pair only
    # 1-3 frames apart) whose true separation implies an unreachable
    # speed. Found via a real origi01 fold0 bench run (2026-09-25): a
    # dense burst of accepted auto player_touch knots at low-but-not-
    # exactly-ground height (_is_ground_knot's _LAUNCH_STATES/height
    # check correctly forced flight for them) produced launch speeds of
    # hundreds of m/s on some pairs despite each knot's OWN position
    # being reasonable — a flight (gravity+drag) model simply cannot
    # explain near-ground short-hop motion between two close points as
    # cleanly as a friction-clamped roll can, and shoot_arc will always
    # find *some* v0 that connects them exactly regardless of how
    # physically absurd. Falling back to the roll fit (which can never
    # explode — its acceleration is Coulomb-clamped) is strictly safer
    # than trusting an unreachable launch speed; this also protects
    # split-and-retry recursion below, whose internal-split knots are
    # exactly where the wild speeds were observed (never visible to
    # ball_hybrid_gating's own probe, which disables splitting).
    if float(np.linalg.norm(v0)) > cfg["max_launch_speed_m_s"]:
        return _roll_branch(fallback=True)

    # Optional bounded spin (Magnus) refinement — IC-E's
    # ball_hybrid_spin.fit_span_spin, gated off by default
    # (ball.hybrid.spin.enabled) until the benchmark decision. Runs right
    # after robust gating converges, on the FINAL inlier set
    # (active_evid) — the spin fit is judged against exactly the evidence
    # the drag-only arc above was judged against. Both knots stay exact
    # (fit_span_spin re-shoots v0 via shoot_arc for every trial omega);
    # accepted only when it clears fit_span_spin's own BIC + residual-gain
    # bars, so a spin-free or under-evidenced span is untouched.
    spin_cfg = cfg.get("spin") or {}
    span_omega_world: tuple[float, float, float] | None = None
    span_omega_rad_s: float | None = None
    shot_span = (bool(spin_cfg.get("shot_spans", False))
                 and a_knot.frame in set(spin_cfg.get("shot_start_frames") or ()))
    spin_on = bool(spin_cfg.get("enabled", False)) or shot_span
    spin_bounds = tuple(spin_cfg.get("bounds", DEFAULT_SPIN_BOUNDS))
    if shot_span and not spin_cfg.get("enabled", False):
        spin_bounds = _shot_spin_bounds(
            spin_bounds, a_xyz, b_xyz, duration_s, cd, cfg["magnus_coeff"],
            float(spin_cfg.get("shot_max_accel_m_s2", 10.0)))
    # A shot span curls, so the drag-only arc above can't explain it and its
    # robust inlier gate collapses -- fit the spin against ALL span evidence
    # under a soft-L1 loss instead (design D6.3). Global spin keeps the
    # validated inlier-set / plain-LS behaviour.
    spin_evid = evid_all if shot_span else active_evid
    spin_robust = float(cfg["inlier_px"]) if shot_span else None
    if spin_on and len(spin_evid) >= int(spin_cfg.get("min_obs", 8)):
        obs_frames = np.array([f for f, _, _ in spin_evid], dtype=float)
        # Absolute clip-time base throughout (obs_times/t_a/t_b all
        # frame/fps) — project_fn maps t_s straight to a frame via
        # round(t_s * fps), matching scripts/eval_ball_spin.py's
        # validated reference wiring exactly (mixing an absolute and a
        # span-relative base across these three is the KeyError IC-E hit
        # — never do that).
        obs_times = obs_frames / ctx.fps
        obs_uv = np.array([uv for _, uv, _ in spin_evid], dtype=float)
        obs_conf = np.array([w for _, _, w in spin_evid], dtype=float)
        t_a = a_knot.frame / ctx.fps
        t_b = b_knot.frame / ctx.fps

        def _spin_project_fn(t_s: float, xyz: np.ndarray, _ctx=ctx) -> np.ndarray:
            return _ctx.project(int(round(t_s * _ctx.fps)), xyz)

        try:
            spin_result = fit_span_spin(
                a_xyz, t_a, b_xyz, t_b, obs_times, obs_uv, _spin_project_fn,
                cd=cd, bounds=spin_bounds,
                magnus_coeff=cfg["magnus_coeff"], obs_conf=obs_conf,
                min_obs=int(spin_cfg.get("min_obs", 8)),
                min_delta_bic=float(spin_cfg.get("min_delta_bic", 6.0)),
                min_resid_gain=float(spin_cfg.get("min_resid_gain", 0.10)),
                robust_scale_px=spin_robust,
            )
        except Exception:  # noqa: BLE001 — spin is best-effort enrichment
            spin_result = None

        if spin_result is not None:
            omega = np.array(spin_result.omega_world, dtype=float)
            v0 = shoot_arc(a_xyz, 0.0, b_xyz, duration_s, cd=cd, omega=omega,
                            magnus_coeff=cfg["magnus_coeff"])
            frames = list(range(a_knot.frame, b_knot.frame + 1))
            times = [(f - a_knot.frame) / ctx.fps for f in frames]
            positions = simulate(a_xyz, v0, times, cd=cd, omega=omega,
                                  magnus_coeff=cfg["magnus_coeff"])
            pts = {f: positions[i] for i, f in enumerate(frames)}
            span_omega_world = spin_result.omega_world
            span_omega_rad_s = spin_result.rad_s

    worst: tuple[int, tuple[float, float], float] | None = None
    for frame, uv, _w in evid_all:
        err = _reproj_px(ctx, frame, pts[frame], uv)
        if worst is None or err > worst[2]:
            worst = (frame, uv, err)

    info = {"span": (a_knot.frame, b_knot.frame), "model": "flight", "cd": cd,
            "n_obs": len(evid_all), "n_inliers": len(active_evid),
            "max_residual_px": worst[2] if worst else None,
            # p0/v0/g fully determine this span's parabola (+ Magnus, via
            # omega_world/rad_s below when present) — carried so a caller
            # (ball.py's hybrid wiring) can build a FlightSegment for
            # BallTrack.flight_segments / ball_orientation.integrate_
            # orientation without re-deriving v0 via a second shoot_arc call.
            "p0": tuple(float(x) for x in a_xyz),
            "v0": tuple(float(x) for x in v0),
            "g": float(G)}
    if span_omega_world is not None:
        info["omega_world"] = span_omega_world
        info["rad_s"] = span_omega_rad_s

    # A shot span whose bounded-curl fit was accepted is explained by one
    # physical arc: a lone far-off detection (false track near the striker)
    # must not split it, which would also strand the sub-spans from the
    # spin refinement.
    shot_spin_accepted = shot_span and span_omega_world is not None
    if (worst is not None
            and not shot_spin_accepted
            and worst[2] > cfg["inlier_px"] * cfg["split_residual_factor"]
            and splits_used < cfg["max_splits_per_span"]):
        split_frame, split_uv, _err = worst
        C, d = ctx.ray(split_frame, split_uv)
        arc_pt = pts[split_frame]
        _, along = point_ray_distance(arc_pt, C, d)
        faithful = C + max(along, 0.0) * d
        z = max(float(faithful[2]), BALL_RADIUS_M)
        split_xyz = (float(faithful[0]), float(faithful[1]), float(z))
        split_knot = Knot(frame=split_frame, xyz=split_xyz, kind="bounce",
                           depth_hard=True, source="auto")
        internal_knot_frames.append(split_frame)
        pts1, info1 = solve_span(ctx, a_knot, split_knot, observations, ray_knots,
                                  cfg, splits_used + 1, internal_knot_frames)
        pts2, info2 = solve_span(ctx, split_knot, b_knot, observations, ray_knots,
                                  cfg, splits_used + 1, internal_knot_frames)
        merged = {**pts1, **pts2}
        return merged, [*info1, *info2]

    return pts, [info]


def build_trajectory(
    ctx: HybridShotCtx,
    hard_knots: Sequence[Knot],
    ray_knots: Sequence[Knot],
    observations: Sequence[Any],
    cfg: Mapping[str, Any],
) -> tuple[dict[int, np.ndarray], list[dict], list[int]]:
    """The full physics track P_phys(frame): interior spans + free-end
    head/tail. Returns ``(positions, span_diagnostics,
    internal_split_frames)``."""
    pts: dict[int, np.ndarray] = {}
    diagnostics: list[dict] = []
    internal_knot_frames: list[int] = []

    for a_knot, b_knot in zip(hard_knots, hard_knots[1:]):
        span_pts, span_info = solve_span(ctx, a_knot, b_knot, observations,
                                          ray_knots, cfg, 0, internal_knot_frames)
        pts.update(span_pts)
        diagnostics.extend(span_info)

    if len(hard_knots) == 1:
        pts[hard_knots[0].frame] = np.asarray(hard_knots[0].xyz)

    if hard_knots:
        all_frames = sorted(set(list(pts)
                                 + [o.frame for o in observations]
                                 + [r.frame for r in ray_knots]))
        first_frame = hard_knots[0].frame
        last_frame = hard_knots[-1].frame
        head_candidates = [f for f in all_frames if f < first_frame]
        tail_candidates = [f for f in all_frames if f > last_frame]
        if head_candidates:
            head_start = min(head_candidates)
            head_evid = _span_evidence(observations, ray_knots,
                                        head_start - 1, first_frame,
                                        cfg["airborne_ray_weight"])
            fitted = _fit_open_end(ctx, hard_knots[0], head_evid, -1, cfg)
            for f in range(head_start, first_frame):
                pts[f] = fitted.get(f, np.asarray(hard_knots[0].xyz))
            diagnostics.append({
                "span": (head_start, first_frame),
                "model": "open_head" if fitted else "hold_head",
                "n_obs": len(head_evid),
            })
        if tail_candidates:
            tail_end = max(tail_candidates)
            tail_evid = _span_evidence(observations, ray_knots,
                                        last_frame, tail_end + 1,
                                        cfg["airborne_ray_weight"])
            fitted = _fit_open_end(ctx, hard_knots[-1], tail_evid, 1, cfg)
            for f in range(last_frame + 1, tail_end + 1):
                pts[f] = fitted.get(f, np.asarray(hard_knots[-1].xyz))
            diagnostics.append({
                "span": (last_frame, tail_end),
                "model": "open_tail" if fitted else "hold_tail",
                "n_obs": len(tail_evid),
            })

    return pts, diagnostics, internal_knot_frames


def smooth_non_event_knot_windows(
    ctx: HybridShotCtx,
    pts: Mapping[int, np.ndarray],
    hard_knots: Sequence[Knot],
    cfg: Mapping[str, Any],
) -> dict[int, np.ndarray]:
    """Local C1 fix for every NON-EVENT knot's velocity kink — see module
    docstring / ``resolve_knots``. Knot positions are unchanged; only the
    velocity direction either side is turned smoothly."""
    frames_sorted = sorted(pts)
    if len(frames_sorted) < 3:
        return dict(pts)
    idx_of = {f: i for i, f in enumerate(frames_sorted)}
    out: dict[int, np.ndarray] = dict(pts)

    def _tangent(i: int) -> np.ndarray:
        i_a = max(0, i - 1)
        i_b = min(len(frames_sorted) - 1, i + 1)
        fa, fb = frames_sorted[i_a], frames_sorted[i_b]
        dt = (fb - fa) / ctx.fps
        return (pts[fb] - pts[fa]) / dt if dt > 1e-9 else np.zeros(3)

    for knot in hard_knots:
        if is_sharp_knot(knot):
            continue
        f0 = knot.frame
        if f0 not in idx_of:
            continue
        i0 = idx_of[f0]
        if i0 == 0 or i0 == len(frames_sorted) - 1:
            continue

        f_before, f_after = frames_sorted[i0 - 1], frames_sorted[i0 + 1]
        dt_in = (f0 - f_before) / ctx.fps
        dt_out = (f_after - f0) / ctx.fps
        v_in = (pts[f0] - pts[f_before]) / dt_in if dt_in > 1e-9 else np.zeros(3)
        v_out = (pts[f_after] - pts[f0]) / dt_out if dt_out > 1e-9 else np.zeros(3)
        speed = 0.5 * (float(np.linalg.norm(v_in)) + float(np.linalg.norm(v_out)))

        k = int(np.clip(
            round(cfg["hermite_k_ref_frames"] * cfg["hermite_k_ref_speed_m_s"]
                  / max(speed, 0.5)),
            cfg["hermite_k_min"], cfg["hermite_k_max"]))

        i_lo = max(0, i0 - k)
        i_hi = min(len(frames_sorted) - 1, i0 + k)
        f_lo, f_hi = frames_sorted[i_lo], frames_sorted[i_hi]
        if f_lo == f0 or f_hi == f0:
            continue

        m_lo, m_hi = _tangent(i_lo), _tangent(i_hi)
        m_avg = 0.5 * (v_in + v_out)
        p_lo, p_knot, p_hi = pts[f_lo], pts[f0], pts[f_hi]

        frames_a = frames_sorted[i_lo:i0 + 1]
        if len(frames_a) >= 2:
            T_a = (f0 - f_lo) / ctx.fps
            fracs_a = np.array([(f - f_lo) / (f0 - f_lo) for f in frames_a])
            pos_a = hermite_blend(p_lo, m_lo * T_a, p_knot, m_avg * T_a, fracs_a)
            for f, p in zip(frames_a, pos_a):
                out[f] = p

        frames_b = frames_sorted[i0:i_hi + 1]
        if len(frames_b) >= 2:
            T_b = (f_hi - f0) / ctx.fps
            fracs_b = np.array([(f - f0) / (f_hi - f0) for f in frames_b])
            pos_b = hermite_blend(p_knot, m_avg * T_b, p_hi, m_hi * T_b, fracs_b)
            for f, p in zip(frames_b, pos_b):
                out[f] = p

    return out


# ---------------------------------------------------------------------------
# Delta (broadcast/physics) blend
# ---------------------------------------------------------------------------

def delta_evidence(
    ctx: HybridShotCtx,
    pts: Mapping[int, np.ndarray],
    observations: Sequence[Any],
    ray_knots: Sequence[Knot],
    cfg: Mapping[str, Any],
) -> dict[int, tuple[tuple[float, float, float], float]]:
    """Per-frame ``(delta_xyz, weight)`` evidence for ``blend_deltas``:
    ``faithful_point - P_phys`` at confident inlier observations and at
    ray knots. A ray knot's "faithful point" is always computed at
    P_phys's OWN depth (``along``) projected onto the ray — i.e. this is
    already a lateral-only pull by construction (see module docstring:
    the depth-hard fix lives in the SPAN FIT's evidence weighting, not
    here — this function was already depth-safe)."""
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
    for r in ray_knots:
        if r.uv is None:
            continue
        p_phys = pts.get(r.frame)
        if p_phys is None:
            continue
        C, d_hat = ctx.ray(r.frame, r.uv)
        perp, along = point_ray_distance(p_phys, C, d_hat)
        if perp > cfg["anchor_delta_max_m"]:
            continue
        faithful = C + max(along, 0.0) * d_hat
        delta = faithful - p_phys
        evidence[r.frame] = (tuple(float(x) for x in delta), float(cfg["ray_anchor_weight"]))
    return evidence


_MODEL_STATE = {"roll": "grounded", "open_head": None, "open_tail": None,
                 "hold_head": None, "hold_tail": None, "flight": "flight"}


def _span_state_by_frame(
    diagnostics: Sequence[Mapping[str, Any]],
    hard_knots: Sequence[Knot],
) -> dict[int, str]:
    """Derive a per-frame ``"grounded"``/``"flight"`` state from
    ``build_trajectory``'s span diagnostics — the ``src.schemas.ball_
    track.State`` the caller (``ball.py``) needs, without changing
    ``build_trajectory``'s return shape (``ball_hybrid_gating.py`` calls
    ``solve_span`` directly and doesn't need this)."""
    by_frame: dict[int, str] = {}
    knot_by_frame = {k.frame: k for k in hard_knots}
    for d in diagnostics:
        a_frame, b_frame = d["span"]
        model = d["model"]
        if model in ("roll", "flight"):
            state = _MODEL_STATE[model]
        else:
            # open_head/open_tail/hold_head/hold_tail: ground-ness comes
            # from the anchoring knot itself (the head span's knot is at
            # a_frame for a tail-direction span or b_frame for a head one
            # — whichever endpoint is the actual hard knot, not the
            # evidence-extent frame).
            anchor_knot = knot_by_frame.get(a_frame) or knot_by_frame.get(b_frame)
            state = "grounded" if anchor_knot is not None and _is_ground_knot(anchor_knot) else "flight"
        for f in range(a_frame, b_frame + 1):
            by_frame[f] = state
    return by_frame


def finalize_track(
    ctx: HybridShotCtx,
    hard_knots: Sequence[Knot],
    ray_knots: Sequence[Knot],
    observations: Sequence[Any],
    cfg: Mapping[str, Any],
) -> tuple[dict[int, dict], dict]:
    """Full trajectory build for an already-resolved (manual + any
    gating-accepted auto) knot set: physics track -> local Hermite
    smoothing at non-event knots -> delta blend -> exact re-snap at sharp
    knots. Returns ``(frames, diagnostics)`` where ``frames`` maps
    ``frame -> {"xyz": Vec3, "state": "grounded"|"flight", "conf": float,
    "mode": "anchor"|"faithful"|"simulated"}``.

    This is the trajectory layer's top-level entry point once the caller
    has already decided which auto-event candidates to fold in (that
    decision is ``ball_hybrid_gating.gate_auto_events``'s job, not
    this module's — see module docstring)."""
    pts, span_diag, internal_knot_frames = build_trajectory(
        ctx, hard_knots, ray_knots, observations, cfg)

    if not pts:
        return {}, {"spans": [], "n_knots": len(hard_knots),
                     "n_ray_knots": len(ray_knots), "anchor_residuals": [],
                     "anchor_not_honoured": []}

    pts = smooth_non_event_knot_windows(ctx, pts, hard_knots, cfg)
    state_by_frame = _span_state_by_frame(span_diag, hard_knots)

    evidence = delta_evidence(ctx, pts, observations, ray_knots, cfg)
    sharp_frames = [k.frame for k in hard_knots if is_sharp_knot(k)]
    event_frames = sharp_frames + internal_knot_frames
    frames_sorted = sorted(pts)
    blended = blend_deltas(frames_sorted, evidence,
                            halflife_frames=cfg["blend_halflife_frames"],
                            event_frames=event_frames)

    max_step_by_frame: dict[int, float] = {}
    frac = cfg["blend_max_delta_step_frac"]
    speed_floor = cfg["blend_max_delta_step_floor_m_s"]
    for i, f in enumerate(frames_sorted):
        f_prev = frames_sorted[max(i - 1, 0)]
        f_next = frames_sorted[min(i + 1, len(frames_sorted) - 1)]
        dt = (f_next - f_prev) / ctx.fps
        speed = (float(np.linalg.norm(pts[f_next] - pts[f_prev])) / dt
                 if dt > 1e-9 else 0.0)
        max_step_by_frame[f] = frac * max(speed, speed_floor) / ctx.fps
    blended = clamp_delta_rate(frames_sorted, blended,
                                max_step_m=max_step_by_frame,
                                event_frames=event_frames)

    knot_by_frame = {k.frame: k for k in hard_knots}
    ray_by_frame = {r.frame: r for r in ray_knots}

    out: dict[int, dict] = {}
    mode_counts = {"anchor": 0, "faithful": 0, "simulated": 0}
    for f in frames_sorted:
        base = np.asarray(pts[f], dtype=float)
        delta, conf = blended.get(f, ((0.0, 0.0, 0.0), 0.0))
        final = base + np.asarray(delta, dtype=float)
        mode = "faithful" if conf >= 0.5 else "simulated"
        out_conf: float = conf

        if f in knot_by_frame and is_sharp_knot(knot_by_frame[f]):
            final = np.array(knot_by_frame[f].xyz, dtype=float)
            mode, out_conf = "anchor", 1.0
        elif f in knot_by_frame:
            mode = "anchor"
        elif f in ray_by_frame:
            mode = "anchor"

        final[2] = max(float(final[2]), BALL_RADIUS_M)
        mode_counts[mode] = mode_counts.get(mode, 0) + 1
        out[f] = {
            "xyz": (float(final[0]), float(final[1]), float(final[2])),
            "state": state_by_frame.get(f, "flight"),
            "conf": float(out_conf) if out_conf is not None else 0.0,
            "mode": mode,
        }

    by_frame_final = {f: v["xyz"] for f, v in out.items()}
    not_honoured_px = cfg["anchor_not_honoured_px"]
    anchor_residuals = []
    for r in ray_knots:
        xyz = by_frame_final.get(r.frame)
        if xyz is None or r.uv is None:
            continue
        res_px = _reproj_px(ctx, r.frame, np.asarray(xyz), r.uv)
        anchor_residuals.append({"frame": r.frame, "kind": r.kind, "residual_px": res_px})
    for k in hard_knots:
        if is_sharp_knot(k):
            continue
        xyz = by_frame_final.get(k.frame)
        if xyz is None:
            continue
        uv = ctx.project(k.frame, np.asarray(k.xyz))
        res_px = _reproj_px(ctx, k.frame, np.asarray(xyz), (float(uv[0]), float(uv[1])))
        anchor_residuals.append({"frame": k.frame, "kind": k.kind, "residual_px": res_px})
    anchor_not_honoured = [a["frame"] for a in anchor_residuals
                            if a["residual_px"] > not_honoured_px]

    diagnostics = {
        "spans": span_diag,
        "n_knots": len(hard_knots),
        "n_ray_knots": len(ray_knots),
        "n_internal_splits": len(internal_knot_frames),
        "mode_counts": mode_counts,
        "anchor_residuals": anchor_residuals,
        "anchor_not_honoured": anchor_not_honoured,
    }
    return out, diagnostics


# ---------------------------------------------------------------------------
# Single orchestrating entry point (resolve manual + gate auto + finalize)
# ---------------------------------------------------------------------------

def run_trajectory(
    ctx: HybridShotCtx,
    observations: Sequence[Any],
    anchors: Sequence[Any],
    *,
    auto_anchors: Sequence[Any] = (),
    fixes: Sequence[Any] = (),
    cues: Sequence[Any] = (),
    cfg: Mapping[str, Any] | None = None,
    gating_cfg: Mapping[str, Any] | None = None,
    player_context: Any = None,
    goal_outcome: str | None = None,
) -> tuple[dict[int, dict], dict]:
    """Convenience one-call entry point: resolve manual ``anchors``/
    ``fixes`` into knots, gate ``auto_anchors`` through
    ``ball_hybrid_gating.gate_auto_events`` (``cues`` is passed through
    as that gate's ``corroboration``), and build the final track. This is
    what ``ball.py``'s ``_solve_shot`` wiring and any other single-shot
    caller (bench harnesses, hold-out eval scripts) should reach for
    instead of re-deriving the resolve -> gate -> finalize sequence
    themselves. ``cfg`` is this module's own cfg (``DEFAULT_CFG``
    overrides); ``gating_cfg`` is ``ball_hybrid_gating.DEFAULT_GATING_CFG``
    overrides.

    Returns ``(frames, diagnostics)`` — same shape as ``finalize_track``,
    with an added ``diagnostics["gate"]`` block from the auto-event gate.
    A lazy import avoids a module-level circular dependency (``ball_
    hybrid_gating`` imports this module for ``solve_span``/
    ``resolve_knots``).
    """
    from src.utils.ball_hybrid_gating import gate_auto_events as _gate_auto_events

    from src.utils.ball_direction_gate import filter_reversed_observations
    from src.utils.ball_goal_constraint import (
        GoalEvent, _goal_end_for_x, contain_in_net, goal_check,
        infer_line_cross_knots, normalize_outcome)

    tcfg = full_cfg(cfg)
    hard, ray = resolve_knots(ctx, anchors, fixes, source="manual",
                               player_context=player_context)

    # D6.1 -- goal-mouth constraint: an in-memory line-cross knot from the
    # operator's airborne anchor nearest a goal_impact (anchor file untouched).
    goal_cfg = tcfg.get("goal") or {}
    goal_event = None
    if goal_cfg.get("enabled", True):
        extra, goal_event = infer_line_cross_knots(
            ctx, anchors, goal_cfg, goal_outcome)
        have = {k.frame for k in hard}
        hard = sorted(list(hard) + [k for k in extra if k.frame not in have],
                      key=lambda k: k.frame)

    # D6.3 -- shot spans get a bounded Magnus refinement even with the
    # global spin switch off: tag the start frames of shot/volley/spin touches.
    # Only for shots that end in a goal event: on non-goal shot/volley spans
    # (s013 gate, 2026-10-03) the unanchored Magnus fit doubled real p50.
    spin_cfg = dict(tcfg.get("spin") or {})
    if spin_cfg.get("shot_spans") and goal_event is not None:
        starts = _shot_start_frames(list(anchors) + list(auto_anchors))
        spin_cfg["shot_start_frames"] = tuple(sorted(starts))
        tcfg = {**tcfg, "spin": spin_cfg}

    gate_result = _gate_auto_events(
        ctx, hard, ray, auto_anchors, observations,
        cfg=gating_cfg, trajectory_cfg=tcfg,
        player_context=player_context, corroboration=cues)

    all_hard = sorted(list(hard) + list(gate_result.accepted_hard),
                       key=lambda k: k.frame)
    all_ray = sorted(list(ray) + list(gate_result.accepted_ray),
                      key=lambda k: k.frame)

    frames, diagnostics = finalize_track(ctx, all_hard, all_ray, observations, tcfg)

    # D6.2 -- direction-consistency gate: drop detection runs that move
    # against the fitted flight, then re-fit once without them.
    dg_cfg = tcfg.get("direction_gate") or {}
    dropped: list[int] = []
    if dg_cfg.get("enabled", True) and frames:
        def _fitted_uv(f: int):
            e = frames.get(f)
            if e is None or e["state"] != "flight" or not ctx.has_frame(f):
                return None
            return ctx.project(f, np.asarray(e["xyz"], dtype=float))
        protect = {k.frame for k in all_hard} | {r.frame for r in all_ray}
        kept, dropped = filter_reversed_observations(
            observations, _fitted_uv, protect_frames=protect, cfg=dg_cfg)
        if dropped:
            observations = kept
            frames, diagnostics = finalize_track(
                ctx, all_hard, all_ray, observations, tcfg)
    diagnostics["direction_gate"] = {"n_dropped": len(dropped),
                                      "frames": sorted(dropped)}
    diagnostics["gate"] = {
        "n_candidates": gate_result.n_candidates,
        "n_accepted_hard": len(gate_result.accepted_hard),
        "n_accepted_ray": len(gate_result.accepted_ray),
        "n_rejected_kind": gate_result.n_rejected_kind,
        "n_rejected_confidence": gate_result.n_rejected_confidence,
        "n_rejected_near_manual": gate_result.n_rejected_near_manual,
        "n_rejected_consistency": gate_result.n_rejected_consistency,
        "n_rejected_residual": gate_result.n_rejected_residual,
        "n_rejected_implausible_velocity": gate_result.n_rejected_implausible_velocity,
    }

    if (goal_event is None and goal_cfg.get("enabled", True)
            and normalize_outcome(goal_outcome) == "goal"):
        # No operator goal_impact but the shot is explicitly a goal: an
        # accepted AUTO goal_impact knot locates it (checked, no line-cross
        # knot inferred). Auto knots never make a goal on their own.
        auto_gi = [k for k in gate_result.accepted_hard if k.kind == "goal_impact"]
        if auto_gi:
            goal_event = GoalEvent(auto_gi[0].frame,
                                   _goal_end_for_x(float(auto_gi[0].xyz[0])),
                                   None, None)
    gc = goal_check(frames, goal_event)
    if gc is not None:
        diagnostics["goal_check"] = gc
        frames, n_net = contain_in_net(
            frames, goal_event, gc,
            protect_frames=[k.frame for k in all_hard] + [r.frame for r in all_ray])
        gc["net_clamped_frames"] = n_net
    return frames, diagnostics


def _shot_start_frames(anchors: Sequence[Any]) -> set[int]:
    """Frames of touches that launch a shot: ``player_touch``/``kick``-like
    anchors tagged ``touch_type`` shot|volley, or carrying a spin preset."""
    out: set[int] = set()
    for a in anchors:
        if isinstance(a, Mapping):
            frame, tt, spin = a.get("frame"), a.get("touch_type"), a.get("spin")
            state = a.get("state")
        else:
            frame = getattr(a, "frame", None)
            tt, spin = getattr(a, "touch_type", None), getattr(a, "spin", None)
            state = getattr(a, "state", None)
        if frame is None or state not in ("player_touch", "kick", "volley"):
            continue
        if tt in ("shot", "volley") or spin:
            out.add(int(frame))
    return out
