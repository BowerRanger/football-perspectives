"""The hybrid ball extractor (Task B / SPIKE core method).

``run_hybrid(ctx, observations, anchors, fixes=(), auto_anchors=(), cfg=None) -> Track``

## Knot taxonomy and the local-join fix (iteration 3.1)

Every RESOLVED anchor (any state that ``ball_eval.anchor_gt_world`` /
goal geometry / the joint-depth convention can pin to a real 3-D point)
is a hard SPAN-BOUNDARY KNOT — this is the iteration-2 per-anchor span
structure, restored. Knots split into two kinds for how their velocity
is treated:

* **Sharp** (``_is_sharp_knot``: ``EVENT_STATES`` — player_touch, kick,
  bounce, header, volley, chest, catch, goal_impact — plus cross-replay
  ``fixes`` and any data-discovered internal split). A real velocity
  break belongs here — exact 3-D re-snap, raw per-span fit kink kept
  as-is.
* **Non-sharp** (everything else — in practice almost always a
  "grounded" click). Still an exact-position span boundary for FITTING
  (recovering per-anchor accuracy), but its velocity is smoothed
  LOCALLY over a small window (``_smooth_non_event_knot_windows``, k~3-5
  frames scaled to speed): a two-piece cubic Hermite blend, each piece
  running from the window EDGE (that span's own position+velocity there,
  so it joins smoothly with the untouched path outside the window) in to
  the knot's own exact position, sharing one tangent AT the knot (the
  average of the incoming/outgoing velocities). The knot's position is
  unchanged; only the velocity direction either side of it turns
  smoothly instead of kinking.

**History**: the first iteration-3 attempt made every non-event anchor
permanently SOFT evidence inside one continuous fit per event-free
chain (chains between rare touches/bounces, potentially 50+ frames).
That removed the kink but chains between real events span genuinely
curved real motion (a dribble with several direction changes) that a
single rigid roll/flight model badly underfit — measured regression on
gberch/base: %<=20cm 0.91->0.36, ground-truth floating >1m, held-out p50
0.21->1.67m. Reverted to per-anchor knots + local Hermite join instead:
keeps the accuracy, still eliminates the kink (validated: gberch/base
heading_break count now matches truth's own count exactly, down from 35
vs. 8 pre-fix).

**This is a documented, narrowly-scoped relaxation of "operator input
always wins"**: a non-sharp knot's reported position may deviate from
its literal click by the Hermite window's own small pass-through
tolerance (tracked per-anchor in ``diagnostics["anchor_residuals"]``,
``>4px`` flagged as ``"anchor_not_honoured"``); its velocity is NEVER
exactly what a naive per-span fit would give, by design. Manual anchors
still always win over auto/detector evidence. EVENT anchors (and fixes)
are pinned exactly with zero relaxation.

## Pipeline (see CONTRACT.md and the task brief)

  a. Resolve knots + ray evidence (``resolve_knots``, taxonomy above). A
     state that can't be pinned to a 3-D point without extra context
     (airborne_*, off_screen_flight, or an EVENT anchor missing its
     joint/goal-geometry hit) becomes a lateral-exact ray constraint,
     depth free.
  a2. Auto-event knots: ``auto_anchors`` (the CURRENT ball stage's own
     auto-generated EVENTS — kinematic touches, velocity-break bounces,
     auto goal impacts) are resolved the same way and folded in as
     extra, SOFT-ish knots on top of the manual layer — this PoC's
     hybrid is a new TRAJECTORY layer over the existing EVENT layer, not
     a replacement for it. Only auto anchors whose state is itself an
     EVENT state are considered at all (an auto "grounded"/"airborne_*"
     entry is a synthetic interpolation of evidence already in the fit,
     not new information, so it's ignored outright). Manual anchors
     always win: an auto knot/ray within ``auto_anchor_min_frame_gap``
     frames of any manual knot/ray is dropped. A surviving auto knot is
     accepted only if inserting it doesn't make its bracketing span's
     own worst-evidence-residual gate any worse (``_integrate_auto_
     knots``); accepted/rejected counts are in
     ``diagnostics["auto_anchors"]``.
  b. Robust evidence: real detector observations are graded per-span
     against that span's own physics fit, ITERATED (fit -> gate outliers
     by reprojection residual -> refit) up to ``robust_gate_max_iters``
     times or until the inlier set stabilises; low-confidence/high-
     residual detections never move the knot-exact fit.
  c. Per span between consecutive knots: pick roll (both ends
     ground-level, no launch state) or flight (gravity + drag, optional
     Cd fit bounded to ``cd_bounds``, both endpoints always hit exactly
     via boundary-value shooting); split-and-retry (up to
     ``max_splits_per_span``, both roll and flight) at the
     worst-residual evidence frame when a single fit can't explain the
     span (treated as an internal, sharp knot — a genuine
     data-discovered event — recursed).
  d. Physics track P_phys(frame) for every frame in [first knot/evidence,
     last knot/evidence]. The clip head/tail gets a genuine free-end fit
     anchored at that one knot (ground-roll or drag-flight, bounded +
     plausibility-checked) covering exactly out to where the evidence in
     that direction ends (``_fit_open_end``); falls back to holding the
     knot only when there's too little evidence or the fit is
     implausible. Every non-sharp knot's local kink is then smoothed
     (``_smooth_non_event_knot_windows``, above).
  e. Hybrid blend: delta = faithful_point - P_phys at confident inlier
     evidence and at ray constraints (gated: a ray whose implied pull
     would move the baseline more than ``anchor_delta_max_m`` is left
     unpulled — see ``_delta_evidence`` — rather than yanking a whole
     smoothing-kernel window off course), smoothed by ``blend.
     blend_deltas`` (a C-infinity Gaussian-family kernel — no corner at
     an evidence frame) and rate-limited by ``blend.clamp_delta_rate``
     (delta's own frame-to-frame change capped to a fraction of local
     physics-track speed) so the smoothing pass itself can't introduce a
     kink either. Final = P_phys + delta_s; sharp knots re-snapped
     exactly afterwards.
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

from .blend import blend_deltas, clamp_delta_rate
from .hybrid_physics import (
    BALL_RADIUS_M,
    CD_BOUNDS,
    CD_DEFAULT,
    DEFAULT_MAGNUS_COEFF,
    fit_roll_segment,
    hermite_blend,
    shoot_arc,
    simulate,
)
from .types import Observation, Track, TrackFrame

GROUND_EXACT_STATES = frozenset(GROUND_LEVEL_STATES) | {"bounce"}
_LAUNCH_STATES = frozenset({
    "kick", "header", "volley", "chest", "goal_impact",
})
# NOTE: "player_touch" is deliberately NOT here. It's a generic catch-all
# contact state (dribble touch, ground pass, reception -- not necessarily
# a launch), unlike kick/header/volley/chest/goal_impact which genuinely
# almost always send the ball into the air. Iteration 3's chains span
# between rare EVENT knots (potentially 50+ frames), so unconditionally
# treating every player_touch as airborne forced long, purely-grounded
# dribble sequences bracketed by two grounded touches into a single
# flight-arc fit -- discovered on gberch/base: a (209, 310) chain whose
# both endpoints resolved at z=0.11m still got model="flight" purely from
# state membership, producing a wildly-elevated internal split knot and a
# ~4m error blip. A player_touch's RESOLVED HEIGHT (the height check in
# _is_ground_knot below) is the correct signal for whether it was
# actually a ground-level contact.

# The knot taxonomy (see module docstring): only these states are real
# physical events (a velocity break there is expected) and become hard
# span-boundary knots. Every other anchor state (grounded, airborne_*,
# off_screen_flight) is a non-event waypoint that becomes weighted pixel
# evidence inside one continuous chain fit instead. Matches
# run_all.py's ``_CONTACT_ANCHOR_STATES`` (kept as an independent
# constant here rather than imported, to avoid a reverse dependency on
# run_all.py from the core extractor module).
EVENT_STATES = frozenset({
    "player_touch", "kick", "bounce", "header", "volley", "chest",
    "catch", "goal_impact",
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
    # Governs the DELTA-BLEND pull/confidence around a non-event anchor
    # (how far the "this frame is anchor-backed" halo extends and how
    # much it can nudge nearby frames) -- NOT how tightly the underlying
    # chain FIT itself honours the click (that's anchor_fit_weight,
    # below). Deliberately modest: a wide/heavy value here previously
    # (iteration 1) inflated "faithful" confidence for several frames
    # around a manual click purely from the anchor's pull rather than
    # actual detector evidence.
    "ray_anchor_weight": 1.2,
    "min_obs_for_cd_fit": 5,
    "split_residual_factor": 2.5,
    # iteration 2 additions
    "robust_gate_max_iters": 3,
    "auto_anchor_min_frame_gap": 2,  # drop an auto anchor within this many
                                     # frames of any manual knot/ray anchor
    "min_evid_for_open_end_fit": 2,
    # iteration 3 additions (knot taxonomy: non-event anchors are soft
    # evidence inside one continuous chain fit, not hard knots -- see
    # module docstring)
    "anchor_fit_weight": 20.0,       # weight of a non-event anchor click in
                                      # the CHAIN fit (roll/flight/open-end)
                                      # relative to a conf~1.0 detector obs;
                                      # tuned so anchors land within ~2px
                                      # despite being outnumbered by noisier
                                      # real detections in a long chain.
    "anchor_not_honoured_px": 4.0,   # diagnostics flag threshold
    "anchor_delta_max_m": 1.5,       # skip a ray anchor's delta-blend pull
                                      # when the physics baseline is already
                                      # farther than this from its ray (see
                                      # _delta_evidence) -- prevents an
                                      # isolated bad/mislabelled anchor from
                                      # yanking a whole smoothing-kernel
                                      # window several metres off course
    # iteration 3.1: local C1 fix at non-event knots (_smooth_non_event_
    # knot_windows), replacing the whole-chain-soft-evidence redesign that
    # cost too much accuracy. k = hermite_k_ref_frames * hermite_k_ref_
    # speed_m_s / local_speed, clamped to [hermite_k_min, hermite_k_max]
    # -- "~3-5 frames, scaled to speed" (slower motion needs a slightly
    # wider window to blend the same amount of curvature away).
    "hermite_k_min": 3,
    "hermite_k_max": 5,
    "hermite_k_ref_frames": 4,
    "hermite_k_ref_speed_m_s": 2.0,
    "blend_max_delta_step_frac": 0.5,  # cap on delta's own frame-to-frame
                                        # change, as a fraction of local
                                        # physics-track speed (C1 delta)
    "blend_max_delta_step_floor_m_s": 1.0,  # speed floor so a near-
                                             # stationary span still has
                                             # SOME correction headroom
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
    """Split ``anchors`` (+ ``fixes``) into hard 3-D span-boundary knots
    and soft ray evidence, restoring the iteration-2 PER-ANCHOR span
    density: every anchor that resolves to a genuine 3-D point (event or
    non-event state — ``ground_exact``/``joint_depth`` via
    ``ball_eval.anchor_gt_world``, goal geometry, or the catch/
    player_touch joint convention) becomes its own span boundary. Only
    states that genuinely can't be pinned to a 3-D point without a second
    constraint (airborne_*, off_screen_flight, or an EVENT anchor that
    failed 3-D resolution) become ray evidence.

    Iteration 3 first tried making NON-EVENT anchors (module docstring
    history) permanently soft, one continuous fit per event-free chain —
    it removed the per-anchor velocity kink, but chains between real
    events can span 50+ frames of a genuinely curved real dribble/roll, and
    a single rigid physical model badly underfit that (gberch/base %<=20cm
    0.91->0.36, ground-truth-vs-track floating up to 1m+). Reverted here:
    every resolved anchor is a knot again (iteration-2 accuracy), and the
    velocity kink at a NON-EVENT knot specifically is fixed LOCALLY
    instead — see ``_smooth_non_event_knot_windows`` — over a small
    (~3-5 frame) window around it, not by giving up per-anchor precision
    everywhere. ``_is_sharp_knot`` (state in ``EVENT_STATES``, ``"fix"``,
    or a data-discovered internal split) marks which knots keep their
    raw, physically-real velocity break.
    """
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
            # Couldn't resolve to hard 3-D (airborne_*/off_screen_flight
            # by nature, or an EVENT anchor missing the extra context it
            # needed -- e.g. a goal_impact ray that missed all goal
            # geometry, or a player_touch/catch with no player_id/bone):
            # lateral-exact ray constraint, depth free.
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


def _is_sharp_knot(knot: "_Knot") -> bool:
    """True when a real velocity break belongs at this knot: an EVENT
    anchor (touch/kick/bounce/header/volley/chest/catch/goal_impact), a
    cross-replay fix, or a data-discovered internal split (a genuine
    hidden bounce/direction-change the residual gate found, not click
    noise). Everything else (a NON-EVENT anchor — almost always a
    "grounded" click) is exact-position for FITTING but gets its velocity
    smoothed locally rather than left kinked — see
    ``_smooth_non_event_knot_windows``.
    """
    return (knot.state in EVENT_STATES or knot.state == "fix"
            or knot.kind == "internal")


_AMBIGUOUS_GROUND_STATES = frozenset({"player_touch", "catch"})


def _quick_roll_worst_px(ctx, a_knot, b_knot, evid, z_level, duration_s, cfg) -> float:
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
    worst = 0.0
    for frame, uv, _w in evid:
        t_s = (frame - a_knot.frame) / ctx.fps
        p = roll.eval([t_s], z_level)[0]
        worst = max(worst, _reproj_px(ctx, frame, p, uv))
    return worst


def _quick_flight_worst_px(ctx, a_knot, b_knot, evid, duration_s, cfg) -> float:
    v0 = shoot_arc(a_knot.xyz, 0.0, b_knot.xyz, duration_s, cd=cfg["cd"])
    worst = 0.0
    for frame, uv, _w in evid:
        t_s = (frame - a_knot.frame) / ctx.fps
        p = simulate(a_knot.xyz, v0, [t_s], cd=cfg["cd"])[0]
        worst = max(worst, _reproj_px(ctx, frame, p, uv))
    return worst


def _choose_model(ctx: Any, a_knot: _Knot, b_knot: _Knot,
                   observations: Sequence[Observation],
                   ray_anchors: Sequence[_RayAnchor],
                   cfg: Mapping[str, Any]) -> str:
    """Roll needs both endpoints at ground level; ANY ``airborne_*`` ray
    inside the span forces flight instead.

    NOTE: iteration 3.0 (briefly) required >=2 airborne rays here for its
    long event-only CHAINS; reverted to the iteration-1/2 "any" rule now
    that spans are per-anchor again (iteration 3.1).

    ``player_touch``/``catch`` are genuinely ambiguous even by resolved
    HEIGHT: contact happens at ground/hand level whether the touch is a
    grounded dribble or a launching chip/lob — the knot's own z can't
    tell those apart, only what happens in BETWEEN the knots can. When
    both endpoints are one of these ambiguous states, both ground-level,
    AND there's no airborne ray to settle it, this runs a cheap one-shot
    roll fit and one-shot flight fit and picks whichever already explains
    the span's real evidence better (lower worst-point reprojection),
    instead of guessing. (A pure ground/bounce/kick pair skips this —
    those states are unambiguous, and the check isn't free.) Discovered
    via a real regression: defaulting player_touch to ground-by-height
    fixed gberch's mostly-grounded dribbles but broke s013, whose
    player_touch-bracketed spans are mostly lofted passes.
    """
    span_rays = [r for r in ray_anchors if a_knot.frame < r.frame < b_knot.frame]
    if any(r.state.startswith("airborne") for r in span_rays):
        return "flight"
    if not (_is_ground_knot(a_knot) and _is_ground_knot(b_knot)):
        return "flight"
    if not (a_knot.state in _AMBIGUOUS_GROUND_STATES
            or b_knot.state in _AMBIGUOUS_GROUND_STATES):
        return "roll"

    duration_s = (b_knot.frame - a_knot.frame) / ctx.fps
    evid = _span_evidence(observations, ray_anchors, a_knot.frame, b_knot.frame,
                           cfg["anchor_fit_weight"])
    if not evid:
        return "roll"
    z_level = 0.5 * (float(a_knot.xyz[2]) + float(b_knot.xyz[2]))
    roll_worst = _quick_roll_worst_px(ctx, a_knot, b_knot, evid, z_level, duration_s, cfg)
    flight_worst = _quick_flight_worst_px(ctx, a_knot, b_knot, evid, duration_s, cfg)
    return "roll" if roll_worst <= flight_worst else "flight"


def _fit_roll_iterative(
    a_xy, b_xy, duration_s: float,
    ground_obs: list[tuple[float, np.ndarray, float]],
    cfg: Mapping[str, Any],
):
    """Endpoint-exact WEIGHTED roll fit with the same iterative
    fit->gate->refit robust-gating idea as the flight branch: drop ground
    observations whose residual from the current fit is a clear outlier
    (> 3x the fit's own median residual, floored at 0.5m so a tight,
    well-behaved fit isn't destabilised by refitting on near-nothing),
    refit, repeat up to ``robust_gate_max_iters`` times. ``ground_obs``
    entries are ``(t_s, xy, weight)`` -- weight lets a non-event manual
    anchor click dominate real detector observations in the SAME chain
    fit (see ``anchor_fit_weight``) without ever being pinned exactly."""
    active = list(ground_obs)
    roll = fit_roll_segment(a_xy, b_xy, duration_s, active,
                             mu_max=cfg["roll_mu_max"])
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

    model = _choose_model(ctx, a_knot, b_knot, observations, ray_anchors, cfg)

    if model == "roll":
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

        # ground_obs mixes real detector observations (weight = their own
        # confidence) with non-event manual anchor clicks (weight =
        # anchor_fit_weight, tuned much higher -- see DEFAULT_CFG -- so
        # the WHOLE chain stays one smooth roll while still landing close
        # to every click, instead of pinning each one exactly).
        ground_obs: list[tuple[float, np.ndarray, float]] = []
        for o in observations:
            if not (a_knot.frame < o.frame < b_knot.frame):
                continue
            P = _ground_project(o.frame, o.uv)
            if P is not None:
                ground_obs.append(((o.frame - a_knot.frame) / ctx.fps, P[:2], float(o.conf)))
        for r in ray_anchors:
            if not (a_knot.frame < r.frame < b_knot.frame):
                continue
            P = _ground_project(r.frame, r.uv)
            if P is not None:
                ground_obs.append(((r.frame - a_knot.frame) / ctx.fps, P[:2],
                                    float(cfg["anchor_fit_weight"])))

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
                                    b_knot.frame, cfg["anchor_fit_weight"])
        worst_roll: tuple[int, tuple[float, float], float] | None = None
        for frame, uv, _w in roll_evid:
            err = _reproj_px(ctx, frame, pts[frame], uv)
            if worst_roll is None or err > worst_roll[2]:
                worst_roll = (frame, uv, err)

        info = {"span": (a_knot.frame, b_knot.frame), "model": "roll",
                "n_obs": len(ground_obs),
                "max_residual_px": worst_roll[2] if worst_roll else None}

        # Split-and-retry, same mechanism as flight: a REAL football chain
        # between two rare events (e.g. a long uninterrupted dribble) can
        # have genuine direction changes a single straight-line+friction
        # roll can't capture. This is data-driven (only fires when the
        # residual demands it, bounded by max_splits_per_span) and thus
        # fundamentally different from the pre-iteration-3 bug: it never
        # fires just because a click exists, only when the single-model
        # fit demonstrably fails -- discovered on real evidence (gberch/
        # kroupi01/origi01 synthetic scenarios all regressed ~3-10x on
        # plain "hybrid" without this, because a chain with dozens of
        # non-event anchors along a genuinely curved real path was being
        # forced through one rigid quadratic).
        if (worst_roll is not None
                and worst_roll[2] > cfg["inlier_px"] * cfg["split_residual_factor"]
                and splits_used < cfg["max_splits_per_span"]):
            split_frame, split_uv, _err = worst_roll
            split_xy = _ground_project(split_frame, split_uv)
            if split_xy is not None:
                split_xyz = np.array([split_xy[0], split_xy[1], z_level])
                split_knot = _Knot(frame=split_frame, xyz=split_xyz,
                                    state="waypoint", kind="internal", is_manual=False)
                internal_knot_frames.append(split_frame)
                pts1, info1 = _solve_span(ctx, a_knot, split_knot, observations,
                                           ray_anchors, cfg, splits_used + 1,
                                           internal_knot_frames)
                pts2, info2 = _solve_span(ctx, split_knot, b_knot, observations,
                                           ray_anchors, cfg, splits_used + 1,
                                           internal_knot_frames)
                return {**pts1, **pts2}, [*info1, *info2]

        return pts, [info]

    # --- flight (iterative robust gating: fit -> gate outliers by
    # reprojection residual -> refit, up to robust_gate_max_iters times or
    # until the inlier set stabilises) --------------------------------
    evid_all = _span_evidence(observations, ray_anchors, a_knot.frame, b_knot.frame,
                               cfg["anchor_fit_weight"])
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
                                        cfg["anchor_fit_weight"])
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
                                        cfg["anchor_fit_weight"])
            fitted = _fit_open_end(ctx, hard_knots[-1], tail_evid, 1, cfg)
            for f in range(last_frame + 1, tail_end + 1):
                pts[f] = fitted.get(f, hard_knots[-1].xyz)
            diagnostics.append({
                "span": (last_frame, tail_end),
                "model": "open_tail" if fitted else "hold_tail",
                "n_obs": len(tail_evid),
            })

    return pts, diagnostics, internal_knot_frames


def _smooth_non_event_knot_windows(
    ctx: Any,
    pts: Mapping[int, np.ndarray],
    hard_knots: Sequence[_Knot],
    cfg: Mapping[str, Any],
) -> dict[int, np.ndarray]:
    """Local C1 fix for every NON-EVENT knot's velocity kink (see
    ``resolve_knots``'s docstring for why this replaced iteration 3's
    event-only chains): over a small window (``+/-k`` frames, k scaled
    down for faster motion) centred on the knot, replace the raw
    per-span-fit path with a two-piece cubic Hermite blend. Each piece
    runs from the window EDGE — using that edge's own position and
    velocity from the untouched per-span fit, so it joins smoothly with
    everything outside the window — in to the knot's own exact position,
    with a SHARED tangent AT the knot (the average of the incoming and
    outgoing velocities there). The knot's own position is therefore
    unchanged (still the exact click, or exactly the internal-split
    position); only the velocity DIRECTION either side of it is turned
    smoothly instead of kinking.
    """
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
        if _is_sharp_knot(knot):
            continue
        f0 = knot.frame
        if f0 not in idx_of:
            continue
        i0 = idx_of[f0]
        if i0 == 0 or i0 == len(frames_sorted) - 1:
            continue  # clip edge -- nothing on one side to blend with

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
        perp, along = point_ray_distance(p_phys, r.C, r.d_hat)
        # Unlike an observation (gated above by px reprojection error), a
        # ray anchor's faithful point sits ON its ray BY CONSTRUCTION, so
        # its own reprojection is always ~0px regardless of how far the
        # physics baseline actually is from that ray -- pixel error can't
        # gate it. Gate on the baseline-to-ray PERPENDICULAR distance
        # (metres) instead: when the physics fit is already far from an
        # anchor's ray (a genuinely mislabelled click, or a chain the fit
        # can't fully honour), forcing the blend to jump onto that ray
        # anyway created a large, spatially-isolated delta spike that
        # leaked into several neighbouring frames via the smoothing
        # kernel (observed: a single inconsistent anchor costing ~4m of
        # error across a ~15-frame window). Such an anchor is left
        # unpulled here -- it's already visible in
        # diagnostics["anchor_not_honoured"] instead of silently
        # distorting nearby frames.
        if perp > cfg["anchor_delta_max_m"]:
            continue
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
        # Only auto EVENTS are candidate knots at all -- an auto
        # "grounded"/"airborne_*" entry is a synthetic interpolation of
        # evidence already folded into the chain fit as soft evidence, so
        # it carries no new information and is ignored outright rather
        # than even being considered.
        auto_events = [a for a in auto_anchors if _anchor_attrs(a)[2] in EVENT_STATES]
        auto_hard, auto_rays = resolve_knots(ctx, auto_events, fixes=())
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
                        "auto_anchors": auto_diag, "anchor_residuals": [],
                        "anchor_not_honoured": []}

    # Local C1 fix at every NON-EVENT knot (see resolve_knots' docstring):
    # smooths the velocity kink over a small window without moving the
    # knot's own position or touching the (physically real) breaks at
    # EVENT/fix/internal-split knots.
    pts = _smooth_non_event_knot_windows(ctx, pts, hard_knots, full_cfg)

    evidence = _delta_evidence(ctx, pts, obs_sorted, ray_anchors, full_cfg)
    sharp_frames = [k.frame for k in hard_knots if _is_sharp_knot(k)]
    event_frames = sharp_frames + internal_knot_frames
    frames_sorted = sorted(pts)
    blended = blend_deltas(frames_sorted, evidence,
                            halflife_frames=full_cfg["blend_halflife_frames"],
                            event_frames=event_frames)

    # delta must be C1 too: cap its own frame-to-frame change to a
    # fraction of the physics track's local speed (centred finite
    # difference), independent of blend_deltas' kernel smoothness. A
    # smooth kernel alone can still respond quickly when evidence is
    # dense; this is the second, direct guard.
    max_step_by_frame: dict[int, float] = {}
    frac = full_cfg["blend_max_delta_step_frac"]
    speed_floor = full_cfg["blend_max_delta_step_floor_m_s"]
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
    ray_by_frame = {r.frame: r for r in ray_anchors}

    out_frames = []
    mode_counts = {"anchor": 0, "faithful": 0, "simulated": 0}
    for f in frames_sorted:
        base = np.asarray(pts[f], dtype=float)
        delta, conf = blended.get(f, ((0.0, 0.0, 0.0), 0.0))
        final = base + np.asarray(delta, dtype=float)
        mode = "faithful" if conf >= 0.5 else "simulated"
        out_conf: float | None = conf

        if f in knot_by_frame and _is_sharp_knot(knot_by_frame[f]):
            # EVENT/fix/internal-split knot: exact re-snap, as always --
            # a real velocity break is expected here.
            final = np.array(knot_by_frame[f].xyz, dtype=float)
            mode, out_conf = "anchor", 1.0
        elif f in knot_by_frame:
            # NON-EVENT knot: position is already exact (or within
            # click-noise tolerance) via _smooth_non_event_knot_windows'
            # Hermite pass-through, so no re-snap is needed here; mode is
            # still reported as "anchor". Residual is recorded below
            # (diagnostics["anchor_residuals"]) instead of forcing it to
            # exactly zero, which is this iteration's documented,
            # narrowly-scoped relaxation of "operator input always wins".
            mode = "anchor"
        elif f in ray_by_frame:
            mode = "anchor"

        final[2] = max(float(final[2]), BALL_RADIUS_M)
        mode_counts[mode] = mode_counts.get(mode, 0) + 1
        out_frames.append(TrackFrame(
            frame=f, xyz=(float(final[0]), float(final[1]), float(final[2])),
            mode=mode, conf=out_conf))

    by_frame_final = {tf.frame: tf.xyz for tf in out_frames if tf.xyz is not None}
    not_honoured_px = full_cfg["anchor_not_honoured_px"]
    anchor_residuals = []
    for r in ray_anchors:
        xyz = by_frame_final.get(r.frame)
        if xyz is None:
            continue
        res_px = _reproj_px(ctx, r.frame, np.asarray(xyz), r.uv)
        anchor_residuals.append({"frame": r.frame, "state": r.state,
                                  "residual_px": res_px})
    for k in hard_knots:
        if _is_sharp_knot(k):
            continue  # trivially ~0px by exact re-snap; not informative
        xyz = by_frame_final.get(k.frame)
        if xyz is None:
            continue
        uv = ctx.project(k.frame, k.xyz)
        res_px = _reproj_px(ctx, k.frame, np.asarray(xyz), (float(uv[0]), float(uv[1])))
        anchor_residuals.append({"frame": k.frame, "state": k.state,
                                  "residual_px": res_px})
    anchor_not_honoured = [a["frame"] for a in anchor_residuals
                            if a["residual_px"] > not_honoured_px]

    track = Track(clip_id=ctx.clip_id, method="hybrid", frames=tuple(out_frames))
    diagnostics = {
        "spans": span_diag,
        "n_knots": len(hard_knots),
        "n_ray_anchors": len(ray_anchors),
        "n_internal_bounces": len(internal_knot_frames),
        "mode_counts": mode_counts,
        "auto_anchors": auto_diag,
        "anchor_residuals": anchor_residuals,
        "anchor_not_honoured": anchor_not_honoured,
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
    anchor_res = diagnostics.get("anchor_residuals", [])
    if anchor_res:
        arr = np.array([a["residual_px"] for a in anchor_res])
        not_honoured = diagnostics.get("anchor_not_honoured", [])
        print(f"non-event anchor residual px: median={np.median(arr):.2f}  "
              f"p95={np.percentile(arr, 95):.2f}  (n={len(arr)}, "
              f"not_honoured>4px={len(not_honoured)})")
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
