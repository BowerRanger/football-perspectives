"""Synthetic 3-D ground truth for the ball-stage regression bench.

Promoted from the ``ball-hybrid-integration`` spike
(``prototypes/ball_hybrid_poc/truth_sim.py`` + ``truth_builder.py``, merged
into one module). Two sections:

1. **Physics** (``simulate_flight``, ``apply_bounce``, roll-distance
   helpers, ...) — a SEPARATE, self-contained implementation from the
   pipeline's own ball solver family. It uses only ``numpy``/``scipy`` and
   must NEVER import ``src.utils.ball_physics``, ``src.utils.
   ball_piecewise_solver``, ``src.stages.ball``, any other ``ball_*``
   *physics/solver* module, or any ``ball_hybrid_*`` module —
   ``tests/test_ball_bench_truth.py`` greps this file's imports to enforce
   it. The point of the bench is to grade the pipeline's ball extraction
   against ground truth manufactured by a model that does NOT share code
   (and, in the ``mismatch``/``sparse`` scenarios, deliberately does not
   share parameters) with the thing under test.

2. **Builder** (``build_truth``) — resolves a clip's manual anchors into
   hard 3-D knots (grounded/kick/bounce via click-ray ∩ ground plane,
   player_touch via the contacting joint projected onto the click ray,
   goal_impact via goal geometry) or ray-only waypoints, then solves the
   physics segment between each pair of hard knots so the simulated arc
   passes exactly through both. This section is allowed to (and does)
   import ``src.utils.ball_eval`` (grading/GT primitives, not a solver)
   and ``src.utils.goal_geometry`` — same as the spike's ``truth_builder
   .py``.

Ball constants (FIFA size-5 ball): mass ``BALL_MASS_KG`` = 0.43 kg, radius
``BALL_RADIUS_M`` = 0.11 m.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq, least_squares

# ---------------------------------------------------------------------------
# Section 1: independent physics (numpy/scipy only — see module docstring)
# ---------------------------------------------------------------------------

G = 9.81
RHO_AIR = 1.2
BALL_RADIUS_M = 0.11
BALL_MASS_KG = 0.43
BALL_AREA_M2 = float(np.pi * BALL_RADIUS_M ** 2)


@dataclass(frozen=True)
class DragParams:
    """Quadratic-drag model. When ``crisis`` is False, ``cd_const`` is used
    at every speed (the ``base`` scenario). When True, Cd smoothly
    transitions from ``cd_low`` (below ``v_low``) to ``cd_high`` (above
    ``v_high``) — the real "drag crisis" soccer balls exhibit around
    10-20 m/s, deliberately NOT how the pipeline's own solver models drag.
    """

    cd_const: float = 0.25
    crisis: bool = False
    cd_low: float = 0.45
    cd_high: float = 0.20
    v_low: float = 10.0
    v_high: float = 20.0


def drag_coefficient(speed: float, p: DragParams) -> float:
    if not p.crisis:
        return p.cd_const
    if speed <= p.v_low:
        return p.cd_low
    if speed >= p.v_high:
        return p.cd_high
    s = (speed - p.v_low) / (p.v_high - p.v_low)
    s = s * s * (3.0 - 2.0 * s)  # smoothstep
    return p.cd_low + (p.cd_high - p.cd_low) * s


def drag_accel(v: np.ndarray, p: DragParams) -> np.ndarray:
    """``F = 1/2 rho Cd A |v| v`` (opposing velocity), returned as ``F/m``."""
    speed = float(np.linalg.norm(v))
    if speed < 1e-9:
        return np.zeros(3)
    cd = drag_coefficient(speed, p)
    f_over_m = 0.5 * RHO_AIR * cd * BALL_AREA_M2 * speed / BALL_MASS_KG
    return -f_over_m * v


@dataclass(frozen=True)
class SpinParams:
    """Spin state and its aerodynamic coupling. ``omega0`` (rad/s, world
    frame) decays exponentially with time constant ``decay_s``. Lift
    coefficient ``CL`` is modelled as ``clamp(cl_coeff * spin_parameter,
    0, cl_max)`` where ``spin_parameter = r*|omega|/|v|`` — CL ~ 0.2-0.3
    for spin parameters ~0.1-0.3, matching measured soccer-ball data.
    """

    omega0: np.ndarray
    cl_coeff: float = 1.0
    cl_max: float = 0.33
    decay_s: float = 4.0

    def omega_at(self, t: float) -> np.ndarray:
        if self.decay_s <= 0:
            return np.asarray(self.omega0, dtype=float)
        return np.asarray(self.omega0, dtype=float) * float(np.exp(-t / self.decay_s))


def magnus_accel(v: np.ndarray, omega: np.ndarray) -> np.ndarray:
    """``F = 1/2 rho A CL |v|^2 n_hat``, ``n_hat`` along ``omega x v``."""
    speed = float(np.linalg.norm(v))
    om = float(np.linalg.norm(omega))
    if speed < 1e-6 or om < 1e-6:
        return np.zeros(3)
    spin_parameter = BALL_RADIUS_M * om / speed
    cl = min(1.0 * spin_parameter, 0.33)
    cross = np.cross(omega, v)
    cn = float(np.linalg.norm(cross))
    if cn < 1e-9:
        return np.zeros(3)
    n_hat = cross / cn
    f_over_m = 0.5 * RHO_AIR * BALL_AREA_M2 * cl * speed ** 2 / BALL_MASS_KG
    return f_over_m * n_hat


def _flight_deriv(t: float, state: np.ndarray, spin: SpinParams,
                   drag: DragParams) -> np.ndarray:
    v = state[3:6]
    omega = spin.omega_at(t)
    a = np.array([0.0, 0.0, -G]) + drag_accel(v, drag) + magnus_accel(v, omega)
    return np.concatenate([v, a])


@dataclass(frozen=True)
class FlightResult:
    """Fixed-step RK4 trajectory sample table. ``state_at`` linearly
    interpolates between the (<=2ms-spaced) samples, which is accurate to
    well under a millimetre given the sub-2ms step."""

    ts: np.ndarray        # (N,)
    pos: np.ndarray        # (N, 3)
    vel: np.ndarray        # (N, 3)
    t_end: float
    pos_end: np.ndarray
    vel_end: np.ndarray

    def state_at(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        t = float(np.clip(t, 0.0, self.t_end))
        pos = np.array([np.interp(t, self.ts, self.pos[:, i]) for i in range(3)])
        vel = np.array([np.interp(t, self.ts, self.vel[:, i]) for i in range(3)])
        return pos, vel


def simulate_flight(pos0: np.ndarray, vel0: np.ndarray, duration_s: float,
                     drag: DragParams, spin: SpinParams,
                     max_step: float = 0.002) -> FlightResult:
    """Fixed-step RK4-integrate a free-flight arc of ``duration_s``
    seconds at <= ``max_step`` (default 2 ms) — faster than an adaptive
    RK45 for the repeated shooting-solver calls in :func:`build_truth`
    without meaningfully changing the arc for a smooth
    gravity+drag+Magnus force field.

    No ground event here — callers that need a bounce mid-arc should
    integrate up to the bounce time (found separately) and then continue
    with a fresh call seeded by :func:`apply_bounce`'s output.
    """
    duration_s = float(duration_s)
    n = max(2, int(np.ceil(duration_s / max_step)) + 1)
    ts_arr = np.linspace(0.0, duration_s, n)
    dt = ts_arr[1] - ts_arr[0] if n > 1 else duration_s
    pos = np.zeros((n, 3))
    vel = np.zeros((n, 3))
    pos[0] = np.asarray(pos0, dtype=float)
    vel[0] = np.asarray(vel0, dtype=float)
    for i in range(n - 1):
        t = ts_arr[i]
        y = np.concatenate([pos[i], vel[i]])
        k1 = _flight_deriv(t, y, spin, drag)
        k2 = _flight_deriv(t + dt / 2, y + dt / 2 * k1, spin, drag)
        k3 = _flight_deriv(t + dt / 2, y + dt / 2 * k2, spin, drag)
        k4 = _flight_deriv(t + dt, y + dt * k3, spin, drag)
        y_next = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        pos[i + 1] = y_next[:3]
        vel[i + 1] = y_next[3:6]
    return FlightResult(ts=ts_arr, pos=pos, vel=vel, t_end=duration_s,
                        pos_end=pos[-1], vel_end=vel[-1])


@dataclass(frozen=True)
class BounceParams:
    e_n: float = 0.70          # normal restitution (grass: 0.6-0.75)
    mu_t: float = 0.40         # tangential friction retention loss coeff
    spin_retention: float = 0.60
    spin_transfer: float = 0.30


def apply_bounce(vel: np.ndarray, omega: np.ndarray,
                 p: BounceParams) -> tuple[np.ndarray, np.ndarray]:
    """Post-bounce ``(velocity, omega)`` given pre-bounce values.

    Normal component reflects with restitution ``e_n`` (energy loss).
    Tangential component loses speed to friction proportional to the
    normal impulse, coupled with contact-point slip from spin (topspin
    -> forward kick, backspin -> check/stop, sidespin -> curl-out).
    """
    vel = np.asarray(vel, dtype=float)
    omega = np.asarray(omega, dtype=float)
    vn = float(vel[2])
    vt = vel.copy()
    vt[2] = 0.0
    vn_new = -p.e_n * vn
    spin_tan = BALL_RADIUS_M * np.array([-omega[1], omega[0], 0.0])
    slip = vt - spin_tan
    slip_speed = float(np.linalg.norm(slip))
    impulse_cap = p.mu_t * (1.0 + p.e_n) * abs(vn)
    if slip_speed > 1e-6:
        reduction = min(impulse_cap, slip_speed)
        vt_new = vt - reduction * (slip / slip_speed)
    else:
        vt_new = vt
    delta_vt = vt - vt_new
    omega_new = omega * p.spin_retention + p.spin_transfer * np.array(
        [delta_vt[1], -delta_vt[0], 0.0]) / BALL_RADIUS_M
    vel_new = np.array([vt_new[0], vt_new[1], vn_new])
    return vel_new, omega_new


@dataclass(frozen=True)
class RollParams:
    friction_decel: float = 0.6     # steady rolling resistance, m/s^2
    skid_extra_decel: float = 1.4   # extra decel during initial skid
    skid_duration_s: float = 0.25


def roll_distance_at(v0: float, duration_s: float, p: RollParams) -> float:
    """1-D distance covered in ``duration_s`` starting at speed ``v0``,
    under the piecewise skid+steady deceleration model. Ball never goes
    backwards (speed clamped to 0 and held)."""
    v0 = max(0.0, float(v0))
    t = 0.0
    v = v0
    dist = 0.0
    t_skid = min(p.skid_duration_s, duration_s)
    a1 = p.friction_decel + p.skid_extra_decel
    t_stop_skid = v / a1 if a1 > 0 else float("inf")
    if t_stop_skid <= t_skid:
        dist += v * v / (2.0 * a1) if a1 > 0 else 0.0
        return dist  # stopped during skid phase, before duration_s elapses
    dist += v * t_skid - 0.5 * a1 * t_skid ** 2
    v -= a1 * t_skid
    t += t_skid
    remaining = duration_s - t
    if remaining <= 0:
        return dist
    a2 = p.friction_decel
    t_stop = v / a2 if a2 > 0 else float("inf")
    if t_stop <= remaining:
        dist += v * v / (2.0 * a2) if a2 > 0 else 0.0
        return dist
    dist += v * remaining - 0.5 * a2 * remaining ** 2
    return dist


def solve_roll_initial_speed(target_dist: float, duration_s: float,
                             p: RollParams) -> float:
    """Initial speed whose ``roll_distance_at`` over ``duration_s`` equals
    ``target_dist`` (monotonic in v0, so a bracketed root-find suffices).
    Returns 0.0 for a non-positive/negligible target distance."""
    target_dist = float(target_dist)
    if target_dist <= 1e-6 or duration_s <= 0:
        return 0.0
    hi = 5.0
    while roll_distance_at(hi, duration_s, p) < target_dist and hi < 60.0:
        hi *= 1.5
    if roll_distance_at(hi, duration_s, p) < target_dist:
        return hi  # can't reach it even at the cap; caller flags infeasible
    return float(brentq(lambda v: roll_distance_at(v, duration_s, p)
                        - target_dist, 0.0, hi, xtol=1e-5))


# ---------------------------------------------------------------------------
# Section 2: builder (allowed to use src.utils.ball_eval / goal_geometry —
# grading/GT primitives, not solvers)
# ---------------------------------------------------------------------------

from src.utils.ball_eval import (  # noqa: E402
    anchor_gt_world, pixel_ray, point_ray_distance, ray_plane_z,
)
from src.utils.goal_geometry import (  # noqa: E402
    GoalGeometry, resolve_goal_impact_world,
)

from .ball_bench_clip import ClipContext  # noqa: E402
from .ball_bench_types import TruthEvent, TruthFrame, TruthTrack  # noqa: E402

# States that get no player/bone association in the real anchor files even
# though the schema permits one (only ``player_touch`` requires it). These
# always demote to ray-only waypoints.
_NO_DEPTH_STATES = frozenset({"catch", "header", "volley", "chest"})
_CONTACT_STATES = frozenset({"player_touch", "kick", "bounce", "goal_impact"})

_ROLL_MAX_SPEED_MPS = 14.0
_CARRY_MAX_SPEED_MPS = 6.0
_CARRY_MAX_FRAMES = 20
_FLIGHT_INFEASIBLE_ERR_M = 0.5
_FLIGHT_INFEASIBLE_SPEED_MPS = 40.0


@dataclass(frozen=True)
class _Resolved:
    frame: int
    xyz: np.ndarray | None   # None => waypoint only
    kind: str
    C: np.ndarray | None
    d: np.ndarray | None
    anchor: object


def _resolve_anchor(ctx: ClipContext, anchor) -> _Resolved:
    if anchor.image_xy is None:
        return _Resolved(anchor.frame, None, "none", None, None, anchor)
    frame = anchor.frame
    if frame not in ctx.per_frame_K:
        return _Resolved(frame, None, "no_camera", None, None, anchor)
    K = ctx.per_frame_K[frame]
    R = ctx.per_frame_R[frame]
    t = ctx.per_frame_t[frame]

    if anchor.state == "goal_impact":
        geom = GoalGeometry.from_pitch_config({})
        try:
            xyz = resolve_goal_impact_world(
                anchor.image_xy, anchor.goal_element, K=K, R=R, t=t,
                distortion=ctx.distortion, geometry=geom)
            return _Resolved(frame, np.asarray(xyz, float), "goal_geometry",
                             None, None, anchor)
        except ValueError:
            C, d = pixel_ray(anchor.image_xy, K, R, t, ctx.distortion)
            xyz = ray_plane_z(C, d, geom.crossbar_z)
            kind = "goal_fallback_plane" if xyz is not None else "ray_only"
            return _Resolved(frame, xyz, kind, C, d, anchor)

    joint = None
    if anchor.state == "player_touch":
        joint = ctx.player_context().joint_world(frame, anchor.player_id,
                                                 anchor.bone)
    xyz, kind = anchor_gt_world(anchor, K, R, t, ctx.distortion,
                                ball_radius=BALL_RADIUS_M,
                                joint_world=joint)
    C, d = pixel_ray(anchor.image_xy, K, R, t, ctx.distortion)
    return _Resolved(frame, (np.asarray(xyz, float) if xyz is not None
                             else None), kind, C, d, anchor)


def _roll_like_segment(xa, xb, fa, fb, dt_s, fps, scenario, rng, seg_kind):
    if scenario == "base":
        p = RollParams(friction_decel=0.6, skid_extra_decel=1.4)
    else:
        p = RollParams(
            friction_decel=float(rng.uniform(0.4, 0.8)),
            skid_extra_decel=float(rng.uniform(1.0, 2.0)),
        )
    dist = float(np.linalg.norm(xb - xa))
    direction = (xb - xa) / dist if dist > 1e-9 else np.zeros(3)
    v0 = solve_roll_initial_speed(dist, dt_s, p)
    positions: dict[int, np.ndarray] = {}
    for f in range(fa, fb + 1):
        trel = (f - fa) / fps
        d_along = roll_distance_at(v0, trel, p)
        positions[f] = xa + d_along * direction
    positions[fa] = xa
    positions[fb] = xb
    info = {
        "type": seg_kind, "frame_a": fa, "frame_b": fb, "dt_s": dt_s,
        "dist_m": dist, "v0_m_s": v0,
        "friction_decel": p.friction_decel,
        "skid_extra_decel": p.skid_extra_decel,
    }
    return positions, info


def _flight_segment(ctx: ClipContext, xa, xb, fa, fb, dt_s, wp_between,
                    scenario, rng):
    fps = ctx.fps
    fit_spin = scenario != "base"
    if scenario == "base":
        cd = DragParams(cd_const=0.25, crisis=False)
        e_n = 0.70
        spin_mag = float(rng.uniform(0.0, 5.0)) * 2 * np.pi  # <=5 rev/s
    else:
        cd = DragParams(
            crisis=True,
            cd_low=float(rng.uniform(0.35, 0.50)),
            cd_high=float(rng.uniform(0.15, 0.25)),
            v_low=float(rng.uniform(8.0, 12.0)),
            v_high=float(rng.uniform(18.0, 22.0)),
        )
        e_n = float(rng.uniform(0.60, 0.75))
        spin_mag = float(rng.uniform(0.0, 10.0)) * 2 * np.pi  # <=10 rev/s
    axis = rng.normal(size=3)
    axis_norm = np.linalg.norm(axis)
    axis = axis / axis_norm if axis_norm > 1e-9 else np.array([0.0, 1.0, 0.0])
    omega0 = axis * spin_mag
    decay_s = float(rng.uniform(3.0, 6.0)) if scenario != "base" else 4.0

    disp = xb - xa
    vz0_guess = (disp[2] + 0.5 * G * dt_s ** 2) / dt_s
    vxy0_guess = disp[:2] / dt_s
    x0 = np.array([vxy0_guess[0], vxy0_guess[1], vz0_guess])

    def integrate(v0, omega):
        spin = SpinParams(omega0=omega, decay_s=decay_s)
        return simulate_flight(xa, v0, dt_s, cd, spin)

    def residuals(x):
        v0 = x[:3]
        omega = x[3:6] if fit_spin else omega0
        res = integrate(v0, omega)
        end_pos, _ = res.state_at(dt_s)
        r = list((end_pos - xb) * 25.0)
        for f, C, d, _wa in wp_between:
            trel = (f - fa) / fps
            p_i, _ = res.state_at(trel)
            lateral, _ = point_ray_distance(p_i, C, d)
            r.append(lateral * 4.0)
        return np.asarray(r, dtype=float)

    if fit_spin:
        x0f = np.concatenate([x0, omega0])
        bounds = ([-45, -45, -45, -80, -80, -80],
                 [45, 45, 45, 80, 80, 80])
    else:
        x0f = x0
        bounds = ([-45, -45, -45], [45, 45, 45])

    sol = least_squares(residuals, x0f, bounds=bounds, max_nfev=200)
    v0_fit = sol.x[:3]
    omega_fit = sol.x[3:6] if fit_spin else omega0
    res = integrate(v0_fit, omega_fit)
    end_pos, _end_vel = res.state_at(dt_s)
    end_err = float(np.linalg.norm(end_pos - xb))
    infeasible = (end_err > _FLIGHT_INFEASIBLE_ERR_M
                 or float(np.linalg.norm(v0_fit)) > _FLIGHT_INFEASIBLE_SPEED_MPS)

    wp_err_m = None
    if wp_between:
        errs = []
        for f, C, d, _wa in wp_between:
            trel = (f - fa) / fps
            p_i, _ = res.state_at(trel)
            lateral, _ = point_ray_distance(p_i, C, d)
            errs.append(lateral)
        wp_err_m = float(np.max(errs))

    if infeasible:
        positions, info = _roll_like_segment(xa, xb, fa, fb, dt_s, fps,
                                             scenario, rng, "flight_fallback_roll")
        info["notes"] = (
            f"flight shoot infeasible (end_err={end_err:.3f}m, "
            f"|v0|={float(np.linalg.norm(v0_fit)):.1f}m/s); fell back to a "
            "straight-line roll profile between the two hard knots")
        return positions, info

    positions = {}
    for f in range(fa, fb + 1):
        trel = (f - fa) / fps
        p_i, _ = res.state_at(trel)
        positions[f] = p_i
    positions[fa] = xa
    positions[fb] = xb
    info = {
        "type": "flight", "frame_a": fa, "frame_b": fb, "dt_s": dt_s,
        "cd": (cd.cd_const if not cd.crisis else
              {"crisis": True, "cd_low": cd.cd_low, "cd_high": cd.cd_high,
               "v_low": cd.v_low, "v_high": cd.v_high}),
        "bounce_e_n_annotation": e_n,
        "omega0_rad_s": omega_fit.tolist(),
        "spin_decay_s": decay_s,
        "v0_m_s": v0_fit.tolist(),
        "end_err_m": end_err,
        "waypoint_max_err_m": wp_err_m,
    }
    return positions, info


def _segment_kind(anchor_a, anchor_b, xa, xb, dt_s, has_waypoints_between):
    ground_states = {"grounded", "kick", "bounce"}
    dist = float(np.linalg.norm(xb - xa))
    speed = dist / dt_s if dt_s > 0 else float("inf")
    if (not has_waypoints_between and anchor_a.state in ground_states
            and anchor_b.state in ground_states
            and speed <= _ROLL_MAX_SPEED_MPS):
        return "roll"
    if (not has_waypoints_between and anchor_a.state == "player_touch"
            and anchor_b.state == "player_touch"
            and anchor_a.player_id == anchor_b.player_id
            and speed <= _CARRY_MAX_SPEED_MPS
            and (anchor_b.frame - anchor_a.frame) <= _CARRY_MAX_FRAMES):
        return "carry"
    return "flight"


def _event_for_hard_knot(resolved: _Resolved) -> TruthEvent | None:
    a = resolved.anchor
    xyz = tuple(float(v) for v in resolved.xyz)
    if a.state in ("player_touch", "kick"):
        return TruthEvent(frame=resolved.frame, kind="touch", xyz=xyz,
                          player_id=a.player_id, bone=a.bone)
    if a.state == "bounce":
        return TruthEvent(frame=resolved.frame, kind="bounce", xyz=xyz)
    if a.state == "goal_impact":
        kind = "net" if a.goal_element in ("back_net", "side_net") else "post"
        return TruthEvent(frame=resolved.frame, kind=kind, xyz=xyz)
    return None


def build_truth(ctx: ClipContext, scenario: str, seed: int = 0) -> TruthTrack:
    """Build synthetic 3-D truth for ``ctx``'s clip under ``scenario``
    (``"base" | "mismatch" | "sparse"`` — ``"hidden"`` is derived from
    ``"mismatch"`` by ``ball_bench_synth.derive_hidden_scenario`` and is
    not built here). ``sparse`` uses IDENTICAL physics to ``mismatch``
    (only the synthetic detector differs — see ``ball_bench_synth.py``);
    implemented by building with the ``mismatch`` parameter distributions
    under any ``seed``, exact by construction rather than by RNG
    coincidence.
    """
    if scenario == "sparse":
        mismatch = build_truth(ctx, "mismatch", seed=seed)
        physics = dict(mismatch.physics)
        physics["scenario_requested"] = "sparse"
        return dataclasses.replace(mismatch, scenario="sparse", physics=physics)
    phys_scenario = scenario
    if phys_scenario not in ("base", "mismatch"):
        raise ValueError(f"unknown scenario {scenario!r}")
    rng = np.random.default_rng(seed)

    anchors = sorted(ctx.anchors.anchors, key=lambda a: a.frame)
    resolved = [_resolve_anchor(ctx, a) for a in anchors]
    hard = [r for r in resolved if r.xyz is not None]
    waypoints = [r for r in resolved if r.xyz is None and r.C is not None]
    if len(hard) < 2:
        raise ValueError(
            f"{ctx.clip_id}: only {len(hard)} hard knot(s) resolved; "
            "need >= 2 to build a truth track")

    notes: list[str] = []
    for r in waypoints:
        if r.anchor.state in _NO_DEPTH_STATES:
            notes.append(
                f"frame {r.frame}: state={r.anchor.state!r} has no "
                "player/bone association in the real anchor file; treated "
                "as a ray-only waypoint, not a hard knot")
        elif r.anchor.state == "player_touch":
            notes.append(
                f"frame {r.frame}: player_touch joint lookup returned "
                "None (missing pose data); demoted to a ray-only waypoint")
    for r in resolved:
        if r.kind == "no_camera":
            notes.append(
                f"frame {r.frame}: no solved camera at this frame "
                "(outside camera_track's frame range); anchor dropped "
                "entirely (neither hard knot nor waypoint)")

    dense: dict[int, np.ndarray] = {}
    segments: list[dict] = []
    n_fallback = 0
    for ra, rb in zip(hard, hard[1:]):
        fa, fb = ra.frame, rb.frame
        if fb <= fa:
            continue
        dt_s = (fb - fa) / ctx.fps
        wp_between = [(w.frame, w.C, w.d, w.anchor) for w in waypoints
                     if fa < w.frame < fb]
        kind = _segment_kind(ra.anchor, rb.anchor, ra.xyz, rb.xyz, dt_s,
                             bool(wp_between))
        if kind in ("roll", "carry"):
            positions, info = _roll_like_segment(
                ra.xyz, rb.xyz, fa, fb, dt_s, ctx.fps, phys_scenario, rng,
                kind)
        else:
            positions, info = _flight_segment(
                ctx, ra.xyz, rb.xyz, fa, fb, dt_s, wp_between, phys_scenario,
                rng)
            if info["type"] == "flight_fallback_roll":
                n_fallback += 1
        segments.append(info)
        for f, xyz in positions.items():
            dense[f] = xyz

    # Safety clamp: no dense position may sit meaningfully below the pitch.
    n_clamped = 0
    for f in list(dense):
        z = dense[f][2]
        if z < BALL_RADIUS_M - 0.01:
            p = dense[f].copy()
            p[2] = BALL_RADIUS_M - 0.01
            dense[f] = p
            n_clamped += 1
    if n_clamped:
        notes.append(f"{n_clamped} frame(s) clamped to z >= r-1cm "
                     "(simulated arc dipped slightly below ground)")

    contact_frames = {r.frame for r in hard if r.anchor.state in _CONTACT_STATES}
    contact_frames |= {w.frame for w in waypoints
                       if w.anchor.state in _NO_DEPTH_STATES}

    frames_out = []
    for f in sorted(dense):
        xyz = dense[f]
        if f in contact_frames:
            state = "contact"
        elif xyz[2] <= BALL_RADIUS_M + 0.02:
            state = "ground"
        else:
            state = "air"
        frames_out.append(TruthFrame(
            frame=f, xyz=(float(xyz[0]), float(xyz[1]), float(xyz[2])),
            state=state))

    events = [e for r in hard if (e := _event_for_hard_knot(r)) is not None]
    if hard[-1].anchor.state == "grounded":
        events.append(TruthEvent(frame=hard[-1].frame, kind="rest",
                                 xyz=tuple(float(v) for v in hard[-1].xyz)))

    physics = {
        "ball_radius_m": BALL_RADIUS_M,
        "ball_mass_kg": BALL_MASS_KG,
        "g": G,
        "rho_air": RHO_AIR,
        "scenario_requested": scenario,
        "scenario_physics": phys_scenario,
        "seed": seed,
        "n_hard_knots": len(hard),
        "n_waypoints": len(waypoints),
        "n_segments": len(segments),
        "n_flight_fallback": n_fallback,
        "segments": segments,
        "notes": notes + [
            "bounce restitution is annotated per flight segment "
            "('bounce_e_n_annotation') for realism, but adjacent segments "
            "are shot independently to their own hard knots rather than "
            "propagated velocity-continuously through the bounce — a "
            "documented bench simplification."],
    }

    return TruthTrack(
        clip_id=ctx.clip_id, scenario=scenario, fps=ctx.fps,
        frames=tuple(frames_out), events=tuple(events),
        seed_anchor_frames=tuple(r.frame for r in hard),
        physics=physics,
    )


__all__ = [
    "G", "RHO_AIR", "BALL_RADIUS_M", "BALL_MASS_KG", "BALL_AREA_M2",
    "DragParams", "drag_coefficient", "drag_accel",
    "SpinParams", "magnus_accel",
    "FlightResult", "simulate_flight",
    "BounceParams", "apply_bounce",
    "RollParams", "roll_distance_at", "solve_roll_initial_speed",
    "build_truth",
]
