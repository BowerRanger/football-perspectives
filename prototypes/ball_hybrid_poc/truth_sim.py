"""Independent ball physics for the hybrid-extraction PoC's synthetic truth.

This is a SEPARATE physics implementation from the pipeline's own ball
solver family. It must not import anything from ``src.utils.ball_physics``,
``src.utils.ball_piecewise_solver``, any other ``src.utils.ball_*`` physics
module, or any ``prototypes.ball_hybrid_poc.hybrid*`` module — a test greps
this file's imports to enforce that. The point of the PoC is to grade the
pipeline's ball extraction against ground truth manufactured by a model
that does NOT share code (and, in the ``mismatch``/``sparse`` scenarios,
deliberately does not share parameters) with the thing under test.

Only ``numpy`` and ``scipy`` are used. Ball constants (FIFA size-5 ball):
mass ``BALL_MASS_KG`` = 0.43 kg, radius ``BALL_RADIUS_M`` = 0.11 m.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

G = 9.81
RHO_AIR = 1.2
BALL_RADIUS_M = 0.11
BALL_MASS_KG = 0.43
BALL_AREA_M2 = float(np.pi * BALL_RADIUS_M ** 2)


# --------------------------------------------------------------------------
# Aerodynamic drag (quadratic, optional "drag crisis" Cd(v) transition)
# --------------------------------------------------------------------------


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


# --------------------------------------------------------------------------
# Magnus lift (bounded, spin-parameter-scaled) + spin decay
# --------------------------------------------------------------------------


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


# --------------------------------------------------------------------------
# Flight ODE (gravity + drag + Magnus), RK45 fine-step integration
# --------------------------------------------------------------------------


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
    seconds at <= ``max_step`` (default 2 ms), per CONTRACT.md's "RK4 at
    <=2 ms" allowance (faster than an adaptive RK45 for the repeated
    shooting-solver calls in ``truth_builder.py`` without meaningfully
    changing the arc for a smooth gravity+drag+Magnus force field).

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


# --------------------------------------------------------------------------
# Bounce map: normal restitution + tangential friction with spin coupling
# --------------------------------------------------------------------------


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
    # Surface velocity of the contact point due to spin: omega x (-r z_hat).
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


# --------------------------------------------------------------------------
# Rolling friction (steady rolling resistance + extra skid immediately
# after touchdown), with a closed-form solver for the launch speed that
# covers a given distance in a given time (used to hit an exact next knot
# without needing an ODE shoot for ground segments).
# --------------------------------------------------------------------------


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


def simulate_roll_path(pos0_xy: np.ndarray, direction_xy: np.ndarray,
                       v0: float, duration_s: float, p: RollParams,
                       ts: np.ndarray) -> np.ndarray:
    """Positions (N,2) along a straight roll at the given sample times."""
    ts = np.asarray(ts, dtype=float)
    out = np.zeros((len(ts), 2))
    for i, t in enumerate(ts):
        d = roll_distance_at(v0, float(t), p)
        out[i] = np.asarray(pos0_xy, float) + d * np.asarray(direction_xy, float)
    return out


__all__ = [
    "G", "RHO_AIR", "BALL_RADIUS_M", "BALL_MASS_KG", "BALL_AREA_M2",
    "DragParams", "drag_coefficient", "drag_accel",
    "SpinParams", "magnus_accel",
    "FlightResult", "simulate_flight",
    "BounceParams", "apply_bounce",
    "RollParams", "roll_distance_at", "solve_roll_initial_speed",
    "simulate_roll_path",
]
