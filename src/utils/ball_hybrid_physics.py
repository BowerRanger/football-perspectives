"""Pure physics primitives for the production ball-hybrid trajectory layer.

Ported verbatim from ``prototypes/ball_hybrid_poc/hybrid_physics.py`` (see
``prototypes/ball_hybrid_poc/CONTRACT.md``) — no behavioural changes, so
outputs must match the PoC module bit-for-bit (``tests/test_ball_hybrid_
physics.py`` cross-checks both).

No camera math, no IO. Everything here is testable on plain numpy arrays.

Model: a size-11 football (r=0.11 m, m=0.43 kg) falling under gravity with
quadratic aerodynamic drag (F_drag = -0.5*rho*Cd*A*|v|*v) and an optional
bounded Magnus (lift) term (F_magnus/m = k_magnus * (omega x v)) integrated
with fixed-step RK4. ``shoot_arc`` solves the two-point boundary-value
problem — "what launch velocity v0 sends the ball from p_a at t=0 to p_b
at t=T under this drag/Magnus model" — which is what lets a drag-aware
arc still land exactly on both hard 3-D knots (endpoint-exactness is
never traded away for physical realism).

Constants are the same order of magnitude as
``src/utils/ball_piecewise_solver.py``'s ``drag_k_over_m`` (0.005): its
docstring notes that coefficient is actually a Magnus term (the existing
solver has no aerodynamic drag at all) — this module adds a genuine drag
term and keeps a comparable Magnus coefficient for continuity.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.optimize import least_squares

# --- Physical constants (FIFA size-5 ball) ---------------------------------

RHO_AIR = 1.2           # kg/m^3, sea-level air density
BALL_RADIUS_M = 0.11    # m
BALL_MASS_KG = 0.43     # kg
BALL_AREA_M2 = math.pi * BALL_RADIUS_M ** 2

G = -9.81
G_VEC = np.array([0.0, 0.0, G])

CD_DEFAULT = 0.25
CD_BOUNDS = (0.15, 0.40)

# a = -0.5*rho*Cd*A/m * |v| * v ; this is the Cd-independent prefactor.
_DRAG_PREFACTOR = 0.5 * RHO_AIR * BALL_AREA_M2 / BALL_MASS_KG

# a_magnus = k_magnus * (omega x v), with k_magnus = 0.5*rho*A*r*Cl/m and
# a unit lift coefficient Cl=1.0 (soccer-ball Magnus lift coefficients are
# commonly cited in the 0.1-1.0 range depending on spin ratio; Cl=1.0 is a
# deliberately generous default since ``magnus`` is only ever enabled when
# it measurably reduces residual — see cfg gating in hybrid.py).
DEFAULT_MAGNUS_COEFF = 0.5 * RHO_AIR * BALL_AREA_M2 * BALL_RADIUS_M / BALL_MASS_KG
MAGNUS_MAX_OMEGA_REV_S = 10.0

Vec3 = np.ndarray


def _accel(v: np.ndarray, cd: float, omega: np.ndarray | None,
           magnus_coeff: float) -> np.ndarray:
    a = G_VEC.copy()
    if cd:
        speed = float(np.linalg.norm(v))
        a = a - (_DRAG_PREFACTOR * cd * speed) * v
    if omega is not None:
        a = a + magnus_coeff * np.cross(omega, v)
    return a


def simulate(
    p0: Vec3,
    v0: Vec3,
    times_s: Sequence[float],
    cd: float = CD_DEFAULT,
    omega: Vec3 | None = None,
    magnus_coeff: float = DEFAULT_MAGNUS_COEFF,
    dt_max: float = 1.0 / 60.0,
) -> np.ndarray:
    """Positions at ``times_s`` (need not be sorted, MAY be negative) of
    the ball at ``p0``/``v0`` at t=0, under gravity + quadratic drag
    (``cd``) + optional Magnus (``omega``, world-frame angular velocity
    rad/s). Fixed-step RK4 integration, linearly interpolated onto the
    requested sample times. Shape ``(N, 3)``.

    A negative time integrates *backward* from ``p0``/``v0`` (used by the
    hybrid extractor's free-end head-span fit, where a knot's position is
    known but frames before it need extrapolating). RK4 is time-symmetric
    — stepping with a negative ``h`` numerically solves the same ODE
    backward — so the forward and backward branches share one stepper;
    they're just integrated as two separate one-directional passes from
    t=0 (positive times forward, negative times' magnitudes backward)
    since a single monotonically-increasing substep grid can't cover
    both directions at once.
    """
    p0 = np.asarray(p0, dtype=float)
    v0 = np.asarray(v0, dtype=float)
    times_s = np.atleast_1d(np.asarray(times_s, dtype=float))
    out = np.empty((len(times_s), 3))
    fwd = times_s >= 0
    bwd = ~fwd
    if fwd.any():
        out[fwd] = _integrate_direction(p0, v0, times_s[fwd], cd, omega,
                                         magnus_coeff, dt_max, direction=1.0)
    if bwd.any():
        out[bwd] = _integrate_direction(p0, v0, -times_s[bwd], cd, omega,
                                         magnus_coeff, dt_max, direction=-1.0)
    return out


def _integrate_direction(
    p0: np.ndarray, v0: np.ndarray, abs_times_s: np.ndarray,
    cd: float, omega: Vec3 | None, magnus_coeff: float, dt_max: float,
    direction: float,
) -> np.ndarray:
    """RK4 from t=0 to ``direction * max(abs_times_s)``, sampled (via
    linear interpolation) at ``direction * abs_times_s``. ``abs_times_s``
    are all >= 0; ``direction`` is +1.0 (forward) or -1.0 (backward)."""
    t_end = float(np.max(abs_times_s))
    if t_end <= 0:
        return np.tile(p0, (len(abs_times_s), 1))

    n = max(1, int(math.ceil(t_end / dt_max)))
    h = direction * (t_end / n)
    ts = np.linspace(0.0, t_end, n + 1)  # magnitudes, for interpolation
    ps = np.empty((n + 1, 3))
    ps[0] = p0
    p, v = p0.copy(), v0.copy()
    for i in range(n):
        k1v = _accel(v, cd, omega, magnus_coeff)
        k1p = v
        k2v = _accel(v + 0.5 * h * k1v, cd, omega, magnus_coeff)
        k2p = v + 0.5 * h * k1v
        k3v = _accel(v + 0.5 * h * k2v, cd, omega, magnus_coeff)
        k3p = v + 0.5 * h * k2v
        k4v = _accel(v + h * k3v, cd, omega, magnus_coeff)
        k4p = v + h * k3v
        v = v + (h / 6.0) * (k1v + 2 * k2v + 2 * k3v + k4v)
        p = p + (h / 6.0) * (k1p + 2 * k2p + 2 * k3p + k4p)
        ps[i + 1] = p

    out = np.empty((len(abs_times_s), 3))
    for ax in range(3):
        out[:, ax] = np.interp(abs_times_s, ts, ps[:, ax])
    return out


def shoot_arc(
    p_a: Vec3,
    t_a: float,
    p_b: Vec3,
    t_b: float,
    cd: float = CD_DEFAULT,
    omega: Vec3 | None = None,
    magnus_coeff: float = DEFAULT_MAGNUS_COEFF,
    v0_guess: Vec3 | None = None,
) -> np.ndarray:
    """Solve the boundary-value problem: launch velocity ``v0`` at
    ``p_a``/``t_a`` that arrives exactly at ``p_b`` at ``t_b`` under this
    drag/Magnus model. ``cd=0`` and ``omega=None`` is the closed-form
    two-knot gravity arc (exact, no optimisation); otherwise a
    Levenberg-Marquardt shoot seeded from that analytic solution.
    """
    p_a = np.asarray(p_a, dtype=float)
    p_b = np.asarray(p_b, dtype=float)
    duration_s = float(t_b - t_a)
    if duration_s <= 0:
        raise ValueError("shoot_arc needs t_b > t_a")

    analytic_v0 = (p_b - p_a - 0.5 * G_VEC * duration_s ** 2) / duration_s
    if v0_guess is None:
        v0_guess = analytic_v0

    if not cd and omega is None:
        return analytic_v0

    def residual(v0: np.ndarray) -> np.ndarray:
        p_end = simulate(p_a, v0, [duration_s], cd=cd, omega=omega,
                          magnus_coeff=magnus_coeff)[0]
        return p_end - p_b

    sol = least_squares(residual, v0_guess, method="lm",
                         xtol=1e-10, ftol=1e-10, max_nfev=200)
    return sol.x


# ---------------------------------------------------------------------------
# Roll model: endpoint-exact constant-deceleration ground roll.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RollFit:
    """Endpoint-exact ground roll: ``x(t) = a + v0*t + 0.5*accel*t^2``
    with ``v0`` solved so ``x(0)=a`` and ``x(T)=b`` exactly; ``accel`` is
    a constant (friction) deceleration fit to interior observations and
    clamped to a Coulomb-friction bound."""

    a_xy: tuple[float, float]
    b_xy: tuple[float, float]
    duration_s: float
    accel_xy: tuple[float, float]

    def eval(self, times_s: Sequence[float], z: float) -> np.ndarray:
        ts = np.atleast_1d(np.asarray(times_s, dtype=float))
        a = np.asarray(self.a_xy, dtype=float)
        b = np.asarray(self.b_xy, dtype=float)
        acc = np.asarray(self.accel_xy, dtype=float)
        T = self.duration_s
        v0 = (b - a - 0.5 * acc * T ** 2) / T
        xy = (a[None, :] + v0[None, :] * ts[:, None]
              + 0.5 * acc[None, :] * (ts ** 2)[:, None])
        out = np.empty((len(ts), 3))
        out[:, :2] = xy
        out[:, 2] = z
        return out


def fit_roll_segment(
    a_xy: Sequence[float],
    b_xy: Sequence[float],
    duration_s: float,
    obs: Sequence[tuple[float, Sequence[float]] | tuple[float, Sequence[float], float]] = (),
    mu_max: float = 0.9,
    g: float = 9.81,
) -> RollFit:
    """Weighted least-squares constant-deceleration roll through both
    endpoints.

    ``obs`` entries are ``(time_s, xy)`` or ``(time_s, xy, weight)``
    interior ground observations (world xy); an omitted weight defaults
    to 1.0. Manual anchor clicks are typically weighted far above real
    detector observations here (see ``hybrid.py``'s ``anchor_fit_weight``)
    so the WHOLE event-free chain stays one smooth path while still
    landing close to every click, instead of being pinned exactly (which
    is what used to create a velocity kink at every anchor). With no
    observations the roll degenerates to constant velocity (accel=0).
    ``|accel|`` is clamped to ``mu_max*g`` (Coulomb friction envelope on
    natural turf).
    """
    if duration_s <= 0:
        raise ValueError("fit_roll_segment needs a positive duration")
    a = np.asarray(a_xy, dtype=float)
    b = np.asarray(b_xy, dtype=float)
    accel = np.zeros(2)
    if obs:
        num = np.zeros(2)
        den = 0.0
        for item in obs:
            if len(item) == 3:
                t_s, xy, w = item
            else:
                t_s, xy = item
                w = 1.0
            phi = 0.5 * (t_s ** 2 - t_s * duration_s)
            line = a + (b - a) * (t_s / duration_s)
            num += w * phi * (np.asarray(xy, dtype=float) - line)
            den += w * phi * phi
        if den > 1e-12:
            accel = num / den
    amax = mu_max * g
    mag = float(np.linalg.norm(accel))
    if amax > 0 and mag > amax:
        accel = accel * (amax / mag)
    return RollFit(
        a_xy=(float(a[0]), float(a[1])),
        b_xy=(float(b[0]), float(b[1])),
        duration_s=float(duration_s),
        accel_xy=(float(accel[0]), float(accel[1])),
    )


# ---------------------------------------------------------------------------
# Bounce model
# ---------------------------------------------------------------------------

def bounce_velocity(
    v_in: Vec3,
    restitution_e: float,
    tangential_retention: float = 1.0,
) -> np.ndarray:
    """Outbound velocity of a ground bounce: vertical component flips and
    scales by ``restitution_e`` (``-v_out_z = e * v_in_z``); horizontal
    component is scaled by ``tangential_retention`` (friction/spin loss,
    1.0 = no tangential loss)."""
    v_in = np.asarray(v_in, dtype=float)
    v_out = v_in.copy()
    v_out[2] = -restitution_e * v_in[2]
    v_out[:2] = v_in[:2] * tangential_retention
    return v_out


# ---------------------------------------------------------------------------
# Cubic Hermite blend (local C1 join, used by the hybrid extractor to
# smooth the velocity transition at a non-event knot over a small window
# without moving the knot's own position)
# ---------------------------------------------------------------------------

def hermite_blend(
    p0: Vec3, m0: Vec3, p1: Vec3, m1: Vec3, frac,
) -> np.ndarray:
    """Standard two-point cubic Hermite interpolation: passes exactly
    through ``p0`` at ``frac=0`` (derivative ``m0``) and ``p1`` at
    ``frac=1`` (derivative ``m1``); ``frac`` may be a scalar or array in
    ``[0, 1]``. ``m0``/``m1`` must already be scaled by the interval's own
    duration (i.e. ``dp/dfrac``, not ``dp/dt`` — a caller integrating in
    seconds over an interval of duration ``T`` passes ``velocity * T``).
    Shape ``(len(frac), 3)``.
    """
    f = np.atleast_1d(np.asarray(frac, dtype=float))
    f2 = f * f
    f3 = f2 * f
    h00 = 2 * f3 - 3 * f2 + 1
    h10 = f3 - 2 * f2 + f
    h01 = -2 * f3 + 3 * f2
    h11 = f3 - f2
    p0 = np.asarray(p0, dtype=float)
    m0 = np.asarray(m0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    m1 = np.asarray(m1, dtype=float)
    return (h00[:, None] * p0 + h10[:, None] * m0
            + h01[:, None] * p1 + h11[:, None] * m1)
