"""Physical primitives for the piecewise ball solver.

Pure functions: ballistic two-knot arcs (gravity fully determines an arc
from two timed positions), endpoint-exact rolling fits with a friction
bound, and bounce restitution. No IO, no camera math.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

G = -9.81
G_VEC = np.array([0.0, 0.0, G])


def two_knot_arc(
    a: np.ndarray, b: np.ndarray, duration_s: float
) -> tuple[np.ndarray, np.ndarray]:
    """The unique gravity arc through ``a`` at t=0 and ``b`` at t=T.

    Returns ``(p0, v0)`` with ``p(t) = p0 + v0 t + ½ g t²``. Position
    continuity at both ends is exact by construction.
    """
    if duration_s <= 0:
        raise ValueError("two_knot_arc needs a positive duration")
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    v0 = (b - a - 0.5 * G_VEC * duration_s**2) / duration_s
    return a, v0


def eval_parabola(
    p0: np.ndarray, v0: np.ndarray, times_s: np.ndarray
) -> np.ndarray:
    """Positions of the gravity arc at ``times_s``; shape (N, 3)."""
    ts = np.asarray(times_s, dtype=float).reshape(-1, 1)
    return (
        np.asarray(p0, dtype=float)
        + np.asarray(v0, dtype=float) * ts
        + 0.5 * G_VEC * ts**2
    )


def parabola_end_velocity(v0: np.ndarray, duration_s: float) -> np.ndarray:
    return np.asarray(v0, dtype=float) + G_VEC * duration_s


@dataclass(frozen=True)
class RollingFit:
    """Endpoint-exact ground roll ``xy(t) = line(t) + c2·(t² − t·T)``.

    The quadratic term vanishes at t=0 and t=T (both endpoints exact)
    and ``|2·c2|`` is the constant acceleration in m/s², bounded by the
    rolling-friction cap. ``z`` is the ball radius throughout.
    """

    a_xy: tuple[float, float]
    b_xy: tuple[float, float]
    duration_s: float
    c2: tuple[float, float]

    def eval(self, times_s: np.ndarray, z: float) -> np.ndarray:
        ts = np.asarray(times_s, dtype=float).reshape(-1, 1)
        a = np.asarray(self.a_xy, dtype=float)
        b = np.asarray(self.b_xy, dtype=float)
        c2 = np.asarray(self.c2, dtype=float)
        frac = ts / self.duration_s
        xy = a + (b - a) * frac + c2 * (ts**2 - ts * self.duration_s)
        out = np.empty((len(ts), 3))
        out[:, :2] = xy
        out[:, 2] = z
        return out


def fit_rolling_segment(
    a_xy: np.ndarray,
    b_xy: np.ndarray,
    duration_s: float,
    obs: list[tuple[float, np.ndarray]],
    decel_max_m_s2: float,
) -> RollingFit:
    """Least-squares constant-acceleration roll through both endpoints.

    ``obs`` are ``(time_s, xy)`` interior ground observations. With no
    observations the roll degenerates to constant velocity (c2 = 0).
    The fitted acceleration ``|2·c2|`` is clamped to the friction cap so
    a handful of bad detections cannot bend the roll unphysically.
    """
    if duration_s <= 0:
        raise ValueError("fit_rolling_segment needs a positive duration")
    a = np.asarray(a_xy, dtype=float)
    b = np.asarray(b_xy, dtype=float)
    c2 = np.zeros(2)
    if obs:
        num = np.zeros(2)
        den = 0.0
        for t_s, xy in obs:
            phi = t_s**2 - t_s * duration_s
            line = a + (b - a) * (t_s / duration_s)
            num += phi * (np.asarray(xy, dtype=float) - line)
            den += phi * phi
        if den > 1e-12:
            c2 = num / den
    accel = 2.0 * float(np.linalg.norm(c2))
    if accel > decel_max_m_s2 > 0:
        c2 = c2 * (decel_max_m_s2 / accel)
    return RollingFit(
        a_xy=(float(a[0]), float(a[1])),
        b_xy=(float(b[0]), float(b[1])),
        duration_s=float(duration_s),
        c2=(float(c2[0]), float(c2[1])),
    )


def restitution(v_in: np.ndarray, v_out: np.ndarray) -> float | None:
    """Bounce restitution ``e = −v_out_z / v_in_z``.

    ``None`` when the inbound vertical speed is too small for the ratio
    to be meaningful (ball arriving flat / rolling into the contact).
    """
    viz = float(np.asarray(v_in, dtype=float)[2])
    voz = float(np.asarray(v_out, dtype=float)[2])
    if viz > -0.1:
        return None
    return -voz / viz


# Moment-of-inertia coefficient for a thin spherical shell (a football is
# close to a hollow sphere): I = α·m·R² with α = 2/3.  This sets the
# rolling-condition split of the friction impulse between the centre-of-
# mass tangential velocity and the spin.  (A solid sphere would use 2/5.)
_INERTIA_ALPHA = 2.0 / 3.0


def bounce(
    v_in: np.ndarray,
    omega_in: np.ndarray,
    e: float,
    mu: float,
    ball_radius: float = 0.11,
) -> tuple[np.ndarray, np.ndarray]:
    """Spin-aware rigid-sphere bounce on the horizontal ground plane.

    Standard oblique-impact impulse model for a sphere striking a fixed
    horizontal surface (Cross, "Measurements of the horizontal coefficient
    of restitution for a superball and a tennis ball", Am. J. Phys. 70
    (2002) 482; see also Cross, "Grip-slip behavior of a bouncing ball",
    AJP 70 (2002) 1093).  The ground normal is world-z (up).

    Convention:
      * The contact point sits at ``r_c = (0, 0, -R)`` below the centre.
      * Contact-point tangential velocity (the quantity friction acts on)
        is ``v_c = v_in_xy + omega_in × r_c``.  With ``r_c`` pointing down,
        ``omega_in × r_c`` adds the surface speed of the spinning ball, so
        backspin (ω about +x for travel along +x) *adds* forward surface
        speed at the contact and the friction impulse then *reverses* the
        ball's horizontal velocity, etc.
      * Normal: ``v_out_z = -e · v_in_z`` (no penetration; e the normal COR).
      * Tangential: a friction impulse opposes the contact-point slip.
        It is capped by Coulomb friction ``μ·J_n`` (slipping/skidding
        bounce) or, when friction is strong enough, just enough to drive
        the contact point to zero tangential velocity (rolling/gripping
        bounce — the slip cannot reverse).

    Args:
        v_in: inbound centre-of-mass velocity (3,), m/s, world frame.
        omega_in: inbound angular velocity (3,), rad/s, world frame.
        e: normal coefficient of restitution (>= 0).
        mu: Coulomb friction coefficient (>= 0).
        ball_radius: ball radius in metres.

    Returns:
        ``(v_out, omega_out)`` — outbound CoM velocity and angular
        velocity, both world-frame (3,).
    """
    v = np.asarray(v_in, dtype=float).reshape(3)
    w = np.asarray(omega_in, dtype=float).reshape(3)
    R = float(ball_radius)
    e = float(e)
    mu = float(mu)

    n = np.array([0.0, 0.0, 1.0])  # ground normal (up)
    r_c = np.array([0.0, 0.0, -R])  # centre -> contact point

    vz_in = v[2]
    # Ball not actually descending into the plane: no impulse (passthrough).
    if vz_in >= 0.0:
        return v.copy(), w.copy()

    # Normal impulse magnitude per unit mass: Δv_z = -(1+e)·v_z, so
    # J_n / m = (1 + e) * |v_z|.
    jn_over_m = (1.0 + e) * (-vz_in)

    # Contact-point velocity and its tangential (horizontal) component.
    v_c = v + np.cross(w, r_c)
    v_c_t = v_c.copy()
    v_c_t[2] = 0.0
    slip_speed = float(np.linalg.norm(v_c_t))

    v_out = v.copy()
    v_out[2] = -e * vz_in  # normal restitution

    if slip_speed < 1e-9:
        # Pure normal impact (no tangential slip, e.g. a dead drop):
        # tangential velocity and spin are unchanged.
        return v_out, w.copy()

    t_hat = v_c_t / slip_speed  # slip direction (friction opposes this)

    # Tangential impulse that would bring the contact point to rest
    # (rolling/gripping bounce).  For a sphere the contact-point tangential
    # velocity changes by (1 + 1/α)·Δv_cm_t under a tangential impulse, so
    # the impulse-per-mass needed to null the slip is
    #     J_t/m = slip_speed / (1 + 1/α).
    grip_jt_over_m = slip_speed / (1.0 + 1.0 / _INERTIA_ALPHA)
    # Coulomb cap (slipping bounce).
    coulomb_jt_over_m = mu * jn_over_m
    jt_over_m = min(grip_jt_over_m, coulomb_jt_over_m)

    # Friction impulse opposes the slip direction.
    dvt = -jt_over_m * t_hat
    v_out[0] += dvt[0]
    v_out[1] += dvt[1]

    # Spin change from the same tangential impulse: the friction force acts
    # at the contact point, producing a torque τ = r_c × F.  Per unit mass
    # the angular-velocity change is
    #     Δω = (r_c × J_t) / (α·R²).
    j_t = np.array([dvt[0], dvt[1], 0.0])
    dw = np.cross(r_c, j_t) / (_INERTIA_ALPHA * R * R)
    omega_out = w + dw

    return v_out, omega_out
