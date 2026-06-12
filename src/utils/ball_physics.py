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
