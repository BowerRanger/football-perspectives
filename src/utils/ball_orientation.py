"""Per-frame ball orientation integration.

Integrates a running unit quaternion over the dense ball track so the
exported ball *visibly rotates / curls*.  The angular velocity per frame
comes from physics, not from any measured ball spin observation:

- **flight** frames use the fitted Magnus spin of their ``FlightSegment``
  (``spin_omega_rad_s * spin_axis_world``);
- **grounded / rolling** frames use a rolling-consistent angular velocity
  ``omega = (|v| / r)`` about ``up x v_hat`` (the contact point does not
  slip — see ``integrate_orientation`` and the slip test);
- **stationary, missing, or world-less** frames contribute no rotation
  (``omega = 0``) and the quaternion is carried unchanged.

The quaternion evolves *continuously*: ``omega`` jumps at segment
boundaries, but ``q`` is never reset at a node.

Quaternion convention: ``(w, x, y, z)`` (scalar-first), Hamilton product,
right-handed.  Pure numpy, deterministic, no IO.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from src.schemas.ball_track import BallFrame, FlightSegment

Quat = tuple[float, float, float, float]

IDENTITY_QUAT: Quat = (1.0, 0.0, 0.0, 0.0)
_UP = np.array([0.0, 0.0, 1.0])
_EPS = 1e-12


# ---------------------------------------------------------------------------
# Quaternion helpers (wxyz convention, Hamilton product).
# ---------------------------------------------------------------------------
def quat_normalize(q: Quat) -> Quat:
    """Return the unit quaternion; identity if ``q`` is degenerate."""
    arr = np.asarray(q, dtype=float)
    norm = float(np.linalg.norm(arr))
    if norm < _EPS:
        return IDENTITY_QUAT
    arr = arr / norm
    return (float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3]))


def quat_mul(a: Quat, b: Quat) -> Quat:
    """Hamilton product ``a * b`` (apply ``b`` first, then ``a``)."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def quat_from_axis_angle(axis: Sequence[float] | np.ndarray, angle: float) -> Quat:
    """Unit quaternion for a rotation of ``angle`` rad about ``axis``.

    ``axis`` need not be normalised; a zero axis or zero angle yields the
    identity quaternion.
    """
    ax = np.asarray(axis, dtype=float)
    n = float(np.linalg.norm(ax))
    if n < _EPS or abs(angle) < _EPS:
        return IDENTITY_QUAT
    ax = ax / n
    half = 0.5 * float(angle)
    s = float(np.sin(half))
    return (float(np.cos(half)), ax[0] * s, ax[1] * s, ax[2] * s)


# ---------------------------------------------------------------------------
# Angular-velocity model.
# ---------------------------------------------------------------------------
def _flight_omega(segment: FlightSegment | None) -> np.ndarray:
    """Magnus spin vector (rad/s) for a flight segment; zero if unknown."""
    if segment is None:
        return np.zeros(3)
    para = segment.parabola
    axis = para.get("spin_axis_world")
    omega_mag = para.get("spin_omega_rad_s")
    if axis is None or omega_mag is None:
        return np.zeros(3)
    axis_arr = np.asarray(axis, dtype=float)
    n = float(np.linalg.norm(axis_arr))
    if n < _EPS or abs(float(omega_mag)) < _EPS:
        return np.zeros(3)
    return (axis_arr / n) * float(omega_mag)


def _rolling_omega(velocity: np.ndarray, radius_m: float) -> np.ndarray:
    """Rolling-consistent spin: ``omega = (|v|/r)`` about ``up x v_hat``.

    Chosen so the contact point (offset ``(0, 0, -r)`` from the centre)
    has zero slip: ``v + omega x r_contact = 0``.
    """
    v = np.asarray(velocity, dtype=float)
    speed = float(np.linalg.norm(v))
    if speed < _EPS or radius_m <= 0:
        return np.zeros(3)
    v_hat = v / speed
    axis = np.cross(_UP, v_hat)
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm < _EPS:
        # Velocity is vertical (no horizontal travel): no rolling spin.
        return np.zeros(3)
    axis = axis / axis_norm
    return axis * (speed / radius_m)


def _world(frame: BallFrame) -> np.ndarray | None:
    if frame.world_xyz is None:
        return None
    return np.asarray(frame.world_xyz, dtype=float)


def _frame_omega(
    idx: int,
    frames: list[BallFrame],
    seg_by_id: dict[int, FlightSegment],
    radius_m: float,
    dt: float,
) -> np.ndarray:
    """Angular velocity (rad/s) used to advance ``q`` from frame ``idx``."""
    frame = frames[idx]
    here = _world(frame)
    if here is None:
        return np.zeros(3)

    if frame.state == "flight" and frame.flight_segment_id is not None:
        return _flight_omega(seg_by_id.get(frame.flight_segment_id))

    if frame.state == "grounded":
        velocity = _finite_velocity(idx, frames, dt)
        return _rolling_omega(velocity, radius_m)

    # occluded / missing / any state without a usable model -> no rotation.
    return np.zeros(3)


def _finite_velocity(
    idx: int, frames: list[BallFrame], dt: float
) -> np.ndarray:
    """World velocity at frame ``idx`` via finite difference of positions.

    Prefers a forward difference to the next world-bearing frame so the
    rotation advanced *from* this frame matches the displacement *to* the
    next; falls back to a backward difference at the tail.
    """
    here = _world(frames[idx])
    if here is None:
        return np.zeros(3)
    # Forward difference to the next world-bearing frame.
    for j in range(idx + 1, len(frames)):
        nxt = _world(frames[j])
        if nxt is not None:
            span = (j - idx) * dt
            if span > _EPS:
                return (nxt - here) / span
            break
    # Backward difference to the previous world-bearing frame.
    for j in range(idx - 1, -1, -1):
        prev = _world(frames[j])
        if prev is not None:
            span = (idx - j) * dt
            if span > _EPS:
                return (here - prev) / span
            break
    return np.zeros(3)


# ---------------------------------------------------------------------------
# Integration.
# ---------------------------------------------------------------------------
def integrate_orientation(
    frames: Sequence[BallFrame],
    flight_segments: Sequence[FlightSegment],
    fps: float,
    ball_radius_m: float,
) -> dict[int, Quat | None]:
    """Integrate a per-frame unit quaternion over the dense ball track.

    Walks ``frames`` in frame order, maintaining a running unit quaternion
    ``q`` (starting at identity).  For each frame the current ``q`` is
    stored, then ``q`` is advanced by ``omega * dt`` about ``omega``'s axis
    ready for the next frame.  Frames with ``world_xyz is None`` store
    ``None`` (mirroring ``world_xyz``) and carry ``q`` unchanged.

    ``q`` is never reset at a segment boundary, so orientation stays
    continuous even though ``omega`` jumps between segments.

    Returns ``{frame_number: (w, x, y, z) | None}``.
    """
    if fps <= 0:
        raise ValueError("integrate_orientation needs a positive fps")
    dt = 1.0 / float(fps)

    ordered = sorted(frames, key=lambda f: f.frame)
    seg_by_id = {s.id: s for s in flight_segments}

    out: dict[int, Quat | None] = {}
    q: Quat = IDENTITY_QUAT
    for idx, frame in enumerate(ordered):
        if frame.world_xyz is None:
            # Mirror world_xyz: no orientation, carry q for the next frame.
            out[frame.frame] = None
            continue
        out[frame.frame] = q
        omega = _frame_omega(idx, ordered, seg_by_id, ball_radius_m, dt)
        omega_mag = float(np.linalg.norm(omega))
        if omega_mag >= _EPS:
            step = quat_from_axis_angle(omega, omega_mag * dt)
            q = quat_normalize(quat_mul(step, q))
    return out
