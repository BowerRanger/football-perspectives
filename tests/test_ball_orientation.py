"""Tests for per-frame ball orientation integration (unit quaternion)."""

import numpy as np

from src.schemas.ball_track import BallFrame, FlightSegment
from src.utils.ball_orientation import (
    integrate_orientation,
    quat_from_axis_angle,
    quat_mul,
    quat_normalize,
)

FPS = 25.0
DT = 1.0 / FPS
R = 0.11  # ball radius (m)


def _quat_to_rotmat(q: tuple[float, float, float, float]) -> np.ndarray:
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _quat_conj(q):
    w, x, y, z = q
    return (w, -x, -y, -z)


def _relative_rotation(qa, qb):
    """Quaternion taking qa to qb: qb = q_rel * qa  ->  q_rel = qb * qa^-1."""
    return quat_mul(qb, _quat_conj(qa))


def _angle_of(q) -> float:
    w = abs(max(-1.0, min(1.0, q[0])))
    return 2.0 * np.arccos(w)


# ---------------------------------------------------------------------------
# (a) Pure flight, known constant omega.
# ---------------------------------------------------------------------------
def test_pure_flight_constant_omega():
    omega_mag = 40.0  # rad/s
    axis = np.array([0.0, 1.0, 0.0])
    seg = FlightSegment(
        id=0,
        frame_range=(0, 4),
        parabola={
            "p0": [0.0, 0.0, 1.0],
            "v0": [10.0, 0.0, 5.0],
            "g": -9.81,
            "spin_axis_world": list(axis),
            "spin_omega_rad_s": omega_mag,
        },
        fit_residual_px=0.5,
    )
    frames = [
        BallFrame(
            frame=i,
            world_xyz=(float(i), 0.0, 1.0),
            state="flight",
            confidence=1.0,
            flight_segment_id=0,
        )
        for i in range(5)
    ]
    quats = integrate_orientation(frames, [seg], FPS, R)

    # q0 must be identity.
    assert np.allclose(quats[0], (1.0, 0.0, 0.0, 0.0), atol=1e-9)

    expected_step = quat_from_axis_angle(axis, omega_mag * DT)
    for i in range(4):
        rel = _relative_rotation(quats[i], quats[i + 1])
        # Relative rotation between consecutive frames == axis-angle(omega, dt).
        # Compare via rotation matrices (sign-agnostic).
        assert np.allclose(
            _quat_to_rotmat(rel), _quat_to_rotmat(expected_step), atol=1e-7
        )

    # Cumulative: final quat == step^4 applied from identity.
    cum = (1.0, 0.0, 0.0, 0.0)
    for _ in range(4):
        cum = quat_normalize(quat_mul(expected_step, cum))
    assert np.allclose(
        _quat_to_rotmat(quats[4]), _quat_to_rotmat(cum), atol=1e-7
    )


# ---------------------------------------------------------------------------
# (b) Rolling at speed v: implied omega = v/r, contact-point slip ~ 0.
# ---------------------------------------------------------------------------
def test_rolling_slip_zero():
    speed = 5.0  # m/s along +x
    frames = [
        BallFrame(
            frame=i,
            world_xyz=(speed * i * DT, 0.0, R),
            state="grounded",
            confidence=1.0,
        )
        for i in range(6)
    ]
    quats = integrate_orientation(frames, [], FPS, R)

    expected_omega_mag = speed / R
    # Implied per-frame rotation angle == omega * dt.
    for i in range(1, 5):
        rel = _relative_rotation(quats[i], quats[i + 1])
        assert abs(_angle_of(rel) - expected_omega_mag * DT) < 1e-6

    # Contact-point slip: slip = v_world + omega x r_contact, r_contact=(0,0,-r).
    # Rolling forward (+x) on the ground => omega about -y (so up x v_hat).
    v_hat = np.array([1.0, 0.0, 0.0])
    up = np.array([0.0, 0.0, 1.0])
    omega_axis = np.cross(up, v_hat)
    omega = expected_omega_mag * omega_axis
    r_contact = np.array([0.0, 0.0, -R])
    v_world = speed * v_hat
    slip = v_world + np.cross(omega, r_contact)
    assert np.linalg.norm(slip) < 1e-9


# ---------------------------------------------------------------------------
# (c) Stationary span: quaternion constant.
# ---------------------------------------------------------------------------
def test_stationary_constant():
    frames = [
        BallFrame(
            frame=i,
            world_xyz=(3.0, 2.0, R),
            state="grounded",
            confidence=1.0,
        )
        for i in range(5)
    ]
    quats = integrate_orientation(frames, [], FPS, R)
    for i in range(5):
        assert np.allclose(quats[i], (1.0, 0.0, 0.0, 0.0), atol=1e-12)


# ---------------------------------------------------------------------------
# (d) Continuity across a flight -> rolling boundary.
# ---------------------------------------------------------------------------
def test_continuity_flight_to_rolling():
    omega_mag = 50.0
    axis = np.array([0.0, 1.0, 0.0])
    seg = FlightSegment(
        id=0,
        frame_range=(0, 3),
        parabola={
            "p0": [0.0, 0.0, 1.0],
            "v0": [8.0, 0.0, 0.0],
            "g": -9.81,
            "spin_axis_world": list(axis),
            "spin_omega_rad_s": omega_mag,
        },
        fit_residual_px=0.5,
    )
    speed = 6.0
    flight_frames = [
        BallFrame(
            frame=i,
            world_xyz=(2.0 * i, 0.0, 1.0),
            state="flight",
            confidence=1.0,
            flight_segment_id=0,
        )
        for i in range(4)
    ]
    # Rolling continues from the last flight x-position.
    x0 = 6.0
    roll_frames = [
        BallFrame(
            frame=4 + i,
            world_xyz=(x0 + speed * (i + 1) * DT, 0.0, R),
            state="grounded",
            confidence=1.0,
        )
        for i in range(4)
    ]
    frames = flight_frames + roll_frames
    quats = integrate_orientation(frames, [seg], FPS, R)

    # No jump at the boundary: the q at the first rolling frame is the carried
    # q advanced by at most one step's worth of rotation.
    rolling_omega_mag = speed / R
    max_step = max(omega_mag, rolling_omega_mag) * DT
    frame_ids = sorted(quats.keys())
    for a, b in zip(frame_ids, frame_ids[1:]):
        rel = _relative_rotation(quats[a], quats[b])
        assert _angle_of(rel) <= max_step + 1e-6


# ---------------------------------------------------------------------------
# (e) All quaternions unit norm.
# ---------------------------------------------------------------------------
def test_all_unit_norm():
    omega_mag = 30.0
    seg = FlightSegment(
        id=0,
        frame_range=(0, 4),
        parabola={
            "p0": [0.0, 0.0, 1.0],
            "v0": [10.0, 0.0, 5.0],
            "g": -9.81,
            "spin_axis_world": [0.3, 0.6, 0.74],
            "spin_omega_rad_s": omega_mag,
        },
        fit_residual_px=0.5,
    )
    frames = []
    for i in range(5):
        frames.append(
            BallFrame(
                frame=i,
                world_xyz=(float(i), 0.0, 1.0),
                state="flight",
                confidence=1.0,
                flight_segment_id=0,
            )
        )
    for i in range(5, 9):
        frames.append(
            BallFrame(
                frame=i,
                world_xyz=(5.0 + (i - 5) * 0.2, 0.0, R),
                state="grounded",
                confidence=1.0,
            )
        )
    quats = integrate_orientation(frames, [seg], FPS, R)
    for q in quats.values():
        if q is None:
            continue
        assert abs(np.linalg.norm(q) - 1.0) < 1e-9


def test_missing_world_yields_none_and_carries_q():
    omega_mag = 40.0
    axis = np.array([0.0, 1.0, 0.0])
    seg = FlightSegment(
        id=0,
        frame_range=(0, 3),
        parabola={
            "p0": [0.0, 0.0, 1.0],
            "v0": [10.0, 0.0, 0.0],
            "g": -9.81,
            "spin_axis_world": list(axis),
            "spin_omega_rad_s": omega_mag,
        },
        fit_residual_px=0.5,
    )
    frames = [
        BallFrame(frame=0, world_xyz=(0.0, 0.0, 1.0), state="flight",
                  confidence=1.0, flight_segment_id=0),
        BallFrame(frame=1, world_xyz=None, state="missing", confidence=0.0),
        BallFrame(frame=2, world_xyz=(2.0, 0.0, 1.0), state="flight",
                  confidence=1.0, flight_segment_id=0),
    ]
    quats = integrate_orientation(frames, [seg], FPS, R)
    assert quats[1] is None
    # Frame 0 stores identity then advances one flight step; frame 1 is missing
    # so it contributes nothing and carries q; frame 2 therefore equals the
    # single advance from frame 0 (one flight step from identity).
    step = quat_from_axis_angle(axis, omega_mag * DT)
    expected = quat_normalize(quat_mul(step, (1.0, 0.0, 0.0, 0.0)))
    assert np.allclose(
        _quat_to_rotmat(quats[2]), _quat_to_rotmat(expected), atol=1e-7
    )
