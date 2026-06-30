"""Recover a known Magnus (spin-augmented) ball trajectory from per-frame
image observations. Companion to test_ball_flight.py — same camera setup,
different physics."""

from __future__ import annotations

import numpy as np
import pytest

from src.utils.bundle_adjust import (
    fit_magnus_trajectory,
    fit_parabola_to_image_observations,
)
from src.utils.ball_piecewise_solver import SolverCfg


def _broadcast_camera() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (K, R, t) matching the test_ball_flight.py broadcast pose."""
    K = np.array([[1500.0, 0, 960], [0, 1500.0, 540], [0, 0, 1]])
    look = np.array([0.0, 64.0, -30.0])
    look /= np.linalg.norm(look)
    right = np.array([1.0, 0.0, 0.0])
    down = np.cross(look, right)
    R = np.array([right, down, look], dtype=float)
    t = -R @ np.array([52.5, -30.0, 30.0])
    return K, R, t


def _integrate_magnus_truth(
    p0: np.ndarray,
    v0: np.ndarray,
    omega: np.ndarray,
    g: float,
    drag_k_over_m: float,
    duration: float,
    substep: float = 0.0005,
) -> tuple[np.ndarray, np.ndarray]:
    """Ground-truth RK4 integration with a very fine step. Returns (times, positions)."""
    g_vec = np.array([0.0, 0.0, g])

    def accel(v: np.ndarray) -> np.ndarray:
        return g_vec + drag_k_over_m * np.cross(omega, v)

    n_steps = int(round(duration / substep)) + 1
    times = np.arange(n_steps) * substep
    pos = np.zeros((n_steps, 3))
    pos[0] = p0
    p, v = p0.copy(), v0.copy()
    for i in range(1, n_steps):
        h = substep
        k1v = accel(v)
        k1p = v
        k2v = accel(v + 0.5 * h * k1v)
        k2p = v + 0.5 * h * k1v
        k3v = accel(v + 0.5 * h * k2v)
        k3p = v + 0.5 * h * k2v
        k4v = accel(v + h * k3v)
        k4p = v + h * k3v
        p = p + (h / 6.0) * (k1p + 2 * k2p + 2 * k3p + k4p)
        v = v + (h / 6.0) * (k1v + 2 * k2v + 2 * k3v + k4v)
        pos[i] = p
    return times, pos


def _project(pts: np.ndarray, K: np.ndarray, R: np.ndarray, t: np.ndarray) -> list[tuple[float, float]]:
    obs = []
    for p in pts:
        cam = R @ p + t
        pix = K @ cam
        obs.append((float(pix[0] / pix[2]), float(pix[1] / pix[2])))
    return obs


def _sample_at_frames(times: np.ndarray, pos: np.ndarray, fps: float, n_frames: int) -> np.ndarray:
    frame_times = np.arange(n_frames) / fps
    idx = np.clip(np.round(frame_times / (times[1] - times[0])).astype(int), 0, len(times) - 1)
    return pos[idx]


@pytest.mark.unit
def test_fit_magnus_recovers_sidespin_axis_and_magnitude():
    K, R, t = _broadcast_camera()
    g = -9.81
    drag = 0.005
    fps = 30.0
    duration = 1.0
    n = int(round(fps * duration)) + 1

    p0_true = np.array([30.0, 40.0, 0.5])
    v0_true = np.array([12.0, -8.0, 9.0])
    # Pure sidespin around the world vertical (z) axis — produces a
    # banana-curve in the horizontal plane reminiscent of a Beckham
    # free-kick.
    omega_true = np.array([0.0, 0.0, 25.0])

    times, pos = _integrate_magnus_truth(p0_true, v0_true, omega_true, g, drag, duration)
    sampled = _sample_at_frames(times, pos, fps, n)
    uv = _project(sampled, K, R, t)
    obs = [(i, uv[i]) for i in range(n)]
    Ks = [K] * n
    Rs = [R] * n
    ts = [t] * n

    # Warm-start from a parabola fit (matches what BallStage will do).
    p0_seed, v0_seed, _ = fit_parabola_to_image_observations(
        obs, Ks=Ks, Rs=Rs, t_world=t, fps=fps,
    )
    p0_hat, v0_hat, omega_hat, residual = fit_magnus_trajectory(
        obs,
        Ks=Ks,
        Rs=Rs,
        t_world=t,
        fps=fps,
        drag_k_over_m=drag,
        p0_seed=p0_seed,
        v0_seed=v0_seed,
    )

    axis_true = omega_true / np.linalg.norm(omega_true)
    axis_hat = omega_hat / np.linalg.norm(omega_hat)
    angle_deg = float(np.degrees(np.arccos(np.clip(np.dot(axis_true, axis_hat), -1.0, 1.0))))
    rel_mag_err = abs(np.linalg.norm(omega_hat) - np.linalg.norm(omega_true)) / np.linalg.norm(omega_true)

    assert angle_deg < 10.0, f"axis off by {angle_deg:.1f}°"
    assert rel_mag_err < 0.15, f"|ω| relative error {rel_mag_err:.2%}"
    assert residual < 1.0, f"residual {residual:.2f} px"


@pytest.mark.unit
def test_fit_magnus_collapses_to_parabola_when_no_spin():
    """With zero ground-truth spin, fit should recover near-zero |ω|."""
    K, R, t = _broadcast_camera()
    g = -9.81
    drag = 0.005
    fps = 30.0
    duration = 1.0
    n = int(round(fps * duration)) + 1

    p0_true = np.array([30.0, 40.0, 0.5])
    v0_true = np.array([12.0, -8.0, 9.0])
    omega_true = np.zeros(3)

    times, pos = _integrate_magnus_truth(p0_true, v0_true, omega_true, g, drag, duration)
    sampled = _sample_at_frames(times, pos, fps, n)
    uv = _project(sampled, K, R, t)
    obs = [(i, uv[i]) for i in range(n)]
    Ks = [K] * n
    Rs = [R] * n

    p0_seed, v0_seed, parab_resid = fit_parabola_to_image_observations(
        obs, Ks=Ks, Rs=Rs, t_world=t, fps=fps,
    )
    _, _, omega_hat, magnus_resid = fit_magnus_trajectory(
        obs,
        Ks=Ks,
        Rs=Rs,
        t_world=t,
        fps=fps,
        drag_k_over_m=drag,
        p0_seed=p0_seed,
        v0_seed=v0_seed,
    )

    # |ω| should be small — anything below ~5 rad/s on a no-spin trajectory
    # is firmly in the noise floor of the fitter at this segment length.
    assert np.linalg.norm(omega_hat) < 5.0
    # And the Magnus residual should be at least as good as parabola.
    assert magnus_resid <= parab_resid + 0.1


# Physical ceiling for free-kick spin: ~15 rev/s = 94.2 rad/s.  Task 1
# lowers the configured cap from 200 to 95 to match this real-world limit.
_PHYSICAL_SPIN_CEILING_RAD_S = 95.0


@pytest.mark.unit
def test_spin_cap_clamps_unphysically_high_omega():
    """An arc generated with an unphysically-high true ω (150 rad/s) must
    be clamped to ≤ 95 rad/s by the SolverCfg.spin_max_omega_rad_s default.

    The cap threads as ``omega_mag_bound`` to ``fit_magnus_trajectory`` via
    the piecewise solver's Magnus-refinement call sites.  We exercise the
    same gate here by reading SolverCfg's class-level default and passing it
    directly to the fitter.

    This test FAILS at the old cap (200 rad/s) because the LM is free to
    recover ω ≈ 150 rad/s which satisfies the 200-bound but violates the
    physical ceiling.  After lowering the default to 95, the optimizer is
    bounded to ≤ 95 rad/s and the test PASSES.
    """
    K, R, t = _broadcast_camera()
    g = -9.81
    drag = 0.005
    fps = 30.0
    duration = 1.0
    n = int(round(fps * duration)) + 1

    p0_true = np.array([30.0, 40.0, 0.5])
    v0_true = np.array([12.0, -8.0, 9.0])
    # Unphysically high sidespin — well above the real free-kick ceiling of
    # ~15 rev/s (≈ 94 rad/s).
    omega_true = np.array([0.0, 0.0, 150.0])

    times, pos = _integrate_magnus_truth(p0_true, v0_true, omega_true, g, drag, duration)
    sampled = _sample_at_frames(times, pos, fps, n)
    uv = _project(sampled, K, R, t)
    obs = [(i, uv[i]) for i in range(n)]
    Ks = [K] * n
    Rs = [R] * n

    # Read the cap from SolverCfg so the test is coupled to the config value.
    cap = SolverCfg.spin_max_omega_rad_s  # class-level default

    p0_seed, v0_seed, _ = fit_parabola_to_image_observations(
        obs, Ks=Ks, Rs=Rs, t_world=t, fps=fps,
    )
    omega_axis = omega_true / np.linalg.norm(omega_true)
    _, _, omega_hat, _ = fit_magnus_trajectory(
        obs,
        Ks=Ks,
        Rs=Rs,
        t_world=t,
        fps=fps,
        drag_k_over_m=drag,
        p0_seed=p0_seed,
        v0_seed=v0_seed,
        # Hinted path: axis fixed, magnitude bounded by the configured cap.
        omega_seed=omega_axis * min(cap, 63.0),
        omega_axis_fixed=omega_axis,
        omega_mag_bound=cap,
    )

    omega_hat_mag = float(np.linalg.norm(omega_hat))
    # After Task 1 the cap is 95; any fit above this is unphysical.
    assert omega_hat_mag <= _PHYSICAL_SPIN_CEILING_RAD_S + 1e-6, (
        f"recovered |ω| {omega_hat_mag:.1f} rad/s exceeds physical ceiling "
        f"{_PHYSICAL_SPIN_CEILING_RAD_S:.1f} rad/s — SolverCfg.spin_max_omega_rad_s "
        f"is {cap:.1f} (should be ≤ {_PHYSICAL_SPIN_CEILING_RAD_S:.1f})"
    )


@pytest.mark.unit
def test_spin_cap_does_not_clamp_normal_freekick_omega():
    """A normal free-kick ω (~63 rad/s ≈ 10 rev/s) must be recovered close to
    truth without the cap clamping it (it sits comfortably below 95 rad/s).

    This test validates that lowering the cap from 200 to 95 does NOT degrade
    recovery of in-range spins.  The fitter must remain free below the cap.
    """
    K, R, t = _broadcast_camera()
    g = -9.81
    drag = 0.005
    fps = 30.0
    duration = 1.0
    n = int(round(fps * duration)) + 1

    p0_true = np.array([30.0, 40.0, 0.5])
    v0_true = np.array([12.0, -8.0, 9.0])
    # ~10 rev/s sidespin — firmly within real free-kick range and below 95.
    omega_true = np.array([0.0, 0.0, 63.0])
    omega_hat_mag_true = float(np.linalg.norm(omega_true))

    times, pos = _integrate_magnus_truth(p0_true, v0_true, omega_true, g, drag, duration)
    sampled = _sample_at_frames(times, pos, fps, n)
    uv = _project(sampled, K, R, t)
    obs = [(i, uv[i]) for i in range(n)]
    Ks = [K] * n
    Rs = [R] * n

    cap = SolverCfg.spin_max_omega_rad_s  # must be ≥ 95 after Task 1

    p0_seed, v0_seed, _ = fit_parabola_to_image_observations(
        obs, Ks=Ks, Rs=Rs, t_world=t, fps=fps,
    )
    omega_axis = omega_true / np.linalg.norm(omega_true)
    _, _, omega_hat, residual = fit_magnus_trajectory(
        obs,
        Ks=Ks,
        Rs=Rs,
        t_world=t,
        fps=fps,
        drag_k_over_m=drag,
        p0_seed=p0_seed,
        v0_seed=v0_seed,
        omega_seed=omega_axis * omega_hat_mag_true,
        omega_axis_fixed=omega_axis,
        omega_mag_bound=cap,
    )

    omega_hat_mag = float(np.linalg.norm(omega_hat))
    # 63 rad/s is well below 95; the cap must not have clamped it.
    assert omega_hat_mag < _PHYSICAL_SPIN_CEILING_RAD_S - 5.0, (
        f"|ω| hat {omega_hat_mag:.1f} rad/s unexpectedly close to or above "
        f"physical ceiling {_PHYSICAL_SPIN_CEILING_RAD_S:.1f} rad/s"
    )
    # Recovered within 20 % of truth — cap is not biting.
    rel_err = abs(omega_hat_mag - omega_hat_mag_true) / omega_hat_mag_true
    assert rel_err < 0.20, (
        f"relative magnitude error {rel_err:.1%} exceeds 20 % for a "
        f"normal free-kick ω ({omega_hat_mag_true:.0f} rad/s)"
    )
    assert residual < 1.0, f"residual {residual:.2f} px"
