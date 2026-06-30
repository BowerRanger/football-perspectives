"""world_fixes: 3D soft constraints in the LM flight fitters."""

from __future__ import annotations

import numpy as np
import pytest

from src.utils.bundle_adjust import (
    fit_magnus_trajectory,
    fit_parabola_to_image_observations,
)
from tests.fixtures.ball_synthetic import broadcast_camera, project_track

FPS = 25.0
G = np.array([0.0, 0.0, -9.81])


def _arc(n, p0, v0):
    return {f: p0 + v0 * (f / FPS) + 0.5 * G * (f / FPS) ** 2
            for f in range(n)}


def _fit_inputs(worlds, K, R, t, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    pix = project_track(worlds, K, R, t)
    obs = []
    for f, (u, v) in sorted(pix.items()):
        du, dv = rng.normal(0.0, noise, 2) if noise else (0.0, 0.0)
        obs.append((f, (u + du, v + dv)))
    n = len(obs)
    return obs, [K] * n, [R] * n, [t] * n


@pytest.mark.unit
def test_parabola_unchanged_without_fixes():
    K, R, t = broadcast_camera()
    worlds = _arc(30, np.array([40.0, 30.0, 0.11]), np.array([8.0, 4.0, 9.0]))
    obs, Ks, Rs, ts = _fit_inputs(worlds, K, R, t)
    a = fit_parabola_to_image_observations(obs, Ks=Ks, Rs=Rs, t_world=ts, fps=FPS)
    b = fit_parabola_to_image_observations(
        obs, Ks=Ks, Rs=Rs, t_world=ts, fps=FPS, world_fixes=[])
    assert np.allclose(a[0], b[0]) and np.allclose(a[1], b[1])
    assert a[2] == pytest.approx(b[2])


@pytest.mark.unit
def test_parabola_fixes_pull_depth_to_truth():
    """Sparse noisy monocular obs leave depth soft; two exact 3D fixes
    must pull the recovered trajectory onto the truth."""
    K, R, t = broadcast_camera()
    p0, v0 = np.array([40.0, 30.0, 0.11]), np.array([8.0, 4.0, 9.0])
    worlds = _arc(30, p0, v0)
    sparse = {f: worlds[f] for f in range(0, 30, 6)}  # 5 obs only
    obs, Ks, Rs, ts = _fit_inputs(sparse, K, R, t, noise=2.0)
    fixes = [(8, np.asarray(worlds[8]), 30.0), (20, np.asarray(worlds[20]), 30.0)]
    p0n, v0n, _ = fit_parabola_to_image_observations(
        obs, Ks=Ks, Rs=Rs, t_world=ts, fps=FPS)
    p0f, v0f, _ = fit_parabola_to_image_observations(
        obs, Ks=Ks, Rs=Rs, t_world=ts, fps=FPS, world_fixes=fixes)
    err_n = np.linalg.norm(p0n - p0) + np.linalg.norm(v0n - v0)
    err_f = np.linalg.norm(p0f - p0) + np.linalg.norm(v0f - v0)
    assert err_f <= err_n + 1e-9
    assert np.linalg.norm(p0f - p0) < 0.5


@pytest.mark.unit
def test_magnus_accepts_fixes_and_respects_weight():
    K, R, t = broadcast_camera()
    p0, v0 = np.array([40.0, 30.0, 0.11]), np.array([8.0, 4.0, 9.0])
    worlds = _arc(30, p0, v0)
    obs, Ks, Rs, ts = _fit_inputs(worlds, K, R, t, noise=1.0)
    wrong_fix = [(15, np.asarray(worlds[15]) + np.array([0.0, 5.0, 0.0]), 1e-6)]
    # Without any fix, get base error.
    p0_base, _, _, _ = fit_magnus_trajectory(
        obs, Ks=Ks, Rs=Rs, t_world=ts, fps=FPS,
        omega_mag_bound=10.0)
    p0w, v0w, _, _ = fit_magnus_trajectory(
        obs, Ks=Ks, Rs=Rs, t_world=ts, fps=FPS,
        omega_mag_bound=10.0, world_fixes=wrong_fix)
    # Near-zero weight: the wrong fix must not drag the solution away
    # from the base fit (error ≤ base + small tolerance).
    base_err = np.linalg.norm(p0_base - p0)
    assert np.linalg.norm(p0w - p0) <= base_err + 0.01


@pytest.mark.unit
def test_magnus_backward_fix_constrains_extrapolation():
    """A world fix placed BEFORE the first observation must constrain the
    backward-extrapolated trajectory — not be silently clamped to t=0.

    Setup:
    - Observations run frames 5..24 (20 frames of a no-spin parabolic arc).
    - A strong fix (weight 30.0) is placed at frame 2, i.e. dt = -3/FPS
      relative to the first observation at frame 5.
    - The ground-truth position at frame 2 is computed from the analytic arc.

    Pass criterion:
    - The fitted trajectory, when evaluated analytically at frame 2's time
      (t2 = -3/FPS relative to first obs), must land within 0.15 m of the
      true position.  This proves the fix was extrapolated backward.

    Before the fix the clamp forces dt=0, so the "fix" residual actually
    measures the mismatch at t=0 (frame 5's position), leaving frame 2
    unconstrained — yielding errors of ~0.77 m.
    """
    K, R, t = broadcast_camera()
    p0_true = np.array([40.0, 30.0, 0.11])
    v0_true = np.array([8.0, 4.0, 9.0])

    # Analytic arc: p(t) = p0_true + v0_true*t + 0.5*G*t^2
    def arc_pos(frame: int) -> np.ndarray:
        ti = frame / FPS
        return p0_true + v0_true * ti + 0.5 * G * ti ** 2

    # Observations start at frame 5 (absolute), run to frame 24.
    obs_frames = list(range(5, 25))
    worlds_obs = {f: arc_pos(f) for f in obs_frames}
    obs, Ks, Rs, ts = _fit_inputs(worlds_obs, K, R, t, noise=0.0)

    # Fix is at absolute frame 2 — two frames before the first observation.
    # dt relative to first obs (frame 5) = (2 - 5)/FPS = -3/FPS  (<0).
    fix_frame = 2
    fix_xyz = arc_pos(fix_frame)
    world_fixes = [(fix_frame, fix_xyz, 30.0)]

    p0_fit, v0_fit, _omega, _res = fit_magnus_trajectory(
        obs, Ks=Ks, Rs=Rs, t_world=ts, fps=FPS,
        world_fixes=world_fixes,
        omega_seed=np.zeros(3),
    )

    # Evaluate fitted arc backward to frame 2.
    # In no-spin / zero-drag the trajectory is a parabola:
    # p(t) = p0_fit + v0_fit * t + 0.5 * G * t^2
    # where t is relative to the first obs (frame 5).
    t_fix_rel = (fix_frame - obs_frames[0]) / FPS  # negative: -3/25 = -0.12 s
    pos_at_fix = p0_fit + v0_fit * t_fix_rel + 0.5 * G * t_fix_rel ** 2

    err = float(np.linalg.norm(pos_at_fix - fix_xyz))
    assert err < 0.15, (
        f"Backward fix not respected: error={err:.3f} m (expected <0.15 m). "
        f"Likely the fix dt was clamped to 0 instead of being extrapolated."
    )
