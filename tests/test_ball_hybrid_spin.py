"""Tests for ``src.utils.ball_hybrid_spin.fit_span_spin``.

Covers: known-spin recovery (sidespin free kick, topspin drive) under
realistic pixel noise on a synthetic broadcast-like pinhole camera AND
real per-frame camera geometry (gberch's solved camera track, when
present on disk); false-acceptance rate on spin-free spans; and the
"both knots stay exact" invariant.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from src.utils.ball_hybrid_physics import (
    BALL_RADIUS_M,
    CD_DEFAULT,
    DEFAULT_MAGNUS_COEFF,
    shoot_arc,
    simulate,
)
from src.utils.ball_hybrid_spin import (
    DEFAULT_BOUNDS,
    MAX_OMEGA_RAD_S,
    _spin_axes,
    fit_span_spin,
)
from src.utils.camera_projection import project_world_to_image

GBERCH_CAMERA_TRACK = Path(
    "/Users/joebower/workplace/football-perspectives/output/camera/"
    "gberch_camera_track.json"
)


# ---------------------------------------------------------------------------
# Synthetic broadcast-like pinhole camera (static, wide-ish FOV so most of
# the pitch stays in frame).
# ---------------------------------------------------------------------------

def _look_at_RT(camera_center: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    world_up = np.array([0.0, 0.0, 1.0])
    forward = target - camera_center
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, world_up)
    right = right / np.linalg.norm(right)
    down = np.cross(forward, right)
    R = np.vstack([right, down, forward])
    t = -R @ camera_center
    return R, t


_CAM_K = np.array([[1800.0, 0.0, 960.0], [0.0, 1800.0, 540.0], [0.0, 0.0, 1.0]])
_CAM_C = np.array([-20.0, 34.0, 20.0])
_CAM_TARGET = np.array([52.5, 34.0, 0.0])
_CAM_R, _CAM_T = _look_at_RT(_CAM_C, _CAM_TARGET)
_CAM_DISTORTION = (0.0, 0.0)


def _synthetic_project_fn(_t_s: float, xyz: np.ndarray) -> np.ndarray:
    """Static synthetic pinhole camera; ignores ``t_s``."""
    return project_world_to_image(
        _CAM_K, _CAM_R, _CAM_T, _CAM_DISTORTION, np.asarray(xyz).reshape(1, 3)
    )[0]


# ---------------------------------------------------------------------------
# Shared span-generation helper.
# ---------------------------------------------------------------------------

def _make_span(
    p_a: np.ndarray,
    v0: np.ndarray,
    duration_s: float,
    omega: np.ndarray | None,
    *,
    n_obs: int,
    noise_px_sigma: float,
    seed: int,
    cd: float = CD_DEFAULT,
    magnus_coeff: float = DEFAULT_MAGNUS_COEFF,
    project_fn=_synthetic_project_fn,
):
    """Simulate a (possibly spinning) flight span and return
    ``(p_b, obs_times, obs_uv)`` — noisy pixel evidence at ``n_obs``
    interior frames, plus the exact endpoint reached at ``duration_s``.
    """
    rng = np.random.default_rng(seed)
    p_b = simulate(p_a, v0, [duration_s], cd=cd, omega=omega,
                    magnus_coeff=magnus_coeff)[0]
    obs_times = np.linspace(0.0, duration_s, n_obs + 2)[1:-1]  # interior only
    positions = simulate(p_a, v0, obs_times, cd=cd, omega=omega,
                          magnus_coeff=magnus_coeff)
    obs_uv = np.empty((len(obs_times), 2))
    for i, (t_s, pos) in enumerate(zip(obs_times, positions)):
        uv = project_fn(float(t_s), pos)
        obs_uv[i] = uv + rng.normal(scale=noise_px_sigma, size=2)
    return p_b, obs_times, obs_uv


def _axis_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    cos_ang = np.clip(np.dot(a, b), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_ang)))


# ---------------------------------------------------------------------------
# Known-spin recovery: free kick (sidespin).
# ---------------------------------------------------------------------------

class TestSidespinRecovery:
    def test_recovers_sidespin_free_kick(self):
        p_a = np.array([30.0, 20.0, 0.11])
        v0_true = np.array([14.0, 8.0, 6.0])  # m/s, a driven free kick
        omega_true = np.array([0.0, 0.0, 20.0])  # pure sidespin, 20 rad/s
        duration_s = 0.8

        p_b, obs_times, obs_uv = _make_span(
            p_a, v0_true, duration_s, omega_true,
            n_obs=18, noise_px_sigma=2.0, seed=1,
        )

        fit = fit_span_spin(
            tuple(p_a), 0.0, tuple(p_b), duration_s,
            obs_times, obs_uv, _synthetic_project_fn,
        )

        assert fit is not None
        assert fit.delta_bic > 0.0
        true_mag = float(np.linalg.norm(omega_true))
        assert abs(fit.rad_s - true_mag) / true_mag <= 0.20
        angle = _axis_angle_deg(np.array(fit.omega_world), omega_true)
        assert angle <= 15.0

    def test_recovers_sidespin_negative_direction(self):
        p_a = np.array([40.0, 30.0, 0.11])
        v0_true = np.array([-14.0, 16.0, 6.0])
        omega_true = np.array([0.0, 0.0, -30.0])  # opposite curl direction
        duration_s = 0.85

        p_b, obs_times, obs_uv = _make_span(
            p_a, v0_true, duration_s, omega_true,
            n_obs=24, noise_px_sigma=1.5, seed=2,
        )

        fit = fit_span_spin(
            tuple(p_a), 0.0, tuple(p_b), duration_s,
            obs_times, obs_uv, _synthetic_project_fn,
        )

        assert fit is not None
        true_mag = float(np.linalg.norm(omega_true))
        assert abs(fit.rad_s - true_mag) / true_mag <= 0.20
        angle = _axis_angle_deg(np.array(fit.omega_world), omega_true)
        assert angle <= 15.0


# ---------------------------------------------------------------------------
# Known-spin recovery: driven shot (topspin), using the module's own axis
# helper so the true omega lies exactly along the recoverable topspin dof.
# ---------------------------------------------------------------------------

class TestTopspinRecovery:
    def test_recovers_topspin_drive(self):
        p_a = np.array([25.0, 40.0, 0.11])
        v0_true = np.array([16.0, -5.0, 4.0])  # a low, driven strike
        axis_top, _axis_side = _spin_axes(v0_true)
        omega_true = 45.0 * axis_top  # driven-shot magnitude (~40-60 rad/s)
        duration_s = 0.6

        p_b, obs_times, obs_uv = _make_span(
            p_a, v0_true, duration_s, omega_true,
            n_obs=16, noise_px_sigma=2.5, seed=3,
        )

        fit = fit_span_spin(
            tuple(p_a), 0.0, tuple(p_b), duration_s,
            obs_times, obs_uv, _synthetic_project_fn,
        )

        assert fit is not None
        true_mag = float(np.linalg.norm(omega_true))
        assert abs(fit.rad_s - true_mag) / true_mag <= 0.20
        angle = _axis_angle_deg(np.array(fit.omega_world), omega_true)
        assert angle <= 15.0

    def test_bounds_cap_matches_ten_rev_per_s(self):
        assert math.isclose(MAX_OMEGA_RAD_S, 10.0 * 2.0 * math.pi, rel_tol=1e-9)
        assert DEFAULT_BOUNDS == (-MAX_OMEGA_RAD_S, MAX_OMEGA_RAD_S)


# ---------------------------------------------------------------------------
# Both knots stay exact, regardless of spin acceptance.
# ---------------------------------------------------------------------------

class TestKnotsStayExact:
    def test_endpoints_exact_for_accepted_spin_fit(self):
        p_a = np.array([30.0, 20.0, 0.11])
        v0_true = np.array([14.0, 8.0, 6.0])
        omega_true = np.array([0.0, 0.0, 20.0])
        duration_s = 0.8

        p_b, obs_times, obs_uv = _make_span(
            p_a, v0_true, duration_s, omega_true,
            n_obs=18, noise_px_sigma=2.0, seed=1,
        )

        fit = fit_span_spin(
            tuple(p_a), 0.0, tuple(p_b), duration_s,
            obs_times, obs_uv, _synthetic_project_fn,
        )
        assert fit is not None

        omega_fit = np.array(fit.omega_world)
        v0_fit = shoot_arc(p_a, 0.0, p_b, duration_s, cd=CD_DEFAULT,
                            omega=omega_fit, magnus_coeff=DEFAULT_MAGNUS_COEFF)
        p_end = simulate(p_a, v0_fit, [duration_s], cd=CD_DEFAULT,
                          omega=omega_fit, magnus_coeff=DEFAULT_MAGNUS_COEFF)[0]
        assert np.allclose(p_end, p_b, atol=1e-6)
        p_start = simulate(p_a, v0_fit, [0.0], cd=CD_DEFAULT,
                            omega=omega_fit, magnus_coeff=DEFAULT_MAGNUS_COEFF)[0]
        assert np.allclose(p_start, p_a, atol=1e-9)


# ---------------------------------------------------------------------------
# False-acceptance rate on spin-free spans.
# ---------------------------------------------------------------------------

class TestFalseAcceptance:
    def test_false_acceptance_rate_le_5_percent(self):
        n_seeds = 50
        n_accepted = 0
        rng_master = np.random.default_rng(12345)
        for seed in range(n_seeds):
            p_a = np.array([
                rng_master.uniform(20.0, 80.0),
                rng_master.uniform(15.0, 50.0),
                0.11,
            ])
            v0 = np.array([
                rng_master.uniform(-10.0, 10.0),
                rng_master.uniform(-10.0, 10.0),
                rng_master.uniform(3.0, 7.0),
            ])
            duration_s = float(rng_master.uniform(0.5, 0.9))
            noise_sigma = float(rng_master.uniform(1.5, 3.0))

            p_b, obs_times, obs_uv = _make_span(
                p_a, v0, duration_s, None,
                n_obs=16, noise_px_sigma=noise_sigma, seed=1000 + seed,
            )

            fit = fit_span_spin(
                tuple(p_a), 0.0, tuple(p_b), duration_s,
                obs_times, obs_uv, _synthetic_project_fn,
            )
            if fit is not None:
                n_accepted += 1

        assert n_accepted <= math.ceil(0.05 * n_seeds), (
            f"false-acceptance {n_accepted}/{n_seeds} exceeds 5% budget"
        )


# ---------------------------------------------------------------------------
# Basic input validation / min_obs gate.
# ---------------------------------------------------------------------------

class TestGuards:
    def test_returns_none_below_min_obs(self):
        p_a = np.array([30.0, 20.0, 0.11])
        v0_true = np.array([14.0, 8.0, 6.0])
        omega_true = np.array([0.0, 0.0, 15.0])
        duration_s = 0.8
        p_b, obs_times, obs_uv = _make_span(
            p_a, v0_true, duration_s, omega_true,
            n_obs=4, noise_px_sigma=1.5, seed=7,
        )
        fit = fit_span_spin(
            tuple(p_a), 0.0, tuple(p_b), duration_s,
            obs_times, obs_uv, _synthetic_project_fn, min_obs=8,
        )
        assert fit is None

    def test_rejects_non_positive_duration(self):
        with pytest.raises(ValueError):
            fit_span_spin(
                (0.0, 0.0, 0.11), 1.0, (1.0, 1.0, 0.11), 1.0,
                [0.5], [(0.0, 0.0)], _synthetic_project_fn,
            )

    def test_filters_observations_outside_span(self):
        p_a = np.array([30.0, 20.0, 0.11])
        v0_true = np.array([14.0, 8.0, 6.0])
        omega_true = np.array([0.0, 0.0, 20.0])
        duration_s = 0.8
        p_b, obs_times, obs_uv = _make_span(
            p_a, v0_true, duration_s, omega_true,
            n_obs=18, noise_px_sigma=2.0, seed=1,
        )
        # Inject a handful of out-of-span observations that would be
        # garbage if not filtered (huge synthetic pixel offsets).
        obs_times_padded = np.concatenate([obs_times, [-0.5, duration_s + 0.5]])
        obs_uv_padded = np.vstack([obs_uv, [[-9999.0, -9999.0], [9999.0, 9999.0]]])

        fit = fit_span_spin(
            tuple(p_a), 0.0, tuple(p_b), duration_s,
            obs_times_padded, obs_uv_padded, _synthetic_project_fn,
        )
        assert fit is not None
        true_mag = float(np.linalg.norm(omega_true))
        assert abs(fit.rad_s - true_mag) / true_mag <= 0.20


# ---------------------------------------------------------------------------
# Real camera geometry (gberch's solved camera track), when present.
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not GBERCH_CAMERA_TRACK.exists(),
                     reason="gberch camera track not available on disk")
class TestRealCameraGeometry:
    @staticmethod
    def _load_camera():
        with GBERCH_CAMERA_TRACK.open() as fh:
            data = json.load(fh)
        fps = float(data["fps"])
        distortion = tuple(data.get("distortion", (0.0, 0.0)))
        per_frame = {}
        for f in data["frames"]:
            per_frame[int(f["frame"])] = (
                np.array(f["K"], dtype=float),
                np.array(f["R"], dtype=float),
                np.array(f["t"], dtype=float),
            )
        return fps, distortion, per_frame

    def test_recovers_sidespin_on_real_camera_track(self):
        fps, distortion, per_frame = self._load_camera()
        frame_a = 60
        assert frame_a in per_frame

        def project_fn(t_s: float, xyz: np.ndarray) -> np.ndarray:
            frame = frame_a + int(round(t_s * fps))
            frame = min(max(frame, min(per_frame)), max(per_frame))
            K, R, t = per_frame[frame]
            return project_world_to_image(
                K, R, t, distortion, np.asarray(xyz).reshape(1, 3)
            )[0]

        p_a = np.array([35.0, 34.0, 0.11])
        v0_true = np.array([15.0, -15.0, 5.0])
        omega_true = np.array([0.0, 0.0, 20.0])
        duration_s = 0.7  # 21 frames at 30fps

        p_b, obs_times, obs_uv = _make_span(
            p_a, v0_true, duration_s, omega_true,
            n_obs=14, noise_px_sigma=2.0, seed=5, project_fn=project_fn,
        )

        fit = fit_span_spin(
            tuple(p_a), 0.0, tuple(p_b), duration_s,
            obs_times, obs_uv, project_fn,
        )

        assert fit is not None
        true_mag = float(np.linalg.norm(omega_true))
        assert abs(fit.rad_s - true_mag) / true_mag <= 0.20
        angle = _axis_angle_deg(np.array(fit.omega_world), omega_true)
        assert angle <= 15.0
