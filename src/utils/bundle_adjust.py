"""Bundle-adjustment helpers for the broadcast-mono pipeline.

Two fitters live here:

* :func:`fit_parabola_to_image_observations` -- Levenberg-Marquardt fit
  of a constant-gravity parabola (p(t) = p0 + v0*t + 0.5*g*t^2) to a
  sequence of per-frame ball pixel observations, projected through the
  per-frame camera-track ``(K_t, R_t)`` and the clip-shared ``t_world``.
* :func:`fit_magnus_trajectory` -- Levenberg-Marquardt fit of a
  Magnus-augmented trajectory, ``dv/dt = g + k * (ω × v)``, integrated
  with RK4 inside the residual loop.  Recovers ``(p0, v0, ω)``.  Warm-
  starts from the parabola fit if seeds are supplied.

The parabola seed is computed by ground-projecting the first/last image
points to a coarse plane (z = 0.5 m, mid-flight) using
:func:`src.utils.foot_anchor.ankle_ray_to_pitch`, and assuming a
symmetric vertical velocity that places apex at mid-flight.
"""

from __future__ import annotations

import numpy as np


def fit_parabola_to_image_observations(
    observations: list[tuple[int, tuple[float, float]]],
    *,
    Ks: list[np.ndarray],
    Rs: list[np.ndarray],
    t_world: np.ndarray | list[np.ndarray],
    fps: float,
    g: float = -9.81,
    max_iter: int = 100,
    distortion: tuple[float, float] = (0.0, 0.0),
    p0_fixed: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Fit a 3D parabola to per-frame image observations.

    Args:
        observations: list of ``(frame_index, (u, v))`` pairs ordered by
            time.  ``frame_index`` is the absolute clip frame, used only
            for time deltas; ``Ks`` and ``Rs`` are looked up positionally.
        Ks: position-parallel to ``observations`` — one entry per
            observation, in the same order.
        Rs: position-parallel to ``observations``.
        t_world: either a clip-shared (3,) translation or a list of
            per-frame (3,) translations parallel to ``observations``.
            Use the per-frame form for static-camera clips where ``t``
            varies with the SLERP'd ``R``.
        fps: frame rate.
        g: gravity along world-z (default -9.81 m/s^2).
        max_iter: LM iteration cap (passed through to scipy as
            ``max_nfev = max_iter * 50``).
        distortion: (k1, k2) radial distortion. Default ``(0, 0)``;
            non-zero values undistort each image observation before
            measuring reprojection residuals.
        p0_fixed: when not ``None``, the world-space starting position is
            pinned to this value and only ``v0`` (3 dof) is optimised,
            reducing the ill-conditioned monocular-depth ambiguity.

    Returns:
        ``(p0, v0, mean_residual_px)`` where ``mean_residual_px`` is
        the RMS reprojection error in pixels.
    """
    from scipy.optimize import least_squares

    obs_array = np.array([o[1] for o in observations], dtype=float)
    frame_idx = np.array([o[0] for o in observations], dtype=int)
    dt = (frame_idx - frame_idx[0]) / fps
    g_vec = np.array([0.0, 0.0, g])

    # Normalise t_world to per-observation form so the residual loop is uniform.
    n_obs = len(observations)
    if isinstance(t_world, list) or (
        isinstance(t_world, np.ndarray) and t_world.ndim == 2
    ):
        ts = [np.asarray(t, dtype=float) for t in t_world]
        if len(ts) != n_obs:
            raise ValueError(
                f"per-frame t_world has {len(ts)} entries, expected {n_obs}"
            )
    else:
        t_shared = np.asarray(t_world, dtype=float)
        ts = [t_shared] * n_obs

    def _residuals(params: np.ndarray) -> np.ndarray:
        p0 = params[:3]
        v0 = params[3:6]
        pts = p0 + np.outer(dt, v0) + 0.5 * np.outer(dt ** 2, g_vec)
        residuals = []
        for i in range(n_obs):
            cam = Rs[i] @ pts[i] + ts[i]
            pix = Ks[i] @ cam
            uv = pix[:2] / pix[2]
            residuals.append(uv - obs_array[i])
        return np.concatenate(residuals)

    # Seed from start/end image points -> ground projection (rough).
    from src.utils.foot_anchor import ankle_ray_to_pitch

    p_start = ankle_ray_to_pitch(
        observations[0][1],
        K=Ks[0],
        R=Rs[0],
        t=ts[0],
        plane_z=0.5,
        distortion=distortion,
    )
    p_end = ankle_ray_to_pitch(
        observations[-1][1],
        K=Ks[-1],
        R=Rs[-1],
        t=ts[-1],
        plane_z=0.5,
        distortion=distortion,
    )
    duration = dt[-1] if dt[-1] > 0 else 1.0
    v_horiz = (p_end - p_start) / duration
    v0_seed = np.array([v_horiz[0], v_horiz[1], 0.5 * abs(g) * duration])
    p0_seed = p_start

    if p0_fixed is None:
        result = least_squares(
            _residuals,
            np.concatenate([p0_seed, v0_seed]),
            method="lm",
            max_nfev=max_iter * 50,
        )
    else:
        p0_pin = np.asarray(p0_fixed, dtype=float).copy()

        def _residuals_v0only(params: np.ndarray) -> np.ndarray:
            v0 = params[:3]
            pts = p0_pin + np.outer(dt, v0) + 0.5 * np.outer(dt ** 2, g_vec)
            residuals = []
            for i in range(n_obs):
                cam = Rs[i] @ pts[i] + ts[i]
                pix = Ks[i] @ cam
                uv = pix[:2] / pix[2]
                residuals.append(uv - obs_array[i])
            return np.concatenate(residuals)

        result = least_squares(
            _residuals_v0only,
            v0_seed,
            method="lm",
            max_nfev=max_iter * 50,
        )
    n = len(observations)
    mean_residual = float(np.linalg.norm(result.fun) / np.sqrt(n))
    if p0_fixed is None:
        p0_opt = result.x[:3]
        v0_opt = result.x[3:6]
    else:
        p0_opt = np.asarray(p0_fixed, dtype=float).copy()
        v0_opt = result.x[:3]
    return p0_opt, v0_opt, mean_residual


def _integrate_magnus_positions(
    p0: np.ndarray,
    v0: np.ndarray,
    omega: np.ndarray,
    g_vec: np.ndarray,
    drag_k_over_m: float,
    sample_times: np.ndarray,
    substeps_per_interval: int = 4,
) -> np.ndarray:
    """RK4-integrate ``dv/dt = g + k * (ω × v)`` and sample at ``sample_times``.

    ``sample_times`` must start at 0 and be monotonically increasing.
    Returns positions of shape ``(len(sample_times), 3)``.
    """
    out = np.zeros((len(sample_times), 3))
    out[0] = p0

    def accel(v: np.ndarray) -> np.ndarray:
        return g_vec + drag_k_over_m * np.cross(omega, v)

    p, v = p0.astype(float).copy(), v0.astype(float).copy()
    for i in range(1, len(sample_times)):
        t_prev = sample_times[i - 1]
        t_next = sample_times[i]
        total = t_next - t_prev
        if total <= 0:
            out[i] = p
            continue
        h = total / substeps_per_interval
        for _ in range(substeps_per_interval):
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
        out[i] = p
    return out


def fit_magnus_trajectory(
    observations: list[tuple[int, tuple[float, float]]],
    *,
    Ks: list[np.ndarray],
    Rs: list[np.ndarray],
    t_world: np.ndarray | list[np.ndarray],
    fps: float,
    g: float = -9.81,
    drag_k_over_m: float = 0.005,
    p0_seed: np.ndarray | None = None,
    v0_seed: np.ndarray | None = None,
    omega_seed: np.ndarray | None = None,
    max_iter: int = 100,
    distortion: tuple[float, float] = (0.0, 0.0),
    p0_fixed: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Fit a Magnus-augmented 3D trajectory to per-frame image observations.

    Optimises ``(p0, v0, ω)`` (9 dof) by minimising pixel reprojection
    residuals under ``dv/dt = g + drag_k_over_m * (ω × v)``, integrated
    with RK4 between observation times.

    Args:
        observations: ``(frame_index, (u, v))`` pairs ordered by time.
        Ks, Rs, t_world: position-parallel to ``observations`` (see
            :func:`fit_parabola_to_image_observations` for details).
        fps: clip frame rate.
        g: gravity along world-z (default -9.81 m/s^2).
        drag_k_over_m: lumped drag/Magnus coefficient (k / m).
        p0_seed, v0_seed: warm-start seeds. If both ``None``, seeds are
            derived from a parabola fit on the same observations.
        omega_seed: 3-vector seed for angular velocity (rad/s).  Default
            zeros — the LM finds a non-zero ω only if it improves the
            pixel residual.
        max_iter: LM iteration cap.
        distortion: unused here (kept for signature symmetry with the
            parabola fitter; image residuals are computed against raw
            observations).

    Returns:
        ``(p0, v0, ω, mean_residual_px)``.
    """
    from scipy.optimize import least_squares

    obs_array = np.array([o[1] for o in observations], dtype=float)
    frame_idx = np.array([o[0] for o in observations], dtype=int)
    dt = (frame_idx - frame_idx[0]) / fps
    g_vec = np.array([0.0, 0.0, g])

    n_obs = len(observations)
    if isinstance(t_world, list) or (
        isinstance(t_world, np.ndarray) and t_world.ndim == 2
    ):
        ts = [np.asarray(t, dtype=float) for t in t_world]
        if len(ts) != n_obs:
            raise ValueError(
                f"per-frame t_world has {len(ts)} entries, expected {n_obs}"
            )
    else:
        t_shared = np.asarray(t_world, dtype=float)
        ts = [t_shared] * n_obs

    if p0_seed is None or v0_seed is None:
        p0_seed, v0_seed, _ = fit_parabola_to_image_observations(
            observations,
            Ks=Ks,
            Rs=Rs,
            t_world=t_world,
            fps=fps,
            g=g,
            max_iter=max_iter,
            distortion=distortion,
        )
    if omega_seed is None:
        omega_seed = np.zeros(3)

    def _residuals(params: np.ndarray) -> np.ndarray:
        p0 = params[:3]
        v0 = params[3:6]
        omega = params[6:9]
        pts = _integrate_magnus_positions(
            p0, v0, omega, g_vec, drag_k_over_m, dt,
        )
        residuals = []
        for i in range(n_obs):
            cam = Rs[i] @ pts[i] + ts[i]
            pix = Ks[i] @ cam
            uv = pix[:2] / pix[2]
            residuals.append(uv - obs_array[i])
        return np.concatenate(residuals)

    if p0_fixed is None:
        x0 = np.concatenate([p0_seed, v0_seed, omega_seed])
        result = least_squares(_residuals, x0, method="lm",
                               max_nfev=max_iter * 50)
        p0_opt = result.x[:3]
        v0_opt = result.x[3:6]
        omega_opt = result.x[6:9]
    else:
        p0_pin = np.asarray(p0_fixed, dtype=float).copy()

        def _residuals_anchored(params: np.ndarray) -> np.ndarray:
            v0 = params[:3]
            omega = params[3:6]
            positions = _integrate_magnus_positions(
                p0_pin, v0, omega, g_vec, drag_k_over_m, dt,
            )
            residuals = []
            for i in range(n_obs):
                cam = Rs[i] @ positions[i] + ts[i]
                pix = Ks[i] @ cam
                uv = pix[:2] / pix[2]
                residuals.append(uv - obs_array[i])
            return np.concatenate(residuals)

        x0 = np.concatenate([v0_seed, omega_seed])
        result = least_squares(_residuals_anchored, x0, method="lm",
                               max_nfev=max_iter * 50)
        p0_opt = p0_pin
        v0_opt = result.x[:3]
        omega_opt = result.x[3:6]

    mean_residual = float(np.linalg.norm(result.fun) / np.sqrt(n_obs))
    return p0_opt, v0_opt, omega_opt, mean_residual
