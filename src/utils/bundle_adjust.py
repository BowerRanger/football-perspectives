"""Bundle-adjustment helpers for the broadcast-mono pipeline.

Currently exposes one fitter:

* :func:`fit_parabola_to_image_observations` -- Levenberg-Marquardt fit
  of a constant-gravity parabola (p(t) = p0 + v0*t + 0.5*g*t^2) to a
  sequence of per-frame ball pixel observations, projected through the
  per-frame camera-track ``(K_t, R_t)`` and the clip-shared ``t_world``.

The seed is computed by ground-projecting the first/last image points
to a coarse plane (z = 0.5 m, mid-flight) using
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
    t_world: np.ndarray,
    fps: float,
    g: float = -9.81,
    max_iter: int = 100,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Fit a 3D parabola to per-frame image observations.

    Args:
        observations: list of ``(frame_index, (u, v))`` pairs ordered by
            time.  ``frame_index`` is the absolute clip frame, used only
            for time deltas; ``Ks`` and ``Rs`` are looked up positionally.
        Ks: position-parallel to ``observations`` — one entry per
            observation, in the same order. ``Ks[i]`` is the intrinsics
            for ``observations[i]``.
        Rs: position-parallel to ``observations`` — one entry per
            observation. ``Rs[i]`` is the rotation for ``observations[i]``.
        t_world: clip-shared world translation (3,).
        fps: frame rate.
        g: gravity along world-z (default -9.81 m/s^2).
        max_iter: LM iteration cap (passed through to scipy as
            ``max_nfev = max_iter * 50``).

    Returns:
        ``(p0, v0, mean_residual_px)`` where ``mean_residual_px`` is
        the RMS reprojection error in pixels.
    """
    from scipy.optimize import least_squares

    obs_array = np.array([o[1] for o in observations], dtype=float)
    frame_idx = np.array([o[0] for o in observations], dtype=int)
    dt = (frame_idx - frame_idx[0]) / fps
    g_vec = np.array([0.0, 0.0, g])

    def _residuals(params: np.ndarray) -> np.ndarray:
        p0 = params[:3]
        v0 = params[3:6]
        pts = p0 + np.outer(dt, v0) + 0.5 * np.outer(dt ** 2, g_vec)
        residuals = []
        for i in range(len(observations)):
            cam = Rs[i] @ pts[i] + t_world
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
        t=t_world,
        plane_z=0.5,
    )
    p_end = ankle_ray_to_pitch(
        observations[-1][1],
        K=Ks[-1],
        R=Rs[-1],
        t=t_world,
        plane_z=0.5,
    )
    duration = dt[-1] if dt[-1] > 0 else 1.0
    v_horiz = (p_end - p_start) / duration
    v0_seed = np.array([v_horiz[0], v_horiz[1], 0.5 * abs(g) * duration])
    p0_seed = p_start

    result = least_squares(
        _residuals,
        np.concatenate([p0_seed, v0_seed]),
        method="lm",
        max_nfev=max_iter * 50,
    )
    n = len(observations)
    mean_residual = float(np.linalg.norm(result.fun) / np.sqrt(n))
    return result.x[:3], result.x[3:6], mean_residual
