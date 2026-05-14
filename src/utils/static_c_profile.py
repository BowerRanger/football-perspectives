"""Camera-centre profile diagnostic.

Sweeps candidate static camera centres on a 3-D grid. At each candidate
``C``, every frame's ``(rvec, fx)`` is solved independently with ``C``
pinned; the per-frame line RMS is aggregated to mean / P95 / max as a
function of ``C``.

Two jobs in one:
  * **seed-finder** — the argmin ``C`` (and the per-frame ``(rvec, fx)``
    at it) seed the static-camera bundle adjustment.
  * **honesty check** — if no grid ``C`` gets the mean RMS sub-pixel,
    a single static camera genuinely cannot fit the detected lines under
    the current lens model.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np
from scipy.optimize import least_squares

from src.schemas.anchor import LineObservation
from src.utils.anchor_solver import _make_K
from src.utils.static_line_solver import _dist5, _line_residuals_distorted

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CProfileResult:
    """Output of :func:`profile_camera_centre`."""

    grid_points: np.ndarray                                   # (M, 3)
    mean_rms: np.ndarray                                      # (M,)
    p95_rms: np.ndarray                                       # (M,)
    max_rms: np.ndarray                                       # (M,)
    argmin_c: np.ndarray                                      # (3,)
    per_frame_seeds: dict[int, tuple[np.ndarray, float]]      # at argmin_c


def make_c_grid(
    c_center: np.ndarray, *, extent_m: float, n_steps: int
) -> np.ndarray:
    """Build an (n_steps**3, 3) grid of candidate centres spanning
    ``c_center +/- extent_m`` on each axis. ``n_steps`` should be odd so
    ``c_center`` itself is on the grid."""
    c_center = np.asarray(c_center, dtype=np.float64)
    axis = np.linspace(-extent_m, extent_m, n_steps)
    gx, gy, gz = np.meshgrid(axis, axis, axis, indexing="ij")
    offsets = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
    return c_center[None, :] + offsets


def _solve_frame_at_fixed_c(
    lines: list[LineObservation],
    cx: float,
    cy: float,
    dist5: np.ndarray,
    C: np.ndarray,
    rvec_seed: np.ndarray,
    fx_seed: float,
) -> tuple[np.ndarray, float, float]:
    """LM-solve one frame's (rvec, fx) with C pinned. Returns
    ``(rvec, fx, line_rms)``."""

    def res(p: np.ndarray) -> np.ndarray:
        rvec = p[0:3]
        fx = float(np.clip(p[3], 50.0, 1e5))
        R, _ = cv2.Rodrigues(rvec)
        t = -R @ C
        K = _make_K(fx, cx, cy)
        return _line_residuals_distorted(lines, K, rvec, t, dist5)

    p0 = np.array([*np.asarray(rvec_seed, float).reshape(3), float(fx_seed)])
    lower = np.array([-np.pi, -np.pi, -np.pi, fx_seed * 0.5])
    upper = np.array([np.pi, np.pi, np.pi, fx_seed * 2.0])
    result = least_squares(
        res, p0, bounds=(lower, upper),
        method="trf", loss="huber", f_scale=2.0, max_nfev=80,
    )
    rvec = result.x[0:3]
    fx = float(result.x[3])
    R, _ = cv2.Rodrigues(rvec)
    t = -R @ C
    K = _make_K(fx, cx, cy)
    r = _line_residuals_distorted(lines, K, rvec, t, dist5)
    rms = float(np.sqrt((r ** 2).mean())) if r.size else float("nan")
    return rvec, fx, rms


def profile_camera_centre(
    per_frame_lines: dict[int, list[LineObservation]],
    image_size: tuple[int, int],
    *,
    c_grid: np.ndarray,
    lens_seed: tuple[float, float, float, float],
    per_frame_bootstrap: dict[int, tuple[np.ndarray, float]],
    max_grid_frames: int = 80,
) -> CProfileResult:
    """Profile line-fitting RMS as a function of the static camera
    centre over ``c_grid``.

    ``lens_seed`` is ``(cx, cy, k1, k2)`` held fixed throughout the
    profile (the profile answers "where is C", not "what is the lens").
    ``per_frame_bootstrap`` provides the per-frame ``(rvec, fx)`` seeds
    for the inner solves.

    The grid sweep runs on an evenly-spaced subsample of at most
    ``max_grid_frames`` frames — the line-RMS-vs-C surface is smooth, so
    a representative pan/tilt spread locates the argmin just as well as
    the full set at a fraction of the cost. Once the argmin is found, a
    final pass re-solves *every* input frame at it, so the returned
    ``per_frame_seeds`` covers all frames (the bundle adjustment needs a
    seed per frame).
    """
    cx, cy, k1, k2 = lens_seed
    dist5 = _dist5((k1, k2))
    fids = sorted(per_frame_lines.keys())

    if len(fids) > max_grid_frames:
        idx = np.linspace(0, len(fids) - 1, max_grid_frames).round().astype(int)
        grid_fids = sorted({fids[i] for i in idx})
    else:
        grid_fids = fids

    m = len(c_grid)
    mean_rms = np.full(m, np.inf)
    p95_rms = np.full(m, np.inf)
    max_rms = np.full(m, np.inf)

    # Warm-start each grid cell from the previous cell's solution so the
    # inner LMs converge fast; first cell uses the bootstrap.
    warm = {f: per_frame_bootstrap[f] for f in grid_fids}
    best = 0
    best_mean = np.inf
    best_cell_seeds: dict[int, tuple[np.ndarray, float]] = dict(warm)
    for gi, C in enumerate(c_grid):
        rms_vals = []
        cell_seeds: dict[int, tuple[np.ndarray, float]] = {}
        for fid in grid_fids:
            rvec_seed, fx_seed = warm[fid]
            rvec, fx, rms = _solve_frame_at_fixed_c(
                per_frame_lines[fid], cx, cy, dist5, np.asarray(C, float),
                rvec_seed, fx_seed,
            )
            cell_seeds[fid] = (rvec, fx)
            rms_vals.append(rms)
        arr = np.array(rms_vals, dtype=np.float64)
        finite = arr[np.isfinite(arr)]
        if finite.size:
            mean_rms[gi] = float(finite.mean())
            p95_rms[gi] = float(np.percentile(finite, 95))
            max_rms[gi] = float(finite.max())
        if mean_rms[gi] < best_mean:
            best_mean = mean_rms[gi]
            best = gi
            best_cell_seeds = cell_seeds
        warm = cell_seeds  # warm-start the next cell

    argmin_c = np.asarray(c_grid[best], dtype=np.float64).copy()

    # Re-seed over ALL frames at the argmin C. Subsample frames reuse the
    # (rvec, fx) already solved at the best grid cell; the remaining
    # frames are solved now from their bootstrap.
    per_frame_seeds: dict[int, tuple[np.ndarray, float]] = dict(best_cell_seeds)
    for fid in fids:
        if fid in per_frame_seeds:
            continue
        rvec_seed, fx_seed = per_frame_bootstrap[fid]
        rvec, fx, _rms = _solve_frame_at_fixed_c(
            per_frame_lines[fid], cx, cy, dist5, argmin_c, rvec_seed, fx_seed,
        )
        per_frame_seeds[fid] = (rvec, fx)

    return CProfileResult(
        grid_points=np.asarray(c_grid, dtype=np.float64),
        mean_rms=mean_rms,
        p95_rms=p95_rms,
        max_rms=max_rms,
        argmin_c=argmin_c,
        per_frame_seeds=per_frame_seeds,
    )
