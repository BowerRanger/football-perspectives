"""Static-camera bundle adjustment from detected painted lines.

Solves one fixed camera centre ``C`` across every frame while pan/tilt
(``rvec``) and zoom (``fx``) vary per frame. The sub-pixel painted-line
observations from ``line_detector.py`` provide the constraints; optional
low-weight point-landmark hints catch gross basin errors without setting
the gauge (the lines' ``world_segment``s already fix it).

This is the reusable core behind ``scripts/global_solve_from_lines.py``
and the camera stage's static-C line path. Unlike the legacy global
solve it has NO per-frame motion budget — ``C`` is strictly one
3-vector.

Lens model:
  - ``pinhole_k1k2``  — shared ``(cx, cy, k1, k2)``; ``p1=p2=k3=0`` fixed.
  - ``brown_conrady`` — shared ``(cx, cy, k1, k2, p1, p2, k3)``.
Both feed a 5-element OpenCV distortion vector into the line residual.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.optimize._numdiff import approx_derivative
from scipy.sparse import lil_matrix

from src.schemas.anchor import LandmarkObservation, LineObservation
from src.utils.anchor_solver import _make_K, _point_residuals_distorted

logger = logging.getLogger(__name__)

LensModel = Literal["pinhole_k1k2", "brown_conrady"]

# How many distortion coefficients are *free parameters* per lens model.
_N_FREE_DIST: dict[str, int] = {"pinhole_k1k2": 2, "brown_conrady": 5}


@dataclass(frozen=True)
class StaticCameraSolution:
    """Result of a static-camera line solve.

    ``camera_centre`` is the single locked C; every entry in
    ``per_frame_KRt`` satisfies ``-R.T @ t == camera_centre``.
    """

    camera_centre: np.ndarray                  # (3,)
    principal_point: tuple[float, float]       # (cx, cy)
    distortion: tuple[float, ...]              # (k1, k2) or (k1, k2, p1, p2, k3)
    per_frame_KRt: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]
    per_frame_line_rms: dict[int, float]
    lens_model: LensModel

    @property
    def line_rms_mean(self) -> float:
        vals = [v for v in self.per_frame_line_rms.values() if np.isfinite(v)]
        return float(np.mean(vals)) if vals else float("nan")


def _dist5(distortion: tuple[float, ...]) -> np.ndarray:
    """Pad a (k1,k2) or (k1,k2,p1,p2,k3) tuple to OpenCV's 5-vector."""
    out = np.zeros(5, dtype=np.float64)
    out[: len(distortion)] = distortion
    return out


def _project_points_distorted(
    K: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    dist5: np.ndarray,
    world_points: np.ndarray,
) -> np.ndarray:
    """Project (N,3) world points to (N,2) pixels with a 5-element
    OpenCV distortion vector."""
    pts = np.asarray(world_points, dtype=np.float64).reshape(-1, 1, 3)
    out, _ = cv2.projectPoints(
        pts,
        np.asarray(rvec, dtype=np.float64).reshape(3, 1),
        np.asarray(tvec, dtype=np.float64).reshape(3, 1),
        K.astype(np.float64),
        dist5,
    )
    return out.reshape(-1, 2)


def _line_residuals_distorted(
    lines: list[LineObservation],
    K: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    dist5: np.ndarray,
) -> np.ndarray:
    """2 residuals per position-known line: perpendicular pixel distance
    from each detected image endpoint to the distortion-projected world
    line. Direction-only lines are not produced by the painted-line
    detector, so they get a large finite sentinel."""
    out = np.empty(2 * len(lines))
    R = cv2.Rodrigues(np.asarray(rvec, np.float64))[0]
    for i, ln in enumerate(lines):
        if ln.world_segment is None:
            out[2 * i] = out[2 * i + 1] = 1e6
            continue
        world = np.array(
            [ln.world_segment[0], ln.world_segment[1]], dtype=np.float64
        )
        # Behind-camera guard.
        cam = world @ R.T + tvec
        if (cam[:, 2] <= 1e-3).all():
            out[2 * i] = out[2 * i + 1] = 1e6
            continue
        proj = _project_points_distorted(K, rvec, tvec, dist5, world)
        pa, pb = proj[0], proj[1]
        d = pb - pa
        norm = float(np.linalg.norm(d))
        if norm < 1e-6:
            out[2 * i] = out[2 * i + 1] = 1e6
            continue
        nx, ny = -d[1] / norm, d[0] / norm
        c = -(nx * pa[0] + ny * pa[1])
        for j, (u, v) in enumerate(ln.image_segment):
            out[2 * i + j] = nx * u + ny * v + c
    return out


def solve_static_camera_from_lines(
    per_frame_lines: dict[int, list[LineObservation]],
    image_size: tuple[int, int],
    *,
    c_seed: np.ndarray,
    lens_seed: tuple[float, float, float, float],
    per_frame_seeds: dict[int, tuple[np.ndarray, float]],
    point_hints: dict[int, list[LandmarkObservation]] | None = None,
    lens_model: LensModel = "pinhole_k1k2",
    point_hint_weight: float = 0.05,
    max_nfev: int = 600,
) -> StaticCameraSolution:
    """Solve one fixed camera centre across all frames in
    ``per_frame_lines``.

    Parameters
    ----------
    per_frame_lines
        frame id -> list of position-known ``LineObservation``.
    image_size
        ``(width, height)`` — used only for ``(cx, cy)`` bounds.
    c_seed
        Initial guess for the shared camera centre (3,). The C-profile
        diagnostic supplies the good seed; a poor seed risks a local
        minimum.
    lens_seed
        ``(cx, cy, k1, k2)`` initial lens guess. ``p1, p2, k3`` seed at 0.
    per_frame_seeds
        frame id -> ``(rvec, fx)`` initial pose/zoom guess. Required for
        every frame in ``per_frame_lines``.
    point_hints
        Optional frame id -> landmark list, added at ``point_hint_weight``
        as a basin regulariser. Does NOT set the gauge.
    lens_model
        ``pinhole_k1k2`` (default) or ``brown_conrady``.
    """
    W, H = image_size
    fids = sorted(per_frame_lines.keys())
    if not fids:
        raise ValueError("solve_static_camera_from_lines: no frames given")

    n_dist = _N_FREE_DIST[lens_model]
    SHARED = 2 + n_dist + 3      # cx, cy, dist..., Cx, Cy, Cz
    PER = 4                      # rvec(3), fx
    n = len(fids)

    cx_s, cy_s, k1_s, k2_s = lens_seed
    p0 = np.empty(SHARED + PER * n)
    lower = np.empty_like(p0)
    upper = np.empty_like(p0)

    p0[0], p0[1] = cx_s, cy_s
    lower[0], upper[0] = W / 2 - 150, W / 2 + 150
    lower[1], upper[1] = H / 2 - 150, H / 2 + 150
    # distortion seeds + bounds
    dist_seed = [k1_s, k2_s, 0.0, 0.0, 0.0][:n_dist]
    dist_lo = [-0.5, -0.5, -0.1, -0.1, -0.5][:n_dist]
    dist_hi = [0.5, 0.5, 0.1, 0.1, 0.5][:n_dist]
    for j in range(n_dist):
        p0[2 + j] = dist_seed[j]
        lower[2 + j] = dist_lo[j]
        upper[2 + j] = dist_hi[j]
    c_base = 2 + n_dist
    p0[c_base : c_base + 3] = c_seed
    lower[c_base : c_base + 3] = np.asarray(c_seed) - 5.0
    upper[c_base : c_base + 3] = np.asarray(c_seed) + 5.0

    for i, fid in enumerate(fids):
        if fid not in per_frame_seeds:
            raise ValueError(f"per_frame_seeds missing frame {fid}")
        rvec0, fx0 = per_frame_seeds[fid]
        base = SHARED + i * PER
        p0[base : base + 3] = np.asarray(rvec0, dtype=np.float64).reshape(3)
        p0[base + 3] = fx0
        lower[base : base + 3] = -np.pi
        upper[base : base + 3] = np.pi
        lower[base + 3] = fx0 * 0.5
        upper[base + 3] = fx0 * 2.0

    hints = point_hints or {}

    def _unpack_shared(p: np.ndarray):
        cx, cy = float(p[0]), float(p[1])
        dist = tuple(float(p[2 + j]) for j in range(n_dist))
        C = p[c_base : c_base + 3]
        return cx, cy, dist, C

    def residuals(p: np.ndarray) -> np.ndarray:
        cx, cy, dist, C = _unpack_shared(p)
        d5 = _dist5(dist)
        parts: list[np.ndarray] = []
        for i, fid in enumerate(fids):
            base = SHARED + i * PER
            rvec = p[base : base + 3]
            fx = float(np.clip(p[base + 3], 50.0, 1e5))
            R_i, _ = cv2.Rodrigues(rvec)
            t_i = -R_i @ C
            K_i = _make_K(fx, cx, cy)
            parts.append(
                _line_residuals_distorted(per_frame_lines[fid], K_i, rvec, t_i, d5)
            )
            hint = hints.get(fid)
            if hint:
                parts.append(
                    point_hint_weight
                    * _point_residuals_distorted(
                        hint, K_i, rvec, t_i, (dist[0], dist[1])
                    )
                )
        return np.concatenate(parts) if parts else np.empty(0)

    # Sparse Jacobian: each frame's residuals touch SHARED cols + its PER cols.
    n_res_per_frame = []
    for fid in fids:
        n_res = 2 * len(per_frame_lines[fid])
        if fid in hints:
            n_res += 2 * len(hints[fid])
        n_res_per_frame.append(n_res)
    total_res = sum(n_res_per_frame)
    total_par = SHARED + PER * n
    spar = lil_matrix((total_res, total_par), dtype=np.uint8)
    row = 0
    for i, n_res in enumerate(n_res_per_frame):
        base = SHARED + i * PER
        for jr in range(n_res):
            spar[row + jr, 0:SHARED] = 1
            spar[row + jr, base : base + PER] = 1
        row += n_res

    spar_csr = spar.tocsr()

    def jac(p: np.ndarray) -> np.ndarray:
        # Grouped sparse finite-difference Jacobian, returned *dense* so the
        # 'exact' trust-region solver can be used. Passing ``jac_sparsity``
        # to ``least_squares`` directly forces scipy onto the 'lsmr'
        # iterative trust-region solver, which fails to converge on this
        # block-arrow problem structure (verified: hits max_nfev at
        # multi-pixel error even on clean synthetic data). The grouped FD
        # itself is cheap — ~(SHARED + PER) evaluations regardless of the
        # frame count — so the dense solver pays no Jacobian-cost penalty.
        J = approx_derivative(
            residuals, p, method="2-point",
            sparsity=spar_csr, bounds=(lower, upper),
        )
        return J.toarray() if hasattr(J, "toarray") else np.asarray(J)

    # Tolerances are 1e-8, not 1e-12: residuals here are *pixels* and the
    # target is sub-pixel, so chasing 1e-12 is chasing numerical noise. On
    # data the model can't fully fit (a hard anchor set → noisy detected
    # lines) the tighter stop condition is never satisfied and the LM
    # grinds toward max_nfev — each step a dense SVD of a large Jacobian,
    # so a non-convergent solve runs for hours. 1e-8 + a modest max_nfev
    # let a hard solve bail in minutes with a usable result; a convergent
    # solve (~60-90 nfev on validation clips) is unaffected.
    result = least_squares(
        residuals, p0, jac=jac, bounds=(lower, upper),
        method="trf", loss="huber", f_scale=2.0,
        tr_solver="exact", max_nfev=max_nfev,
        xtol=1e-8, ftol=1e-8, gtol=1e-8,
    )

    cx, cy, dist, C = _unpack_shared(result.x)
    d5 = _dist5(dist)
    per_frame_KRt: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    per_frame_line_rms: dict[int, float] = {}
    for i, fid in enumerate(fids):
        base = SHARED + i * PER
        rvec = result.x[base : base + 3]
        fx = float(result.x[base + 3])
        R_i, _ = cv2.Rodrigues(rvec)
        t_i = -R_i @ np.asarray(C)
        K_i = _make_K(fx, cx, cy)
        per_frame_KRt[fid] = (K_i, R_i, t_i)
        r = _line_residuals_distorted(per_frame_lines[fid], K_i, rvec, t_i, d5)
        per_frame_line_rms[fid] = (
            float(np.sqrt((r ** 2).mean())) if r.size else float("nan")
        )

    return StaticCameraSolution(
        camera_centre=np.asarray(C, dtype=np.float64).copy(),
        principal_point=(cx, cy),
        distortion=tuple(dist),
        per_frame_KRt=per_frame_KRt,
        per_frame_line_rms=per_frame_line_rms,
        lens_model=lens_model,
    )
