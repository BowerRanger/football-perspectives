"""Fixed-position PnP solver.

Option B1 of the calibration densification plan.  PnLCalib's full
calibration solver struggles on panning/close-up shots where the pitch
geometry is ambiguous, but its HRNet keypoint detector still produces
reasonable 2D landmarks on most of those frames.  Given the known
camera world position ``C`` (established from the sparse keyframes
that *do* converge), we only need to recover rotation + focal length
per frame — 4 unknowns instead of 7.

The solver works entirely in PnLCalib's internal pitch coordinate
frame (origin at pitch centre, y positive toward the top of the
broadcast image, z negative up) because the 2D → 3D keypoint mapping
comes straight from PnLCalib's ``keypoint_world_coords_2D`` table.
Results are converted to our near-left-corner / z-up frame at the
very end via the same transformation used by
:func:`src.utils.neural_calibrator.convert_pnlcalib_to_ours`.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from src.utils.neural_calibrator import convert_pnlcalib_to_ours

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PNLCALIB_ROOT = _REPO_ROOT / "third_party" / "PnLCalib"

# PnLCalib assigns zw = -2.44 to four goalpost-top keypoints (the
# crossbars), and 0 to everything else.  See utils_calib.py line 134.
_GOALPOST_TOP_KEYS = {12, 15, 16, 19}
_GOALPOST_Z = -2.44


def _load_pnlcalib_keypoint_table() -> tuple[
    dict[int, tuple[float, float, float]], dict[int, tuple[float, float, float]],
]:
    """Load PnLCalib's keypoint world-coordinate tables at module init.

    Returns ``(main_world, aux_world)`` mapping PnLCalib keypoint IDs
    (1–57 and 58–73) to ``(xw, yw, zw)`` tuples in PnLCalib's centred
    pitch frame.
    """
    src_path = str(_REPO_ROOT / "src")
    removed = [p for p in sys.path if p == src_path or p.rstrip("/") == src_path]
    for p in removed:
        sys.path.remove(p)
    sys.path.insert(0, str(_PNLCALIB_ROOT))
    try:
        from utils.utils_calib import (  # type: ignore
            keypoint_aux_world_coords_2D,
            keypoint_world_coords_2D,
        )
    finally:
        try:
            sys.path.remove(str(_PNLCALIB_ROOT))
        except ValueError:
            pass
        for p in removed:
            sys.path.append(p)

    main: dict[int, tuple[float, float, float]] = {}
    for idx, (xw, yw) in enumerate(keypoint_world_coords_2D):
        kp_id = idx + 1  # PnLCalib is 1-indexed
        zw = _GOALPOST_Z if kp_id in _GOALPOST_TOP_KEYS else 0.0
        main[kp_id] = (float(xw), float(yw), float(zw))

    aux: dict[int, tuple[float, float, float]] = {}
    for idx, (xw, yw) in enumerate(keypoint_aux_world_coords_2D):
        kp_id = idx + 1 + 57
        aux[kp_id] = (float(xw), float(yw), 0.0)
    return main, aux


_MAIN_WORLD, _AUX_WORLD = _load_pnlcalib_keypoint_table()


def _world_coords_for(kp_id: int) -> tuple[float, float, float] | None:
    """Return PnLCalib's 3D world coords for a keypoint ID, or None if unknown."""
    if kp_id in _MAIN_WORLD:
        return _MAIN_WORLD[kp_id]
    if kp_id in _AUX_WORLD:
        return _AUX_WORLD[kp_id]
    return None


@dataclass(frozen=True)
class FixedPositionResult:
    """Result of a fixed-position PnP solve, already converted to our frame."""

    K: np.ndarray     # 3x3 intrinsic
    rvec: np.ndarray  # 3-vector Rodrigues rotation (our frame)
    tvec: np.ndarray  # 3-vector translation (our frame)
    reprojection_error: float
    num_inliers: int


def solve_fixed_position(
    kp_pixels: dict[int, tuple[float, float]],
    camera_position_ours: np.ndarray,
    image_size: tuple[int, int],
    fx_init: float = 3000.0,
    max_reprojection_error: float = 5.0,
    min_inliers: int = 4,
) -> FixedPositionResult | None:
    """Solve for rotation + focal length with a fixed camera world position.

    Args:
        kp_pixels: ``{pnl_keypoint_id: (x_pixel, y_pixel)}`` from
            ``PnLCalibrator.extract_keypoints_pixels``.
        camera_position_ours: camera world position in OUR frame
            (near-left corner origin, z-up).  This is converted to
            PnLCalib's frame internally.
        image_size: ``(width, height)`` of the frame.
        fx_init: initial focal length estimate (pixels).
        max_reprojection_error: maximum per-point reprojection error
            (pixels) for an observation to be considered an inlier.
        min_inliers: minimum number of inlier observations to accept
            a solution.

    Returns:
        ``FixedPositionResult`` or ``None`` when insufficient keypoints
        or no low-error solution exists.
    """
    # Build 3D/2D correspondence arrays in PnLCalib's pitch frame.
    obj_pts: list[tuple[float, float, float]] = []
    img_pts: list[tuple[float, float]] = []
    for kp_id, (px, py) in kp_pixels.items():
        world = _world_coords_for(kp_id)
        if world is None:
            continue
        obj_pts.append(world)
        img_pts.append((float(px), float(py)))

    if len(obj_pts) < min_inliers:
        return None

    obj_np = np.array(obj_pts, dtype=np.float64)
    img_np = np.array(img_pts, dtype=np.float64)

    width, height = image_size
    cx = width / 2.0
    cy = height / 2.0

    # Convert known camera position from our frame → PnLCalib's frame:
    #   x_pnl = x_ours - 52.5
    #   y_pnl = 34 - y_ours
    #   z_pnl = -z_ours
    C_ours = np.asarray(camera_position_ours, dtype=np.float64).reshape(3)
    C_pnl = np.array(
        [C_ours[0] - 52.5, 34.0 - C_ours[1], -C_ours[2]], dtype=np.float64,
    )

    # Two-stage solve:
    # 1. For each fx candidate, run unconstrained solvePnPRansac to
    #    find an inlier keypoint subset robust to PnLCalib's occasional
    #    misidentifications.
    # 2. On the best inlier subset, run a constrained optimiser that
    #    minimises reprojection error over (rvec, fx) with
    #    tvec = -R @ C_pnl hard-enforced.
    fx_seeds = [fx_init * s for s in (0.5, 0.7, 0.85, 1.0, 1.15, 1.3, 1.5)]
    best_inliers: list[int] = []
    best_seed_fx = fx_init
    ransac_threshold = max(10.0, max_reprojection_error * 2.0)
    for fx_seed in fx_seeds:
        K_seed = np.array(
            [[fx_seed, 0, cx], [0, fx_seed, cy], [0, 0, 1]], dtype=np.float64,
        )
        try:
            ok, rvec_r, tvec_r, inliers_arr = cv2.solvePnPRansac(
                obj_np, img_np, K_seed, None,
                reprojectionError=ransac_threshold,
                iterationsCount=200,
                confidence=0.99,
                flags=cv2.SOLVEPNP_EPNP,
            )
        except cv2.error:
            continue
        if not ok or inliers_arr is None:
            continue
        inliers = sorted(int(i) for i in inliers_arr.flatten())
        if len(inliers) < min_inliers:
            continue
        # Sanity: unconstrained camera position should be at least vaguely
        # near the known C.  Reject obvious nonsense.
        R_r, _ = cv2.Rodrigues(rvec_r)
        C_r = (-R_r.T @ tvec_r).flatten()
        if float(np.linalg.norm(C_r - C_pnl)) > 100.0:
            continue
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_seed_fx = float(fx_seed)

    if len(best_inliers) < min_inliers:
        return None

    obj_in = obj_np[best_inliers]
    img_in = img_np[best_inliers]

    # Constrained refinement: optimise (rvec, fx) minimising reprojection
    # error with tvec = -R @ C_pnl fixed.
    refined = _refine_fixed_position(obj_in, img_in, C_pnl, best_seed_fx, cx, cy)
    if refined is None:
        return None
    rvec_pnl_opt, fx_opt, mean_err = refined
    if mean_err > max_reprojection_error:
        return None

    K_pnl = np.array(
        [[fx_opt, 0, cx], [0, fx_opt, cy], [0, 0, 1]], dtype=np.float64,
    )
    R_pnl, _ = cv2.Rodrigues(rvec_pnl_opt)

    # Convert (R, t) from PnLCalib's frame to ours.  The world position
    # is the known C_ours we started with; convert_pnlcalib_to_ours takes
    # the PnLCalib rotation matrix and the PnLCalib camera position.
    rvec_ours, tvec_ours, _ = convert_pnlcalib_to_ours(R_pnl, C_pnl)

    return FixedPositionResult(
        K=K_pnl.astype(np.float64),
        rvec=rvec_ours.astype(np.float64),
        tvec=tvec_ours.astype(np.float64),
        reprojection_error=float(mean_err),
        num_inliers=len(best_inliers),
    )


def _refine_fixed_position(
    obj_np: np.ndarray,
    img_np: np.ndarray,
    C_pnl: np.ndarray,
    fx_seed: float,
    cx: float,
    cy: float,
) -> tuple[np.ndarray, float, float] | None:
    """Optimise ``(rvec, fx)`` on an inlier subset with ``t = -R @ C`` fixed.

    Uses ``scipy.optimize.least_squares`` with the Levenberg–Marquardt
    solver.  The residual projects the 3D keypoints through
    ``K(fx) @ [R(rvec) | -R(rvec) @ C]``.

    Returns ``(rvec, fx, mean_reprojection_error)`` or ``None`` on
    failure.
    """
    from scipy.optimize import least_squares

    K_seed = np.array(
        [[fx_seed, 0, cx], [0, fx_seed, cy], [0, 0, 1]], dtype=np.float64,
    )
    ok, rvec_init, _tvec_init = cv2.solvePnP(
        obj_np.astype(np.float64),
        img_np.astype(np.float64),
        K_seed,
        None,
        flags=cv2.SOLVEPNP_EPNP if len(obj_np) >= 4 else cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return None

    def residual(params: np.ndarray) -> np.ndarray:
        rvec = params[:3]
        fx = params[3]
        K = np.array([[fx, 0, cx], [0, fx, cy], [0, 0, 1]], dtype=np.float64)
        R, _ = cv2.Rodrigues(rvec)
        tvec = -R @ C_pnl
        proj_h = (K @ (R @ obj_np.T + tvec[:, None])).T
        proj = proj_h[:, :2] / proj_h[:, 2:3]
        return (proj - img_np).flatten()

    x0 = np.array([rvec_init[0, 0], rvec_init[1, 0], rvec_init[2, 0], fx_seed])
    try:
        result = least_squares(residual, x0, method="lm", max_nfev=200)
    except Exception as exc:  # noqa: BLE001
        logger.debug("least_squares raised: %s", exc)
        return None

    rvec_opt = result.x[:3].astype(np.float64)
    fx_opt = float(result.x[3])
    if fx_opt < 200 or fx_opt > 20000:
        return None

    final_res = residual(result.x).reshape(-1, 2)
    per_pt = np.linalg.norm(final_res, axis=1)
    mean_err = float(np.mean(per_pt))
    return rvec_opt, fx_opt, mean_err
