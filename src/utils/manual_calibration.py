"""Manual landmark calibration solver.

Given a set of manually-annotated pitch landmarks (each is a known
3D world point ↔ a clicked 2D pixel), recover the camera ``(K, R, t)``
via PnP.  Used to anchor the calibration pipeline at user-trusted
frames when PnLCalib is unreliable.

The solver works in three regimes:

- **Fixed position** (any annotations ≥ 3 plus a known camera world
  position): lock ``tvec = -R @ C`` and refine ``rvec`` + ``fx``
  only.  This is the preferred path inside a shot whose camera
  position is already estimated by the static-camera fuser — it
  handles clustered clicks robustly because the position lever-arm
  can't drift.
- **Pose-only** (4–5 annotations with no known position): fix ``K``
  to the supplied initial value and solve for ``rvec``, ``tvec`` via
  :func:`cv2.solvePnP`.
- **Full** (6+ annotations with no known position): jointly refine
  ``rvec``, ``tvec``, and ``fx`` via :func:`scipy.optimize.least_squares`
  starting from a sensible initial guess.

In all modes the principal point is pinned to the image centre.
Returned :class:`CameraFrame` carries the recovered camera with a
sub-pixel ``reprojection_error`` for inspection in the dashboard.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

from src.schemas.calibration import CameraFrame
from src.utils.pitch import FIFA_LANDMARKS

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ManualCalibrationResult:
    """The output of :func:`solve_from_annotations`."""

    camera_frame: CameraFrame
    n_points: int
    mean_reprojection_error_px: float
    mode: str  # "fixed_position" | "pose_only" | "full"


def solve_from_annotations(
    annotations: dict[str, list[float]],
    image_size: tuple[int, int],
    *,
    fx_init: float = 3500.0,
    frame_idx: int = 0,
    camera_position_world: np.ndarray | None = None,
) -> ManualCalibrationResult | None:
    """Recover ``(K, R, t)`` from a dict of ``{landmark_name: [px, py]}``.

    Args:
        annotations: ``{landmark_name: [pixel_x, pixel_y]}`` mapping
            for the frame.  Landmark names must be keys of
            :data:`src.utils.pitch.FIFA_LANDMARKS`.
        image_size: ``(width, height)`` of the source video frame.
            Used to fix the principal point at the image centre.
        fx_init: initial guess for the focal length in pixels.  When
            we have ≥6 annotations and no known camera position, the
            solver also refines this value; with fewer annotations or
            a fixed position it is still refined but more carefully.
        frame_idx: video frame index this calibration corresponds to.
            Stored on the returned :class:`CameraFrame` for downstream
            use.
        camera_position_world: optional known camera world position
            ``C`` in pitch metres.  When provided the solver locks
            ``tvec = -R @ C`` and only refines ``(rvec, fx)``.  This
            makes clustered annotations reliable — the dominant
            failure mode of the free-``t`` path.  Minimum landmark
            count drops from 4 to 3.

    Returns:
        :class:`ManualCalibrationResult` or ``None`` when the
        annotations are insufficient (<3 in fixed mode, <4 in free
        mode) or the solver fails.
    """
    obj_pts: list[np.ndarray] = []
    img_pts: list[np.ndarray] = []
    used_names: list[str] = []

    for name, pixel in annotations.items():
        world = FIFA_LANDMARKS.get(name)
        if world is None:
            logger.debug("manual_calibration: unknown landmark %s", name)
            continue
        if not isinstance(pixel, (list, tuple)) or len(pixel) != 2:
            continue
        try:
            px = float(pixel[0])
            py = float(pixel[1])
        except (TypeError, ValueError):
            continue
        obj_pts.append(np.asarray(world, dtype=np.float64))
        img_pts.append(np.array([px, py], dtype=np.float64))
        used_names.append(name)

    min_points = 3 if camera_position_world is not None else 4
    if len(obj_pts) < min_points:
        return None

    obj = np.array(obj_pts, dtype=np.float64)
    img = np.array(img_pts, dtype=np.float64)

    width, height = image_size
    cx = float(width) / 2.0
    cy = float(height) / 2.0

    # ── Fixed-position path ──
    if camera_position_world is not None:
        C = np.asarray(camera_position_world, dtype=np.float64).reshape(3)
        refined = _refine_with_fixed_position(obj, img, C, fx_init, cx, cy)
        if refined is None:
            return None
        rvec_opt, tvec_opt, fx_opt, mean_err = refined
        K_final = _build_K(fx_opt, cx, cy)
        cf = CameraFrame(
            frame=frame_idx,
            intrinsic_matrix=K_final.tolist(),
            rotation_vector=rvec_opt.flatten().tolist(),
            translation_vector=tvec_opt.flatten().tolist(),
            reprojection_error=float(mean_err),
            num_correspondences=len(obj),
            confidence=1.0,
            tracked_landmark_types=list(used_names),
        )
        return ManualCalibrationResult(
            camera_frame=cf,
            n_points=len(obj),
            mean_reprojection_error_px=float(mean_err),
            mode="fixed_position",
        )

    # ── Free-position path (existing behaviour) ──
    K_init = np.array(
        [[fx_init, 0.0, cx],
         [0.0, fx_init, cy],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )

    # Stage 1: solvePnP for an initial pose.  Use ITERATIVE which
    # works well with a planar-ish landmark set.
    try:
        ok, rvec, tvec = cv2.solvePnP(
            obj, img, K_init, None,
            flags=cv2.SOLVEPNP_ITERATIVE if len(obj) >= 4 else cv2.SOLVEPNP_AP3P,
        )
    except cv2.error as exc:
        logger.debug("manual_calibration: solvePnP raised %s", exc)
        return None
    if not ok:
        return None

    # Stage 2: if we have 6+ points, refine fx jointly with pose.
    # Below 6 points the system is too under-determined to learn fx
    # reliably; keep fx fixed.
    n = len(obj)
    if n >= 6:
        refined = _refine_with_focal(obj, img, K_init, rvec, tvec, cx, cy)
        if refined is not None:
            rvec, tvec, fx_refined, mean_err = refined
            mode = "full"
        else:
            fx_refined = fx_init
            mean_err = _mean_reprojection_error(
                obj, img, _build_K(fx_init, cx, cy), rvec, tvec,
            )
            mode = "pose_only"
    else:
        fx_refined = fx_init
        mean_err = _mean_reprojection_error(
            obj, img, K_init, rvec, tvec,
        )
        mode = "pose_only"

    K_final = _build_K(fx_refined, cx, cy)
    cf = CameraFrame(
        frame=frame_idx,
        intrinsic_matrix=K_final.tolist(),
        rotation_vector=rvec.flatten().tolist(),
        translation_vector=tvec.flatten().tolist(),
        reprojection_error=float(mean_err),
        num_correspondences=n,
        confidence=1.0,
        tracked_landmark_types=list(used_names),
    )
    return ManualCalibrationResult(
        camera_frame=cf,
        n_points=n,
        mean_reprojection_error_px=float(mean_err),
        mode=mode,
    )


def _refine_with_fixed_position(
    obj_pts: np.ndarray,
    img_pts: np.ndarray,
    C: np.ndarray,
    fx_seed: float,
    cx: float, cy: float,
) -> tuple[np.ndarray, np.ndarray, float, float] | None:
    """Joint LM refinement of ``(rvec, fx)`` with ``tvec = -R @ C`` fixed.

    Mirrors the pattern in
    :func:`src.utils.fixed_position_solver._refine_fixed_position` but
    works directly in our near-left-corner pitch frame (no PnLCalib
    frame conversion).  Seeds ``rvec`` from a geometric "look at
    pitch centre from C" guess and, when ``len(obj) >= 4``, also
    from :func:`cv2.solvePnP`, then picks whichever gives the lower
    initial residual before LM.

    Returns ``(rvec_opt, tvec_opt, fx_opt, mean_err_px)`` or
    ``None`` on failure.
    """
    from scipy.optimize import least_squares

    def residual(params: np.ndarray) -> np.ndarray:
        rvec = params[:3]
        fx = float(params[3])
        K = _build_K(fx, cx, cy)
        R, _ = cv2.Rodrigues(rvec)
        tvec = -R @ C
        proj_h = (K @ (R @ obj_pts.T + tvec[:, None])).T
        proj = proj_h[:, :2] / proj_h[:, 2:3]
        return (proj - img_pts).flatten()

    candidates: list[np.ndarray] = []

    # Candidate 1: geometric "look at pitch centre" seed
    forward = np.array([52.5, 34.0, 0.0], dtype=np.float64) - C
    if float(forward @ forward) > 1e-6:
        forward /= np.linalg.norm(forward)
        world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        right = np.cross(forward, world_up)
        if float(right @ right) > 1e-6:
            right /= np.linalg.norm(right)
            up = np.cross(right, forward)
            R_init = np.stack([right, -up, forward], axis=0)
            rvec_geom, _ = cv2.Rodrigues(R_init)
            candidates.append(
                np.array([rvec_geom[0, 0], rvec_geom[1, 0],
                          rvec_geom[2, 0], float(fx_seed)],
                         dtype=np.float64)
            )

    # Candidate 2: cv2.solvePnP when we have enough points for it.
    # Even when free-t gives a bad global pose, its rvec is often
    # close enough to bootstrap the fixed-t LM.
    if len(obj_pts) >= 4:
        K_seed = _build_K(fx_seed, cx, cy)
        try:
            ok, rvec_pnp, _tvec_pnp = cv2.solvePnP(
                obj_pts, img_pts, K_seed, None,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            if ok:
                candidates.append(
                    np.array([rvec_pnp[0, 0], rvec_pnp[1, 0],
                              rvec_pnp[2, 0], float(fx_seed)],
                             dtype=np.float64)
                )
        except cv2.error as exc:
            logger.debug("fixed_position seed solvePnP raised: %s", exc)

    if not candidates:
        return None

    # Rank candidates by initial residual before LM
    candidates.sort(key=lambda p: float(np.mean(residual(p) ** 2)))

    best: tuple[np.ndarray, float] | None = None
    for x0 in candidates:
        try:
            result = least_squares(residual, x0, method="lm", max_nfev=300)
        except Exception as exc:  # noqa: BLE001
            logger.debug("fixed_position LM raised: %s", exc)
            continue
        final_res = residual(result.x).reshape(-1, 2)
        mean_err = float(np.mean(np.linalg.norm(final_res, axis=1)))
        if best is None or mean_err < best[1]:
            best = (result.x.astype(np.float64), mean_err)

    if best is None:
        return None
    params_opt, mean_err = best
    fx_opt = float(params_opt[3])
    if fx_opt < 200 or fx_opt > 20000:
        return None

    rvec_opt = params_opt[:3].reshape(3, 1)
    R_opt, _ = cv2.Rodrigues(rvec_opt)
    tvec_opt = (-R_opt @ C).reshape(3, 1)
    return rvec_opt, tvec_opt, fx_opt, mean_err


def _build_K(fx: float, cx: float, cy: float) -> np.ndarray:
    return np.array(
        [[fx, 0.0, cx],
         [0.0, fx, cy],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _mean_reprojection_error(
    obj_pts: np.ndarray,
    img_pts: np.ndarray,
    K: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
) -> float:
    proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, None)
    proj = proj.reshape(-1, 2)
    return float(np.mean(np.linalg.norm(proj - img_pts, axis=1)))


def _refine_with_focal(
    obj_pts: np.ndarray,
    img_pts: np.ndarray,
    K_seed: np.ndarray,
    rvec_seed: np.ndarray,
    tvec_seed: np.ndarray,
    cx: float, cy: float,
) -> tuple[np.ndarray, np.ndarray, float, float] | None:
    """Joint LM refinement of ``(rvec, tvec, fx)`` from a PnP seed."""
    from scipy.optimize import least_squares

    def residual(params: np.ndarray) -> np.ndarray:
        rv = params[:3]
        tv = params[3:6]
        fx = float(params[6])
        K = _build_K(fx, cx, cy)
        proj, _ = cv2.projectPoints(obj_pts, rv, tv, K, None)
        return (proj.reshape(-1, 2) - img_pts).flatten()

    x0 = np.concatenate([
        rvec_seed.flatten(),
        tvec_seed.flatten(),
        [float(K_seed[0, 0])],
    ])
    try:
        result = least_squares(residual, x0, method="lm", max_nfev=300)
    except Exception as exc:  # noqa: BLE001
        logger.debug("manual_calibration LM raised: %s", exc)
        return None

    rvec = result.x[:3].astype(np.float64).reshape(3, 1)
    tvec = result.x[3:6].astype(np.float64).reshape(3, 1)
    fx = float(result.x[6])
    if fx < 200 or fx > 20000:
        return None

    final_res = residual(result.x).reshape(-1, 2)
    mean_err = float(np.mean(np.linalg.norm(final_res, axis=1)))
    return rvec, tvec, fx, mean_err
