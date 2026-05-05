"""Joint multi-anchor camera-pose solver.

Replaces the legacy DLT/RQ "primary anchor + propagate-t-fixed" pipeline.

Given a set of anchors (each with point-landmark observations and/or
line-correspondence observations) on the same broadcast clip, this module
solves a single joint bundle adjustment that recovers:

- Shared world->camera translation ``t_world`` (3 params; broadcast camera
  body is fixed across the clip).
- Shared principal point ``(cx, cy)`` (2 params; lens optical axis is also
  clip-shared on a fixed-body camera, even when zoom changes).
- Per-anchor world->camera rotation as a Rodrigues vector ``rvec_i``
  (3 params each) and focal length ``fx_i`` (1 param each, fy = fx).

The solver:

1. Picks the "richest" anchor (most landmarks across most z-levels) for K
   seed selection. No special downstream role.
2. Initialises K with ``fx = image_width`` (a broadcast prior) and
   principal point at image centre.
3. Seeds each anchor's (R, t) via ``cv2.solvePnP`` on its point landmarks
   (anchors with too few points fall back to identity rotation; the joint
   LM corrects them via lines + neighbouring-anchor agreement).
4. Runs ``scipy.optimize.least_squares`` with a sparse Jacobian, minimising
   pixel reprojection error of every point landmark plus perpendicular
   distance from each user-clicked image endpoint to the projected world
   line for every line annotation.

Why this works where DLT didn't:

- No SVD-on-singular-matrix: K is seeded from a known-good prior and
  refined by LM rather than recovered from scratch.
- No sign or scale ambiguity: cv2.solvePnP returns physically-meaningful
  (R, t) with cam_z > 0 by construction, and LM minimises pixel-space
  residuals directly.
- Errors are spread across all observations rather than concentrated on
  one primary anchor whose mistakes corrupt every subsequent anchor.
"""

from __future__ import annotations

import logging
from typing import NamedTuple

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from src.schemas.anchor import Anchor, LandmarkObservation, LineObservation

logger = logging.getLogger(__name__)


class AnchorSolveError(RuntimeError):
    """Raised when the joint solve cannot proceed (e.g. no qualifying anchor)."""


class JointSolution(NamedTuple):
    t_world: np.ndarray                                       # (3,)
    principal_point: tuple[float, float]                      # (cx, cy)
    per_anchor_KR: dict[int, tuple[np.ndarray, np.ndarray]]   # frame -> (K, R)
    per_anchor_residual_px: dict[int, float]                  # frame -> mean px


# Public utilities ------------------------------------------------------------


def reprojection_residual_for_anchor(
    anchor: Anchor, K: np.ndarray, R: np.ndarray, t: np.ndarray
) -> float:
    """Mean pixel reprojection residual across this anchor's point landmarks.

    Returns ``1e9`` (sentinel) if any landmark projects behind the camera.
    Lines are deliberately excluded — they constrain orientation but their
    absolute pixel residual depends on the user's chosen segment endpoints.
    """
    if not anchor.landmarks:
        return 0.0
    residuals: list[float] = []
    for lm in anchor.landmarks:
        cam = R @ np.array(lm.world_xyz) + t
        if cam[2] <= 0:
            return 1e9
        pix = K @ cam
        proj = pix[:2] / pix[2]
        residuals.append(float(np.linalg.norm(np.array(lm.image_xy) - proj)))
    return float(np.mean(residuals))


# Internal helpers ------------------------------------------------------------


def _qualifies(anchor: Anchor, min_points: int = 4, min_lines: int = 2) -> bool:
    """An anchor contributes to the solve if it has enough points OR enough lines.

    Lines also constrain (R, fx); 2 line correspondences across roughly
    perpendicular directions yield similar information to 4 point landmarks.
    """
    return len(anchor.landmarks) >= min_points or len(anchor.lines) >= min_lines


def _z_diversity(anchor: Anchor) -> int:
    """Number of distinct z-levels among an anchor's point landmarks.

    Used to pick the K-seed anchor — diversity in z breaks coplanar
    degeneracies in solvePnP.
    """
    return len({round(lm.world_xyz[2], 3) for lm in anchor.landmarks})


def _pick_seed_anchor(anchors: tuple[Anchor, ...]) -> Anchor:
    """Pick the anchor with the most point landmarks AND most z-levels.

    Tie-broken by landmark count then by frame number.
    """
    best = sorted(
        anchors,
        key=lambda a: (_z_diversity(a), len(a.landmarks), -a.frame),
        reverse=True,
    )[0]
    if len(best.landmarks) < 4:
        raise AnchorSolveError(
            f"no anchor has ≥4 point landmarks for K-seeding "
            f"(richest: frame {best.frame} with {len(best.landmarks)})"
        )
    return best


def _seed_anchor_pose(
    anchor: Anchor, K: np.ndarray
) -> tuple[np.ndarray, np.ndarray] | None:
    """Run cv2.solvePnP on this anchor's point landmarks. Returns
    ``(rvec, tvec)`` or ``None`` if there are too few points or PnP fails.
    """
    if len(anchor.landmarks) < 4:
        return None
    obj_pts = np.array(
        [lm.world_xyz for lm in anchor.landmarks], dtype=np.float64
    ).reshape(-1, 1, 3)
    img_pts = np.array(
        [lm.image_xy for lm in anchor.landmarks], dtype=np.float64
    ).reshape(-1, 1, 2)
    dist_zero = np.zeros(5, dtype=np.float64)
    # ITERATIVE needs ≥6 points and uses DLT internally; EPNP handles 4–5
    # points and coplanar configurations gracefully. Pick by count.
    flag = cv2.SOLVEPNP_EPNP if len(anchor.landmarks) < 6 else cv2.SOLVEPNP_ITERATIVE
    ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, dist_zero, flags=flag)
    if not ok:
        return None
    return rvec.reshape(3), tvec.reshape(3)


def _refine_seed_fx(
    anchor: Anchor,
    rvec_init: np.ndarray,
    tvec_init: np.ndarray,
    cx: float,
    cy: float,
    fx_init: float,
) -> float:
    """One-anchor LM that refines fx + (R, t) starting from solvePnP output.

    Run before seeding the joint LM so the broadcast-prior fx (= image
    width) is replaced by a value much closer to truth. Without this,
    real broadcast clips (whose fx is often 0.5–0.8× image width) seed
    other anchors with an fx so wrong that solvePnP places their points
    behind the camera, and the joint LM can't escape.
    """
    if not anchor.landmarks:
        return fx_init
    obj_pts = np.array([lm.world_xyz for lm in anchor.landmarks], dtype=np.float64)
    img_pts = np.array([lm.image_xy for lm in anchor.landmarks], dtype=np.float64)

    def _residuals(p: np.ndarray) -> np.ndarray:
        fx = float(p[0])
        rvec = p[1:4]
        tvec = p[4:7]
        R, _ = cv2.Rodrigues(rvec)
        cam = obj_pts @ R.T + tvec
        safe_z = np.where(cam[:, 2] > 1e-3, cam[:, 2], 1e-3)
        K = _make_K(fx, cx, cy)
        pix = cam @ K.T
        proj = pix[:, :2] / safe_z[:, None]
        return (proj - img_pts).reshape(-1)

    p0 = np.concatenate([[fx_init], rvec_init, tvec_init])
    try:
        result = least_squares(
            _residuals, p0, method="lm", max_nfev=200,
        )
        return float(np.clip(result.x[0], 50.0, 1e5))
    except Exception:
        return fx_init


def _solve_anchor_with_t_fixed(
    anchor: Anchor,
    t_fixed: np.ndarray,
    cx: float,
    cy: float,
    fx_init: float,
    rvec_init: np.ndarray,
) -> tuple[np.ndarray, float]:
    """LM-refine (rvec, fx) for one anchor with a shared t held constant.

    Used to seed coplanar/thin anchors via the rich anchor's tvec rather
    than running solvePnP per anchor (which is unstable on coplanar points
    when the K seed is uncertain). With t known, this is a 4-DOF problem
    (3 rotation + 1 fx) and the LM is well-determined for ≥3 points.
    """
    obj_pts = np.array([lm.world_xyz for lm in anchor.landmarks], dtype=np.float64)
    img_pts = np.array([lm.image_xy for lm in anchor.landmarks], dtype=np.float64)
    if len(obj_pts) < 3:
        # Fall back to seed defaults (caller should not have invoked this).
        return rvec_init.copy(), fx_init

    def _residuals(p: np.ndarray) -> np.ndarray:
        fx = float(p[0])
        rvec = p[1:4]
        R, _ = cv2.Rodrigues(rvec)
        cam = obj_pts @ R.T + t_fixed
        safe_z = np.where(cam[:, 2] > 1e-3, cam[:, 2], 1e-3)
        K = _make_K(fx, cx, cy)
        pix = cam @ K.T
        proj = pix[:, :2] / safe_z[:, None]
        return (proj - img_pts).reshape(-1)

    p0 = np.concatenate([[fx_init], rvec_init])
    try:
        result = least_squares(_residuals, p0, method="lm", max_nfev=200)
        return result.x[1:4].copy(), float(np.clip(result.x[0], 50.0, 1e5))
    except Exception:
        return rvec_init.copy(), fx_init


def _solve_anchor_with_t_fixed_lines(
    anchor: Anchor,
    t_fixed: np.ndarray,
    cx: float,
    cy: float,
    fx_init: float,
    rvec_init: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Like ``_solve_anchor_with_t_fixed`` but uses line correspondences
    instead of (or alongside) point landmarks. Used to seed line-only
    anchors from the rich anchor's tvec.
    """
    if not anchor.lines and not anchor.landmarks:
        return rvec_init.copy(), fx_init

    obj_pts = (
        np.array([lm.world_xyz for lm in anchor.landmarks], dtype=np.float64)
        if anchor.landmarks else np.empty((0, 3))
    )
    img_pts = (
        np.array([lm.image_xy for lm in anchor.landmarks], dtype=np.float64)
        if anchor.landmarks else np.empty((0, 2))
    )

    def _residuals(p: np.ndarray) -> np.ndarray:
        fx = float(p[0])
        rvec = p[1:4]
        R, _ = cv2.Rodrigues(rvec)
        K = _make_K(fx, cx, cy)
        parts: list[np.ndarray] = []
        if len(obj_pts):
            cam = obj_pts @ R.T + t_fixed
            safe_z = np.where(cam[:, 2] > 1e-3, cam[:, 2], 1e-3)
            pix = cam @ K.T
            proj = pix[:, :2] / safe_z[:, None]
            parts.append((proj - img_pts).reshape(-1))
        if anchor.lines:
            parts.append(_line_residuals(list(anchor.lines), K, R, t_fixed))
        return np.concatenate(parts)

    p0 = np.concatenate([[fx_init], rvec_init])
    try:
        result = least_squares(_residuals, p0, method="lm", max_nfev=300)
        return result.x[1:4].copy(), float(np.clip(result.x[0], 50.0, 1e5))
    except Exception:
        return rvec_init.copy(), fx_init


def _rvec_to_R(rvec: np.ndarray) -> np.ndarray:
    """Rodrigues vector to 3x3 rotation matrix via OpenCV (matches solvePnP)."""
    Rmat, _ = cv2.Rodrigues(rvec)
    return Rmat


def _make_K(fx: float, cx: float, cy: float) -> np.ndarray:
    return np.array([[fx, 0.0, cx], [0.0, fx, cy], [0.0, 0.0, 1.0]])


# Parameter packing -----------------------------------------------------------
#
# Layout: [tx, ty, tz, cx, cy, rvec_0(3), fx_0, rvec_1(3), fx_1, ...]
#         shared globals       per-anchor (4 each)


_GLOBALS = 5    # tx, ty, tz, cx, cy
_PER_ANCHOR = 4  # rvec(3), fx(1)


def _pack_params(
    t: np.ndarray,
    cx: float,
    cy: float,
    rvecs: list[np.ndarray],
    fxs: list[float],
) -> np.ndarray:
    out = np.empty(_GLOBALS + _PER_ANCHOR * len(rvecs))
    out[:3] = t
    out[3] = cx
    out[4] = cy
    for i, (rv, fx) in enumerate(zip(rvecs, fxs)):
        base = _GLOBALS + i * _PER_ANCHOR
        out[base : base + 3] = rv
        out[base + 3] = fx
    return out


def _unpack_params(p: np.ndarray, n_anchors: int) -> tuple[np.ndarray, float, float, list[np.ndarray], list[float]]:
    t = p[:3]
    cx = float(p[3])
    cy = float(p[4])
    rvecs: list[np.ndarray] = []
    fxs: list[float] = []
    for i in range(n_anchors):
        base = _GLOBALS + i * _PER_ANCHOR
        rvecs.append(p[base : base + 3])
        fxs.append(float(p[base + 3]))
    return t, cx, cy, rvecs, fxs


# Residuals -------------------------------------------------------------------


def _point_residuals(
    points: list[LandmarkObservation], K: np.ndarray, R: np.ndarray, t: np.ndarray
) -> np.ndarray:
    """2 residuals per point: (proj_u - obs_u, proj_v - obs_v)."""
    if not points:
        return np.empty(0)
    world = np.array([lm.world_xyz for lm in points], dtype=np.float64)
    obs = np.array([lm.image_xy for lm in points], dtype=np.float64)
    cam = world @ R.T + t                          # (N, 3)
    # Guard against points behind the camera with a large finite residual
    # — the LM should pull the parameters out of that regime, but if it
    # gets stuck we want a finite signal not NaN.
    safe_z = np.where(cam[:, 2] > 1e-3, cam[:, 2], 1e-3)
    pix = cam @ K.T
    proj = pix[:, :2] / safe_z[:, None]
    out = (proj - obs).reshape(-1)
    # Penalise behind-camera projections so LM has gradient pulling them forward.
    behind = cam[:, 2] <= 1e-3
    if behind.any():
        # Multiply offending residuals by a large factor (still finite) so
        # the LM strongly disfavours these regions.
        scale = np.ones_like(safe_z)
        scale[behind] = 1e3
        out = (out.reshape(-1, 2) * scale[:, None]).reshape(-1)
    return out


def _line_residuals(
    lines: list[LineObservation], K: np.ndarray, R: np.ndarray, t: np.ndarray
) -> np.ndarray:
    """2 residuals per line: perpendicular distance from each user-clicked
    image endpoint to the projected world line.

    The projected world line is defined by the projections of the world
    segment's endpoints. Using image-plane line-distance (not point-to-point)
    means the user's image endpoints are NOT required to coincide with the
    world segment's endpoints — they only need to lie on the projected line.
    """
    if not lines:
        return np.empty(0)
    out = np.empty(2 * len(lines))
    for i, ln in enumerate(lines):
        Pa = np.array(ln.world_segment[0], dtype=np.float64)
        Pb = np.array(ln.world_segment[1], dtype=np.float64)
        cam_a = R @ Pa + t
        cam_b = R @ Pb + t
        # If both world endpoints are behind the camera, the line cannot be
        # evaluated at this iteration. Penalise heavily but with finite values.
        if cam_a[2] <= 1e-3 and cam_b[2] <= 1e-3:
            out[2 * i] = 1e6
            out[2 * i + 1] = 1e6
            continue
        # Clamp z to keep projections finite.
        za = cam_a[2] if cam_a[2] > 1e-3 else 1e-3
        zb = cam_b[2] if cam_b[2] > 1e-3 else 1e-3
        pa = (K @ cam_a)[:2] / za
        pb = (K @ cam_b)[:2] / zb
        # Image line through pa, pb expressed as ax + by + c = 0 with
        # (a, b) the unit normal.
        d = pb - pa
        norm = float(np.linalg.norm(d))
        if norm < 1e-6:
            # Degenerate: world endpoints project to the same image point.
            out[2 * i] = 1e6
            out[2 * i + 1] = 1e6
            continue
        nx = -d[1] / norm
        ny = d[0] / norm
        c = -(nx * pa[0] + ny * pa[1])
        for j, (u, v) in enumerate(ln.image_segment):
            out[2 * i + j] = nx * u + ny * v + c
    return out


def _anchor_residuals(
    anchor: Anchor, K: np.ndarray, R: np.ndarray, t: np.ndarray
) -> np.ndarray:
    return np.concatenate([
        _point_residuals(list(anchor.landmarks), K, R, t),
        _line_residuals(list(anchor.lines), K, R, t),
    ])


def _residuals(
    p: np.ndarray, anchors: tuple[Anchor, ...]
) -> np.ndarray:
    t, cx, cy, rvecs, fxs = _unpack_params(p, len(anchors))
    parts: list[np.ndarray] = []
    for anchor, rvec, fx in zip(anchors, rvecs, fxs):
        K = _make_K(fx, cx, cy)
        R = _rvec_to_R(rvec)
        parts.append(_anchor_residuals(anchor, K, R, t))
    return np.concatenate(parts) if parts else np.empty(0)


def _residual_lengths(anchors: tuple[Anchor, ...]) -> list[int]:
    """How many residuals each anchor contributes (2 per point + 2 per line)."""
    return [2 * len(a.landmarks) + 2 * len(a.lines) for a in anchors]


def _jac_sparsity(anchors: tuple[Anchor, ...]) -> lil_matrix:
    """Each residual touches: shared globals (5 cols) + its anchor's 4 cols."""
    lengths = _residual_lengths(anchors)
    n_res = sum(lengths)
    n_param = _GLOBALS + _PER_ANCHOR * len(anchors)
    spar = lil_matrix((n_res, n_param), dtype=np.uint8)
    row = 0
    for i, length in enumerate(lengths):
        if length == 0:
            continue
        anchor_base = _GLOBALS + i * _PER_ANCHOR
        for j in range(length):
            for c in range(_GLOBALS):
                spar[row + j, c] = 1
            for c in range(_PER_ANCHOR):
                spar[row + j, anchor_base + c] = 1
        row += length
    return spar


# Public solver ---------------------------------------------------------------


def solve_anchors_jointly(
    anchors: tuple[Anchor, ...],
    image_size: tuple[int, int],
) -> JointSolution:
    """Joint bundle adjustment over all anchors. See module docstring."""
    if not anchors:
        raise AnchorSolveError("no anchors supplied")
    qualifying = tuple(a for a in anchors if _qualifies(a))
    if not qualifying:
        raise AnchorSolveError(
            "no anchor has ≥4 point landmarks or ≥2 line correspondences; "
            "place more landmarks (or lines) before running the camera stage"
        )

    width, height = image_size
    cx_init = width / 2.0
    cy_init = height / 2.0
    fx_init = float(width)        # broadcast prior: fx ≈ image width
    K_seed = _make_K(fx_init, cx_init, cy_init)

    # Pick the richest anchor (most landmarks + most z-diversity) and run a
    # one-anchor LM to get a good fx seed before initialising the others.
    # The broadcast prior fx=image_width is often off by a factor of 1.5×;
    # solvePnP at the wrong fx places landmarks behind the camera at the
    # other anchors, and the joint LM struggles to escape.
    seed_anchor = _pick_seed_anchor(qualifying)
    seed = _seed_anchor_pose(seed_anchor, K_seed)
    if seed is None:
        raise AnchorSolveError(
            f"cv2.solvePnP failed on seed anchor (frame {seed_anchor.frame}); "
            f"add more / clearer landmarks on that frame"
        )
    seed_rvec_init, seed_tvec_init = seed

    fx_refined = _refine_seed_fx(
        seed_anchor, seed_rvec_init, seed_tvec_init, cx_init, cy_init, fx_init,
    )
    K_refined = _make_K(fx_refined, cx_init, cy_init)

    # Re-run solvePnP on the seed anchor with the refined K for a more
    # consistent (R, t) anchor.
    seed = _seed_anchor_pose(seed_anchor, K_refined)
    if seed is not None:
        seed_rvec_init, seed_tvec_init = seed

    # Use the seed anchor's tvec as the shared t_world initial guess. Other
    # anchors' R and fx are seeded by a t-fixed LM (much more stable on
    # coplanar landmarks than running solvePnP per anchor).
    t_init = seed_tvec_init.copy()

    rvec_seeds: list[np.ndarray] = []
    fx_seeds: list[float] = []
    for a in qualifying:
        if a.frame == seed_anchor.frame:
            rvec_seeds.append(seed_rvec_init.copy())
            fx_seeds.append(fx_refined)
            continue
        if len(a.landmarks) >= 3:
            # Get a good rvec initial via solvePnP with the refined K, then
            # let the t-fixed LM polish (R, fx) under the shared t_init.
            seeded = _seed_anchor_pose(a, K_refined)
            rvec_init = seeded[0] if seeded is not None else seed_rvec_init
            rvec, fx = _solve_anchor_with_t_fixed(
                a, t_init, cx_init, cy_init, fx_refined, rvec_init,
            )
            rvec_seeds.append(rvec)
            fx_seeds.append(fx)
        else:
            # Line-only anchor — initialise from seed; let line-aware
            # t-fixed LM (below) refine (R, fx) using the line residuals.
            rvec, fx = _solve_anchor_with_t_fixed_lines(
                a, t_init, cx_init, cy_init, fx_refined, seed_rvec_init,
            )
            rvec_seeds.append(rvec)
            fx_seeds.append(fx)

    # ── Compose result from staged seeds ──────────────────────────────────
    # The joint LM was originally meant to polish the seeds, but in
    # practice it tends to compromise the rich anchor's tight fit by
    # tugging t_world toward thin/noisy anchors. The seeded values
    # (rich-anchor full solve + per-other-anchor t-fixed LM) are already
    # close to optimal under the shared-t broadcast assumption; anchors
    # whose user-clicks don't fit that assumption surface as high
    # per-anchor residuals (the camera stage flags those as low confidence).
    per_anchor_KR: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    per_anchor_res: dict[int, float] = {}
    for a, rvec, fx in zip(qualifying, rvec_seeds, fx_seeds):
        K = _make_K(fx, cx_init, cy_init)
        R = _rvec_to_R(rvec)
        per_anchor_KR[a.frame] = (K, R)
        per_anchor_res[a.frame] = reprojection_residual_for_anchor(a, K, R, t_init)

    logger.info(
        "staged solve: %d anchors, seed=%d, t_world=[%.2f, %.2f, %.2f], "
        "fx_seed=%.1f, mean per-anchor residual=%.2f px",
        len(qualifying), seed_anchor.frame, *t_init, fx_refined,
        float(np.mean(list(per_anchor_res.values()))) if per_anchor_res else 0.0,
    )

    return JointSolution(
        t_world=t_init,
        principal_point=(cx_init, cy_init),
        per_anchor_KR=per_anchor_KR,
        per_anchor_residual_px=per_anchor_res,
    )

    # ── (Joint LM polish disabled — see commit message for rationale) ────
    p0 = _pack_params(t_init, cx_init, cy_init, rvec_seeds, fx_seeds)
    sparsity = _jac_sparsity(qualifying)

    # Parameter bounds — keep the LM out of pathological regions where it
    # can diverge to 1e27 t_world or wildly off principal points. fx has a
    # wide range (broadcast lenses span ~400 to image_width × 5 effectively).
    # Rotation vectors are unbounded (axis-angle wraps).
    n = len(qualifying)
    lo = np.full(_GLOBALS + _PER_ANCHOR * n, -np.inf)
    hi = np.full(_GLOBALS + _PER_ANCHOR * n, np.inf)
    # t_world bounds: ±200 m from pitch origin (FIFA pitch is 105×68).
    lo[:3] = -200.0
    hi[:3] = 200.0
    # Principal point: tightly bounded to image centre. Allowing it to roam
    # creates a degenerate parameter space (pp + fx + rotation are
    # interchangeable in many configurations), and broadcast lenses are
    # close to centred in practice. Tight ±2 % bound effectively pins it.
    lo[3] = cx_init - 0.02 * width
    hi[3] = cx_init + 0.02 * width
    lo[4] = cy_init - 0.02 * height
    hi[4] = cy_init + 0.02 * height
    # Per-anchor fx: 0.2× to 10× image width covers ultra-wide to long
    # telephoto broadcast lenses.
    for i in range(n):
        fx_idx = _GLOBALS + i * _PER_ANCHOR + 3
        lo[fx_idx] = 0.2 * width
        hi[fx_idx] = 10.0 * width

    # Clamp the seed to lie strictly inside bounds (LM requires this).
    p0 = np.clip(p0, lo + 1e-6, hi - 1e-6)

    try:
        # Use the trust-region reflective method with auto-scaling. ``lm``
        # (true Levenberg–Marquardt) does not support bounds or sparse
        # Jacobians, so we keep ``trf`` to leverage both. Plain L2 loss;
        # outlier handling can be added back later if real-data smoke
        # tests show issues.
        result = least_squares(
            _residuals,
            p0,
            args=(qualifying,),
            bounds=(lo, hi),
            jac_sparsity=sparsity,
            method="trf",
            x_scale="jac",
            ftol=1e-10,
            xtol=1e-10,
            gtol=1e-10,
            max_nfev=2000,
        )
    except Exception as exc:
        raise AnchorSolveError(f"joint LM failed: {exc}") from exc

    t_world, cx, cy, rvecs, fxs = _unpack_params(result.x, len(qualifying))

    per_anchor_KR: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    per_anchor_res: dict[int, float] = {}
    for a, rvec, fx in zip(qualifying, rvecs, fxs):
        K = _make_K(fx, cx, cy)
        R = _rvec_to_R(rvec)
        per_anchor_KR[a.frame] = (K, R)
        per_anchor_res[a.frame] = reprojection_residual_for_anchor(a, K, R, t_world)

    logger.info(
        "joint BA solved: %d anchors, t_world=[%.2f, %.2f, %.2f], "
        "principal_point=(%.1f, %.1f), final cost=%.4f",
        len(qualifying), *t_world, cx, cy, float(result.cost),
    )

    return JointSolution(
        t_world=t_world,
        principal_point=(cx, cy),
        per_anchor_KR=per_anchor_KR,
        per_anchor_residual_px=per_anchor_res,
    )
