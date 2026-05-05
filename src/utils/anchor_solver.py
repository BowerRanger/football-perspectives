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
    t_world: np.ndarray                                            # (3,) — median across anchors
    principal_point: tuple[float, float]                           # (cx, cy)
    per_anchor_KRt: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]
    """Per-anchor (K, R, t). Each anchor has its own translation because
    the broadcast-fixed-body assumption doesn't hold reliably on real
    clips (steadicam, broadcast cuts between cameras). Solving each
    anchor independently gives much tighter per-anchor calibration.
    """
    per_anchor_residual_px: dict[int, float]                       # frame -> mean px


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
    # SOLVEPNP_SQPNP (OpenCV ≥ 4.5) is the modern globally-optimal PnP that
    # handles ≥3 points including coplanar configurations more stably than
    # EPNP, which is highly sensitive to small fx perturbations on coplanar
    # data. Falls back to ITERATIVE for ≥6 points (well-conditioned non-
    # coplanar cases benefit from its iterative refinement).
    flag = cv2.SOLVEPNP_SQPNP if len(anchor.landmarks) < 6 else cv2.SOLVEPNP_ITERATIVE
    ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, dist_zero, flags=flag)
    if not ok:
        return None
    return rvec.reshape(3), tvec.reshape(3)


_LINE_RESIDUAL_WEIGHT = 0.2


def _refine_seed_pose(
    anchor: Anchor,
    rvec_init: np.ndarray,
    tvec_init: np.ndarray,
    cx: float,
    cy: float,
    fx_init: float,
) -> tuple[float, np.ndarray, np.ndarray]:
    """One-anchor LM that refines (fx, R, t) on the seed anchor.

    Uses point landmarks (full weight) and line annotations (down-weighted
    by ``_LINE_RESIDUAL_WEIGHT``). Lines have inherently large residual
    magnitudes when the projected line is even slightly off; without
    down-weighting they dominate the LM and pull the solution away from
    the tight point-only fit (observed empirically on real data — frame
    429's point-only residual jumped from 5 px to 117 px when lines
    were given equal weight).

    Returns ``(fx, rvec, tvec)`` — caller updates K and the seed pose.
    """
    if not anchor.landmarks:
        return fx_init, rvec_init.copy(), tvec_init.copy()
    obj_pts = np.array([lm.world_xyz for lm in anchor.landmarks], dtype=np.float64)
    img_pts = np.array([lm.image_xy for lm in anchor.landmarks], dtype=np.float64)
    # When the seed already has plenty of well-conditioned points (≥6 with
    # non-coplanar), skip lines entirely on the seed solve — they only add
    # noise. Lines kick in below this threshold to rescue thinner seeds.
    n_pts = len(obj_pts)
    z_levels = len({round(z, 3) for _, _, z in obj_pts})
    use_lines_on_seed = anchor.lines and (n_pts < 6 or z_levels < 2)

    def _residuals(p: np.ndarray) -> np.ndarray:
        fx = float(p[0])
        rvec = p[1:4]
        tvec = p[4:7]
        R, _ = cv2.Rodrigues(rvec)
        K = _make_K(fx, cx, cy)
        cam = obj_pts @ R.T + tvec
        safe_z = np.where(cam[:, 2] > 1e-3, cam[:, 2], 1e-3)
        pix = cam @ K.T
        proj = pix[:, :2] / safe_z[:, None]
        parts: list[np.ndarray] = [(proj - img_pts).reshape(-1)]
        if use_lines_on_seed:
            parts.append(
                _LINE_RESIDUAL_WEIGHT
                * _line_residuals(list(anchor.lines), K, R, tvec)
            )
        return np.concatenate(parts)

    p0 = np.concatenate([[fx_init], rvec_init, tvec_init])
    try:
        result = least_squares(_residuals, p0, method="lm", max_nfev=300)
        return (
            float(np.clip(result.x[0], 50.0, 1e5)),
            result.x[1:4].copy(),
            result.x[4:7].copy(),
        )
    except Exception:
        return fx_init, rvec_init.copy(), tvec_init.copy()


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
    obj_pts = (
        np.array([lm.world_xyz for lm in anchor.landmarks], dtype=np.float64)
        if anchor.landmarks else np.empty((0, 3))
    )
    img_pts = (
        np.array([lm.image_xy for lm in anchor.landmarks], dtype=np.float64)
        if anchor.landmarks else np.empty((0, 2))
    )
    if len(obj_pts) < 3 and not anchor.lines:
        return rvec_init.copy(), fx_init

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
            # Down-weight lines (see _refine_seed_pose). Heavier weight when
            # there are no points (line-only anchors rely entirely on lines).
            weight = 1.0 if len(obj_pts) == 0 else _LINE_RESIDUAL_WEIGHT
            parts.append(
                weight * _line_residuals(list(anchor.lines), K, R, t_fixed)
            )
        return np.concatenate(parts) if parts else np.empty(0)

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
    """2 residuals per line.

    For *position-known* lines (``world_segment`` set): perpendicular
    distance from each user-clicked image endpoint to the projected world
    line. The projected line is defined by the projections of the world
    segment's endpoints.

    For *direction-only* lines (``world_direction`` set, used for vanishing-
    point constraints like ``vertical_separator``): the line is parallel to
    a known world direction but its world position is unknown. The residual
    is the perpendicular distance from the *vanishing point* of that
    direction (i.e. ``K @ R @ d`` perspective-divided) to the line through
    the user's two image clicks. Each click contributes one residual via
    its perpendicular distance from the line, computed against the line
    that would pass through the VP and bisect the user's clicks.

    Either way, each line contributes 2 residuals.
    """
    if not lines:
        return np.empty(0)
    out = np.empty(2 * len(lines))
    for i, ln in enumerate(lines):
        if ln.world_direction is not None:
            # Vanishing-point residual.
            d_world = np.array(ln.world_direction, dtype=np.float64)
            d_cam = R @ d_world          # direction in camera frame
            # The vanishing point is K @ d_cam, perspective-divided. If
            # d_cam[2] ≈ 0 the VP is at infinity in image plane (line is
            # parallel to image plane); fall through to a "line direction"
            # residual using d_cam[:2] as the image-plane direction.
            (u1, v1), (u2, v2) = ln.image_segment
            if abs(d_cam[2]) < 1e-3:
                # VP at infinity — image line should be parallel to d_cam[:2].
                # Residual: cross-product magnitude between user line direction
                # and projected world direction.
                user_dir = np.array([u2 - u1, v2 - v1])
                un = float(np.linalg.norm(user_dir))
                if un < 1e-6:
                    out[2 * i] = 1e6; out[2 * i + 1] = 1e6
                    continue
                user_dir /= un
                proj_dir = d_cam[:2]
                pn = float(np.linalg.norm(proj_dir))
                if pn < 1e-6:
                    out[2 * i] = 1e6; out[2 * i + 1] = 1e6
                    continue
                proj_dir /= pn
                # Cross product (scalar, signed) — pixels per metre of line
                # length, scaled to match magnitude with point residuals.
                cross = float(user_dir[0] * proj_dir[1] - user_dir[1] * proj_dir[0])
                out[2 * i] = un * cross / 2
                out[2 * i + 1] = un * cross / 2
                continue
            vp = (K @ d_cam)[:2] / d_cam[2]
            # Build image line through the user's two clicks and measure
            # perpendicular distance from VP to that line.
            a = np.array([u1, v1])
            b = np.array([u2, v2])
            ab = b - a
            ab_norm = float(np.linalg.norm(ab))
            if ab_norm < 1e-6:
                out[2 * i] = 1e6; out[2 * i + 1] = 1e6
                continue
            # Unit normal of the user's line.
            nx = -ab[1] / ab_norm
            ny = ab[0] / ab_norm
            cc = -(nx * a[0] + ny * a[1])
            dist = nx * vp[0] + ny * vp[1] + cc
            # Two residuals: distance from VP applied symmetrically to keep
            # parameter count consistent with the position-known branch.
            out[2 * i] = dist
            out[2 * i + 1] = dist
            continue

        # Position-known line residual (existing path).
        if ln.world_segment is None:
            out[2 * i] = 1e6; out[2 * i + 1] = 1e6
            continue
        Pa = np.array(ln.world_segment[0], dtype=np.float64)
        Pb = np.array(ln.world_segment[1], dtype=np.float64)
        cam_a = R @ Pa + t
        cam_b = R @ Pb + t
        if cam_a[2] <= 1e-3 and cam_b[2] <= 1e-3:
            out[2 * i] = 1e6
            out[2 * i + 1] = 1e6
            continue
        za = cam_a[2] if cam_a[2] > 1e-3 else 1e-3
        zb = cam_b[2] if cam_b[2] > 1e-3 else 1e-3
        pa = (K @ cam_a)[:2] / za
        pb = (K @ cam_b)[:2] / zb
        d = pb - pa
        norm = float(np.linalg.norm(d))
        if norm < 1e-6:
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


def _is_degenerate_solo(t: np.ndarray, fx: float) -> bool:
    """Reject solo solves where the LM walked into a non-physical region:
    camera position more than a soccer-field-and-a-half from the pitch, or
    fx outside any realistic broadcast lens range. These configs sometimes
    yield low local-residuals via massive K↔t compensation but the
    geometry is meaningless.
    """
    if abs(t[0]) > 200 or abs(t[1]) > 200 or abs(t[2]) > 500:
        return True
    if fx < 200 or fx > 20000:
        return True
    return False


def _solve_one_anchor_full(
    anchor: Anchor,
    cx: float,
    cy: float,
    fx_init: float,
    K_init: np.ndarray,
    fallback_seed: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float] | None:
    """Solo solve: cv2.solvePnP for a (R, t) seed, then LM-refine (fx, R, t)
    using both point landmarks and line annotations.

    Tries the caller-supplied ``fx_init`` first; if the LM lands in a
    degenerate region (very small fx or very far |t|), retries with a
    small ladder of alternative fx priors and returns the best
    non-degenerate result. This handles real broadcast clips where the
    nearest-rich-anchor's fx differs enough from the thin anchor's true
    fx that LM falls into a wrong basin.

    ``fallback_seed`` is an optional ``(rvec, tvec)`` pair used as the LM
    starting pose when the anchor has too few point landmarks (<4) to
    bootstrap from cv2.solvePnP. Pass the nearest rich anchor's pose so
    the LM starts in a sensible region of the parameter space; without it
    the fallback is identity-rotation + ``(0, 0, 50) m`` translation,
    which is rarely close to truth on a thin line-heavy anchor.

    Returns ``(K, R, t, fx)`` or ``None`` on failure / no valid result.
    """
    if len(anchor.landmarks) < 4 and len(anchor.lines) < 2:
        return None

    def _try(fx_seed: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, float] | None:
        K_seed = _make_K(fx_seed, cx, cy)
        if len(anchor.landmarks) >= 4:
            seed = _seed_anchor_pose(anchor, K_seed)
            if seed is None:
                return None
            rvec_init, tvec_init = seed
        elif fallback_seed is not None:
            rvec_init = fallback_seed[0].copy()
            tvec_init = fallback_seed[1].copy()
        else:
            rvec_init = np.array([0.001, 0.001, 0.001])
            tvec_init = np.array([0.0, 0.0, 50.0])
        fx, rvec, tvec = _refine_seed_pose(
            anchor, rvec_init, tvec_init, cx, cy, fx_seed,
        )
        K = _make_K(fx, cx, cy)
        R = _rvec_to_R(rvec)
        return K, R, tvec, fx

    # First attempt with caller's prior.
    primary = _try(fx_init)
    if primary is not None and not _is_degenerate_solo(primary[2], primary[3]):
        return primary

    # Primary attempt was degenerate (or failed). Try alternative priors.
    candidate_best = primary
    candidate_res = (
        reprojection_residual_for_anchor(anchor, primary[0], primary[1], primary[2])
        if primary is not None and not _is_degenerate_solo(primary[2], primary[3])
        else float("inf")
    )
    for mult in (1.1, 0.9, 1.2, 0.8, 1.5, 0.7, 2.0, 0.5, 3.0, 0.3):
        alt = _try(fx_init * mult)
        if alt is None:
            continue
        if _is_degenerate_solo(alt[2], alt[3]):
            continue
        res = reprojection_residual_for_anchor(anchor, alt[0], alt[1], alt[2])
        if res < candidate_res:
            candidate_res = res
            candidate_best = alt
    return candidate_best


def _is_rich(anchor: Anchor, min_points: int = 6) -> bool:
    """Whether an anchor has enough non-coplanar points to solo-solve (K, R, t).

    A rich anchor can recover its own translation independently of other
    anchors — used for the v2 hybrid solve where the broadcast-fixed-body
    assumption is dropped.
    """
    if len(anchor.landmarks) < min_points:
        return False
    z_levels = {round(lm.world_xyz[2], 3) for lm in anchor.landmarks}
    return len(z_levels) >= 2


def _interp_t(
    frame: int, rich_frames: list[int], t_by_frame: dict[int, np.ndarray]
) -> np.ndarray:
    """Linearly interpolate t at ``frame`` from the bracketing rich anchors.
    Clamps to the first/last rich anchor outside the bracket range."""
    if frame <= rich_frames[0]:
        return t_by_frame[rich_frames[0]].copy()
    if frame >= rich_frames[-1]:
        return t_by_frame[rich_frames[-1]].copy()
    for a, b in zip(rich_frames, rich_frames[1:]):
        if a <= frame <= b:
            w = (frame - a) / (b - a) if b > a else 0.0
            return (1.0 - w) * t_by_frame[a] + w * t_by_frame[b]
    return t_by_frame[rich_frames[0]].copy()


def solve_anchors_jointly(
    anchors: tuple[Anchor, ...],
    image_size: tuple[int, int],
) -> JointSolution:
    """Hybrid per-anchor solve.

    Pass 1: every "rich" anchor (≥6 non-coplanar landmarks) is solo-solved
    via solvePnP + LM refine — each gets its own (K, R, t).
    Pass 2: thin anchors inherit a t linearly interpolated between the
    bracketing rich anchors and run a t-fixed LM for (R, fx).

    This drops the "broadcast camera body fixed" assumption (which doesn't
    hold for steadicam / multi-camera-stitched clips). Camera position
    can vary smoothly between rich anchors; thin anchors trust the
    interpolated value.
    """
    if not anchors:
        raise AnchorSolveError("no anchors supplied")
    qualifying = tuple(a for a in anchors if _qualifies(a))
    if not qualifying:
        raise AnchorSolveError(
            "no anchor has ≥4 point landmarks or ≥2 line correspondences; "
            "place more landmarks (or lines) before running the camera stage"
        )

    width, height = image_size
    cx = width / 2.0
    cy = height / 2.0
    fx_init = float(width)              # broadcast prior, refined per-anchor
    K_init = _make_K(fx_init, cx, cy)

    per_anchor_KRt: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    per_anchor_res: dict[int, float] = {}

    # ── Pass 1: solo-solve every rich anchor ────────────────────────────
    rich_anchors = [a for a in qualifying if _is_rich(a)]
    if not rich_anchors:
        raise AnchorSolveError(
            "no anchor has ≥6 non-coplanar landmarks; place at least one "
            "rich anchor (e.g. corners + a corner-flag-top + a goal-crossbar "
            "endpoint) so the solver can recover the camera pose"
        )

    rich_t: dict[int, np.ndarray] = {}
    rich_rvec: dict[int, np.ndarray] = {}
    rich_fx: dict[int, float] = {}
    for a in sorted(rich_anchors, key=lambda x: x.frame):
        result = _solve_one_anchor_full(a, cx, cy, fx_init, K_init)
        if result is None:
            continue
        K, R, t, fx = result
        per_anchor_KRt[a.frame] = (K, R, t)
        per_anchor_res[a.frame] = reprojection_residual_for_anchor(a, K, R, t)
        rich_t[a.frame] = t
        rvec_arr, _ = cv2.Rodrigues(R)
        rich_rvec[a.frame] = rvec_arr.reshape(3)
        rich_fx[a.frame] = fx

    if not rich_t:
        raise AnchorSolveError(
            "all rich-anchor solo solves failed (cv2.solvePnP refused). "
            "Check that landmark world_xyz values match the FIFA catalogue."
        )

    # ── Pass 2: thin anchors — pick the better of solo-solve or t-fixed ──
    # Try BOTH a solo full-solve (uses the anchor's own constraints to
    # recover its own t) and a t-fixed LM (inherits the t interpolated
    # between bracketing rich anchors). Keep whichever produces a lower
    # reprojection residual. Solo wins when the anchor has enough info
    # to disagree with the rich-anchor t (e.g. clip with camera motion).
    # T-fixed wins when the anchor is geometrically degenerate alone.
    rich_frames_sorted = sorted(rich_t.keys())
    for a in qualifying:
        if a.frame in per_anchor_KRt:
            continue
        nearest_idx = min(
            range(len(rich_frames_sorted)),
            key=lambda i: abs(rich_frames_sorted[i] - a.frame),
        )
        nearest_frame = rich_frames_sorted[nearest_idx]
        fx_prior = rich_fx[nearest_frame]
        K_prior = _make_K(fx_prior, cx, cy)
        # Warm-start the LM from the nearest rich anchor's pose. Crucial
        # for line-heavy thin anchors (<4 points) where solvePnP can't
        # bootstrap and the previous identity-rotation default landed the
        # LM in arbitrary local minima.
        fallback_seed = (rich_rvec[nearest_frame], rich_t[nearest_frame])

        # Candidate A: solo solve.
        solo_KRt: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        solo_res = float("inf")
        solo_result = _solve_one_anchor_full(
            a, cx, cy, fx_prior, K_prior, fallback_seed=fallback_seed,
        )
        if solo_result is not None:
            K_s, R_s, t_s, _ = solo_result
            solo_res = reprojection_residual_for_anchor(a, K_s, R_s, t_s)
            solo_KRt = (K_s, R_s, t_s)

        # Candidate B: t-fixed (inherited from interpolation).
        t_inherited = _interp_t(a.frame, rich_frames_sorted, rich_t)
        if len(a.landmarks) >= 4:
            seed = _seed_anchor_pose(a, K_prior)
            rvec_init = seed[0] if seed is not None else rich_rvec[nearest_frame].copy()
        else:
            rvec_init = rich_rvec[nearest_frame].copy()
        rvec, fx = _solve_anchor_with_t_fixed(
            a, t_inherited, cx, cy, fx_prior, rvec_init,
        )
        K_t = _make_K(fx, cx, cy)
        R_t = _rvec_to_R(rvec)
        tfx_res = reprojection_residual_for_anchor(a, K_t, R_t, t_inherited)

        if solo_KRt is not None and solo_res < tfx_res:
            per_anchor_KRt[a.frame] = solo_KRt
            per_anchor_res[a.frame] = solo_res
        else:
            per_anchor_KRt[a.frame] = (K_t, R_t, t_inherited)
            per_anchor_res[a.frame] = tfx_res

    # Representative t_world: median across rich-anchor t values.
    ts_rich = np.stack(list(rich_t.values()))
    t_world_median = np.median(ts_rich, axis=0)

    logger.info(
        "hybrid solve: %d total (%d rich solo + %d thin t-interpolated), "
        "t_world median=[%.2f, %.2f, %.2f], mean residual=%.2f px",
        len(per_anchor_KRt), len(rich_t),
        len(per_anchor_KRt) - len(rich_t), *t_world_median,
        float(np.mean(list(per_anchor_res.values()))) if per_anchor_res else 0.0,
    )

    return JointSolution(
        t_world=t_world_median,
        principal_point=(cx, cy),
        per_anchor_KRt=per_anchor_KRt,
        per_anchor_residual_px=per_anchor_res,
    )
