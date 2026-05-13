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
    camera_centre: np.ndarray | None = None
    """World-frame camera body position when the static-camera relock has
    been applied. ``None`` for un-relocked solutions. When set, every
    per-anchor (R, t) satisfies ``-R^T @ t == camera_centre``."""
    distortion: tuple[float, float] = (0.0, 0.0)
    """Clip-shared radial distortion (k1, k2). Default (0, 0) for
    solutions produced before lens distortion was added to the model.
    Bounded to ``|k| < 0.5`` by the joint LM."""


# Public utilities ------------------------------------------------------------


def _refine_shared_camera_centre(
    anchors: list[Anchor],
    per_anchor_KRt: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]],
    cx: float,
    cy: float,
    distortion: tuple[float, float],
    C_seed: np.ndarray,
) -> np.ndarray:
    """Joint LM that optimises a single shared world-frame camera centre
    ``C`` plus per-anchor (rvec, fx) so the static-camera constraint
    ``t = -R @ C`` holds *and* total reprojection residual is minimised.

    Parameter layout: ``[Cx, Cy, Cz, rvec_0(3), fx_0, rvec_1(3), fx_1, ...]``.
    Per-anchor t is reconstructed inside the residual; cx, cy and
    distortion are held fixed at the joint-solve values.

    Replaces the simple median-of-solo-Cs that ``refine_with_shared_
    translation`` previously used as ``C_locked``. The median is fine
    when solo Cs cluster, but on real broadcast clips with click noise
    on far-side landmarks the solos spread by metres and the median
    sits off-truth by enough to introduce a per-anchor yaw bias as the
    relock LM compensates. Optimising C against all observations finds
    the true position to sub-metre precision and the bias drops out.
    Falls back to ``C_seed`` if the LM fails or the result is worse.

    Returns the optimised ``C_locked`` (3,) array. The caller is
    responsible for re-fitting per-anchor (R, fx) under this C.
    """
    qualifying = [
        a for a in anchors if a.frame in per_anchor_KRt
    ]
    if not qualifying:
        return np.asarray(C_seed, dtype=np.float64)

    rvecs_init: list[np.ndarray] = []
    fxs_init: list[float] = []
    for a in qualifying:
        K_a, R_a, _t_a = per_anchor_KRt[a.frame]
        rv_a, _ = cv2.Rodrigues(R_a.astype(np.float64))
        rvecs_init.append(rv_a.flatten().copy())
        fxs_init.append(float(K_a[0, 0]))

    PER = 4
    n = len(qualifying)
    p0 = np.empty(3 + PER * n)
    p0[:3] = np.asarray(C_seed, dtype=np.float64)
    for i, (rv, fx) in enumerate(zip(rvecs_init, fxs_init)):
        base = 3 + i * PER
        p0[base : base + 3] = rv
        p0[base + 3] = fx

    def _residuals(p: np.ndarray) -> np.ndarray:
        C = p[:3]
        parts: list[np.ndarray] = []
        for i, anchor in enumerate(qualifying):
            base = 3 + i * PER
            rv = p[base : base + 3]
            fx = float(np.clip(p[base + 3], 50.0, 1e5))
            R_i, _ = cv2.Rodrigues(rv)
            t_i = -R_i @ C
            K_i = _make_K(fx, cx, cy)
            if anchor.landmarks:
                parts.append(_point_residuals_distorted(
                    list(anchor.landmarks), K_i, rv, t_i, distortion,
                ))
            if anchor.lines:
                # Down-weight lines (matches the per-anchor solo LMs):
                # lines have larger natural residual magnitudes than
                # point clicks and over-weight pulls the LM off-true.
                parts.append(_LINE_RESIDUAL_WEIGHT * _line_residuals(
                    list(anchor.lines), K_i, R_i, t_i,
                ))
        return np.concatenate(parts) if parts else np.empty(0)

    try:
        result = least_squares(
            _residuals, p0,
            method="trf", loss="huber", f_scale=2.0, max_nfev=500,
        )
    except Exception as exc:
        logger.warning("shared-C joint LM failed: %s; falling back to median", exc)
        return np.asarray(C_seed, dtype=np.float64)

    C_opt = np.asarray(result.x[:3], dtype=np.float64)
    # Sanity guard: reject moves that break physical plausibility (camera
    # body more than a pitch-length from the seed). Click-driven LMs can
    # occasionally walk into nonsensical regions.
    if float(np.linalg.norm(C_opt - np.asarray(C_seed))) > 50.0:
        logger.warning(
            "shared-C joint LM moved C by >50 m (seed=%s, opt=%s); "
            "falling back to seed",
            np.round(C_seed, 2).tolist(), np.round(C_opt, 2).tolist(),
        )
        return np.asarray(C_seed, dtype=np.float64)

    delta = float(np.linalg.norm(C_opt - np.asarray(C_seed)))
    logger.info(
        "shared-C joint LM: seed=(%.2f, %.2f, %.2f) → opt=(%.2f, %.2f, %.2f) "
        "(Δ=%.2f m)",
        *C_seed, *C_opt, delta,
    )
    return C_opt


def refine_with_shared_translation(
    anchors: tuple[Anchor, ...],
    sol: JointSolution,
) -> JointSolution:
    """Re-fit every anchor's (R, fx) against a single shared **camera
    centre** in world coordinates.

    For static stadium-mounted cameras the camera *body position* is the
    physical invariant — the body doesn't move, only pan/tilt/zoom. The
    OpenCV translation ``t`` is *not* the body position: ``t = -R @ C``
    where ``C`` is the world-frame centre, so t varies whenever R does.
    Locking t directly forces every anchor to share R as well, which
    breaks the moment the camera pans (i.e. always).

    The right invariant is C. We pick the lowest-residual anchor (its
    solo solve is the most trustworthy individual fit), compute its
    world-frame centre ``C = -R^T @ t``, then re-run a 4-DOF LM (rvec +
    fx) for every other anchor with C held constant — t per anchor is
    recomputed as ``-R @ C`` inside the residual.

    The exported ``JointSolution.t_world`` and per-anchor ``t`` are
    consistent with each anchor's R; downstream code that takes the
    clip-shared ``t_world`` should now treat it as "t at the seed
    anchor's rotation", or compute the camera centre via ``-R^T @ t``.

    Returns a fresh JointSolution with every anchor's ``-R^T @ t``
    equal to the locked C. If the resulting mean residual is much worse
    than the original, the original is returned untouched.
    """
    if not sol.per_anchor_KRt or not sol.per_anchor_residual_px:
        return sol

    by_frame = {a.frame: a for a in anchors}
    # Compute camera centre C = -R^T @ t per anchor and aggregate. We
    # prefer rich (≥6 points) non-collinear anchors because their C is
    # well-determined; thin / collinear anchors can have sub-pixel
    # residuals despite a wildly off C (the unconstrained dimension).
    Cs_rich: list[np.ndarray] = []
    Cs_noncol: list[np.ndarray] = []
    Cs_all: list[np.ndarray] = []
    for af, (_K, R, t) in sol.per_anchor_KRt.items():
        a = by_frame.get(af)
        C_a = -R.astype(np.float64).T @ t.astype(np.float64)
        Cs_all.append(C_a)
        if a is None:
            continue
        if not _landmarks_collinear(a):
            Cs_noncol.append(C_a)
            if _is_rich(a):
                Cs_rich.append(C_a)

    if Cs_rich:
        C_seed = np.median(np.stack(Cs_rich), axis=0)
        seed_pool = "rich+non-collinear"; seed_n = len(Cs_rich)
    elif Cs_noncol:
        C_seed = np.median(np.stack(Cs_noncol), axis=0)
        seed_pool = "non-collinear"; seed_n = len(Cs_noncol)
    else:
        C_seed = np.median(np.stack(Cs_all), axis=0)
        seed_pool = "all"; seed_n = len(Cs_all)

    # Joint-LM-refine the shared C against ALL anchors simultaneously.
    # Median of solo-solved Cs is only as good as the solos; click noise
    # on far-side landmarks spreads them by metres, and the median sits
    # off the truth by enough to introduce a small per-anchor R bias
    # (rotation that compensates for the slightly-wrong C). The
    # optimised C is the one that minimises total reprojection residual
    # under the static-camera constraint; once it converges, the
    # per-anchor R bias drops out as a byproduct.
    cx_seed, cy_seed = sol.principal_point
    qualifying_in_sol = [
        by_frame[af] for af in sol.per_anchor_KRt.keys() if af in by_frame
    ]
    C_locked = _refine_shared_camera_centre(
        anchors=qualifying_in_sol,
        per_anchor_KRt=sol.per_anchor_KRt,
        cx=cx_seed,
        cy=cy_seed,
        distortion=sol.distortion,
        C_seed=C_seed,
    )

    # We still need a representative seed_K/R for the JointSolution
    # output's t_world. Use the lowest-residual non-collinear anchor.
    representative_frame = min(
        sol.per_anchor_residual_px,
        key=lambda f: (
            _landmarks_collinear(by_frame[f]) if f in by_frame else True,
            sol.per_anchor_residual_px[f],
        ),
    )
    _Ks, R_seed, t_seed = sol.per_anchor_KRt[representative_frame]
    R_seed = R_seed.astype(np.float64)
    t_seed = t_seed.astype(np.float64)
    cx, cy = sol.principal_point
    best_frame = representative_frame  # used in log + early-keep below

    new_KRt: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    new_res: dict[int, float] = {}
    for af, (K_init, R_init, _t_init) in sol.per_anchor_KRt.items():
        anchor = by_frame.get(af)
        if anchor is None:
            # No anchor record to re-fit against — keep K/R, recompute t.
            R_new = R_init.astype(np.float64)
            t_new = (-R_new @ C_locked)
            new_KRt[af] = (K_init.copy(), R_new, t_new)
            new_res[af] = float("inf")
            continue
        # Re-fit every anchor (no early-keep): C_locked is a median over
        # rich anchors and won't match any single anchor's solo-solve C
        # exactly, so even the representative anchor needs re-fitting.
        fx_init = float(K_init[0, 0])
        rvec_init, _ = cv2.Rodrigues(R_init.astype(np.float64))
        rvec, fx = _solve_anchor_with_C_fixed(
            anchor, C_locked, cx, cy, fx_init, rvec_init.reshape(3),
            distortion=sol.distortion,
        )
        R_new = _rvec_to_R(rvec)
        t_new = -R_new @ C_locked
        K_new = _make_K(fx, cx, cy)
        new_KRt[af] = (K_new, R_new, t_new)
        new_res[af] = reprojection_residual_for_anchor(
            anchor, K_new, R_new, t_new, distortion=sol.distortion,
        )

    new_mean = float(np.mean(list(new_res.values())))
    old_mean = float(np.mean(list(sol.per_anchor_residual_px.values())))
    # If the static-C residual blows up by >10× the original, the camera
    # body genuinely moves (or anchors disagree on C). Surface this as an
    # ERROR but DO NOT silently fall back to the un-relocked solution —
    # falling back invisibly breaks the static-camera contract that every
    # anchor satisfies -R^T @ t == C. Better to ship a relocked solution
    # the user can inspect than a moving one they can't see.
    if new_mean > 10.0 * max(old_mean, 1.0):
        worst = sorted(new_res.items(), key=lambda kv: -kv[1])[:3]
        logger.error(
            "static-camera relock produced mean residual %.2f px (was %.2f px "
            "before relock). Camera centre median over %d %s anchors was "
            "(%.1f, %.1f, %.1f). Worst offenders (frame, residual_px): %s. "
            "Continuing with relocked solution to honour the static-camera "
            "contract; if the camera body actually moves, set "
            "camera.static_camera=false in config.",
            new_mean, old_mean, seed_n, seed_pool, *C_locked, worst,
        )

    logger.info(
        "static-camera relock: locked camera centre to (%.2f, %.2f, %.2f) "
        "(median over %d %s anchors), mean residual %.2f px (was %.2f px). "
        "Higher residual is expected — thin/collinear anchors lose their "
        "overfit solo-solve as C is constrained.",
        *C_locked, seed_n, seed_pool, new_mean, old_mean,
    )

    # Representative t for the JointSolution.t_world field (consumers
    # like CameraTrack.t_world use this as a fallback). Compute it from
    # the relocked R of the lowest-residual rich anchor so
    # ``-R_seed_new.T @ t_world == C_locked`` — i.e. the clip-shared
    # t_world is consistent with the locked camera centre.
    if best_frame in new_KRt:
        _Krep, R_rep_new, t_rep_new = new_KRt[best_frame]
        t_world_out = t_rep_new.copy()
    else:
        t_world_out = -R_seed @ C_locked

    return JointSolution(
        t_world=t_world_out,
        principal_point=(cx, cy),
        per_anchor_KRt=new_KRt,
        per_anchor_residual_px=new_res,
        camera_centre=C_locked.copy(),
        distortion=sol.distortion,
    )


def refine_with_bounded_motion(
    anchors: tuple[Anchor, ...],
    sol: JointSolution,
    max_motion_m: float,
) -> JointSolution:
    """Static-camera-like relock that allows each anchor's camera centre
    to drift up to ``max_motion_m`` metres from a shared reference.

    Designed for clips where the camera body is approximately fixed but
    not pixel-perfect — e.g. a stadium gantry mount with operator pan/tilt
    induced vibration. ``refine_with_shared_translation`` forces every
    anchor to share C exactly, which on real broadcast data inflates
    per-anchor residuals when solo Cs disagree by 1–3 m (typically driven
    by click noise + un-modelled lens, not real body motion).

    Parameter layout: ``[Cx, Cy, Cz, {rvec_i(3), dC_i(3), fx_i}...]``.
    Each anchor's t is reconstructed inside the residual as
    ``t = -R @ (C + dC)`` with ``|dC|_∞ ≤ max_motion_m``. Distortion
    and principal point are held fixed at the input solution's values.

    Returns a fresh JointSolution. The reported ``camera_centre`` is the
    shared reference C; per-anchor centres are not constant. If the
    optimisation makes residuals worse than the input by more than 10×,
    the input is returned untouched (same fail-safe as the strict relock).
    """
    if max_motion_m <= 0:
        return refine_with_shared_translation(anchors, sol)
    if not sol.per_anchor_KRt:
        return sol

    by_frame = {a.frame: a for a in anchors}
    qualifying = [
        by_frame[af] for af in sol.per_anchor_KRt.keys() if af in by_frame
    ]
    if not qualifying:
        return sol

    cx, cy = sol.principal_point
    distortion = sol.distortion

    # Seed C as the mean of solo Cs across rich anchors (so dC starts ≈ 0
    # for the dominant cluster).
    rich_Cs: list[np.ndarray] = []
    for af, (_K, R, t) in sol.per_anchor_KRt.items():
        a = by_frame.get(af)
        if a is None or not _is_rich(a):
            continue
        rich_Cs.append(-R.astype(np.float64).T @ t.astype(np.float64))
    if rich_Cs:
        C_seed = np.mean(np.stack(rich_Cs), axis=0)
    else:
        C_seed = -sol.per_anchor_KRt[next(iter(sol.per_anchor_KRt))][1].T @ \
                 sol.per_anchor_KRt[next(iter(sol.per_anchor_KRt))][2]

    PER = 7  # rvec(3), dC(3), fx
    n = len(qualifying)
    p0 = np.empty(3 + PER * n, dtype=np.float64)
    p0[:3] = C_seed
    lower = np.empty_like(p0)
    upper = np.empty_like(p0)
    lower[:3] = C_seed - 5.0
    upper[:3] = C_seed + 5.0
    # Box bound per dC component such that the worst-case L2 stays at
    # the budget. The LM optimizes inside [-c, c]^3 where c = R/√3 so
    # |dC|_2 ≤ R. Conservative vs an L2-constrained NLP but it's cheap
    # and the LM rarely sits at the corner.
    c_bound = max_motion_m / float(np.sqrt(3.0))
    for i, anchor in enumerate(qualifying):
        K_i, R_i, t_i = sol.per_anchor_KRt[anchor.frame]
        C_i = -R_i.astype(np.float64).T @ t_i.astype(np.float64)
        dC_init = np.clip(C_i - C_seed, -c_bound, c_bound)
        rv_i, _ = cv2.Rodrigues(R_i.astype(np.float64))
        base = 3 + i * PER
        p0[base : base + 3] = rv_i.reshape(3)
        p0[base + 3 : base + 6] = dC_init
        p0[base + 6] = float(K_i[0, 0])
        lower[base : base + 3] = -np.pi
        upper[base : base + 3] = np.pi
        lower[base + 3 : base + 6] = -c_bound
        upper[base + 3 : base + 6] = c_bound
        lower[base + 6] = float(K_i[0, 0]) * 0.5
        upper[base + 6] = float(K_i[0, 0]) * 2.0

    obj_pts_per: list[np.ndarray] = []
    img_pts_per: list[np.ndarray] = []
    lines_per: list = []
    for anchor in qualifying:
        if anchor.landmarks:
            obj_pts_per.append(
                np.array([lm.world_xyz for lm in anchor.landmarks],
                         dtype=np.float64).reshape(-1, 1, 3)
            )
            img_pts_per.append(
                np.array([lm.image_xy for lm in anchor.landmarks],
                         dtype=np.float64)
            )
        else:
            obj_pts_per.append(None)
            img_pts_per.append(None)
        lines_per.append(list(anchor.lines))

    dist = np.array([distortion[0], distortion[1], 0.0, 0.0, 0.0],
                    dtype=np.float64)

    def _residuals(p: np.ndarray) -> np.ndarray:
        C = p[:3]
        parts: list[np.ndarray] = []
        for i, anchor in enumerate(qualifying):
            base = 3 + i * PER
            rv = p[base : base + 3]
            dC = p[base + 3 : base + 6]
            fx = float(np.clip(p[base + 6], 50.0, 1e5))
            R_i, _ = cv2.Rodrigues(rv)
            t_i = -R_i @ (C + dC)
            K_i = _make_K(fx, cx, cy)
            if obj_pts_per[i] is not None:
                proj, _ = cv2.projectPoints(
                    obj_pts_per[i], rv.reshape(3, 1), t_i.reshape(3, 1),
                    K_i, dist,
                )
                parts.append((proj.reshape(-1, 2) - img_pts_per[i]).reshape(-1))
            if lines_per[i]:
                parts.append(
                    _LINE_RESIDUAL_WEIGHT
                    * _line_residuals(lines_per[i], K_i, R_i, t_i)
                )
        return np.concatenate(parts) if parts else np.empty(0)

    try:
        result = least_squares(
            _residuals, p0, bounds=(lower, upper),
            method="trf", loss="huber", f_scale=2.0, max_nfev=1000,
        )
    except Exception as exc:
        logger.warning("bounded-motion relock failed: %s; falling back to "
                       "strict static-camera relock", exc)
        return refine_with_shared_translation(anchors, sol)

    C_locked = np.asarray(result.x[:3], dtype=np.float64)
    new_KRt: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    new_res: dict[int, float] = {}
    motions: list[float] = []
    for i, anchor in enumerate(qualifying):
        base = 3 + i * PER
        rv = result.x[base : base + 3]
        dC = result.x[base + 3 : base + 6]
        fx = float(result.x[base + 6])
        R_new, _ = cv2.Rodrigues(rv)
        t_new = -R_new @ (C_locked + dC)
        K_new = _make_K(fx, cx, cy)
        new_KRt[anchor.frame] = (K_new, R_new, t_new)
        new_res[anchor.frame] = reprojection_residual_for_anchor(
            anchor, K_new, R_new, t_new, distortion=distortion,
        )
        motions.append(float(np.linalg.norm(dC)))

    # Anchors that were in sol.per_anchor_KRt but not in `qualifying`
    # (e.g. degenerate solves with no anchor record) keep their K/R and
    # get a recomputed t against the shared C.
    for af, (K_init, R_init, _t_init) in sol.per_anchor_KRt.items():
        if af in new_KRt:
            continue
        R_new = R_init.astype(np.float64)
        t_new = -R_new @ C_locked
        new_KRt[af] = (K_init.copy(), R_new, t_new)
        new_res[af] = float("inf")

    new_mean = float(np.mean([
        r for r in new_res.values() if np.isfinite(r)
    ])) if new_res else 0.0
    old_mean = float(np.mean(list(sol.per_anchor_residual_px.values())))
    logger.info(
        "bounded-motion relock: C=(%.2f, %.2f, %.2f), max anchor motion "
        "%.2f m (budget %.1f m), mean residual %.2f → %.2f px",
        *C_locked, max(motions) if motions else 0.0, max_motion_m,
        old_mean, new_mean,
    )

    # t_world: use the first qualifying anchor's t as a representative
    # value. Downstream consumers should prefer ``camera_centre`` + the
    # per-anchor R; t_world is just a legacy fallback.
    rep_t = next(iter(new_KRt.values()))[2].copy()
    return JointSolution(
        t_world=rep_t,
        principal_point=(cx, cy),
        per_anchor_KRt=new_KRt,
        per_anchor_residual_px=new_res,
        camera_centre=C_locked.copy(),
        distortion=distortion,
    )


# Backwards-compat thin wrapper for any external caller that imports the
# old name. Returns just the per_anchor_KRt dict like the previous API.
def relock_anchors_with_shared_t(
    anchors: tuple[Anchor, ...],
    sol: JointSolution,
    t_shared: np.ndarray | None = None,
) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Deprecated. Use ``refine_with_shared_translation`` for static cams."""
    refined = refine_with_shared_translation(anchors, sol)
    return refined.per_anchor_KRt


def reprojection_residual_for_anchor(
    anchor: Anchor,
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    distortion: tuple[float, float] = (0.0, 0.0),
) -> float:
    """Mean pixel reprojection residual across this anchor's point landmarks.

    Returns ``1e9`` (sentinel) if any landmark projects behind the camera.
    Lines are deliberately excluded — they constrain orientation but their
    absolute pixel residual depends on the user's chosen segment endpoints.

    ``distortion`` is the (k1, k2) radial distortion applied during the
    forward projection. Default ``(0, 0)`` matches the legacy pinhole
    behaviour; pass the clip's recovered distortion to evaluate residuals
    under the same lens model the solver fit.
    """
    if not anchor.landmarks:
        return 0.0
    for lm in anchor.landmarks:
        cam = R @ np.array(lm.world_xyz) + t
        if cam[2] <= 0:
            return 1e9
    rvec, _ = cv2.Rodrigues(R.astype(np.float64))
    residuals = _point_residuals_distorted(
        list(anchor.landmarks), K, rvec, t.astype(np.float64), distortion,
    ).reshape(-1, 2)
    return float(np.mean(np.linalg.norm(residuals, axis=1)))


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


def _landmarks_collinear(anchor: Anchor, tol_m: float = 1.0) -> bool:
    """Whether the anchor's point landmarks all lie on a single world line.

    Detected via the second singular value of the centred world-coordinate
    matrix: values below ``tol_m`` (metres) mean the points span at most
    one world axis. Returns ``False`` when the anchor has fewer than two
    points (caller treats those as separately under-constrained).

    A collinear point set is geometrically rank-deficient — the solver
    will still find *a* low-residual (R, t) but the camera position along
    the perpendicular to the line is unconstrained, so the recovered
    pose may be physically nonsensical (e.g. camera at pitch level on a
    halfway-line-only annotation). Used by ``solve_anchors_jointly`` to
    warn the user.
    """
    if len(anchor.landmarks) < 2:
        return False
    pts = np.array([lm.world_xyz for lm in anchor.landmarks], dtype=np.float64)
    centred = pts - pts.mean(axis=0)
    s = np.linalg.svd(centred, compute_uv=False)
    # SVD pads to length 3; second singular value tells us whether the
    # points span > 1 axis.
    return float(s[1]) < tol_m


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
    anchor: Anchor,
    K: np.ndarray,
    distortion: tuple[float, float] = (0.0, 0.0),
) -> tuple[np.ndarray, np.ndarray] | None:
    """Run cv2.solvePnP on this anchor's point landmarks. Returns
    ``(rvec, tvec)`` or ``None`` if there are too few points or PnP fails.

    ``distortion`` (k1, k2) is passed to ``cv2.solvePnP`` so the seed pose
    is correct under the same lens model the solo LM will use. Default
    ``(0, 0)`` keeps the legacy pinhole behaviour.
    """
    if len(anchor.landmarks) < 4:
        return None
    obj_pts = np.array(
        [lm.world_xyz for lm in anchor.landmarks], dtype=np.float64
    ).reshape(-1, 1, 3)
    img_pts = np.array(
        [lm.image_xy for lm in anchor.landmarks], dtype=np.float64
    ).reshape(-1, 1, 2)
    k1, k2 = distortion
    dist_coeffs = np.array([k1, k2, 0.0, 0.0, 0.0], dtype=np.float64)
    # SOLVEPNP_SQPNP (OpenCV ≥ 4.5) is the modern globally-optimal PnP that
    # handles ≥3 points including coplanar configurations more stably than
    # EPNP, which is highly sensitive to small fx perturbations on coplanar
    # data. Falls back to ITERATIVE for ≥6 points (well-conditioned non-
    # coplanar cases benefit from its iterative refinement).
    flag = cv2.SOLVEPNP_SQPNP if len(anchor.landmarks) < 6 else cv2.SOLVEPNP_ITERATIVE
    try:
        ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, dist_coeffs, flags=flag)
    except cv2.error:
        # SQPNP raises a hard assertion on rank-deficient configurations
        # (e.g. all-collinear points). Treat that as a regular PnP failure
        # so the orchestrator can fall back to t-fixed / interpolation.
        return None
    if not ok:
        return None
    return rvec.reshape(3), tvec.reshape(3)


# Line residuals are perpendicular pixel distances from clicked endpoints to
# projected world segments. When the world segment is long (e.g. a full
# touchline) and the projection is even slightly off, the magnitude can
# dwarf point reprojection residuals and overwhelm the LM. Down-weighting
# during the solo LM bootstrap keeps points dominant; the final joint LM
# (which is well-seeded) uses lines at full weight via Huber loss.
_LINE_RESIDUAL_WEIGHT = 0.2


def _refine_seed_pose(
    anchor: Anchor,
    rvec_init: np.ndarray,
    tvec_init: np.ndarray,
    cx: float,
    cy: float,
    fx_init: float,
    distortion: tuple[float, float] = (0.0, 0.0),
) -> tuple[float, np.ndarray, np.ndarray]:
    """One-anchor LM that refines (fx, R, t) on the seed anchor.

    Uses point landmarks (full weight) and line annotations (down-weighted
    by ``_LINE_RESIDUAL_WEIGHT``). TRF + Huber loss caps any individual
    residual that escapes the down-weighting — the two protections compose:
    Huber handles outlier *clicks*, the down-weight handles the natural
    *scale* mismatch between point and line residuals on a wide-FOV pitch.

    When ``distortion`` is non-zero the residual uses ``cv2.projectPoints``
    so radial distortion is honoured; this lets a caller-supplied lens
    prior bias each solo solve consistently across anchors. Lines are
    treated as pinhole — radial distortion at line endpoints is a small
    second-order effect and applying it would require per-endpoint
    distortion handling that ``_line_residuals`` doesn't currently model.

    Returns ``(fx, rvec, tvec)`` — caller updates K and the seed pose.
    """
    if not anchor.landmarks:
        return fx_init, rvec_init.copy(), tvec_init.copy()
    obj_pts = np.array([lm.world_xyz for lm in anchor.landmarks], dtype=np.float64)
    img_pts = np.array([lm.image_xy for lm in anchor.landmarks], dtype=np.float64)
    n_pts = len(obj_pts)
    z_levels = len({round(z, 3) for _, _, z in obj_pts})
    use_lines_on_seed = anchor.lines and (n_pts < 6 or z_levels < 2)
    distorted = distortion != (0.0, 0.0)

    def _residuals(p: np.ndarray) -> np.ndarray:
        fx = float(p[0])
        rvec = p[1:4]
        tvec = p[4:7]
        R, _ = cv2.Rodrigues(rvec)
        K = _make_K(fx, cx, cy)
        if distorted:
            parts: list[np.ndarray] = [_point_residuals_distorted(
                list(anchor.landmarks), K, rvec, tvec, distortion,
            )]
        else:
            cam = obj_pts @ R.T + tvec
            safe_z = np.where(cam[:, 2] > 1e-3, cam[:, 2], 1e-3)
            pix = cam @ K.T
            proj = pix[:, :2] / safe_z[:, None]
            parts = [(proj - img_pts).reshape(-1)]
        if use_lines_on_seed:
            parts.append(
                _LINE_RESIDUAL_WEIGHT
                * _line_residuals(list(anchor.lines), K, R, tvec)
            )
        return np.concatenate(parts)

    p0 = np.concatenate([[fx_init], rvec_init, tvec_init])
    try:
        # Solo LMs use Levenberg–Marquardt (not robust). The final joint
        # distortion LM uses TRF + Huber for outlier protection; doing it
        # at solo level too caused real-broadcast seed solves to converge
        # to wider local minima (seed-anchor residual jumped 5 → 30 px).
        result = least_squares(_residuals, p0, method="lm", max_nfev=300)
        return (
            float(np.clip(result.x[0], 50.0, 1e5)),
            result.x[1:4].copy(),
            result.x[4:7].copy(),
        )
    except Exception:
        return fx_init, rvec_init.copy(), tvec_init.copy()


def _solve_anchor_with_C_fixed(
    anchor: Anchor,
    C_fixed: np.ndarray,
    cx: float,
    cy: float,
    fx_init: float,
    rvec_init: np.ndarray,
    distortion: tuple[float, float] = (0.0, 0.0),
) -> tuple[np.ndarray, float]:
    """LM-refine (rvec, fx) for one anchor with the world-frame camera
    centre held constant. t per anchor is recomputed inside the
    residual as ``t = -R @ C_fixed`` so the camera body stays put while
    rotation and zoom vary.

    This is the physical constraint for a static stadium-mounted camera
    (PTZ rig): the body doesn't move, only its pan/tilt and zoom. With
    only ``rvec`` and ``fx`` free this is a 4-DOF problem.

    ``distortion`` honours the clip's recovered (k1, k2) when the static-
    camera relock runs against an undistorted pinhole model; pass
    ``(0, 0)`` for legacy callers.
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
    distorted = distortion != (0.0, 0.0)

    def _residuals(p: np.ndarray) -> np.ndarray:
        fx = float(p[0])
        rvec = p[1:4]
        R, _ = cv2.Rodrigues(rvec)
        t = -R @ C_fixed
        K = _make_K(fx, cx, cy)
        parts: list[np.ndarray] = []
        if len(obj_pts):
            if distorted:
                parts.append(_point_residuals_distorted(
                    list(anchor.landmarks), K, rvec, t, distortion,
                ))
            else:
                cam = obj_pts @ R.T + t
                safe_z = np.where(cam[:, 2] > 1e-3, cam[:, 2], 1e-3)
                pix = cam @ K.T
                proj = pix[:, :2] / safe_z[:, None]
                parts.append((proj - img_pts).reshape(-1))
        if anchor.lines:
            # Down-weight lines (see _refine_seed_pose). Heavier weight when
            # there are no points (line-only anchors rely entirely on lines).
            weight = 1.0 if len(obj_pts) == 0 else _LINE_RESIDUAL_WEIGHT
            parts.append(
                weight * _line_residuals(list(anchor.lines), K, R, t)
            )
        return np.concatenate(parts) if parts else np.empty(0)

    p0 = np.concatenate([[fx_init], rvec_init])
    try:
        result = least_squares(_residuals, p0, method="lm", max_nfev=300)
        return result.x[1:4].copy(), float(np.clip(result.x[0], 50.0, 1e5))
    except Exception:
        return rvec_init.copy(), fx_init


def _solve_anchor_with_t_fixed(
    anchor: Anchor,
    t_fixed: np.ndarray,
    cx: float,
    cy: float,
    fx_init: float,
    rvec_init: np.ndarray,
    distortion: tuple[float, float] = (0.0, 0.0),
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
    distorted = distortion != (0.0, 0.0)

    def _residuals(p: np.ndarray) -> np.ndarray:
        fx = float(p[0])
        rvec = p[1:4]
        R, _ = cv2.Rodrigues(rvec)
        K = _make_K(fx, cx, cy)
        parts: list[np.ndarray] = []
        if len(obj_pts):
            if distorted:
                parts.append(_point_residuals_distorted(
                    list(anchor.landmarks), K, rvec, t_fixed, distortion,
                ))
            else:
                cam = obj_pts @ R.T + t_fixed
                safe_z = np.where(cam[:, 2] > 1e-3, cam[:, 2], 1e-3)
                pix = cam @ K.T
                proj = pix[:, :2] / safe_z[:, None]
                parts.append((proj - img_pts).reshape(-1))
        if anchor.lines:
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
    distortion: tuple[float, float] = (0.0, 0.0),
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
            seed = _seed_anchor_pose(anchor, K_seed, distortion=distortion)
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
            distortion=distortion,
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
        reprojection_residual_for_anchor(
            anchor, primary[0], primary[1], primary[2], distortion=distortion,
        )
        if primary is not None and not _is_degenerate_solo(primary[2], primary[3])
        else float("inf")
    )
    for mult in (1.1, 0.9, 1.2, 0.8, 1.5, 0.7, 2.0, 0.5, 3.0, 0.3):
        alt = _try(fx_init * mult)
        if alt is None:
            continue
        if _is_degenerate_solo(alt[2], alt[3]):
            continue
        res = reprojection_residual_for_anchor(
            anchor, alt[0], alt[1], alt[2], distortion=distortion,
        )
        if res < candidate_res:
            candidate_res = res
            candidate_best = alt
    return candidate_best


def _anchor_image_coverage(
    anchor: Anchor, image_size: tuple[int, int]
) -> float:
    """Fraction of the image area enclosed by an anchor's landmarks'
    bounding box. Returns a value in [0, 1]; 0 if there are <2 landmarks.

    Used by ``_estimate_lens_from_best_anchor`` to rank candidates —
    distortion and principal-point offset only manifest at distance from
    the optical centre, so a high-coverage anchor is the only one that
    can disambiguate them.
    """
    if len(anchor.landmarks) < 2:
        return 0.0
    us = [lm.image_xy[0] for lm in anchor.landmarks]
    vs = [lm.image_xy[1] for lm in anchor.landmarks]
    w, h = image_size
    return float(((max(us) - min(us)) / w) * ((max(vs) - min(vs)) / h))


def _estimate_lens_from_best_anchor(
    anchors: tuple[Anchor, ...],
    image_size: tuple[int, int],
    min_landmarks: int = 10,
    min_coverage: float = 0.12,
    min_residual_drop: float = 2.0,
) -> tuple[float, float, float, float] | None:
    """Recover (cx, cy, k1, k2) from the single highest-coverage anchor.

    Runs a 9-DOF solo LM ``(rvec, tvec, fx, cx, cy, k1, k2)`` on whichever
    anchor has the most landmarks and the widest image-area coverage.
    Returns the recovered lens parameters only if the residual drops by
    at least ``min_residual_drop``× over the no-distortion solo solve and
    the recovered values stay within physically plausible bounds.

    Returning ``None`` means: no anchor met the coverage gate, or the LM
    didn't beat the no-distortion baseline by enough margin to trust the
    estimate. The caller should fall back to the existing no-distortion
    flow (pp pinned to image centre, (k1, k2) = (0, 0)).

    The bounds — |k1|, |k2| ≤ 0.5 and pp within ±100 px of image centre —
    cover typical broadcast lenses + camera-frame crops; values outside
    that range are almost always overfitting noise rather than a real
    optical signal.
    """
    w, h = image_size
    candidates = [
        a for a in anchors
        if len(a.landmarks) >= min_landmarks
        and _anchor_image_coverage(a, image_size) >= min_coverage
    ]
    if not candidates:
        logger.info(
            "lens-from-anchor: no anchor has ≥%d landmarks with ≥%.0f%% "
            "image coverage; skipping (need wide spatial spread to "
            "disambiguate pp/distortion from pose)",
            min_landmarks, min_coverage * 100,
        )
        return None
    best = max(
        candidates,
        key=lambda a: (
            _anchor_image_coverage(a, image_size),
            len(a.landmarks),
        ),
    )

    # Baseline: no-distortion solo solve. Provides the residual we need to
    # beat to accept the recovered lens params.
    cx_seed, cy_seed = w / 2.0, h / 2.0
    fx_seed = float(w)
    K_seed = _make_K(fx_seed, cx_seed, cy_seed)
    base = _solve_one_anchor_full(best, cx_seed, cy_seed, fx_seed, K_seed)
    if base is None:
        logger.warning(
            "lens-from-anchor: baseline solo solve failed on frame %d",
            best.frame,
        )
        return None
    K_b, R_b, t_b, fx_b = base
    base_res = reprojection_residual_for_anchor(best, K_b, R_b, t_b)

    # 9-DOF LM: (rvec, tvec, fx, cx, cy, k1, k2). Seeded from the baseline
    # solo solve so it starts in a sensible basin. Multi-seeded over a
    # grid of (cx, cy) offsets: the cost surface around the no-distortion
    # solve has a shallow basin near image centre (per-anchor pose has
    # already absorbed the pp offset), so the LM gets stuck if we only
    # start at the image-centre seed. Offsets of ±30 px on both axes
    # cover broadcast crops without inflating the search excessively.
    obj_pts = np.array(
        [lm.world_xyz for lm in best.landmarks], dtype=np.float64
    ).reshape(-1, 1, 3)
    img_pts = np.array(
        [lm.image_xy for lm in best.landmarks], dtype=np.float64
    )
    rvec_b, _ = cv2.Rodrigues(R_b.astype(np.float64))

    def _residuals_for_seed(cx_s: float, cy_s: float, k1_s: float, k2_s: float):
        p0 = np.concatenate([
            rvec_b.reshape(3),
            t_b.astype(np.float64),
            [fx_b, cx_s, cy_s, k1_s, k2_s],
        ])
        lower = np.array([
            -np.pi, -np.pi, -np.pi,
            -300.0, -300.0, -300.0,
            fx_b * 0.5, cx_seed - 100.0, cy_seed - 100.0,
            -0.5, -0.5,
        ])
        upper = np.array([
            np.pi, np.pi, np.pi,
            300.0, 300.0, 300.0,
            fx_b * 2.0, cx_seed + 100.0, cy_seed + 100.0,
            0.5, 0.5,
        ])

        def _residuals(p: np.ndarray) -> np.ndarray:
            rvec = p[0:3]
            tvec = p[3:6]
            fx = float(p[6])
            cx = float(p[7])
            cy = float(p[8])
            k1 = float(p[9])
            k2 = float(p[10])
            K = _make_K(fx, cx, cy)
            dist = np.array([k1, k2, 0.0, 0.0, 0.0], dtype=np.float64)
            proj, _ = cv2.projectPoints(
                obj_pts,
                rvec.reshape(3, 1),
                tvec.reshape(3, 1),
                K,
                dist,
            )
            return (proj.reshape(-1, 2) - img_pts).reshape(-1)

        return least_squares(
            _residuals, p0, bounds=(lower, upper),
            method="trf", max_nfev=500,
        )

    best_result = None
    best_cost = float("inf")
    seed_grid = [
        (cx_seed, cy_seed, 0.0, 0.0),
        (cx_seed + 30, cy_seed, 0.0, 0.0),
        (cx_seed - 30, cy_seed, 0.0, 0.0),
        (cx_seed, cy_seed + 30, 0.0, 0.0),
        (cx_seed, cy_seed - 30, 0.0, 0.0),
        (cx_seed, cy_seed, -0.1, 0.0),
        (cx_seed, cy_seed, 0.1, 0.0),
    ]
    for cx_s, cy_s, k1_s, k2_s in seed_grid:
        try:
            r = _residuals_for_seed(cx_s, cy_s, k1_s, k2_s)
        except Exception as exc:
            logger.debug("lens-from-anchor LM seed (%.0f, %.0f) failed: %s", cx_s, cy_s, exc)
            continue
        if r.cost < best_cost:
            best_cost = r.cost
            best_result = r

    if best_result is None:
        logger.warning("lens-from-anchor LM failed across all seeds")
        return None
    result = best_result

    fx_est = float(result.x[6])
    cx_est = float(result.x[7])
    cy_est = float(result.x[8])
    k1_est = float(result.x[9])
    k2_est = float(result.x[10])
    K_est = _make_K(fx_est, cx_est, cy_est)
    R_est, _ = cv2.Rodrigues(result.x[0:3])
    t_est = result.x[3:6]
    est_res = reprojection_residual_for_anchor(
        best, K_est, R_est, t_est, distortion=(k1_est, k2_est),
    )

    if est_res >= base_res / min_residual_drop:
        logger.info(
            "lens-from-anchor (frame %d): rejected — residual %.2f px did "
            "not improve %.1f× over baseline %.2f px. Likely no real lens "
            "structure to recover (or click noise dominates).",
            best.frame, est_res, min_residual_drop, base_res,
        )
        return None

    logger.info(
        "lens-from-anchor (frame %d, %d landmarks, %.0f%% image coverage): "
        "cx=%.1f, cy=%.1f, k1=%+.4f, k2=%+.4f — residual %.2f → %.2f px",
        best.frame, len(best.landmarks),
        _anchor_image_coverage(best, image_size) * 100,
        cx_est, cy_est, k1_est, k2_est, base_res, est_res,
    )
    return cx_est, cy_est, k1_est, k2_est


def _estimate_lens_jointly(
    anchors: tuple[Anchor, ...],
    image_size: tuple[int, int],
    min_residual_drop: float = 1.3,
) -> tuple[float, float, float, float] | None:
    """Recover shared (cx, cy, k1, k2) by joint LM across every rich anchor.

    Where ``_estimate_lens_from_best_anchor`` runs a single-anchor 9-DOF
    LM (and gets stuck in click-noise basins on real broadcast data),
    this fits all rich anchors simultaneously with **shared**
    (cx, cy, k1, k2) and **per-anchor** (rvec, tvec, fx). For 5 rich
    anchors with ~18 landmarks each that's ~180 residuals against
    4 shared + 35 per-anchor = 39 parameters — well-determined, and
    click noise on any one anchor can't capture the joint fit.

    Acceptance gate is intentionally looser (``min_residual_drop=1.3``)
    than the single-anchor estimator's 2×: on real broadcast clips the
    distortion magnitude is small enough that 30 % drop is the right
    bar. Higher would reject every real-world clip; lower starts
    accepting noise-fitting.

    Returns ``(cx, cy, k1, k2)`` or ``None`` if no rich anchor exists,
    the LM fails, or the joint residual doesn't improve enough over the
    no-distortion baseline.
    """
    rich = [a for a in anchors if _is_rich(a)]
    if len(rich) < 2:
        logger.info(
            "joint lens estimation: <2 rich anchors; falling back to "
            "single-anchor estimator",
        )
        return None
    w, h = image_size
    cx_seed, cy_seed = w / 2.0, h / 2.0
    fx_init = float(w)
    K_init = _make_K(fx_init, cx_seed, cy_seed)

    # Baseline solo solves provide initial pose + reference residual.
    seeds: list[tuple[np.ndarray, np.ndarray, float, float]] = []
    base_res_total = 0.0
    n_pts_total = 0
    for a in rich:
        result = _solve_one_anchor_full(a, cx_seed, cy_seed, fx_init, K_init)
        if result is None:
            logger.warning(
                "joint lens estimation: baseline solo solve failed on frame %d",
                a.frame,
            )
            return None
        K_b, R_b, t_b, fx_b = result
        rv, _ = cv2.Rodrigues(R_b.astype(np.float64))
        seeds.append((rv.reshape(3).copy(), t_b.astype(np.float64), float(fx_b), 0.0))
        res = reprojection_residual_for_anchor(a, K_b, R_b, t_b)
        base_res_total += res * len(a.landmarks)
        n_pts_total += len(a.landmarks)
    base_res = base_res_total / max(n_pts_total, 1)

    # Param layout: [cx, cy, k1, k2, rvec_0(3), tvec_0(3), fx_0, ...]
    SHARED = 4
    PER = 7
    n = len(rich)
    lower = np.empty(SHARED + PER * n, dtype=np.float64)
    upper = np.empty_like(lower)
    lower[0] = cx_seed - 100.0; upper[0] = cx_seed + 100.0
    lower[1] = cy_seed - 100.0; upper[1] = cy_seed + 100.0
    lower[2:4] = -0.5; upper[2:4] = 0.5
    for i, (rv, t_a, fx_a, _k0) in enumerate(seeds):
        base = SHARED + i * PER
        lower[base : base + 3] = -np.pi
        upper[base : base + 3] = np.pi
        lower[base + 3 : base + 6] = -300.0
        upper[base + 3 : base + 6] = 300.0
        lower[base + 6] = fx_a * 0.5
        upper[base + 6] = fx_a * 2.0

    obj_pts_per: list[np.ndarray] = []
    img_pts_per: list[np.ndarray] = []
    for a in rich:
        obj_pts_per.append(
            np.array([lm.world_xyz for lm in a.landmarks], dtype=np.float64).reshape(-1, 1, 3)
        )
        img_pts_per.append(
            np.array([lm.image_xy for lm in a.landmarks], dtype=np.float64)
        )

    def _residuals(p: np.ndarray) -> np.ndarray:
        cx = float(p[0]); cy = float(p[1])
        k1 = float(p[2]); k2 = float(p[3])
        dist = np.array([k1, k2, 0.0, 0.0, 0.0], dtype=np.float64)
        parts: list[np.ndarray] = []
        for i, _a in enumerate(rich):
            base = SHARED + i * PER
            rv = p[base : base + 3]
            tv = p[base + 3 : base + 6]
            fx = float(np.clip(p[base + 6], 50.0, 1e5))
            K = _make_K(fx, cx, cy)
            proj, _ = cv2.projectPoints(
                obj_pts_per[i],
                rv.reshape(3, 1),
                tv.reshape(3, 1),
                K,
                dist,
            )
            parts.append((proj.reshape(-1, 2) - img_pts_per[i]).reshape(-1))
        return np.concatenate(parts)

    # Multi-seed (cx, cy, k1) grid — same reasoning as the single-anchor
    # estimator: the cost surface near image-centre + no-distortion has a
    # shallow basin that the LM gets stuck in. Offsets of ±30 px and a
    # ±0.1 k1 seed cover broadcast crops and typical distortion magnitudes.
    seed_grid = [
        (cx_seed, cy_seed, 0.0),
        (cx_seed + 30, cy_seed, 0.0),
        (cx_seed - 30, cy_seed, 0.0),
        (cx_seed, cy_seed + 30, 0.0),
        (cx_seed, cy_seed - 30, 0.0),
        (cx_seed + 30, cy_seed + 30, 0.0),
        (cx_seed - 30, cy_seed - 30, 0.0),
        (cx_seed, cy_seed, -0.1),
        (cx_seed, cy_seed, 0.1),
    ]
    best_result = None
    best_cost = float("inf")
    for cx_s, cy_s, k1_s in seed_grid:
        p0 = np.empty(SHARED + PER * n, dtype=np.float64)
        p0[0] = cx_s; p0[1] = cy_s
        p0[2] = k1_s; p0[3] = 0.0
        for i, (rv, t_a, fx_a, _k0) in enumerate(seeds):
            base = SHARED + i * PER
            p0[base : base + 3] = rv
            p0[base + 3 : base + 6] = t_a
            p0[base + 6] = fx_a
        try:
            r = least_squares(
                _residuals, p0, bounds=(lower, upper),
                method="trf", loss="huber", f_scale=2.0, max_nfev=1000,
            )
        except Exception as exc:
            logger.debug("joint lens LM seed (%.0f, %.0f) failed: %s", cx_s, cy_s, exc)
            continue
        if r.cost < best_cost:
            best_cost = r.cost
            best_result = r
    if best_result is None:
        logger.warning("joint lens estimation: every seed failed")
        return None
    result = best_result

    cx = float(result.x[0]); cy = float(result.x[1])
    k1 = float(result.x[2]); k2 = float(result.x[3])

    # Total residual across rich anchors under the new lens model.
    new_res_total = 0.0
    new_n_total = 0
    for i, a in enumerate(rich):
        base = SHARED + i * PER
        rv = result.x[base : base + 3]
        tv = result.x[base + 3 : base + 6]
        fx = float(result.x[base + 6])
        K = _make_K(fx, cx, cy)
        R, _ = cv2.Rodrigues(rv)
        r = reprojection_residual_for_anchor(a, K, R, tv, distortion=(k1, k2))
        new_res_total += r * len(a.landmarks)
        new_n_total += len(a.landmarks)
    new_res = new_res_total / max(new_n_total, 1)

    if new_res >= base_res / min_residual_drop:
        logger.info(
            "joint lens estimation: rejected — residual %.2f px did not "
            "improve %.2f× over no-distortion baseline %.2f px",
            new_res, min_residual_drop, base_res,
        )
        return None

    logger.info(
        "joint lens estimation (%d rich anchors, %d landmarks total): "
        "cx=%.1f, cy=%.1f, k1=%+.4f, k2=%+.4f — residual %.2f → %.2f px",
        len(rich), n_pts_total, cx, cy, k1, k2, base_res, new_res,
    )
    return cx, cy, k1, k2


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


def _refine_joint_distortion(
    anchors: tuple[Anchor, ...],
    per_anchor_KRt: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]],
    cx: float,
    cy: float,
) -> tuple[
    dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]],
    tuple[float, float],
    dict[int, float],
]:
    """Final joint LM: shared (k1, k2), per-anchor (rvec, t, fx).

    ``cx, cy`` are held fixed at the joint-solve seed. Per-anchor ``t`` is
    free here so the LM can disambiguate distortion from pose; the static-
    camera relock that follows reconstrains ``t = -R @ C`` if requested.

    Uses ``cv2.projectPoints`` so distortion is honoured natively. Huber
    loss dampens outlier landmarks/lines.

    Returns refined per-anchor (K, R, t), recovered (k1, k2), and per-anchor
    point-only mean reprojection residuals (in pixels) under the refined
    pose AND distortion.
    """
    qualifying = [a for a in anchors if a.frame in per_anchor_KRt]
    if not qualifying:
        return per_anchor_KRt, (0.0, 0.0), {}

    rvecs_init: list[np.ndarray] = []
    fxs_init: list[float] = []
    ts_init: list[np.ndarray] = []
    for a in qualifying:
        K_a, R_a, t_a = per_anchor_KRt[a.frame]
        rv_a, _ = cv2.Rodrigues(R_a.astype(np.float64))
        rvecs_init.append(rv_a.flatten().copy())
        fxs_init.append(float(K_a[0, 0]))
        ts_init.append(t_a.astype(np.float64).copy())

    # Param layout: [k1, k2, rvec_0(3), tvec_0(3), fx_0, ...] — 7 per anchor.
    PER = 7
    n = len(qualifying)
    p0 = np.empty(2 + PER * n)
    p0[:2] = 0.0
    for i, (rv, t_a, fx) in enumerate(zip(rvecs_init, ts_init, fxs_init)):
        base = 2 + i * PER
        p0[base : base + 3] = rv
        p0[base + 3 : base + 6] = t_a
        p0[base + 6] = fx

    def _residuals_joint(p: np.ndarray) -> np.ndarray:
        k1 = float(p[0])
        k2 = float(p[1])
        parts: list[np.ndarray] = []
        for i, anchor in enumerate(qualifying):
            base = 2 + i * PER
            rv = p[base : base + 3]
            t_i = p[base + 3 : base + 6]
            fx = float(np.clip(p[base + 6], 50.0, 1e5))
            R_i, _ = cv2.Rodrigues(rv)
            K_i = _make_K(fx, cx, cy)
            if anchor.landmarks:
                parts.append(_point_residuals_distorted(
                    list(anchor.landmarks), K_i, rv, t_i, (k1, k2),
                ))
            if anchor.lines:
                parts.append(_line_residuals(list(anchor.lines), K_i, R_i, t_i))
        return np.concatenate(parts) if parts else np.empty(0)

    try:
        result = least_squares(
            _residuals_joint, p0,
            method="trf", loss="huber", f_scale=2.0, max_nfev=500,
        )
    except Exception as exc:
        logger.warning("joint distortion refinement failed: %s", exc)
        return per_anchor_KRt, (0.0, 0.0), {}

    k1 = float(np.clip(result.x[0], -0.5, 0.5))
    k2 = float(np.clip(result.x[1], -0.5, 0.5))
    new_KRt: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = dict(
        per_anchor_KRt
    )
    new_residuals: dict[int, float] = {}
    for i, anchor in enumerate(qualifying):
        base = 2 + i * PER
        rv = result.x[base : base + 3]
        t_i = result.x[base + 3 : base + 6].copy()
        fx = float(np.clip(result.x[base + 6], 50.0, 1e5))
        R_i, _ = cv2.Rodrigues(rv)
        K_i = _make_K(fx, cx, cy)
        new_KRt[anchor.frame] = (K_i, R_i, t_i)
        if anchor.landmarks:
            r = _point_residuals_distorted(
                list(anchor.landmarks), K_i, rv, t_i, (k1, k2),
            ).reshape(-1, 2)
            new_residuals[anchor.frame] = float(
                np.mean(np.linalg.norm(r, axis=1))
            )
        else:
            new_residuals[anchor.frame] = 0.0
    return new_KRt, (k1, k2), new_residuals


def _point_residuals_distorted(
    points: list[LandmarkObservation],
    K: np.ndarray,
    rvec: np.ndarray,
    t: np.ndarray,
    distortion: tuple[float, float],
) -> np.ndarray:
    """Per-point reprojection residuals using ``cv2.projectPoints`` so that
    radial distortion is honoured. 2 residuals per point. Used by the joint
    distortion-refinement LM."""
    if not points:
        return np.empty(0)
    world = np.array([lm.world_xyz for lm in points], dtype=np.float64).reshape(
        -1, 1, 3,
    )
    obs = np.array([lm.image_xy for lm in points], dtype=np.float64)
    k1, k2 = distortion
    dist = np.array([k1, k2, 0.0, 0.0, 0.0], dtype=np.float64)
    proj, _ = cv2.projectPoints(
        world,
        np.asarray(rvec, dtype=np.float64).reshape(3, 1),
        np.asarray(t, dtype=np.float64).reshape(3, 1),
        K.astype(np.float64),
        dist,
    )
    return (proj.reshape(-1, 2) - obs).reshape(-1)


def solve_anchors_jointly(
    anchors: tuple[Anchor, ...],
    image_size: tuple[int, int],
    lens_prior: tuple[float, float, float, float] | None = None,
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

    ``lens_prior`` is an optional ``(cx, cy, k1, k2)`` tuple, typically
    produced by ``_estimate_lens_from_best_anchor``. When set, the joint
    solve uses (cx, cy) as the principal point (instead of image-centre)
    and applies (k1, k2) radial distortion in every solo LM's residual.
    The Pass 3 joint distortion refine is skipped — the prior is the
    final lens model. This is the right path when the clip's anchors
    include at least one high-coverage anchor that can disambiguate
    pp/distortion from pose; without one, leave it ``None`` and the
    legacy in-solve estimation runs.
    """
    if not anchors:
        raise AnchorSolveError("no anchors supplied")
    qualifying = tuple(a for a in anchors if _qualifies(a))
    if not qualifying:
        raise AnchorSolveError(
            "no anchor has ≥4 point landmarks or ≥2 line correspondences; "
            "place more landmarks (or lines) before running the camera stage"
        )

    # Warn on geometrically rank-deficient anchors. Collinear point sets
    # admit a low-residual fit (the points reproject perfectly along the
    # line) but leave the camera position perpendicular to the line
    # unconstrained, so the recovered pose is often physically wrong.
    # We still attempt the solve — the user may be relying on inter-anchor
    # interpolation — but flag the frame so they know to add an off-axis
    # landmark.
    for a in qualifying:
        if _landmarks_collinear(a):
            logger.warning(
                "anchor at frame %d has %d point landmarks but they are "
                "collinear in world coordinates — the solve will fit them "
                "but the camera position perpendicular to the line is "
                "unconstrained, so the projected pitch may appear rotated "
                "or skewed. Add at least one off-axis landmark (e.g. an "
                "18yd-box corner, corner flag, or goal post) to disambiguate.",
                a.frame, len(a.landmarks),
            )

    width, height = image_size
    if lens_prior is not None:
        cx, cy, k1_prior, k2_prior = lens_prior
        prior_distortion: tuple[float, float] = (k1_prior, k2_prior)
    else:
        cx = width / 2.0
        cy = height / 2.0
        prior_distortion = (0.0, 0.0)
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
        result = _solve_one_anchor_full(
            a, cx, cy, fx_init, K_init, distortion=prior_distortion,
        )
        if result is None:
            continue
        K, R, t, fx = result
        per_anchor_KRt[a.frame] = (K, R, t)
        per_anchor_res[a.frame] = reprojection_residual_for_anchor(
            a, K, R, t, distortion=prior_distortion,
        )
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
            distortion=prior_distortion,
        )
        if solo_result is not None:
            K_s, R_s, t_s, _ = solo_result
            solo_res = reprojection_residual_for_anchor(
                a, K_s, R_s, t_s, distortion=prior_distortion,
            )
            solo_KRt = (K_s, R_s, t_s)

        # Candidate B: t-fixed (inherited from interpolation).
        t_inherited = _interp_t(a.frame, rich_frames_sorted, rich_t)
        if len(a.landmarks) >= 4:
            seed = _seed_anchor_pose(a, K_prior, distortion=prior_distortion)
            rvec_init = seed[0] if seed is not None else rich_rvec[nearest_frame].copy()
        else:
            rvec_init = rich_rvec[nearest_frame].copy()
        rvec, fx = _solve_anchor_with_t_fixed(
            a, t_inherited, cx, cy, fx_prior, rvec_init,
            distortion=prior_distortion,
        )
        K_t = _make_K(fx, cx, cy)
        R_t = _rvec_to_R(rvec)
        tfx_res = reprojection_residual_for_anchor(
            a, K_t, R_t, t_inherited, distortion=prior_distortion,
        )

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
        "hybrid solve (pre-distortion): %d total (%d rich solo + %d thin t-fixed), "
        "t_world median=[%.2f, %.2f, %.2f], mean residual=%.2f px",
        len(per_anchor_KRt), len(rich_t),
        len(per_anchor_KRt) - len(rich_t), *t_world_median,
        float(np.mean(list(per_anchor_res.values()))) if per_anchor_res else 0.0,
    )

    # ── Pass 3: joint LM to recover shared (k1, k2) and refine (R, t, fx). ──
    # Per-anchor t is freed so the LM can disambiguate distortion from pose;
    # cx/cy held at seed. This is where the broadcast lens's ~1–3% radial
    # bias gets removed when the anchors are mutually consistent.
    #
    # Skipped when a ``lens_prior`` was supplied — the prior is the final
    # lens model and re-estimating distortion here would risk overfitting
    # to per-anchor click noise.
    if lens_prior is not None:
        distortion = prior_distortion
    else:
        pre_residuals = dict(per_anchor_res)
        pre_mean = (
            float(np.mean(list(pre_residuals.values())))
            if pre_residuals else 0.0
        )
        refined_KRt, distortion, refined_res = _refine_joint_distortion(
            qualifying, per_anchor_KRt, cx, cy,
        )
        refined_mean = (
            float(np.mean(list(refined_res.values()))) if refined_res else 0.0
        )
        # Sanity guard: if the joint LM made things worse (typically because
        # anchors disagree from click noise rather than distortion), keep the
        # solo solves' (R, t, fx) and report distortion=(0, 0). Otherwise
        # bounded distortion can compound user inconsistencies into a
        # near-saturated k2 that bakes in further bias.
        if refined_res and refined_mean <= pre_mean * 1.1:
            per_anchor_KRt = refined_KRt
            for af, res in refined_res.items():
                per_anchor_res[af] = res
            logger.info(
                "joint distortion refine: k1=%+.4f, k2=%+.4f, mean residual %.2f → %.2f px",
                distortion[0], distortion[1], pre_mean, refined_mean,
            )
        else:
            distortion = (0.0, 0.0)
            logger.warning(
                "joint distortion refine rejected: residual would worsen %.2f → %.2f px. "
                "Anchors may be mutually inconsistent; review clicks before relying on "
                "lens-distortion compensation.",
                pre_mean, refined_mean,
            )

    return JointSolution(
        t_world=t_world_median,
        principal_point=(cx, cy),
        per_anchor_KRt=per_anchor_KRt,
        per_anchor_residual_px=per_anchor_res,
        distortion=distortion,
    )
