"""Iterative Closest Line calibration refinement.

Takes a per-frame initial calibration (typically the output of
PnLCalib + the single-VP rotation refinement in
:mod:`src.utils.calibration_refine`) and tightens it using detected
pitch + ad-board line segments as 3D-line constraints in a
Levenberg-Marquardt solve.

Three line categories drive the residual:

1. **Painted pitch markings** with fully known geometry — touchlines,
   goal lines, halfway, 18-yard fronts, 6-yard fronts.  Sourced from
   :func:`src.utils.pitch_lines.pitch_line_families`.

2. **Near ad board** top edges at known ``z = 0.9 m`` parallel to the
   touchline direction, but at an unknown ``y`` (stadium-specific).
   The ``y`` is a free parameter optimised jointly with the camera.

3. **Far ad board** top edges at the same height, also free ``y``.

The solver alternates between:

- **Assignment**: project every line family using the current
  estimate, then assign each detected segment to whichever family it
  sits closest to in image space (perpendicular pixel distance from
  the segment midpoint to the projected polyline).  Segments outside
  a max distance are dropped from this iteration.

- **Refinement**: an LM solve over
  ``(rvec[3], fx, y_board_near, y_board_far)`` minimising the sum of
  squared perpendicular distances between every assigned segment's
  midpoint and its projected canonical line.  The camera world
  position is held fixed (it comes from the static-camera fuser),
  which keeps the optimisation well-conditioned.

ICL terminates when assignments stop changing or after a small number
of iterations.  The refined frame is accepted only if the total
residual *strictly* decreases and the recovered focal length stays
within a sane band around PnLCalib's initial estimate.  Any acceptance
failure returns the input unchanged, matching the silent-rollback
pattern used elsewhere in the calibration refinement code.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np
from scipy.optimize import least_squares

from src.schemas.calibration import CameraFrame
from src.utils.camera import camera_world_position
from src.utils.pitch_line_detector import (
    DetectedLine,
    detect_board_lines,
    detect_pitch_lines,
)
from src.utils.pitch_lines import PitchLineFamily, pitch_line_families

logger = logging.getLogger(__name__)


# Distance threshold (pixels) for assigning a detected segment to a
# projected line family.  Anything farther than this is considered
# unrelated noise and dropped from the iteration.
_ASSIGN_MAX_DIST_PX = 30.0

# Acceptance band on the recovered focal length (relative to the input).
# Outside this band the refinement is rejected — focal length jumps of
# more than 2× usually mean the LM landed in a bad basin.
_FX_MIN_FACTOR = 0.5
_FX_MAX_FACTOR = 2.0

# Default search range for the ad-board ``y`` offset (metres beyond the
# touchline).  The board must be behind the touchline and within a
# stadium-realistic distance.  ``y_board_near = -offset`` (so positive
# offsets push the board behind the near touchline at y=0).
_BOARD_Y_OFFSET_INIT = 3.0
_BOARD_Y_OFFSET_MIN = 0.5
_BOARD_Y_OFFSET_MAX = 12.0

# Ad board top-edge height (metres above the pitch).  Stadium-specific
# in reality but ~0.9 m is a safe broadcast default.
_BOARD_TOP_Z = 0.9

# Maximum ICL iterations.  Convergence is usually within 2–3.
_MAX_ICL_ITERS = 5

# Minimum assigned segments to attempt the LM solve.  Below this we
# don't have enough constraints to refine 6 parameters.
_MIN_ASSIGNED_SEGMENTS = 8


@dataclass(frozen=True)
class ICLResult:
    """Diagnostics for one ICL refinement attempt on a single keyframe."""

    iterations: int
    n_assigned: int
    n_pitch_segments: int
    n_board_segments: int
    initial_residual_px: float
    refined_residual_px: float
    focal_length_change_factor: float
    accepted: bool


def refine_with_lines(
    cf: CameraFrame,
    frame_bgr: np.ndarray,
    *,
    max_iters: int = _MAX_ICL_ITERS,
    pitch_segments: list[DetectedLine] | None = None,
    board_segments: list[DetectedLine] | None = None,
    max_rotation_delta_deg: float | None = None,
) -> tuple[CameraFrame, ICLResult]:
    """Refine ``cf`` against detected pitch + board lines via ICL + LM.

    Args:
        cf: Initial calibration frame (typically post single-VP refinement).
        frame_bgr: Source BGR image — used for line detection if
            ``pitch_segments`` / ``board_segments`` aren't supplied.
        max_iters: Maximum ICL outer iterations.
        pitch_segments / board_segments: Optional pre-computed
            detection results.  Useful for tests; production callers
            usually let the function detect.

    Returns ``(refined_cf, diagnostics)``.  If refinement makes things
    worse or violates the focal-length sanity band, ``cf`` is returned
    unchanged with ``accepted=False`` in the diagnostics.
    """
    if pitch_segments is None or board_segments is None:
        det_pitch, mask = detect_pitch_lines(frame_bgr)
        det_board = detect_board_lines(frame_bgr, mask)
        pitch_segments = pitch_segments if pitch_segments is not None else det_pitch
        board_segments = board_segments if board_segments is not None else det_board

    K = np.asarray(cf.intrinsic_matrix, dtype=np.float64)
    rvec = np.asarray(cf.rotation_vector, dtype=np.float64).reshape(3)
    tvec = np.asarray(cf.translation_vector, dtype=np.float64).reshape(3)
    fx_init = float(K[0, 0])
    cx = float(K[0, 2])
    cy = float(K[1, 2])
    cam_pos = camera_world_position(rvec, tvec)

    canonical_families = pitch_line_families()
    fail = ICLResult(
        iterations=0,
        n_assigned=0,
        n_pitch_segments=len(pitch_segments),
        n_board_segments=len(board_segments),
        initial_residual_px=float("inf"),
        refined_residual_px=float("inf"),
        focal_length_change_factor=1.0,
        accepted=False,
    )
    if not pitch_segments and not board_segments:
        return cf, fail

    # Initial parameter vector and residual baseline.
    params = np.array(
        [rvec[0], rvec[1], rvec[2], fx_init, _BOARD_Y_OFFSET_INIT, _BOARD_Y_OFFSET_INIT],
        dtype=np.float64,
    )
    initial_residual = _total_residual(
        params, pitch_segments, board_segments, canonical_families,
        cx, cy, cam_pos,
    )
    if not np.isfinite(initial_residual):
        return cf, fail

    last_assignment: tuple[tuple[int, int], ...] | None = None
    icl_iters = 0
    best_params = params
    best_residual = initial_residual
    best_assigned = 0

    for it in range(max_iters):
        icl_iters = it + 1
        # Project current line families and assign segments.
        K_cur = np.array([[params[3], 0.0, cx], [0.0, params[3], cy], [0.0, 0.0, 1.0]],
                         dtype=np.float64)
        rvec_cur = params[:3]
        R_cur, _ = cv2.Rodrigues(rvec_cur)
        tvec_cur = -R_cur @ cam_pos

        canonical_polys = [
            _project_polyline(fam.polyline, K_cur, rvec_cur, tvec_cur)
            for fam in canonical_families
        ]
        board_near_poly = _project_board_polyline(
            -float(params[4]), K_cur, rvec_cur, tvec_cur,
        )
        board_far_poly = _project_board_polyline(
            68.0 + float(params[5]), K_cur, rvec_cur, tvec_cur,
        )

        # Assignment: each detected segment → (family_idx, family_kind)
        assignments = _assign(
            pitch_segments, board_segments,
            canonical_polys, board_near_poly, board_far_poly,
            max_dist_px=_ASSIGN_MAX_DIST_PX,
        )
        if len(assignments) < _MIN_ASSIGNED_SEGMENTS:
            return cf, ICLResult(
                iterations=icl_iters,
                n_assigned=len(assignments),
                n_pitch_segments=len(pitch_segments),
                n_board_segments=len(board_segments),
                initial_residual_px=initial_residual,
                refined_residual_px=initial_residual,
                focal_length_change_factor=1.0,
                accepted=False,
            )

        # Stable representation for change detection
        assign_key = tuple(sorted((sid, fid, kind) for sid, fid, kind in assignments))
        if last_assignment is not None and assign_key == last_assignment:
            break
        last_assignment = assign_key

        # LM refinement on the current assignment
        try:
            result = least_squares(
                _residuals_for_assignment,
                params,
                method="lm",
                max_nfev=200,
                args=(
                    assignments, pitch_segments, board_segments,
                    canonical_families, cx, cy, cam_pos,
                ),
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("ICL LM raised: %s", exc)
            break

        new_params = result.x.astype(np.float64)
        new_residual = _total_residual(
            new_params, pitch_segments, board_segments, canonical_families,
            cx, cy, cam_pos,
        )
        if not np.isfinite(new_residual):
            break
        if new_residual >= best_residual:
            break
        best_residual = new_residual
        best_params = new_params
        best_assigned = len(assignments)
        params = new_params

    fx_factor = float(best_params[3] / fx_init) if fx_init > 0 else 1.0
    # Rotation delta from the seed: how far did LM move us in
    # rotation space?  When the seed is already approximately right
    # (e.g., per-frame mode where the seed is the SLERP-interpolated
    # keyframe value) any large delta is a sign that LM landed in a
    # wrong basin — better to keep the seed unchanged.
    R_seed, _ = cv2.Rodrigues(rvec)
    R_refined, _ = cv2.Rodrigues(best_params[:3])
    R_delta = R_refined @ R_seed.T
    rotation_delta_deg = float(np.degrees(np.arccos(
        np.clip((np.trace(R_delta) - 1) / 2, -1.0, 1.0),
    )))
    accepted = (
        best_residual < initial_residual
        and _FX_MIN_FACTOR <= fx_factor <= _FX_MAX_FACTOR
        and _BOARD_Y_OFFSET_MIN <= float(best_params[4]) <= _BOARD_Y_OFFSET_MAX
        and _BOARD_Y_OFFSET_MIN <= float(best_params[5]) <= _BOARD_Y_OFFSET_MAX
        and (max_rotation_delta_deg is None or rotation_delta_deg <= max_rotation_delta_deg)
    )
    diagnostics = ICLResult(
        iterations=icl_iters,
        n_assigned=best_assigned,
        n_pitch_segments=len(pitch_segments),
        n_board_segments=len(board_segments),
        initial_residual_px=initial_residual,
        refined_residual_px=best_residual,
        focal_length_change_factor=fx_factor,
        accepted=accepted,
    )
    if not accepted:
        return cf, diagnostics

    new_K = np.array([[best_params[3], 0.0, cx],
                      [0.0, best_params[3], cy],
                      [0.0, 0.0, 1.0]], dtype=np.float64)
    new_rvec = best_params[:3]
    R_new, _ = cv2.Rodrigues(new_rvec)
    new_tvec = -R_new @ cam_pos
    refined_cf = CameraFrame(
        frame=cf.frame,
        intrinsic_matrix=new_K.tolist(),
        rotation_vector=new_rvec.tolist(),
        translation_vector=new_tvec.tolist(),
        reprojection_error=cf.reprojection_error,
        num_correspondences=cf.num_correspondences,
        confidence=cf.confidence,
        tracked_landmark_types=list(cf.tracked_landmark_types),
    )
    return refined_cf, diagnostics


# ── Internals ─────────────────────────────────────────────────────────────


def _project_polyline(
    world_pts: np.ndarray, K: np.ndarray, rvec: np.ndarray, tvec: np.ndarray,
) -> np.ndarray:
    """Project a (N, 3) world polyline through (K, rvec, tvec).

    Points behind the camera are removed before projection so the
    perspective divide doesn't produce nonsense.
    """
    R, _ = cv2.Rodrigues(rvec.reshape(3))
    cam = (R @ world_pts.T).T + tvec.reshape(3)
    front = cam[:, 2] > 0.05
    if not np.any(front):
        return np.empty((0, 2), dtype=np.float64)
    proj, _ = cv2.projectPoints(
        world_pts[front], rvec.reshape(3), tvec.reshape(3), K, None,
    )
    return proj.reshape(-1, 2).astype(np.float64)


def _project_board_polyline(
    y_world: float, K: np.ndarray, rvec: np.ndarray, tvec: np.ndarray,
) -> np.ndarray:
    """Project a synthetic ad-board top edge at (y, z) = (y_world, _BOARD_TOP_Z)."""
    poly = np.column_stack([
        np.linspace(0.0, 105.0, 64),
        np.full(64, y_world),
        np.full(64, _BOARD_TOP_Z),
    ])
    return _project_polyline(poly, K, rvec, tvec)


def _segment_to_polyline_distance(seg: DetectedLine, polyline: np.ndarray) -> float:
    """Perpendicular distance from a segment midpoint to a projected polyline.

    Vectorised: computes the distance from the midpoint to every
    polyline edge in one numpy operation rather than a Python loop.
    Returns ``inf`` when the polyline is empty (entirely behind camera).
    """
    if polyline.shape[0] < 2:
        return float("inf")
    mid = seg.midpoint()
    return _point_to_polyline_distance(mid, polyline)


def _point_to_polyline_distance(point: np.ndarray, polyline: np.ndarray) -> float:
    """Vectorised point-to-polyline distance.

    For an (N, 2) polyline of N-1 line segments, computes the
    perpendicular distance from ``point`` to each segment in a single
    numpy expression and returns the minimum.
    """
    if polyline.shape[0] < 2:
        return float("inf")
    a = polyline[:-1]                    # (N-1, 2)
    b = polyline[1:]                     # (N-1, 2)
    ab = b - a                           # (N-1, 2)
    ab_len_sq = np.einsum("ij,ij->i", ab, ab)  # (N-1,)
    # Avoid division-by-zero on degenerate edges (consecutive identical points)
    safe = ab_len_sq > 1e-12
    ap = point - a                       # (N-1, 2) via broadcasting
    t = np.zeros(len(a), dtype=np.float64)
    if np.any(safe):
        t[safe] = np.clip(
            np.einsum("ij,ij->i", ap[safe], ab[safe]) / ab_len_sq[safe],
            0.0, 1.0,
        )
    closest = a + ab * t[:, None]        # (N-1, 2)
    diffs = point - closest
    dists_sq = np.einsum("ij,ij->i", diffs, diffs)
    if np.any(safe):
        dists_sq = dists_sq[safe]
    return float(np.sqrt(np.min(dists_sq)))


def _assign(
    pitch_segments: list[DetectedLine],
    board_segments: list[DetectedLine],
    canonical_polys: list[np.ndarray],
    board_near_poly: np.ndarray,
    board_far_poly: np.ndarray,
    max_dist_px: float,
) -> list[tuple[int, int, str]]:
    """Assign each detected segment to its closest projected line family.

    ``kind`` is ``'canonical'``, ``'board_near'``, or ``'board_far'``.
    Pitch segments may be assigned to canonical families OR (in case
    of false positives in the pitch mask) to a board family.  Board
    segments only consider the two board families.

    Returns a list of ``(segment_idx_in_combined, family_idx, kind)``
    where ``segment_idx_in_combined`` indexes into
    ``pitch_segments + board_segments`` (concatenated).
    """
    assignments: list[tuple[int, int, str]] = []
    n_pitch = len(pitch_segments)
    for i, seg in enumerate(pitch_segments):
        best_dist = max_dist_px
        best: tuple[int, str] | None = None
        for fid, poly in enumerate(canonical_polys):
            d = _segment_to_polyline_distance(seg, poly)
            if d < best_dist:
                best_dist = d
                best = (fid, "canonical")
        if best is not None:
            assignments.append((i, best[0], best[1]))

    for j, seg in enumerate(board_segments):
        best_dist = max_dist_px
        best: tuple[int, str] | None = None
        d_near = _segment_to_polyline_distance(seg, board_near_poly)
        if d_near < best_dist:
            best_dist = d_near
            best = (0, "board_near")
        d_far = _segment_to_polyline_distance(seg, board_far_poly)
        if d_far < best_dist:
            best_dist = d_far
            best = (0, "board_far")
        if best is not None:
            assignments.append((n_pitch + j, best[0], best[1]))
    return assignments


def _residuals_for_assignment(
    params: np.ndarray,
    assignments: list[tuple[int, int, str]],
    pitch_segments: list[DetectedLine],
    board_segments: list[DetectedLine],
    canonical_families: list[PitchLineFamily],
    cx: float, cy: float, cam_pos: np.ndarray,
) -> np.ndarray:
    """LM residual vector for a fixed assignment.

    Each assignment contributes one scalar residual: the perpendicular
    distance from the segment midpoint to its projected canonical
    line.  LM minimises the sum of squares.
    """
    rvec = params[:3]
    fx = float(params[3])
    K = np.array([[fx, 0.0, cx], [0.0, fx, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    R, _ = cv2.Rodrigues(rvec)
    tvec = -R @ cam_pos

    canonical_polys = [
        _project_polyline(fam.polyline, K, rvec, tvec)
        for fam in canonical_families
    ]
    board_near_poly = _project_board_polyline(
        -float(params[4]), K, rvec, tvec,
    )
    board_far_poly = _project_board_polyline(
        68.0 + float(params[5]), K, rvec, tvec,
    )

    n_pitch = len(pitch_segments)
    out = np.empty(len(assignments), dtype=np.float64)
    for i, (seg_idx, fam_idx, kind) in enumerate(assignments):
        seg = pitch_segments[seg_idx] if seg_idx < n_pitch else board_segments[seg_idx - n_pitch]
        if kind == "canonical":
            poly = canonical_polys[fam_idx]
        elif kind == "board_near":
            poly = board_near_poly
        else:
            poly = board_far_poly
        d = _segment_to_polyline_distance(seg, poly)
        # Cap the residual so a single bad assignment can't blow up
        # the whole LM step.
        out[i] = float(min(d, 200.0))
    return out


def _total_residual(
    params: np.ndarray,
    pitch_segments: list[DetectedLine],
    board_segments: list[DetectedLine],
    canonical_families: list[PitchLineFamily],
    cx: float, cy: float, cam_pos: np.ndarray,
) -> float:
    """Re-assign and compute the mean residual for arbitrary parameters.

    Used for before/after comparisons that must be on equal footing
    (same assignment criterion, just different params).
    """
    rvec = params[:3]
    fx = float(params[3])
    K = np.array([[fx, 0.0, cx], [0.0, fx, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    R, _ = cv2.Rodrigues(rvec)
    tvec = -R @ cam_pos
    canonical_polys = [
        _project_polyline(fam.polyline, K, rvec, tvec)
        for fam in canonical_families
    ]
    board_near_poly = _project_board_polyline(-float(params[4]), K, rvec, tvec)
    board_far_poly = _project_board_polyline(68.0 + float(params[5]), K, rvec, tvec)
    assignments = _assign(
        pitch_segments, board_segments,
        canonical_polys, board_near_poly, board_far_poly,
        max_dist_px=_ASSIGN_MAX_DIST_PX,
    )
    if not assignments:
        return float("inf")
    res = _residuals_for_assignment(
        params, assignments, pitch_segments, board_segments,
        canonical_families, cx, cy, cam_pos,
    )
    return float(np.mean(res))
