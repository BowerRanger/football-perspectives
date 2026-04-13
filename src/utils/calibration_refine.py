"""Per-shot calibration refinement using detected pitch lines + boards.

Takes the initial PnLCalib calibration and refines each keyframe's
rotation using line-based cues PnLCalib doesn't see well:

- **Mowing stripes + touchlines + advertising-board edges** — long
  straight lines whose 3D direction is the touchline direction
  ``(1, 0, 0)``.  Their image-space vanishing point is a strong
  constraint on the camera rotation: the world vector ``(1, 0, 0)``
  must project exactly to that pixel.

The refinement keeps camera **position** fixed (it comes from the
static-camera fuser's median across keyframes — robust even when
individual rotations are wrong) and keeps **focal length** at the
PnLCalib estimate, since accurately recovering focal length needs a
second orthogonal vanishing point that we can't reliably extract from
typical broadcast frames where mowing stripes dominate the line
detector output.

Only the **rotation** is updated.  We compute the smallest rotation
that takes PnLCalib's current image of the touchline direction onto
the detected vanishing-point direction — this preserves PnLCalib's
estimate of the orthogonal rotation components (camera tilt and roll)
while correcting the dominant pan error.

A per-frame residual is computed both before and after refinement by
projecting all canonical pitch polylines and measuring the average
perpendicular distance from each detected line segment midpoint to
the nearest projected polyline.  When refinement *increases* the
residual, the original calibration is kept — failure is silent and
non-destructive.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from src.schemas.calibration import CalibrationResult, CameraFrame
from src.utils.camera import camera_world_position
from src.utils.iterative_line_refinement import (
    ICLResult,
    refine_with_lines,
)
from src.utils.pitch_line_detector import (
    DetectedLine,
    detect_board_lines,
    detect_pitch_lines,
)
from src.utils.pitch_lines import pitch_polylines
from src.utils.vp_calibration import (
    cluster_lines_by_orientation,
    vanishing_point_from_lines,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RefinementResult:
    """Diagnostics for a single-frame refinement attempt."""

    frame: int
    initial_residual_px: float
    refined_residual_px: float
    accepted: bool
    n_pitch_lines: int
    n_board_lines: int
    icl_iterations: int = 0
    icl_n_assigned: int = 0
    icl_initial_residual_px: float = float("inf")
    icl_refined_residual_px: float = float("inf")
    icl_focal_change_factor: float = 1.0
    icl_accepted: bool = False


# Standard board height above the pitch plane in metres (LED boards).
# Stadium-specific in reality, but ~0.9 m is a safe broadcast default
# and only affects the absolute scale of the board residual term.
_BOARD_HEIGHT_M = 0.9


def _project_polylines(
    K: np.ndarray, rvec: np.ndarray, tvec: np.ndarray,
) -> list[np.ndarray]:
    """Project all canonical pitch polylines and return their pixel forms.

    Polylines that fall entirely behind the camera are returned as
    empty arrays.
    """
    R, _ = cv2.Rodrigues(rvec.reshape(3))
    out: list[np.ndarray] = []
    for poly in pitch_polylines():
        cam = (R @ poly.T).T + tvec.reshape(3)
        front = cam[:, 2] > 0.05
        if not np.any(front):
            out.append(np.empty((0, 2), dtype=np.float64))
            continue
        front_world = poly[front]
        proj, _ = cv2.projectPoints(front_world, rvec.reshape(3), tvec.reshape(3), K, None)
        out.append(proj.reshape(-1, 2).astype(np.float64))
    return out


def _vp_consistency_residual(
    segments: list[DetectedLine],
    K: np.ndarray, rvec: np.ndarray, tvec: np.ndarray,
    world_direction: np.ndarray = np.array([1.0, 0.0, 0.0]),
) -> float:
    """Mean angular error (degrees) between each segment and the VP direction.

    For each detected segment we compute the angle between the
    segment's direction and the line from the segment midpoint to the
    touchline vanishing point implied by the candidate ``(K, R, t)``.
    A perfectly consistent segment has angle 0.

    Returns ``inf`` when there are no segments or the VP is degenerate.
    """
    if not segments:
        return float("inf")
    R, _ = cv2.Rodrigues(rvec.reshape(3))
    d_world = np.asarray(world_direction, dtype=np.float64)
    d_cam = R @ d_world
    if abs(d_cam[2]) < 1e-9:
        return float("inf")
    vp_h = K @ d_cam
    vp = vp_h[:2] / vp_h[2]
    if not np.all(np.isfinite(vp)):
        return float("inf")

    angles_deg: list[float] = []
    for seg in segments:
        seg_dir = seg.direction()
        mid = seg.midpoint()
        to_vp = vp - mid
        n = float(np.linalg.norm(to_vp))
        if n < 1e-3:
            continue
        to_vp /= n
        cos = float(np.clip(abs(seg_dir @ to_vp), 0.0, 1.0))
        angles_deg.append(float(np.degrees(np.arccos(cos))))
    if not angles_deg:
        return float("inf")
    return float(np.mean(angles_deg))


def _select_touchline_cluster(
    clusters: list[list[DetectedLine]],
    K: np.ndarray, rvec: np.ndarray, tvec: np.ndarray,
) -> list[DetectedLine] | None:
    """Pick the cluster of lines that best matches the touchline direction.

    Projects the world touchline direction (1,0,0) through the existing
    calibration to predict its image-space angle, then picks whichever
    detected cluster is closest to that angle.  Falls back to the
    longest cluster if the projection fails.
    """
    if not clusters:
        return None
    polylines = _project_polylines(K, rvec, tvec)
    near_touch = polylines[0]  # near touchline polyline

    def _cluster_angle(cluster: list[DetectedLine]) -> float:
        ws = np.array([ln.length for ln in cluster], dtype=np.float64)
        a = np.array([ln.angle for ln in cluster], dtype=np.float64)
        s = float(np.sum(ws * np.sin(2 * a)))
        c = float(np.sum(ws * np.cos(2 * a)))
        ang = 0.5 * np.arctan2(s, c)
        if ang < 0:
            ang += np.pi
        return float(ang)

    if near_touch.shape[0] >= 2:
        d = near_touch[-1] - near_touch[0]
        if np.all(np.isfinite(d)) and float(d @ d) > 1e-6:
            ref_angle = float(np.arctan2(d[1], d[0]))
            if ref_angle < 0:
                ref_angle += np.pi

            def _diff(a: float) -> float:
                dd = abs(a - ref_angle) % np.pi
                return float(min(dd, np.pi - dd))

            best = min(clusters, key=lambda c: _diff(_cluster_angle(c)))
            return best
    # Fallback: longest cluster
    return max(clusters, key=lambda c: sum(ln.length for ln in c))


def _rotation_aligning(
    a: np.ndarray, b: np.ndarray,
) -> np.ndarray:
    """Rotation matrix that takes unit vector ``a`` to unit vector ``b``.

    Uses Rodrigues' rotation formula.  Falls back to identity when the
    vectors are already aligned, or to a 180° rotation around an
    arbitrary perpendicular axis when they're antiparallel.
    """
    a = np.asarray(a, dtype=np.float64).reshape(3)
    b = np.asarray(b, dtype=np.float64).reshape(3)
    a /= np.linalg.norm(a)
    b /= np.linalg.norm(b)
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    if dot > 0.9999:
        return np.eye(3, dtype=np.float64)
    if dot < -0.9999:
        # Pick any axis perpendicular to a
        axis = np.cross(a, np.array([1.0, 0.0, 0.0]))
        if float(axis @ axis) < 1e-6:
            axis = np.cross(a, np.array([0.0, 1.0, 0.0]))
        axis /= np.linalg.norm(axis)
        K = _skew(axis)
        return np.eye(3) + 2 * (K @ K)
    axis = np.cross(a, b)
    s = float(np.linalg.norm(axis))
    axis /= s
    angle = float(np.arctan2(s, dot))
    K = _skew(axis)
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def _skew(v: np.ndarray) -> np.ndarray:
    return np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0],
    ], dtype=np.float64)


def refine_keyframe(
    frame_bgr: np.ndarray,
    cf: CameraFrame,
) -> tuple[CameraFrame, RefinementResult]:
    """Refine a single keyframe's calibration using detected pitch lines.

    Two-pass refinement:

    1. **Single-VP pan correction**: detect the dominant pitch-line
       cluster, compute its image vanishing point, and rotate the
       existing calibration so the world touchline direction projects
       exactly through that VP.  Corrects the dominant pan error while
       keeping PnLCalib's tilt + roll + focal length intact.

    2. **ICL refinement**: takes the result of step 1 (or the original
       calibration if step 1 failed) and runs Iterative Closest Line
       against the painted markings + ad boards to jointly refine
       rotation + focal length.  See
       :mod:`src.utils.iterative_line_refinement`.

    Returns ``(refined_camera_frame, diagnostics)``.  When neither
    step accepts a change, the original ``cf`` is returned unchanged.
    """
    pitch_lines, mask = detect_pitch_lines(frame_bgr)
    board_lines = detect_board_lines(frame_bgr, mask)
    n_pitch = len(pitch_lines)
    n_board = len(board_lines)

    # ── Step 1: single-VP pan correction ──
    vp_cf, vp_initial, vp_refined, vp_accepted = _refine_with_touchline_vp(
        cf, pitch_lines, board_lines,
    )

    # ── Step 2: ICL refinement ──
    icl_cf, icl_diag = refine_with_lines(
        vp_cf, frame_bgr,
        pitch_segments=pitch_lines,
        board_segments=board_lines,
    )

    # Pick the better of (vp_cf, icl_cf): ICL is only used if it
    # improved the line-distance residual *and* its diagnostics flagged
    # acceptance.  Otherwise fall back to the VP result (which is
    # itself either the refined or the original frame).
    final_cf = icl_cf if icl_diag.accepted else vp_cf
    final_accepted = vp_accepted or icl_diag.accepted
    return final_cf, RefinementResult(
        frame=cf.frame,
        initial_residual_px=vp_initial,
        refined_residual_px=vp_refined,
        accepted=final_accepted,
        n_pitch_lines=n_pitch,
        n_board_lines=n_board,
        icl_iterations=icl_diag.iterations,
        icl_n_assigned=icl_diag.n_assigned,
        icl_initial_residual_px=icl_diag.initial_residual_px,
        icl_refined_residual_px=icl_diag.refined_residual_px,
        icl_focal_change_factor=icl_diag.focal_length_change_factor,
        icl_accepted=icl_diag.accepted,
    )


def _refine_with_touchline_vp(
    cf: CameraFrame,
    pitch_lines: list[DetectedLine],
    board_lines: list[DetectedLine],
) -> tuple[CameraFrame, float, float, bool]:
    """Single-VP pan correction (the original refinement step).

    Returns ``(refined_or_original_cf, initial_residual, refined_residual,
    accepted)``.  Always returns a usable CameraFrame — the original
    when the VP refinement fails or makes things worse.
    """
    K = np.asarray(cf.intrinsic_matrix, dtype=np.float64)
    rvec = np.asarray(cf.rotation_vector, dtype=np.float64).reshape(3)
    tvec = np.asarray(cf.translation_vector, dtype=np.float64).reshape(3)

    if len(pitch_lines) < 6:
        return cf, float("inf"), float("inf"), False

    candidates: list[DetectedLine] = list(pitch_lines)
    candidates.extend(board_lines)

    clusters = cluster_lines_by_orientation(candidates, n_clusters=3, angle_tol_deg=6.0)
    if not clusters:
        return cf, float("inf"), float("inf"), False
    touch_cluster = _select_touchline_cluster(clusters, K, rvec, tvec)
    if touch_cluster is None or len(touch_cluster) < 4:
        return cf, float("inf"), float("inf"), False

    vp = vanishing_point_from_lines(touch_cluster)
    if not np.all(np.isfinite(vp)):
        return cf, float("inf"), float("inf"), False

    initial_residual = _vp_consistency_residual(touch_cluster, K, rvec, tvec)

    K_inv = np.linalg.inv(K)
    d_target_cam = K_inv @ np.array([vp[0], vp[1], 1.0], dtype=np.float64)
    d_target_cam /= np.linalg.norm(d_target_cam)

    R, _ = cv2.Rodrigues(rvec)
    d_current_cam = R @ np.array([1.0, 0.0, 0.0], dtype=np.float64)
    d_current_cam /= np.linalg.norm(d_current_cam)
    if float(d_current_cam @ d_target_cam) < float(d_current_cam @ -d_target_cam):
        d_target_cam = -d_target_cam

    R_align = _rotation_aligning(d_current_cam, d_target_cam)
    new_R_world_to_cam = R_align @ R
    angle_change = float(np.arccos(np.clip((np.trace(R_align) - 1) / 2, -1.0, 1.0)))
    if angle_change > np.deg2rad(60):
        return cf, initial_residual, initial_residual, False

    new_rvec, _ = cv2.Rodrigues(new_R_world_to_cam)
    camera_pos = camera_world_position(rvec, tvec)
    new_tvec = -new_R_world_to_cam @ camera_pos

    refined_residual = _vp_consistency_residual(
        touch_cluster, K, new_rvec.reshape(3), new_tvec,
    )
    if not np.isfinite(refined_residual) or refined_residual >= initial_residual:
        return cf, initial_residual, refined_residual, False

    new_cf = CameraFrame(
        frame=cf.frame,
        intrinsic_matrix=cf.intrinsic_matrix,
        rotation_vector=new_rvec.reshape(3).tolist(),
        translation_vector=new_tvec.tolist(),
        reprojection_error=cf.reprojection_error,
        num_correspondences=cf.num_correspondences,
        confidence=cf.confidence,
        tracked_landmark_types=list(cf.tracked_landmark_types),
    )
    return new_cf, initial_residual, refined_residual, True


def refine_shot_calibration(
    cal: CalibrationResult,
    clip_path: Path,
) -> tuple[CalibrationResult, list[RefinementResult]]:
    """Refine every keyframe in a shot's calibration.

    Loads the clip, samples each keyframe's frame, runs
    :func:`refine_keyframe`, and returns a new
    :class:`CalibrationResult` plus per-frame diagnostics.
    """
    if not cal.frames:
        return cal, []

    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        logger.warning("refine: cannot open %s", clip_path)
        return cal, []

    new_frames: list[CameraFrame] = []
    diagnostics: list[RefinementResult] = []
    try:
        for cf in cal.frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, cf.frame)
            ok, frame = cap.read()
            if not ok:
                new_frames.append(cf)
                continue
            new_cf, diag = refine_keyframe(frame, cf)
            new_frames.append(new_cf)
            diagnostics.append(diag)
    finally:
        cap.release()

    return CalibrationResult(
        shot_id=cal.shot_id,
        camera_type=cal.camera_type,
        frames=new_frames,
    ), diagnostics
