"""Homography-chain gap filling for PnLCalib calibration outputs.

PnLCalib fails on frames where the major pitch features (centre
circle, penalty box) aren't visible — typically zoomed-in corner
shots, throw-ins, or tight goal-area views.  Lowering the detector
thresholds doesn't help: it replaces "no output" with wrong-but-
confident output.

Within a single shot the broadcast camera is physically static;
it only pans, tilts, and zooms.  That means the image of the pitch
plane transforms by a **2D homography** between consecutive frames.
We can estimate that homography from frame-to-frame feature
tracking (ORB + Lucas–Kanade) *without* any pitch features, then
chain it from a known-good PnLCalib frame into the gap to
synthesise calibrations for the missing frames.

The result: every frame in a gap gets a pitch-to-image homography
equivalent to the anchor's calibration transformed by the
accumulated inter-frame camera motion.  We then decompose that
homography back into ``(K, rvec, tvec)`` so it fits into the
existing :class:`CameraFrame` schema and flows through the rest of
the pipeline unchanged.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from src.schemas.calibration import CalibrationResult, CameraFrame
from src.utils.pitch_line_detector import _largest_pitch_region, _pitch_mask

logger = logging.getLogger(__name__)


# ── Tunables ────────────────────────────────────────────────────────────────
_MIN_FEATURES = 40
_MAX_FEATURES = 400
_LK_WIN = (21, 21)
_LK_LEVELS = 3
_MIN_INLIERS_FOR_HOMOGRAPHY = 30
# Tighter RANSAC: LK tracking on pitch features is typically sub-pixel;
# anything >1.5 px off is probably tracking a moving player boundary.
_RANSAC_REPROJ_THRESHOLD_PX = 1.5
# Per-frame homography must be close to an identity-plus-pan — the
# camera can't zoom by >5 % in one frame, so |det(M) - 1| should be
# tiny.  Reject anything outside a safe band.
_DET_MIN = 0.85
_DET_MAX = 1.18
# Round-trip LK check: after forward tracking a feature, track it
# back and require the round-trip error to be <1 px.  Catches LK
# drift onto moving content.
_LK_ROUND_TRIP_MAX_PX = 1.0


@dataclass(frozen=True)
class PropagationStats:
    """Diagnostics for one shot's gap-filling pass."""

    n_good_frames_before: int
    n_good_frames_after: int
    n_gaps: int
    n_filled: int
    n_left_missing: int


def propagate_calibration_across_gaps(
    cal: CalibrationResult,
    clip_path: Path,
) -> tuple[CalibrationResult, PropagationStats]:
    """Fill empty frames in ``cal`` using homography chains from
    bracketing good frames.

    Args:
        cal: PnLCalib output for a single shot.  Frames that
            PnLCalib produced are the "good" anchors; frames
            missing from ``cal.frames`` are the gaps to fill.
        clip_path: Source clip on disk.

    Returns:
        ``(augmented_cal, stats)`` — a new :class:`CalibrationResult`
        with the synthesised gap frames merged into the existing
        frames (sorted by frame index), plus per-shot diagnostics.
    """
    if not cal.frames:
        return cal, PropagationStats(0, 0, 0, 0, 0)
    if not clip_path.exists():
        logger.warning("propagate: clip not found %s", clip_path)
        return cal, PropagationStats(
            len(cal.frames), len(cal.frames), 0, 0, 0,
        )

    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        logger.warning("propagate: cannot open clip %s", clip_path)
        return cal, PropagationStats(
            len(cal.frames), len(cal.frames), 0, 0, 0,
        )
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Build a frame_idx → CameraFrame lookup.
    good_by_idx: dict[int, CameraFrame] = {
        cf.frame: cf for cf in cal.frames
    }
    good_indices = sorted(good_by_idx.keys())
    if not good_indices:
        cap.release()
        return cal, PropagationStats(0, 0, 0, 0, 0)

    # Find gaps: the frame range is [0, n_total), any frame not in
    # good_by_idx within that range is a gap.  We only fill gaps
    # *between* good frames (or extending out from a good frame
    # toward the shot edges).
    gaps = _find_gaps(good_indices, n_total)
    if not gaps:
        cap.release()
        return cal, PropagationStats(
            len(good_indices), len(good_indices), 0, 0, 0,
        )

    # Read every frame into memory for one forward pass.  On a
    # 500-frame 1080p clip that's ~1.5 GB of RAM which is
    # acceptable on any reasonable dev box, and avoids repeated
    # seeks which dominate runtime on VFR clips.
    #
    # The alternative would be to walk the clip linearly once per
    # gap, but VideoCapture seeks on mp4 are cheap enough for this
    # not to matter at the frame counts we're dealing with.
    synthesised: dict[int, CameraFrame] = {}

    try:
        for gap_start, gap_end in gaps:
            filled = _fill_gap(
                cap=cap,
                gap_start=gap_start,
                gap_end=gap_end,
                good_by_idx=good_by_idx,
            )
            synthesised.update(filled)
    finally:
        cap.release()

    # Merge: keep original good frames, add synthesised ones,
    # sort by frame index.
    merged_list: list[CameraFrame] = list(cal.frames)
    for fi, cf in synthesised.items():
        merged_list.append(cf)
    merged_list.sort(key=lambda cf: cf.frame)

    new_cal = CalibrationResult(
        shot_id=cal.shot_id,
        camera_type=cal.camera_type,
        frames=merged_list,
    )
    total_gap_frames = sum(e - s + 1 for s, e in gaps)
    stats = PropagationStats(
        n_good_frames_before=len(good_indices),
        n_good_frames_after=len(merged_list),
        n_gaps=len(gaps),
        n_filled=len(synthesised),
        n_left_missing=total_gap_frames - len(synthesised),
    )
    return new_cal, stats


def _find_gaps(
    good_indices: list[int], n_total: int,
) -> list[tuple[int, int]]:
    """Return contiguous (start, end) inclusive ranges of missing frames.

    Gaps outside the bracket of good frames (before the first or
    after the last) are included, so that a shot whose first good
    frame is at 150 will have a gap (0, 149) filled by
    backward-chaining from frame 150.
    """
    gaps: list[tuple[int, int]] = []
    if not good_indices:
        return [(0, n_total - 1)]

    # Leading gap
    if good_indices[0] > 0:
        gaps.append((0, good_indices[0] - 1))
    # Interior gaps
    for a, b in zip(good_indices[:-1], good_indices[1:]):
        if b - a > 1:
            gaps.append((a + 1, b - 1))
    # Trailing gap
    if good_indices[-1] < n_total - 1:
        gaps.append((good_indices[-1] + 1, n_total - 1))
    return gaps


def _camera_frame_to_homography(cf: CameraFrame) -> np.ndarray:
    """Return the 3×3 pitch-plane → image homography for ``cf``.

    ``H = K [r1 | r2 | t]`` where ``r1``, ``r2`` are the first two
    columns of the rotation matrix.  Any point ``(x, y)`` on the
    pitch plane ``z = 0`` projects to ``H @ [x, y, 1]^T`` (up to
    scale).
    """
    K = np.asarray(cf.intrinsic_matrix, dtype=np.float64)
    rvec = np.asarray(cf.rotation_vector, dtype=np.float64).reshape(3)
    tvec = np.asarray(cf.translation_vector, dtype=np.float64).reshape(3)
    R, _ = cv2.Rodrigues(rvec)
    H = K @ np.column_stack([R[:, 0], R[:, 1], tvec])
    return H


def _homography_to_camera_frame(
    H: np.ndarray,
    anchor_cf: CameraFrame,
    frame_idx: int,
) -> CameraFrame:
    """Decompose a pitch-plane → image homography into a CameraFrame.

    We reuse ``anchor_cf``'s ``K`` (constant-zoom assumption within
    a gap — broadcast cameras almost never zoom while panning, and
    when they do the error over a ~30-frame gap is small).

    The decomposition: ``H = K [r1 | r2 | t]`` so
    ``K^{-1} H = [r1 | r2 | t]``.  ``r1``, ``r2`` must be unit
    vectors and mutually orthogonal in the ideal case; we SVD them
    to project onto the nearest orthonormal pair and recover the
    full rotation ``R = [r1 | r2 | r3]`` with ``r3 = r1 × r2``.
    """
    K = np.asarray(anchor_cf.intrinsic_matrix, dtype=np.float64)
    K_inv = np.linalg.inv(K)
    B = K_inv @ H  # 3×3: [r1' | r2' | t']

    r1p = B[:, 0]
    r2p = B[:, 1]
    tp = B[:, 2]

    # Average scale: r1 and r2 should have unit norm.  The
    # homography is only defined up to scale, so normalise by the
    # geometric mean to keep the scale consistent with translation.
    s1 = np.linalg.norm(r1p)
    s2 = np.linalg.norm(r2p)
    if s1 < 1e-9 or s2 < 1e-9:
        # Degenerate — fall back to the anchor calibration
        return CameraFrame(
            frame=frame_idx,
            intrinsic_matrix=anchor_cf.intrinsic_matrix,
            rotation_vector=list(anchor_cf.rotation_vector),
            translation_vector=list(anchor_cf.translation_vector),
            reprojection_error=anchor_cf.reprojection_error,
            num_correspondences=anchor_cf.num_correspondences,
            confidence=anchor_cf.confidence,
            tracked_landmark_types=list(anchor_cf.tracked_landmark_types),
        )
    scale = float(np.sqrt(s1 * s2))
    r1 = r1p / scale
    r2 = r2p / scale
    tvec = tp / scale

    # Project (r1, r2) onto the nearest orthonormal pair via SVD of
    # the 3×2 matrix [r1 | r2].  This also gives us an orthogonal
    # r3 via the cross product after normalisation.
    M = np.column_stack([r1, r2])
    U, _, Vt = np.linalg.svd(M, full_matrices=False)
    R_partial = U @ Vt  # 3×2 nearest orthonormal
    r1_ortho = R_partial[:, 0]
    r2_ortho = R_partial[:, 1]
    r3 = np.cross(r1_ortho, r2_ortho)
    R = np.column_stack([r1_ortho, r2_ortho, r3])
    # Ensure right-handedness
    if np.linalg.det(R) < 0:
        R[:, 2] = -R[:, 2]
    rvec, _ = cv2.Rodrigues(R)

    return CameraFrame(
        frame=frame_idx,
        intrinsic_matrix=K.tolist(),
        rotation_vector=rvec.reshape(3).tolist(),
        translation_vector=tvec.reshape(3).tolist(),
        reprojection_error=0.0,
        num_correspondences=0,
        confidence=0.8,  # slightly less trusted than a PnLCalib anchor
        tracked_landmark_types=["propagated"],
    )


def _estimate_frame_homography(
    frame_a: np.ndarray,
    frame_b: np.ndarray,
) -> tuple[np.ndarray, int] | None:
    """Estimate the 2D homography ``M`` such that ``p_b ~= M @ p_a``
    for pixels on the (nearly planar) pitch surface.

    Uses corner features on the pitch mask region of ``frame_a``
    tracked forward via Lucas–Kanade to ``frame_b``, then tracked
    back to ``frame_a`` as a round-trip consistency check.  Features
    whose tracked point leaves the pitch mask in ``frame_b`` or
    whose round-trip error exceeds a sub-pixel threshold are
    rejected (they're likely following moving content such as
    player silhouettes).  A RANSAC homography is then fit on the
    surviving pairs.

    Returns ``(M, n_inliers)`` or ``None`` if the estimation is
    unreliable (too few features, too few inliers, or degenerate
    homography).
    """
    mask_a = _largest_pitch_region(_pitch_mask(frame_a))
    mask_b = _largest_pitch_region(_pitch_mask(frame_b))
    gray_a = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY)

    corners_a = cv2.goodFeaturesToTrack(
        gray_a,
        maxCorners=_MAX_FEATURES,
        qualityLevel=0.01,
        minDistance=12,
        mask=mask_a,
    )
    if corners_a is None or len(corners_a) < _MIN_FEATURES:
        return None

    lk_params = dict(winSize=_LK_WIN, maxLevel=_LK_LEVELS)

    tracked_b, status_fwd, _ = cv2.calcOpticalFlowPyrLK(
        gray_a, gray_b, corners_a, None, **lk_params,
    )
    status_fwd = status_fwd.reshape(-1).astype(bool)
    if int(status_fwd.sum()) < _MIN_FEATURES:
        return None

    # Round-trip check: track back and measure displacement
    tracked_back, status_bwd, _ = cv2.calcOpticalFlowPyrLK(
        gray_b, gray_a, tracked_b, None, **lk_params,
    )
    status_bwd = status_bwd.reshape(-1).astype(bool)
    round_trip_err = np.linalg.norm(
        corners_a.reshape(-1, 2) - tracked_back.reshape(-1, 2),
        axis=1,
    )
    round_trip_ok = round_trip_err < _LK_ROUND_TRIP_MAX_PX

    # Check tracked points are still inside the pitch mask in frame B
    tracked_b_xy = tracked_b.reshape(-1, 2)
    h_b, w_b = mask_b.shape
    tb_int_x = np.clip(np.round(tracked_b_xy[:, 0]).astype(int), 0, w_b - 1)
    tb_int_y = np.clip(np.round(tracked_b_xy[:, 1]).astype(int), 0, h_b - 1)
    mask_b_ok = mask_b[tb_int_y, tb_int_x] > 0

    valid = status_fwd & status_bwd & round_trip_ok & mask_b_ok
    if int(valid.sum()) < _MIN_FEATURES:
        return None

    pts_a = corners_a[valid].reshape(-1, 2)
    pts_b = tracked_b_xy[valid]

    M, inlier_mask = cv2.findHomography(
        pts_a, pts_b,
        method=cv2.RANSAC,
        ransacReprojThreshold=_RANSAC_REPROJ_THRESHOLD_PX,
        maxIters=2000,
        confidence=0.995,
    )
    if M is None:
        return None
    n_inliers = int(inlier_mask.sum()) if inlier_mask is not None else 0
    if n_inliers < _MIN_INLIERS_FOR_HOMOGRAPHY:
        return None
    # Reject homographies whose determinant is out of the safe
    # per-frame band (camera can't zoom > ~18 % in one frame) or
    # numerically degenerate.
    det = float(np.linalg.det(M))
    if not (_DET_MIN < abs(det) < _DET_MAX):
        return None
    return M, n_inliers


def _fill_gap(
    cap: cv2.VideoCapture,
    gap_start: int,
    gap_end: int,
    good_by_idx: dict[int, CameraFrame],
) -> dict[int, CameraFrame]:
    """Fill a single contiguous gap with forward + backward homography chains.

    Returns ``{frame_idx: CameraFrame}`` for every frame in
    ``[gap_start, gap_end]`` that was successfully filled.  Frames
    where neither chain reaches stay missing.
    """
    anchor_before: CameraFrame | None = None
    anchor_after: CameraFrame | None = None
    # Scan back / forward from the gap for bracketing anchors
    for i in range(gap_start - 1, -1, -1):
        if i in good_by_idx:
            anchor_before = good_by_idx[i]
            break
    for i in range(gap_end + 1, gap_end + 1 + 10_000):
        if i in good_by_idx:
            anchor_after = good_by_idx[i]
            break

    if anchor_before is None and anchor_after is None:
        return {}

    forward: dict[int, np.ndarray] = {}  # frame_idx → pitch-to-image H
    backward: dict[int, np.ndarray] = {}

    # Forward chain from the anchor before the gap.
    if anchor_before is not None:
        H_prev = _camera_frame_to_homography(anchor_before)
        prev_frame_bgr = _read_frame(cap, anchor_before.frame)
        if prev_frame_bgr is not None:
            for fi in range(anchor_before.frame + 1, gap_end + 1):
                cur_frame_bgr = _read_frame(cap, fi)
                if cur_frame_bgr is None:
                    break
                est = _estimate_frame_homography(prev_frame_bgr, cur_frame_bgr)
                if est is None:
                    break
                M, _ = est
                H_cur = M @ H_prev
                if fi >= gap_start:
                    forward[fi] = H_cur
                H_prev = H_cur
                prev_frame_bgr = cur_frame_bgr

    # Backward chain from the anchor after the gap.
    if anchor_after is not None:
        H_prev = _camera_frame_to_homography(anchor_after)
        prev_frame_bgr = _read_frame(cap, anchor_after.frame)
        if prev_frame_bgr is not None:
            for fi in range(anchor_after.frame - 1, gap_start - 1, -1):
                cur_frame_bgr = _read_frame(cap, fi)
                if cur_frame_bgr is None:
                    break
                est = _estimate_frame_homography(prev_frame_bgr, cur_frame_bgr)
                if est is None:
                    break
                M, _ = est
                H_cur = M @ H_prev
                if fi <= gap_end:
                    backward[fi] = H_cur
                H_prev = H_cur
                prev_frame_bgr = cur_frame_bgr

    # For each frame pick the chain whose anchor is CLOSER — no
    # matrix averaging.  Linear interpolation of two 3×3 homographies
    # doesn't live on the homography manifold; blended matrices
    # correspond to invalid camera transforms and the projected
    # pitch lines sweep through degenerate rotations as the blend
    # weight shifts.
    out: dict[int, CameraFrame] = {}
    anchor_before_frame = anchor_before.frame if anchor_before else None
    anchor_after_frame = anchor_after.frame if anchor_after else None
    for fi in range(gap_start, gap_end + 1):
        H_f = forward.get(fi)
        H_b = backward.get(fi)
        dist_fwd = (fi - anchor_before_frame) if anchor_before_frame is not None else float("inf")
        dist_bck = (anchor_after_frame - fi) if anchor_after_frame is not None else float("inf")

        chosen_H: np.ndarray | None = None
        anchor_for_K: CameraFrame | None = None
        if H_f is not None and H_b is not None:
            if dist_fwd <= dist_bck:
                chosen_H = H_f
                anchor_for_K = anchor_before
            else:
                chosen_H = H_b
                anchor_for_K = anchor_after
        elif H_f is not None:
            chosen_H = H_f
            anchor_for_K = anchor_before
        elif H_b is not None:
            chosen_H = H_b
            anchor_for_K = anchor_after

        if chosen_H is not None and anchor_for_K is not None:
            out[fi] = _homography_to_camera_frame(chosen_H, anchor_for_K, fi)
        # else: no estimate reaches this frame; leave missing
    return out


def _read_frame(
    cap: cv2.VideoCapture, frame_idx: int,
) -> np.ndarray | None:
    """Seek to ``frame_idx`` and decode one frame."""
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    if not ok:
        return None
    return frame
