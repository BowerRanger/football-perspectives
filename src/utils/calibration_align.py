"""Cross-shot calibration alignment via player-foot Procrustes.

Each shot's calibration stage estimates a static camera position +
per-keyframe rotation independently, in its own world frame.  When two
shots cover the same play but their world frames differ by a small
rotation/translation, the same physical player back-projects to
**different** world positions in each shot — which appears in the
bird's-eye view as a sudden jump when triangulation switches between
shots.

This module aligns every non-reference shot's world frame to a chosen
reference shot using a Procrustes fit on matched player foot
positions:

1. **Pick a reference shot** — the one with the most calibration
   keyframes (most stable static-camera estimate).
2. **For each other shot**, find frames where matched players appear
   in both shots.  Back-project the player's foot pixel using each
   shot's calibration to get a foot world position in each frame.
3. **Fit a rigid transform** ``(R_diff, T_diff)`` mapping the other
   shot's foot positions onto the reference shot's foot positions.
4. **Bake the transform into the other shot's calibration**.  The
   camera world position becomes ``R_diff @ pos + T_diff``; the
   per-keyframe rotation becomes ``R_old @ R_diff^T``; the per-frame
   translation is recomputed from the new position to keep the
   pinhole equation consistent.

The transform is restricted to a yaw rotation (rotation around the
vertical axis) plus a 2D translation in the pitch plane, since the
pitch is a horizontal plane and any vertical or out-of-plane
component would represent a calibration impossibility.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

from src.schemas.calibration import CalibrationResult, CameraFrame
from src.schemas.player_matches import PlayerMatches
from src.schemas.sync_map import SyncMap
from src.schemas.tracks import TracksResult
from src.utils.camera import camera_world_position, project_to_pitch
from src.utils.triangulation_calib import CalibrationInterpolator

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ShotAlignment:
    """Diagnostics for one shot's alignment to the reference frame."""

    shot_id: str
    n_correspondences: int
    yaw_correction_deg: float
    translation_xy_m: tuple[float, float]
    residual_before_m: float
    residual_after_m: float
    accepted: bool


def _foot_centre(bbox: list[float]) -> tuple[float, float]:
    """Bottom-centre of bbox = foot pixel position estimate."""
    return ((bbox[0] + bbox[2]) / 2.0, float(bbox[3]))


def _build_track_frame_lookup(
    tracks: TracksResult,
) -> dict[str, dict[int, list[float]]]:
    """``{track_id: {frame_idx: bbox}}`` for fast cross-frame access."""
    out: dict[str, dict[int, list[float]]] = {}
    for t in tracks.tracks:
        if t.class_name == "ball":
            continue
        out[t.track_id] = {tf.frame: tf.bbox for tf in t.frames}
    return out


def _back_project_foot(
    foot_px: tuple[float, float],
    interp: CalibrationInterpolator,
    local_frame: int,
) -> np.ndarray | None:
    """Back-project a foot pixel through the per-frame calibration.

    Returns ``(x, y)`` in pitch-metres, or ``None`` if no calibration
    is available at ``local_frame`` or the back-projection fails
    (camera ray nearly parallel to the pitch plane).
    """
    cal = interp.at_nearest(local_frame, max_extrapolation_frames=200)
    if cal is None:
        return None
    try:
        xy = project_to_pitch(
            np.array(foot_px, dtype=np.float64),
            cal.K, cal.rvec, cal.tvec,
        )
    except (np.linalg.LinAlgError, cv2.error):
        return None
    if not np.all(np.isfinite(xy)):
        return None
    return np.array([float(xy[0]), float(xy[1])], dtype=np.float64)


def _yaw_procrustes(
    src: np.ndarray, dst: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Find ``(yaw, t)`` that minimises ``||dst - R(yaw) @ src - t||²``.

    Operates entirely in 2D (the pitch plane).  Closed-form: subtract
    centroids, find the optimal rotation via the cross-correlation
    angle, then recover the translation.  Returns ``(yaw_radians,
    t_xy)``.
    """
    src_c = src - src.mean(axis=0)
    dst_c = dst - dst.mean(axis=0)
    # Cross-correlation matrix
    H = src_c.T @ dst_c  # (2, 2)
    # Optimal yaw = angle of the rotation that aligns src_c onto dst_c
    yaw = float(np.arctan2(H[0, 1] - H[1, 0], H[0, 0] + H[1, 1]))
    R = np.array([[np.cos(yaw), -np.sin(yaw)],
                  [np.sin(yaw),  np.cos(yaw)]], dtype=np.float64)
    t = dst.mean(axis=0) - R @ src.mean(axis=0)
    return yaw, t


def _apply_yaw_translation_to_frame(
    cf: CameraFrame, yaw: float, t_xy: np.ndarray,
) -> CameraFrame:
    """Bake a yaw + planar translation into a single calibration frame.

    The world frame is rotated by ``yaw`` around the vertical axis and
    translated by ``(t_x, t_y, 0)``.  The camera position transforms
    accordingly; rotation and translation vectors are recomputed so
    the pinhole projection of any point ``p_old`` produces the same
    pixel as the projection of ``R(yaw) @ p_old + t``.
    """
    K = np.array(cf.intrinsic_matrix, dtype=np.float64)
    rvec_old = np.array(cf.rotation_vector, dtype=np.float64)
    tvec_old = np.array(cf.translation_vector, dtype=np.float64)
    R_old, _ = cv2.Rodrigues(rvec_old)

    R_diff_3d = np.array([[np.cos(yaw), -np.sin(yaw), 0.0],
                          [np.sin(yaw),  np.cos(yaw), 0.0],
                          [0.0, 0.0, 1.0]], dtype=np.float64)
    T_diff_3d = np.array([t_xy[0], t_xy[1], 0.0], dtype=np.float64)

    # Original camera position
    C_old = camera_world_position(rvec_old, tvec_old)
    # New camera position after the world frame transform
    C_new = R_diff_3d @ C_old + T_diff_3d

    # New rotation: world_new → world_old: pos_old = R_diff^T @ (pos_new - T)
    # cam = R_old @ pos_old + tvec_old = R_old @ R_diff^T @ (pos_new - T) + tvec_old
    # So new R_world_to_cam = R_old @ R_diff^T
    R_new = R_old @ R_diff_3d.T
    # And new t = -R_new @ C_new (so the pinhole equation holds with C_new)
    t_new = -R_new @ C_new

    rvec_new, _ = cv2.Rodrigues(R_new)
    return CameraFrame(
        frame=cf.frame,
        intrinsic_matrix=cf.intrinsic_matrix,
        rotation_vector=rvec_new.reshape(3).tolist(),
        translation_vector=t_new.tolist(),
        reprojection_error=cf.reprojection_error,
        num_correspondences=cf.num_correspondences,
        confidence=cf.confidence,
        tracked_landmark_types=list(cf.tracked_landmark_types),
    )


def _gather_correspondences(
    ref_shot_id: str,
    other_shot_id: str,
    matches: PlayerMatches,
    track_lookups: dict[str, dict[str, dict[int, list[float]]]],
    interps: dict[str, CalibrationInterpolator],
    sync_offsets: dict[str, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Collect ``(other_world_xy, ref_world_xy)`` pairs from matched players.

    For every matched player that has a track in *both* the reference
    and the other shot, walk every frame the two tracks share (in
    reference time) and back-project both feet to world coordinates.
    Returns two ``(N, 2)`` arrays.  Pairs whose back-projection fails
    on either side are skipped.
    """
    ref_offset = sync_offsets.get(ref_shot_id, 0)
    other_offset = sync_offsets.get(other_shot_id, 0)

    src_pts: list[np.ndarray] = []
    dst_pts: list[np.ndarray] = []
    for player in matches.matched_players:
        ref_track = next(
            (v.track_id for v in player.views if v.shot_id == ref_shot_id), None,
        )
        other_track = next(
            (v.track_id for v in player.views if v.shot_id == other_shot_id), None,
        )
        if ref_track is None or other_track is None:
            continue
        ref_frames = track_lookups[ref_shot_id].get(ref_track, {})
        other_frames = track_lookups[other_shot_id].get(other_track, {})
        if not ref_frames or not other_frames:
            continue

        # Walk frames in reference time
        for ref_local, ref_bbox in ref_frames.items():
            ref_global = ref_local + ref_offset
            other_local = ref_global - other_offset
            if other_local not in other_frames:
                continue
            ref_xy = _back_project_foot(
                _foot_centre(ref_bbox), interps[ref_shot_id], ref_local,
            )
            other_xy = _back_project_foot(
                _foot_centre(other_frames[other_local]),
                interps[other_shot_id],
                other_local,
            )
            if ref_xy is None or other_xy is None:
                continue
            # Reject foot positions that fall well outside the pitch — they
            # come from numerically-bad back-projections (camera ray
            # nearly parallel to the plane).
            for xy in (ref_xy, other_xy):
                if not (-15.0 < xy[0] < 120.0 and -15.0 < xy[1] < 83.0):
                    break
            else:
                src_pts.append(other_xy)
                dst_pts.append(ref_xy)
    return (
        np.asarray(src_pts, dtype=np.float64) if src_pts else np.empty((0, 2)),
        np.asarray(dst_pts, dtype=np.float64) if dst_pts else np.empty((0, 2)),
    )


def align_shots(
    calibrations: dict[str, CalibrationResult],
    tracks_by_shot: dict[str, TracksResult],
    matches: PlayerMatches,
    sync_map: SyncMap,
    *,
    min_correspondences: int = 20,
    max_yaw_deg: float = 25.0,
    max_translation_m: float = 30.0,
) -> tuple[dict[str, CalibrationResult], list[ShotAlignment]]:
    """Align every non-reference shot's world frame to the reference shot.

    The reference shot is whichever calibrated shot has the most
    keyframes.  For each other shot we fit a yaw + planar translation
    using matched player foot positions.  Alignments outside the
    sanity bounds (``max_yaw_deg``, ``max_translation_m``) are
    rejected and the original calibration is kept unchanged.

    Returns the (possibly modified) calibrations dict plus a list of
    :class:`ShotAlignment` diagnostics, one per non-reference shot.
    """
    sync_offsets: dict[str, int] = {sync_map.reference_shot: 0}
    for a in sync_map.alignments:
        sync_offsets[a.shot_id] = a.frame_offset

    # Build per-shot calibration interpolators (also serves as a way
    # to check which shots actually have usable calibration data).
    interps: dict[str, CalibrationInterpolator] = {}
    for sid, cal in calibrations.items():
        interps[sid] = CalibrationInterpolator(cal)

    calibrated_shots = [sid for sid, interp in interps.items() if not interp.is_empty]
    if len(calibrated_shots) < 2:
        return calibrations, []

    # Pick reference: most keyframes
    ref_shot_id = max(
        calibrated_shots,
        key=lambda sid: len(calibrations[sid].frames),
    )

    track_lookups: dict[str, dict[str, dict[int, list[float]]]] = {
        sid: _build_track_frame_lookup(tr) for sid, tr in tracks_by_shot.items()
    }

    diagnostics: list[ShotAlignment] = []
    refined = dict(calibrations)
    for shot_id in calibrated_shots:
        if shot_id == ref_shot_id:
            continue
        if shot_id not in track_lookups or ref_shot_id not in track_lookups:
            continue

        src, dst = _gather_correspondences(
            ref_shot_id=ref_shot_id,
            other_shot_id=shot_id,
            matches=matches,
            track_lookups=track_lookups,
            interps=interps,
            sync_offsets=sync_offsets,
        )
        if len(src) < min_correspondences:
            diagnostics.append(ShotAlignment(
                shot_id=shot_id,
                n_correspondences=int(len(src)),
                yaw_correction_deg=0.0,
                translation_xy_m=(0.0, 0.0),
                residual_before_m=float("inf"),
                residual_after_m=float("inf"),
                accepted=False,
            ))
            continue

        # Robust subset via simple inlier filtering: drop the worst 10% of
        # raw distances, fit, then re-fit on inliers to the fitted residual.
        diffs = dst - src
        d_norms = np.linalg.norm(diffs, axis=1)
        residual_before = float(np.median(d_norms))

        cutoff = float(np.quantile(d_norms, 0.9))
        keep = d_norms <= cutoff
        if int(keep.sum()) < min_correspondences:
            keep = np.ones(len(src), dtype=bool)

        yaw, t_xy = _yaw_procrustes(src[keep], dst[keep])
        # Re-evaluate full residual after the fit
        c, s = float(np.cos(yaw)), float(np.sin(yaw))
        R = np.array([[c, -s], [s, c]], dtype=np.float64)
        residual_after = float(
            np.median(np.linalg.norm(dst - (src @ R.T + t_xy), axis=1))
        )
        yaw_deg = float(np.degrees(yaw))
        t_norm = float(np.hypot(t_xy[0], t_xy[1]))

        accept = (
            abs(yaw_deg) <= max_yaw_deg
            and t_norm <= max_translation_m
            and residual_after < residual_before
        )
        diagnostics.append(ShotAlignment(
            shot_id=shot_id,
            n_correspondences=int(len(src)),
            yaw_correction_deg=yaw_deg,
            translation_xy_m=(float(t_xy[0]), float(t_xy[1])),
            residual_before_m=residual_before,
            residual_after_m=residual_after,
            accepted=accept,
        ))
        if not accept:
            continue

        cal = calibrations[shot_id]
        new_frames = [
            _apply_yaw_translation_to_frame(cf, yaw, t_xy) for cf in cal.frames
        ]
        refined[shot_id] = CalibrationResult(
            shot_id=cal.shot_id,
            camera_type=cal.camera_type,
            frames=new_frames,
        )

    return refined, diagnostics
