"""Cross-view bundle adjustment for camera calibration refinement.

Uses tracked player foot positions across multiple synchronized views
as 2D observations that must project to consistent 3D points on the
pitch plane (z=0). Optimizes camera parameters to minimize cross-view
reprojection error.
"""

import logging
from dataclasses import dataclass

import cv2
import numpy as np
from scipy.optimize import least_squares

from src.schemas.calibration import CalibrationResult, CameraFrame
from src.schemas.player_matches import PlayerMatches
from src.schemas.sync_map import SyncMap
from src.schemas.tracks import TracksResult
from src.utils.camera import build_projection_matrix


@dataclass
class _Observation:
    """A 2D observation of a player foot position in a specific view."""
    shot_id: str
    frame_idx: int  # in that shot's local time
    ref_frame: int  # in reference time
    pixel_uv: np.ndarray  # (2,) foot pixel position
    player_id: str
    cal_frame_idx: int  # index into the shot's calibration frames list


def _foot_centre(bbox: list[float]) -> np.ndarray:
    """Bottom-centre of bounding box as foot position estimate."""
    return np.array([(bbox[0] + bbox[2]) / 2.0, bbox[3]], dtype=np.float64)


def _gather_observations(
    calibrations: dict[str, CalibrationResult],
    tracks: dict[str, TracksResult],
    sync_offsets: dict[str, int],
    matches: PlayerMatches,
    max_frames: int = 200,
) -> list[_Observation]:
    """Gather cross-view player foot observations for bundle adjustment."""
    observations: list[_Observation] = []

    # Build frame lookup: {shot_id: {track_id: {frame: TrackFrame}}}
    track_lookup: dict[str, dict[str, dict[int, list[float]]]] = {}
    for shot_id, tr in tracks.items():
        by_track: dict[str, dict[int, list[float]]] = {}
        for track in tr.tracks:
            if track.class_name == "ball":
                continue
            frame_map: dict[int, list[float]] = {}
            for tf in track.frames:
                frame_map[tf.frame] = tf.bbox
            by_track[track.track_id] = frame_map
        track_lookup[shot_id] = by_track

    # For each matched player, find frames where they're visible in 2+ views
    for player in matches.matched_players:
        if len(player.views) < 2:
            continue

        # Collect all reference frames where this player is visible in any view
        ref_frames_per_view: dict[str, set[int]] = {}
        for view in player.views:
            shot_id = view.shot_id
            if shot_id not in calibrations or not calibrations[shot_id].frames:
                continue
            offset = sync_offsets.get(shot_id, 0)
            frames_for_track = track_lookup.get(shot_id, {}).get(view.track_id, {})
            ref_frames = {f + offset for f in frames_for_track.keys()}
            ref_frames_per_view[shot_id] = ref_frames

        if len(ref_frames_per_view) < 2:
            continue

        # Find reference frames where player is visible in 2+ views simultaneously
        all_ref = set.intersection(*ref_frames_per_view.values())
        if not all_ref:
            continue

        # Sample frames (don't use all — too many for optimization)
        sampled = sorted(all_ref)
        if len(sampled) > max_frames:
            step = len(sampled) // max_frames
            sampled = sampled[::step][:max_frames]

        for ref_frame in sampled:
            for view in player.views:
                shot_id = view.shot_id
                if shot_id not in ref_frames_per_view:
                    continue
                offset = sync_offsets.get(shot_id, 0)
                local_frame = ref_frame - offset
                bbox = track_lookup.get(shot_id, {}).get(view.track_id, {}).get(local_frame)
                if bbox is None:
                    continue

                # Find nearest calibration frame
                cal = calibrations[shot_id]
                if not cal.frames:
                    continue
                nearest_cf_idx = min(
                    range(len(cal.frames)),
                    key=lambda i: abs(cal.frames[i].frame - local_frame),
                )

                observations.append(_Observation(
                    shot_id=shot_id,
                    frame_idx=local_frame,
                    ref_frame=ref_frame,
                    pixel_uv=_foot_centre(bbox),
                    player_id=player.player_id,
                    cal_frame_idx=nearest_cf_idx,
                ))

    return observations


def refine_calibrations(
    calibrations: dict[str, CalibrationResult],
    tracks: dict[str, TracksResult],
    sync_map: SyncMap,
    matches: PlayerMatches,
) -> dict[str, CalibrationResult]:
    """Refine camera calibrations using cross-view player correspondences.

    Uses scipy.optimize.least_squares to jointly optimize camera extrinsics
    (rvec, tvec) and player 3D positions (on the z=0 plane) to minimize
    cross-view reprojection error.
    """
    sync_offsets: dict[str, int] = {sync_map.reference_shot: 0}
    for a in sync_map.alignments:
        sync_offsets[a.shot_id] = a.frame_offset

    observations = _gather_observations(calibrations, tracks, sync_offsets, matches)
    if len(observations) < 10:
        logging.warning("Too few cross-view observations (%d) for refinement", len(observations))
        return calibrations

    # Build parameter vector: [rvec, tvec] per calibration frame per shot
    # and [x, y] per unique 3D point (player at reference frame)
    shot_ids = sorted(calibrations.keys())
    cam_param_index: dict[tuple[str, int], int] = {}  # (shot_id, cf_idx) → param offset
    param_list: list[float] = []

    for shot_id in shot_ids:
        for ci, cf in enumerate(calibrations[shot_id].frames):
            cam_param_index[(shot_id, ci)] = len(param_list)
            param_list.extend(cf.rotation_vector)     # 3 params
            param_list.extend(cf.translation_vector)   # 3 params

    n_cam_params = len(param_list)

    # 3D point parameters: (x, y) per unique (player_id, ref_frame)
    point_keys: list[tuple[str, int]] = []
    point_index: dict[tuple[str, int], int] = {}

    for obs in observations:
        key = (obs.player_id, obs.ref_frame)
        if key not in point_index:
            point_index[key] = n_cam_params + len(point_keys) * 2
            point_keys.append(key)
            # Initialize 3D point at pitch centre
            param_list.extend([52.5, 34.0])

    x0 = np.array(param_list, dtype=np.float64)

    # K matrices per (shot_id, cf_idx)
    K_by_cam: dict[tuple[str, int], np.ndarray] = {}
    for shot_id in shot_ids:
        for ci, cf in enumerate(calibrations[shot_id].frames):
            K_by_cam[(shot_id, ci)] = np.array(cf.intrinsic_matrix, dtype=np.float64)

    def residuals(x: np.ndarray) -> np.ndarray:
        res = np.empty(len(observations) * 2, dtype=np.float64)
        for i, obs in enumerate(observations):
            cam_key = (obs.shot_id, obs.cal_frame_idx)
            cam_off = cam_param_index[cam_key]
            rvec = x[cam_off:cam_off + 3]
            tvec = x[cam_off + 3:cam_off + 6]

            pt_key = (obs.player_id, obs.ref_frame)
            pt_off = point_index[pt_key]
            pt_x, pt_y = x[pt_off], x[pt_off + 1]

            K = K_by_cam[cam_key]
            pt_3d = np.array([[pt_x, pt_y, 0.0]], dtype=np.float64)
            projected, _ = cv2.projectPoints(pt_3d, rvec, tvec, K, None)
            proj = projected.reshape(2)

            res[i * 2] = proj[0] - obs.pixel_uv[0]
            res[i * 2 + 1] = proj[1] - obs.pixel_uv[1]
        return res

    logging.info(
        "Bundle adjustment: %d observations, %d camera params, %d 3D points",
        len(observations), n_cam_params // 6, len(point_keys),
    )

    initial_err = np.mean(np.abs(residuals(x0)))
    result = least_squares(residuals, x0, method="trf", max_nfev=200, verbose=0)
    final_err = np.mean(np.abs(result.fun))

    logging.info("Bundle adjustment: mean error %.1f → %.1f px", initial_err, final_err)
    print(f"     bundle adjustment: {len(observations)} observations, mean error {initial_err:.0f} → {final_err:.0f} px")

    # Extract refined calibrations
    refined = {}
    for shot_id in shot_ids:
        cal = calibrations[shot_id]
        new_frames = []
        for ci, cf in enumerate(cal.frames):
            cam_off = cam_param_index[(shot_id, ci)]
            new_rvec = result.x[cam_off:cam_off + 3].tolist()
            new_tvec = result.x[cam_off + 3:cam_off + 6].tolist()
            new_frames.append(CameraFrame(
                frame=cf.frame,
                intrinsic_matrix=cf.intrinsic_matrix,
                rotation_vector=new_rvec,
                translation_vector=new_tvec,
                reprojection_error=cf.reprojection_error,
                num_correspondences=cf.num_correspondences,
                confidence=cf.confidence,
                tracked_landmark_types=cf.tracked_landmark_types,
            ))
        refined[shot_id] = CalibrationResult(
            shot_id=shot_id,
            camera_type=cal.camera_type,
            frames=new_frames,
        )

    return refined
