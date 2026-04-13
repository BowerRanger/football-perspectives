"""Stage 6: 3D Triangulation.

Strict multi-view triangulation.  A player is only output if they have at
least 2 views where the corresponding shot has a non-empty calibration.
For each pose frame and each COCO joint, we need at least 2 high-confidence
2D observations before running weighted DLT + RANSAC.  Joints (or whole
frames) that fail that bar stay NaN; no monocular fallback.

Per-frame camera parameters are interpolated between the sparse
PnLCalib keyframes using SLERP on rotation and linear interpolation on
focal length (the world position is constant per shot thanks to the
static-camera calibration stage).

Player display names are sourced live from ``tracks/<shot>_tracks.json``
(first non-empty ``Track.player_name`` walking ``MatchedPlayer.views`` in
order), so manual renames in the web UI propagate on the next
triangulation run without re-running matching or pose.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from src.pipeline.base import BaseStage
from src.schemas.calibration import CalibrationResult
from src.schemas.player_matches import MatchedPlayer, PlayerMatches, PlayerView
from src.schemas.poses import Keypoint, PosesResult
from src.schemas.shots import ShotsManifest
from src.schemas.sync_map import SyncMap
from src.schemas.tracks import TracksResult
from src.schemas.triangulated import TriangulatedPlayer
from src.utils.camera import build_projection_matrix
from src.utils.triangulation import (
    enforce_bone_lengths,
    ransac_triangulate,
    snap_feet_to_ground,
    temporal_smooth_savgol,
)
from src.utils.triangulation_calib import CalibrationInterpolator

logger = logging.getLogger(__name__)

_N_COCO_JOINTS = 17


def _build_pose_lookup(
    poses: PosesResult,
) -> dict[str, dict[int, list[Keypoint]]]:
    """Build ``{track_id: {frame: [Keypoint]}}`` for O(1) lookup."""
    lookup: dict[str, dict[int, list[Keypoint]]] = {}
    for player in poses.players:
        frame_map: dict[int, list[Keypoint]] = {}
        for pf in player.frames:
            frame_map[pf.frame] = pf.keypoints
        lookup[player.track_id] = frame_map
    return lookup


def _build_track_metadata(
    tracks: TracksResult,
) -> dict[str, tuple[str, str, str]]:
    """Return ``{track_id: (player_id, player_name, team)}`` for a shot."""
    return {
        t.track_id: (t.player_id or "", t.player_name or "", t.team or "")
        for t in tracks.tracks
    }


def _pick_player_identity(
    player: MatchedPlayer,
    track_meta_by_shot: dict[str, dict[str, tuple[str, str, str]]],
) -> tuple[str, str]:
    """Pick ``(player_name, team)`` for a matched player.

    Walks ``player.views`` in order and takes the first non-empty name from
    the corresponding track.  Team follows the same rule, falling back to
    ``player.team`` if no view has a team set.
    """
    name = ""
    team = player.team or ""
    for view in player.views:
        meta = track_meta_by_shot.get(view.shot_id, {}).get(view.track_id)
        if meta is None:
            continue
        _, track_name, track_team = meta
        if not name and track_name:
            name = track_name
        if not team and track_team:
            team = track_team
        if name and team:
            break
    return name, team


class TriangulationStage(BaseStage):
    name = "triangulation"

    def is_complete(self) -> bool:
        tri_dir = self.output_dir / "triangulated"
        return tri_dir.exists() and any(tri_dir.glob("*.npz"))

    def run(self) -> None:  # noqa: C901 — orchestration
        cfg = self.config.get("triangulation", {})
        min_conf = float(cfg.get("min_keypoint_confidence", 0.3))
        ransac_thresh = float(cfg.get("ransac_threshold", 15.0))
        savgol_window = int(cfg.get("savgol_window", 7))
        savgol_order = int(cfg.get("savgol_order", 3))
        bone_tol = float(cfg.get("bone_length_tolerance", 0.2))
        ground_snap_vel = float(cfg.get("ground_snap_velocity", 0.1))

        # ── Load inputs ──
        matches = PlayerMatches.load(
            self.output_dir / "matching" / "player_matches.json"
        )
        sync_map = SyncMap.load(self.output_dir / "sync" / "sync_map.json")
        manifest = ShotsManifest.load_or_infer(
            self.output_dir / "shots", persist=False
        )

        # Build sync offsets: {shot_id: frame_offset} (reference shot is 0)
        sync_offsets: dict[str, int] = {sync_map.reference_shot: 0}
        for alignment in sync_map.alignments:
            sync_offsets[alignment.shot_id] = alignment.frame_offset

        # ── Per-shot inputs ──
        interps_by_shot: dict[str, CalibrationInterpolator] = {}
        poses_by_shot: dict[str, dict[str, dict[int, list[Keypoint]]]] = {}
        track_meta_by_shot: dict[str, dict[str, tuple[str, str, str]]] = {}

        cal_dir = self.output_dir / "calibration"
        poses_dir = self.output_dir / "poses"
        tracks_dir = self.output_dir / "tracks"

        calibrated_shot_ids: set[str] = set()

        for shot in manifest.shots:
            cal_path = cal_dir / f"{shot.id}_calibration.json"
            if cal_path.exists():
                cal = CalibrationResult.load(cal_path)
                interp = CalibrationInterpolator(cal)
                interps_by_shot[shot.id] = interp
                if not interp.is_empty:
                    calibrated_shot_ids.add(shot.id)

            poses_path = poses_dir / f"{shot.id}_poses.json"
            if poses_path.exists():
                poses_by_shot[shot.id] = _build_pose_lookup(PosesResult.load(poses_path))

            tracks_path = tracks_dir / f"{shot.id}_tracks.json"
            if tracks_path.exists():
                track_meta_by_shot[shot.id] = _build_track_metadata(
                    TracksResult.load(tracks_path)
                )

        if not calibrated_shot_ids:
            logger.warning("No calibrated shots — skipping triangulation")
            return

        # Determine reference frame range across all shots (in reference time)
        all_ref_frames: set[int] = set()
        for shot_id, shot_poses in poses_by_shot.items():
            offset = sync_offsets.get(shot_id, 0)
            for track_frames in shot_poses.values():
                for local_frame in track_frames:
                    all_ref_frames.add(local_frame + offset)

        if not all_ref_frames:
            logger.warning("No pose frames found — skipping triangulation")
            return

        frame_min = min(all_ref_frames)
        frame_max = max(all_ref_frames)
        frame_range = list(range(frame_min, frame_max + 1))
        n_frames = len(frame_range)
        fps = manifest.fps if manifest.fps > 0 else 25.0

        tri_dir = self.output_dir / "triangulated"
        tri_dir.mkdir(parents=True, exist_ok=True)

        print(
            f"  -> triangulating {len(matches.matched_players)} players "
            f"across {n_frames} frames ({len(calibrated_shot_ids)} calibrated shots)"
        )

        n_saved = 0
        for player in matches.matched_players:
            calibrated_views = [
                v for v in player.views if v.shot_id in calibrated_shot_ids
            ]
            if len(calibrated_views) < 2:
                logger.warning(
                    "Skipping %s: only %d calibrated view(s) — triangulation "
                    "requires ≥2",
                    player.player_id,
                    len(calibrated_views),
                )
                continue

            player_name, team = _pick_player_identity(player, track_meta_by_shot)

            triangulated = self._triangulate_player(
                player=player,
                calibrated_views=calibrated_views,
                frame_range=frame_range,
                sync_offsets=sync_offsets,
                interps_by_shot=interps_by_shot,
                poses_by_shot=poses_by_shot,
                min_conf=min_conf,
                ransac_thresh=ransac_thresh,
            )
            if triangulated is None:
                logger.warning(
                    "Player %s (%s): no valid frames after triangulation — skipping",
                    player.player_id,
                    player_name or "<unnamed>",
                )
                continue

            positions, confidences, reproj_errors, n_views_arr = triangulated
            positions = temporal_smooth_savgol(
                positions, window=savgol_window, order=savgol_order,
            )
            positions = enforce_bone_lengths(positions, tolerance=bone_tol)
            positions = snap_feet_to_ground(
                positions, velocity_threshold=ground_snap_vel,
            )

            result = TriangulatedPlayer(
                player_id=player.player_id,
                player_name=player_name,
                team=team,
                positions=positions,
                confidences=confidences,
                reprojection_errors=reproj_errors,
                num_views=n_views_arr,
                fps=fps,
                start_frame=frame_min,
            )
            result.save(tri_dir / f"{player.player_id}_3d_joints.npz")
            n_saved += 1

        print(f"  -> saved {n_saved} player triangulations to triangulated/")

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _triangulate_player(
        *,
        player: MatchedPlayer,
        calibrated_views: list[PlayerView],
        frame_range: list[int],
        sync_offsets: dict[str, int],
        interps_by_shot: dict[str, CalibrationInterpolator],
        poses_by_shot: dict[str, dict[str, dict[int, list[Keypoint]]]],
        min_conf: float,
        ransac_thresh: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
        """Triangulate a single player across all frames.

        Returns ``(positions, confidences, reproj_errors, n_views)`` or
        ``None`` if every frame ends up NaN.
        """
        n_frames = len(frame_range)
        positions = np.full((n_frames, _N_COCO_JOINTS, 3), np.nan, dtype=np.float32)
        confidences = np.zeros((n_frames, _N_COCO_JOINTS), dtype=np.float32)
        reproj_errors = np.full((n_frames, _N_COCO_JOINTS), np.nan, dtype=np.float32)
        n_views_arr = np.zeros((n_frames, _N_COCO_JOINTS), dtype=np.int8)

        for fi, ref_frame in enumerate(frame_range):
            # Gather interpolated calibrations + keypoints for each view.
            view_data: list[tuple[np.ndarray, list[Keypoint]]] = []
            for view in calibrated_views:
                offset = sync_offsets.get(view.shot_id, 0)
                local_frame = ref_frame - offset

                track_poses = poses_by_shot.get(view.shot_id, {}).get(view.track_id)
                if track_poses is None:
                    continue
                keypoints = track_poses.get(local_frame)
                if keypoints is None:
                    continue

                interp = interps_by_shot.get(view.shot_id)
                if interp is None or interp.is_empty:
                    continue
                cal = interp.at(local_frame)
                if cal is None:
                    continue

                P = build_projection_matrix(cal.K, cal.rvec, cal.tvec)
                view_data.append((P, keypoints))

            # Strict multi-view: require ≥ 2 observations per joint.
            if len(view_data) < 2:
                continue

            for j in range(_N_COCO_JOINTS):
                obs_P: list[np.ndarray] = []
                obs_uv: list[np.ndarray] = []
                obs_w: list[float] = []
                for P, keypoints in view_data:
                    if j >= len(keypoints):
                        continue
                    kp = keypoints[j]
                    if kp.conf < min_conf:
                        continue
                    obs_P.append(P)
                    obs_uv.append(np.array([kp.x, kp.y], dtype=np.float64))
                    obs_w.append(kp.conf)

                if len(obs_P) < 2:
                    continue

                pt, err, nv = ransac_triangulate(
                    obs_P, obs_uv, obs_w, threshold=ransac_thresh,
                )
                if np.any(np.isnan(pt)):
                    continue
                positions[fi, j] = pt
                confidences[fi, j] = float(np.mean(obs_w))
                reproj_errors[fi, j] = err
                n_views_arr[fi, j] = nv

        if not np.any(~np.isnan(positions[:, :, 0])):
            return None
        return positions, confidences, reproj_errors, n_views_arr
