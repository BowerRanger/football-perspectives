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
from src.schemas.triangulated import TriangulatedBall, TriangulatedPlayer
from src.utils.ball_reconstruction import reconstruct_ball
from src.utils.camera import build_projection_matrix
from src.utils.single_shot_reconstruction import reconstruct_player as _single_shot_reconstruct
from src.utils.triangulation import (
    enforce_bone_lengths,
    ransac_triangulate,
    snap_feet_to_ground,
    temporal_smooth_savgol,
)
from src.utils.calibration_align import align_shots
from src.utils.triangulation_calib import CalibrationInterpolator
from src.utils.triangulation_dedupe import deduplicate_players

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


def _dedupe_views_by_shot(
    views: list[PlayerView],
    poses_by_shot: dict[str, dict[str, dict[int, list[Keypoint]]]],
) -> list[PlayerView]:
    """Collapse a player's views to at most one per shot.

    The matching stage can group several tracks from the same shot under
    one ``player_id``.  Triangulation needs **distinct cameras** — two
    observations from the same camera are geometrically degenerate (both
    rays originate at the same camera centre), so feeding duplicate-shot
    views into weighted DLT produces garbage.

    For each shot we keep the view whose track has the most pose frames
    in that shot, which is a simple heuristic for "the main track" when
    matching has merged a primary track with a brief spurious copy.
    """
    by_shot: dict[str, PlayerView] = {}
    for view in views:
        existing = by_shot.get(view.shot_id)
        if existing is None:
            by_shot[view.shot_id] = view
            continue
        existing_frames = len(
            poses_by_shot.get(view.shot_id, {}).get(existing.track_id, {})
        )
        new_frames = len(
            poses_by_shot.get(view.shot_id, {}).get(view.track_id, {})
        )
        if new_frames > existing_frames:
            by_shot[view.shot_id] = view
    return list(by_shot.values())


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
        savgol_window = int(cfg.get("savgol_window", 11))
        savgol_order = int(cfg.get("savgol_order", 2))
        savgol_max_gap_fill = int(cfg.get("savgol_max_gap_fill", 5))
        method_switch_hysteresis = int(cfg.get("method_switch_hysteresis", 3))
        bone_tol = float(cfg.get("bone_length_tolerance", 0.2))
        ground_snap_vel = float(cfg.get("ground_snap_velocity", 0.1))
        # Single-shot fallback: when only one calibrated view has the
        # player at a given frame, reconstruct the 3D pose via
        # foot-grounding + vertical-body assumption.  Multi-view
        # triangulation is still used for frames where ≥2 calibrated
        # views see the player.
        allow_single_shot = bool(cfg.get("allow_single_shot", True))
        # When single-shot is enabled, allow bounded extrapolation of
        # the per-frame calibration outside the keyframe range.  The
        # fallback is tolerant of a small rotation error and we'd
        # rather have approximate coverage of the un-keyframed frames
        # than no coverage at all.  Set to 0 to disable extrapolation.
        single_shot_extrapolation_frames = int(
            cfg.get("single_shot_extrapolation_frames", 200)
        )
        dedupe_distance_m = float(cfg.get("dedupe_distance_m", 1.5))
        dedupe_min_overlap_frames = int(cfg.get("dedupe_min_overlap_frames", 30))

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
        cal_dir = self.output_dir / "calibration"
        poses_dir = self.output_dir / "poses"
        tracks_dir = self.output_dir / "tracks"

        calibrations_by_shot: dict[str, CalibrationResult] = {}
        tracks_full_by_shot: dict[str, TracksResult] = {}
        poses_by_shot: dict[str, dict[str, dict[int, list[Keypoint]]]] = {}
        track_meta_by_shot: dict[str, dict[str, tuple[str, str, str]]] = {}

        for shot in manifest.shots:
            cal_path = cal_dir / f"{shot.id}_calibration.json"
            if cal_path.exists():
                calibrations_by_shot[shot.id] = CalibrationResult.load(cal_path)

            poses_path = poses_dir / f"{shot.id}_poses.json"
            if poses_path.exists():
                poses_by_shot[shot.id] = _build_pose_lookup(PosesResult.load(poses_path))

            tracks_path = tracks_dir / f"{shot.id}_tracks.json"
            if tracks_path.exists():
                tr = TracksResult.load(tracks_path)
                tracks_full_by_shot[shot.id] = tr
                track_meta_by_shot[shot.id] = _build_track_metadata(tr)

        # ── Cross-shot alignment ──
        # Each shot's calibration sits in its own world frame; we align
        # every non-reference shot's frame onto the reference shot's
        # using matched-player foot positions.  Skipped when there's
        # only one calibrated shot or align_shots is disabled.
        align_enabled = bool(cfg.get("align_shots", True))
        if align_enabled and len(calibrations_by_shot) >= 2:
            try:
                aligned, align_diags = align_shots(
                    calibrations_by_shot,
                    tracks_full_by_shot,
                    matches,
                    sync_map,
                )
                if align_diags:
                    accepted = sum(1 for d in align_diags if d.accepted)
                    print(
                        f"  -> cross-shot alignment: {accepted}/{len(align_diags)} "
                        f"shots aligned to reference"
                    )
                    for d in align_diags:
                        flag = "✓" if d.accepted else "·"
                        print(
                            f"     {flag} {d.shot_id}: yaw={d.yaw_correction_deg:+.1f}° "
                            f"t=({d.translation_xy_m[0]:+.1f},{d.translation_xy_m[1]:+.1f})m "
                            f"residual {d.residual_before_m:.2f}→{d.residual_after_m:.2f}m "
                            f"({d.n_correspondences} obs)"
                        )
                calibrations_by_shot = aligned
                # Persist the refined calibrations so the web dashboard
                # picks up the aligned versions on next reload.
                for sid, cal in aligned.items():
                    cal.save(cal_dir / f"{sid}_calibration.json")
            except Exception as exc:  # noqa: BLE001
                logger.warning("cross-shot alignment failed: %s", exc)

        # Build interpolators from (possibly aligned) calibrations
        interps_by_shot: dict[str, CalibrationInterpolator] = {}
        calibrated_shot_ids: set[str] = set()
        for shot_id, cal in calibrations_by_shot.items():
            interp = CalibrationInterpolator(cal)
            interps_by_shot[shot_id] = interp
            if not interp.is_empty:
                calibrated_shot_ids.add(shot_id)

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
        # Clear stale outputs from a previous run.  Without this, players
        # that get merged away by deduplication on this run would still
        # have their old .npz files on disk and get picked up by the
        # viewer as ghost duplicates.
        for stale in tri_dir.glob("*_3d_joints.npz"):
            stale.unlink()

        print(
            f"  -> triangulating {len(matches.matched_players)} players "
            f"across {n_frames} frames ({len(calibrated_shot_ids)} calibrated shots)"
        )

        triangulated_players: list[TriangulatedPlayer] = []
        min_shots = 1 if allow_single_shot else 2
        for player in matches.matched_players:
            calibrated_views = [
                v for v in player.views if v.shot_id in calibrated_shot_ids
            ]
            # Collapse duplicate-shot entries (matching stage sometimes groups
            # multiple tracks from the same shot under one player_id, which
            # triangulation treats as the same camera twice → degenerate).
            calibrated_views = _dedupe_views_by_shot(calibrated_views, poses_by_shot)
            unique_shots = {v.shot_id for v in calibrated_views}
            if len(unique_shots) < min_shots:
                logger.warning(
                    "Skipping %s: only %d unique calibrated shot(s) — "
                    "requires ≥%d",
                    player.player_id,
                    len(unique_shots),
                    min_shots,
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
                allow_single_shot=allow_single_shot,
                single_shot_extrapolation_frames=single_shot_extrapolation_frames,
                method_switch_hysteresis=method_switch_hysteresis,
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
                positions,
                window=savgol_window,
                order=savgol_order,
                max_gap_fill=savgol_max_gap_fill,
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
            triangulated_players.append(result)

        # ── Deduplicate co-located players ──
        # The matching stage occasionally emits two MatchedPlayer
        # records for one physical player (cross-view re-id failure).
        # Merge them now so the bird's-eye view doesn't show ghost
        # duplicates.
        n_before = len(triangulated_players)
        triangulated_players = deduplicate_players(
            triangulated_players,
            distance_m=dedupe_distance_m,
            min_overlap_frames=dedupe_min_overlap_frames,
        )
        n_after = len(triangulated_players)
        if n_after < n_before:
            print(
                f"  -> merged {n_before - n_after} duplicate player(s) "
                f"({n_before} → {n_after})"
            )

        for result in triangulated_players:
            result.save(tri_dir / f"{result.player_id}_3d_joints.npz")

        print(f"  -> saved {len(triangulated_players)} player triangulations to triangulated/")

        # ── Ball reconstruction ──
        if bool(cfg.get("reconstruct_ball", True)):
            try:
                ball = reconstruct_ball(
                    tracks_full_by_shot,
                    interps_by_shot,
                    sync_offsets,
                    frame_range,
                    fps,
                )
                if ball is not None:
                    ball.save(tri_dir / "ball_3d_trajectory.npz")
                    n_valid = int(np.sum(~np.isnan(ball.positions[:, 0])))
                    n_multi = int(np.sum(ball.methods == 1))
                    n_ground = int(np.sum(ball.methods == 2))
                    n_flight = int(np.sum(ball.methods == 3))
                    print(
                        f"  -> ball: {n_valid}/{len(frame_range)} frames "
                        f"(multi={n_multi}, ground={n_ground}, flight={n_flight})"
                    )
                else:
                    # Remove any stale ball file from a prior run
                    stale = tri_dir / "ball_3d_trajectory.npz"
                    if stale.exists():
                        stale.unlink()
                    print("  -> ball: no ball detections found in any shot")
            except Exception as exc:  # noqa: BLE001
                logger.warning("ball reconstruction failed: %s", exc)

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
        allow_single_shot: bool,
        single_shot_extrapolation_frames: int = 0,
        method_switch_hysteresis: int = 3,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
        """Triangulate a single player across all frames.

        Three-pass design so the per-frame method choice (multi-view vs
        single-shot) can be smoothed with hysteresis before any positions
        are computed:

          1. **Gather**: collect strict + extrapolated views per frame.
          2. **Choose**: pick a method per frame, then suppress short
             single-shot islands embedded inside a multi-view run — those
             are the visible "flicker" frames where the geometry
             discontinuity would otherwise be papered over by savgol.
          3. **Compute**: run the chosen method per frame.

        Returns ``(positions, confidences, reproj_errors, n_views)`` or
        ``None`` if every frame ends up NaN.
        """
        n_frames = len(frame_range)
        positions = np.full((n_frames, _N_COCO_JOINTS, 3), np.nan, dtype=np.float32)
        confidences = np.zeros((n_frames, _N_COCO_JOINTS), dtype=np.float32)
        reproj_errors = np.full((n_frames, _N_COCO_JOINTS), np.nan, dtype=np.float32)
        n_views_arr = np.zeros((n_frames, _N_COCO_JOINTS), dtype=np.int8)

        # ── Pass 1: gather views per frame ──
        view_data: list[
            tuple[
                list[tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[Keypoint]]],
                list[tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[Keypoint]]],
            ]
        ] = []
        for ref_frame in frame_range:
            strict_views: list[
                tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[Keypoint]]
            ] = []
            extrapolated_views: list[
                tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[Keypoint]]
            ] = []

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

                cal_strict = interp.at(local_frame)
                if cal_strict is not None:
                    P = build_projection_matrix(cal_strict.K, cal_strict.rvec, cal_strict.tvec)
                    strict_views.append(
                        (view.shot_id, P, cal_strict.K, cal_strict.rvec, cal_strict.tvec, keypoints),
                    )
                    continue

                if not allow_single_shot or single_shot_extrapolation_frames <= 0:
                    continue
                cal_ext = interp.at_nearest(
                    local_frame,
                    max_extrapolation_frames=single_shot_extrapolation_frames,
                )
                if cal_ext is None:
                    continue
                P = build_projection_matrix(cal_ext.K, cal_ext.rvec, cal_ext.tvec)
                extrapolated_views.append(
                    (view.shot_id, P, cal_ext.K, cal_ext.rvec, cal_ext.tvec, keypoints),
                )

            view_data.append((strict_views, extrapolated_views))

        # ── Pass 2: choose method per frame with hysteresis ──
        methods = _choose_methods(
            view_data,
            allow_single_shot=allow_single_shot,
            hysteresis=method_switch_hysteresis,
        )

        # ── Pass 3: compute positions ──
        for fi, method in enumerate(methods):
            strict_views, extrapolated_views = view_data[fi]
            if method == "multi":
                for j in range(_N_COCO_JOINTS):
                    obs_P: list[np.ndarray] = []
                    obs_uv: list[np.ndarray] = []
                    obs_w: list[float] = []
                    for _sid, P, _K, _rv, _tv, keypoints in strict_views:
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
                    # Plausibility: a player's joint can't sit below
                    # the pitch surface or absurdly high above it.
                    # Multi-view DLT produces wildly off 3D points
                    # when the two views' calibrations disagree
                    # significantly; reject those rather than
                    # polluting the bird's-eye view.
                    if pt[2] < -0.5 or pt[2] > 3.5:
                        continue
                    if not (-15.0 <= pt[0] <= 120.0 and -15.0 <= pt[1] <= 83.0):
                        continue
                    positions[fi, j] = pt
                    confidences[fi, j] = float(np.mean(obs_w))
                    reproj_errors[fi, j] = err
                    n_views_arr[fi, j] = nv
            elif method == "single":
                fallback_view = (
                    strict_views[0] if strict_views else
                    (extrapolated_views[0] if extrapolated_views else None)
                )
                if fallback_view is not None:
                    _sid, _P, K, rvec, tvec, keypoints = fallback_view
                    result = _single_shot_reconstruct(
                        keypoints=keypoints,
                        K=K, rvec=rvec, tvec=tvec,
                        min_foot_confidence=min_conf,
                        min_joint_confidence=min_conf,
                    )
                    if result is not None:
                        positions[fi] = result.positions
                        confidences[fi] = result.confidences
                        valid_mask = ~np.isnan(result.positions[:, 0])
                        n_views_arr[fi, valid_mask] = 1
            # else: NaN — savgol short-gap fill will smooth across it

        if not np.any(~np.isnan(positions[:, :, 0])):
            return None
        return positions, confidences, reproj_errors, n_views_arr


def _choose_methods(
    view_data: list[tuple[list, list]],
    *,
    allow_single_shot: bool,
    hysteresis: int,
) -> list[str | None]:
    """Choose ``'multi'`` / ``'single'`` / ``None`` per frame, with hysteresis.

    Raw eligibility:
      - ``multi`` if ≥2 unique strict shots are available
      - ``single`` if at least one view (strict or extrapolated) is
        available and single-shot is enabled
      - ``None`` otherwise

    Hysteresis: a run of ``single`` shorter than ``hysteresis`` frames
    that is bounded on both sides by ``multi`` is downgraded to ``None``.
    Those frames would otherwise produce a geometric discontinuity
    inside an otherwise consistent multi-view trajectory; better to let
    the savgol short-gap fill carry the trajectory across the hole.
    """
    raw: list[str | None] = []
    for strict, ext in view_data:
        unique_strict = {sv[0] for sv in strict}
        if len(unique_strict) >= 2:
            raw.append("multi")
        elif allow_single_shot and (strict or ext):
            raw.append("single")
        else:
            raw.append(None)

    if hysteresis <= 1:
        return raw

    out: list[str | None] = list(raw)
    n = len(out)
    i = 0
    while i < n:
        if out[i] != "single":
            i += 1
            continue
        j = i
        while j < n and out[j] == "single":
            j += 1
        run_len = j - i
        left = out[i - 1] if i > 0 else None
        right = out[j] if j < n else None
        if run_len < hysteresis and left == "multi" and right == "multi":
            for k in range(i, j):
                out[k] = None
        i = j
    return out
