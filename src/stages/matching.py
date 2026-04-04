import logging
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.pipeline.base import BaseStage
from src.schemas.player_matches import MatchedPlayer, PlayerMatches, PlayerView
from src.schemas.shots import ShotsManifest
from src.schemas.sync_map import SyncMap
from src.schemas.tracks import TracksResult

_DEFAULT_MAX_DISTANCE_M = 5.0  # reject matches further apart than this on the pitch


def _mean_pitch_position(
    tracks_result: TracksResult, track_id: str, frames: list[int]
) -> np.ndarray | None:
    """Average pitch position for a track over the given frames. Returns None if no data."""
    positions = []
    frame_set = set(frames)
    for track in tracks_result.tracks:
        if track.track_id != track_id:
            continue
        for tf in track.frames:
            if tf.frame in frame_set and tf.pitch_position is not None:
                positions.append(tf.pitch_position)
    if not positions:
        return None
    return np.mean(positions, axis=0)


def hungarian_match_players(
    shot_a_tracks: TracksResult,
    shot_b_tracks: TracksResult,
    sync_offset: int,
    reference_frames: list[int],
    max_distance_m: float = _DEFAULT_MAX_DISTANCE_M,
) -> list[tuple[str, str]]:
    """
    Match player track IDs between two shots using the Hungarian algorithm.

    sync_offset: alignment.frame_offset — so shot_b_frame = shot_a_frame - sync_offset.
    reference_frames: frame indices in shot_a's time domain used to compute positions.

    Returns list of (track_id_in_shot_a, track_id_in_shot_b) pairs whose pitch
    distance is within max_distance_m.
    """
    tracks_a = [t for t in shot_a_tracks.tracks if t.class_name != "ball"]
    tracks_b = [t for t in shot_b_tracks.tracks if t.class_name != "ball"]
    if not tracks_a or not tracks_b:
        return []

    b_frames = [f - sync_offset for f in reference_frames if f - sync_offset >= 0]

    pos_a = {t.track_id: _mean_pitch_position(shot_a_tracks, t.track_id, reference_frames)
             for t in tracks_a}
    pos_b = {t.track_id: _mean_pitch_position(shot_b_tracks, t.track_id, b_frames)
             for t in tracks_b}

    valid_a = [t.track_id for t in tracks_a if pos_a.get(t.track_id) is not None]
    valid_b = [t.track_id for t in tracks_b if pos_b.get(t.track_id) is not None]
    if not valid_a or not valid_b:
        return []

    # Build cost matrix: (len(valid_a), len(valid_b))
    inf = max_distance_m * 2
    cost = np.full((len(valid_a), len(valid_b)), fill_value=inf)
    for i, tid_a in enumerate(valid_a):
        for j, tid_b in enumerate(valid_b):
            cost[i, j] = float(np.linalg.norm(pos_a[tid_a] - pos_b[tid_b]))

    row_ind, col_ind = linear_sum_assignment(cost)
    return [
        (valid_a[r], valid_b[c])
        for r, c in zip(row_ind, col_ind)
        if cost[r, c] <= max_distance_m
    ]


class CrossViewMatchingStage(BaseStage):
    name = "matching"

    def is_complete(self) -> bool:
        return (self.output_dir / "matching" / "player_matches.json").exists()

    def run(self) -> None:
        matching_dir = self.output_dir / "matching"
        matching_dir.mkdir(parents=True, exist_ok=True)
        cfg = self.config.get("matching", {})
        max_distance_m = cfg.get("max_distance_m", _DEFAULT_MAX_DISTANCE_M)
        n_reference_frames = cfg.get("n_reference_frames", 10)

        manifest = ShotsManifest.load(self.output_dir / "shots" / "shots_manifest.json")
        sync_map = SyncMap.load(self.output_dir / "sync" / "sync_map.json")
        tracks_dir = self.output_dir / "tracks"

        tracks_by_shot: dict[str, TracksResult] = {}
        for shot in manifest.shots:
            path = tracks_dir / f"{shot.id}_tracks.json"
            if path.exists():
                tracks_by_shot[shot.id] = TracksResult.load(path)

        if not tracks_by_shot:
            logging.warning("No track files found in %s — player_matches will be empty", tracks_dir)

        # Assign a global player_id to every track in the reference shot first
        player_counter = 0
        player_id_map: dict[tuple[str, str], str] = {}  # (shot_id, track_id) -> player_id

        ref_id = sync_map.reference_shot
        if ref_id in tracks_by_shot:
            for track in tracks_by_shot[ref_id].tracks:
                if track.class_name == "ball":
                    continue
                player_counter += 1
                pid = f"P{player_counter:03d}"
                player_id_map[(ref_id, track.track_id)] = pid

        # Match each non-reference shot to the reference
        for alignment in sync_map.alignments:
            other_id = alignment.shot_id
            if ref_id not in tracks_by_shot or other_id not in tracks_by_shot:
                continue
            overlap_start, overlap_end = alignment.overlap_frames
            if overlap_end <= overlap_start:
                continue
            step = max(1, (overlap_end - overlap_start) // n_reference_frames)
            ref_frames = list(range(overlap_start, overlap_end, step))[:n_reference_frames]
            matches = hungarian_match_players(
                tracks_by_shot[ref_id],
                tracks_by_shot[other_id],
                sync_offset=alignment.frame_offset,
                reference_frames=ref_frames,
                max_distance_m=max_distance_m,
            )
            logging.info(
                "  -> %s <-> %s: %d matches", ref_id, other_id, len(matches)
            )
            for track_id_ref, track_id_other in matches:
                pid = player_id_map.get((ref_id, track_id_ref))
                if pid is not None:
                    player_id_map[(other_id, track_id_other)] = pid

        # Collect all views per player_id and build output
        pid_to_views: dict[str, list[PlayerView]] = {}
        pid_to_team: dict[str, str] = {}
        for (shot_id, track_id), pid in player_id_map.items():
            pid_to_views.setdefault(pid, []).append(PlayerView(shot_id=shot_id, track_id=track_id))
            if pid not in pid_to_team and shot_id in tracks_by_shot:
                for t in tracks_by_shot[shot_id].tracks:
                    if t.track_id == track_id:
                        pid_to_team[pid] = t.team
                        break

        matched_players = [
            MatchedPlayer(
                player_id=pid,
                team=pid_to_team.get(pid, "unknown"),
                views=views,
            )
            for pid, views in sorted(pid_to_views.items())
        ]
        PlayerMatches(matched_players=matched_players).save(
            matching_dir / "player_matches.json"
        )
        logging.info("  -> %d matched players", len(matched_players))
