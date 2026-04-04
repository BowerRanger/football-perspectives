import json
import pytest
import numpy as np
from pathlib import Path
from src.pipeline.config import load_config
from src.schemas.player_matches import PlayerMatches
from src.schemas.shots import Shot, ShotsManifest
from src.schemas.sync_map import Alignment, SyncMap
from src.schemas.tracks import Track, TrackFrame, TracksResult
from src.stages.matching import CrossViewMatchingStage, hungarian_match_players


def _make_track(track_id: str, pitch_positions: list[list[float]], team: str = "A") -> Track:
    frames = [
        TrackFrame(frame=i, bbox=[0.0, 0.0, 60.0, 180.0],
                   confidence=0.9, pitch_position=pos)
        for i, pos in enumerate(pitch_positions)
    ]
    return Track(track_id=track_id, class_name="player", team=team, frames=frames)


def test_hungarian_match_nearby_players():
    # Two players in shot A, two in shot B at nearly the same pitch positions
    tracks_a = TracksResult(shot_id="shot_001", tracks=[
        _make_track("T001", [[10.0, 5.0]] * 5),
        _make_track("T002", [[30.0, 10.0]] * 5),
    ])
    tracks_b = TracksResult(shot_id="shot_002", tracks=[
        _make_track("T003", [[10.2, 5.1]] * 5),  # close to T001
        _make_track("T004", [[30.1, 10.2]] * 5), # close to T002
    ])
    matches = hungarian_match_players(
        tracks_a, tracks_b, sync_offset=0, reference_frames=[0, 1, 2, 3, 4]
    )
    assert ("T001", "T003") in matches
    assert ("T002", "T004") in matches


def test_hungarian_match_rejects_distant_players():
    tracks_a = TracksResult(shot_id="shot_001", tracks=[
        _make_track("T001", [[10.0, 5.0]] * 3),
    ])
    tracks_b = TracksResult(shot_id="shot_002", tracks=[
        _make_track("T002", [[80.0, 60.0]] * 3),  # far away — beyond max_distance
    ])
    matches = hungarian_match_players(
        tracks_a, tracks_b, sync_offset=0, reference_frames=[0, 1, 2],
        max_distance_m=5.0,
    )
    assert len(matches) == 0


@pytest.fixture(scope="module")
def two_shot_dir(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("matching_stage")
    shots_dir = root / "shots"
    shots_dir.mkdir()
    tracks_dir = root / "tracks"
    tracks_dir.mkdir()
    sync_dir = root / "sync"
    sync_dir.mkdir()

    ShotsManifest(
        source_file="test.mp4", fps=10.0, total_frames=20,
        shots=[
            Shot(id="shot_001", start_frame=0, end_frame=9,
                 start_time=0.0, end_time=1.0, clip_file="shots/shot_001.mp4"),
            Shot(id="shot_002", start_frame=0, end_frame=9,
                 start_time=0.0, end_time=1.0, clip_file="shots/shot_002.mp4"),
        ],
    ).save(shots_dir / "shots_manifest.json")

    TracksResult(shot_id="shot_001", tracks=[
        _make_track("T001", [[10.0, 5.0]] * 10),
        _make_track("T002", [[30.0, 10.0]] * 10),
    ]).save(tracks_dir / "shot_001_tracks.json")

    TracksResult(shot_id="shot_002", tracks=[
        _make_track("T003", [[10.1, 5.0]] * 10),
        _make_track("T004", [[30.0, 9.9]] * 10),
    ]).save(tracks_dir / "shot_002_tracks.json")

    SyncMap(
        reference_shot="shot_001",
        alignments=[Alignment(shot_id="shot_002", frame_offset=0,
                               confidence=0.9, method="ball_trajectory",
                               overlap_frames=[0, 10])],
    ).save(sync_dir / "sync_map.json")
    return root


def test_matching_stage_writes_player_matches(two_shot_dir):
    cfg = load_config()
    stage = CrossViewMatchingStage(config=cfg, output_dir=two_shot_dir)
    stage.run()
    assert (two_shot_dir / "matching" / "player_matches.json").exists()


def test_matching_stage_is_complete_after_run(two_shot_dir):
    cfg = load_config()
    stage = CrossViewMatchingStage(config=cfg, output_dir=two_shot_dir)
    assert stage.is_complete()


def test_matching_stage_output_has_two_players(two_shot_dir):
    result = PlayerMatches.load(two_shot_dir / "matching" / "player_matches.json")
    assert len(result.matched_players) == 2
    # Each player should have views in both shots
    for mp in result.matched_players:
        shot_ids = {v.shot_id for v in mp.views}
        assert "shot_001" in shot_ids
        assert "shot_002" in shot_ids
