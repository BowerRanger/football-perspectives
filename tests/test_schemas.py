import json
from pathlib import Path
from src.schemas.shots import Shot, ShotsManifest
from src.schemas.calibration import CameraFrame, CalibrationResult
from src.schemas.sync_map import Alignment, SyncMap

def test_shots_manifest_round_trip(tmp_path):
    m = ShotsManifest(
        source_file="input.mp4", fps=25.0, total_frames=100,
        shots=[Shot(id="shot_001", start_frame=0, end_frame=50,
                    start_time=0.0, end_time=2.0,
                    clip_file="shots/shot_001.mp4")]
    )
    p = tmp_path / "manifest.json"
    m.save(p)
    loaded = ShotsManifest.load(p)
    assert loaded.fps == 25.0
    assert loaded.shots[0].id == "shot_001"


def test_shots_manifest_loads_legacy_thumbnail_field(tmp_path):
    p = tmp_path / "legacy_manifest.json"
    p.write_text(json.dumps({
        "source_file": "input.mp4",
        "fps": 25.0,
        "total_frames": 100,
        "shots": [{
            "id": "shot_001",
            "start_frame": 0,
            "end_frame": 50,
            "start_time": 0.0,
            "end_time": 2.0,
            "clip_file": "shots/shot_001.mp4",
            "thumbnail": "shots/shot_001_thumb.jpg",
        }],
    }))

    loaded = ShotsManifest.load(p)
    assert len(loaded.shots) == 1
    assert loaded.shots[0].clip_file == "shots/shot_001.mp4"

def test_calibration_round_trip(tmp_path):
    frame = CameraFrame(
        frame=0,
        intrinsic_matrix=[[1500,0,960],[0,1500,540],[0,0,1]],
        rotation_vector=[0.1,0.2,0.05],
        translation_vector=[-52.5,-34.0,50.0],
        reprojection_error=3.2,
        num_correspondences=8,
        confidence=0.79,
    )
    result = CalibrationResult(shot_id="shot_001", camera_type="static", frames=[frame])
    p = tmp_path / "cal.json"
    result.save(p)
    loaded = CalibrationResult.load(p)
    assert loaded.frames[0].reprojection_error == 3.2

def test_sync_map_round_trip(tmp_path):
    sm = SyncMap(
        reference_shot="shot_001",
        alignments=[Alignment(shot_id="shot_003", frame_offset=-47,
                              confidence=0.92, method="ball_trajectory",
                              overlap_frames=[120, 280])]
    )
    p = tmp_path / "sync.json"
    sm.save(p)
    loaded = SyncMap.load(p)
    assert loaded.alignments[0].frame_offset == -47


from src.schemas.tracks import TrackFrame, Track, TracksResult
from src.schemas.poses import Keypoint, PlayerPoseFrame, PlayerPoses, PosesResult, COCO_KEYPOINT_NAMES
from src.schemas.player_matches import PlayerView, MatchedPlayer, PlayerMatches

def test_tracks_result_round_trip(tmp_path):
    tf = TrackFrame(frame=0, bbox=[10.0, 20.0, 80.0, 200.0], confidence=0.9, pitch_position=[34.2, 21.5])
    track = Track(track_id="T001", class_name="player", team="A", frames=[tf])
    result = TracksResult(shot_id="shot_001", tracks=[track])
    path = tmp_path / "tracks.json"
    result.save(path)
    loaded = TracksResult.load(path)
    assert loaded.shot_id == "shot_001"
    assert len(loaded.tracks) == 1
    assert loaded.tracks[0].track_id == "T001"
    assert loaded.tracks[0].frames[0].pitch_position == [34.2, 21.5]

def test_poses_result_round_trip(tmp_path):
    kp = Keypoint(name="nose", x=100.0, y=50.0, conf=0.91)
    pf = PlayerPoseFrame(frame=0, keypoints=[kp])
    pp = PlayerPoses(track_id="T001", frames=[pf])
    result = PosesResult(shot_id="shot_001", players=[pp])
    path = tmp_path / "poses.json"
    result.save(path)
    loaded = PosesResult.load(path)
    assert loaded.shot_id == "shot_001"
    assert loaded.players[0].frames[0].keypoints[0].name == "nose"
    assert loaded.players[0].frames[0].keypoints[0].conf == 0.91

def test_player_matches_round_trip(tmp_path):
    view = PlayerView(shot_id="shot_001", track_id="T001")
    player = MatchedPlayer(player_id="P001", team="A", views=[view])
    result = PlayerMatches(matched_players=[player])
    path = tmp_path / "matches.json"
    result.save(path)
    loaded = PlayerMatches.load(path)
    assert len(loaded.matched_players) == 1
    assert loaded.matched_players[0].player_id == "P001"
    assert loaded.matched_players[0].views[0].shot_id == "shot_001"

def test_coco_keypoint_names_length():
    assert len(COCO_KEYPOINT_NAMES) == 17
