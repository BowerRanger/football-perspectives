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
                    clip_file="shots/shot_001.mp4",
                    thumbnail="shots/shot_001_thumb.jpg")]
    )
    p = tmp_path / "manifest.json"
    m.save(p)
    loaded = ShotsManifest.load(p)
    assert loaded.fps == 25.0
    assert loaded.shots[0].id == "shot_001"

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
