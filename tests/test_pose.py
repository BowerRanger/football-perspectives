import cv2
import numpy as np
import pytest
from pathlib import Path
from src.pipeline.config import load_config
from src.schemas.shots import Shot, ShotsManifest
from src.schemas.tracks import Track, TrackFrame, TracksResult
from src.schemas.poses import PosesResult
from src.stages.pose import PoseEstimationStage, smooth_keypoints
from src.utils.pose_estimator import FakePoseEstimator, PoseEstimator
from src.schemas.poses import COCO_KEYPOINT_NAMES


@pytest.fixture(scope="module")
def shot_with_tracks(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("pose_stage")
    shots_dir = root / "shots"
    shots_dir.mkdir()
    tracks_dir = root / "tracks"
    tracks_dir.mkdir()

    clip_path = shots_dir / "shot_001.mp4"
    writer = cv2.VideoWriter(
        str(clip_path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (320, 240)
    )
    for _ in range(10):
        writer.write(np.full((240, 320, 3), [50, 200, 50], dtype=np.uint8))
    writer.release()

    shot = Shot(id="shot_001", start_frame=0, end_frame=9,
                start_time=0.0, end_time=1.0, clip_file="shots/shot_001.mp4")
    ShotsManifest(source_file="test.mp4", fps=10.0, total_frames=10, shots=[shot]).save(
        shots_dir / "shots_manifest.json"
    )

    frames = [TrackFrame(frame=i, bbox=[50.0, 30.0, 150.0, 200.0],
                         confidence=0.9, pitch_position=None) for i in range(10)]
    track = Track(track_id="T001", class_name="player", team="A", frames=frames)
    TracksResult(shot_id="shot_001", tracks=[track]).save(
        tracks_dir / "shot_001_tracks.json"
    )
    return root


def test_pose_stage_writes_poses_file(shot_with_tracks):
    cfg = load_config()
    stage = PoseEstimationStage(
        config=cfg,
        output_dir=shot_with_tracks,
        pose_estimator=FakePoseEstimator(),
    )
    stage.run()
    assert (shot_with_tracks / "poses" / "shot_001_poses.json").exists()


def test_pose_stage_is_complete_after_run(shot_with_tracks):
    cfg = load_config()
    stage = PoseEstimationStage(
        config=cfg,
        output_dir=shot_with_tracks,
        pose_estimator=FakePoseEstimator(),
    )
    assert stage.is_complete()


def test_pose_stage_keypoints_in_frame_coords(shot_with_tracks):
    result = PosesResult.load(shot_with_tracks / "poses" / "shot_001_poses.json")
    assert len(result.players) >= 1
    kps = result.players[0].frames[0].keypoints
    assert len(kps) == 17
    # x should be >= 50 (bbox x1), not in crop-local coords
    assert all(kp.x >= 50.0 for kp in kps)


def test_smooth_keypoints_preserves_track_id():
    from src.schemas.poses import Keypoint, PlayerPoseFrame, PlayerPoses
    frames = [
        PlayerPoseFrame(frame=i, keypoints=[Keypoint(name="nose", x=float(i), y=float(i), conf=0.9)])
        for i in range(5)
    ]
    pp = PlayerPoses(track_id="T001", frames=frames)
    smoothed = smooth_keypoints(pp)
    assert smoothed.track_id == "T001"
    assert len(smoothed.frames) == 5


def test_fake_pose_estimator_returns_17_keypoints():
    estimator = FakePoseEstimator()
    crop = np.zeros((120, 60, 3), dtype=np.uint8)
    kps = estimator.estimate(crop, bbox_offset=(100.0, 50.0))
    assert len(kps) == 17


def test_fake_pose_estimator_applies_offset():
    estimator = FakePoseEstimator()
    crop = np.zeros((120, 60, 3), dtype=np.uint8)
    kps = estimator.estimate(crop, bbox_offset=(100.0, 50.0))
    # All keypoints should have x >= 100 (offset applied)
    assert all(kp.x >= 100.0 for kp in kps)


def test_fake_pose_estimator_keypoint_names():
    estimator = FakePoseEstimator()
    crop = np.zeros((120, 60, 3), dtype=np.uint8)
    kps = estimator.estimate(crop, bbox_offset=(0.0, 0.0))
    names = [kp.name for kp in kps]
    assert names == COCO_KEYPOINT_NAMES


def test_pose_estimator_is_abstract():
    assert issubclass(FakePoseEstimator, PoseEstimator)
    with pytest.raises(TypeError):
        PoseEstimator()
