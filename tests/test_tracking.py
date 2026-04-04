import cv2
import numpy as np
import pytest
from pathlib import Path
from src.pipeline.config import load_config
from src.schemas.shots import Shot, ShotsManifest
from src.schemas.tracks import TracksResult
from src.stages.tracking import PlayerTrackingStage
from src.utils.player_detector import Detection, FakePlayerDetector
from src.utils.team_classifier import FakeTeamClassifier


@pytest.fixture(scope="module")
def tiny_shot_dir(tmp_path_factory) -> Path:
    """Output directory with a shots manifest and a 1-second synthetic clip."""
    root = tmp_path_factory.mktemp("tracking_stage")
    shots_dir = root / "shots"
    shots_dir.mkdir()

    clip_path = shots_dir / "shot_001.mp4"
    writer = cv2.VideoWriter(
        str(clip_path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (320, 240)
    )
    for _ in range(10):
        writer.write(np.full((240, 320, 3), [50, 200, 50], dtype=np.uint8))
    writer.release()

    shot = Shot(
        id="shot_001",
        start_frame=0,
        end_frame=9,
        start_time=0.0,
        end_time=1.0,
        clip_file="shots/shot_001.mp4",
    )
    ShotsManifest(
        source_file="test.mp4", fps=10.0, total_frames=10, shots=[shot]
    ).save(shots_dir / "shots_manifest.json")
    return root


def _one_player_det() -> Detection:
    return Detection(bbox=(50.0, 30.0, 150.0, 200.0), confidence=0.9, class_name="player")


def test_tracking_stage_writes_tracks_file(tiny_shot_dir):
    cfg = load_config()
    stage = PlayerTrackingStage(
        config=cfg,
        output_dir=tiny_shot_dir,
        player_detector=FakePlayerDetector([[_one_player_det()]]),
        team_classifier=FakeTeamClassifier("A"),
    )
    stage.run()
    assert (tiny_shot_dir / "tracks" / "shot_001_tracks.json").exists()


def test_tracking_stage_is_complete_after_run(tiny_shot_dir):
    cfg = load_config()
    stage = PlayerTrackingStage(
        config=cfg,
        output_dir=tiny_shot_dir,
        player_detector=FakePlayerDetector([[_one_player_det()]]),
        team_classifier=FakeTeamClassifier("A"),
    )
    assert stage.is_complete()


def test_tracking_stage_tracks_have_correct_schema(tiny_shot_dir):
    result = TracksResult.load(tiny_shot_dir / "tracks" / "shot_001_tracks.json")
    assert result.shot_id == "shot_001"
    assert len(result.tracks) >= 1
    t = result.tracks[0]
    assert t.team == "A"
    assert len(t.frames) >= 1
    assert len(t.frames[0].bbox) == 4


# Unit tests for PlayerDetector (from plan Task 2)
def test_fake_player_detector_cycles():
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    from src.utils.player_detector import Detection, FakePlayerDetector, PlayerDetector
    dets = [
        [Detection(bbox=(10.0, 20.0, 80.0, 200.0), confidence=0.9, class_name="player")],
        [],
    ]
    detector = FakePlayerDetector(dets)
    assert len(detector.detect(frame)) == 1
    assert len(detector.detect(frame)) == 0
    assert len(detector.detect(frame)) == 1  # cycles


def test_player_detector_is_abstract():
    from src.utils.player_detector import PlayerDetector, FakePlayerDetector
    with pytest.raises(TypeError):
        PlayerDetector()
    assert issubclass(FakePlayerDetector, PlayerDetector)


# Unit tests for TeamClassifier (from plan Task 3)
def test_fake_team_classifier_returns_fixed_label():
    from src.utils.team_classifier import FakeTeamClassifier
    crops = [np.zeros((60, 40, 3), dtype=np.uint8) for _ in range(3)]
    clf = FakeTeamClassifier("B")
    labels = clf.classify(crops)
    assert labels == ["B", "B", "B"]


def test_fake_team_classifier_empty_input():
    from src.utils.team_classifier import FakeTeamClassifier
    clf = FakeTeamClassifier("A")
    assert clf.classify([]) == []


def test_team_classifier_is_abstract():
    from src.utils.team_classifier import TeamClassifier, FakeTeamClassifier
    with pytest.raises(TypeError):
        TeamClassifier()
    assert issubclass(FakeTeamClassifier, TeamClassifier)
