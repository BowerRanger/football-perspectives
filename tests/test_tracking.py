import cv2
import numpy as np
import pytest
from pathlib import Path
from src.pipeline.config import load_config
from src.schemas.shots import Shot, ShotsManifest
from src.schemas.tracks import Track, TrackFrame, TracksResult
from src.stages.tracking import PlayerTrackingStage
from src.utils.player_detector import Detection, FakePlayerDetector
from src.utils.team_classifier import FakeTeamClassifier
from src.utils.track_backfill import BackfillConfig


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


def _test_cfg() -> dict:
    """Load default config but pin the tracker to bytetrack so these
    fast unit tests don't need BoxMOT installed or the OSNet ReID
    checkpoint on disk. The production default (botsort) is exercised
    end-to-end via the dispatch unit test in test_tracker_dispatch.py."""
    cfg = load_config()
    cfg.setdefault("tracking", {})["tracker"] = "bytetrack"
    return cfg


class _RequiresFitTeamClassifier:
    def __init__(self) -> None:
        self._fitted = False
        self.fit_call_count = 0

    def fit(self, crops: list[np.ndarray]) -> None:
        self.fit_call_count += 1
        if not crops:
            raise ValueError("Need at least one crop")
        self._fitted = True

    def classify(self, crops: list[np.ndarray]) -> list[str]:
        if not self._fitted:
            raise RuntimeError("Call fit() before classify()")
        return ["A"] * len(crops)


def test_tracking_stage_writes_tracks_file(tiny_shot_dir):
    cfg = _test_cfg()
    stage = PlayerTrackingStage(
        config=cfg,
        output_dir=tiny_shot_dir,
        player_detector=FakePlayerDetector([[_one_player_det()]]),
        team_classifier=FakeTeamClassifier("A"),
    )
    stage.run()
    assert (tiny_shot_dir / "tracks" / "shot_001_tracks.json").exists()


def test_tracking_stage_is_complete_after_run(tiny_shot_dir):
    cfg = _test_cfg()
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


def test_tracking_stage_fits_team_classifier_before_classify(tiny_shot_dir):
    cfg = _test_cfg()
    classifier = _RequiresFitTeamClassifier()
    stage = PlayerTrackingStage(
        config=cfg,
        output_dir=tiny_shot_dir,
        player_detector=FakePlayerDetector([[_one_player_det()]]),
        team_classifier=classifier,
    )

    stage.run()

    result = TracksResult.load(tiny_shot_dir / "tracks" / "shot_001_tracks.json")
    assert classifier._fitted
    assert classifier.fit_call_count >= 1
    assert len(result.tracks) >= 1
    assert result.tracks[0].team == "A"


def test_backfill_disabled_by_default_in_config():
    """tracking.backfill.enabled must default to False — the in-stage
    backfill path is opt-in so existing tracks.json outputs stay stable
    unless an operator explicitly turns it on (ball-stage campaign
    Workstream 4)."""
    cfg = load_config()
    assert cfg["tracking"]["backfill"]["enabled"] is False


def test_backfill_shot_extends_a_late_track_when_enabled(tiny_shot_dir):
    """Exercises PlayerTrackingStage._backfill_shot end to end against
    the fixture's real (tiny) clip via VideoFrameSource, proving the
    in-stage wiring (not just the underlying algorithm, covered by
    tests/test_track_backfill.py) works."""
    cfg = _test_cfg()
    stage = PlayerTrackingStage(
        config=cfg,
        output_dir=tiny_shot_dir,
        player_detector=FakePlayerDetector([[_one_player_det()]]),
        team_classifier=FakeTeamClassifier("A"),
    )
    late_track = Track(
        track_id="T005", class_name="player", team="A",
        player_id="P005", player_name="Late",
        frames=[TrackFrame(frame=5, bbox=[50.0, 30.0, 150.0, 200.0], confidence=0.9, pitch_position=None)],
    )
    result = TracksResult(shot_id="shot_001", tracks=[late_track])
    backfill_cfg = BackfillConfig(min_late_start_frames=1, patience=0)
    # _one_player_det() always returns the SAME bbox, so every backward
    # step is an unambiguous IoU=1.0 match all the way to frame 0.
    always_match = FakePlayerDetector([[_one_player_det()]])

    new_result = stage._backfill_shot(
        "shots/shot_001.mp4", result, always_match, backfill_cfg
    )

    assert new_result.tracks[0].track_id == "T005"
    assert new_result.tracks[0].player_id == "P005"
    assert new_result.tracks[0].frames[0].frame == 0
    assert new_result.tracks[0].frames[0].source == "backfill"
    assert new_result.tracks[0].frames[-1].frame == 5
    assert new_result.tracks[0].frames[-1].source == "detector"


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
