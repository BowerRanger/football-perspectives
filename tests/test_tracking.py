import numpy as np
from src.utils.player_detector import Detection, FakePlayerDetector, PlayerDetector
from src.utils.team_classifier import FakeTeamClassifier, TeamClassifier


def test_fake_player_detector_cycles():
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    dets = [
        [Detection(bbox=(10.0, 20.0, 80.0, 200.0), confidence=0.9, class_name="player")],
        [],
    ]
    detector = FakePlayerDetector(dets)
    assert len(detector.detect(frame)) == 1
    assert len(detector.detect(frame)) == 0
    assert len(detector.detect(frame)) == 1  # cycles


def test_detection_is_player_detector():
    assert issubclass(FakePlayerDetector, PlayerDetector)


def test_fake_team_classifier_returns_fixed_label():
    crops = [np.zeros((60, 40, 3), dtype=np.uint8) for _ in range(3)]
    clf = FakeTeamClassifier("B")
    labels = clf.classify(crops)
    assert labels == ["B", "B", "B"]


def test_fake_team_classifier_empty_input():
    clf = FakeTeamClassifier("A")
    assert clf.classify([]) == []


def test_team_classifier_is_abstract():
    assert issubclass(FakeTeamClassifier, TeamClassifier)
