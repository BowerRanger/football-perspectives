import numpy as np
import pytest
from src.utils.player_detector import Detection, FakePlayerDetector, PlayerDetector


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
