import numpy as np
import pytest
from src.utils.ball_detector import BallDetector, FakeBallDetector


def test_fake_ball_detector_returns_position():
    frames = [np.zeros((240, 320, 3), dtype=np.uint8) for _ in range(5)]
    positions = [(50.0, 60.0), (55.0, 65.0), None, (60.0, 70.0), (65.0, 75.0)]
    detector = FakeBallDetector(positions)
    results = [detector.detect(f) for f in frames]
    assert results[0] == pytest.approx((50.0, 60.0))
    assert results[2] is None


def test_ball_detector_is_abstract():
    import inspect
    assert inspect.isabstract(BallDetector)
