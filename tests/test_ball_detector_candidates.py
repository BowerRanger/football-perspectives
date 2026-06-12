"""detect_candidates / reset on the BallDetector interface."""

from __future__ import annotations

import numpy as np
import pytest

from src.utils.ball_detector import FakeBallDetector

_FRAME = np.zeros((720, 1280, 3), dtype=np.uint8)


@pytest.mark.unit
def test_default_adapter_wraps_detect():
    det = FakeBallDetector([(100.0, 200.0, 0.8), None])
    assert det.detect_candidates(_FRAME, min_score=0.5, top_k=5) == [
        (100.0, 200.0, 0.8)
    ]
    assert det.detect_candidates(_FRAME, min_score=0.5, top_k=5) == []


@pytest.mark.unit
def test_default_adapter_applies_min_score():
    det = FakeBallDetector([(100.0, 200.0, 0.2)])
    assert det.detect_candidates(_FRAME, min_score=0.5, top_k=5) == []


@pytest.mark.unit
def test_fake_scripted_candidates_filter_and_truncate():
    det = FakeBallDetector(
        [None],
        candidates=[[(10.0, 10.0, 0.9), (20.0, 20.0, 0.4), (30.0, 30.0, 0.7)]],
    )
    out = det.detect_candidates(_FRAME, min_score=0.5, top_k=2)
    assert out == [(10.0, 10.0, 0.9), (20.0, 20.0, 0.7)] or out == [
        (10.0, 10.0, 0.9), (30.0, 30.0, 0.7)
    ]
    assert len(out) == 2


@pytest.mark.unit
def test_reset_is_counted_on_fake_and_noop_on_base():
    det = FakeBallDetector([None])
    det.reset()
    det.reset()
    assert det.reset_count == 2
