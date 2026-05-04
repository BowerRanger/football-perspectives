import pytest
from src.utils.camera_confidence import FrameSignals, confidence_from_signals


@pytest.mark.unit
def test_confidence_perfect_signals_returns_1():
    s = FrameSignals(inlier_ratio=1.0, fwd_bwd_disagreement_deg=0.0, pitch_line_residual_px=0.0)
    assert confidence_from_signals(s) == 1.0


@pytest.mark.unit
def test_confidence_low_inlier_dominates():
    s = FrameSignals(inlier_ratio=0.2, fwd_bwd_disagreement_deg=0.0, pitch_line_residual_px=0.0)
    assert confidence_from_signals(s) < 0.3


@pytest.mark.unit
def test_confidence_no_pitch_lines_does_not_penalise():
    s = FrameSignals(inlier_ratio=1.0, fwd_bwd_disagreement_deg=0.0, pitch_line_residual_px=None)
    assert confidence_from_signals(s) == 1.0
