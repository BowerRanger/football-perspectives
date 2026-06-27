import numpy as np
import pytest

from src.utils.ball_kinematic_touch import KinematicTouchCfg, interpolate_ball_uvs


def test_cfg_defaults_are_high_recall():
    cfg = KinematicTouchCfg()
    assert cfg.enabled is True
    assert cfg.contact_gap_m == pytest.approx(0.30)
    assert cfg.min_emit_score == pytest.approx(0.25)
    assert cfg.max_ball_gap_frames == 6


def test_interpolate_fills_short_gap_and_flags_it():
    uvs = {0: np.array([0.0, 0.0]), 3: np.array([3.0, 6.0])}
    filled, interp = interpolate_ball_uvs(uvs, max_gap_frames=6)
    assert set(filled) == {0, 1, 2, 3}
    assert filled[1] == pytest.approx(np.array([1.0, 2.0]))
    assert filled[2] == pytest.approx(np.array([2.0, 4.0]))
    assert interp == frozenset({1, 2})


def test_interpolate_leaves_long_gap_empty():
    uvs = {0: np.array([0.0, 0.0]), 10: np.array([10.0, 0.0])}
    filled, interp = interpolate_ball_uvs(uvs, max_gap_frames=6)
    assert set(filled) == {0, 10}
    assert interp == frozenset()
