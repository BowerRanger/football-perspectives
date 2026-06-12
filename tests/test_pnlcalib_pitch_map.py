import pytest

from src.utils.pnlcalib_pitch_map import (
    keypoint_world_xyz_ours,
    NUM_KEYPOINTS,
)


def test_total_keypoint_count():
    assert NUM_KEYPOINTS == 73  # 57 main + 16 aux


def test_far_left_corner_keypoint_is_y_flipped():
    """PnLCalib kp1 -> far-left corner; in our frame (0, 68, 0)."""
    assert keypoint_world_xyz_ours(1) == pytest.approx((0.0, 68.0, 0.0))


def test_ground_keypoint_y_flips():
    """kp4 -> ours (0, 54.16, 0)."""
    assert keypoint_world_xyz_ours(4) == pytest.approx((0.0, 54.16, 0.0))


def test_goalpost_top_is_z_up_244():
    """kp16 -> ours (0, 30.34, +2.44)."""
    assert keypoint_world_xyz_ours(16) == pytest.approx((0.0, 30.34, 2.44))


def test_unknown_keypoint_returns_none():
    assert keypoint_world_xyz_ours(999) is None
