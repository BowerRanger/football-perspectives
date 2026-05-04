import pytest

from src.utils.pitch_landmarks import LANDMARK_CATALOGUE, get_landmark, has_non_coplanar


@pytest.mark.unit
def test_known_landmarks_present():
    assert get_landmark("near_left_corner").world_xyz == (0.0, 0.0, 0.0)
    assert get_landmark("near_right_corner").world_xyz == (105.0, 0.0, 0.0)
    assert get_landmark("far_left_corner").world_xyz == (0.0, 68.0, 0.0)
    assert get_landmark("halfway_near").world_xyz == (52.5, 0.0, 0.0)


@pytest.mark.unit
def test_non_coplanar_landmarks_exist():
    """K-recovery requires non-coplanar landmarks."""
    crossbar = get_landmark("left_goal_crossbar_left")
    corner_flag_top = get_landmark("near_left_corner_flag_top")
    assert crossbar.world_xyz[2] > 0
    assert corner_flag_top.world_xyz[2] > 0


@pytest.mark.unit
def test_unknown_landmark_raises():
    with pytest.raises(KeyError):
        get_landmark("bogus_landmark")


@pytest.mark.unit
def test_has_non_coplanar_returns_true_with_crossbar():
    landmarks = [
        get_landmark("near_left_corner"),
        get_landmark("near_right_corner"),
        get_landmark("far_left_corner"),
        get_landmark("left_goal_crossbar_left"),
    ]
    assert has_non_coplanar(landmarks)


@pytest.mark.unit
def test_has_non_coplanar_returns_false_for_pitch_only():
    landmarks = [
        get_landmark("near_left_corner"),
        get_landmark("near_right_corner"),
        get_landmark("far_left_corner"),
        get_landmark("halfway_near"),
    ]
    assert not has_non_coplanar(landmarks)
