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


@pytest.mark.unit
def test_d_intersection_landmarks_present():
    """Penalty arc / 18-yard box intersections (the 'D')."""
    near = get_landmark("left_18yd_d_near")
    far = get_landmark("left_18yd_d_far")
    assert near.world_xyz[0] == 16.5
    assert far.world_xyz[0] == 16.5
    # 9.15 m arc radius from penalty spot at (11, 34); intersection at x=16.5
    # gives y = 34 ± sqrt(9.15² - 5.5²) ≈ 34 ± 7.3125.
    assert abs(near.world_xyz[1] - (34.0 - 7.3125)) < 1e-3
    assert abs(far.world_xyz[1] - (34.0 + 7.3125)) < 1e-3
    # Mirror on right half of pitch.
    assert get_landmark("right_18yd_d_near").world_xyz[0] == 88.5
    assert get_landmark("right_18yd_d_far").world_xyz[0] == 88.5


@pytest.mark.unit
def test_18yd_goalline_corners_present():
    """Where the 18-yard box meets the goal line (touchline x=0 / x=105)."""
    assert get_landmark("left_18yd_goal_near").world_xyz == (0.0, 13.84, 0.0)
    assert get_landmark("left_18yd_goal_far").world_xyz == (0.0, 54.16, 0.0)
    assert get_landmark("right_18yd_goal_near").world_xyz == (105.0, 13.84, 0.0)
    assert get_landmark("right_18yd_goal_far").world_xyz == (105.0, 54.16, 0.0)


@pytest.mark.unit
def test_6yd_goalline_corners_present():
    """Where the 6-yard box meets the goal line."""
    assert get_landmark("left_6yd_goal_near").world_xyz == (0.0, 24.84, 0.0)
    assert get_landmark("left_6yd_goal_far").world_xyz == (0.0, 43.16, 0.0)
    assert get_landmark("right_6yd_goal_near").world_xyz == (105.0, 24.84, 0.0)
    assert get_landmark("right_6yd_goal_far").world_xyz == (105.0, 43.16, 0.0)


@pytest.mark.unit
def test_goal_post_bases_present():
    """Goal post bases at z=0, paired with the existing crossbar endpoints at z=2.44."""
    assert get_landmark("left_goal_left_post_base").world_xyz == (0.0, 34.0 - 7.32 / 2, 0.0)
    assert get_landmark("left_goal_right_post_base").world_xyz == (0.0, 34.0 + 7.32 / 2, 0.0)
    assert get_landmark("right_goal_left_post_base").world_xyz == (105.0, 34.0 - 7.32 / 2, 0.0)
    assert get_landmark("right_goal_right_post_base").world_xyz == (105.0, 34.0 + 7.32 / 2, 0.0)


@pytest.mark.unit
def test_catalogue_size_matches_legacy():
    """Catalogue has at least the legacy 42 landmarks (regression guard)."""
    assert len(LANDMARK_CATALOGUE) >= 42
