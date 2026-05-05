"""Tests for the pitch-line catalogue, focusing on the advertising-board
entries which give thin anchors a z>0 depth cue when goal frames aren't
clearly visible.
"""

from __future__ import annotations

import pytest

from src.utils.pitch_lines_catalogue import (
    _AD_BOARD_HEIGHT,
    _AD_FAR_Y,
    _AD_LEFT_X,
    _AD_NEAR_Y,
    _AD_RIGHT_X,
    LINE_CATALOGUE,
    get_line,
)


_AD_BOARD_NAMES = (
    "near_advertising_board_base",
    "near_advertising_board_top",
    "far_advertising_board_base",
    "far_advertising_board_top",
    "behind_left_goal_ad_board_base",
    "behind_left_goal_ad_board_top",
    "behind_right_goal_ad_board_base",
    "behind_right_goal_ad_board_top",
)


@pytest.mark.unit
def test_all_ad_board_entries_present():
    for name in _AD_BOARD_NAMES:
        assert name in LINE_CATALOGUE, f"missing catalogue entry: {name}"


@pytest.mark.unit
def test_ad_board_base_entries_are_at_ground():
    for name in _AD_BOARD_NAMES:
        if not name.endswith("_base"):
            continue
        a, b = LINE_CATALOGUE[name]
        assert a[2] == 0.0
        assert b[2] == 0.0


@pytest.mark.unit
def test_ad_board_top_entries_are_at_nominal_height():
    for name in _AD_BOARD_NAMES:
        if not name.endswith("_top"):
            continue
        a, b = LINE_CATALOGUE[name]
        assert a[2] == _AD_BOARD_HEIGHT
        assert b[2] == _AD_BOARD_HEIGHT


@pytest.mark.unit
def test_touchline_parallel_boards_share_y_along_their_length():
    for prefix in ("near_advertising_board", "far_advertising_board"):
        for suffix in ("_base", "_top"):
            a, b = LINE_CATALOGUE[prefix + suffix]
            assert a[1] == b[1], f"{prefix+suffix} should be parallel to x-axis"


@pytest.mark.unit
def test_goalline_parallel_boards_share_x_along_their_length():
    for prefix in ("behind_left_goal_ad_board", "behind_right_goal_ad_board"):
        for suffix in ("_base", "_top"):
            a, b = LINE_CATALOGUE[prefix + suffix]
            assert a[0] == b[0], f"{prefix+suffix} should be parallel to y-axis"


@pytest.mark.unit
def test_near_board_base_endpoints_match_constants():
    seg = get_line("near_advertising_board_base")
    assert seg == ((0.0, _AD_NEAR_Y, 0.0), (105.0, _AD_NEAR_Y, 0.0))


@pytest.mark.unit
def test_far_board_top_endpoints_match_constants():
    seg = get_line("far_advertising_board_top")
    assert seg == ((0.0, _AD_FAR_Y, _AD_BOARD_HEIGHT), (105.0, _AD_FAR_Y, _AD_BOARD_HEIGHT))


@pytest.mark.unit
def test_behind_left_goal_board_base_endpoints_match_constants():
    seg = get_line("behind_left_goal_ad_board_base")
    # Spans the 18-yard-box y extent (13.84 to 54.16) at x = _AD_LEFT_X.
    assert seg[0] == (_AD_LEFT_X, 13.84, 0.0)
    assert seg[1] == (_AD_LEFT_X, 54.16, 0.0)


@pytest.mark.unit
def test_behind_right_goal_board_top_endpoints_match_constants():
    seg = get_line("behind_right_goal_ad_board_top")
    assert seg[0] == (_AD_RIGHT_X, 13.84, _AD_BOARD_HEIGHT)
    assert seg[1] == (_AD_RIGHT_X, 54.16, _AD_BOARD_HEIGHT)


@pytest.mark.unit
def test_catalogue_size_after_ad_boards():
    # Pre-existing catalogue had 23 entries; +8 ad-board entries = 31.
    assert len(LINE_CATALOGUE) == 31
    # And no overlap between the legacy and ad-board names.
    legacy = {n for n in LINE_CATALOGUE if n not in _AD_BOARD_NAMES}
    assert len(legacy) == 23


@pytest.mark.unit
def test_get_line_raises_on_unknown_name():
    with pytest.raises(KeyError):
        get_line("bogus_advertising_board")
