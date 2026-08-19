"""Track outlier rejection (cleanup item 1): drop out-of-image positions and
isolated teleport spikes before direction-change segmentation."""

from __future__ import annotations

from src.utils.ball_track_clean import clean_pixel_track


def test_drops_out_of_image_positions():
    uvs = {
        0: (100.0, 100.0),
        1: (110.0, 100.0),
        2: (-5.0, 100.0),      # negative u
        3: (120.0, 800.0),     # v beyond 720
        4: (130.0, 100.0),
    }
    out = clean_pixel_track(uvs, image_size=(1280, 720))
    assert set(out) == {0, 1, 4}


def test_drops_isolated_teleport_spike():
    # frame 2 jumps ~600px away then returns — a spike, not real motion
    uvs = {
        0: (100.0, 100.0),
        1: (106.0, 100.0),
        2: (700.0, 100.0),     # teleport
        3: (118.0, 100.0),
        4: (124.0, 100.0),
    }
    out = clean_pixel_track(uvs, image_size=(1280, 720), max_jump_px=200.0)
    assert 2 not in out
    assert set(out) == {0, 1, 3, 4}


def test_keeps_smooth_track():
    uvs = {f: (100.0 + 6.0 * f, 100.0) for f in range(10)}
    assert clean_pixel_track(uvs, image_size=(1280, 720)) == uvs


def test_jump_budget_scales_with_frame_gap():
    # a 300px move across a 3-frame gap (100px/frame) is plausible, kept
    uvs = {0: (100.0, 100.0), 3: (400.0, 100.0), 6: (700.0, 100.0)}
    out = clean_pixel_track(uvs, image_size=(1280, 720), max_jump_px=150.0)
    assert set(out) == {0, 3, 6}


class TestClickContradictedVeto:
    """Sub-20cm W5s: detections the operator's bracketing clicks prove
    false (interpolated click path >veto_px away) are removed from the
    observation stream BEFORE the IMM/fits consume them."""

    def test_contradicted_detection_removed_click_consistent_kept(self):
        from src.utils.ball_track_clean import veto_click_contradicted

        clicks = {10: (100.0, 100.0), 14: (180.0, 100.0)}
        uvs = {
            11: (120.0, 100.0),   # on the interpolated path → kept
            12: (400.0, 300.0),   # contradicted → vetoed
            13: (160.0, 100.0),   # on path → kept
            20: (500.0, 500.0),   # outside any bracket → kept
        }
        out = veto_click_contradicted(uvs, clicks, max_px=60.0,
                                      max_gap_frames=6)
        assert set(out) == {11, 13, 20}

    def test_wide_bracket_not_interpolated(self):
        from src.utils.ball_track_clean import veto_click_contradicted

        clicks = {10: (100.0, 100.0), 40: (900.0, 100.0)}  # 30-frame gap
        uvs = {25: (300.0, 400.0)}
        out = veto_click_contradicted(uvs, clicks, max_px=60.0,
                                      max_gap_frames=6)
        assert set(out) == {25}   # no close bracket → operator silent
