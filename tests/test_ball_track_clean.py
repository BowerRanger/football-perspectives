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
