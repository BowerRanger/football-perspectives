from collections import namedtuple

from src.utils.auto_anchor import snap_keypoints

_SP = namedtuple("SP", "xy snapped mode_used confidence")


def test_snap_replaces_high_confidence_and_keeps_low():
    pixels = {1: (100.0, 200.0), 2: (300.0, 400.0), 3: (500.0, 600.0)}

    def fake_snap(frame, xy):
        if xy == (100.0, 200.0):
            return _SP((101.5, 201.5), True, "line_intersection", 0.9)   # snapped, high conf
        if xy == (300.0, 400.0):
            return _SP((305.0, 405.0), True, "line_intersection", 0.3)   # snapped, LOW conf -> keep
        return _SP(xy, False, "line_intersection", 0.0)                  # not snapped -> keep

    out = snap_keypoints(pixels, frame_bgr=None, min_confidence=0.5, snap_fn=fake_snap)
    assert out[1] == (101.5, 201.5)   # refined
    assert out[2] == (300.0, 400.0)   # low conf -> original
    assert out[3] == (500.0, 600.0)   # not snapped -> original
