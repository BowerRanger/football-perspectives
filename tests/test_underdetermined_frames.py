from src.utils.line_camera_refine import drop_underdetermined_frames


def test_drops_frames_below_min_lines():
    pf = {0: [1, 2, 3, 4], 1: [1, 2], 2: [1, 2, 3], 3: []}
    out = drop_underdetermined_frames(pf, min_lines=3)
    assert set(out) == {0, 2}
    assert out[0] == [1, 2, 3, 4]  # surviving entries unchanged


def test_min_one_keeps_all_nonempty():
    pf = {0: [1], 1: [], 2: [1, 2]}
    assert set(drop_underdetermined_frames(pf, min_lines=1)) == {0, 2}


def test_min_zero_or_one_is_a_noop_on_nonempty():
    pf = {5: [1, 2, 3, 4, 5]}
    assert drop_underdetermined_frames(pf, min_lines=1) == pf
