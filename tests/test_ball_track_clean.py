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


class TestFlightContradictedVeto:
    """W9: detections inside a BALLISTIC span never shaped the arc (two-
    knot arcs come from anchors), so the arc can honestly judge them. A
    hard detection far off the arc that is locally STATIC (net corner,
    goal furniture) or a TELEPORTER (no continuous ball reaches it) is
    not a ball observation — relabel it so solve sidecars and grading
    stop treating it as truth."""

    @staticmethod
    def _mk(uvs):
        return {f: (float(u), float(v)) for f, (u, v) in uvs.items()}

    def test_static_far_cluster_vetoed(self):
        from src.utils.ball_track_clean import veto_flight_contradicted
        # Arc reprojection moves ~30 px/frame; obs 200-206 follow it, but
        # 207-210 sit still 100 px away (net corner lock-on).
        arc_px = {f: (400.0 + 30 * (f - 200), 500.0) for f in range(200, 212)}
        uvs = {f: arc_px[f] for f in range(200, 207)}
        for f in range(207, 211):
            uvs[f] = (447.0 + 0.5 * (f - 207), 605.0)
        vetoed = veto_flight_contradicted(
            self._mk(uvs), arc_px, mpp_by_frame={f: 0.02 for f in arc_px})
        assert set(vetoed) == {207, 208, 209, 210}

    def test_teleporter_vetoed_smooth_kept(self):
        from src.utils.ball_track_clean import veto_flight_contradicted
        arc_px = {f: (400.0 + 30 * (f - 0), 500.0) for f in range(0, 8)}
        uvs = {f: arc_px[f] for f in range(0, 8)}
        # Frame 4: oscillating goal-mouth lock 200 px off the arc, its
        # neighbours 180+ px away in pixel space (infeasible ball step).
        uvs[4] = (400.0 + 30 * 4 + 200.0, 700.0)
        vetoed = veto_flight_contradicted(
            self._mk(uvs), arc_px, mpp_by_frame={f: 0.02 for f in arc_px})
        assert vetoed == [4]

    def test_on_arc_and_moving_obs_kept(self):
        from src.utils.ball_track_clean import veto_flight_contradicted
        arc_px = {f: (400.0 + 30 * f, 500.0) for f in range(0, 8)}
        uvs = {f: (arc_px[f][0] + 5.0, arc_px[f][1] - 4.0)
               for f in range(0, 8)}   # small honest residual, moving
        vetoed = veto_flight_contradicted(
            self._mk(uvs), arc_px, mpp_by_frame={f: 0.02 for f in arc_px})
        assert vetoed == []

    def test_far_but_moving_smoothly_kept(self):
        from src.utils.ball_track_clean import veto_flight_contradicted
        # 100 px off the arc but moving coherently (a REAL ball the arc
        # missed — e.g. a deflection the solve got wrong): never veto;
        # the metric must keep charging the track for it.
        arc_px = {f: (400.0 + 30 * f, 500.0) for f in range(0, 8)}
        uvs = {f: (arc_px[f][0] + 100.0, arc_px[f][1] + 80.0)
               for f in range(0, 8)}
        vetoed = veto_flight_contradicted(
            self._mk(uvs), arc_px, mpp_by_frame={f: 0.02 for f in arc_px})
        assert vetoed == []
