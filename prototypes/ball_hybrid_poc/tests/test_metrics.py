"""Hand-built tiny tracks with known errors -> known metrics.

No dependency on A1 (truth_builder/synth_detector), A2 (run_current) or B
(hybrid) — pure unit tests of ``metrics.py`` against literal
``Track``/``TruthTrack`` fixtures.
"""

from __future__ import annotations

import math

import pytest

from .. import metrics
from ..types import Track, TrackFrame, TruthEvent, TruthFrame, TruthTrack


def _truth(frames, events=()):
    return TruthTrack(clip_id="t", scenario="s", fps=25.0,
                       frames=tuple(frames), events=tuple(events))


def _track(frames, method="m"):
    return Track(clip_id="t", method=method, frames=tuple(frames))


# ---------------------------------------------------------------------------
# per_frame_error / error_stats: null frames count as FAILS
# ---------------------------------------------------------------------------

def test_per_frame_error_known_offsets_and_null_frame():
    truth = _truth([
        TruthFrame(0, (0.0, 0.0, 0.11), "ground"),
        TruthFrame(1, (1.0, 0.0, 0.11), "ground"),
        TruthFrame(2, (2.0, 0.0, 1.0), "air"),
        TruthFrame(3, (3.0, 0.0, 0.11), "contact"),
        TruthFrame(4, (4.0, 0.0, 0.11), "ground"),
    ], events=[TruthEvent(3, "touch", (3.0, 0.0, 0.11))])

    track = _track([
        TrackFrame(0, (0.1, 0.0, 0.11), "faithful"),
        TrackFrame(1, None, "faithful"),          # detector gap -> FAIL
        TrackFrame(2, (2.0, 0.0, 1.1), "simulated"),
        TrackFrame(3, (3.1, 0.0, 0.11), "anchor"),
        TrackFrame(4, (4.0, 0.0, 0.11), "faithful"),
    ])

    errs = metrics.per_frame_error(track, truth)
    assert errs[1] is None
    assert errs[0] == pytest.approx(0.1)
    assert errs[2] == pytest.approx(0.1)
    assert errs[3] == pytest.approx(0.1)
    assert errs[4] == pytest.approx(0.0, abs=1e-9)

    stats = metrics.error_stats(errs)
    assert stats["n"] == 5
    assert stats["n_valid"] == 4
    assert stats["coverage"] == pytest.approx(0.8)
    assert stats["p50"] == pytest.approx(0.1)
    assert stats["p95"] == pytest.approx(0.1)
    assert stats["max"] == pytest.approx(0.1)
    # pct_le_20cm denominator is ALL truth frames (5), not just valid ones:
    # the null frame counts as a fail even though every valid error is
    # well under the 20cm threshold.
    assert stats["pct_le_20cm"] == pytest.approx(0.8)

    by_state = metrics.split_by_state(track, truth)
    assert by_state["ground"]["n"] == 3          # frames 0, 1, 4
    assert by_state["ground"]["n_valid"] == 2    # frame 1 is the null
    assert by_state["ground"]["coverage"] == pytest.approx(2 / 3)
    assert by_state["ground"]["p50"] == pytest.approx(0.05)
    assert by_state["ground"]["max"] == pytest.approx(0.1)
    assert by_state["air"]["n"] == 1
    assert by_state["air"]["p50"] == pytest.approx(0.1)
    assert by_state["contact"]["n"] == 1
    assert by_state["contact"]["p50"] == pytest.approx(0.1)


def test_error_stats_all_null_is_zero_coverage_not_crash():
    stats = metrics.error_stats([None, None, None])
    assert stats["n"] == 3
    assert stats["n_valid"] == 0
    assert stats["coverage"] == 0.0
    assert stats["pct_le_20cm"] == 0.0
    assert stats["p50"] is None
    assert stats["max"] is None


# ---------------------------------------------------------------------------
# contact gap: exact vs. min-over-+/-1 separates timing error from position
# ---------------------------------------------------------------------------

def test_contact_gap_separates_timing_error_from_position_error():
    truth = _truth([
        TruthFrame(0, (0.0, 0.0, 0.11), "ground"),
        TruthFrame(1, (1.0, 0.0, 0.11), "ground"),
        TruthFrame(2, (2.0, 0.0, 0.11), "ground"),
    ], events=[TruthEvent(1, "bounce", (1.0, 0.0, 0.11))])

    # The method's bounce position is emitted one frame LATE: nothing at
    # frame 1 (the graded frame), but frame 2 lands exactly on the truth
    # bounce location.
    track = _track([
        TrackFrame(0, (0.0, 0.0, 0.11), "faithful"),
        TrackFrame(1, None, "faithful"),
        TrackFrame(2, (1.0, 0.0, 0.11), "simulated"),
    ])

    gap = metrics.contact_gap(track, truth)
    assert len(gap["events"]) == 1
    ev = gap["events"][0]
    assert ev["err_m"] is None                       # naive exact-frame grading: total miss
    assert ev["err_min_pm1_m"] == pytest.approx(0.0)  # +/-1 window: actually spot-on


def test_contact_gap_position_error_shows_even_with_pm1_window():
    truth = _truth([
        TruthFrame(0, (0.0, 0.0, 0.11), "ground"),
        TruthFrame(1, (1.0, 0.0, 0.11), "ground"),
        TruthFrame(2, (2.0, 0.0, 0.11), "ground"),
    ], events=[TruthEvent(1, "touch", (1.0, 0.0, 0.11))])

    # Genuinely wrong position at and around the event frame.
    track = _track([
        TrackFrame(0, (5.0, 5.0, 0.11), "faithful"),
        TrackFrame(1, (5.0, 5.0, 0.11), "faithful"),
        TrackFrame(2, (5.0, 5.0, 0.11), "faithful"),
    ])
    gap = metrics.contact_gap(track, truth)
    ev = gap["events"][0]
    dist = math.hypot(4.0, 5.0)
    assert ev["err_m"] == pytest.approx(dist)
    assert ev["err_min_pm1_m"] == pytest.approx(dist)


# ---------------------------------------------------------------------------
# ground float/sink
# ---------------------------------------------------------------------------

def test_ground_float_sink_counts_and_stats():
    truth = _truth([
        TruthFrame(0, (0.0, 0.0, 0.11), "ground"),
        TruthFrame(1, (1.0, 0.0, 0.11), "ground"),
        TruthFrame(2, (2.0, 0.0, 0.11), "ground"),
        TruthFrame(3, (3.0, 0.0, 0.11), "ground"),
        TruthFrame(4, (4.0, 0.0, 1.0), "air"),   # not scored (not ground)
    ])
    track = _track([
        TrackFrame(0, (0.0, 0.0, 0.11), "faithful"),   # perfect
        TrackFrame(1, (1.0, 0.0, 0.05), "faithful"),   # sink (< 0.09)
        TrackFrame(2, (2.0, 0.0, 0.30), "faithful"),   # float (> 0.25)
        TrackFrame(3, (3.0, 0.0, 0.20), "faithful"),   # neither
        TrackFrame(4, (4.0, 0.0, 1.0), "faithful"),
    ])
    gfs = metrics.ground_float_sink(track, truth)
    assert gfs["n_ground_frames"] == 4
    assert gfs["n_scored"] == 4
    assert gfs["mean_abs_dev_m"] == pytest.approx((0.0 + 0.06 + 0.19 + 0.09) / 4)
    assert gfs["n_sink"] == 1
    assert gfs["n_float"] == 1


# ---------------------------------------------------------------------------
# side camera sees a pure-depth error the broadcast camera can't
# ---------------------------------------------------------------------------

class _FakeBroadcastCtx:
    """Minimal ``ctx``-shaped stand-in: a fixed camera looking straight
    down the +x axis from very far away, so it's blind to x-only shifts
    (x is its depth axis) but sensitive to y/z. Mirrors
    ``ClipContext.per_frame_K``/``.project`` just enough for
    ``broadcast_px_error``."""

    def __init__(self, frames):
        self.per_frame_K = {f: True for f in frames}
        self._cam = metrics.SideCamera(
            eye=(-100000.0, 50.0, 0.11), target=(0.0, 50.0, 0.11),
            hfov_deg=1.0, image_size=(1920, 1080))

    def project(self, frame, xyz):
        uv = self._cam.project(xyz)
        assert uv is not None
        return uv


def test_side_camera_sees_depth_error_broadcast_cannot():
    truth_pts = [(0.0, 50.0, 0.11), (1.0, 50.0, 0.11),
                 (2.0, 50.0, 0.11), (3.0, 50.0, 0.11)]
    truth = _truth([TruthFrame(i, p, "ground") for i, p in enumerate(truth_pts)])

    # Method disagrees with truth ONLY along x -- x is depth for the
    # broadcast camera (which looks straight down +x) but lateral for the
    # side camera (perpendicular to the trajectory's principal x-direction).
    track = _track([
        TrackFrame(i, (p[0] + 5.0, p[1], p[2]), "faithful")
        for i, p in enumerate(truth_pts)
    ])

    ctx = _FakeBroadcastCtx(frames=range(len(truth_pts)))
    bpx = metrics.broadcast_px_error(ctx, track, truth)
    assert bpx["p50"] == pytest.approx(0.0, abs=1e-6)

    side_cam = metrics.build_side_camera(truth_pts, distance_m=30.0, height_m=6.0)
    spx = metrics.side_px_error(side_cam, track, truth)
    assert spx["p50"] is not None
    assert spx["p50"] > 5.0  # the broadcast camera reports ~0; side reports a real miss


def test_build_side_camera_pose_is_offset_from_centroid():
    pts = [(0.0, 0.0, 0.11), (10.0, 0.0, 0.11)]
    cam = metrics.build_side_camera(pts, distance_m=30.0, height_m=6.0)
    assert cam.eye[2] == pytest.approx(6.0)
    dist = math.hypot(cam.eye[0] - 5.0, cam.eye[1] - 0.0)
    assert dist == pytest.approx(30.0, rel=1e-6)
    assert cam.target == pytest.approx((5.0, 0.0, 0.11))


# ---------------------------------------------------------------------------
# naturalness: truth control + violations_minus_truth
# ---------------------------------------------------------------------------

def _lob_truth(n=20, fps=25.0):
    frames = []
    for i in range(n):
        t = i / fps
        z = max(0.11, 5.0 * t - 4.9 * t * t + 0.11)
        state = "ground" if z <= 0.11 + 1e-6 else "air"
        frames.append(TruthFrame(i, (float(i) * 0.5, 0.0, z), state))
    return _truth(frames, events=[TruthEvent(0, "touch", frames[0].xyz)])


def test_naturalness_matching_track_has_no_extra_violations():
    truth = _lob_truth()
    track = _track([TrackFrame(f.frame, f.xyz, "simulated") for f in truth.frames])
    nat = metrics.naturalness_summary(track, truth, fps=truth.fps)
    assert nat["violations_minus_truth"] == 0


def test_naturalness_erratic_track_flags_more_than_truth():
    truth = _lob_truth()
    frames = []
    for i, f in enumerate(truth.frames):
        # Inject a sharp lateral zig-zag mid-flight that no event explains.
        jitter_x = 3.0 if (f.state == "air" and i % 2 == 0) else 0.0
        frames.append(TrackFrame(f.frame, (f.xyz[0] + jitter_x, f.xyz[1], f.xyz[2]),
                                  "simulated"))
    track = _track(frames)
    nat = metrics.naturalness_summary(track, truth, fps=truth.fps)
    assert nat["violations_minus_truth"] > 0


# ---------------------------------------------------------------------------
# jitter
# ---------------------------------------------------------------------------

def test_jitter_p95_zero_for_smooth_linear_motion():
    truth = _truth([TruthFrame(i, (float(i), 0.0, 0.11), "ground") for i in range(6)])
    track = _track([TrackFrame(i, (float(i), 0.0, 0.11), "faithful") for i in range(6)])
    assert metrics.jitter_p95(track) == pytest.approx(0.0, abs=1e-9)
    assert metrics.jitter_p95_truth(truth) == pytest.approx(0.0, abs=1e-9)


def test_jitter_p95_nonzero_for_noisy_track():
    frames = []
    for i in range(8):
        wig = 0.5 if i % 2 == 0 else -0.5
        frames.append(TrackFrame(i, (float(i), wig, 0.11), "faithful"))
    track = _track(frames)
    assert metrics.jitter_p95(track) > 0.0


def test_jitter_skips_non_contiguous_windows():
    # A gap at frame 2 means no 4-frame window is unit-spaced; jitter is
    # undefined (None), not a spurious huge value from bridging the gap.
    track = _track([
        TrackFrame(0, (0.0, 0.0, 0.11), "faithful"),
        TrackFrame(1, (1.0, 0.0, 0.11), "faithful"),
        TrackFrame(3, (3.0, 0.0, 0.11), "faithful"),
        TrackFrame(4, (4.0, 0.0, 0.11), "faithful"),
    ])
    assert metrics.jitter_p95(track) is None


# ---------------------------------------------------------------------------
# compute_scenario_metrics: flat dict stays flat (viewer calls .toFixed on
# every value), detail is the nested breakdown
# ---------------------------------------------------------------------------

def test_compute_scenario_metrics_flat_dict_has_only_scalars():
    truth = _lob_truth()
    track = _track([TrackFrame(f.frame, f.xyz, "simulated") for f in truth.frames])
    side_cam = metrics.build_side_camera([f.xyz for f in truth.frames])

    class _Ctx:
        fps = truth.fps
        per_frame_K: dict = {}

        def project(self, frame, xyz):
            raise AssertionError("no camera frames registered")

    flat, detail = metrics.compute_scenario_metrics(_Ctx(), track, truth,
                                                      side_camera=side_cam)
    for key, value in flat.items():
        assert value is None or isinstance(value, (int, float)), (
            f"{key!r} is not a scalar: {value!r}")
    assert "by_state" in detail
    assert "naturalness" in detail
