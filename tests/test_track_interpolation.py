"""Unit tests for src.utils.track_interpolation.

The interpolation utility is a post-processing pass run from the
dashboard's "Interp Missing Frames" button. It linearly fills the
bounding box between the last known frame and the next known frame
for gaps shorter than ``max_gap``, so GVHMR and the ball-anchor
lookup get continuous tracks instead of having to span detector
dropouts with their own smoothing fallbacks.
"""

from __future__ import annotations

import pytest

from src.schemas.tracks import Track, TrackFrame
from src.utils.track_interpolation import interpolate_track_gaps


def _frame(idx: int, bbox: list[float], conf: float = 0.9) -> TrackFrame:
    return TrackFrame(
        frame=idx, bbox=bbox, confidence=conf, pitch_position=None,
    )


def _make_track(frames: list[TrackFrame]) -> Track:
    return Track(
        track_id="T001",
        class_name="player",
        team="A",
        player_id="P001",
        player_name="Test",
        frames=frames,
    )


def test_no_gap_returns_unchanged() -> None:
    track = _make_track([
        _frame(0, [0, 0, 10, 10]),
        _frame(1, [1, 1, 11, 11]),
        _frame(2, [2, 2, 12, 12]),
    ])
    out, filled = interpolate_track_gaps(track, max_gap=4)
    assert filled == 0
    assert [f.frame for f in out.frames] == [0, 1, 2]
    assert all(not f.interpolated for f in out.frames)


def test_fills_single_frame_gap() -> None:
    # Gap of 1 frame between 0 and 2 — interpolate frame 1.
    track = _make_track([
        _frame(0, [0, 0, 10, 10], conf=1.0),
        _frame(2, [4, 4, 14, 14], conf=0.5),
    ])
    out, filled = interpolate_track_gaps(track, max_gap=4)
    assert filled == 1
    assert [f.frame for f in out.frames] == [0, 1, 2]
    mid = out.frames[1]
    assert mid.bbox == [2.0, 2.0, 12.0, 12.0]
    assert mid.confidence == pytest.approx(0.75)
    assert mid.interpolated is True
    # Originals are untouched.
    assert out.frames[0].interpolated is False
    assert out.frames[2].interpolated is False


def test_fills_multi_frame_gap_linearly() -> None:
    # Gap of 3 frames between 10 and 14 — interpolate 11, 12, 13.
    track = _make_track([
        _frame(10, [0.0, 0.0, 0.0, 0.0], conf=1.0),
        _frame(14, [40.0, 40.0, 40.0, 40.0], conf=0.0),
    ])
    out, filled = interpolate_track_gaps(track, max_gap=4)
    assert filled == 3
    assert [f.frame for f in out.frames] == [10, 11, 12, 13, 14]
    f11, f12, f13 = out.frames[1], out.frames[2], out.frames[3]
    assert f11.bbox == [10.0, 10.0, 10.0, 10.0]
    assert f12.bbox == [20.0, 20.0, 20.0, 20.0]
    assert f13.bbox == [30.0, 30.0, 30.0, 30.0]
    assert f11.confidence == pytest.approx(0.75)
    assert f12.confidence == pytest.approx(0.5)
    assert f13.confidence == pytest.approx(0.25)
    for f in (f11, f12, f13):
        assert f.interpolated is True


def test_skips_gap_larger_than_max() -> None:
    # Gap of 10 frames — exceeds max_gap=8, leave it alone.
    track = _make_track([
        _frame(0, [0, 0, 10, 10]),
        _frame(11, [50, 50, 60, 60]),
    ])
    out, filled = interpolate_track_gaps(track, max_gap=8)
    assert filled == 0
    assert [f.frame for f in out.frames] == [0, 11]


def test_max_gap_inclusive_boundary() -> None:
    # Gap of exactly max_gap=4 frames — should be filled.
    track = _make_track([
        _frame(0, [0, 0, 10, 10]),
        _frame(5, [50, 50, 60, 60]),
    ])
    out, filled = interpolate_track_gaps(track, max_gap=4)
    assert filled == 4
    assert [f.frame for f in out.frames] == [0, 1, 2, 3, 4, 5]


def test_max_gap_plus_one_excluded() -> None:
    # max_gap=4 means a 5-frame gap is rejected.
    track = _make_track([
        _frame(0, [0, 0, 10, 10]),
        _frame(6, [60, 60, 70, 70]),
    ])
    out, filled = interpolate_track_gaps(track, max_gap=4)
    assert filled == 0


def test_handles_multiple_gaps_in_one_track() -> None:
    # Two short gaps (1 frame each), one long gap (>max).
    track = _make_track([
        _frame(0, [0, 0, 10, 10]),
        _frame(2, [2, 2, 12, 12]),     # gap 0→2: fill 1
        _frame(4, [4, 4, 14, 14]),     # gap 2→4: fill 3
        _frame(20, [100, 100, 110, 110]),  # gap 4→20: too wide
    ])
    out, filled = interpolate_track_gaps(track, max_gap=4)
    assert filled == 2
    assert [f.frame for f in out.frames] == [0, 1, 2, 3, 4, 20]


def test_empty_track_is_noop() -> None:
    track = _make_track([])
    out, filled = interpolate_track_gaps(track, max_gap=4)
    assert filled == 0
    assert out.frames == []


def test_single_frame_track_is_noop() -> None:
    track = _make_track([_frame(7, [0, 0, 1, 1])])
    out, filled = interpolate_track_gaps(track, max_gap=4)
    assert filled == 0
    assert [f.frame for f in out.frames] == [7]


def test_max_gap_zero_disables_interpolation() -> None:
    track = _make_track([
        _frame(0, [0, 0, 10, 10]),
        _frame(2, [2, 2, 12, 12]),
    ])
    out, filled = interpolate_track_gaps(track, max_gap=0)
    assert filled == 0
    assert [f.frame for f in out.frames] == [0, 2]


def test_preserves_track_metadata() -> None:
    track = _make_track([
        _frame(0, [0, 0, 10, 10]),
        _frame(2, [2, 2, 12, 12]),
    ])
    out, _ = interpolate_track_gaps(track, max_gap=4)
    assert out.track_id == "T001"
    assert out.player_id == "P001"
    assert out.player_name == "Test"
    assert out.team == "A"
    assert out.class_name == "player"


def test_pitch_position_is_none_on_interpolated_frame() -> None:
    # We don't fabricate pitch_position; the camera stage owns that.
    track = _make_track([
        _frame(0, [0, 0, 10, 10]),
        _frame(2, [2, 2, 12, 12]),
    ])
    # Stamp a pitch_position on the endpoints — the interpolated frame
    # should still be None because the camera projection is not run here.
    track.frames[0].pitch_position = [1.0, 2.0]
    track.frames[1].pitch_position = [3.0, 4.0]
    out, _ = interpolate_track_gaps(track, max_gap=4)
    assert out.frames[1].pitch_position is None


def test_does_not_reorder_when_frames_already_sorted() -> None:
    track = _make_track([
        _frame(5, [0, 0, 10, 10]),
        _frame(8, [3, 3, 13, 13]),
        _frame(9, [4, 4, 14, 14]),
    ])
    out, filled = interpolate_track_gaps(track, max_gap=4)
    assert filled == 2  # frames 6, 7
    assert [f.frame for f in out.frames] == [5, 6, 7, 8, 9]
