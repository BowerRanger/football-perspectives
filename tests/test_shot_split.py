"""PySceneDetect wrapper: span detection + hygiene."""
from pathlib import Path

from src.utils.shot_split import ShotSpan, detect_spans, merge_short_spans
from tests.fixtures.synthetic_reel import FPS, build_reel


def test_detect_spans_finds_cuts(tmp_path: Path):
    reel = tmp_path / "reel.mp4"
    build_reel(reel, [("green", 3.0), ("crowd", 2.0), ("green", 3.0)])
    spans = detect_spans(reel, detector="content", threshold=27.0,
                         min_scene_len_frames=8)
    assert len(spans) == 3
    assert spans[0].start_frame == 0
    assert abs(spans[1].start_frame - int(3.0 * FPS)) <= 2


def test_min_duration_filter(tmp_path: Path):
    reel = tmp_path / "reel.mp4"
    build_reel(reel, [("green", 3.0), ("black", 0.4), ("crowd", 3.0)])
    spans = detect_spans(reel, detector="content", threshold=27.0,
                         min_scene_len_frames=8, min_shot_duration_s=1.0)
    assert all(s.end_s - s.start_s >= 1.0 for s in spans)


def test_no_cuts_returns_whole_video_span(tmp_path: Path):
    reel = tmp_path / "reel.mp4"
    info = build_reel(reel, [("green", 3.0)])
    spans = detect_spans(reel, detector="content", threshold=27.0,
                         min_scene_len_frames=8)
    assert len(spans) == 1
    assert spans[0].start_frame == 0
    assert spans[0].end_frame == info["total_frames"] - 1


def test_merge_short_spans_glues_false_cuts():
    a = ShotSpan(0, 10, 0.0, 0.44)
    b = ShotSpan(11, 80, 0.44, 3.24)
    merged = merge_short_spans([a, b], max_short_duration_s=1.2,
                               max_gap_s=0.08)
    assert len(merged) == 1
    assert merged[0].start_frame == 0 and merged[0].end_frame == 80


def test_merge_short_spans_keeps_distinct_long_spans():
    a = ShotSpan(0, 80, 0.0, 3.24)
    b = ShotSpan(81, 160, 3.24, 6.44)
    merged = merge_short_spans([a, b], max_short_duration_s=1.2,
                               max_gap_s=0.08)
    assert len(merged) == 2
