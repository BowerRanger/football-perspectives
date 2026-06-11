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


def test_spike_cuts_finds_hard_cut(tmp_path: Path):
    from src.utils.shot_split import diff_spike_cuts

    reel = tmp_path / "reel.mp4"
    build_reel(reel, [("green", 3.0), ("crowd", 2.0)])
    cuts = diff_spike_cuts(reel)
    assert any(abs(c - int(3.0 * FPS)) <= 2 for c in cuts)


def test_spike_rescue_recovers_cut_the_detector_missed(tmp_path: Path):
    # threshold=255 blinds the content detector; only spike rescue can
    # split the reel.
    reel = tmp_path / "reel.mp4"
    info = build_reel(reel, [("green", 3.0), ("crowd", 2.0), ("green", 3.0)])
    blind = detect_spans(reel, detector="content", threshold=255.0,
                         min_scene_len_frames=8, spike_rescue=False)
    assert len(blind) == 1
    rescued = detect_spans(reel, detector="content", threshold=255.0,
                           min_scene_len_frames=8, spike_rescue=True)
    assert len(rescued) == 3
    assert rescued[-1].end_frame == info["total_frames"] - 1
    # spans stay contiguous
    for a, b in zip(rescued, rescued[1:]):
        assert b.start_frame == a.end_frame + 1


def test_spike_rescue_does_not_duplicate_detector_cuts(tmp_path: Path):
    reel = tmp_path / "reel.mp4"
    build_reel(reel, [("green", 3.0), ("crowd", 2.0), ("green", 3.0)])
    normal = detect_spans(reel, detector="content", threshold=27.0,
                          min_scene_len_frames=8, spike_rescue=False)
    rescued = detect_spans(reel, detector="content", threshold=27.0,
                           min_scene_len_frames=8, spike_rescue=True)
    # detector already finds both cuts; rescue must not add near-dupes
    assert [s.start_frame for s in rescued] == [s.start_frame for s in normal]


def test_dissolve_cuts_finds_crossfade(tmp_path: Path):
    """Cross-dissolves have high frame-diff but ~zero optical flow —
    invisible to both the content detector (per-frame delta too small)
    and spike rescue (no isolated outlier). The dissolve pass must
    split them."""
    from src.utils.shot_split import dissolve_cuts

    reel = tmp_path / "reel.mp4"
    # green->black maximises blend contrast so the per-frame change
    # during the 0.32 s fade clears the uniformity floor with margin,
    # like real broadcast dissolves do (reel fades measure 12-27).
    build_reel(reel, [("green", 2.0), ("xfade:green:black", 0.32),
                      ("black", 2.0)])
    cuts = dissolve_cuts(reel)
    fade_mid = int(2.16 * FPS)
    assert any(abs(c - fade_mid) <= 6 for c in cuts), cuts


def test_dissolve_cuts_ignores_pans(tmp_path: Path):
    """Fast pans also produce sustained diff — the flow gate must
    reject them."""
    from src.utils.shot_split import dissolve_cuts

    reel = tmp_path / "reel.mp4"
    build_reel(reel, [("pan", 4.0)])
    assert dissolve_cuts(reel) == []


def test_detect_spans_splits_at_dissolve(tmp_path: Path):
    reel = tmp_path / "reel.mp4"
    build_reel(reel, [("green", 2.0), ("xfade:green:black", 0.32),
                      ("black", 2.0)])
    spans = detect_spans(reel, detector="content", threshold=255.0,
                         min_scene_len_frames=8, spike_rescue=False,
                         dissolve_split=True)
    assert len(spans) == 2
    for a, b in zip(spans, spans[1:]):
        assert b.start_frame == a.end_frame + 1
