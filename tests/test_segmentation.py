import pytest

pytest.skip(
    "awaiting later phase: imports a module deleted in Phase 0 of the "
    "broadcast-mono pipeline rewrite",
    allow_module_level=True,
)


import cv2
import numpy as np
import pytest
import subprocess
from pathlib import Path
from src.utils.ffmpeg import extract_clip, extract_thumbnail

@pytest.fixture(scope="module")
def tiny_video(tmp_path_factory) -> Path:
    """Synthetic 2-second video with a hard cut at 1 second."""
    path = tmp_path_factory.mktemp("fixtures") / "test.mp4"
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25, (320, 240)
    )
    for _ in range(25):  # blue frames
        writer.write(np.full((240, 320, 3), [200, 50, 50], dtype=np.uint8))
    for _ in range(25):  # green frames (new shot)
        writer.write(np.full((240, 320, 3), [50, 200, 50], dtype=np.uint8))
    writer.release()
    return path

def test_extract_clip_creates_file(tmp_path, tiny_video):
    out = tmp_path / "clip.mp4"
    extract_clip(tiny_video, out, start_s=0.0, end_s=1.0)
    assert out.exists()
    assert out.stat().st_size > 0

def test_extract_thumbnail_creates_jpg(tmp_path, tiny_video):
    out = tmp_path / "thumb.jpg"
    extract_thumbnail(tiny_video, out, time_s=0.5)
    assert out.exists()
    img = cv2.imread(str(out))
    assert img is not None
    assert img.shape[:2] == (240, 320)


from src.stages.segmentation import (
    _ShotSpan,
    _is_fade_transition_shot,
    _is_reaction_shot,
    _merge_adjacent_short_spans,
    detect_shots,
    ShotSegmentationStage,
)
from src.schemas.shots import ShotsManifest
from src.utils.ball_detector import FakeBallDetector


def test_detect_shots_finds_cut(tiny_video):
    shots = detect_shots(tiny_video, threshold=20.0)
    # Synthetic video has a hard cut between blue and green frames
    assert len(shots) == 2


def test_detect_shots_returns_correct_timings(tiny_video):
    shots = detect_shots(tiny_video, threshold=20.0)
    assert shots[0].start_frame == 0
    assert shots[1].start_frame > 0


def test_detect_shots_with_adaptive_detector(tiny_video):
    shots = detect_shots(
        tiny_video,
        detector="adaptive",
        adaptive_threshold=3.0,
        adaptive_min_content_val=10.0,
    )
    assert len(shots) >= 1


def test_detect_shots_rejects_invalid_detector(tiny_video):
    with pytest.raises(ValueError):
        detect_shots(tiny_video, detector="invalid")


def test_stage1_writes_manifest(tmp_path, tiny_video):
    cfg = {
        "shot_segmentation": {
            "threshold": 20.0,
            "min_shot_duration_s": 0.1,
            "merge_short_shots_max_duration_s": 0.0,
            "reaction_max_duration_s": 0.0,
            "exclude_fade_transitions": False,
            "require_ball_in_shot": False,
        }
    }
    stage = ShotSegmentationStage(
        config=cfg, output_dir=tmp_path, video_path=tiny_video
    )
    stage.run()
    manifest_path = tmp_path / "shots" / "shots_manifest.json"
    assert manifest_path.exists()
    manifest = ShotsManifest.load(manifest_path)
    assert len(manifest.shots) == 2
    assert manifest.fps == 25.0
    assert not list((tmp_path / "shots").glob("*_thumb.jpg"))
    assert not hasattr(manifest.shots[0], "thumbnail")


def test_stage1_is_complete_after_run(tmp_path, tiny_video):
    cfg = {
        "shot_segmentation": {
            "threshold": 20.0,
            "min_shot_duration_s": 0.1,
            "merge_short_shots_max_duration_s": 0.0,
            "reaction_max_duration_s": 0.0,
            "exclude_fade_transitions": False,
            "require_ball_in_shot": False,
        }
    }
    stage = ShotSegmentationStage(config=cfg, output_dir=tmp_path, video_path=tiny_video)
    assert not stage.is_complete()
    stage.run()
    assert stage.is_complete()


def test_stage1_filters_short_reaction_shots(tmp_path):
    path = tmp_path / "reaction.mp4"
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25, (320, 240)
    )
    # Gameplay-like frame region with high green pitch occupancy.
    for _ in range(30):
        writer.write(np.full((240, 320, 3), [40, 170, 40], dtype=np.uint8))
    # Short reaction/crowd-like insert with low green occupancy.
    for _ in range(10):
        writer.write(np.full((240, 320, 3), [20, 20, 200], dtype=np.uint8))
    for _ in range(30):
        writer.write(np.full((240, 320, 3), [40, 170, 40], dtype=np.uint8))
    writer.release()

    cfg = {
        "shot_segmentation": {
            "threshold": 20.0,
            "detector": "content",
            "min_shot_duration_s": 0.2,
            "reaction_max_duration_s": 0.6,
            "min_pitch_ratio": 0.1,
            "merge_short_shots_max_duration_s": 0.0,
            "require_ball_in_shot": False,
        }
    }
    stage = ShotSegmentationStage(config=cfg, output_dir=tmp_path, video_path=path)
    stage.run()
    manifest = ShotsManifest.load(tmp_path / "shots" / "shots_manifest.json")
    assert len(manifest.shots) == 2


def test_stage1_filters_longer_reaction_shot_when_pitch_stays_low(tmp_path):
    path = tmp_path / "long_reaction.mp4"
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25, (320, 240)
    )
    for _ in range(30):
        writer.write(np.full((240, 320, 3), [40, 170, 40], dtype=np.uint8))
    for _ in range(60):
        writer.write(np.full((240, 320, 3), [20, 20, 200], dtype=np.uint8))
    for _ in range(30):
        writer.write(np.full((240, 320, 3), [40, 170, 40], dtype=np.uint8))
    writer.release()

    cfg = {
        "shot_segmentation": {
            "threshold": 20.0,
            "detector": "content",
            "min_shot_duration_s": 0.2,
            "reaction_max_duration_s": 3.0,
            "min_pitch_ratio": 0.1,
            "reaction_max_peak_pitch_ratio": 0.2,
            "reaction_sample_points": [0.2, 0.5, 0.8],
            "merge_short_shots_max_duration_s": 0.0,
            "require_ball_in_shot": False,
        }
    }
    stage = ShotSegmentationStage(config=cfg, output_dir=tmp_path, video_path=path)
    stage.run()
    manifest = ShotsManifest.load(tmp_path / "shots" / "shots_manifest.json")
    assert len(manifest.shots) == 2


def test_reaction_shot_requires_consistently_low_pitch_coverage():
    assert _is_reaction_shot(
        duration=2.4,
        pitch_ratios=[0.02, 0.04, 0.05],
        reaction_max_duration_s=3.0,
        min_pitch_ratio=0.1,
        reaction_max_peak_pitch_ratio=0.2,
    )
    assert not _is_reaction_shot(
        duration=2.4,
        pitch_ratios=[0.02, 0.24, 0.05],
        reaction_max_duration_s=3.0,
        min_pitch_ratio=0.1,
        reaction_max_peak_pitch_ratio=0.2,
    )


def test_merge_adjacent_short_spans_merges_false_cuts():
    spans = [
        _ShotSpan(start_frame=0, end_frame=24, start_time=0.0, end_time=1.0),
        _ShotSpan(start_frame=25, end_frame=34, start_time=1.0, end_time=1.4),
        _ShotSpan(start_frame=35, end_frame=59, start_time=1.4, end_time=2.4),
    ]
    merged = _merge_adjacent_short_spans(
        spans,
        max_short_duration_s=1.2,
        max_gap_s=0.08,
    )
    assert len(merged) == 1
    assert merged[0].start_time == 0.0
    assert merged[0].end_time == 2.4


def test_stage1_keeps_sequential_ids_when_clip_extraction_fails(tmp_path, tiny_video, monkeypatch):
    cfg = {
        "shot_segmentation": {
            "threshold": 20.0,
            "min_shot_duration_s": 0.1,
            "merge_short_shots_max_duration_s": 0.0,
            "reaction_max_duration_s": 0.0,
            "exclude_fade_transitions": False,
            "require_ball_in_shot": False,
        }
    }

    call_count = 0

    def fake_extract_clip(src, out, start_s, end_s):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(b"clip")
            return
        raise subprocess.CalledProcessError(returncode=1, cmd="ffmpeg")

    monkeypatch.setattr("src.stages.segmentation.extract_clip", fake_extract_clip)

    stage = ShotSegmentationStage(config=cfg, output_dir=tmp_path, video_path=tiny_video)
    stage.run()

    manifest = ShotsManifest.load(tmp_path / "shots" / "shots_manifest.json")
    assert [shot.id for shot in manifest.shots] == ["shot_001"]


def test_stage1_discards_shots_without_ball_visibility(tmp_path, tiny_video):
    cfg = {
        "shot_segmentation": {
            "threshold": 20.0,
            "min_shot_duration_s": 0.1,
            "merge_short_shots_max_duration_s": 0.0,
            "reaction_max_duration_s": 0.0,
            "exclude_fade_transitions": False,
            "require_ball_in_shot": True,
        }
    }
    detector = FakeBallDetector(
        positions=[None] * 25 + [None, None, None, None, (100.0, 80.0)] + [None] * 20
    )
    stage = ShotSegmentationStage(
        config=cfg,
        output_dir=tmp_path,
        video_path=tiny_video,
        ball_detector=detector,
    )
    stage.run()

    manifest = ShotsManifest.load(tmp_path / "shots" / "shots_manifest.json")
    assert len(manifest.shots) == 1
    assert manifest.shots[0].start_frame >= 25


def test_stage1_keeps_shot_with_single_ball_frame(tmp_path, tiny_video):
    cfg = {
        "shot_segmentation": {
            "threshold": 20.0,
            "min_shot_duration_s": 0.1,
            "merge_short_shots_max_duration_s": 0.0,
            "reaction_max_duration_s": 0.0,
            "exclude_fade_transitions": False,
            "require_ball_in_shot": True,
        }
    }
    detector = FakeBallDetector(
        positions=[None] * 12 + [(15.0, 20.0)] + [None] * 12 + [None] * 12 + [(18.0, 22.0)] + [None] * 12
    )
    stage = ShotSegmentationStage(
        config=cfg,
        output_dir=tmp_path,
        video_path=tiny_video,
        ball_detector=detector,
    )
    stage.run()

    manifest = ShotsManifest.load(tmp_path / "shots" / "shots_manifest.json")
    assert len(manifest.shots) == 2


@pytest.fixture
def video_with_fade_transition(tmp_path) -> Path:
    path = tmp_path / "fade_transition.mp4"
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25, (320, 240)
    )
    for _ in range(20):
        writer.write(np.full((240, 320, 3), [40, 170, 40], dtype=np.uint8))
    for level in [140, 110, 80, 50, 20, 20, 50, 80, 110, 140]:
        writer.write(np.full((240, 320, 3), [level, level, level], dtype=np.uint8))
    for _ in range(20):
        writer.write(np.full((240, 320, 3), [40, 170, 40], dtype=np.uint8))
    writer.release()
    return path


def test_fade_transition_requires_dark_low_point_and_brightness_swing():
    assert _is_fade_transition_shot(
        duration=0.4,
        brightness_values=[0.55, 0.08, 0.57],
        max_duration_s=1.0,
        black_frame_threshold=0.18,
        min_brightness_range=0.25,
    )
    assert not _is_fade_transition_shot(
        duration=0.4,
        brightness_values=[0.55, 0.42, 0.57],
        max_duration_s=1.0,
        black_frame_threshold=0.18,
        min_brightness_range=0.25,
    )


def test_stage1_excludes_fade_transition_spans(tmp_path, video_with_fade_transition, monkeypatch):
    cfg = {
        "shot_segmentation": {
            "min_shot_duration_s": 0.1,
            "merge_short_shots_max_duration_s": 0.0,
            "reaction_max_duration_s": 0.0,
            "require_ball_in_shot": False,
            "exclude_fade_transitions": True,
            "fade_transition_max_duration_s": 1.0,
            "fade_black_frame_threshold": 0.18,
            "fade_min_brightness_range": 0.25,
        }
    }

    monkeypatch.setattr(
        "src.stages.segmentation.detect_shots",
        lambda *args, **kwargs: [
            _ShotSpan(start_frame=0, end_frame=19, start_time=0.0, end_time=0.8),
            _ShotSpan(start_frame=20, end_frame=29, start_time=0.8, end_time=1.2),
            _ShotSpan(start_frame=30, end_frame=49, start_time=1.2, end_time=2.0),
        ],
    )

    stage = ShotSegmentationStage(
        config=cfg,
        output_dir=tmp_path,
        video_path=video_with_fade_transition,
    )
    stage.run()

    manifest = ShotsManifest.load(tmp_path / "shots" / "shots_manifest.json")
    assert len(manifest.shots) == 2
    assert manifest.shots[0].end_frame == 19
    assert manifest.shots[1].start_frame == 30


def test_stage1_keeps_fade_transition_span_when_filter_disabled(tmp_path, video_with_fade_transition, monkeypatch):
    cfg = {
        "shot_segmentation": {
            "min_shot_duration_s": 0.1,
            "merge_short_shots_max_duration_s": 0.0,
            "reaction_max_duration_s": 0.0,
            "require_ball_in_shot": False,
            "exclude_fade_transitions": False,
        }
    }

    monkeypatch.setattr(
        "src.stages.segmentation.detect_shots",
        lambda *args, **kwargs: [
            _ShotSpan(start_frame=0, end_frame=19, start_time=0.0, end_time=0.8),
            _ShotSpan(start_frame=20, end_frame=29, start_time=0.8, end_time=1.2),
            _ShotSpan(start_frame=30, end_frame=49, start_time=1.2, end_time=2.0),
        ],
    )

    stage = ShotSegmentationStage(
        config=cfg,
        output_dir=tmp_path,
        video_path=video_with_fade_transition,
    )
    stage.run()

    manifest = ShotsManifest.load(tmp_path / "shots" / "shots_manifest.json")
    assert len(manifest.shots) == 3
