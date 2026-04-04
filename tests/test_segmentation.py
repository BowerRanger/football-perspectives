import cv2
import numpy as np
import pytest
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


from src.stages.segmentation import detect_shots, ShotSegmentationStage
from src.schemas.shots import ShotsManifest


def test_detect_shots_finds_cut(tiny_video):
    shots = detect_shots(tiny_video, threshold=20.0)
    # Synthetic video has a hard cut between blue and green frames
    assert len(shots) == 2


def test_detect_shots_returns_correct_timings(tiny_video):
    shots = detect_shots(tiny_video, threshold=20.0)
    assert shots[0].start_frame == 0
    assert shots[1].start_frame > 0


def test_stage1_writes_manifest(tmp_path, tiny_video):
    cfg = {
        "shot_segmentation": {"threshold": 20.0, "min_shot_duration_s": 0.1}
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


def test_stage1_is_complete_after_run(tmp_path, tiny_video):
    cfg = {"shot_segmentation": {"threshold": 20.0, "min_shot_duration_s": 0.1}}
    stage = ShotSegmentationStage(config=cfg, output_dir=tmp_path, video_path=tiny_video)
    assert not stage.is_complete()
    stage.run()
    assert stage.is_complete()
