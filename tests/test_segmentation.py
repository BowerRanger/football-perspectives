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
