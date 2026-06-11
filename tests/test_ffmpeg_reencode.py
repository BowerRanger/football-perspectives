"""Frame-accurate re-encode extraction (highlights split mode)."""
import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.utils.ffmpeg import extract_clip_reencode

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None,
                                reason="ffmpeg not on PATH")
FPS = 25.0


@pytest.fixture()
def source_video(tmp_path: Path) -> Path:
    """4 s of 64x64 frames; blue channel ramps with the frame index."""
    p = tmp_path / "src.mp4"
    w = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (64, 64))
    for i in range(100):
        frame = np.full((64, 64, 3), (min(255, i * 2), 40, 200), np.uint8)
        w.write(frame)
    w.release()
    return p


def _frame_count(p: Path) -> int:
    cap = cv2.VideoCapture(str(p))
    try:
        return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()


def test_reencode_extracts_requested_span(source_video, tmp_path):
    out = tmp_path / "clip.mp4"
    extract_clip_reencode(source_video, out, start_s=1.0, end_s=3.0, fps=FPS)
    assert out.exists()
    assert abs(_frame_count(out) - 50) <= 2  # 2 s at 25 fps, codec tolerance


def test_reencode_retimes_slow_motion(source_video, tmp_path):
    out = tmp_path / "clip.mp4"
    # speed_factor 2.0 == span is 2x slow-mo -> retimed output halves it
    extract_clip_reencode(source_video, out, start_s=0.0, end_s=4.0, fps=FPS,
                          speed_factor=2.0)
    assert abs(_frame_count(out) - 50) <= 3


def test_reencode_source_without_audio_still_works(source_video, tmp_path):
    # cv2.VideoWriter sources have no audio stream; the audio mapping
    # must degrade gracefully rather than fail the extraction.
    out = tmp_path / "clip.mp4"
    extract_clip_reencode(source_video, out, start_s=0.0, end_s=2.0, fps=FPS,
                          speed_factor=1.5)
    assert out.exists() and _frame_count(out) > 0
