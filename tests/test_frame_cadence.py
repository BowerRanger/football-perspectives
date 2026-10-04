"""Frame cadence: repeated-frame detection (25->30 pulldown) and the
content-time correction the Ball Studio solver uses."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from src.utils import frame_cadence as fc

FPS = 30.0


def pulldown_repeats(n: int, phase: int = 5) -> list[int]:
    """Every 6th display frame repeats its predecessor (25 -> 30 fps)."""
    return [f for f in range(1, n) if f % 6 == phase]


def test_no_repeats_means_no_shift():
    shift = fc.content_time_shift(120, [], FPS)
    assert shift.shape == (120,)
    assert np.allclose(shift, 0.0)


def test_repeat_frame_shows_the_same_instant_as_its_predecessor():
    n = 180
    reps = pulldown_repeats(n)
    shift = fc.content_time_shift(n, reps, FPS)
    t = np.arange(n) / FPS + shift
    for f in reps:
        assert t[f] == pytest.approx(t[f - 1], abs=1e-9)


def test_fresh_frames_are_spaced_at_the_source_rate():
    n = 180
    reps = set(pulldown_repeats(n))
    shift = fc.content_time_shift(n, sorted(reps), FPS)
    t = np.arange(n) / FPS + shift
    fresh = [f for f in range(n) if f not in reps]
    steps = np.diff(t[fresh])[10:-10]  # away from the detrend edges
    assert np.allclose(steps, 1 / 25.0, atol=1e-6)


def test_shift_is_locally_zero_mean_and_bounded_by_a_frame():
    n = 300
    shift = fc.content_time_shift(n, pulldown_repeats(n), FPS)
    assert abs(float(shift[30:-30].mean())) < 1e-3
    assert np.abs(shift).max() < 1.0 / FPS


def test_detect_repeats_finds_duplicated_frames(tmp_path: Path):
    path = tmp_path / "clip.avi"
    w, h = 160, 96
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (w, h))
    rng = np.random.default_rng(0)
    frames = []
    for i in range(24):
        if i in (5, 11, 17, 23):
            frames.append(frames[-1])
        else:
            frames.append(rng.integers(0, 255, (h, w, 3), dtype=np.uint8))
    for fr in frames:
        vw.write(fr)
    vw.release()
    assert fc.detect_repeats(path) == [5, 11, 17, 23]


def test_cadence_cache_round_trip(tmp_path: Path):
    video = tmp_path / "v.avi"
    vw = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (64, 48))
    a = np.full((48, 64, 3), 10, np.uint8)
    b = np.full((48, 64, 3), 200, np.uint8)
    for fr in (a, b, b, a):
        vw.write(fr)
    vw.release()
    cache = tmp_path / "cache"
    first = fc.load_or_detect(video, cache)
    assert first.repeats == (2,) and first.n_frames == 4
    assert (cache / "v.json").exists()
    again = fc.load_or_detect(video, cache)
    assert again == first
