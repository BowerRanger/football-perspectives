"""Tests for ball_cue_blur on synthetic moving blobs with known
orientation/length (a drawn streak), verifying PCA recovers them, the
implied-speed scaling, the angle-wrap helper, and the end-to-end
direction-change flagging."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.utils.ball_cue_blur import (
    analyze_blob,
    angle_delta_deg,
    compute_blur_cues,
    crop_around,
    implied_speed_px_s,
)


def _draw_streak(shape, center, length, angle_deg, thickness=3, value=220):
    img = np.zeros(shape, dtype=np.uint8)
    rad = np.radians(angle_deg)
    dx, dy = np.cos(rad) * length / 2, np.sin(rad) * length / 2
    p0 = (int(round(center[0] - dx)), int(round(center[1] - dy)))
    p1 = (int(round(center[0] + dx)), int(round(center[1] + dy)))
    cv2.line(img, p0, p1, value, thickness)
    return img


def test_analyze_blob_recovers_angle_and_length():
    img = _draw_streak((60, 60), (30, 30), length=30, angle_deg=30.0)
    shape = analyze_blob(img)
    assert shape is not None
    assert shape.length_px == pytest.approx(30.0, abs=5.0)
    assert angle_delta_deg(shape.angle_deg, 30.0) < 10.0


def test_analyze_blob_recovers_near_vertical_angle():
    img = _draw_streak((60, 60), (30, 30), length=25, angle_deg=88.0)
    shape = analyze_blob(img)
    assert shape is not None
    assert angle_delta_deg(shape.angle_deg, 88.0) < 10.0


def test_analyze_blob_none_on_empty_crop():
    img = np.zeros((40, 40), dtype=np.uint8)
    assert analyze_blob(img) is None


def test_analyze_blob_none_on_empty_array():
    assert analyze_blob(np.zeros((0, 0), dtype=np.uint8)) is None


def test_implied_speed_scales_with_length_and_fps():
    s1 = implied_speed_px_s(10.0, fps=30.0)
    s2 = implied_speed_px_s(20.0, fps=30.0)
    assert s2 == pytest.approx(2 * s1)
    s_high_fps = implied_speed_px_s(10.0, fps=60.0)
    assert s_high_fps > s1  # shorter assumed exposure -> higher implied speed


def test_angle_delta_wraps_at_180():
    assert angle_delta_deg(5.0, 175.0) == pytest.approx(10.0, abs=0.5)
    assert angle_delta_deg(10.0, 100.0) == pytest.approx(90.0, abs=0.5)
    assert angle_delta_deg(40.0, 40.0) == pytest.approx(0.0, abs=0.5)


def test_crop_around_clamps_to_bounds():
    img = np.zeros((50, 50), dtype=np.uint8)
    crop = crop_around(img, (2, 2), radius=10)
    assert crop.shape[0] <= 13 and crop.shape[1] <= 13
    crop_full = crop_around(img, (25, 25), radius=10)
    assert crop_full.shape == (21, 21)


def test_compute_blur_cues_flags_sharp_direction_change_only():
    frames = {
        0: _draw_streak((80, 80), (40, 40), length=25, angle_deg=10.0),
        1: _draw_streak((80, 80), (40, 40), length=25, angle_deg=80.0),   # sharp turn
        2: _draw_streak((80, 80), (40, 40), length=25, angle_deg=85.0),   # small change
    }

    def lookup(f):
        return frames.get(f)

    detections = [(0, (40.0, 40.0)), (1, (40.0, 40.0)), (2, (40.0, 40.0))]
    events = compute_blur_cues(detections, lookup, fps=30.0, crop_radius=39,
                                angle_change_deg=25.0)
    flagged = [e.frame for e in events]
    assert 1 in flagged
    assert 2 not in flagged
    assert all(e.cue == "blur_direction_change" for e in events)


def test_compute_blur_cues_skips_missing_frames():
    detections = [(0, (10.0, 10.0)), (5, (10.0, 10.0))]
    events = compute_blur_cues(detections, lambda f: None, fps=30.0)
    assert events == []


def test_compute_blur_cues_respects_max_frame_gap():
    frames = {
        0: _draw_streak((80, 80), (40, 40), length=25, angle_deg=0.0),
        10: _draw_streak((80, 80), (40, 40), length=25, angle_deg=90.0),
    }
    detections = [(0, (40.0, 40.0)), (10, (40.0, 40.0))]
    events = compute_blur_cues(detections, lambda f: frames.get(f), fps=30.0,
                                crop_radius=39, max_frame_gap=3)
    assert events == []
