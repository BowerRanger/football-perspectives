"""Unit tests for the Layer 4 appearance bridge."""

from __future__ import annotations

import numpy as np
import pytest

from src.utils.ball_appearance_bridge import (
    AppearanceBridge,
    AppearanceBridgeCfg,
)


def _cfg(**over) -> AppearanceBridgeCfg:
    base = dict(
        enabled=True,
        max_gap_frames=8,
        template_size_px=32,
        search_radius_px=64,
        min_ncc=0.6,
        template_max_age_frames=30,
        template_update_confidence=0.5,
    )
    base.update(over)
    return AppearanceBridgeCfg(**base)


def _frame_with_ball(uv: tuple[int, int], shape=(720, 1280)) -> np.ndarray:
    """A pitch-green frame with a white-ish ball at uv."""
    img = np.full((*shape, 3), [50, 200, 50], dtype=np.uint8)
    u, v = int(uv[0]), int(uv[1])
    for du in range(-6, 7):
        for dv in range(-6, 7):
            if du * du + dv * dv <= 36 and 0 <= v + dv < shape[0] and 0 <= u + du < shape[1]:
                img[v + dv, u + du] = [240, 240, 240]
    return img


def test_bridge_finds_ball_in_predicted_window():
    bridge = AppearanceBridge(_cfg())
    f0 = _frame_with_ball((640, 360))
    bridge.update_template(frame=0, frame_image=f0, uv=(640.0, 360.0), confidence=0.9)
    # Next frame: ball moved by (10, 5).
    f1 = _frame_with_ball((650, 365))
    result = bridge.try_bridge(frame=1, frame_image=f1, predicted_uv=(648.0, 364.0), consecutive_misses=1)
    assert result is not None
    uv, conf = result
    assert abs(uv[0] - 650.0) < 2.0
    assert abs(uv[1] - 365.0) < 2.0
    assert 0.0 < conf < 1.0


def test_bridge_returns_none_when_no_ball_in_window():
    bridge = AppearanceBridge(_cfg())
    f0 = _frame_with_ball((640, 360))
    bridge.update_template(frame=0, frame_image=f0, uv=(640.0, 360.0), confidence=0.9)
    # Plain green frame, no ball anywhere.
    green = np.full((720, 1280, 3), [50, 200, 50], dtype=np.uint8)
    result = bridge.try_bridge(frame=1, frame_image=green, predicted_uv=(648.0, 364.0), consecutive_misses=1)
    assert result is None


def test_bridge_disabled_after_max_gap():
    bridge = AppearanceBridge(_cfg(max_gap_frames=8))
    f0 = _frame_with_ball((640, 360))
    bridge.update_template(frame=0, frame_image=f0, uv=(640.0, 360.0), confidence=0.9)
    f9 = _frame_with_ball((730, 405))
    result = bridge.try_bridge(frame=9, frame_image=f9, predicted_uv=(728.0, 404.0), consecutive_misses=9)
    assert result is None


def test_bridge_disabled_when_template_stale():
    bridge = AppearanceBridge(_cfg(template_max_age_frames=5))
    f0 = _frame_with_ball((640, 360))
    bridge.update_template(frame=0, frame_image=f0, uv=(640.0, 360.0), confidence=0.9)
    f10 = _frame_with_ball((650, 365))
    result = bridge.try_bridge(frame=10, frame_image=f10, predicted_uv=(648.0, 364.0), consecutive_misses=1)
    assert result is None


def test_bridge_disabled_by_config_flag():
    bridge = AppearanceBridge(_cfg(enabled=False))
    f0 = _frame_with_ball((640, 360))
    bridge.update_template(frame=0, frame_image=f0, uv=(640.0, 360.0), confidence=0.9)
    f1 = _frame_with_ball((650, 365))
    result = bridge.try_bridge(frame=1, frame_image=f1, predicted_uv=(648.0, 364.0), consecutive_misses=1)
    assert result is None


def test_update_template_ignored_when_low_confidence():
    bridge = AppearanceBridge(_cfg(template_update_confidence=0.5))
    f0 = _frame_with_ball((640, 360))
    bridge.update_template(frame=0, frame_image=f0, uv=(640.0, 360.0), confidence=0.2)
    f1 = _frame_with_ball((650, 365))
    result = bridge.try_bridge(frame=1, frame_image=f1, predicted_uv=(648.0, 364.0), consecutive_misses=1)
    assert result is None
