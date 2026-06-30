"""detect_events uses robust segmentation for direction changes when
ball.segment is enabled (Phase B4)."""

from __future__ import annotations

import numpy as np

from src.utils.ball_auto_events import AutoEventCfg, _select_breaks, detect_events


class _Step:
    def __init__(self, frame, uv):
        self.frame = frame
        self.uv = uv


class _NoCtx:
    def joints_near_pixel(self, frame, uv, r):
        return []

    def joint_world(self, *a):
        return None


def _turn_uvs():
    a = {f: (100.0 + 6.0 * f, 300.0) for f in range(16)}     # rightward
    b = {f: (190.0, 300.0 - 6.0 * (f - 15)) for f in range(15, 31)}  # upward
    return {**a, **b}


def test_select_breaks_uses_segmentation_when_enabled():
    uvs = {f: np.asarray(p, float) for f, p in _turn_uvs().items()}
    breaks = _select_breaks(uvs, AutoEventCfg(use_segmentation=True))
    assert any(abs(b.frame - 15) <= 2 for b in breaks)


def test_detect_events_finds_turn_via_segmentation():
    uvs = _turn_uvs()
    steps = [_Step(f, p) for f, p in sorted(uvs.items())]
    evs = detect_events(
        steps=steps,
        confidences={f: 0.9 for f in uvs},
        player_ctx=_NoCtx(),
        per_frame_K={}, per_frame_R={}, per_frame_t={},
        cfg=AutoEventCfg(use_segmentation=True),
    )
    assert any(abs(e.frame - 15) <= 2 for e in evs)
