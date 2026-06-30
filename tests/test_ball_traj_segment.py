"""Robust piecewise direction-change segmentation (Phase B2/B3): corners of
the fitted track are touch/bounce candidates. Pure, deterministic."""

from __future__ import annotations

import random

from src.utils.ball_auto_events import AutoEventCfg, _Break
from src.utils.ball_traj_segment import segment_track


def _line(f0, f1, p0, v):
    return {f: (p0[0] + v[0] * (f - f0), p0[1] + v[1] * (f - f0))
            for f in range(f0, f1 + 1)}


def test_straight_roll_has_no_breaks():
    uvs = _line(0, 30, (100.0, 100.0), (5.0, 0.0))
    assert segment_track(uvs, cfg=AutoEventCfg()) == []


def test_sharp_corner_is_detected():
    # moving right to frame 15, then turning to move straight up
    a = _line(0, 15, (100.0, 300.0), (6.0, 0.0))
    b = {f: (190.0, 300.0 - 6.0 * (f - 15)) for f in range(15, 31)}
    uvs = {**a, **b}
    breaks = segment_track(uvs, cfg=AutoEventCfg())
    near = [bk for bk in breaks if abs(bk.frame - 15) <= 2]
    assert near, f"expected a break near frame 15, got {[b.frame for b in breaks]}"
    assert isinstance(near[0], _Break)
    assert near[0].dir_change_deg > 60.0  # ~90-degree turn


def test_robust_to_dropped_points():
    a = _line(0, 15, (100.0, 300.0), (6.0, 0.0))
    b = {f: (190.0, 300.0 - 6.0 * (f - 15)) for f in range(15, 31)}
    uvs = {**a, **b}
    rng = random.Random(0)
    droppable = [k for k in uvs if k not in (0, 15, 30)]
    for k in rng.sample(droppable, int(0.4 * len(droppable))):
        uvs.pop(k)
    breaks = segment_track(uvs, cfg=AutoEventCfg())
    assert any(abs(bk.frame - 15) <= 3 for bk in breaks)


def test_deterministic():
    a = _line(0, 15, (100.0, 300.0), (6.0, 0.0))
    b = {f: (190.0, 300.0 - 6.0 * (f - 15)) for f in range(15, 31)}
    uvs = {**a, **b}
    r1 = segment_track(uvs, cfg=AutoEventCfg())
    r2 = segment_track(uvs, cfg=AutoEventCfg())
    assert [bk.frame for bk in r1] == [bk.frame for bk in r2]
