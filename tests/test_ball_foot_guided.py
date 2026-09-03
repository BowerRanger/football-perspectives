"""Foot-guided ball detection (inference-only): zoom around fast-moving
contact joints to recover the ball at touches. Pure core tested with a fake
player context + injected zoom fn."""

from __future__ import annotations

import inspect

from src.pipeline.config import load_config
from src.utils.ball_foot_guided import foot_ball_detections, gated_feet
from src.utils.ball_player_context import JointSample


class _Ctx:
    def __init__(self, by_frame):
        self._bf = by_frame

    def joints_at(self, f):
        return tuple(self._bf.get(f, ()))


def _js(pid, bone, uv):
    return JointSample(pid, bone, (1.0, 2.0, 0.3), uv, 0.8)


def test_gated_feet_keeps_fast_foot_drops_slow():
    # r_foot moves 10px/frame (kick); l_foot static
    ctx = _Ctx({
        9: [_js("P1", "r_foot", (100.0, 100.0)), _js("P1", "l_foot", (50.0, 50.0))],
        10: [_js("P1", "r_foot", (110.0, 100.0)), _js("P1", "l_foot", (50.0, 50.0))],
        11: [_js("P1", "r_foot", (120.0, 100.0)), _js("P1", "l_foot", (50.0, 50.0))],
    })
    g = gated_feet(ctx, n_frames=12, min_foot_speed_px=8.0)
    assert 10 in g
    bones = {b for _, b, _ in g[10]}
    assert "r_foot" in bones and "l_foot" not in bones


def test_foot_ball_detections_accepts_ball_near_foot():
    gated = {10: [("P1", "r_foot", (110.0, 100.0))]}
    # zoom finds a ball 12px from the foot, good score
    def zoom(f, center):
        return ((118.0, 104.0), 0.6)
    out = foot_ball_detections(gated, zoom, ball_near_foot_px=40.0, min_score=0.2)
    assert len(out) == 1
    f, pid, bone, buv, score = out[0]
    assert f == 10 and pid == "P1" and bone == "r_foot" and score == 0.6


def test_foot_ball_detections_rejects_ball_far_from_foot():
    gated = {10: [("P1", "r_foot", (110.0, 100.0))]}
    def zoom(f, center):
        return ((300.0, 300.0), 0.9)  # found something, but nowhere near the foot
    assert foot_ball_detections(gated, zoom, ball_near_foot_px=40.0) == []


def test_foot_ball_detections_best_per_frame():
    gated = {10: [("P1", "r_foot", (110.0, 100.0)), ("P2", "l_foot", (112.0, 100.0))]}
    def zoom(f, center):
        # the P2 foot's zoom finds a stronger ball
        return ((center[0] + 5, center[1]), 0.4 if center[0] == 110.0 else 0.8)
    out = foot_ball_detections(gated, zoom, ball_near_foot_px=40.0)
    assert len(out) == 1 and out[0][1] == "P2"


def test_ball_near_foot_px_matches_measured_corridor():
    """Corridor radius, tightened 2026-09-03 (workstream 5) after the FK
    contact-anchoring fix. Measured accepted-detection distance from the
    seeding foot/knee/head/chest on the raw (gap_fill=False) foot_guided
    frames of the observations sidecars across gberch/kroupi01/japan-s013/
    origi01/origi02: the one fresh-FK same-run sample (gberch, n=6) maxed
    at 32.1px (median 5.2px); all 4 clips combined (n=19, most from
    pre-FK-fix ball runs) had 16/19 (84%) <= 32.1px. 40.0 keeps ~8px
    margin over the clean max while cutting accept area ~21% vs the old
    45.0px default. config/default.yaml and the ball stage's fallback
    default must move together, so pin both here."""
    cfg = load_config(None)
    configured = float(cfg["ball"]["foot_guided"]["ball_near_foot_px"])
    module_default = inspect.signature(foot_ball_detections).parameters[
        "ball_near_foot_px"].default
    assert configured == 40.0
    assert module_default == 40.0
