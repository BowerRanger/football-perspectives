"""Pose-anchored touch attribution (Phase C): relaxed radius + kinematic
alignment turn direction-change breaks into touches the 25px gate missed."""

from __future__ import annotations

from math import hypot

from src.utils.ball_auto_events import AutoEventCfg, _Break
from src.utils.ball_player_context import JointSample
from src.utils.ball_pose_touch import classify_touch, joint_pixel_velocity


class _Ctx:
    def __init__(self, by_frame):
        self._bf = by_frame  # {frame: [JointSample]}

    def joints_at(self, f):
        return tuple(self._bf.get(f, ()))

    def joints_near_pixel(self, f, uv, r):
        out = [s for s in self._bf.get(f, ())
               if s.uv and hypot(s.uv[0] - uv[0], s.uv[1] - uv[1]) <= r]
        out.sort(key=lambda s: hypot(s.uv[0] - uv[0], s.uv[1] - uv[1]))
        return out


def _js(player, bone, uv, conf=0.8):
    return JointSample(player, bone, (1.0, 2.0, 0.3), uv, conf)


def _brk(frame=10):
    return _Break(frame=frame, strength=0.8, dir_change_deg=40.0, dspeed_px=10.0,
                  speed_before=5.0, speed_after=12.0, vy_before=0.0, vy_after=-12.0)


def test_relaxed_radius_rescues_touch_outside_25px():
    # foot 40px from the ball: the old 25px gate misses it, relaxed 60px gets it
    ctx = _Ctx({10: [_js("P1", "r_foot", (140.0, 100.0))]})
    ev = classify_touch(_brk(), (100.0, 100.0), ctx,
                        AutoEventCfg(touch_relaxed_px=60.0))
    assert ev is not None and ev.kind == "touch"
    assert ev.player_id == "P1" and ev.bone == "r_foot"


def test_no_joint_in_relaxed_radius_returns_none():
    ctx = _Ctx({10: [_js("P1", "r_foot", (300.0, 300.0))]})
    assert classify_touch(_brk(), (100.0, 100.0), ctx,
                         AutoEventCfg(touch_relaxed_px=60.0)) is None


def test_joint_pixel_velocity_central_difference():
    ctx = _Ctx({
        9: [_js("P1", "r_foot", (90.0, 100.0))],
        11: [_js("P1", "r_foot", (110.0, 100.0))],
    })
    v = joint_pixel_velocity(ctx, 10, "P1", "r_foot")
    assert v is not None
    assert abs(v[0] - 10.0) < 1e-6 and abs(v[1]) < 1e-6  # (110-90)/2 per frame


def test_kinematic_alignment_prefers_the_kicking_foot():
    # two feet within radius; only r_foot is moving with the ball's outbound
    # direction (+x). r_foot should win on score (be the attributed bone).
    ctx = _Ctx({
        10: [_js("P1", "r_foot", (120.0, 100.0)), _js("P2", "l_foot", (122.0, 100.0))],
        9: [_js("P1", "r_foot", (108.0, 100.0)), _js("P2", "l_foot", (122.0, 100.0))],
        11: [_js("P1", "r_foot", (132.0, 100.0)), _js("P2", "l_foot", (122.0, 100.0))],
    })
    # ball turns to move +x fast (v_after along +x)
    brk = _Break(frame=10, strength=0.6, dir_change_deg=50.0, dspeed_px=10.0,
                 speed_before=4.0, speed_after=14.0, vy_before=0.0, vy_after=0.0)
    ev = classify_touch(brk, (100.0, 100.0), ctx,
                        AutoEventCfg(touch_relaxed_px=60.0, kinematic_bonus_weight=0.5))
    assert ev is not None and ev.player_id == "P1" and ev.bone == "r_foot"
