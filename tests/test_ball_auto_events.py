"""Auto event detection: touches, bounces, goal impacts, stationary spans.

Synthetic scenes with analytic world trajectories projected through a
known camera. Each scene asserts both recall (the event is found at the
right frame with the right kind/metadata) and precision (no spurious
events on smooth motion).
"""
from __future__ import annotations

import numpy as np
import pytest

from src.utils.ball_auto_events import AutoEventCfg, BallEvent, detect_events
from src.utils.ball_player_context import JointSample
from src.utils.goal_geometry import GoalGeometry
from tests.fixtures.ball_synthetic import (
    FPS,
    FakePlayerContext,
    ballistic_worlds,
    broadcast_camera,
    per_frame_cams,
    project_track,
    rolling_worlds,
    steps_from_pixels,
)

PITCH_CFG = {
    "length_m": 105.0, "width_m": 68.0,
    "goal_height_m": 2.44, "goal_width_m": 7.32, "goal_depth_m": 1.5,
}
CFG = AutoEventCfg()


def _detect(steps, n, *, ctx=None, confidences=None, cams=None,
            goal_geometry=None, cfg=CFG):
    if cams is None:
        cams = per_frame_cams(n)
    Ks, Rs, ts = cams
    return detect_events(
        steps=steps,
        confidences=confidences or {},
        player_ctx=ctx or FakePlayerContext(),
        per_frame_K=Ks, per_frame_R=Rs, per_frame_t=ts,
        distortion=(0.0, 0.0),
        goal_geometry=goal_geometry,
        cfg=cfg,
    )


def _events_of(events, kind):
    return [e for e in events if e.kind == kind]


def _joint(player_id, bone, world, uv, conf=0.8):
    return JointSample(
        player_id=player_id, bone=bone,
        world_xyz=tuple(float(x) for x in world),
        uv=(float(uv[0]), float(uv[1])), confidence=conf,
    )


class TestTouch:
    def test_pass_direction_change_at_foot(self):
        n = 40
        K, R, t = broadcast_camera()
        # Roll -x for 20 frames, then +y for 20 (sharp turn at frame 20).
        worlds = rolling_worlds((50.0, 30.0), (-0.30, 0.0), 21)
        worlds.update(rolling_worlds((44.0, 30.0), (0.0, 0.30), 20,
                                     start_frame=20))
        pixels = project_track(worlds, K, R, t)
        touch_world = worlds[20]
        ctx = FakePlayerContext({
            20: [_joint("P007", "l_foot", touch_world, pixels[20])],
        })
        events = _detect(steps_from_pixels(pixels, n), n, ctx=ctx)
        touches = _events_of(events, "touch")
        assert len(touches) == 1
        ev = touches[0]
        assert abs(ev.frame - 20) <= 1
        assert ev.player_id == "P007"
        assert ev.bone == "l_foot"
        assert 0.0 < ev.score <= 1.0
        assert not _events_of(events, "bounce")
        assert not _events_of(events, "goal_impact")

    def test_break_without_player_is_not_touch(self):
        n = 40
        K, R, t = broadcast_camera()
        worlds = rolling_worlds((50.0, 30.0), (-0.30, 0.0), 21)
        worlds.update(rolling_worlds((44.0, 30.0), (0.0, 0.30), 20,
                                     start_frame=20))
        pixels = project_track(worlds, K, R, t)
        events = _detect(steps_from_pixels(pixels, n), n)
        assert not _events_of(events, "touch")
        # The kink is still surfaced for the solver as a generic break.
        breaks = _events_of(events, "velocity_break")
        assert breaks and abs(breaks[0].frame - 20) <= 1


class TestBounce:
    def test_bounce_detected_at_ground_contact(self):
        n = 40
        K, R, t = broadcast_camera()
        worlds = ballistic_worlds(
            (60.0, 40.0, 3.0), (-6.0, 0.0, 0.0), n, restitution=0.7,
        )
        # Ground contact frame: first frame where z stops decreasing.
        zs = [worlds[i][2] for i in range(n)]
        bounce_frame = int(np.argmin(zs))
        pixels = project_track(worlds, K, R, t)
        events = _detect(
            steps_from_pixels(pixels, n, p_flight=0.9), n,
        )
        bounces = _events_of(events, "bounce")
        assert len(bounces) == 1
        assert abs(bounces[0].frame - bounce_frame) <= 2

    def test_bounce_near_player_is_touch_instead(self):
        n = 40
        K, R, t = broadcast_camera()
        worlds = ballistic_worlds(
            (60.0, 40.0, 3.0), (-6.0, 0.0, 0.0), n, restitution=0.7,
        )
        zs = [worlds[i][2] for i in range(n)]
        bounce_frame = int(np.argmin(zs))
        pixels = project_track(worlds, K, R, t)
        ctx = FakePlayerContext({
            bounce_frame: [_joint(
                "P002", "r_foot", worlds[bounce_frame], pixels[bounce_frame]
            )],
        })
        events = _detect(steps_from_pixels(pixels, n, p_flight=0.9), n, ctx=ctx)
        assert not _events_of(events, "bounce")
        touches = _events_of(events, "touch")
        assert touches and abs(touches[0].frame - bounce_frame) <= 2


class TestGoalImpact:
    def _goal_scene(self):
        """Flight into the near goal (x=0), camera near that goal."""
        cam = broadcast_camera(
            cam_centre=(30.0, 10.0, 12.0), target=(0.0, 34.0, 1.5),
        )
        return cam, GoalGeometry.from_pitch_config(PITCH_CFG)

    def test_crossbar_hit_classified(self):
        (K, R, t), geometry = self._goal_scene()
        n = 40
        # Inbound flight ends exactly on the crossbar centre at frame 20,
        # then rebounds back out (+x).
        bar_point = np.array([0.0, 34.0, 2.44])
        v_in = np.array([-12.0, 0.0, 1.0])
        worlds = {}
        for i in range(21):
            dt = (20 - i) / FPS
            worlds[i] = bar_point - v_in * dt
        v_out = np.array([9.0, 0.0, -1.0])
        for i in range(21, n):
            dt = (i - 20) / FPS
            worlds[i] = bar_point + v_out * dt
        pixels = project_track(worlds, K, R, t)
        events = _detect(
            steps_from_pixels(pixels, n, p_flight=0.9), n,
            cams=({i: K for i in range(n)}, {i: R for i in range(n)},
                  {i: t for i in range(n)}),
            goal_geometry=geometry,
        )
        impacts = _events_of(events, "goal_impact")
        assert len(impacts) == 1
        assert abs(impacts[0].frame - 20) <= 1
        assert impacts[0].goal_element == "crossbar"

    def test_net_hit_needs_speed_collapse(self):
        (K, R, t), geometry = self._goal_scene()
        n = 40
        # Shot into the middle of the goal mouth; ball stops dead in the
        # net at frame 20 (speed collapse).
        net_point = np.array([-0.7, 34.0, 1.2])
        v_in = np.array([-14.0, 0.0, 0.5])
        worlds = {}
        for i in range(21):
            dt = (20 - i) / FPS
            worlds[i] = net_point - v_in * dt
        for i in range(21, n):
            # Caught in the net: barely moving.
            worlds[i] = net_point + np.array([0.0, 0.0, -0.002 * (i - 20)])
        pixels = project_track(worlds, K, R, t)
        events = _detect(
            steps_from_pixels(pixels, n, p_flight=0.9), n,
            cams=({i: K for i in range(n)}, {i: R for i in range(n)},
                  {i: t for i in range(n)}),
            goal_geometry=geometry,
        )
        impacts = _events_of(events, "goal_impact")
        assert len(impacts) == 1
        assert abs(impacts[0].frame - 20) <= 1
        assert impacts[0].goal_element == "back_net"

    def test_keeper_catch_in_goal_mouth_is_touch_not_net(self):
        (K, R, t), geometry = self._goal_scene()
        n = 40
        catch_point = np.array([1.0, 34.0, 1.4])
        v_in = np.array([-13.0, 0.0, 0.0])
        worlds = {}
        for i in range(21):
            dt = (20 - i) / FPS
            worlds[i] = catch_point - v_in * dt
        for i in range(21, n):
            worlds[i] = catch_point
        pixels = project_track(worlds, K, R, t)
        ctx = FakePlayerContext({
            20: [_joint("P001", "r_hand", catch_point, pixels[20])],
        })
        events = _detect(
            steps_from_pixels(pixels, n, p_flight=0.9), n, ctx=ctx,
            cams=({i: K for i in range(n)}, {i: R for i in range(n)},
                  {i: t for i in range(n)}),
            goal_geometry=geometry,
        )
        touches = _events_of(events, "touch")
        assert touches and touches[0].bone == "r_hand"
        assert not _events_of(events, "goal_impact")


class TestPrecisionAndStationary:
    def test_smooth_roll_produces_no_events(self):
        n = 50
        K, R, t = broadcast_camera()
        worlds = rolling_worlds((60.0, 30.0), (-0.25, 0.05), n)
        pixels = project_track(worlds, K, R, t)
        events = _detect(steps_from_pixels(pixels, n), n)
        assert events == ()

    def test_smooth_flight_produces_no_events(self):
        n = 30
        K, R, t = broadcast_camera()
        worlds = ballistic_worlds((50.0, 30.0, 0.11), (8.0, 2.0, 6.0), n)
        pixels = project_track(worlds, K, R, t)
        events = _detect(steps_from_pixels(pixels, n, p_flight=0.9), n)
        assert _events_of(events, "bounce") == []
        assert _events_of(events, "touch") == []

    def test_stationary_span_then_kick(self):
        n = 40
        K, R, t = broadcast_camera()
        worlds = {i: np.array([55.0, 30.0, 0.11]) for i in range(14)}
        worlds.update(rolling_worlds((55.0, 30.0), (0.45, 0.0), n - 14,
                                     start_frame=14))
        pixels = project_track(worlds, K, R, t)
        conf = {i: 0.9 for i in range(n)}
        events = _detect(
            steps_from_pixels(pixels, n), n, confidences=conf,
        )
        spans = _events_of(events, "stationary")
        assert len(spans) == 1
        assert spans[0].frame <= 2
        assert spans[0].end_frame is not None
        assert abs(spans[0].end_frame - 13) <= 2
        # The launch is surfaced as a break (no joint nearby).
        assert any(
            e.kind in ("velocity_break", "touch") and abs(e.frame - 14) <= 2
            for e in events
        )

    def test_events_sorted_and_immutable(self):
        n = 40
        K, R, t = broadcast_camera()
        worlds = rolling_worlds((50.0, 30.0), (-0.30, 0.0), 21)
        worlds.update(rolling_worlds((44.0, 30.0), (0.0, 0.30), 20,
                                     start_frame=20))
        pixels = project_track(worlds, K, R, t)
        events = _detect(steps_from_pixels(pixels, n), n)
        assert isinstance(events, tuple)
        frames = [e.frame for e in events]
        assert frames == sorted(frames)
        with pytest.raises(AttributeError):
            events[0].frame = 0  # frozen
