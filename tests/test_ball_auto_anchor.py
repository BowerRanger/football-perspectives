"""Auto-anchor generation: events + grounded sampling -> BallAnchor records.

Covers event->anchor mapping (incl. shot tagging and the contact-gap
gate), grounded keyframe sampling, reachability validation, the
manual-wins merge policy, and sidecar round-trip through the existing
BallAnchorSet schema.
"""
from __future__ import annotations

import pytest
import numpy as np

from src.schemas.ball_anchor import BallAnchor, BallAnchorSet
from src.utils.ball_auto_anchor import (
    AutoAnchorCfg,
    auto_anchor_path,
    generate_auto_anchors,
    merge_anchors,
)
from src.utils.ball_auto_events import BallEvent
from src.utils.ball_player_context import JointSample
from src.utils.camera_projection import project_world_to_image
from tests.fixtures.ball_synthetic import (
    FPS,
    FakePlayerContext,
    broadcast_camera,
    per_frame_cams,
    project_track,
    rolling_worlds,
    steps_from_pixels,
)

CFG = AutoAnchorCfg()
PITCH_CFG = {
    "length_m": 105.0, "width_m": 68.0,
    "goal_height_m": 2.44, "goal_width_m": 7.32, "goal_depth_m": 1.5,
}


def _generate(events, steps, n, *, ctx=None, confidences=None, cfg=CFG):
    Ks, Rs, ts = per_frame_cams(n)
    return generate_auto_anchors(
        events=events,
        steps=steps,
        confidences=confidences or {},
        player_ctx=ctx or FakePlayerContext(),
        per_frame_K=Ks, per_frame_R=Rs, per_frame_t=ts,
        distortion=(0.0, 0.0),
        fps=FPS,
        pitch_cfg=PITCH_CFG,
        cfg=cfg,
    )


def _rolling_scene(n=60):
    K, R, t = broadcast_camera()
    worlds = rolling_worlds((60.0, 30.0), (-0.20, 0.0), n)
    pixels = project_track(worlds, K, R, t)
    return worlds, pixels, steps_from_pixels(pixels, n)


def _joint(player_id, bone, world, uv, conf=0.8):
    return JointSample(
        player_id=player_id, bone=bone,
        world_xyz=tuple(float(x) for x in world),
        uv=(float(uv[0]), float(uv[1])), confidence=conf,
    )


class TestEventMapping:
    def test_touch_event_becomes_player_touch_anchor(self):
        worlds, pixels, steps = _rolling_scene()
        ctx = FakePlayerContext({
            30: [_joint("P004", "r_foot", worlds[30], pixels[30])],
        })
        events = (BallEvent(frame=30, kind="touch", score=0.8,
                            player_id="P004", bone="r_foot"),)
        anchors = _generate(events, steps, 60, ctx=ctx)
        touch = [a for a in anchors if a.state == "player_touch"]
        assert len(touch) == 1
        assert touch[0].frame == 30
        assert touch[0].player_id == "P004"
        assert touch[0].bone == "r_foot"
        assert touch[0].image_xy is not None
        np.testing.assert_allclose(touch[0].image_xy, pixels[30], atol=1e-6)

    def test_fast_outbound_touch_is_tagged_shot(self):
        n = 60
        K, R, t = broadcast_camera()
        # Slow roll in, very fast straight exit (a struck shot).
        worlds = rolling_worlds((60.0, 30.0), (-0.10, 0.0), 31)
        worlds.update(rolling_worlds((57.0, 30.0), (-1.2, 0.0), n - 30,
                                     start_frame=30))
        pixels = project_track(worlds, K, R, t)
        steps = steps_from_pixels(pixels, n)
        ctx = FakePlayerContext({
            30: [_joint("P009", "l_foot", worlds[30], pixels[30])],
        })
        events = (BallEvent(frame=30, kind="touch", score=0.9,
                            player_id="P009", bone="l_foot"),)
        anchors = _generate(events, steps, n, ctx=ctx)
        touch = [a for a in anchors if a.state == "player_touch"][0]
        assert touch.touch_type == "shot"

    def test_contact_gap_gate_rejects_drifted_pose(self):
        worlds, pixels, steps = _rolling_scene()
        # Joint world 2 m off the ball's camera ray -> pose drift too
        # large to trust as a contact.
        drifted = worlds[30] + np.array([0.0, 0.0, 2.0])
        ctx = FakePlayerContext({
            30: [_joint("P004", "r_foot", drifted, pixels[30])],
        })
        events = (BallEvent(frame=30, kind="touch", score=0.8,
                            player_id="P004", bone="r_foot"),)
        anchors = _generate(events, steps, 60, ctx=ctx)
        assert not [a for a in anchors if a.state == "player_touch"]

    def test_bounce_and_goal_events_map_to_states(self):
        worlds, pixels, steps = _rolling_scene()
        events = (
            BallEvent(frame=20, kind="bounce", score=0.7),
            BallEvent(frame=40, kind="goal_impact", score=0.9,
                      goal_element="crossbar"),
        )
        anchors = _generate(events, steps, 60)
        by_frame = {a.frame: a for a in anchors}
        assert by_frame[20].state == "bounce"
        assert by_frame[40].state == "goal_impact"
        assert by_frame[40].goal_element == "crossbar"

    def test_stationary_span_brackets_grounded_anchors(self):
        worlds, pixels, steps = _rolling_scene()
        events = (BallEvent(frame=5, kind="stationary", score=0.9,
                            end_frame=25),)
        anchors = _generate(events, steps, 60)
        grounded_frames = {a.frame for a in anchors if a.state == "grounded"}
        assert 5 in grounded_frames
        assert 25 in grounded_frames

    def test_low_score_events_and_breaks_are_skipped(self):
        worlds, pixels, steps = _rolling_scene()
        events = (
            BallEvent(frame=20, kind="bounce", score=0.05),
            BallEvent(frame=40, kind="velocity_break", score=0.9),
        )
        anchors = _generate(events, steps, 60)
        assert [a for a in anchors if a.frame in (20, 40)] == []


class TestGroundedSampling:
    def test_confident_grounded_run_sampled_at_interval(self):
        n = 120
        K, R, t = broadcast_camera()
        worlds = rolling_worlds((70.0, 30.0), (-0.15, 0.0), n)
        pixels = project_track(worlds, K, R, t)
        steps = steps_from_pixels(pixels, n, p_flight=0.05)
        conf = {i: 0.9 for i in range(n)}
        cfg = AutoAnchorCfg(grounded_interval=25)
        anchors = _generate((), steps, n, confidences=conf, cfg=cfg)
        grounded = [a for a in anchors if a.state == "grounded"]
        assert len(grounded) >= 4
        frames = [a.frame for a in grounded]
        assert all(b - a >= cfg.grounded_interval for a, b in zip(frames, frames[1:]))

    def test_low_confidence_and_flight_frames_not_sampled(self):
        n = 60
        K, R, t = broadcast_camera()
        worlds = rolling_worlds((70.0, 30.0), (-0.15, 0.0), n)
        pixels = project_track(worlds, K, R, t)
        # Flight posterior high -> not grounded evidence.
        steps = steps_from_pixels(pixels, n, p_flight=0.9)
        conf = {i: 0.9 for i in range(n)}
        anchors = _generate((), steps, n, confidences=conf)
        assert anchors == ()

    def test_sampling_suppressed_near_event_anchors(self):
        n = 60
        K, R, t = broadcast_camera()
        worlds = rolling_worlds((70.0, 30.0), (-0.15, 0.0), n)
        pixels = project_track(worlds, K, R, t)
        steps = steps_from_pixels(pixels, n, p_flight=0.05)
        conf = {i: 0.9 for i in range(n)}
        cfg = AutoAnchorCfg(grounded_interval=10, suppress_radius_frames=5)
        events = (BallEvent(frame=30, kind="bounce", score=0.8),)
        anchors = _generate(events, steps, n, confidences=conf, cfg=cfg)
        for a in anchors:
            if a.state == "grounded":
                assert abs(a.frame - 30) > cfg.suppress_radius_frames


    @pytest.mark.unit
    def test_second_pass_frames_never_become_anchors(self):
        """Frames whose observation source is 'second_pass' are excluded from
        anchor candidacy even when their confidence clears every gate."""
        n = 120
        K, R, t = broadcast_camera()
        worlds = rolling_worlds((70.0, 30.0), (-0.15, 0.0), n)
        pixels = project_track(worlds, K, R, t)
        steps = steps_from_pixels(pixels, n, p_flight=0.05)
        conf = {i: 0.9 for i in range(n)}
        cfg = AutoAnchorCfg(grounded_interval=25)
        Ks, Rs, ts = per_frame_cams(n)

        # All frames marked second_pass -> no anchors should be minted.
        sources_sp = {s.frame: "second_pass" for s in steps}
        anchors = generate_auto_anchors(
            events=(),
            steps=steps,
            confidences=conf,
            player_ctx=FakePlayerContext(),
            per_frame_K=Ks, per_frame_R=Rs, per_frame_t=ts,
            distortion=(0.0, 0.0),
            fps=FPS,
            pitch_cfg=PITCH_CFG,
            cfg=cfg,
            sources=sources_sp,
        )
        assert anchors == ()

        # All frames marked detector -> anchors produced normally.
        sources_ok = {s.frame: "detector" for s in steps}
        anchors_ok = generate_auto_anchors(
            events=(),
            steps=steps,
            confidences=conf,
            player_ctx=FakePlayerContext(),
            per_frame_K=Ks, per_frame_R=Rs, per_frame_t=ts,
            distortion=(0.0, 0.0),
            fps=FPS,
            pitch_cfg=PITCH_CFG,
            cfg=cfg,
            sources=sources_ok,
        )
        assert len(anchors_ok) > 0

        # Backward compatibility: no sources kwarg -> same as detector path.
        anchors_compat = generate_auto_anchors(
            events=(),
            steps=steps,
            confidences=conf,
            player_ctx=FakePlayerContext(),
            per_frame_K=Ks, per_frame_R=Rs, per_frame_t=ts,
            distortion=(0.0, 0.0),
            fps=FPS,
            pitch_cfg=PITCH_CFG,
            cfg=cfg,
        )
        assert len(anchors_compat) > 0


class TestValidationGates:
    def test_unreachable_neighbor_dropped(self):
        n = 60
        K, R, t = broadcast_camera()
        worlds = rolling_worlds((60.0, 30.0), (-0.20, 0.0), n)
        pixels = project_track(worlds, K, R, t)
        # Corrupt one frame's pixel so its ground projection teleports
        # ~25 m away: an impossible 1-frame move.
        far_uv = project_world_to_image(
            K, R, t, (0.0, 0.0), np.array([[35.0, 50.0, 0.11]])
        )[0]
        events = (
            BallEvent(frame=20, kind="bounce", score=0.7),
            BallEvent(frame=21, kind="bounce", score=0.6),
        )
        pixels[21] = (float(far_uv[0]), float(far_uv[1]))
        steps = steps_from_pixels(pixels, n)
        anchors = _generate(events, steps, n)
        frames = {a.frame for a in anchors}
        assert 20 in frames
        assert 21 not in frames

    def test_off_pitch_candidate_dropped(self):
        n = 60
        K, R, t = broadcast_camera()
        worlds = rolling_worlds((60.0, 30.0), (-0.20, 0.0), n)
        pixels = project_track(worlds, K, R, t)
        off_uv = project_world_to_image(
            K, R, t, (0.0, 0.0), np.array([[52.5, -20.0, 0.11]])
        )[0]
        pixels[40] = (float(off_uv[0]), float(off_uv[1]))
        events = (BallEvent(frame=40, kind="bounce", score=0.8),)
        steps = steps_from_pixels(pixels, n)
        anchors = _generate(events, steps, n)
        assert not [a for a in anchors if a.frame == 40]


class TestMergeAndSidecar:
    def _anchor(self, frame, state="grounded", **kw):
        return BallAnchor(frame=frame, image_xy=(100.0, 100.0),
                          state=state, **kw)

    def test_manual_wins_and_suppresses_nearby_auto(self):
        manual = {30: self._anchor(30, state="kick")}
        auto = {
            28: self._anchor(28),   # within radius -> dropped
            30: self._anchor(30),   # same frame -> manual wins
            50: self._anchor(50),   # kept
        }
        merged = merge_anchors(manual, auto, suppress_radius_frames=3)
        assert merged[30].state == "kick"
        assert 28 not in merged
        assert 50 in merged

    def test_sidecar_roundtrip_via_schema(self, tmp_path):
        worlds, pixels, steps = _rolling_scene()
        ctx = FakePlayerContext({
            30: [_joint("P004", "r_foot", worlds[30], pixels[30])],
        })
        events = (
            BallEvent(frame=30, kind="touch", score=0.8,
                      player_id="P004", bone="r_foot"),
            BallEvent(frame=45, kind="bounce", score=0.7),
        )
        anchors = _generate(events, steps, 60, ctx=ctx)
        aset = BallAnchorSet(
            clip_id="shotX", image_size=(1920, 1080), anchors=anchors,
        )
        path = auto_anchor_path(tmp_path, "shotX")
        aset.save(path)
        loaded = BallAnchorSet.load(path)
        assert loaded.anchors == anchors
        assert path.name == "shotX_ball_anchors_auto.json"
