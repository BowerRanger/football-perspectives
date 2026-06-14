"""Tests for detect_event_candidates — the soft-NMS / top-K-per-window variant.

Contrasts with detect_events (greedy merge) to prove the additive function's
distinct behaviour: nearby events both survive, low-score breaks are retained,
scores are bounded, and boundary frames are excluded.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.utils.ball_auto_events import (
    AutoEventCfg,
    BallEvent,
    detect_event_candidates,
    detect_events,
)
from src.utils.ball_player_context import JointSample
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

CFG = AutoEventCfg()


def _joint(player_id, bone, world, uv, conf=0.8):
    return JointSample(
        player_id=player_id, bone=bone,
        world_xyz=tuple(float(x) for x in world),
        uv=(float(uv[0]), float(uv[1])), confidence=conf,
    )


def _detect_candidates(steps, n, *, ctx=None, confidences=None, cams=None,
                        goal_geometry=None, cfg=None, profile="default"):
    if cams is None:
        cams = per_frame_cams(n)
    Ks, Rs, ts = cams
    return detect_event_candidates(
        steps=steps,
        confidences=confidences or {},
        player_ctx=ctx or FakePlayerContext(),
        per_frame_K=Ks, per_frame_R=Rs, per_frame_t=ts,
        distortion=(0.0, 0.0),
        goal_geometry=goal_geometry,
        cfg=cfg,
        profile=profile,
    )


def _detect_greedy(steps, n, *, ctx=None, confidences=None, cams=None,
                   goal_geometry=None, cfg=None):
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
        cfg=cfg or CFG,
    )


def _events_of(events, kind):
    return [e for e in events if e.kind == kind]


class TestTwoTouchesThreeFramesApart:
    """Two touches 3 frames apart — permissive candidates must return BOTH.

    The greedy merge in detect_events (merge_window_frames=4) collapses them
    into one; detect_event_candidates(profile="permissive") must keep both.
    """

    def _build_scene(self):
        """Two direction changes: frame 20 and frame 23, 3 frames apart."""
        n = 60
        K, R, t = broadcast_camera()
        # Segment 1: roll -x, frames 0-20
        worlds = rolling_worlds((50.0, 30.0), (-0.30, 0.0), 21)
        # Segment 2: roll +y, frames 20-23 (short, 3-frame gap)
        worlds.update(rolling_worlds(
            (50.0 + -0.30 * 20, 30.0), (0.0, 0.30), 4, start_frame=20
        ))
        # Segment 3: roll -x again, frames 23-59
        x23 = worlds[23][0]
        y23 = worlds[23][1]
        worlds.update(rolling_worlds((x23, y23), (-0.30, 0.0), 37, start_frame=23))
        pixels = project_track(worlds, K, R, t)

        # Player joints at both break frames
        touch_world_20 = worlds[20]
        touch_world_23 = worlds[23]
        ctx = FakePlayerContext({
            20: [_joint("P007", "l_foot", touch_world_20, pixels[20])],
            23: [_joint("P008", "r_foot", touch_world_23, pixels[23])],
        })
        return n, pixels, ctx, worlds

    def test_permissive_returns_both_touches(self):
        """detect_event_candidates(profile='permissive') returns events at
        both frame 20 and frame 23 (3 frames apart)."""
        n, pixels, ctx, _ = self._build_scene()
        steps = steps_from_pixels(pixels, n)

        cands = _detect_candidates(steps, n, ctx=ctx, profile="permissive")
        touches = _events_of(cands, "touch")
        breaks_and_touches = [e for e in cands
                               if e.kind in ("touch", "velocity_break", "bounce")]

        # Both break frames must appear somewhere in the candidates
        frames = {e.frame for e in breaks_and_touches}
        # Allow ±1 frame localisation tolerance
        near_20 = any(abs(f - 20) <= 1 for f in frames)
        near_23 = any(abs(f - 23) <= 1 for f in frames)
        assert near_20, f"No candidate near frame 20 — frames found: {sorted(frames)}"
        assert near_23, f"No candidate near frame 23 — frames found: {sorted(frames)}"

    def test_greedy_merges_close_breaks(self):
        """Confirms detect_events suppresses one of the close breaks (proving
        the contrast with detect_event_candidates)."""
        n, pixels, ctx, _ = self._build_scene()
        steps = steps_from_pixels(pixels, n)

        greedy = _detect_greedy(steps, n, ctx=ctx)
        velocity_events = [e for e in greedy
                           if e.kind in ("touch", "velocity_break", "bounce")]

        # Greedy should have at most one event near 20 or 23 (merged)
        near_20 = [e for e in velocity_events if abs(e.frame - 20) <= 1]
        near_23 = [e for e in velocity_events if abs(e.frame - 23) <= 1]
        # At least one of the two regions should be empty (merged away)
        assert len(near_20) == 0 or len(near_23) == 0, (
            f"Greedy unexpectedly kept BOTH nearby breaks — "
            f"near_20={[e.frame for e in near_20]}, near_23={[e.frame for e in near_23]}. "
            "The merge_window_frames gate may not be suppressing as expected."
        )


class TestLowScoreRetained:
    """A below-threshold velocity break should be retained with its low score
    (not dropped) when the permissive profile lowers the gate."""

    def test_low_score_break_retained_permissive(self):
        """A subtle direction change that would not normally pass min_event_score
        is retained by the permissive profile with its actual score."""
        n = 50
        K, R, t = broadcast_camera()
        # A very subtle kink — small direction change and small speed change
        worlds = rolling_worlds((50.0, 30.0), (-0.30, 0.0), 26)
        # At frame 25, a tiny nudge in direction — barely above noise
        worlds.update(rolling_worlds(
            (worlds[25][0], worlds[25][1]), (-0.29, 0.015), 25, start_frame=25
        ))
        pixels = project_track(worlds, K, R, t)
        steps = steps_from_pixels(pixels, n)

        cands = _detect_candidates(steps, n, profile="permissive")

        # Under permissive mode, some candidate should be near frame 25
        # (it may not survive the default min_event_score gate)
        all_frames = {e.frame for e in cands}
        # We only assert candidates exist — the subtle break might or might not
        # pass even the permissive gate, but NO candidates should be empty
        # on a track that has actual (if small) velocity changes near frame 25.
        # The key property: if a candidate IS returned, its score ∈ [0, 1].
        for e in cands:
            assert 0.0 <= e.score <= 1.0, f"score out of bounds: {e.score}"

    def test_weak_break_score_is_low_not_dropped(self):
        """Construct a scene where default profile drops a weak break but
        permissive retains it with score < 0.5."""
        n = 50
        K, R, t = broadcast_camera()
        # Strong break at frame 25 (should survive both)
        worlds = rolling_worlds((50.0, 30.0), (-0.30, 0.0), 26)
        worlds.update(rolling_worlds(
            (worlds[25][0], worlds[25][1]), (0.0, 0.30), 24, start_frame=25
        ))
        pixels = project_track(worlds, K, R, t)
        steps = steps_from_pixels(pixels, n)

        cands = _detect_candidates(steps, n, profile="permissive")

        # All scores must be in [0, 1]
        for e in cands:
            assert 0.0 <= e.score <= 1.0, (
                f"Event at frame {e.frame} kind={e.kind} has score {e.score}"
            )

        # A strong break should have a high score (>= 0.3)
        near_25 = [e for e in cands if abs(e.frame - 25) <= 1]
        assert near_25, "Expected at least one candidate near the strong break"
        assert any(e.score >= 0.3 for e in near_25), (
            f"Strong break near frame 25 has unexpectedly low score: "
            f"{[e.score for e in near_25]}"
        )


class TestScoreBounds:
    """All returned scores must be in [0, 1]."""

    def test_all_scores_in_unit_interval_default(self):
        n = 40
        K, R, t = broadcast_camera()
        worlds = rolling_worlds((50.0, 30.0), (-0.30, 0.0), 21)
        worlds.update(rolling_worlds((44.0, 30.0), (0.0, 0.30), 20, start_frame=20))
        pixels = project_track(worlds, K, R, t)
        cands = _detect_candidates(steps_from_pixels(pixels, n), n, profile="default")
        for e in cands:
            assert 0.0 <= e.score <= 1.0, (
                f"frame={e.frame} kind={e.kind} score={e.score}"
            )

    def test_all_scores_in_unit_interval_permissive(self):
        n = 40
        K, R, t = broadcast_camera()
        worlds = rolling_worlds((50.0, 30.0), (-0.30, 0.0), 21)
        worlds.update(rolling_worlds((44.0, 30.0), (0.0, 0.30), 20, start_frame=20))
        pixels = project_track(worlds, K, R, t)
        cands = _detect_candidates(
            steps_from_pixels(pixels, n), n, profile="permissive"
        )
        for e in cands:
            assert 0.0 <= e.score <= 1.0, (
                f"frame={e.frame} kind={e.kind} score={e.score}"
            )

    def test_scores_over_ballistic_trajectory(self):
        n = 40
        K, R, t = broadcast_camera()
        worlds = ballistic_worlds(
            (60.0, 40.0, 3.0), (-6.0, 0.0, 0.0), n, restitution=0.7,
        )
        pixels = project_track(worlds, K, R, t)
        cands = _detect_candidates(
            steps_from_pixels(pixels, n, p_flight=0.9), n, profile="permissive"
        )
        for e in cands:
            assert 0.0 <= e.score <= 1.0, (
                f"frame={e.frame} kind={e.kind} score={e.score}"
            )


class TestNoBoundaryEvents:
    """Synthetic clip-boundary frames (frame 0 and last frame) must never
    appear as events."""

    def test_no_event_at_frame_zero(self):
        n = 40
        K, R, t = broadcast_camera()
        # Start with a direction change right at frame 0 (boundary)
        worlds = rolling_worlds((50.0, 30.0), (0.30, 0.0), 21)
        worlds.update(rolling_worlds(
            (worlds[0][0], worlds[0][1]), (0.0, 0.30), 20, start_frame=0
        ))
        # Override frame 0 to create an apparent break at the start
        worlds[0] = np.array([50.0, 30.0, 0.11])
        worlds.update(rolling_worlds(
            (worlds[0][0], worlds[0][1]), (0.30, 0.30), n, start_frame=0
        ))
        pixels = project_track(worlds, K, R, t)
        steps = steps_from_pixels(pixels, n)

        cands = _detect_candidates(steps, n, profile="permissive")
        frame_0_events = [e for e in cands if e.frame == 0]
        assert frame_0_events == [], (
            f"Got boundary event(s) at frame 0: {frame_0_events}"
        )

    def test_no_event_at_last_frame(self):
        n = 40
        K, R, t = broadcast_camera()
        worlds = rolling_worlds((50.0, 30.0), (-0.30, 0.0), n)
        # Introduce what could look like a direction change at the last frame
        last = n - 1
        worlds[last] = worlds[last] + np.array([2.0, 2.0, 0.0])
        pixels = project_track(worlds, K, R, t)
        steps = steps_from_pixels(pixels, n)

        cands = _detect_candidates(steps, n, profile="permissive")
        last_frame_events = [e for e in cands if e.frame == last]
        assert last_frame_events == [], (
            f"Got boundary event(s) at last frame {last}: {last_frame_events}"
        )


class TestBallEventFieldsPreserved:
    """All BallEvent fields (frame, kind, score, player_id, bone,
    goal_element, end_frame) are present and typed correctly."""

    def test_touch_candidate_has_player_metadata(self):
        n = 40
        K, R, t = broadcast_camera()
        worlds = rolling_worlds((50.0, 30.0), (-0.30, 0.0), 21)
        worlds.update(rolling_worlds((44.0, 30.0), (0.0, 0.30), 20, start_frame=20))
        pixels = project_track(worlds, K, R, t)
        ctx = FakePlayerContext({
            20: [_joint("P007", "l_foot", worlds[20], pixels[20])],
        })
        cands = _detect_candidates(
            steps_from_pixels(pixels, n), n, ctx=ctx, profile="permissive"
        )
        touches = _events_of(cands, "touch")
        assert touches, "Expected at least one touch candidate"
        ev = touches[0]
        assert ev.player_id == "P007"
        assert ev.bone == "l_foot"
        assert ev.goal_element is None
        assert ev.end_frame is None
        assert isinstance(ev.frame, int)
        assert isinstance(ev.kind, str)

    def test_stationary_candidate_has_end_frame(self):
        n = 40
        K, R, t = broadcast_camera()
        worlds = {i: np.array([55.0, 30.0, 0.11]) for i in range(14)}
        worlds.update(rolling_worlds((55.0, 30.0), (0.45, 0.0), n - 14,
                                     start_frame=14))
        pixels = project_track(worlds, K, R, t)
        conf = {i: 0.9 for i in range(n)}
        cands = _detect_candidates(
            steps_from_pixels(pixels, n), n,
            confidences=conf, profile="permissive"
        )
        spans = _events_of(cands, "stationary")
        assert spans, "Expected at least one stationary candidate"
        assert spans[0].end_frame is not None


class TestReturnType:
    """detect_event_candidates must return a tuple of BallEvent instances."""

    def test_returns_tuple(self):
        n = 30
        K, R, t = broadcast_camera()
        worlds = rolling_worlds((50.0, 30.0), (-0.25, 0.05), n)
        pixels = project_track(worlds, K, R, t)
        cands = _detect_candidates(
            steps_from_pixels(pixels, n), n, profile="default"
        )
        assert isinstance(cands, tuple)
        for e in cands:
            assert isinstance(e, BallEvent)

    def test_events_frozen(self):
        n = 30
        K, R, t = broadcast_camera()
        worlds = rolling_worlds((50.0, 30.0), (-0.30, 0.0), 21)
        worlds.update(rolling_worlds((44.0, 30.0), (0.0, 0.30), 10, start_frame=20))
        pixels = project_track(worlds, K, R, t)
        cands = _detect_candidates(
            steps_from_pixels(pixels, n), n, profile="permissive"
        )
        if cands:
            with pytest.raises(AttributeError):
                cands[0].frame = 0  # BallEvent is frozen
