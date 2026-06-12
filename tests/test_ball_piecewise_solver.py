"""Piecewise-physical ball solver: continuity, accuracy, physics gates.

Synthetic scenes with analytic ground truth. The bar throughout is the
10 cm goal wherever the trajectory is bracketed by nodes, and exact
position continuity at every node.
"""
from __future__ import annotations

import numpy as np

from src.utils.ball_physics import (
    eval_parabola,
    parabola_end_velocity,
    two_knot_arc,
)
from src.utils.ball_piecewise_solver import (
    SolverCfg,
    TrajectoryNode,
    solve_piecewise,
)
from tests.fixtures.ball_synthetic import (
    FPS,
    broadcast_camera,
    per_frame_cams,
    project_track,
    rolling_worlds,
    steps_from_pixels,
)

CFG = SolverCfg()


def _node(frame, world, state, conf=1.0):
    return TrajectoryNode(
        frame=frame,
        world_xyz=(float(world[0]), float(world[1]), float(world[2])),
        state=state,
        confidence=conf,
    )


def _solve(nodes, steps, n, *, confidences=None, split_hints=(),
           p_flight_default=None, cfg=CFG):
    Ks, Rs, ts = per_frame_cams(n)
    return solve_piecewise(
        nodes=nodes,
        steps=steps,
        confidences=confidences or {},
        per_frame_K=Ks, per_frame_R=Rs, per_frame_t=ts,
        distortion=(0.0, 0.0),
        fps=FPS,
        n_frames=n,
        pitch_length_m=105.0, pitch_width_m=68.0,
        split_hints=tuple(split_hints),
        cfg=cfg,
    )


def _arc_worlds(a, b, fa, fb):
    """Per-frame analytic gravity arc through (fa, a) and (fb, b)."""
    T = (fb - fa) / FPS
    p0, v0 = two_knot_arc(np.asarray(a), np.asarray(b), T)
    times = np.array([(f - fa) / FPS for f in range(fa, fb + 1)])
    pts = eval_parabola(p0, v0, times)
    return {fa + i: pts[i] for i in range(len(pts))}


def _rms_error(result, truth, frames):
    errs = []
    for f in frames:
        got = result.world_by_frame.get(f)
        assert got is not None, f"frame {f} missing from solve"
        errs.append(np.linalg.norm(np.asarray(got[0]) - truth[f]))
    return float(np.sqrt(np.mean(np.square(errs))))


class TestBallisticSegments:
    def test_two_knot_arc_without_observations(self):
        # Kick at f10, bounce at f35 — no detections in between (the
        # detector lost the ball). Gravity + the two knots still fully
        # determine the arc.
        n = 50
        a = np.array([50.0, 30.0, 0.11])
        b = np.array([42.0, 33.0, 0.11])
        truth = _arc_worlds(a, b, 10, 35)
        nodes = [_node(10, a, "kick"), _node(35, b, "bounce")]
        steps = steps_from_pixels({}, n)
        result = _solve(nodes, steps, n)
        rms = _rms_error(result, truth, range(10, 36))
        assert rms <= 0.10
        assert result.state_by_frame[20] == "flight"
        assert len(result.flight_segments) == 1

    def test_arc_with_noisy_observations(self):
        n = 50
        K, R, t = broadcast_camera()
        a = np.array([50.0, 30.0, 0.11])
        b = np.array([42.0, 33.0, 0.11])
        truth = _arc_worlds(a, b, 10, 35)
        rng = np.random.default_rng(7)
        pixels = project_track(truth, K, R, t)
        noisy = {
            f: (uv[0] + rng.normal(0, 1.0), uv[1] + rng.normal(0, 1.0))
            for f, uv in pixels.items()
        }
        nodes = [_node(10, a, "kick"), _node(35, b, "bounce")]
        steps = steps_from_pixels(noisy, n, p_flight=0.9)
        result = _solve(nodes, steps, n)
        rms = _rms_error(result, truth, range(10, 36))
        assert rms <= 0.10
        seg = result.flight_segments[0]
        assert seg.fit_residual_px <= CFG.flight_max_residual_px

    def test_goal_impact_pinned_exactly(self):
        n = 40
        a = np.array([10.0, 34.0, 0.11])
        g_pt = np.array([0.0, 34.0, 2.44])
        truth = _arc_worlds(a, g_pt, 5, 30)
        nodes = [_node(5, a, "kick"), _node(30, g_pt, "goal_impact")]
        steps = steps_from_pixels({}, n)
        result = _solve(nodes, steps, n)
        got, _ = result.world_by_frame[30]
        np.testing.assert_allclose(got, g_pt, atol=1e-9)
        assert result.state_by_frame[20] == "flight"
        rms = _rms_error(result, truth, range(5, 31))
        assert rms <= 0.10


class TestRollingSegments:
    def test_decelerating_roll_within_10cm(self):
        n = 60
        K, R, t = broadcast_camera()
        # True roll: constant deceleration 2 m/s² against +x motion.
        v0 = np.array([3.0, 0.5])
        a_xy = np.array([45.0, 30.0])
        decel = 2.0
        truth = {}
        for f in range(5, 56):
            ts = (f - 5) / FPS
            direction = v0 / np.linalg.norm(v0)
            xy = a_xy + v0 * ts - 0.5 * decel * direction * ts**2
            truth[f] = np.array([xy[0], xy[1], 0.11])
        pixels = project_track(truth, K, R, t)
        nodes = [_node(5, truth[5], "grounded"), _node(55, truth[55], "grounded")]
        steps = steps_from_pixels(pixels, n)
        result = _solve(nodes, steps, n)
        rms = _rms_error(result, truth, range(5, 56))
        assert rms <= 0.10
        assert all(
            result.state_by_frame[f] == "grounded" for f in range(5, 56)
        )
        assert result.flight_segments == ()

    def test_fast_ground_pass_stays_grounded_despite_flight_posterior(self):
        # A driven ground pass: IMM flags it as flight (pixel velocity),
        # but the rolling fit explains the observations -> grounded.
        n = 40
        K, R, t = broadcast_camera()
        truth = rolling_worlds((55.0, 30.0), (-0.45, 0.0), 31, start_frame=5)
        pixels = project_track(truth, K, R, t)
        nodes = [_node(5, truth[5], "grounded"), _node(35, truth[35], "grounded")]
        steps = steps_from_pixels(pixels, n, p_flight=0.85)
        result = _solve(nodes, steps, n)
        assert result.flight_segments == ()
        rms = _rms_error(result, truth, range(5, 36))
        assert rms <= 0.10


class TestBouncePhysics:
    def _chain(self, e_out):
        """kick -> bounce -> elevated touch 0.4 s later (still rising)."""
        fa, fn, fb = 5, 25, 35
        a = np.array([50.0, 30.0, 0.11])
        bounce = np.array([44.0, 30.0, 0.11])
        t1 = (fn - fa) / FPS
        _, v_in0 = two_knot_arc(a, bounce, t1)
        v_in_end = parabola_end_velocity(v_in0, t1)
        v_out = np.array([v_in_end[0], v_in_end[1], -e_out * v_in_end[2]])
        t2 = (fb - fn) / FPS
        b = bounce + v_out * t2 + 0.5 * np.array([0, 0, -9.81]) * t2**2
        truth = _arc_worlds(a, bounce, fa, fn)
        truth.update(_arc_worlds(bounce, b, fn, fb))
        nodes = [
            _node(fa, a, "kick"),
            _node(fn, bounce, "bounce"),
            _node(fb, b, "player_touch"),
        ]
        return nodes, truth, (fa, fn, fb)

    def test_plausible_restitution_recorded_not_flagged(self):
        nodes, truth, (fa, fn, fb) = self._chain(0.7)
        n = 60
        result = _solve(nodes, steps_from_pixels({}, n), n)
        rms = _rms_error(result, truth, range(fa, fb + 1))
        assert rms <= 0.10
        bounces = result.diagnostics["bounces"]
        assert len(bounces) == 1
        assert bounces[0]["frame"] == fn
        assert abs(bounces[0]["restitution"] - 0.7) < 0.05
        assert not bounces[0]["flagged"]

    def test_impossible_restitution_flagged(self):
        nodes, truth, (fa, fn, fb) = self._chain(1.25)
        n = 60
        result = _solve(nodes, steps_from_pixels({}, n), n)
        bounces = result.diagnostics["bounces"]
        assert len(bounces) == 1
        assert bounces[0]["flagged"]
        assert bounces[0]["restitution"] > 1.0


class TestSplitAndRetry:
    def test_missed_bounce_recovered_from_split_hint(self):
        # True path: two arcs with a ground bounce at f25 that event
        # detection only surfaced as a velocity break (no anchor). The
        # single-arc fit fails its residual gate; the solver splits at
        # the hint and refits both halves.
        n = 60
        K, R, t = broadcast_camera()
        fa, fm, fb = 5, 25, 45
        a = np.array([52.0, 30.0, 0.11])
        mid = np.array([46.0, 31.0, 0.11])
        b = np.array([40.0, 32.0, 0.11])
        truth = _arc_worlds(a, mid, fa, fm)
        truth.update(_arc_worlds(mid, b, fm, fb))
        pixels = project_track(truth, K, R, t)
        nodes = [_node(fa, a, "kick"), _node(fb, b, "bounce")]
        steps = steps_from_pixels(pixels, n, p_flight=0.9)
        result = _solve(
            nodes, steps, n, split_hints=((fm, 0.8),),
        )
        rms = _rms_error(result, truth, range(fa, fb + 1))
        assert rms <= 0.15
        assert len(result.flight_segments) == 2
        assert result.diagnostics["splits"] >= 1
        for seg in result.flight_segments:
            assert seg.fit_residual_px <= CFG.flight_max_residual_px

    def test_unfixable_span_flagged_never_silent(self):
        # Same two-arc truth but NO split hint: the solver must accept
        # its best fit AND flag the span as underconstrained.
        n = 60
        K, R, t = broadcast_camera()
        fa, fm, fb = 5, 25, 45
        a = np.array([52.0, 30.0, 0.11])
        mid = np.array([46.0, 31.0, 0.11])
        b = np.array([40.0, 32.0, 0.11])
        truth = _arc_worlds(a, mid, fa, fm)
        truth.update(_arc_worlds(mid, b, fm, fb))
        pixels = project_track(truth, K, R, t)
        nodes = [_node(fa, a, "kick"), _node(fb, b, "bounce")]
        steps = steps_from_pixels(pixels, n, p_flight=0.9)
        result = _solve(nodes, steps, n)
        flagged = result.diagnostics["underconstrained_spans"]
        assert any(s["start"] == fa and s["end"] == fb for s in flagged)


class TestContinuityAndOpenSpans:
    def test_exact_continuity_at_every_node(self):
        n = 70
        a = np.array([52.0, 30.0, 0.11])
        m1 = np.array([46.0, 31.0, 0.11])
        m2 = np.array([40.0, 30.0, 0.11])
        truth = _arc_worlds(a, m1, 5, 25)
        truth.update(_arc_worlds(m1, m2, 25, 45))
        # then a roll out to f65
        roll = rolling_worlds((40.0, 30.0), (-0.1, 0.0), 21, start_frame=45)
        truth.update(roll)
        K, R, t = broadcast_camera()
        pixels = project_track(truth, K, R, t)
        nodes = [
            _node(5, a, "kick"),
            _node(25, m1, "bounce"),
            _node(45, m2, "bounce"),
            _node(65, truth[65], "grounded"),
        ]
        steps = steps_from_pixels(pixels, n, p_flight=0.9)
        result = _solve(nodes, steps, n)
        for node in nodes:
            got, _ = result.world_by_frame[node.frame]
            np.testing.assert_allclose(got, node.world_xyz, atol=1e-9)
        # No teleports anywhere inside the covered range.
        prev = None
        for f in range(5, 66):
            cur = result.world_by_frame.get(f)
            assert cur is not None
            if prev is not None:
                assert np.linalg.norm(np.asarray(cur[0]) - prev) < 1.2
            prev = np.asarray(cur[0])

    def test_leading_grounded_span_raycast_and_missing(self):
        n = 40
        K, R, t = broadcast_camera()
        truth = rolling_worlds((55.0, 30.0), (-0.2, 0.0), 20)
        pixels = project_track(truth, K, R, t)
        # detections only on frames 0..14; nodes start at f20.
        pixels = {f: uv for f, uv in pixels.items() if f <= 14}
        roll2 = rolling_worlds((51.0, 30.0), (-0.2, 0.0), 20, start_frame=20)
        nodes = [
            _node(20, roll2[20], "grounded"),
            _node(39, roll2[39], "grounded"),
        ]
        steps = steps_from_pixels(pixels, n)
        conf = {f: 0.8 for f in pixels}
        result = _solve(nodes, steps, n, confidences=conf)
        # Leading span: ray-cast ground positions within 10 cm of truth.
        for f in range(0, 15):
            got = result.world_by_frame.get(f)
            assert got is not None
            assert np.linalg.norm(np.asarray(got[0]) - truth[f]) <= 0.10
        # Frames with no detection and no segment: missing.
        for f in range(15, 20):
            assert result.state_by_frame[f] == "missing"

    def test_no_nodes_at_all_still_produces_grounded_track(self):
        n = 30
        K, R, t = broadcast_camera()
        truth = rolling_worlds((55.0, 30.0), (-0.2, 0.0), n)
        pixels = project_track(truth, K, R, t)
        steps = steps_from_pixels(pixels, n)
        conf = {f: 0.8 for f in pixels}
        result = _solve([], steps, n, confidences=conf)
        for f in range(n):
            got = result.world_by_frame.get(f)
            assert got is not None
            assert np.linalg.norm(np.asarray(got[0]) - truth[f]) <= 0.10

class TestMagnusArcConsistency:
    def test_end_velocity_and_eval_share_magnus_dynamics(self):
        # A spinning arc's terminal velocity must come from the same
        # dynamics as its positions: finite-differencing eval() near the
        # end frame has to agree with end_velocity(), and both must
        # differ from the no-spin formula for a strong sideways Magnus.
        from src.utils.ball_piecewise_solver import _Arc
        from src.utils.ball_physics import parabola_end_velocity

        fps = 25.0
        arc = _Arc(
            fa=0, fb=25,
            p0=np.array([50.0, 30.0, 0.11]),
            v0=np.array([15.0, 0.0, 5.0]),
            residual_px=0.5, n_obs=10,
            omega_world=np.array([0.0, 0.0, 40.0]),
            drag_k_over_m=0.01,
        )
        v_end = arc.end_velocity(fps)
        eps_frames = 0.01
        p_hi = arc.eval(25, fps)
        p_lo = arc.eval(25 - eps_frames, fps)
        v_fd = (p_hi - p_lo) / (eps_frames / fps)
        np.testing.assert_allclose(v_end, v_fd, atol=0.05)
        plain = parabola_end_velocity(arc.v0, 1.0)
        assert np.linalg.norm(v_end - plain) > 0.5, (
            "Magnus end velocity should differ from the no-spin formula"
        )
