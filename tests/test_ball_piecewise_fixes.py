"""world_fixes consumption in the piecewise ball solver.

Synthetic single-camera arc with sparse noisy detections between two
grounded runs. Exact 3D fixes (truth points, weight 30.0) must improve
the depth estimate compared to no fixes, and the diagnostics must report
fixes_used counts correctly.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.utils.ball_physics import eval_parabola, two_knot_arc
from src.utils.ball_piecewise_solver import (
    SolverCfg,
    TrajectoryNode,
    solve_piecewise,
)
from tests.fixtures.ball_synthetic import (
    FPS,
    broadcast_camera,
    ballistic_worlds,
    per_frame_cams,
    project_track,
    rolling_worlds,
    steps_from_pixels,
)

G = np.array([0.0, 0.0, -9.81])


def _node(frame, world, state, conf=1.0):
    return TrajectoryNode(
        frame=frame,
        world_xyz=(float(world[0]), float(world[1]), float(world[2])),
        state=state,
        confidence=conf,
    )


def _solve(nodes, steps, n, *, world_fixes=None, cfg=None):
    Ks, Rs, ts = per_frame_cams(n)
    return solve_piecewise(
        nodes=nodes,
        steps=steps,
        confidences={},
        per_frame_K=Ks, per_frame_R=Rs, per_frame_t=ts,
        distortion=(0.0, 0.0),
        fps=FPS,
        n_frames=n,
        pitch_length_m=105.0, pitch_width_m=68.0,
        split_hints=(),
        world_fixes=world_fixes,
        cfg=cfg or SolverCfg(),
    )


def _arc_worlds(a, b, fa, fb):
    T = (fb - fa) / FPS
    p0, v0 = two_knot_arc(np.asarray(a), np.asarray(b), T)
    times = np.array([(f - fa) / FPS for f in range(fa, fb + 1)])
    pts = eval_parabola(p0, v0, times)
    return {fa + i: pts[i] for i in range(len(pts))}


@pytest.mark.unit
def test_world_fixes_improve_depth_estimate():
    """Sparse noisy monocular obs leave depth ambiguous; 3 exact 3D fixes
    must reduce the mean 3D error over the flight span."""
    n = 70
    K, R, t = broadcast_camera()
    rng = np.random.default_rng(42)

    # Ground roll, then flight, then ground roll.
    fa, fb = 10, 50   # flight span
    a = np.array([52.0, 30.0, 0.11])
    b = np.array([42.0, 33.0, 0.11])
    truth_flight = _arc_worlds(a, b, fa, fb)
    truth_roll_pre = rolling_worlds((55.0, 30.0), (-0.3, 0.0), fa, z=0.11)
    truth_roll_post = rolling_worlds(
        (float(b[0]), float(b[1])), (-0.2, 0.0), n - fb,
        start_frame=fb, z=0.11,
    )
    truth = {**truth_roll_pre, **truth_flight, **truth_roll_post}

    # Sparse (every 5th frame) noisy detections during the flight only.
    all_pixels = project_track(truth, K, R, t)
    sparse_frames = set(range(fa, fb + 1, 5))
    noisy_pixels = {}
    for f, (u, v) in all_pixels.items():
        if f in sparse_frames:
            du, dv = rng.normal(0.0, 2.0, 2)
            noisy_pixels[f] = (u + du, v + dv)

    steps = steps_from_pixels(noisy_pixels, n, p_flight={f: 0.9 for f in range(fa, fb + 1)})
    nodes = [
        _node(fa, a, "kick"),
        _node(fb, b, "bounce"),
    ]

    # 3 exact 3D fixes from truth at interior frames.
    fix_frames = [18, 30, 42]
    fixes_map = {f: (np.asarray(truth_flight[f]), 30.0) for f in fix_frames}

    result_no_fix = _solve(nodes, steps, n, world_fixes=None)
    result_with_fix = _solve(nodes, steps, n, world_fixes=fixes_map)

    # (a) with-fixes mean 3D error <= without-fixes error.
    flight_frames = list(range(fa + 1, fb))
    def _mean_err(result):
        errs = []
        for f in flight_frames:
            got = result.world_by_frame.get(f)
            if got is not None:
                errs.append(np.linalg.norm(np.asarray(got[0]) - truth_flight[f]))
        return float(np.mean(errs)) if errs else float("inf")

    err_no = _mean_err(result_no_fix)
    err_with = _mean_err(result_with_fix)
    assert err_with <= err_no + 1e-9, (
        f"fixes did not help: err_with={err_with:.3f} > err_no={err_no:.3f}"
    )

    # (b) diagnostics: at least one segment has fixes_used >= 1 with fixes,
    #     and all segments have fixes_used == 0 without.
    segs_with = result_with_fix.diagnostics["segments"]
    segs_no = result_no_fix.diagnostics["segments"]

    assert any(s.get("fixes_used", 0) >= 1 for s in segs_with), (
        "no segment reported fixes_used >= 1 in the with-fixes run"
    )
    assert all(s.get("fixes_used", 0) == 0 for s in segs_no), (
        "some segment reported fixes_used > 0 in the no-fix run"
    )

    # (c) deterministic across two identical calls.
    result_with_fix2 = _solve(nodes, steps, n, world_fixes=fixes_map)
    for f in flight_frames:
        w1 = result_with_fix.world_by_frame.get(f)
        w2 = result_with_fix2.world_by_frame.get(f)
        if w1 is not None and w2 is not None:
            assert np.allclose(w1[0], w2[0], atol=1e-10), (
                f"frame {f} not deterministic: {w1[0]} vs {w2[0]}"
            )


@pytest.mark.unit
def test_no_fixes_fixes_used_always_zero():
    """Calling solve_piecewise without world_fixes must produce
    fixes_used == 0 on every segment (backward-compatible default)."""
    n = 40
    a = np.array([52.0, 30.0, 0.11])
    b = np.array([44.0, 32.0, 0.11])
    nodes = [_node(5, a, "kick"), _node(35, b, "bounce")]
    steps = steps_from_pixels({}, n)
    result = _solve(nodes, steps, n)  # no world_fixes arg at all
    for seg in result.diagnostics["segments"]:
        assert seg.get("fixes_used", 0) == 0
