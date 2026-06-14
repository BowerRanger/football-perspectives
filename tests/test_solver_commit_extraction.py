"""Golden snapshot test pinning ``solve_piecewise`` output assembly.

Task 0 of the Phase-2 plan extracts the inline output-assembler blocks of
``_Solver.solve()`` (the ``_commit_span`` closure, node-authority block,
restitution-flag block, and the leading/trailing open-span commit) into
importable module-level free functions. That refactor MUST be byte-identical
to the prior closure.

These tests snapshot the *current* piecewise output on three synthetic
scenarios that between them exercise every inline block:

* **rolling-only** — open span, no nodes (open-span grounded ray-casts);
* **node-bracketed arc** — kick→bounce nodes with a fitted ballistic arc
  (node-authority + ballistic-kind + leading/trailing open-span commit);
* **open-span trailing flight** — a single kick node followed by an open-span
  lob (the ``kind == "open"`` per-arc state-membership rule + the
  ``n_frames`` vs ``n_frames-1`` trailing boundary, F15).

The expected dicts are stored inline. Run BEFORE the refactor → PASS (pins
behavior); run AFTER → MUST still pass byte-identical.
"""
from __future__ import annotations

import numpy as np

from src.utils.ball_piecewise_solver import (
    SolverCfg,
    TrajectoryNode,
    solve_piecewise,
)
from tests.fixtures.ball_synthetic import (
    FPS,
    ballistic_worlds,
    broadcast_camera,
    per_frame_cams,
    project_track,
    rolling_worlds,
    steps_from_pixels,
)

CFG = SolverCfg()


def _node(frame: int, world, state: str, conf: float = 1.0) -> TrajectoryNode:
    return TrajectoryNode(
        frame=frame,
        world_xyz=(float(world[0]), float(world[1]), float(world[2])),
        state=state,
        confidence=conf,
    )


def _solve(nodes, steps, n, *, split_hints=(), cfg=CFG):
    Ks, Rs, ts = per_frame_cams(n)
    return solve_piecewise(
        nodes=nodes,
        steps=steps,
        confidences={},
        per_frame_K=Ks,
        per_frame_R=Rs,
        per_frame_t=ts,
        distortion=(0.0, 0.0),
        fps=FPS,
        n_frames=n,
        pitch_length_m=105.0,
        pitch_width_m=68.0,
        split_hints=tuple(split_hints),
        cfg=cfg,
    )


def _snapshot(result) -> dict:
    """Reduce a SolveResult to the byte-stable golden representation."""
    return {
        "world_by_frame": {
            f: [round(float(x), 6) for x in result.world_by_frame[f][0]]
            for f in sorted(result.world_by_frame)
        },
        "state_by_frame": {
            f: result.state_by_frame[f]
            for f in sorted(result.state_by_frame)
        },
        "flight_segments": [
            (tuple(s.frame_range), round(float(s.fit_residual_px), 4))
            for s in result.flight_segments
        ],
        "diag_segments": result.diagnostics["segments"],
    }


# ----------------------------------------------------------------------
# Scenario A: rolling-only roll (no nodes → single open span, grounded).


def _scenario_rolling():
    n = 30
    K, R, t = broadcast_camera()
    truth = rolling_worlds((50.0, 30.0), (-0.4, 0.05), n, z=0.11)
    pixels = project_track(truth, K, R, t)
    steps = steps_from_pixels(pixels, n, p_flight=0.0)
    return _solve([], steps, n)


_GOLDEN_ROLLING = {
    "world_by_frame": {
        0: [50.0, 30.0, 0.11], 1: [49.6, 30.05, 0.11], 2: [49.2, 30.1, 0.11],
        3: [48.8, 30.15, 0.11], 4: [48.4, 30.2, 0.11], 5: [48.0, 30.25, 0.11],
        6: [47.6, 30.3, 0.11], 7: [47.2, 30.35, 0.11], 8: [46.8, 30.4, 0.11],
        9: [46.4, 30.45, 0.11], 10: [46.0, 30.5, 0.11], 11: [45.6, 30.55, 0.11],
        12: [45.2, 30.6, 0.11], 13: [44.8, 30.65, 0.11], 14: [44.4, 30.7, 0.11],
        15: [44.0, 30.75, 0.11], 16: [43.6, 30.8, 0.11], 17: [43.2, 30.85, 0.11],
        18: [42.8, 30.9, 0.11], 19: [42.4, 30.95, 0.11], 20: [42.0, 31.0, 0.11],
        21: [41.6, 31.05, 0.11], 22: [41.2, 31.1, 0.11], 23: [40.8, 31.15, 0.11],
        24: [40.4, 31.2, 0.11], 25: [40.0, 31.25, 0.11], 26: [39.6, 31.3, 0.11],
        27: [39.2, 31.35, 0.11], 28: [38.8, 31.4, 0.11], 29: [38.4, 31.45, 0.11],
    },
    "state_by_frame": {f: "grounded" for f in range(30)},
    "flight_segments": [],
    "diag_segments": [
        {"start": 0, "end": 29, "kind": "open", "residual_px": None,
         "underconstrained": False, "fixes_used": 0},
    ],
}


def test_golden_rolling_only_open_span():
    assert _snapshot(_scenario_rolling()) == _GOLDEN_ROLLING


# ----------------------------------------------------------------------
# Scenario B: node-bracketed ballistic arc (kick → bounce, no detections).


def _scenario_node_bracketed_arc():
    n = 50
    a = np.array([50.0, 30.0, 0.11])
    b = np.array([42.0, 33.0, 0.11])
    nodes = [_node(10, a, "kick"), _node(35, b, "bounce")]
    steps = steps_from_pixels({}, n)
    return _solve(nodes, steps, n)


_GOLDEN_ARC = {
    "world_by_frame": {
        10: [50.0, 30.0, 0.11], 11: [49.68, 30.12, 0.298352],
        12: [49.36, 30.24, 0.471008], 13: [49.04, 30.36, 0.627968],
        14: [48.72, 30.48, 0.769232], 15: [48.4, 30.6, 0.8948],
        16: [48.08, 30.72, 1.004672], 17: [47.76, 30.84, 1.098848],
        18: [47.44, 30.96, 1.177328], 19: [47.12, 31.08, 1.240112],
        20: [46.8, 31.2, 1.2872], 21: [46.48, 31.32, 1.318592],
        22: [46.16, 31.44, 1.334288], 23: [45.84, 31.56, 1.334288],
        24: [45.52, 31.68, 1.318592], 25: [45.2, 31.8, 1.2872],
        26: [44.88, 31.92, 1.240112], 27: [44.56, 32.04, 1.177328],
        28: [44.24, 32.16, 1.098848], 29: [43.92, 32.28, 1.004672],
        30: [43.6, 32.4, 0.8948], 31: [43.28, 32.52, 0.769232],
        32: [42.96, 32.64, 0.627968], 33: [42.64, 32.76, 0.471008],
        34: [42.32, 32.88, 0.298352], 35: [42.0, 33.0, 0.11],
    },
    "state_by_frame": {
        **{f: "missing" for f in range(10)},
        **{f: "flight" for f in range(10, 36)},
        **{f: "missing" for f in range(36, 50)},
    },
    "flight_segments": [((10, 35), 0.0)],
    "diag_segments": [
        {"start": 10, "end": 35, "kind": "ballistic", "residual_px": None,
         "underconstrained": False, "fixes_used": 0},
        {"start": 0, "end": 10, "kind": "open", "residual_px": None,
         "underconstrained": False, "fixes_used": 0},
        {"start": 36, "end": 49, "kind": "open", "residual_px": None,
         "underconstrained": False, "fixes_used": 0},
    ],
}


def test_golden_node_bracketed_arc():
    assert _snapshot(_scenario_node_bracketed_arc()) == _GOLDEN_ARC


# ----------------------------------------------------------------------
# Scenario C: open-span trailing flight (single kick node + lob).
# Exercises the kind=="open" per-arc state-membership rule and the
# n_frames-1 trailing boundary (F15).


def _scenario_open_trailing_flight():
    n = 30
    K, R, t = broadcast_camera()
    flight = ballistic_worlds(
        (52.0, 30.0, 0.11), (-1.0, 0.8, 3.0), 18, start_frame=5, fps=FPS,
    )
    pixels = project_track(flight, K, R, t)
    steps = steps_from_pixels(pixels, n, p_flight=0.9)
    nodes = [_node(5, (52.0, 30.0, 0.11), "kick")]
    return _solve(nodes, steps, n)


_GOLDEN_OPEN_FLIGHT = {
    "world_by_frame": {
        5: [52.0, 30.0, 0.11], 6: [51.96, 30.032, 0.214304],
        7: [51.92, 30.064, 0.302912], 8: [51.88, 30.096, 0.375824],
        9: [51.84, 30.128, 0.43304], 10: [51.8, 30.16, 0.47456],
        11: [51.76, 30.192, 0.500384], 12: [51.72, 30.224, 0.510512],
        13: [51.68, 30.256, 0.504944], 14: [51.64, 30.288, 0.48368],
        15: [51.6, 30.32, 0.44672], 16: [51.56, 30.352, 0.394064],
        17: [51.52, 30.384, 0.325712], 18: [51.48, 30.416, 0.241664],
        19: [51.44, 30.448, 0.14192], 20: [51.4, 30.48, 0.02648],
        21: [51.36, 30.512, -0.104656], 22: [51.32, 30.544, -0.251488],
    },
    "state_by_frame": {
        **{f: "missing" for f in range(5)},
        **{f: "flight" for f in range(5, 23)},
        **{f: "missing" for f in range(23, 30)},
    },
    "flight_segments": [((5, 22), 0.0)],
    "diag_segments": [
        {"start": 0, "end": 5, "kind": "open", "residual_px": None,
         "underconstrained": False, "fixes_used": 0},
        {"start": 6, "end": 29, "kind": "open", "residual_px": None,
         "underconstrained": False, "fixes_used": 0},
    ],
}


def test_golden_open_span_trailing_flight():
    assert _snapshot(_scenario_open_trailing_flight()) == _GOLDEN_OPEN_FLIGHT
