"""Corridor + frame-to-frame speed profile for a moving-camera track.

Diagnostic for the camera.static_camera=auto moving path
(src/stages/camera.py's _refine_with_line_extraction +
src/utils/line_camera_refine.py's refine_camera_from_lines): for every
frame, computes the implied camera centre (C = -R^T @ t) and its
distance from the piecewise-LERP corridor between the anchor frames'
own centres, then reports percentiles, the worst runs, a confidence-vs-
distance correlation check, and the max frame-to-frame speed implied by
consecutive frames' centres.

Built to reproduce and verify the fix for the 2026-09-09 corridor-drift
defect (see docs/superpowers/specs/2026-09-09-moving-camera-support.md,
addendum): under-determined (1-2 line) frames in the per-frame line
solve used to wander tens-to-hundreds of metres from the corridor while
reporting high confidence.

Usage:
  .venv311/bin/python scripts/eval_moving_camera_corridor.py TRACK.json
  .venv311/bin/python scripts/eval_moving_camera_corridor.py TRACK.json \
      --corridor-threshold-m 5.0 --speed-budget-m-s 6.0
"""
from __future__ import annotations

import argparse
import json

import numpy as np


def _centre(frame: dict) -> np.ndarray:
    R = np.array(frame["R"])
    t = np.array(frame["t"])
    return -R.T @ t


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("track", help="path to a *_camera_track.json")
    parser.add_argument("--corridor-threshold-m", type=float, default=5.0)
    parser.add_argument("--speed-budget-m-s", type=float, default=6.0)
    args = parser.parse_args()

    track = json.load(open(args.track))
    fps = float(track["fps"])
    frames = {f["frame"]: f for f in track["frames"]}
    anchor_frames = sorted(f["frame"] for f in track["frames"] if f["is_anchor"])
    print(f"track span {min(frames)}..{max(frames)}  fps={fps}")
    print(f"anchor frames: {anchor_frames}")

    anchor_c = {af: _centre(frames[af]) for af in anchor_frames}
    for af in anchor_frames:
        print(f"  anchor f{af}: C={np.round(anchor_c[af], 1).tolist()}")

    def corridor_at(idx: int) -> np.ndarray:
        """Piecewise LERP between the bracketing anchor centres — the
        same corridor _run_shot's Step 2 interpolation and
        _refine_with_line_extraction's per-frame corridor bound use."""
        for a, b in zip(anchor_frames, anchor_frames[1:]):
            if a <= idx <= b:
                w = (idx - a) / (b - a) if b > a else 0.0
                return (1 - w) * anchor_c[a] + w * anchor_c[b]
        if idx <= anchor_frames[0]:
            return anchor_c[anchor_frames[0]]
        return anchor_c[anchor_frames[-1]]

    rows = []
    for fr in sorted(frames):
        f = frames[fr]
        c = _centre(f)
        dist = float(np.linalg.norm(c - corridor_at(fr)))
        rows.append((fr, dist, float(f["confidence"]), bool(f["is_anchor"]), c))

    dists = np.array([r[1] for r in rows])
    confs = np.array([r[2] for r in rows])
    is_anchor_arr = np.array([r[3] for r in rows])

    print(f"\nn frames: {len(rows)}")
    print(
        f"corridor distance: p50={np.percentile(dists, 50):.2f}m "
        f"p90={np.percentile(dists, 90):.2f}m "
        f"p99={np.percentile(dists, 99):.2f}m max={dists.max():.2f}m"
    )

    thresh = args.corridor_threshold_m
    bad = [r for r in rows if not r[3] and r[1] > thresh]
    print(f"\nnon-anchor frames > {thresh}m off corridor: {len(bad)}")
    runs: list[list] = []
    cur: list = []
    for r in sorted(bad, key=lambda r: r[0]):
        if cur and r[0] == cur[-1][0] + 1:
            cur.append(r)
        else:
            if cur:
                runs.append(cur)
            cur = [r]
    if cur:
        runs.append(cur)
    for run in runs:
        frs = [r[0] for r in run]
        worst = max(run, key=lambda r: r[1])
        print(
            f"  f{frs[0]}-f{frs[-1]}: worst {worst[1]:.1f}m off, "
            f"C={np.round(worst[4], 0).tolist()}, conf={worst[2]:.2f}"
        )

    near_mask = (~is_anchor_arr) & (dists <= thresh)
    far_mask = (~is_anchor_arr) & (dists > thresh)
    near_conf = float(confs[near_mask].mean()) if near_mask.any() else float("nan")
    far_conf = float(confs[far_mask].mean()) if far_mask.any() else float("nan")
    print(
        f"\nmean confidence: far frames (n={int(far_mask.sum())}) "
        f"conf={far_conf:.2f} | near frames (n={int(near_mask.sum())}) "
        f"conf={near_conf:.2f}"
    )

    # Frame-to-frame speed.
    budget = args.speed_budget_m_s
    max_step_m = budget / fps
    ordered = sorted(rows, key=lambda r: r[0])
    speeds = []
    for a, b in zip(ordered, ordered[1:]):
        if b[0] - a[0] != 1:
            continue
        step = float(np.linalg.norm(b[4] - a[4]))
        speeds.append((a[0], b[0], step * fps, a[3], b[3]))
    violations = [s for s in speeds if s[2] > budget + 1e-6]
    print(
        f"\nframe-to-frame speed: budget={budget:.1f} m/s "
        f"(max_step={max_step_m:.4f}m/frame at {fps} fps)"
    )
    print(f"  {len(violations)}/{len(speeds)} transitions exceed the budget")
    for fa, fb, speed, aa, ab in violations:
        tag = " [anchor-adjacent, exempt by design]" if (aa or ab) else " [INTERIOR]"
        print(f"    f{fa}->f{fb}: {speed:.2f} m/s{tag}")


if __name__ == "__main__":
    main()
