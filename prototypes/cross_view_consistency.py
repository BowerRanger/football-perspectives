#!/usr/bin/env python3
"""cross_view_consistency.py — Compare two shot ball tracks at co-temporal frames.

Loads two shots' solved ball_track.json files plus (optionally) the reference
shot's ball_fixes.json to obtain the refined temporal offset between them.
Maps shotB frames onto shotA's timeline via the offset:

    shotA frame  ↔  shotB frame f_b  where  f_a ≈ f_b - refined_offset

and prints median / p90 3D Euclidean distance between the two tracks at
co-temporal frames where BOTH have a non-null world_xyz — split into flight vs
grounded buckets using each BallFrame's `state` field.

Usage
-----
    python prototypes/cross_view_consistency.py [output_dir [shotA [shotB]]]

Arguments
---------
    output_dir  Path to the pipeline output directory (default: output-origi).
    shotA       Shot id for the reference shot (default: origi01).
    shotB       Shot id for the partner shot   (default: origi02).

The refined offset is read from::

    output_dir/ball/<shotA>_ball_fixes.json  →  cross_replay["refined_offset"]

or, if the fixes sidecar does not exist, from::

    output_dir/ball/<shotA>_ball_diag.json   →  cross_replay["refined_offset"]

If neither source provides an offset, a fallback of 0.0 is used and a warning
is printed so the distances are still reported (though they may be meaningless
without a correct alignment).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Locate the repo root (parent of prototypes/) and inject it into sys.path so
# that src.schemas.* imports work without installation.
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.schemas.ball_track import BallTrack  # noqa: E402


def _load_track(ball_dir: Path, shot_id: str) -> BallTrack | None:
    path = ball_dir / f"{shot_id}_ball_track.json"
    if not path.exists():
        print(f"[WARN] Ball track not found: {path}")
        return None
    return BallTrack.load(path)


def _refined_offset(ball_dir: Path, shot_a: str) -> float | None:
    """Return the refined_offset (shotB_frame - shotA_frame) from sidecars."""
    # Prefer fixes sidecar (written after triangulation pass)
    fixes_path = ball_dir / f"{shot_a}_ball_fixes.json"
    if fixes_path.exists():
        import json
        data = json.loads(fixes_path.read_text())
        cr = data.get("cross_replay", {})
        if "refined_offset" in cr:
            return float(cr["refined_offset"])
        # Also check partners dict (older schema)
        for _partner, pdata in cr.get("partners", {}).items():
            if "refined_offset" in pdata:
                return float(pdata["refined_offset"])

    # Fall back to diag sidecar
    diag_path = ball_dir / f"{shot_a}_ball_diag.json"
    if diag_path.exists():
        import json
        data = json.loads(diag_path.read_text())
        cr = data.get("cross_replay", {})
        if "refined_offset" in cr:
            return float(cr["refined_offset"])

    return None


def _bucket(state_a: str, state_b: str) -> str:
    """Classify a frame pair into flight / grounded / mixed."""
    flight = {"flight"}
    grounded = {"grounded"}
    if state_a in flight and state_b in flight:
        return "flight"
    if state_a in grounded and state_b in grounded:
        return "grounded"
    return "mixed"


def run(output_dir: str, shot_a: str, shot_b: str) -> None:
    ball_dir = Path(output_dir) / "ball"
    if not ball_dir.exists():
        print(f"[ERROR] Ball directory not found: {ball_dir}")
        sys.exit(1)

    track_a = _load_track(ball_dir, shot_a)
    track_b = _load_track(ball_dir, shot_b)

    if track_a is None or track_b is None:
        print("[ERROR] One or both ball tracks could not be loaded — aborting.")
        sys.exit(1)

    offset = _refined_offset(ball_dir, shot_a)
    if offset is None:
        print(
            f"[WARN] No refined_offset found for {shot_a} in ball_fixes.json or "
            f"ball_diag.json.  Using offset=0.0 (distances may be meaningless)."
        )
        offset = 0.0
    else:
        print(f"Refined offset ({shot_a} → {shot_b}): {offset:+.1f} frames")

    # Build lookup: shotB frame_index → BallFrame
    b_by_frame = {f.frame: f for f in track_b.frames}

    buckets: dict[str, list[float]] = {"flight": [], "grounded": [], "mixed": []}

    for fa in track_a.frames:
        if fa.world_xyz is None:
            continue
        # Map shotA frame fa.frame → expected shotB frame
        fb_idx = round(fa.frame + offset)
        fb = b_by_frame.get(fb_idx)
        if fb is None or fb.world_xyz is None:
            continue
        dist = float(np.linalg.norm(np.array(fa.world_xyz) - np.array(fb.world_xyz)))
        bkt = _bucket(fa.state, fb.state)
        buckets[bkt].append(dist)

    total = sum(len(v) for v in buckets.values())
    if total == 0:
        print(
            "[WARN] No co-temporal frames found where both tracks have world_xyz.  "
            "Check the offset and track coverage."
        )
        return

    print(f"\nCo-temporal frames compared: {total}")
    print(f"{'Bucket':<12}  {'N':>5}  {'Median (m)':>12}  {'P90 (m)':>10}")
    print("-" * 44)
    for bkt in ("flight", "grounded", "mixed"):
        vals = buckets[bkt]
        if not vals:
            print(f"{bkt:<12}  {'0':>5}  {'—':>12}  {'—':>10}")
            continue
        arr = np.array(vals)
        median = float(np.median(arr))
        p90 = float(np.percentile(arr, 90))
        print(f"{bkt:<12}  {len(vals):>5}  {median:>12.3f}  {p90:>10.3f}")


def main() -> None:
    args = sys.argv[1:]
    output_dir = args[0] if len(args) > 0 else "output-origi"
    shot_a = args[1] if len(args) > 1 else "origi01"
    shot_b = args[2] if len(args) > 2 else "origi02"
    print(f"Cross-view consistency: {shot_a} vs {shot_b}  (output_dir={output_dir})")
    run(output_dir, shot_a, shot_b)


if __name__ == "__main__":
    main()
