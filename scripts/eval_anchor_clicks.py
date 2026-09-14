"""Honest accuracy check: reproject hand-clicked anchor landmarks under a
camera track. The manual TRACK can be wrong (bad solo anchors), but the
clicked pixels are user ground truth wherever they exist.

Scoring lives in src.utils.anchor_click_eval.score_track — the same
path the camera regression gate uses (tests/test_camera_regression.py),
so the two can never drift apart.

Usage:
  .venv/bin/python scripts/eval_anchor_clicks.py ANCHORS.json TRACK.json
e.g.
  .venv/bin/python scripts/eval_anchor_clicks.py \
      output-origi/camera/origi01_anchors__manual.json \
      output-origi/camera/origi01_camera_track.json
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.anchor_click_eval import score_track  # noqa: E402


def main() -> None:
    anchors = json.load(open(sys.argv[1]))
    track = json.load(open(sys.argv[2]))

    frames = [f["frame"] for f in track["frames"]]
    dist = tuple(track.get("distortion", (0, 0))[:2])
    print(f"track span {min(frames)}..{max(frames)}  "
          f"dist={np.round(dist, 4).tolist()}")

    metrics = score_track(anchors, track)
    scored = {a["frame"]: a for a in metrics["per_anchor"]}
    for anchor in anchors["anchors"]:
        f = anchor["frame"]
        if f not in scored:
            print(f"  anchor f{f:>3}: NOT COVERED "
                  f"({len(anchor.get('landmarks', []))} clicks)")
            continue
        a = scored[f]
        print(f"  anchor f{f:>3}: {a['clicks']} clicks  reproj "
              f"med {a['med_px']:6.1f}px  max {a['max_px']:6.1f}px")

    if metrics["clicks"]:
        print(f"\nALL: {metrics['clicks']} clicks  "
              f"med {metrics['med_px']:.1f}px  "
              f"p90 {metrics['p90_px']:.1f}px  "
              f"max {metrics['max_px']:.1f}px")


if __name__ == "__main__":
    main()
