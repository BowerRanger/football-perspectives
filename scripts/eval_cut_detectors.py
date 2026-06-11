"""Score PySceneDetect configurations against spike-confirmed cuts.

Ground truth: the strong frame-diff spike candidates from
``diagnose_cuts.py`` (visually confirmed on the Liverpool reel). A
detector config scores a hit when it places a cut within ±3 frames of a
ground-truth cut. ``extra`` counts detected cuts not near any
ground-truth spike — many are genuine softer cuts (spikes are only the
violent ones), so extra is a *soft* over-split indicator, not an error
count.

Usage:
    python scripts/eval_cut_detectors.py REEL.mp4 OUTPUT_DIR
(reads OUTPUT_DIR/cut_diagnostics.json written by diagnose_cuts.py)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from src.utils.shot_split import detect_spans


def cut_frames_from_spans(spans) -> list[int]:
    # A cut sits at each span start (except the first at frame 0).
    return [s.start_frame for s in spans[1:]]


def score(cuts: list[int], truth: list[int], tol: int = 3):
    hits = sum(1 for t in truth if any(abs(c - t) <= tol for c in cuts))
    extra = sum(1 for c in cuts if not any(abs(c - t) <= tol for t in truth))
    return hits, extra


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("reel", type=Path)
    ap.add_argument("output_dir", type=Path)
    args = ap.parse_args()

    diag = json.loads((args.output_dir / "cut_diagnostics.json").read_text())
    truth = [c["frame"] for c in diag["candidates"]]
    print(f"ground truth: {len(truth)} spike-confirmed cuts\n")

    configs = [
        # current default
        dict(detector="adaptive", adaptive_threshold=3.0,
             min_scene_len_frames=13),
        dict(detector="adaptive", adaptive_threshold=2.0,
             min_scene_len_frames=13),
        dict(detector="adaptive", adaptive_threshold=2.0,
             adaptive_min_content_val=10.0, min_scene_len_frames=6),
        dict(detector="content", threshold=32.0, min_scene_len_frames=13),
        dict(detector="content", threshold=27.0, min_scene_len_frames=13),
        dict(detector="content", threshold=27.0, min_scene_len_frames=6),
        dict(detector="content", threshold=22.0, min_scene_len_frames=6),
        dict(detector="content", threshold=18.0, min_scene_len_frames=6),
    ]

    header = f"{'config':<58} {'cuts':>5} {'hit':>4}/{len(truth):<3} {'extra':>5} {'secs':>5}"
    print(header)
    print("-" * len(header))
    for cfg in configs:
        t0 = time.time()
        spans = detect_spans(args.reel, **cfg)
        dt = time.time() - t0
        cuts = cut_frames_from_spans(spans)
        hits, extra = score(cuts, truth)
        label = ", ".join(f"{k.replace('_frames','').replace('min_scene_len','msl')}={v}"
                          for k, v in cfg.items())
        print(f"{label:<58} {len(cuts):>5} {hits:>4}/{len(truth):<3} "
              f"{extra:>5} {dt:>5.0f}")


if __name__ == "__main__":
    main()
