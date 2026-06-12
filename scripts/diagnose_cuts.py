"""Diagnostic: find camera-cut candidates the splitter missed.

Computes a per-frame difference curve over the source reel (downscaled
gray |frame-diff| mean — hard cuts are large isolated spikes even inside
high-motion footage), then compares spike locations against the cut
boundaries recorded in an ingested manifest. Spikes far from any
recorded boundary are candidate missed cuts.

Usage:
    python scripts/diagnose_cuts.py REEL.mp4 OUTPUT_DIR [--top 40]

Writes ``OUTPUT_DIR/cut_diagnostics.json`` and prints a table.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def diff_curve(video_path: Path, width_px: int = 160) -> tuple[np.ndarray, float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    diffs: list[float] = []
    prev = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        scale = width_px / w
        small = cv2.resize(frame, (width_px, max(1, int(h * scale))),
                           interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
        if prev is not None:
            diffs.append(float(np.mean(np.abs(gray - prev))))
        prev = gray
    cap.release()
    return np.asarray(diffs), fps


def spike_candidates(curve: np.ndarray, *, window: int = 25,
                     z_min: float = 4.0, abs_min: float = 18.0) -> list[dict]:
    """Frames whose diff hugely exceeds the local neighbourhood.

    A cut at frame boundary i->i+1 shows at curve index i. The local
    median/MAD window excludes the frame itself so the spike doesn't
    suppress its own score.
    """
    out = []
    n = len(curve)
    for i in range(n):
        lo, hi = max(0, i - window), min(n, i + window + 1)
        neigh = np.concatenate([curve[lo:i], curve[i + 1:hi]])
        if len(neigh) < 5:
            continue
        med = float(np.median(neigh))
        mad = float(np.median(np.abs(neigh - med))) + 1e-6
        z = (curve[i] - med) / (1.4826 * mad)
        if z >= z_min and curve[i] >= abs_min:
            out.append({"frame": i + 1, "diff": round(float(curve[i]), 1),
                        "z": round(float(z), 1)})
    # Collapse runs (dissolves spread over a few frames): keep the max
    # per run of adjacent candidates.
    collapsed: list[dict] = []
    for c in out:
        if collapsed and c["frame"] - collapsed[-1]["frame"] <= 3:
            if c["diff"] > collapsed[-1]["diff"]:
                collapsed[-1] = c
        else:
            collapsed.append(c)
    return collapsed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("reel", type=Path)
    ap.add_argument("output_dir", type=Path)
    ap.add_argument("--top", type=int, default=40)
    args = ap.parse_args()

    curve, fps = diff_curve(args.reel)
    candidates = spike_candidates(curve)

    manifest = json.loads(
        (args.output_dir / "shots" / "shots_manifest.json").read_text())
    boundaries = sorted({
        int(round(s["source_start_s"] * fps)) for s in manifest["shots"]
    } | {
        int(round(s["source_end_s"] * fps)) for s in manifest["shots"]
    })

    def near_boundary(frame: int, tol: int = 3) -> bool:
        return any(abs(frame - b) <= tol for b in boundaries)

    missed = [c for c in candidates if not near_boundary(c["frame"])]
    matched = [c for c in candidates if near_boundary(c["frame"])]

    print(f"fps={fps:.2f} frames={len(curve) + 1}")
    print(f"spike candidates: {len(candidates)} "
          f"(matched to recorded cuts: {len(matched)}, "
          f"MISSED: {len(missed)})")
    print("\ntop missed-cut candidates (frame, t, diff, z):")
    for c in sorted(missed, key=lambda c: -c["diff"])[: args.top]:
        t = c["frame"] / fps
        print(f"  f={c['frame']:6d}  t={t:7.2f}s  diff={c['diff']:6.1f}  "
              f"z={c['z']:5.1f}")

    out = {
        "fps": fps,
        "boundaries": boundaries,
        "candidates": candidates,
        "missed": missed,
    }
    (args.output_dir / "cut_diagnostics.json").write_text(
        json.dumps(out, indent=1))
    print(f"\nwrote {args.output_dir / 'cut_diagnostics.json'}")


if __name__ == "__main__":
    main()
