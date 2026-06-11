"""Locate hand-cut ground-truth clips inside their source reel.

The ground-truth workflow: the operator cuts the *expected* output clips
(in CapCut or similar) and drops them under
``test-media/ground_truth/<reel-stem>/<group>/<NN>_<name>.mp4``. This
script finds each clip's exact frame range in the source by perceptual
frame hashing (dHash — survives re-encoding), giving labelled cut
boundaries + keep decisions + groupings without anyone writing
timestamps down.

Usage:
    python scripts/match_ground_truth.py SOURCE.mp4 CLIPS_DIR [--out gt.json]

Prints per-clip match locations and confidence; writes a JSON manifest
usable as an evaluation target for prepare_shots.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

_HASH_BITS = 8 * 8  # dHash on a 9x8 gray thumbnail


def dhash(frame_bgr: np.ndarray) -> np.ndarray:
    """64-bit difference hash as a bool array (row-wise gradient sign)."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    thumb = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    return (thumb[:, 1:] > thumb[:, :-1]).reshape(-1)


def video_hashes(path: Path) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    hashes: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        hashes.append(dhash(frame))
    cap.release()
    return hashes


def locate_clip(
    clip_hashes: list[np.ndarray],
    source_hashes: list[np.ndarray],
    n_anchors: int = 9,
) -> tuple[int, float]:
    """Best source offset for the clip; returns (offset, mean_hamming).

    Anchors a handful of clip frames and scans every source offset —
    O(len(source) * n_anchors) hash comparisons, fast enough for
    multi-minute reels.
    """
    n_clip = len(clip_hashes)
    n_src = len(source_hashes)
    if n_clip == 0 or n_src < n_clip:
        return -1, float(_HASH_BITS)
    anchor_idx = sorted({
        int(round(f * (n_clip - 1)))
        for f in np.linspace(0.0, 1.0, n_anchors)
    })
    anchors = np.stack([clip_hashes[i] for i in anchor_idx])  # (A, 64)
    src = np.stack(source_hashes)  # (S, 64)

    best_offset, best_score = -1, float(_HASH_BITS)
    for offset in range(0, n_src - n_clip + 1):
        idx = [offset + i for i in anchor_idx]
        score = float(np.mean(np.sum(anchors != src[idx], axis=1)))
        if score < best_score:
            best_score = score
            best_offset = offset
    return best_offset, best_score


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("source", type=Path)
    ap.add_argument("clips_dir", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--max-hamming", type=float, default=10.0,
                    help="mean per-frame bit distance above which a "
                         "match is reported as unreliable")
    args = ap.parse_args()

    print(f"hashing source {args.source.name} …")
    source_hashes = video_hashes(args.source)
    fps_cap = cv2.VideoCapture(str(args.source))
    fps = float(fps_cap.get(cv2.CAP_PROP_FPS) or 25.0)
    fps_cap.release()
    print(f"  {len(source_hashes)} frames @ {fps:.2f} fps")

    clips = sorted(p for p in args.clips_dir.rglob("*.mp4"))
    results = []
    for clip in clips:
        clip_hashes = video_hashes(clip)
        offset, score = locate_clip(clip_hashes, source_hashes)
        ok = offset >= 0 and score <= args.max_hamming
        rel = clip.relative_to(args.clips_dir)
        group = rel.parts[0] if len(rel.parts) > 1 else ""
        results.append({
            "clip": str(rel),
            "group": group,
            "frames": len(clip_hashes),
            "source_start_frame": offset,
            "source_end_frame": offset + len(clip_hashes) - 1,
            "source_start_s": round(offset / fps, 2) if offset >= 0 else None,
            "source_end_s": (
                round((offset + len(clip_hashes)) / fps, 2)
                if offset >= 0 else None
            ),
            "mean_hamming": round(score, 2),
            "reliable": bool(ok),
        })
        status = "OK " if ok else "??? "
        span = (f"{results[-1]['source_start_s']}-"
                f"{results[-1]['source_end_s']}s"
                if offset >= 0 else "no match")
        print(f"  {status} {rel}  ->  {span}  "
              f"(hamming {score:.1f}, {len(clip_hashes)} frames)")

    if args.out:
        args.out.write_text(json.dumps({
            "source": str(args.source),
            "fps": fps,
            "clips": results,
        }, indent=1))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
