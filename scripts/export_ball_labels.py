#!/usr/bin/env python3
"""Export a clip's manual ball anchors as a WASB soccer-format training set
(frames + CVAT annotation XML), for fine-tuning the ball detector.

Usage:
    python scripts/export_ball_labels.py \
        --output ./output --clip-id gberch \
        --dataset-root ./output/ball_finetune

Produces under <dataset-root>:
    frames/<clip-id>/{00000.png,...}   (extracted clip frames)
    annos/<clip-id>.xml                (gold ball labels from manual anchors)

Then fine-tune with the vendored WASB repo (see
docs/superpowers/specs/2026-06-15-ball-finetune-README.md).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.schemas.ball_anchor import BallAnchorSet
from src.utils.ball_finetune_export import export_dataset


def _anchor_path(output: Path, clip_id: str) -> Path:
    p = output / "ball" / f"{clip_id}_ball_anchors.json"
    return p if p.exists() else output / "ball" / "ball_anchors.json"


def _clip_path(output: Path, clip_id: str) -> Path:
    p = output / "shots" / f"{clip_id}.mp4"
    if not p.exists():
        raise FileNotFoundError(f"clip video not found: {p}")
    return p


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, default=Path("./output"),
                    help="pipeline output dir (holds ball/ + shots/)")
    ap.add_argument("--clip-id", required=True, help="shot/clip id, e.g. gberch")
    ap.add_argument("--dataset-root", type=Path, required=True,
                    help="WASB soccer-format dataset dir to write")
    ap.add_argument("--anchors", type=Path, default=None,
                    help="override path to the ball anchors JSON")
    args = ap.parse_args()

    anchors_path = args.anchors or _anchor_path(args.output, args.clip_id)
    if not anchors_path.exists():
        raise FileNotFoundError(f"no manual anchors at {anchors_path}")
    aset = BallAnchorSet.load(anchors_path)
    info = export_dataset(
        clip_path=_clip_path(args.output, args.clip_id),
        anchors=aset.anchors,
        out_root=args.dataset_root,
        video_id=args.clip_id,
    )
    print(f"exported {info['n_labels']} ball labels over {info['n_frames']} "
          f"frames for '{args.clip_id}'")
    print(f"  annos:  {info['anno_path']}")
    print(f"  frames: {info['frames_dir']}")
    if info["n_labels"] < 200:
        print(f"  NOTE: only {info['n_labels']} labels — thin for fine-tuning. "
              "Label more clips/frames (or confirm auto-anchors) before training.")


if __name__ == "__main__":
    main()
