#!/usr/bin/env python3
"""Inspect / materialize the auto-accumulated ball-label corpus.

The web ball editor appends gold ball labels to ``<output>/ball_finetune``
every time anchors are saved (the fine-tuning flywheel). This CLI reports
what's accumulated and extracts frames before training.

    python scripts/ball_corpus.py status
    python scripts/ball_corpus.py materialize --output ./output
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.ball_label_corpus import load_manifest, materialize_frames  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("status", help="summarise accumulated labels")
    s.add_argument("--corpus", type=Path, default=Path("./output/ball_finetune"))

    mt = sub.add_parser("materialize", help="extract frames for all corpus clips")
    mt.add_argument("--corpus", type=Path, default=Path("./output/ball_finetune"))
    mt.add_argument("--output", type=Path, default=Path("./output"),
                    help="pipeline output dir clip_path entries resolve against")
    mt.add_argument("--force", action="store_true", help="re-extract even if cached")

    args = ap.parse_args()
    if args.cmd == "status":
        clips = load_manifest(args.corpus)["clips"]
        total = sum(c["n_labels"] for c in clips.values())
        print(f"corpus: {args.corpus}")
        print(f"  clips: {len(clips)}   total labels: {total}")
        for cid, c in sorted(clips.items()):
            print(f"    {cid}: {c['n_labels']} labels ({c['saves']} saves)")
        if total < 200:
            print(f"  NOTE: {total} labels — thin for fine-tuning; keep "
                  "reconstructing clips to grow the corpus.")
    elif args.cmd == "materialize":
        res = materialize_frames(args.corpus, args.output, force=args.force)
        for cid, st in sorted(res.items()):
            print(f"  {cid}: {st}")
        print(f"materialized {sum(1 for v in res.values() if v.startswith('extracted'))} "
              f"clip(s); {sum(1 for v in res.values() if v=='cached')} cached")


if __name__ == "__main__":
    main()
