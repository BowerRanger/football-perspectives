"""Thin CLI over ``src.utils.short_compositor`` (logic lives there).

Usage:
    python scripts/compose_short.py --edl config/shorts/gberch_matchday.yaml \\
        --out output-shorts/shorts/gberch_matchday.mp4 [--audio mix.wav]

See ``src/utils/short_compositor.py`` for the EDL schema.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml  # noqa: E402

from src.utils.short_compositor import compose  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--edl", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--audio", type=Path, default=None)
    args = ap.parse_args()
    edl = yaml.safe_load(args.edl.read_text())
    print(json.dumps(compose(edl, args.out, audio=args.audio), indent=2))


if __name__ == "__main__":
    main()
