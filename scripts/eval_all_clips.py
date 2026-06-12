"""Run the per-clip camera dashboard (_clip_eval) across all test clips.

Usage: .venv/bin/python scripts/eval_all_clips.py [clip ...]
  clip is one of: gberch, kroupi01, origi01, origi02 (default: all)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts._clip_eval import ev  # noqa: E402

CLIPS = {
    "gberch": ("output/camera/gberch", 429),
    "kroupi01": ("output-kroupi/camera/kroupi01", 156),
    "origi01": ("output-origi/camera/origi01", 506),
    "origi02": ("output-origi/camera/origi02", 334),
}


def main() -> None:
    names = sys.argv[1:] or list(CLIPS)
    for name in names:
        if name not in CLIPS:
            print(f"unknown clip {name!r}; choose from {sorted(CLIPS)}")
            continue
        base, n = CLIPS[name]
        ev(base, name, n)
        print()


if __name__ == "__main__":
    main()
