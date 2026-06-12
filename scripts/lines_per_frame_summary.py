"""Summarise detected straight-line counts per frame for a shot's
detected-lines JSON — where are the 0-line (interp) spans?

Usage: .venv/bin/python scripts/lines_per_frame_summary.py PATH.json
"""
import json
import sys
from itertools import groupby


def main() -> None:
    dl = json.load(open(sys.argv[1]))
    frames = {int(k): v for k, v in dl["frames"].items()}
    if not frames:
        print("no frames")
        return
    lo, hi = min(frames), max(frames)
    counts = {}
    for f in range(lo, hi + 1):
        lines = frames.get(f, {}).get("lines", [])
        straight = [ln for ln in lines if "circle" not in ln["name"]]
        counts[f] = len(straight)

    # run-length encode zero / nonzero spans
    print(f"span {lo}..{hi}")
    runs = []
    for is_zero, grp in groupby(sorted(counts), key=lambda f: counts[f] == 0):
        g = list(grp)
        runs.append((is_zero, g[0], g[-1]))
    for is_zero, a, b in runs:
        n = b - a + 1
        if is_zero:
            print(f"  {a:>4}..{b:<4} ({n:>3}f)  ZERO lines")
        else:
            cs = [counts[f] for f in range(a, b + 1)]
            print(f"  {a:>4}..{b:<4} ({n:>3}f)  lines min/med/max = "
                  f"{min(cs)}/{sorted(cs)[len(cs)//2]}/{max(cs)}")

    # name frequency
    from collections import Counter
    names = Counter()
    for f, v in frames.items():
        for ln in v.get("lines", []):
            if "circle" not in ln["name"]:
                names[ln["name"]] += 1
    print("\nline-name frequency:")
    for name, n in names.most_common():
        print(f"  {name:<28} {n}")


if __name__ == "__main__":
    main()
