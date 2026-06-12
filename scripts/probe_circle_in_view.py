"""Probe _circle_in_view_fraction under a track's cameras at given frames.

Usage: .venv/bin/python scripts/probe_circle_in_view.py TRACK.json f1,f2,...
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.stages.camera import _circle_in_view_fraction  # noqa: E402


def main() -> None:
    track = json.load(open(sys.argv[1]))
    cams = {f["frame"]: f for f in track["frames"]}
    size = tuple(track.get("image_size", (1920, 1080)))
    for fid in [int(x) for x in sys.argv[2].split(",")]:
        if fid not in cams:
            print(f"f{fid}: not in track")
            continue
        c = cams[fid]
        frac = _circle_in_view_fraction(
            np.array(c["K"]), np.array(c["R"]), np.array(c["t"]), size)
        print(f"f{fid}: circle in-view fraction = {frac:.2f}")


if __name__ == "__main__":
    main()
