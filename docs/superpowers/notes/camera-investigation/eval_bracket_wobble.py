"""Per-frame ROLL deviation from the bracketing-anchor interpolation.

The user perceives inter-anchor instability as the pitch "angling left and
right" — i.e. roll excursions relative to the (trustworthy, click-snapped)
anchor frames. Pan/tilt deviation from a SLERP is expected (real camera
motion is non-linear); ROLL deviation is not — broadcast rigs barely roll.

Usage:
  .venv/bin/python eval_bracket_wobble.py TRACK_JSON ANCHORS_JSON [f1 f2 ...]

Prints the worst 12 frames by |roll deviation| plus any explicitly listed
frames of interest.
"""
import json
import sys

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def main() -> None:
    track = json.load(open(sys.argv[1]))
    anchors = json.load(open(sys.argv[2]))
    interest = [int(a) for a in sys.argv[3:]]

    frames = {f["frame"]: f for f in track["frames"] if f.get("R")}
    a_fids = sorted(
        a["frame"] for a in anchors["anchors"]
        if len(a.get("landmarks", [])) >= 4
        and not all(lm["name"].startswith("pnl_")
                    for lm in a["landmarks"])
        and a["frame"] in frames)
    if len(a_fids) < 2:
        print("need >=2 user-anchor frames in track")
        return

    def R_of(f):
        return np.asarray(frames[f]["R"], float)

    rows = []
    for i in range(len(a_fids) - 1):
        a, b = a_fids[i], a_fids[i + 1]
        sl = Slerp([float(a), float(b)],
                   Rotation.from_matrix([R_of(a), R_of(b)]))
        for f in range(a + 1, b):
            if f not in frames:
                continue
            R_br = sl([float(f)]).as_matrix()[0]
            # relative rotation, decomposed about the view axis (roll)
            R_rel = R_of(f) @ R_br.T
            rv = Rotation.from_matrix(R_rel).as_rotvec()
            view = R_br[2]
            roll = np.degrees(float(np.dot(rv, view)))
            total = np.degrees(float(np.linalg.norm(rv)))
            rows.append((f, roll, total, a, b))

    rows_by_f = {r[0]: r for r in rows}
    worst = sorted(rows, key=lambda r: -abs(r[1]))[:12]
    print(f"anchors: {a_fids}")
    print(f"inter-anchor frames: {len(rows)}  "
          f"|roll| mean {np.mean([abs(r[1]) for r in rows]):.2f} "
          f"p95 {np.percentile([abs(r[1]) for r in rows], 95):.2f} "
          f"max {max(abs(r[1]) for r in rows):.2f} deg")
    print("worst by |roll dev from bracket|:")
    for f, roll, total, a, b in worst:
        print(f"  f{f:4d} roll {roll:+6.2f} deg  total {total:5.2f} deg  "
              f"(bracket {a}..{b})")
    for f in interest:
        r = rows_by_f.get(f)
        if r:
            print(f"  INTEREST f{f}: roll {r[1]:+.2f} total {r[2]:.2f} "
                  f"(bracket {r[3]}..{r[4]})")
        else:
            print(f"  INTEREST f{f}: not inter-anchor (anchor frame or "
                  f"outside span)")


if __name__ == "__main__":
    main()
