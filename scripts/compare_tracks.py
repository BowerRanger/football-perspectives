"""Compare two camera tracks of the same shot: C, lens, per-frame deltas.

Usage: .venv/bin/python scripts/compare_tracks.py OLD.json NEW.json
"""
import json
import sys

import numpy as np


def _geo(a, b):
    c = (np.trace(np.array(a).T @ np.array(b)) - 1) / 2
    return float(np.degrees(np.arccos(max(-1.0, min(1.0, c)))))


def main() -> None:
    old = json.load(open(sys.argv[1]))
    new = json.load(open(sys.argv[2]))
    print(f"C   old={np.round(old.get('camera_centre', []), 3).tolist()}")
    print(f"    new={np.round(new.get('camera_centre', []), 3).tolist()}")
    print(f"dist old={np.round(old.get('distortion', [0, 0])[:2], 4).tolist()} "
          f"new={np.round(new.get('distortion', [0, 0])[:2], 4).tolist()}")
    print(f"pp  old={np.round(old.get('principal_point', []), 1).tolist()} "
          f"new={np.round(new.get('principal_point', []), 1).tolist()}")
    of = {f["frame"]: f for f in old["frames"]}
    nf = {f["frame"]: f for f in new["frames"]}
    common = sorted(set(of) & set(nf))
    only_old = sorted(set(of) - set(nf))
    only_new = sorted(set(nf) - set(of))
    print(f"frames: old {len(of)} new {len(nf)} common {len(common)} "
          f"only-old {only_old[:5]}{'...' if len(only_old) > 5 else ''} "
          f"only-new {only_new[:5]}{'...' if len(only_new) > 5 else ''}")
    drot = [_geo(of[f]["R"], nf[f]["R"]) for f in common]
    dfx = [abs(of[f]["K"][0][0] - nf[f]["K"][0][0]) for f in common]
    drot = np.array(drot)
    dfx = np.array(dfx)
    print(f"dRot deg: med {np.median(drot):.3f} p95 {np.percentile(drot, 95):.3f} "
          f"max {drot.max():.3f} (frame {common[int(drot.argmax())]})")
    print(f"dFx px : med {np.median(dfx):.1f} p95 {np.percentile(dfx, 95):.1f} "
          f"max {dfx.max():.1f}")
    big = [f for f, d in zip(common, drot) if d > 0.5]
    if big:
        print(f"frames with dRot>0.5deg: {big[:20]}{'...' if len(big) > 20 else ''}")


if __name__ == "__main__":
    main()
