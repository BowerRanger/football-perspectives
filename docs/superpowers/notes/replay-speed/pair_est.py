"""Pairwise replay-speed estimator prototype + GT evaluation.

For each (live L, replay R): search rate r (replay frame = r live frames) and offset o
(live frame of replay frame 0). Replay tempo is divided by r (slow-mo shows players
moving r times as fast) and compared with live tempo at t = o + j*r.

pair_est.py <gt.json> [<gt.json> ...]
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d

HERE = Path(__file__).parent
TAG = {"liverpool": "output-highlights", "mancity": "output-mancity", "saka": "output-saka"}


def curve(tag: str, shot: str, key: str = "p75") -> tuple[np.ndarray, np.ndarray]:
    d = json.loads((HERE / "tempo" / f"{tag}_{shot}.json").read_text())
    fr = np.array([r["frame"] for r in d["rows"]], float)
    v = np.array([r[key] for r in d["rows"]], float) * d["fps"]
    return fr, v


def estimate(Lf, Lv, Rf, Rv, rates=np.geomspace(0.12, 1.3, 90), min_cover=0.7, w_shape=0.5,
             smooth=2.0):
    eps = 0.05
    lL = np.log(gaussian_filter1d(Lv, smooth) + eps)
    lR = np.log(gaussian_filter1d(Rv, smooth) + eps)
    best = (np.inf, None, None)
    table = []
    for r in rates:
        t_rel = (Rf - Rf[0]) * r
        span = t_rel[-1]
        o_lo = Lf[0] - (1 - min_cover) * span
        o_hi = Lf[-1] - min_cover * span
        if o_hi < o_lo:
            continue
        best_r = (np.inf, None)
        for o in np.arange(o_lo, o_hi + 0.5, 1.0):
            t = o + t_rel
            inside = (t >= Lf[0]) & (t <= Lf[-1])
            if inside.mean() < min_cover or inside.sum() < 6:
                continue
            l = np.interp(t[inside], Lf, lL)
            x = lR[inside] - np.log(r)
            level = np.median(np.abs(x - l))
            a, b = x - x.mean(), l - l.mean()
            den = np.sqrt((a * a).sum() * (b * b).sum())
            ncc = (a * b).sum() / den if den > 1e-9 else 0.0
            cost = level + w_shape * (1 - ncc)
            if cost < best_r[0]:
                best_r = (cost, o)
        table.append((r, best_r[0], best_r[1]))
        if best_r[0] < best[0]:
            best = (best_r[0], r, best_r[1])
    return best, table


def main():
    rows = []
    for p in sys.argv[1:]:
        rows += json.loads(Path(p).read_text())
    errs = []
    for g in rows:
        if not g.get("match") or g.get("rate") is None:
            continue
        tag = TAG[g["reel"]]
        try:
            Lf, Lv = curve(tag, g["live"])
            Rf, Rv = curve(tag, g["replay"])
        except FileNotFoundError:
            continue
        (cost, r, o), table = estimate(Lf, Lv, Rf, Rv)
        if r is None:
            print(f"{g['reel']:9s} {g['replay']} no estimate")
            continue
        gt = g["rate"]
        e = np.log(r / gt)
        errs.append(abs(e))
        ev = g["events"][0] if g.get("events") else None
        o_gt = (ev["live_frame"] - ev["replay_frame"] * gt) if ev else None
        print(f"{g['reel']:9s} {g['group']} {g['replay']}: gt {gt:.3f} ({g.get('grade')}{' ramp' if g.get('ramp') else ''})"
              f"  est {r:.3f}  ratio {r / gt:5.2f}  off est {o:7.1f} gt {o_gt if o_gt is None else round(o_gt, 1)}  cost {cost:.3f}")
    if errs:
        errs = np.array(errs)
        print(f"n {len(errs)}  median |log ratio| {np.median(errs):.3f}  within 10% {np.mean(errs < np.log(1.1)):.2f}"
              f"  within 20% {np.mean(errs < np.log(1.2)):.2f}")


if __name__ == "__main__":
    main()
