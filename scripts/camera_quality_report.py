"""Camera-track quality metrics for the landmark-free iteration experiments.

Reads a camera_track.json (+ optional detected_lines.json) and emits a compact
metrics block to stdout and, with --out, a markdown summary. Used to compare
each improvement step (baseline / two-knob detection / smoothing / coverage)
on the gberch clip.

  python scripts/camera_quality_report.py \
      --track output/camera/gberch_camera_track.json \
      --lines output/camera/gberch_detected_lines.json \
      --label "step 0 baseline" --out output/camera/gberch_summary__0_baseline.md
"""

from __future__ import annotations

import argparse
import json
from collections import Counter

import numpy as np


def _geodesic_deg(a, b) -> float:
    c = (np.trace(np.asarray(a).T @ np.asarray(b)) - 1) / 2
    c = max(-1.0, min(1.0, c))
    return float(np.degrees(np.arccos(c)))


def compute(track_path: str, lines_path: str | None) -> dict:
    ct = json.load(open(track_path))
    frames = {f["frame"]: f for f in ct["frames"]}
    idx = sorted(frames)
    fx = np.array([frames[i]["K"][0][0] for i in idx])
    conf = np.array([frames[i].get("confidence", 1.0) for i in idx])

    dfx, drot = [], []
    for a, b in zip(idx, idx[1:]):
        if b - a == 1:
            dfx.append(abs(frames[b]["K"][0][0] - frames[a]["K"][0][0]))
            drot.append(_geodesic_deg(frames[a]["R"], frames[b]["R"]))
    dfx = np.array(dfx)
    drot = np.array(drot)

    m = {
        "frames": len(idx),
        "span": (idx[0], idx[-1]),
        "fx_min": float(fx.min()),
        "fx_max": float(fx.max()),
        "fx_median": float(np.median(fx)),
        "fx_over_6000": int((fx > 6000).sum()),
        "drot_median": float(np.median(drot)),
        "drot_p95": float(np.percentile(drot, 95)),
        "drot_max": float(drot.max()),
        "drot_gt2": int((drot > 2).sum()),
        "drot_gt5": int((drot > 5).sum()),
        "dfx_p95": float(np.percentile(dfx, 95)),
        "dfx_max": float(dfx.max()),
        "conf_mean": float(conf.mean()),
    }

    if lines_path:
        dl = json.load(open(lines_path))["frames"]
        counts = {int(k): len(v["lines"]) for k, v in dl.items()}
        span = range(idx[0], idx[-1] + 1)
        nl = np.array([counts.get(f, 0) for f in span])
        namefreq = Counter()
        for v in dl.values():
            for ln in v["lines"]:
                namefreq[ln["name"]] += 1
        nrec = len(dl)
        m["lines_per_frame_mean"] = float(nl.mean())
        m["frames_zero_lines"] = int((nl == 0).sum())
        m["frames_ge4_lines"] = int((nl >= 4).sum())
        m["far_touchline_pct"] = round(
            100 * namefreq.get("far_touchline", 0) / max(1, nrec), 1
        )
        m["line_name_freq"] = {
            k: round(100 * c / max(1, nrec), 0) for k, c in namefreq.most_common()
        }
    return m


def to_markdown(label: str, m: dict) -> str:
    lines = [
        f"# Camera quality — {label}",
        "",
        f"- frames: {m['frames']}  span {m['span'][0]}..{m['span'][1]}",
        f"- focal fx: median {m['fx_median']:.0f}  range [{m['fx_min']:.0f}, {m['fx_max']:.0f}]  (frames fx>6000: {m['fx_over_6000']})",
        "",
        "## Temporal jitter (consecutive frames)",
        f"- rotation deg/frame: median {m['drot_median']:.3f}  p95 {m['drot_p95']:.2f}  max {m['drot_max']:.2f}",
        f"- focal px/frame:     p95 {m['dfx_p95']:.1f}  max {m['dfx_max']:.0f}",
        f"- glitch frames: Δrot>2° = {m['drot_gt2']}   Δrot>5° = {m['drot_gt5']}",
        f"- mean confidence: {m['conf_mean']:.3f}",
    ]
    if "lines_per_frame_mean" in m:
        lines += [
            "",
            "## Line detection",
            f"- lines/frame mean: {m['lines_per_frame_mean']:.2f}",
            f"- frames with 0 detected lines: {m['frames_zero_lines']}",
            f"- frames with ≥4 lines (solved): {m['frames_ge4_lines']}",
            f"- far_touchline detected on: {m['far_touchline_pct']}% of recorded frames",
            "",
            "### per-line detection %",
            *[f"  - {k}: {int(v)}%" for k, v in m["line_name_freq"].items()],
        ]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", required=True)
    ap.add_argument("--lines", default=None)
    ap.add_argument("--label", default="camera")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    m = compute(args.track, args.lines)
    md = to_markdown(args.label, m)
    print(md)
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(md)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
