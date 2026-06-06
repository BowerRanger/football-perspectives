"""Compact per-clip eval for the camera iteration: span, jitter, line-RMS,
0-line count, bad-solve outliers. Usage: python scripts/_clip_eval.py BASE LABEL NCLIP."""
import json
import os
import sys

import numpy as np

from src.utils.camera_projection import project_world_to_image


def ev(base, label, clip):
    if not os.path.exists(f"{base}_camera_track.json"):
        print(f"### {label}: NO TRACK")
        return
    tr = json.load(open(f"{base}_camera_track.json"))
    cam = {f["frame"]: f for f in tr["frames"]}
    fr = sorted(cam)
    dl = (
        json.load(open(f"{base}_detected_lines.json"))
        if os.path.exists(f"{base}_detected_lines.json") else {"frames": {}}
    )

    def g(a, b):
        c = (np.trace(np.array(a).T @ np.array(b)) - 1) / 2
        return np.degrees(np.arccos(max(-1.0, min(1.0, c))))

    dr = [g(cam[a]["R"], cam[b]["R"]) for a, b in zip(fr, fr[1:]) if b - a == 1]
    dist = tuple(tr.get("distortion", (0, 0))[:2])
    res, perf = [], {}
    for fk, fv in dl["frames"].items():
        f = int(fk)
        if f not in cam:
            continue
        K = np.array(cam[f]["K"]); R = np.array(cam[f]["R"]); t = np.array(cam[f]["t"])
        rr = []
        for ln in fv["lines"]:
            proj = project_world_to_image(K, R, t, dist, np.array(ln["world_segment"]))
            pa, pb = proj[0], proj[1]
            d = pb - pa
            nn = np.array([-d[1], d[0]])
            if np.linalg.norm(nn) < 1e-6:
                continue
            nn = nn / np.linalg.norm(nn)
            for ip in ln["image_segment"]:
                rr.append(abs(np.dot(np.array(ip) - pa, nn)))
        res += rr
        if rr:
            perf[f] = float(np.mean(rr))
    res = np.array(res) if res else np.array([np.nan])
    span = range(fr[0], fr[-1] + 1)
    zero = sum(1 for f in span if len(dl["frames"].get(str(f), {}).get("lines", [])) == 0)
    circ = sum(
        1 for v in dl["frames"].values() for ln in v["lines"] if "circle" in ln["name"]
    )
    print(
        f"### {label} ({clip}f): {len(fr)}f span {fr[0]}..{fr[-1]} | "
        f"jit max {max(dr):.2f} p95 {np.percentile(dr, 95):.2f} | "
        f"RMS mean {np.nanmean(res):.2f} med {np.nanmedian(res):.2f} "
        f"<1px {np.mean(res < 1):.0%} | 0-line {zero} | "
        f"out>5px {sum(1 for v in perf.values() if v > 5)} | circle-frames {circ}"
    )


if __name__ == "__main__":
    ev(sys.argv[1], sys.argv[2], int(sys.argv[3]))
