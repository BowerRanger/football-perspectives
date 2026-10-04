"""Per-shot cadence signature: duplicate ratio, blend-frame ratio, motion rate.

A blend frame b is ~ (a + c) / 2 of its neighbours (frame-blend slow-mo).
Usage: cadence_scan.py <output_dir> [shot ...]  (default: all kept shots)
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

out = Path(sys.argv[1])
m = json.loads((out / "shots" / "shots_manifest.json").read_text())
want = set(sys.argv[2:])
shots = [s for s in m["shots"] if (s["id"] in want) or (not want and not s.get("excluded"))]
for s in shots:
    cap = cv2.VideoCapture(str(out / "shots" / f"{s['id']}.mp4"))
    frames = []
    while True:
        ok, img = cap.read()
        if not ok:
            break
        frames.append(cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (320, 180)).astype(np.float32))
    cap.release()
    if len(frames) < 3:
        continue
    d = np.array([np.abs(frames[i] - frames[i - 1]).mean() for i in range(1, len(frames))])
    dup = (d < 0.6).mean()
    blend = []
    for i in range(1, len(frames) - 1):
        a, b, c = frames[i - 1], frames[i], frames[i + 1]
        mid = 0.5 * (a + c)
        e_mid = np.abs(b - mid).mean()
        e_nb = min(np.abs(b - a).mean(), np.abs(b - c).mean())
        blend.append(e_nb > 1.0 and e_mid < 0.35 * e_nb)
    print(f"{s['id']:6s} grp {s.get('group_id') or '-':4s} n {len(frames):4d} dup {dup:.2f} "
          f"blend {np.mean(blend):.2f} meddiff {np.median(d):5.2f} sf {s.get('speed_factor', 1):.2f}")
