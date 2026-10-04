"""Duplicate-frame (pulldown) scan over every active shot of the scratch output dirs."""
import json
from pathlib import Path

import cv2
import numpy as np

M = Path("/Users/joebower/workplace/football-perspectives")
rows = []
for d in ["output-shorts", "output-origi-shorts", "output-kroupi-shorts", "output-saka-shorts"]:
    for mp4 in sorted((M / d / "shots").glob("*.mp4")):
        cap = cv2.VideoCapture(str(mp4))
        fps = cap.get(cv2.CAP_PROP_FPS)
        prev, diffs = None, []
        while True:
            ok, img = cap.read()
            if not ok:
                break
            g = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (480, 270)).astype(np.int16)
            if prev is not None:
                diffs.append(float(np.abs(g - prev).mean()))
            prev = g
        diffs = np.array(diffs)
        dup = np.nonzero(diffs < 0.6)[0] + 1
        spacing = np.diff(dup)
        mode = int(np.bincount(spacing).argmax()) if len(spacing) else None
        rows.append({"dir": d, "shot": mp4.stem, "fps": round(fps, 3), "frames": len(diffs) + 1,
                     "dup": int(len(dup)), "dup_ratio": round(len(dup) / (len(diffs) + 1), 3), "spacing_mode": mode})
        print(rows[-1], flush=True)
Path("/Users/joebower/.claude/jobs/88133eb1/tmp/dup_scan.json").write_text(json.dumps(rows, indent=1))
