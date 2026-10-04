"""Player-tempo probe: camera-compensated player motion in body-heights per second.

tempo_probe.py <output_dir> <shot> [start end]  -> prints per-shot stats, writes <shot>_tempo.json
Per frame pair: background homography (RANSAC on corners outside person boxes) cancels pan/zoom;
corners inside each person box give residual motion; normalised by box height.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

out_dir, shot = Path(sys.argv[1]), sys.argv[2]
start = int(sys.argv[3]) if len(sys.argv) > 3 else 0
end = int(sys.argv[4]) if len(sys.argv) > 4 else 10 ** 9
cap = cv2.VideoCapture(str(out_dir / "shots" / f"{shot}.mp4"))
fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
model = YOLO("/Users/joebower/workplace/football-perspectives/yolov8n.pt")
W = 960
prev = prev_boxes = None
rows = []
i = -1
while True:
    ok, img = cap.read()
    i += 1
    if not ok or i > end:
        break
    if i < start:
        continue
    s = W / img.shape[1]
    small = cv2.resize(img, (W, int(img.shape[0] * s)))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    det = model.predict(small, classes=[0], conf=0.35, verbose=False)[0]
    boxes = det.boxes.xyxy.cpu().numpy() if det.boxes is not None else np.zeros((0, 4))
    boxes = boxes[(boxes[:, 3] - boxes[:, 1]) > 18] if len(boxes) else boxes
    if prev is not None and len(prev_boxes):
        mask_bg = np.full(prev.shape, 255, np.uint8)
        for x0, y0, x1, y1 in prev_boxes.astype(int):
            mask_bg[max(0, y0 - 4):y1 + 4, max(0, x0 - 4):x1 + 4] = 0
        bg = cv2.goodFeaturesToTrack(prev, 400, 0.01, 8, mask=mask_bg)
        if bg is not None and len(bg) >= 12:
            nb, st, _ = cv2.calcOpticalFlowPyrLK(prev, gray, bg, None)
            good = st.reshape(-1) == 1
            H, inl = cv2.findHomography(bg[good], nb[good], cv2.RANSAC, 2.0) if good.sum() >= 8 else (None, None)
            if H is not None:
                per = []
                for x0, y0, x1, y1 in prev_boxes:
                    h = y1 - y0
                    m = np.zeros(prev.shape, np.uint8)
                    m[int(y0):int(y1), int(x0):int(x1)] = 255
                    pts = cv2.goodFeaturesToTrack(prev, 40, 0.01, 3, mask=m)
                    if pts is None or len(pts) < 4:
                        continue
                    npts, st2, _ = cv2.calcOpticalFlowPyrLK(prev, gray, pts, None)
                    g2 = st2.reshape(-1) == 1
                    if g2.sum() < 4:
                        continue
                    pred = cv2.perspectiveTransform(pts[g2], H)
                    res = np.linalg.norm((npts[g2] - pred).reshape(-1, 2), axis=1)
                    per.append(float(np.median(res)) / h)  # heights per frame
                if per:
                    per = np.sort(per)
                    rows.append({"frame": i, "n": len(per), "med": float(np.median(per)),
                                 "p75": float(np.percentile(per, 75)), "max": float(per[-1])})
    prev, prev_boxes = gray, boxes
cap.release()
med = np.array([r["med"] for r in rows]) * fps
p75 = np.array([r["p75"] for r in rows]) * fps
mx = np.array([r["max"] for r in rows]) * fps
print(f"{shot} frames {len(rows)} fps {fps:.0f} tempo(h/s) med-of-med {np.median(med):.3f} "
      f"med-of-p75 {np.median(p75):.3f} med-of-max {np.median(mx):.3f} mean-n {np.mean([r['n'] for r in rows]):.1f}")
Path(f"/Users/joebower/.claude/jobs/88133eb1/tmp/speed/{shot}_tempo.json").write_text(json.dumps({"fps": fps, "rows": rows}))
