"""Stride-cadence probe: per tracked player, the stride frequency from ankle keypoints (ViTPose-small).

stride_probe.py <output_dir> <shot> [max_tracks]
Signal per track: signed ankle separation (L - R ankle, projected on its principal axis) / box height.
Its period is one STRIDE (two steps); step frequency = 2 / period.
"""
import sys
from pathlib import Path

import cv2
import numpy as np
from mmpose.apis import inference_topdown, init_model
from scipy.signal import detrend
from ultralytics import YOLO

CK = Path("/Users/joebower/workplace/football-perspectives/checkpoints")
out_dir, shot = Path(sys.argv[1]), sys.argv[2]
max_tracks = int(sys.argv[3]) if len(sys.argv) > 3 else 12
video = str(out_dir / "shots" / f"{shot}.mp4")
fps = cv2.VideoCapture(video).get(cv2.CAP_PROP_FPS) or 25.0
det = YOLO("/Users/joebower/workplace/football-perspectives/yolov8x.pt")
pose = init_model(str(CK / "td-hm_ViTPose-small_8xb64-210e_coco-256x192.py"),
                  str(CK / "td-hm_ViTPose-small_8xb64-210e_coco-256x192-62d7a712_20230314.pth"), device="cpu")

tracks: dict[int, list] = {}
frame = -1
for r in det.track(source=video, classes=[0], conf=0.3, persist=True, stream=True, verbose=False,
                   tracker="bytetrack.yaml"):
    frame += 1
    if r.boxes is None or r.boxes.id is None:
        continue
    boxes = r.boxes.xyxy.cpu().numpy()
    ids = r.boxes.id.cpu().numpy().astype(int)
    keep = (boxes[:, 3] - boxes[:, 1]) > 40
    boxes, ids = boxes[keep], ids[keep]
    if not len(boxes):
        continue
    res = inference_topdown(pose, r.orig_img, boxes, bbox_format="xyxy")
    for b, tid, pr in zip(boxes, ids, res):
        kp = pr.pred_instances.keypoints[0]
        sc = pr.pred_instances.keypoint_scores[0]
        tracks.setdefault(int(tid), []).append((frame, b, kp[15], kp[16], min(sc[15], sc[16])))


def period_frames(sig: np.ndarray, fps: float) -> tuple[float, float]:
    sig = detrend(sig)
    ac = np.correlate(sig, sig, mode="full")[len(sig) - 1:]
    if ac[0] <= 0:
        return 0.0, 0.0
    ac = ac / ac[0]
    hi = min(len(ac) - 2, int(fps / 0.25))
    neg = np.nonzero(ac[1:hi] < 0)[0]
    if not len(neg):
        return 0.0, 0.0
    s = neg[0] + 1
    for k in range(s + 1, hi):
        if ac[k] >= ac[k - 1] and ac[k] >= ac[k + 1] and ac[k] > 0:
            a, b, c = ac[k - 1], ac[k], ac[k + 1]
            den = a - 2 * b + c
            frac = 0.5 * (a - c) / den if abs(den) > 1e-9 else 0.0
            return k + frac, float(b)
    return 0.0, 0.0


out = []
for tid, rows in tracks.items():
    fr = np.array([r[0] for r in rows])
    if len(rows) < 16:
        continue
    cuts = np.nonzero(np.diff(fr) > 2)[0]
    seg = max(np.split(np.arange(len(rows)), cuts + 1), key=len)
    rows = [rows[i] for i in seg]
    if len(rows) < 16:
        continue
    h = np.array([r[1][3] - r[1][1] for r in rows])
    d = np.array([np.asarray(r[2]) - np.asarray(r[3]) for r in rows])  # L - R ankle
    conf = np.array([r[4] for r in rows])
    if np.median(conf) < 0.4:
        continue
    # principal axis of the separation vectors
    u = np.linalg.svd(d - d.mean(0), full_matrices=False)[2][0]
    sig = (d @ u) / h
    per, strength = period_frames(sig, fps)
    if per <= 0:
        continue
    stride_hz = fps / per
    out.append((tid, len(rows), float(np.median(h)), 2 * stride_hz, strength, float(np.std(sig))))
out.sort(key=lambda r: -r[4])
print(f"{shot} fps {fps:.0f} tracks {len(out)}")
for tid, n, h, step, s, amp in out[:max_tracks]:
    print(f"  track {tid:3d} n {n:3d} h {h:5.0f}px  step {step:4.2f} Hz  ac {s:.2f}  amp {amp:.3f}")
good = [r for r in out if r[4] > 0.3 and r[5] > 0.05]
if good:
    w = np.array([r[4] for r in good])
    print(f"  weighted step freq (ac>0.3, amp>0.05): {np.average([r[3] for r in good], weights=w):.2f} Hz "
          f"median {np.median([r[3] for r in good]):.2f} over {len(good)} tracks")
