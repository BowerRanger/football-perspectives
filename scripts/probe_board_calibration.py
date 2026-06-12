"""Run the hoarding (d, h) calibration on a real clip's solved track and
visualise the result.

Usage: .venv/bin/python scripts/probe_board_calibration.py BASE f1,f2,... OUT.jpg
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.camera_projection import project_world_to_image  # noqa: E402
from src.utils.hoarding_detector import (  # noqa: E402
    calibrate_board_line,
    detect_board_line,
)


def main() -> None:
    base, frames_s, out = sys.argv[1], sys.argv[2], sys.argv[3]
    fids = [int(x) for x in frames_s.split(",")]
    track = json.load(open(base + "_camera_track.json"))
    cams_all = {f["frame"]: f for f in track["frames"]}
    dist = tuple(track.get("distortion", (0, 0))[:2])

    vid = base.replace("/camera/", "/shots/") + ".mp4"
    cap = cv2.VideoCapture(vid)
    frames = {}
    for fid in fids:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
        ok, img = cap.read()
        if ok and fid in cams_all:
            frames[fid] = img
    cap.release()

    cams = {fid: cams_all[fid] for fid in frames}
    import time
    t0 = time.time()
    model = calibrate_board_line(frames, cams, dist)
    print(f"calibration took {time.time() - t0:.1f}s")
    if model is None:
        print("calibration FAILED (no consistent board line)")
        return
    print(f"board line: d={model.d:.2f}m  h={model.h:.2f}m  "
          f"frames={model.frames}  contrast={model.contrast:.0f}  "
          f"residual={model.residual:.2f}px")

    # visualise on the first frame: detected points (red) + model line (cyan)
    fid = fids[0]
    img = frames[fid].copy()
    c = cams[fid]
    K = np.asarray(c["K"]); R = np.asarray(c["R"]); t = np.asarray(c["t"])
    det = detect_board_line(img, K, R, t, dist, model.d, model.h)
    xs = np.linspace(0, 105, 128)
    world = np.stack([xs, np.full_like(xs, 68.0 + model.d),
                      np.full_like(xs, model.h)], axis=1)
    cam = world @ R.T + t
    proj = project_world_to_image(K, R, t, dist, world)
    for i in range(127):
        if cam[i, 2] > 1 and cam[i + 1, 2] > 1:
            a, b = proj[i], proj[i + 1]
            if np.isfinite(a).all() and np.isfinite(b).all() and abs(a[0]) < 4000:
                cv2.line(img, tuple(np.round(a).astype(int)),
                         tuple(np.round(b).astype(int)), (255, 255, 0), 2)
    if det is not None:
        print(f"f{fid}: {len(det.image_points)} edge points, "
              f"contrast {det.contrast:.0f}")
        for (u, v) in det.image_points:
            cv2.circle(img, (int(round(u)), int(round(v))), 3, (0, 0, 255), -1)
    cv2.imwrite(out, img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
