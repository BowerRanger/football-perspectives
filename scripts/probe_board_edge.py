"""Probe the advertising-board region's photometric structure.

For a frame, project candidate board lines (y = 68 + d, z = h) under the
track camera for a (d, h) grid and report the mean |vertical gradient| along
each projected line — where does the grass/board boundary actually sit, and
is it a step or a ridge?

Also dumps a crop of the board region with candidate lines drawn.

Usage:
  .venv/bin/python scripts/probe_board_edge.py BASE FRAME OUT.jpg
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.camera_projection import project_world_to_image  # noqa: E402


def main() -> None:
    base, fid_s, out = sys.argv[1], sys.argv[2], sys.argv[3]
    fid = int(fid_s)
    track = json.load(open(base + "_camera_track.json"))
    cams = {f["frame"]: f for f in track["frames"]}
    c = cams[fid]
    K = np.array(c["K"]); R = np.array(c["R"]); t = np.array(c["t"])
    dist = tuple(track.get("distortion", (0, 0))[:2])

    vid = base.replace("/camera/", "/shots/") + ".mp4"
    cap = cv2.VideoCapture(vid)
    cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
    ok, img = cap.read()
    cap.release()
    if not ok:
        print("no frame")
        return
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gy = cv2.Sobel(grey, cv2.CV_32F, 0, 1, ksize=3)
    h_img, w_img = grey.shape

    print(f"f{fid}: scanning board candidates (d = lateral m beyond far "
          "touchline, h = height m)")
    results = []
    for d in np.arange(0.0, 8.1, 0.5):
        for hh in (0.0, 0.5, 1.0):
            pts = np.stack([
                np.linspace(0, 105, 64),
                np.full(64, 68.0 + d),
                np.full(64, hh),
            ], axis=1)
            cam = pts @ R.T + t
            proj = project_world_to_image(K, R, t, dist, pts)
            vis = ((cam[:, 2] > 1.0) & np.isfinite(proj).all(axis=1)
                   & (proj[:, 0] >= 0) & (proj[:, 0] < w_img)
                   & (proj[:, 1] >= 0) & (proj[:, 1] < h_img))
            if vis.sum() < 8:
                continue
            uv = proj[vis]
            g = np.abs(gy[uv[:, 1].astype(int), uv[:, 0].astype(int)])
            results.append((float(np.mean(g)), d, hh, int(vis.sum())))
    results.sort(reverse=True)
    for g, d, hh, n in results[:12]:
        print(f"  |grad_y|={g:7.1f}  d={d:4.1f}  h={hh:.1f}  ({n} px)")

    # draw the top-3 candidates + the far touchline itself for reference
    colours = [(0, 0, 255), (0, 165, 255), (0, 255, 255)]
    for rank, (g, d, hh, n) in enumerate(results[:3]):
        pts = np.stack([
            np.linspace(0, 105, 128),
            np.full(128, 68.0 + d),
            np.full(128, hh),
        ], axis=1)
        cam = pts @ R.T + t
        proj = project_world_to_image(K, R, t, dist, pts)
        for i in range(127):
            if cam[i, 2] > 1 and cam[i + 1, 2] > 1:
                a = proj[i]; b = proj[i + 1]
                if np.isfinite(a).all() and np.isfinite(b).all():
                    if abs(a[0]) < 4000 and abs(b[0]) < 4000:
                        cv2.line(img, tuple(np.round(a).astype(int)),
                                 tuple(np.round(b).astype(int)),
                                 colours[rank], 2)
    ftl = np.stack([np.linspace(0, 105, 128), np.full(128, 68.0),
                    np.zeros(128)], axis=1)
    cam = ftl @ R.T + t
    proj = project_world_to_image(K, R, t, dist, ftl)
    for i in range(127):
        if cam[i, 2] > 1 and cam[i + 1, 2] > 1:
            a = proj[i]; b = proj[i + 1]
            if np.isfinite(a).all() and np.isfinite(b).all():
                if abs(a[0]) < 4000 and abs(b[0]) < 4000:
                    cv2.line(img, tuple(np.round(a).astype(int)),
                             tuple(np.round(b).astype(int)), (0, 255, 0), 1)
    cv2.imwrite(out, img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(f"wrote {out} (red/orange/yellow = top board candidates, "
          "green = far touchline)")


if __name__ == "__main__":
    main()
