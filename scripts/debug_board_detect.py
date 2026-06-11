"""Step through detect_board_line's gates on one frame/candidate and report
sample attrition.

Usage: .venv/bin/python scripts/debug_board_detect.py BASE FRAME D H
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.camera_projection import project_world_to_image  # noqa: E402
from src.utils.hoarding_detector import (  # noqa: E402
    BOARD_X0,
    BOARD_X1,
    _step_edge_offset,
)
from src.utils.line_detector import DetectorConfig, _prepare_frame  # noqa: E402


def main() -> None:
    base, fid, d, h = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
    track = json.load(open(base + "_camera_track.json"))
    cams = {f["frame"]: f for f in track["frames"]}
    c = cams[fid]
    K = np.asarray(c["K"]); R = np.asarray(c["R"]); t = np.asarray(c["t"])
    dist = tuple(track.get("distortion", (0, 0))[:2])
    vid = base.replace("/camera/", "/shots/") + ".mp4"
    cap = cv2.VideoCapture(vid)
    cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
    ok, img = cap.read()
    cap.release()

    cfg = DetectorConfig()
    gray, green = _prepare_frame(img, cfg)
    h_img, w_img = gray.shape
    n_samples, strip_px, min_step = 48, 30, 25.0

    xs = np.linspace(BOARD_X0, BOARD_X1, n_samples)
    world = np.stack([xs, np.full_like(xs, 68.0 + d), np.full_like(xs, h)], axis=1)
    cam = world @ R.T + t
    proj = project_world_to_image(K, R, t, dist, world)
    grass = world.copy(); grass[:, 1] -= 1.0; grass[:, 2] = 0.0
    proj_g = project_world_to_image(K, R, t, dist, grass)
    in_view = ((cam[:, 2] > 1.0) & np.isfinite(proj).all(axis=1)
               & (proj[:, 0] >= 0) & (proj[:, 0] < w_img)
               & (proj[:, 1] >= -strip_px) & (proj[:, 1] < h_img))
    print(f"in_view: {int(in_view.sum())}/{n_samples}")
    stats = dict(short=0, nostep=0, oob=0, polarity=0, ok=0)
    offsets = np.arange(-strip_px, strip_px + 1)
    for idx in np.where(in_view)[0]:
        centre = proj[idx]
        normal = proj_g[idx] - centre
        nn = np.linalg.norm(normal)
        if nn < 1e-6:
            continue
        normal = normal / nn
        sx = centre[0] + offsets * normal[0]
        sy = centre[1] + offsets * normal[1]
        okm = (sx >= 0) & (sx < w_img - 1) & (sy >= 0) & (sy < h_img - 1)
        if okm.sum() < 15:
            stats["short"] += 1
            continue
        x0 = np.floor(sx[okm]).astype(np.int32)
        y0 = np.floor(sy[okm]).astype(np.int32)
        fx = sx[okm] - x0; fy = sy[okm] - y0
        g = (gray[y0, x0] * (1 - fx) * (1 - fy)
             + gray[y0, x0 + 1] * fx * (1 - fy)
             + gray[y0 + 1, x0] * (1 - fx) * fy
             + gray[y0 + 1, x0 + 1] * fx * fy)
        hit = _step_edge_offset(g.astype(np.float32), min_step)
        if hit is None:
            stats["nostep"] += 1
            if idx % 8 == 0:
                kr = np.concatenate([np.full(6, 1 / 6), [0.0], np.full(6, -1 / 6)])
                resp = np.convolve(g, kr[::-1], mode="same")
                print(f"  sample {idx}: profile min/max {g.min():.0f}/{g.max():.0f} "
                      f"best step resp {np.nanmax(resp[7:-7]):.1f}")
            continue
        sub_idx, contrast = hit
        valid_offsets = offsets[okm].astype(np.float64)
        base_i = int(np.floor(sub_idx))
        if base_i < 0 or base_i >= len(valid_offsets) - 1:
            stats["oob"] += 1
            continue
        frac = sub_idx - base_i
        off = (1 - frac) * valid_offsets[base_i] + frac * valid_offsets[base_i + 1]
        u = float(centre[0] + off * normal[0]); v = float(centre[1] + off * normal[1])
        below = (int(round(u + 6 * normal[0])), int(round(v + 6 * normal[1])))
        above = (int(round(u - 6 * normal[0])), int(round(v - 6 * normal[1])))
        if not (0 <= below[0] < w_img and 0 <= below[1] < h_img
                and 0 <= above[0] < w_img and 0 <= above[1] < h_img):
            stats["oob"] += 1
            continue
        gb = green[below[1], below[0]]
        ga = green[above[1], above[0]]
        if gb == 0 or ga != 0:
            stats["polarity"] += 1
            if stats["polarity"] <= 3:
                print(f"  sample {idx}: step at ({u:.0f},{v:.0f}) contrast "
                      f"{contrast:.0f} green-below={gb} green-above={ga}")
            continue
        stats["ok"] += 1
    print(f"attrition: {stats}")


if __name__ == "__main__":
    main()
