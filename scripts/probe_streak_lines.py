"""Identify catalogue lines that project as cross-image streaks for a frame.

For each named straight catalogue line: camera-space z of endpoints, projected
endpoints, and the normalized radius vs the distortion polynomial's monotonic
fold radius — points beyond the fold get pulled back INSIDE the image and
render as garbage streaks.

Usage: .venv/bin/python scripts/probe_streak_lines.py BASE FRAME
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.camera_projection import project_world_to_image  # noqa: E402
from src.utils.line_camera_refine import PITCH_LINE_CATALOGUE  # noqa: E402


def fold_radius(k1: float, k2: float) -> float:
    """Largest r where r*(1 + k1 r^2 + k2 r^4) is still increasing."""
    # d/dr = 1 + 3 k1 r^2 + 5 k2 r^4 = 0 -> quadratic in r^2
    if abs(k2) < 1e-12:
        if k1 >= 0:
            return float("inf")
        return float(np.sqrt(-1.0 / (3 * k1)))
    a, b, c = 5 * k2, 3 * k1, 1.0
    disc = b * b - 4 * a * c
    if disc < 0:
        return float("inf")
    roots = [(-b + s * np.sqrt(disc)) / (2 * a) for s in (+1, -1)]
    r2 = [r for r in roots if r > 0]
    return float(np.sqrt(min(r2))) if r2 else float("inf")


def main() -> None:
    base, fid = sys.argv[1], int(sys.argv[2])
    track = json.load(open(base + "_camera_track.json"))
    cams = {f["frame"]: f for f in track["frames"]}
    c = cams[fid]
    K = np.array(c["K"]); R = np.array(c["R"]); t = np.array(c["t"])
    k1, k2 = track.get("distortion", (0, 0))[:2]
    rf = fold_radius(k1, k2)
    w, h = track.get("image_size", (1920, 1080))
    print(f"f{fid}: dist=({k1:+.3f},{k2:+.3f}) fold radius={rf:.3f} "
          f"(normalized); image {w}x{h} fx={K[0][0]:.0f}")

    for name, seg in PITCH_LINE_CATALOGUE.items():
        pts = np.linspace(np.array(seg[0], float), np.array(seg[1], float), 9)
        cam = pts @ R.T + t
        z = cam[:, 2]
        if (z <= 0.1).all():
            continue
        rad = np.where(z > 0.1,
                       np.linalg.norm(cam[:, :2], axis=1) / np.maximum(z, 1e-9),
                       np.inf)
        proj = project_world_to_image(K, R, t, (k1, k2), pts)
        inside = ((proj[:, 0] >= 0) & (proj[:, 0] < w)
                  & (proj[:, 1] >= 0) & (proj[:, 1] < h)
                  & np.isfinite(proj).all(axis=1))
        beyond = (rad > rf) & (z > 0.1)
        flag = " STREAK-RISK" if (beyond & inside).any() else ""
        ips = "; ".join(
            f"({p[0]:.0f},{p[1]:.0f})" for p in proj[[0, 4, 8]])
        print(f"  {name:<26} z {z.min():7.1f}..{z.max():7.1f}  "
              f"rad max {np.nanmax(rad[np.isfinite(rad)]):6.2f}  "
              f"in-image pts {int(inside.sum())}/9  proj[0,4,8]={ips}{flag}")


if __name__ == "__main__":
    main()
