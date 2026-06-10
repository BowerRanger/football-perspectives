"""Dump clip frames as JPEGs with the full camera-debug overlay:

  YELLOW — projected pitch catalogue under the track camera (z-clipped)
  GREEN  — detected lines from <shot>_detected_lines.json (what the solver fit)
  RED    — anchor landmark clicks (manual or PnLCalib) for anchor frames

Usage:
  .venv/bin/python scripts/dump_overlay_frames.py BASE TRACKSUFFIX OUTDIR f1,f2,...
e.g.
  .venv/bin/python scripts/dump_overlay_frames.py output-origi/camera/origi01 \
      _camera_track.json /tmp/origi01 0,60,120
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.camera_projection import project_world_to_image  # noqa: E402
from src.utils.pitch_lines import pitch_polylines  # noqa: E402


def _fold_r2(k1: float, k2: float) -> float:
    """Largest normalized r^2 where the radial polynomial is monotonic;
    beyond it, far-outside points fold back INSIDE the image (streaks)."""
    if abs(k2) < 1e-12:
        return float("inf") if k1 >= 0 else -1.0 / (3 * k1)
    a, b, c = 5 * k2, 3 * k1, 1.0
    disc = b * b - 4 * a * c
    if disc < 0:
        return float("inf")
    roots = [(-b + s * np.sqrt(disc)) / (2 * a) for s in (+1, -1)]
    r2 = [r for r in roots if r > 0]
    return min(r2) if r2 else float("inf")


def _draw_polyline(img, K, R, t, dist, pts, colour, thickness=2):
    """Project a world polyline and draw only segments fully in FRONT of the
    camera, inside the distortion polynomial's monotonic radius, and with
    both endpoints inside a sane border around the image — distortion
    polynomials are garbage far outside the calibrated area and produce
    streaks across the frame."""
    pts = np.asarray(pts, dtype=float)
    K = np.asarray(K)
    cam = pts @ np.asarray(R).T + np.asarray(t)
    z = cam[:, 2]
    r2n = np.where(z > 0.5,
                   (cam[:, 0] ** 2 + cam[:, 1] ** 2) / np.maximum(z, 1e-9) ** 2,
                   np.inf)
    in_front = (z > 0.5) & (r2n < 0.9 * _fold_r2(dist[0], dist[1]))
    proj = project_world_to_image(K, R, t, dist, pts)
    h, w = img.shape[:2]
    m = 0.5 * max(w, h)
    for i in range(len(pts) - 1):
        if not (in_front[i] and in_front[i + 1]):
            continue
        a, b = proj[i], proj[i + 1]
        if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
            continue
        if (a[0] < -m or a[0] > w + m or a[1] < -m or a[1] > h + m
                or b[0] < -m or b[0] > w + m or b[1] < -m or b[1] > h + m):
            continue
        cv2.line(img, tuple(np.round(a).astype(int)),
                 tuple(np.round(b).astype(int)), colour, thickness)


def main() -> None:
    base, suffix, outdir, frames = sys.argv[1:5]
    fids = [int(x) for x in frames.split(",")]
    track = json.load(open(base + suffix))
    cams = {f["frame"]: f for f in track["frames"]}
    dist = tuple(track.get("distortion", (0, 0))[:2])

    dl_path = Path(base + "_detected_lines.json")
    detected = {}
    if dl_path.exists():
        detected = {
            int(k): v["lines"]
            for k, v in json.load(open(dl_path))["frames"].items()
        }

    anchors_by_frame = {}
    for suffix_a in ("_anchors.json", "_anchors__manual.json"):
        p = Path(base + suffix_a)
        if p.exists():
            for a in json.load(open(p))["anchors"]:
                anchors_by_frame.setdefault(a["frame"], []).extend(
                    a.get("landmarks", []))
            break

    vid = base.replace("/camera/", "/shots/") + ".mp4"
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    catalogue = pitch_polylines()
    cap = cv2.VideoCapture(vid)
    for fid in fids:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
        ok, img = cap.read()
        if not ok:
            print(f"f{fid}: no frame")
            continue
        if fid in cams:
            K = np.array(cams[fid]["K"]); R = np.array(cams[fid]["R"])
            t = np.array(cams[fid]["t"])
            for pts in catalogue:
                _draw_polyline(img, K, R, t, dist, pts, (0, 255, 255), 2)
            fx = cams[fid]["K"][0][0]
            label = f"f{fid} fx={fx:.0f} conf={cams[fid].get('confidence', 0):.2f}"
        else:
            label = f"f{fid} (no camera)"
        for ln in detected.get(fid, []):
            a = tuple(int(round(v)) for v in ln["image_segment"][0])
            b = tuple(int(round(v)) for v in ln["image_segment"][1])
            cv2.line(img, a, b, (0, 220, 0), 2)
        for ob in anchors_by_frame.get(fid, []):
            x, y = ob["image_xy"]
            cv2.circle(img, (int(round(x)), int(round(y))), 6, (0, 0, 255), -1)
        cv2.putText(img, label, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.3,
                    (0, 255, 255), 3)
        path = out / f"f{fid:04d}.jpg"
        cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 82])
        print(f"wrote {path}")
    cap.release()


if __name__ == "__main__":
    main()
