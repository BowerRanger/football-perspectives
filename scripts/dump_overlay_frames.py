"""Dump clip frames as JPEGs with the projected pitch catalogue overlaid.

Usage:
  .venv/bin/python scripts/dump_overlay_frames.py BASE TRACKSUFFIX OUTDIR f1,f2,...
e.g.
  .venv/bin/python scripts/dump_overlay_frames.py output-origi/camera/origi01 \
      _track__manual.json /tmp/origi01_start 0,60,120,176
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.camera_projection import project_world_to_image  # noqa: E402
from src.utils.pitch_lines import pitch_polylines  # noqa: E402


def main() -> None:
    base, suffix, outdir, frames = sys.argv[1:5]
    fids = [int(x) for x in frames.split(",")]
    track = json.load(open(base + suffix))
    cams = {f["frame"]: f for f in track["frames"]}
    dist = tuple(track.get("distortion", (0, 0))[:2])
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
                proj = project_world_to_image(K, R, t, dist, np.asarray(pts))
                for a, b in zip(proj, proj[1:]):
                    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
                        continue
                    if abs(a[0]) > 1e5 or abs(b[0]) > 1e5:
                        continue
                    cv2.line(img, tuple(np.round(a).astype(int)),
                             tuple(np.round(b).astype(int)), (0, 255, 255), 2)
            fx = cams[fid]["K"][0][0]
            cv2.putText(img, f"f{fid} fx={fx:.0f}", (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
        else:
            cv2.putText(img, f"f{fid} (no camera)", (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        path = out / f"f{fid:04d}.jpg"
        cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        print(f"wrote {path}")
    cap.release()


if __name__ == "__main__":
    main()
