"""Honest accuracy check: reproject hand-clicked anchor landmarks under a
camera track. The manual TRACK can be wrong (bad solo anchors), but the
clicked pixels are user ground truth wherever they exist.

Usage:
  .venv/bin/python scripts/eval_anchor_clicks.py ANCHORS.json TRACK.json
e.g.
  .venv/bin/python scripts/eval_anchor_clicks.py \
      output-origi/camera/origi01_anchors__manual.json \
      output-origi/camera/origi01_camera_track.json
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.camera_projection import project_world_to_image  # noqa: E402


def main() -> None:
    anchors = json.load(open(sys.argv[1]))
    track = json.load(open(sys.argv[2]))
    cams = {f["frame"]: f for f in track["frames"]}
    dist = tuple(track.get("distortion", (0, 0))[:2])

    all_res = []
    print(f"track span {min(cams)}..{max(cams)}  dist={np.round(dist, 4).tolist()}")
    for a in anchors["anchors"]:
        f = a["frame"]
        lm = a.get("landmarks", [])
        if f not in cams:
            print(f"  anchor f{f:>3}: NOT COVERED ({len(lm)} clicks)")
            continue
        K = np.array(cams[f]["K"]); R = np.array(cams[f]["R"])
        t = np.array(cams[f]["t"])
        res = []
        for ob in lm:
            w = np.array([ob["world_xyz"]], dtype=float)
            p = project_world_to_image(K, R, t, dist, w)[0]
            res.append(float(np.linalg.norm(p - np.array(ob["image_xy"]))))
        if res:
            all_res += res
            print(f"  anchor f{f:>3}: {len(res)} clicks  reproj "
                  f"med {np.median(res):6.1f}px  max {max(res):6.1f}px")
    if all_res:
        print(f"\nALL: {len(all_res)} clicks  med {np.median(all_res):.1f}px  "
              f"p90 {np.percentile(all_res, 90):.1f}px  max {max(all_res):.1f}px")


if __name__ == "__main__":
    main()
