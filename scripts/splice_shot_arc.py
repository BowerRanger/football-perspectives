"""PROTOTYPE (gberch shorts, gap G16) — Splice the bounded-curl two-knot shot arc (strike 371 -> line cross 394
at the operator anchor ray, curl 8 m/s^2) into the pipeline ball track.

Arc runs until it meets the back netting (x <= -NET_DEPTH); from there it
blends linearly into the pipeline's own post-impact track at frame 404.
Gap log G16 — a manual step outside the pipeline."""
import json
import shutil
import sys

import numpy as np

out = sys.argv[1]
ns: dict = {}
src = open(__file__.replace("splice_shot_arc.py", "fit_shot_arc.py")).read().split("best = None")[0]
sys.argv = ["x", out, "/dev/null"]
exec(src, ns)

X_END, CURL = 0.0, 8.0
NET_DEPTH = 1.95
SIDE_NET_Y = 37.55
BLEND_TO = 404
v0 = ns["solve"](X_END, CURL)
arc = ns["simulate"](v0, CURL, 410)
impact = next(f for f in range(371, 411) if arc[f][0] <= -NET_DEPTH or (arc[f][0] < 0 and arc[f][1] >= SIDE_NET_Y))

path = f"{out}/ball/gberch_ball_track.json"
backup = f"{out}/ball/gberch_ball_track.pipeline.json"
import os
if not os.path.exists(backup):  # keep the pristine pipeline track; re-runs splice from it
    shutil.copy(path, backup)
track = json.load(open(backup))
frames = {f["frame"]: f for f in track["frames"]}
p_imp = np.array(arc[impact])
p_end = np.array(frames[BLEND_TO]["world_xyz"])
for f in range(371, BLEND_TO):
    if f <= impact:
        p = arc[f]
    else:
        a = (f - impact) / (BLEND_TO - impact)
        p = (1 - a) * p_imp + a * p_end
    frames[f]["world_xyz"] = [float(v) for v in p]
    frames[f]["state"] = "flight"
    frames[f]["confidence"] = 0.9
json.dump(track, open(path, "w"))
print("v0", np.round(v0, 2), "speed km/h", round(float(np.linalg.norm(v0)) * 3.6, 1),
      "impact frame", impact, np.round(p_imp, 2))
for f in (371, 380, 390, 394, impact, 400, 404, 410):
    print(f, np.round(frames[f]["world_xyz"], 2))
