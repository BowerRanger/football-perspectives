"""Semi-synthetic slow motion from a REAL calibrated replay: per-track pitch positions of origi02
(real tracking + camera noise) re-sampled at rate r (replay frame j = native time offset + r*j),
then estimated against the real live shot origi01."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/Users/joebower/workplace/football-perspectives/.claude/worktrees/gberch-shorts")
from src.schemas.camera_track import CameraTrack  # noqa: E402
from src.utils import replay_speed as rs  # noqa: E402
from src.utils.camera_projection import pixel_ray  # noqa: E402

OUT = Path("/Users/joebower/workplace/football-perspectives/output-origi-shorts")


def per_track(shot):
    tr = CameraTrack.load(OUT / "camera" / f"{shot}_camera_track.json")
    by = {f.frame: f for f in tr.frames}
    tracks = json.loads((OUT / "tracks" / f"{shot}_tracks.json").read_text())["tracks"]
    out = []
    for t in tracks:
        if t.get("class_name") not in ("player", "goalkeeper", "referee"):
            continue
        rows = {}
        for fr in t["frames"]:
            f = by.get(fr["frame"])
            if f is None or fr.get("interpolated"):
                continue
            x0, y0, x1, y1 = fr["bbox"]
            C, d = pixel_ray(((x0 + x1) / 2, y1), np.array(f.K), np.array(f.R),
                             np.array(f.t if f.t is not None else tr.t_world), tuple(tr.distortion))
            if d[2] >= 0:
                continue
            p = C - C[2] / d[2] * d
            rows[fr["frame"]] = p[:2]
        if len(rows) > 5:
            out.append(rows)
    return out


def resample(tracks, times):
    rep = {}
    for j, t in enumerate(times):
        pts = []
        for rows in tracks:
            a, b = int(np.floor(t)), int(np.floor(t)) + 1
            if a in rows and b in rows:
                w = t - a
                pts.append((1 - w) * rows[a] + w * rows[b])
        if pts:
            rep[j] = np.array(pts)
    return rep


def live_points():
    return {f: p for f, p in rs.feet_on_pitch(
        json.loads((OUT / "tracks" / "origi01_tracks.json").read_text()), _cam("origi01")).items()}


def _cam(shot):
    tr = CameraTrack.load(OUT / "camera" / f"{shot}_camera_track.json")
    by = {f.frame: f for f in tr.frames}
    return lambda fi: (lambda f: None if f is None else (f.K, f.R, f.t if f.t is not None else tr.t_world,
                                                         tuple(tr.distortion)))(by.get(fi))


tracks = per_track("origi02")
live = live_points()
native0 = 120.0   # start inside origi02; origi02 native f -> origi01 f + ~143.7*... (real time, rate ~0.991)
for label, rates in [("1.00x", [1.0] * 200), ("0.34x", [0.34] * 300), ("0.50x", [0.5] * 240), ("0.20x", [0.2] * 400),
                     ("ramp 0.27->0.41", [0.27] * 150 + [0.41] * 150)]:
    times = native0 + np.concatenate([[0.0], np.cumsum(rates[:-1])])
    times = times[times < 330]
    rep = resample(tracks, times)
    est = rs.estimate_speed(live, rep)
    true_rate = 0.991 * rates[0]
    # live frame of replay frame 0: origi02 native frame 120 -> origi01 frame 120 + 143.7 (r~0.991)
    true_off = 143.7 + 0.991 * native0
    print(f"{label:16s} est rate {est.rate:.3f} (true {true_rate:.3f})  offset {est.offset:.1f} (true {true_off:.1f})"
          f"  conf {est.confidence:.2f} cost {est.cost_m:.2f}  ramp {est.ramp} halves {est.rate_first:.3f}/{est.rate_second:.3f}")
