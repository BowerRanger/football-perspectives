"""Sample per-player kit colours (shirt / shorts / socks) from broadcast footage.

Prototype for the "kit palette from evidence" pipeline gap: reads a shot's
tracks, crops each non-interpolated detector bbox, masks out pitch green,
and takes the per-zone median colour (zones = bbox height bands). Frames
where the bbox overlaps another track are skipped (occlusion bleed).

Per-role aggregates use ``players.json`` kit roles, so the output can be
pasted into ``render.teams.defaults``.

Usage:
    python scripts/sample_kit_colors.py --output output/ --shot gberch
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

# bbox height fractions (0 = top of bbox, 1 = bottom)
ZONES = {
    "shirt": (0.20, 0.42),
    "shorts": (0.47, 0.58),
    "socks": (0.74, 0.88),
}
# central width fraction of the bbox sampled (arms/background bleed at edges)
X_BAND = (0.30, 0.70)
FRAME_STRIDE = 3
MIN_BBOX_H_PX = 60
MAX_OVERLAP_IOU = 0.02
MIN_ZONE_PIXELS = 12


def _iou(a, b) -> float:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    if inter == 0.0:
        return 0.0
    area = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / area


def _non_grass(pixels_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(pixels_bgr.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    grass = (hsv[:, 0] >= 30) & (hsv[:, 0] <= 85) & (hsv[:, 1] >= 60)
    return pixels_bgr.reshape(-1, 3)[~grass]


def _hex(bgr) -> str:
    b, g, r = (int(round(float(c))) for c in bgr)
    return f"#{r:02x}{g:02x}{b:02x}"


def sample(output_dir: Path, shot: str) -> dict:
    tracks = json.loads((output_dir / "tracks" / f"{shot}_tracks.json").read_text())["tracks"]
    by_frame: dict[int, list[tuple[str, list[float]]]] = {}
    for t in tracks:
        pid = t.get("player_id") or t["track_id"]
        for f in t["frames"]:
            if f.get("interpolated"):
                continue
            by_frame.setdefault(int(f["frame"]), []).append((pid, f["bbox"]))

    samples: dict[str, dict[str, list[np.ndarray]]] = {}
    cap = cv2.VideoCapture(str(output_dir / "shots" / f"{shot}.mp4"))
    idx = 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        if idx % FRAME_STRIDE == 0:
            boxes = by_frame.get(idx, [])
            for pid, bb in boxes:
                if bb[3] - bb[1] < MIN_BBOX_H_PX:
                    continue
                if any(_iou(bb, ob) > MAX_OVERLAP_IOU for opid, ob in boxes if opid != pid):
                    continue
                x0, y0, x1, y1 = (int(round(v)) for v in bb)
                h, w = y1 - y0, x1 - x0
                cx0, cx1 = x0 + int(w * X_BAND[0]), x0 + int(w * X_BAND[1])
                for zone, (f0, f1) in ZONES.items():
                    crop = img[y0 + int(h * f0): y0 + int(h * f1), cx0:cx1]
                    if crop.size == 0:
                        continue
                    px = _non_grass(crop)
                    if len(px) < MIN_ZONE_PIXELS:
                        continue
                    samples.setdefault(pid, {}).setdefault(zone, []).append(
                        np.median(px, axis=0))
        idx += 1
    cap.release()

    per_player = {
        pid: {zone: _hex(np.median(np.stack(v), axis=0)) for zone, v in zones.items()}
        | {"n": min(len(v) for v in zones.values())}
        for pid, zones in samples.items()
    }
    roles = {}
    players_json = output_dir / "players.json"
    if players_json.exists():
        roles = {k: v.get("kit_role") for k, v in json.loads(players_json.read_text()).items()}
    per_role: dict[str, dict[str, str]] = {}
    for role in sorted({r for r in roles.values() if r}):
        pids = [p for p, r in roles.items() if r == role and p in samples]
        if not pids:
            continue
        per_role[role] = {
            zone: _hex(np.median(np.stack(
                [m for p in pids for m in samples[p].get(zone, [])]), axis=0))
            for zone in ZONES
        } | {"players": pids}
    return {"per_player": per_player, "per_role": per_role}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--shot", required=True)
    ap.add_argument("--json-out", type=Path)
    args = ap.parse_args()
    result = sample(args.output, args.shot)
    text = json.dumps(result, indent=2)
    if args.json_out:
        args.json_out.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
