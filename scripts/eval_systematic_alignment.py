"""The honest pitch-alignment metric: per-frame SYSTEMATIC camera offset.

Individual click reprojection conflates two things: the camera's true offset
from the painted pitch (what matters for player placement) and per-click
placement scatter (~5-10 px at far field — the metric's noise floor). This
separates them: per anchor frame, the MEAN SIGNED residual vector over far
landmarks is the systematic camera offset; the scatter around it is click
noise. Coherence = |mean vector| / mean |residual| (1.0 = fully systematic,
~0 = pure noise).

Usage: .venv/bin/python scripts/eval_systematic_alignment.py BASE [min_depth_m]
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.camera_projection import project_world_to_image  # noqa: E402


def main() -> None:
    base = sys.argv[1]
    min_z = float(sys.argv[2]) if len(sys.argv) > 2 else 80.0
    anchors = json.load(open(base + "_anchors__manual.json"))
    track = json.load(open(base + "_camera_track.json"))
    cams = {f["frame"]: f for f in track["frames"]}
    dist = tuple(track.get("distortion", (0, 0))[:2])

    worst_m = 0.0
    print(f"{base}  (k1={dist[0]:+.3f}, far = Z>{min_z:.0f}m)")
    for a in anchors["anchors"]:
        c = cams.get(a["frame"])
        if c is None:
            continue
        K = np.array(c["K"]); R = np.array(c["R"]); t = np.array(c["t"])
        vecs, mags, zs = [], [], []
        for lm in a.get("landmarks", []):
            w = np.array(lm["world_xyz"], float)
            Z = float((R @ w + t)[2])
            if Z <= min_z:
                continue
            p = project_world_to_image(K, R, t, dist, w[None])[0]
            d = p - np.array(lm["image_xy"], float)
            vecs.append(d); mags.append(float(np.linalg.norm(d))); zs.append(Z)
        if len(vecs) < 3:
            continue
        mv = np.mean(vecs, axis=0)
        sys_px = float(np.linalg.norm(mv))
        sys_m = sys_px * float(np.mean(zs)) / float(K[0, 0])
        coher = sys_px / max(1e-6, float(np.mean(mags)))
        worst_m = max(worst_m, sys_m)
        print(f"  f{a['frame']:>3}: systematic {sys_px:5.1f}px = {sys_m:.2f}m"
              f"  (coherence {coher:.2f}, n={len(vecs)},"
              f" scatter {np.mean(mags):.1f}px)")
    bar = 0.3048
    print(f"  WORST systematic offset: {worst_m:.2f} m"
          f"  -> {'WITHIN' if worst_m <= bar else 'ABOVE'} the 1-foot bar")


if __name__ == "__main__":
    main()
