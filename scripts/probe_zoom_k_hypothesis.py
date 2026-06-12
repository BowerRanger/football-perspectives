"""Reality-check the zoom-dependent-distortion hypothesis on origi01.

For each manual anchor frame, solve (rvec, fx, k1, k2) at a FIXED candidate C
directly against the hand clicks (LM, point reprojection). If a per-frame
lens can fit the midfield clicks at <=10 px where the shared lens gives
100-180 px, zoom-dependent distortion explains the span and the k(fx) model
is worth building. Tested at both the bundle C and the anchor-stage C.

Usage: .venv/bin/python scripts/probe_zoom_k_hypothesis.py
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.camera_projection import project_world_to_image  # noqa: E402

BASE = "output-origi/camera/origi01"


def fit_frame(clicks, C, rvec0, fx0):
    """LM over (rvec, fx, k1, k2) at fixed C against clicked landmarks."""
    world = np.array([c["world_xyz"] for c in clicks], dtype=float)
    image = np.array([c["image_xy"] for c in clicks], dtype=float)

    def res(p):
        rvec = p[:3]
        fx = float(np.clip(p[3], 500, 20000))
        k1 = float(np.clip(p[4], -0.5, 0.5))
        k2 = float(np.clip(p[5], -0.5, 0.5))
        R, _ = cv2.Rodrigues(rvec)
        t = -R @ C
        K = np.array([[fx, 0, 960.0], [0, fx, 540.0], [0, 0, 1.0]])
        proj = project_world_to_image(K, R, t, (k1, k2), world)
        return (proj - image).ravel()

    p0 = np.array([*np.asarray(rvec0, float).reshape(3), fx0, 0.05, 0.0])
    sol = least_squares(res, p0, method="lm", max_nfev=400)
    r = res(sol.x).reshape(-1, 2)
    err = np.linalg.norm(r, axis=1)
    return float(np.median(err)), float(err.max()), float(sol.x[3]), \
        float(sol.x[4]), float(sol.x[5])


def main() -> None:
    track = json.load(open(BASE + "_camera_track.json"))
    anchors = json.load(open(BASE + "_anchors__manual.json"))
    cams = {f["frame"]: f for f in track["frames"]}

    for label, C in (
        ("bundle C (48.9,-31.4,15.1)", np.array([48.93, -31.40, 15.05])),
        ("anchor C (52.4,-36.4,16.2)", np.array([52.37, -36.43, 16.20])),
        ("manual C (51.2,-33.6,15.7)", np.array([51.19, -33.62, 15.7])),
    ):
        print(f"=== {label}")
        for a in anchors["anchors"]:
            f = a["frame"]
            clicks = a.get("landmarks", [])
            if len(clicks) < 5 or f not in cams:
                continue
            c = cams[f]
            rv, _ = cv2.Rodrigues(np.asarray(c["R"]))
            med, mx, fx, k1, k2 = fit_frame(
                clicks, C, rv.reshape(3), float(c["K"][0][0]))
            print(f"  f{f:>3}: best-fit med {med:6.1f}px max {mx:6.1f}px  "
                  f"fx={fx:6.0f} k1={k1:+.3f} k2={k2:+.3f} ({len(clicks)} clicks)")


if __name__ == "__main__":
    main()
