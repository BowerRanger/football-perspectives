"""Isolate the camera-centre hypothesis on origi01: re-solve every covered
frame's (rvec, fx) at a CANDIDATE C (default: the manual track's centre)
against the track's stored constraints (straight lines + circle points),
then reproject the manual anchor clicks. If clicks improve wholesale, C is
the binding error.

Usage: .venv/bin/python scripts/probe_c_hypothesis.py [Cx Cy Cz]
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.schemas.anchor import LandmarkObservation, LineObservation  # noqa: E402
from src.utils.camera_projection import project_world_to_image  # noqa: E402
from src.utils.static_c_profile import _solve_frame_at_fixed_c  # noqa: E402
from src.utils.static_line_solver import _dist5  # noqa: E402

BASE = "output-origi/camera/origi01"


def main() -> None:
    track = json.load(open(BASE + "_camera_track.json"))
    dl = json.load(open(BASE + "_detected_lines.json"))["frames"]
    anchors = json.load(open(BASE + "_anchors__manual.json"))
    cams = {f["frame"]: f for f in track["frames"]}
    dist = tuple(track.get("distortion", (0, 0))[:2])
    d5 = _dist5(dist)
    cx, cy = track["principal_point"]

    if len(sys.argv) >= 4:
        C = np.array([float(x) for x in sys.argv[1:4]])
    else:
        C = np.array([51.19, -33.62, 15.7])  # manual track's centre
    C0 = np.array(track["camera_centre"], dtype=float)
    print(f"current C={np.round(C0, 2).tolist()}  candidate C="
          f"{np.round(C, 2).tolist()}  |dC|={np.linalg.norm(C - C0):.2f} m")

    # Re-solve every frame that has stored constraints, at BOTH centres, and
    # compare the constraint misfit — if the stored evidence alone prefers
    # the candidate C, an automatic grid re-profile can find it.
    new_cams: dict[int, dict] = {}
    n_solved = 0
    rms_cand: list[float] = []
    rms_cur: list[float] = []
    for fk, fv in dl.items():
        f = int(fk)
        if f not in cams:
            continue
        lns = [
            LineObservation(
                name=ln["name"],
                image_segment=(tuple(ln["image_segment"][0]),
                               tuple(ln["image_segment"][1])),
                world_segment=(tuple(ln["world_segment"][0]),
                               tuple(ln["world_segment"][1])))
            for ln in fv["lines"] if "circle" not in ln["name"]
        ]
        pts = [
            LandmarkObservation(
                name="centre_circle",
                image_xy=tuple(ln["image_segment"][0]),
                world_xyz=tuple(ln["world_segment"][0]))
            for ln in fv["lines"] if "circle" in ln["name"]
        ] or None
        if not lns and not pts:
            continue
        c = cams[f]
        rv, _ = cv2.Rodrigues(np.asarray(c["R"]))
        rvec, fx, rms = _solve_frame_at_fixed_c(
            lns, cx, cy, d5, C, rv.reshape(3), float(c["K"][0][0]),
            circle_obs=pts)
        if not np.isfinite(rms):
            continue
        rms_cand.append(rms)
        _, _, rms0 = _solve_frame_at_fixed_c(
            lns, cx, cy, d5, C0, rv.reshape(3), float(c["K"][0][0]),
            circle_obs=pts)
        if np.isfinite(rms0):
            rms_cur.append(rms0)
        R, _ = cv2.Rodrigues(rvec)
        K = np.array([[fx, 0, cx], [0, fx, cy], [0, 0, 1.0]])
        new_cams[f] = {"K": K, "R": R, "t": -R @ C}
        n_solved += 1
    print(f"re-solved {n_solved} constrained frame(s) at candidate C")
    print(f"constraint misfit: candidate C med {np.median(rms_cand):.2f} "
          f"p90 {np.percentile(rms_cand, 90):.2f} | current C med "
          f"{np.median(rms_cur):.2f} p90 {np.percentile(rms_cur, 90):.2f}")

    # Clicks under (a) the live track, (b) the candidate-C re-solves
    for label, getcam in (
        ("live", lambda f: (np.asarray(cams[f]["K"]), np.asarray(cams[f]["R"]),
                            np.asarray(cams[f]["t"])) if f in cams else None),
        ("candidate-C", lambda f: (new_cams[f]["K"], new_cams[f]["R"],
                                   new_cams[f]["t"]) if f in new_cams else None),
    ):
        print(f"--- {label}")
        allr = []
        for a in anchors["anchors"]:
            got = getcam(a["frame"])
            if got is None:
                continue
            K, R, t = got
            res = []
            for ob in a.get("landmarks", []):
                p = project_world_to_image(
                    K, R, t, dist, np.array([ob["world_xyz"]], float))[0]
                res.append(float(np.linalg.norm(p - np.array(ob["image_xy"]))))
            if res:
                allr += res
                print(f"  f{a['frame']:>3}: med {np.median(res):7.1f}px "
                      f"max {max(res):7.1f}px  ({len(res)})")
        if allr:
            print(f"  ALL: med {np.median(allr):.1f}px p90 "
                  f"{np.percentile(allr, 90):.1f}px")


if __name__ == "__main__":
    main()
