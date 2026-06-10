"""Replicate the cold-start orientation sweep for chosen start frames and
report what each candidate finds (lines, circle, rms) — why does the sweep
yield no seeds on origi01's start?

Usage: .venv/bin/python scripts/probe_cold_start_sweep.py BASE REF_FRAME f1,f2,...
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.schemas.anchor import LandmarkObservation  # noqa: E402
from src.stages.camera import _circle_in_view_fraction  # noqa: E402
from src.utils.circle_detector import detect_circle  # noqa: E402
from src.utils.line_camera_refine import detect_lines_for_frames  # noqa: E402
from src.utils.line_detector import DetectorConfig  # noqa: E402
from src.utils.static_c_profile import _solve_frame_at_fixed_c  # noqa: E402
from src.utils.static_line_solver import _dist5  # noqa: E402


def main() -> None:
    base, ref_s, frames_s = sys.argv[1], sys.argv[2], sys.argv[3]
    track = json.load(open(base + "_camera_track.json"))
    cams = {f["frame"]: f for f in track["frames"]}
    ref = int(ref_s)
    C = np.array(track["camera_centre"], dtype=float)
    cx, cy = track["principal_point"]
    dist = tuple(track.get("distortion", (0, 0))[:2])
    d5 = _dist5(dist)
    size = tuple(track.get("image_size", (1920, 1080)))

    R_ref = np.array(cams[ref]["R"])
    fx_ref = float(cams[ref]["K"][0][0])
    print(f"ref f{ref}: fx={fx_ref:.0f} pp=({cx:.1f},{cy:.1f}) dist={np.round(dist,3).tolist()}")

    covered = sorted(cams)
    pairs = [(a, b) for a, b in zip(covered, covered[1:]) if b - a == 1]
    axes = []
    for a, b in pairs:
        v = Rotation.from_matrix(
            np.array(cams[b]["R"]) @ np.array(cams[a]["R"]).T).as_rotvec()
        if np.linalg.norm(v) > 1e-4:
            axes.append(v / np.linalg.norm(v))
    pan_ax = np.mean(axes, axis=0)
    pan_ax /= np.linalg.norm(pan_ax)
    tilt_ax = np.cross(pan_ax, R_ref[2])
    tilt_ax /= np.linalg.norm(tilt_ax)

    vid = base.replace("/camera/", "/shots/") + ".mp4"
    cap = cv2.VideoCapture(vid)

    for fid in [int(x) for x in frames_s.split(",")]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
        ok, img = cap.read()
        if not ok:
            print(f"f{fid}: no frame")
            continue
        best = None
        n_cand_lines = 0
        n_cand_circ = 0
        for dp in np.arange(-24, 25, 4):
            for dt in np.arange(-10, 11, 5):
                Rc = (Rotation.from_rotvec(pan_ax * np.radians(dp))
                      * Rotation.from_rotvec(tilt_ax * np.radians(dt))
                      ).as_matrix() @ R_ref
                sK = np.array([[fx_ref, 0, cx], [0, fx_ref, cy], [0, 0, 1.0]])
                det = detect_lines_for_frames(
                    {fid: img}, {fid: {"K": sK, "R": Rc, "t": -Rc @ C}},
                    dist, DetectorConfig(search_strip_px=40, min_gradient=10.0),
                    min_confidence=0.5, min_n_samples=40,
                    min_lines=1).get(fid, [])
                if det:
                    n_cand_lines += 1
                circ_obs = None
                if len(det) < 3:
                    if _circle_in_view_fraction(sK, Rc, -Rc @ C, size) >= 0.3:
                        cd = detect_circle(
                            img, sK, Rc, -Rc @ C, (0.0, 0.0),
                            DetectorConfig(search_strip_px=100,
                                           min_gradient=10.0))
                        if cd is not None:
                            n_cand_circ += 1
                            k = min(20, len(cd.image_points))
                            idx = np.linspace(0, len(cd.image_points) - 1,
                                              k).astype(int)
                            circ_obs = [
                                LandmarkObservation(
                                    name=cd.name,
                                    image_xy=cd.image_points[j],
                                    world_xyz=cd.world_points[j])
                                for j in idx]
                if len(det) < 3 and not circ_obs:
                    continue
                rv, _ = cv2.Rodrigues(Rc)
                rvec, fx, rms = _solve_frame_at_fixed_c(
                    det, cx, cy, d5, C, rv.reshape(3), fx_ref,
                    fx_rel=0.05 if len(det) < 4 else None,
                    circle_obs=circ_obs)
                if not np.isfinite(rms):
                    continue
                R_solved, _ = cv2.Rodrigues(rvec)
                from src.stages.camera import _angle_between
                pull = _angle_between(Rc, R_solved)
                if circ_obs is not None and pull > 3.0:
                    continue  # the in-stage dev-guard
                cand = (len(det), -rms)
                if best is None or cand > best[0]:
                    best = (cand, R_solved, fx, det, circ_obs is not None,
                            pull)
        if best is None:
            print(f"f{fid}: lines={n_cand_lines} circle={n_cand_circ} "
                  f"NO candidate survived dev-guard")
            continue
        (cand, Re, fx, det, had_circ, pull) = best
        # replicate the refine loop: re-detect under the solve, strip 30
        rms = -cand[1]
        trace = [f"sweep rms={rms:.1f} pull={pull:.1f} circ={had_circ}"]
        for it in range(4):
            sK = np.array([[fx, 0, cx], [0, fx, cy], [0, 0, 1.0]])
            det = detect_lines_for_frames(
                {fid: img}, {fid: {"K": sK, "R": Re, "t": -Re @ C}},
                dist, DetectorConfig(search_strip_px=30, min_gradient=10.0),
                min_confidence=0.5, min_n_samples=40, min_lines=1
            ).get(fid, [])
            circ_obs = None
            if len(det) < 3:
                if _circle_in_view_fraction(sK, Re, -Re @ C, size) >= 0.3:
                    cd = detect_circle(
                        img, sK, Re, -Re @ C, (0.0, 0.0),
                        DetectorConfig(search_strip_px=100,
                                       min_gradient=10.0))
                    if cd is not None:
                        k = min(20, len(cd.image_points))
                        idx = np.linspace(0, len(cd.image_points) - 1,
                                          k).astype(int)
                        circ_obs = [
                            LandmarkObservation(
                                name=cd.name, image_xy=cd.image_points[j],
                                world_xyz=cd.world_points[j])
                            for j in idx]
                if circ_obs is None:
                    trace.append(f"it{it}: no circle ({len(det)} lines) STOP")
                    break
            rv, _ = cv2.Rodrigues(Re)
            rvec, fx2, rms2 = _solve_frame_at_fixed_c(
                det, cx, cy, d5, C, rv.reshape(3), fx,
                fx_rel=0.05 if len(det) < 4 else None, circle_obs=circ_obs)
            R_new, _ = cv2.Rodrigues(rvec)
            pull2 = _angle_between(Re, R_new)
            if circ_obs is not None and pull2 > 3.0:
                trace.append(f"it{it}: pull {pull2:.1f} > 3 STOP")
                break
            Re, fx, rms = R_new, fx2, rms2
            trace.append(f"it{it}: rms={rms:.1f} fx={fx:.0f}")
        verdict = "SEED" if rms <= 12.0 else "REJECT(rms)"
        print(f"f{fid}: {' | '.join(trace)} -> {verdict}")
    cap.release()


if __name__ == "__main__":
    main()
