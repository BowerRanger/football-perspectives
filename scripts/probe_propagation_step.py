"""Probe one backward-propagation step on a real clip: from the track's
first covered frame, attempt the exact _solve_at recipe on the frame before
it and report every intermediate (lines, circle, rms at each gate).

Usage: .venv/bin/python scripts/probe_propagation_step.py BASE [FRAME]
  BASE e.g. output-origi/camera/origi01 ; FRAME defaults to first-covered - 1
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.schemas.anchor import LandmarkObservation  # noqa: E402
from src.utils.circle_detector import detect_circle  # noqa: E402
from src.utils.line_camera_refine import detect_lines_for_frames  # noqa: E402
from src.utils.line_detector import DetectorConfig  # noqa: E402
from src.utils.static_c_profile import _solve_frame_at_fixed_c  # noqa: E402
from src.utils.static_line_solver import _dist5  # noqa: E402


def main() -> None:
    base = sys.argv[1]
    track = json.load(open(base + "_camera_track.json"))
    cams = {f["frame"]: f for f in track["frames"]}
    first = min(cams)
    target = int(sys.argv[2]) if len(sys.argv) > 2 else first - 1
    nb1, nb2 = first, first + 1
    print(f"track span {first}..{max(cams)}; probing f{target} "
          f"seeded from f{nb1} (velocity from f{nb2})")

    C = np.array(track["camera_centre"], dtype=float)
    cx, cy = track["principal_point"]
    dist = tuple(track.get("distortion", (0, 0))[:2])
    d5 = _dist5(dist)
    print(f"C={np.round(C, 2).tolist()} pp=({cx:.1f},{cy:.1f}) "
          f"dist={np.round(dist, 4).tolist()}")

    R1 = np.array(cams[nb1]["R"]); fx1 = float(cams[nb1]["K"][0][0])
    from scipy.spatial.transform import Rotation
    steps = nb1 - target
    if nb2 in cams:
        R2 = np.array(cams[nb2]["R"])
        D = Rotation.from_matrix(R1 @ R2.T)
        Dk = Rotation.from_rotvec(D.as_rotvec() * steps)
        seed_R = (Dk * Rotation.from_matrix(R1)).as_matrix()
        dfx = fx1 - float(cams[nb2]["K"][0][0])
        seed_fx = float(np.clip(fx1 + dfx * steps, 0.7 * fx1, 1.3 * fx1))
    else:
        seed_R, seed_fx = R1, fx1
    seed_t = -seed_R @ C
    seed_K = np.array([[seed_fx, 0, cx], [0, seed_fx, cy], [0, 0, 1.0]])

    vid = base.replace("/camera/", "/shots/") + ".mp4"
    cap = cv2.VideoCapture(vid)
    cap.set(cv2.CAP_PROP_POS_FRAMES, target)
    ok, img = cap.read()
    cap.release()
    if not ok:
        print("no frame")
        return

    det = detect_lines_for_frames(
        {target: img}, {target: {"K": seed_K, "R": seed_R, "t": seed_t}},
        dist, DetectorConfig(), min_confidence=0.5, min_n_samples=40,
        min_lines=1).get(target, [])
    print(f"straight lines: {len(det)} -> {[d.name for d in det]}")

    cd = detect_circle(img, seed_K, seed_R, seed_t, (0.0, 0.0),
                       DetectorConfig())
    print(f"circle: {cd and f'{len(cd.image_points)} pts conf {cd.confidence:.2f}'}")

    circ_obs = None
    if cd is not None:
        k = min(20, len(cd.image_points))
        idx = np.linspace(0, len(cd.image_points) - 1, k).astype(int)
        circ_obs = [LandmarkObservation(name=cd.name,
                                        image_xy=cd.image_points[j],
                                        world_xyz=cd.world_points[j])
                    for j in idx]

    rv, _ = cv2.Rodrigues(seed_R)
    if det or circ_obs:
        rvec, fx, rms = _solve_frame_at_fixed_c(
            det, cx, cy, d5, C, rv.reshape(3), seed_fx,
            fx_rel=0.05 if len(det) < 4 else None, circle_obs=circ_obs)
        R_solved, _ = cv2.Rodrigues(rvec)
        from src.stages.camera import _angle_between
        dev = _angle_between(np.asarray(seed_R), R_solved)
        print(f"lines+circle solve: rms={rms:.2f} fx={fx:.0f} "
              f"dev-from-seed={dev:.2f}deg "
              f"(gates: rms 12.0 with circle / 4.0 without; dev 3.0)")
        if det:
            rvec2, fx2, rms2 = _solve_frame_at_fixed_c(
                det, cx, cy, d5, C, rv.reshape(3), seed_fx,
                fx_rel=0.05 if len(det) < 4 else None)
            print(f"lines-only solve  : rms={rms2:.2f} fx={fx2:.0f}")


if __name__ == "__main__":
    main()
