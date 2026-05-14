"""Iterative detect -> static-C solve -> re-detect loop.

Each outer iteration: detect painted lines under the current cameras,
solve one shared static camera centre, then re-detect under the
coherent cameras. Tests the hypothesis that per-frame detection bias
(from biased bootstrap cameras) closes once the cameras agree.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from src.schemas.anchor import AnchorSet
from src.utils.anchor_solver import _is_rich
from src.utils.camera_projection import project_world_to_image
from src.utils.line_camera_refine import detect_lines_for_frames
from src.utils.line_detector import DetectorConfig
from src.utils.static_line_solver import solve_static_camera_from_lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("clip", type=Path)
    parser.add_argument("bootstrap_camera", type=Path)
    parser.add_argument("anchors", type=Path)
    parser.add_argument("--strip-px", type=int, default=20)
    parser.add_argument("--point-hint-weight", type=float, default=0.05)
    parser.add_argument("--lens-model", default="pinhole_k1k2",
                        choices=["pinhole_k1k2", "brown_conrady"])
    parser.add_argument("--n-outer-iters", type=int, default=3)
    args = parser.parse_args()

    with open(args.bootstrap_camera) as f:
        track = json.load(f)
    distortion = tuple(track.get("distortion", [0.0, 0.0]))
    W, H = track["image_size"]
    cams_init = {fr["frame"]: fr for fr in track["frames"]}

    aset = AnchorSet.load(args.anchors)
    anchor_landmarks = {
        a.frame: list(a.landmarks) for a in aset.anchors if a.landmarks
    }
    rich_frames = {a.frame for a in aset.anchors if _is_rich(a)}

    cap = cv2.VideoCapture(str(args.clip))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_bgr: dict[int, np.ndarray] = {}
    for fi in range(n_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        if ok and fi in cams_init:
            frames_bgr[fi] = frame
    cap.release()
    print(f"loaded {len(frames_bgr)} frames")

    cameras = {
        fi: {
            "K": np.array(cams_init[fi]["K"]),
            "R": np.array(cams_init[fi]["R"]),
            "t": np.array(cams_init[fi]["t"]),
        }
        for fi in frames_bgr
    }
    det_cfg = DetectorConfig(search_strip_px=args.strip_px)
    dist2 = (float(distortion[0]), float(distortion[1]))

    for outer in range(args.n_outer_iters):
        print(f"\n=== outer iter {outer + 1} ===")
        per_frame_lines = detect_lines_for_frames(
            frames_bgr, cameras, dist2, det_cfg,
        )
        print(f"  detected lines on {len(per_frame_lines)} frames")
        if len(per_frame_lines) < 2:
            print("  too few frames with lines; stopping")
            break

        seeds = {}
        seed_cs = []
        for fid in per_frame_lines:
            rv, _ = cv2.Rodrigues(cameras[fid]["R"])
            seeds[fid] = (rv.reshape(3), float(cameras[fid]["K"][0, 0]))
            if fid in rich_frames:
                seed_cs.append(
                    -cameras[fid]["R"].T @ cameras[fid]["t"]
                )
        if not seed_cs:
            seed_cs = [
                -cameras[f]["R"].T @ cameras[f]["t"] for f in per_frame_lines
            ]
        c_seed = np.median(np.stack(seed_cs), axis=0)

        sol = solve_static_camera_from_lines(
            per_frame_lines, (int(W), int(H)),
            c_seed=c_seed, lens_seed=(W / 2.0, H / 2.0, dist2[0], dist2[1]),
            per_frame_seeds=seeds, point_hints=anchor_landmarks,
            lens_model=args.lens_model, point_hint_weight=args.point_hint_weight,
        )
        cameras = {
            fid: {"K": K, "R": R, "t": t}
            for fid, (K, R, t) in sol.per_frame_KRt.items()
        }
        dist2 = (sol.distortion[0], sol.distortion[1])

        rms = np.array(
            [v for v in sol.per_frame_line_rms.values() if np.isfinite(v)]
        )
        print(f"  C = {np.round(sol.camera_centre, 3)}")
        print(f"  line RMS: mean={rms.mean():.3f} median={np.median(rms):.3f} "
              f"max={rms.max():.3f}  frac<1px={(rms < 1.0).mean():.3f}")
        for fid in sorted(anchor_landmarks):
            if fid not in cameras or fid not in rich_frames:
                continue
            K, R, t = cameras[fid]["K"], cameras[fid]["R"], cameras[fid]["t"]
            pts = np.array([lm.world_xyz for lm in anchor_landmarks[fid]])
            obs = np.array([lm.image_xy for lm in anchor_landmarks[fid]])
            proj = project_world_to_image(K, R, t, dist2, pts)
            devs = np.linalg.norm(proj - obs, axis=1)
            print(f"    f{fid:>4}: pt mean={devs.mean():5.2f} "
                  f"max={devs.max():5.2f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
