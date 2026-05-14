"""Iterative loop: detect → global static-C solve → re-detect with new
cameras → re-solve. The hypothesis is that per-frame detected lines
disagree on C because each detection was done with a per-frame
bootstrap camera that had its own bias; re-detecting under a coherent
static-camera model should produce internally consistent lines.

Each iteration overwrites ``output/camera/<shot>_detected_lines.json``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from src.schemas.anchor import AnchorSet, LineObservation
from src.utils.anchor_solver import (
    _is_rich,
    _line_residuals,
    _make_K,
    _point_residuals_distorted,
)
from src.utils.camera_projection import project_world_to_image
from src.utils.line_detector import (
    DetectorConfig,
    detect_painted_lines_in_frame,
)
from src.utils.pitch_lines_catalogue import LINE_CATALOGUE


logger = logging.getLogger(__name__)


def _is_pitch_line(segment) -> bool:
    a, b = segment
    return all(
        0.0 <= p[0] <= 105.0 and 0.0 <= p[1] <= 68.0 and p[2] == 0.0
        for p in (a, b)
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("clip", type=Path)
    parser.add_argument("bootstrap_camera", type=Path)
    parser.add_argument("anchors", type=Path)
    parser.add_argument("--strip-px", type=int, default=20)
    parser.add_argument("--max-motion-m", type=float, default=0.5)
    parser.add_argument("--point-hint-weight", type=float, default=0.1)
    parser.add_argument("--n-outer-iters", type=int, default=3)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    with open(args.bootstrap_camera) as f:
        track = json.load(f)
    distortion = tuple(track.get("distortion", [0.0, 0.0]))
    image_size = tuple(track["image_size"])
    W, H = image_size
    cameras_by_frame_init = {fr["frame"]: fr for fr in track["frames"]}

    aset = AnchorSet.load(args.anchors)
    anchor_landmarks = {
        a.frame: list(a.landmarks) for a in aset.anchors if a.landmarks
    }
    rich_frames = {a.frame for a in aset.anchors if _is_rich(a)}

    pitch_lines = {
        n: s for n, s in LINE_CATALOGUE.items() if _is_pitch_line(s)
    }

    cap = cv2.VideoCapture(str(args.clip))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"clip has {n_frames} frames")

    # Pre-extract frames (memory-hungry but avoids seeking each iter)
    print("loading frames into memory...")
    frames_bgr: dict[int, np.ndarray] = {}
    for fi in range(n_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        if ok:
            frames_bgr[fi] = frame
    cap.release()
    print(f"loaded {len(frames_bgr)} frames")

    # Initial per-frame cameras from existing pipeline
    cameras: dict[int, dict] = {}
    for fi in sorted(frames_bgr.keys()):
        if fi not in cameras_by_frame_init:
            continue
        cam = cameras_by_frame_init[fi]
        cameras[fi] = {
            "K": np.array(cam["K"]),
            "R": np.array(cam["R"]),
            "t": np.array(cam["t"]),
        }

    detector_cfg = DetectorConfig(search_strip_px=args.strip_px)

    def detect_all(cur_cameras, cur_distortion):
        per_frame: dict[int, list[LineObservation]] = {}
        for fi, frame in frames_bgr.items():
            if fi not in cur_cameras:
                continue
            K = cur_cameras[fi]["K"]; R = cur_cameras[fi]["R"]; t = cur_cameras[fi]["t"]
            all_dets = detect_painted_lines_in_frame(
                frame, K, R, t, cur_distortion, pitch_lines, detector_cfg,
            )
            dets = [d for d in all_dets if d.confidence >= 0.5 and d.n_samples >= 40]
            if len(dets) >= 2:
                per_frame[fi] = [
                    LineObservation(name=d.name, image_segment=d.image_segment,
                                    world_segment=d.world_segment)
                    for d in dets
                ]
        return per_frame

    def joint_solve(per_frame_lines, cur_cameras, lens_seed):
        cx_s, cy_s, k1_s, k2_s = lens_seed
        fids = sorted(per_frame_lines.keys())
        # Shared (cx, cy, k1, k2, Cx, Cy, Cz) + per-frame (rvec, dC, fx)
        rich_seed_Cs = [
            -cur_cameras[f]["R"].T @ cur_cameras[f]["t"]
            for f in fids if f in rich_frames
        ] or [
            -cur_cameras[f]["R"].T @ cur_cameras[f]["t"] for f in fids
        ]
        C_seed = np.median(np.stack(rich_seed_Cs), axis=0)

        SHARED = 7; PER = 7; n = len(fids)
        p0 = np.empty(SHARED + PER * n)
        p0[0] = cx_s; p0[1] = cy_s
        p0[2] = k1_s; p0[3] = k2_s
        p0[4:7] = C_seed
        lower = np.empty_like(p0); upper = np.empty_like(p0)
        lower[0] = W/2 - 100; upper[0] = W/2 + 100
        lower[1] = H/2 - 100; upper[1] = H/2 + 100
        lower[2:4] = -0.5; upper[2:4] = 0.5
        lower[4:7] = C_seed - 5; upper[4:7] = C_seed + 5
        c_b = args.max_motion_m
        for i, fi in enumerate(fids):
            cam = cur_cameras[fi]
            rv, _ = cv2.Rodrigues(cam["R"])
            b = SHARED + i * PER
            p0[b:b+3] = rv.reshape(3)
            p0[b+3:b+6] = 0.0
            p0[b+6] = float(cam["K"][0, 0])
            lower[b:b+3] = -np.pi; upper[b:b+3] = np.pi
            lower[b+3:b+6] = -c_b if c_b > 0 else -1e-6
            upper[b+3:b+6] = +c_b if c_b > 0 else +1e-6
            lower[b+6] = float(cam["K"][0, 0]) * 0.5
            upper[b+6] = float(cam["K"][0, 0]) * 2.0

        def res(p):
            cx, cy = float(p[0]), float(p[1])
            k1, k2 = float(p[2]), float(p[3])
            C = p[4:7]
            parts = []
            for i, fi in enumerate(fids):
                b = SHARED + i * PER
                rv = p[b:b+3]; dC = p[b+3:b+6]; fx = float(p[b+6])
                R_i, _ = cv2.Rodrigues(rv); K_i = _make_K(fx, cx, cy)
                t_i = -R_i @ (C + dC)
                if per_frame_lines.get(fi):
                    parts.append(_line_residuals(per_frame_lines[fi], K_i, R_i, t_i))
                if fi in anchor_landmarks:
                    parts.append(args.point_hint_weight * _point_residuals_distorted(
                        anchor_landmarks[fi], K_i, rv, t_i, (k1, k2),
                    ))
            return np.concatenate(parts)

        # Sparse Jacobian
        n_res_per = []
        for fi in fids:
            n_r = 2 * len(per_frame_lines.get(fi, []))
            if fi in anchor_landmarks:
                n_r += 2 * len(anchor_landmarks[fi])
            n_res_per.append(n_r)
        total = sum(n_res_per); n_p = SHARED + PER * n
        spar = lil_matrix((total, n_p), dtype=np.uint8)
        row = 0
        for i, n_r in enumerate(n_res_per):
            b = SHARED + i * PER
            for jr in range(n_r):
                spar[row + jr, 0:SHARED] = 1
                spar[row + jr, b:b+PER] = 1
            row += n_r

        result = least_squares(
            res, p0, bounds=(lower, upper),
            method="trf", loss="huber", f_scale=2.0,
            max_nfev=400, jac_sparsity=spar.tocsr(),
            xtol=1e-9, ftol=1e-9, gtol=1e-9,
        )
        cx, cy = float(result.x[0]), float(result.x[1])
        k1, k2 = float(result.x[2]), float(result.x[3])
        C = result.x[4:7]
        new_cameras = {}
        for i, fi in enumerate(fids):
            b = SHARED + i * PER
            rv = result.x[b:b+3]; dC = result.x[b+3:b+6]; fx = float(result.x[b+6])
            R_i, _ = cv2.Rodrigues(rv); K_i = _make_K(fx, cx, cy)
            t_i = -R_i @ (C + dC)
            new_cameras[fi] = {"K": K_i, "R": R_i, "t": t_i}
        return new_cameras, (cx, cy, k1, k2), C

    lens = (W/2, H/2, 0.0, 0.0)
    for outer in range(args.n_outer_iters):
        print(f"\n=== outer iter {outer + 1} ===")
        per_frame_lines = detect_all(cameras, (lens[2], lens[3]))
        print(f"  detected lines on {len(per_frame_lines)} frames")
        cameras, lens, C = joint_solve(per_frame_lines, cameras, lens)
        # Per-iteration metrics
        rms_vals = []
        for fi, lines in per_frame_lines.items():
            K = cameras[fi]["K"]; R = cameras[fi]["R"]; t = cameras[fi]["t"]
            rms_vals.append(float(np.sqrt((_line_residuals(lines, K, R, t)**2).mean())))
        rms = np.array(rms_vals)
        print(f"  lens: cx={lens[0]:.1f} cy={lens[1]:.1f} k1={lens[2]:+.4f} k2={lens[3]:+.4f}")
        print(f"  C   : ({C[0]:.3f}, {C[1]:.3f}, {C[2]:.3f})")
        print(f"  line RMS: mean={rms.mean():.3f}  median={np.median(rms):.3f}  max={rms.max():.3f}")
        print(f"  frac <1px={(rms < 1.0).mean():.3f}  <2px={(rms < 2.0).mean():.3f}  <3px={(rms < 3.0).mean():.3f}")
        # Point sanity
        for fid in sorted(anchor_landmarks):
            if fid not in cameras: continue
            cam = cameras[fid]; lm = anchor_landmarks[fid]
            pts = np.array([l.world_xyz for l in lm]); obs = np.array([l.image_xy for l in lm])
            proj = project_world_to_image(cam["K"], cam["R"], cam["t"], (lens[2], lens[3]), pts)
            devs = np.linalg.norm(proj - obs, axis=1)
            if fid in rich_frames:
                print(f'    f{fid:>4}: pt mean={devs.mean():5.2f} max={devs.max():5.2f}')

    return 0


if __name__ == "__main__":
    sys.exit(main())
