"""Global joint solve from per-frame detected lines.

Reads ``output/camera/<shot>_detected_lines.json`` (already-detected per-
frame painted-line observations) and re-fits one camera body across
every frame:

  Shared params (7):  cx, cy, k1, k2, Cx, Cy, Cz
  Per-frame params (4 × n_frames):  rvec(3), fx
  Each frame's t reconstructed as t = -R @ C.

This enforces the static-camera contract (body fixed) across the whole
shot while letting pan/tilt/zoom vary per frame. The sub-pixel-accurate
line observations from line_detector.py provide the constraints; click-
noisy point landmarks (from the anchor JSON) appear as a small-weight
sanity check inside the cost.

Optionally allows a small bounded per-frame dC (≤ max_motion_m metres)
for clips where the camera body genuinely shifts. Set to 0 for pure
static; the default 0.5 m absorbs gantry vibration without violating
the user's 2 m body-motion budget.

Reports per-frame line RMS + per-anchor-frame point-landmark deviance.
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


logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("detected_lines", type=Path)
    parser.add_argument("anchors", type=Path)
    parser.add_argument("--max-motion-m", type=float, default=0.5,
                        help="L_inf per-component motion budget per frame "
                             "(L2 worst-case = budget * sqrt(3) m). Set 0 for "
                             "pure static.")
    parser.add_argument("--point-hint-weight", type=float, default=0.05)
    parser.add_argument("--output-camera-track", type=Path, default=None,
                        help="If set, write a CameraTrack JSON here with the "
                             "global-solve cameras.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(message)s",
    )

    with open(args.detected_lines) as f:
        data = json.load(f)
    W, H = data["image_size"]
    fps = data.get("fps", 30.0)
    shot_id = data["shot_id"]
    frames = data["frames"]
    frame_ids = sorted(int(k) for k in frames.keys())
    print(f"loaded {len(frame_ids)} frames with detected lines from {args.detected_lines}")

    aset = AnchorSet.load(args.anchors)
    anchor_landmarks: dict[int, list] = {
        a.frame: list(a.landmarks) for a in aset.anchors if a.landmarks
    }
    rich_frames = {a.frame for a in aset.anchors if _is_rich(a)}

    # Per-frame data:
    per_frame_lines: dict[int, list[LineObservation]] = {}
    per_frame_seed_rvec: dict[int, np.ndarray] = {}
    per_frame_seed_fx: dict[int, float] = {}
    for fid in frame_ids:
        body = frames[str(fid)]
        lines = [
            LineObservation(
                name=ln["name"],
                image_segment=tuple(map(tuple, ln["image_segment"])),
                world_segment=tuple(map(tuple, ln["world_segment"])),
            )
            for ln in body["lines"]
        ]
        per_frame_lines[fid] = lines
        K = np.array(body["K"]); R = np.array(body["R"])
        rv, _ = cv2.Rodrigues(R.astype(np.float64))
        per_frame_seed_rvec[fid] = rv.reshape(3).copy()
        per_frame_seed_fx[fid] = float(K[0, 0])

    # Seed shared C from the seed-camera Cs of rich anchor frames
    rich_seed_Cs = []
    for fid in frame_ids:
        if fid not in rich_frames:
            continue
        body = frames[str(fid)]
        R = np.array(body["R"]); t = np.array(body["t"])
        rich_seed_Cs.append(-R.T @ t)
    if not rich_seed_Cs:
        # Fall back: use all frames' seed Cs
        for fid in frame_ids:
            body = frames[str(fid)]
            R = np.array(body["R"]); t = np.array(body["t"])
            rich_seed_Cs.append(-R.T @ t)
    C_seed = np.median(np.stack(rich_seed_Cs), axis=0)
    print(f"seed C = {C_seed}  (median across {len(rich_seed_Cs)} seed frames)")

    SHARED = 7  # cx, cy, k1, k2, Cx, Cy, Cz
    PER = 7     # rvec(3), dC(3), fx
    n = len(frame_ids)
    p0 = np.empty(SHARED + PER * n)
    p0[0] = W / 2.0; p0[1] = H / 2.0
    p0[2] = 0.0; p0[3] = 0.0
    p0[4:7] = C_seed
    lower = np.empty_like(p0); upper = np.empty_like(p0)
    lower[0] = W/2 - 100; upper[0] = W/2 + 100
    lower[1] = H/2 - 100; upper[1] = H/2 + 100
    lower[2:4] = -0.5; upper[2:4] = 0.5
    lower[4:7] = C_seed - 5.0; upper[4:7] = C_seed + 5.0

    # Per-frame fx bounds: start from seed, ±2x
    for i, fid in enumerate(frame_ids):
        base = SHARED + i * PER
        p0[base : base + 3] = per_frame_seed_rvec[fid]
        # dC starts at 0 (assume static-ish camera)
        p0[base + 3 : base + 6] = 0.0
        p0[base + 6] = per_frame_seed_fx[fid]
        fx_a = per_frame_seed_fx[fid]
        lower[base : base + 3] = -np.pi
        upper[base : base + 3] = np.pi
        if args.max_motion_m > 0:
            lower[base + 3 : base + 6] = -args.max_motion_m
            upper[base + 3 : base + 6] = +args.max_motion_m
        else:
            # static: pin dC to 0 by giving 0-width bounds (with epsilon)
            lower[base + 3 : base + 6] = -1e-6
            upper[base + 3 : base + 6] = +1e-6
        lower[base + 6] = fx_a * 0.5
        upper[base + 6] = fx_a * 2.0

    # Pre-extract residual templates
    point_hint_w = float(args.point_hint_weight)

    def residuals(p: np.ndarray) -> np.ndarray:
        cx, cy = float(p[0]), float(p[1])
        k1, k2 = float(p[2]), float(p[3])
        C = p[4:7]
        parts: list[np.ndarray] = []
        for i, fid in enumerate(frame_ids):
            base = SHARED + i * PER
            rv = p[base : base + 3]
            dC = p[base + 3 : base + 6]
            fx = float(np.clip(p[base + 6], 50.0, 1e5))
            R_i, _ = cv2.Rodrigues(rv)
            t_i = -R_i @ (C + dC)
            K_i = _make_K(fx, cx, cy)
            lines = per_frame_lines[fid]
            if lines:
                parts.append(_line_residuals(lines, K_i, R_i, t_i))
            hint = anchor_landmarks.get(fid)
            if hint:
                parts.append(point_hint_w * _point_residuals_distorted(
                    hint, K_i, rv, t_i, (k1, k2),
                ))
        return np.concatenate(parts) if parts else np.empty(0)

    # Sparse Jacobian sparsity pattern.
    # Each frame's lines residuals touch: shared (7) + this frame's PER (7).
    # Same for point hints.
    n_res_per_frame = []
    for fid in frame_ids:
        n_res = 2 * len(per_frame_lines[fid])
        if fid in anchor_landmarks:
            n_res += 2 * len(anchor_landmarks[fid])
        n_res_per_frame.append(n_res)
    total_res = sum(n_res_per_frame)
    total_par = SHARED + PER * n
    spar = lil_matrix((total_res, total_par), dtype=np.uint8)
    row = 0
    for i, n_res in enumerate(n_res_per_frame):
        base = SHARED + i * PER
        for jr in range(n_res):
            spar[row + jr, 0:SHARED] = 1
            spar[row + jr, base : base + PER] = 1
        row += n_res

    print(f"Joint solve: {total_par} params, {total_res} residuals")
    print(f"Running scipy.least_squares (this may take a few minutes)...")
    result = least_squares(
        residuals, p0, bounds=(lower, upper),
        method="trf", loss="huber", f_scale=2.0,
        max_nfev=800, jac_sparsity=spar.tocsr(),
        xtol=1e-10, ftol=1e-10, gtol=1e-10,
        verbose=1,
    )
    print(f"final cost: {result.cost:.3f}")

    cx, cy = float(result.x[0]), float(result.x[1])
    k1, k2 = float(result.x[2]), float(result.x[3])
    C_locked = result.x[4:7]
    print(f"\nrecovered lens: cx={cx:.1f} cy={cy:.1f} k1={k1:+.4f} k2={k2:+.4f}")
    print(f"recovered C:    ({C_locked[0]:.3f}, {C_locked[1]:.3f}, {C_locked[2]:.3f})")

    # Aggregate per-frame metrics
    line_rms_all = []
    motion_all = []
    point_rms_anchor = {}
    refined_cameras: dict[int, dict] = {}
    for i, fid in enumerate(frame_ids):
        base = SHARED + i * PER
        rv = result.x[base : base + 3]
        dC = result.x[base + 3 : base + 6]
        fx = float(result.x[base + 6])
        R_i, _ = cv2.Rodrigues(rv)
        t_i = -R_i @ (C_locked + dC)
        K_i = _make_K(fx, cx, cy)
        motion_all.append(float(np.linalg.norm(dC)))
        refined_cameras[fid] = {
            "K": K_i.tolist(),
            "R": R_i.tolist(),
            "t": list(t_i),
        }
        lines = per_frame_lines[fid]
        if lines:
            line_rms_all.append(float(np.sqrt((_line_residuals(lines, K_i, R_i, t_i) ** 2).mean())))
        if fid in anchor_landmarks:
            pts = np.array([lm.world_xyz for lm in anchor_landmarks[fid]])
            obs = np.array([lm.image_xy for lm in anchor_landmarks[fid]])
            proj = project_world_to_image(K_i, R_i, t_i, (k1, k2), pts)
            devs = np.linalg.norm(proj - obs, axis=1)
            point_rms_anchor[fid] = (float(devs.mean()), float(devs.max()))

    line_rms_all = np.array(line_rms_all)
    motion_all = np.array(motion_all)
    print(f"\nLine RMS distribution across {len(line_rms_all)} frames:")
    print(f"  mean={line_rms_all.mean():.3f}  median={np.median(line_rms_all):.3f}  max={line_rms_all.max():.3f}")
    print(f"  P95={np.percentile(line_rms_all, 95):.3f}  P99={np.percentile(line_rms_all, 99):.3f}")
    print(f"  frac <1px={(line_rms_all < 1.0).mean():.3f}  <2px={(line_rms_all < 2.0).mean():.3f}  <3px={(line_rms_all < 3.0).mean():.3f}")

    print(f"\nPer-frame |dC| (motion from shared C):")
    print(f"  mean={motion_all.mean():.3f}  max={motion_all.max():.3f}  budget={args.max_motion_m:.2f} m")

    print(f"\nSanity check — point-landmark devs on rich anchor frames:")
    for fid in sorted(point_rms_anchor):
        if fid not in rich_frames: continue
        m, mx = point_rms_anchor[fid]
        line_idx = frame_ids.index(fid)
        line_r = line_rms_all[line_idx] if line_idx < len(line_rms_all) else 0
        print(f"  f{fid:>4}: line RMS={line_r:5.3f}  pt mean={m:5.2f}  pt max={mx:5.2f}")

    if args.output_camera_track:
        from src.schemas.camera_track import CameraFrame, CameraTrack
        frames_out = []
        for fid in frame_ids:
            cam = refined_cameras[fid]
            frames_out.append(CameraFrame(
                frame=fid,
                K=cam["K"],
                R=cam["R"],
                confidence=1.0,
                is_anchor=fid in rich_frames,
                t=cam["t"],
            ))
        track = CameraTrack(
            clip_id=shot_id,
            fps=fps,
            image_size=(int(W), int(H)),
            t_world=list(refined_cameras[frame_ids[0]]["t"]),
            frames=tuple(frames_out),
            principal_point=(cx, cy),
            camera_centre=(float(C_locked[0]), float(C_locked[1]), float(C_locked[2])),
            distortion=(float(k1), float(k2)),
        )
        track.save(args.output_camera_track)
        print(f"\nWrote camera track to {args.output_camera_track}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
