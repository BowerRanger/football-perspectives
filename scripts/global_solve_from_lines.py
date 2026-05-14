"""Global static-camera solve from per-frame detected lines.

Reads ``output/camera/<shot>_detected_lines.json`` (per-frame painted-
line observations with per-frame bootstrap cameras) and fits one camera
body across every frame via :func:`solve_static_camera_from_lines`:

  Shared:     cx, cy, distortion, Cx, Cy, Cz   (one fixed camera centre)
  Per-frame:  rvec(3), fx

Reports per-frame line RMS and per-anchor-frame point-landmark deviance.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from src.schemas.anchor import AnchorSet, LineObservation
from src.schemas.camera_track import CameraFrame, CameraTrack
from src.utils.anchor_solver import _is_rich
from src.utils.camera_projection import project_world_to_image
from src.utils.static_line_solver import solve_static_camera_from_lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("detected_lines", type=Path)
    parser.add_argument("anchors", type=Path)
    parser.add_argument("--lens-model", default="pinhole_k1k2",
                        choices=["pinhole_k1k2", "brown_conrady"])
    parser.add_argument("--point-hint-weight", type=float, default=0.05)
    parser.add_argument("--output-camera-track", type=Path, default=None)
    args = parser.parse_args()

    with open(args.detected_lines) as f:
        data = json.load(f)
    W, H = data["image_size"]
    fps = data.get("fps", 30.0)
    shot_id = data["shot_id"]
    frames = data["frames"]

    aset = AnchorSet.load(args.anchors)
    anchor_landmarks = {
        a.frame: list(a.landmarks) for a in aset.anchors if a.landmarks
    }
    rich_frames = {a.frame for a in aset.anchors if _is_rich(a)}

    per_frame_lines: dict[int, list[LineObservation]] = {}
    per_frame_seeds: dict[int, tuple[np.ndarray, float]] = {}
    seed_cs: list[np.ndarray] = []
    for fid_str, body in frames.items():
        fid = int(fid_str)
        lines = [
            LineObservation(
                name=ln["name"],
                image_segment=tuple(map(tuple, ln["image_segment"])),
                world_segment=tuple(map(tuple, ln["world_segment"])),
            )
            for ln in body["lines"]
        ]
        if len(lines) < 2 or "K" not in body:
            continue
        K = np.array(body["K"])
        R = np.array(body["R"])
        t = np.array(body["t"])
        per_frame_lines[fid] = lines
        rv, _ = cv2.Rodrigues(R)
        per_frame_seeds[fid] = (rv.reshape(3), float(K[0, 0]))
        if fid in rich_frames:
            seed_cs.append(-R.T @ t)
    if not seed_cs:
        seed_cs = [
            -np.array(frames[str(f)]["R"]).T @ np.array(frames[str(f)]["t"])
            for f in per_frame_lines
        ]
    c_seed = np.median(np.stack(seed_cs), axis=0)
    print(f"loaded {len(per_frame_lines)} frames; C seed = {np.round(c_seed, 3)}")

    sol = solve_static_camera_from_lines(
        per_frame_lines, (int(W), int(H)),
        c_seed=c_seed, lens_seed=(W / 2.0, H / 2.0, 0.0, 0.0),
        per_frame_seeds=per_frame_seeds, point_hints=anchor_landmarks,
        lens_model=args.lens_model, point_hint_weight=args.point_hint_weight,
    )

    rms = np.array([v for v in sol.per_frame_line_rms.values() if np.isfinite(v)])
    print(f"\nrecovered C = {np.round(sol.camera_centre, 3)}")
    print(f"recovered lens: pp={np.round(sol.principal_point, 1)} "
          f"distortion={np.round(sol.distortion, 4)}")
    print(f"line RMS: mean={rms.mean():.3f} median={np.median(rms):.3f} "
          f"max={rms.max():.3f}")
    print(f"  frac <1px={(rms < 1.0).mean():.3f}  <2px={(rms < 2.0).mean():.3f}")

    print("\nPoint-landmark devs on rich anchor frames:")
    for fid in sorted(anchor_landmarks):
        if fid not in sol.per_frame_KRt or fid not in rich_frames:
            continue
        K, R, t = sol.per_frame_KRt[fid]
        pts = np.array([lm.world_xyz for lm in anchor_landmarks[fid]])
        obs = np.array([lm.image_xy for lm in anchor_landmarks[fid]])
        proj = project_world_to_image(
            K, R, t, (sol.distortion[0], sol.distortion[1]), pts
        )
        devs = np.linalg.norm(proj - obs, axis=1)
        line_rms = sol.per_frame_line_rms.get(fid, float("nan"))
        print(f"  f{fid:>4}: line RMS={line_rms:5.3f}  "
              f"pt mean={devs.mean():5.2f}  pt max={devs.max():5.2f}")

    if args.output_camera_track:
        frames_out = [
            CameraFrame(
                frame=fid, K=K.tolist(), R=R.tolist(),
                confidence=1.0, is_anchor=fid in rich_frames, t=list(t),
            )
            for fid, (K, R, t) in sorted(sol.per_frame_KRt.items())
        ]
        first = frames_out[0]
        CameraTrack(
            clip_id=shot_id, fps=fps, image_size=(int(W), int(H)),
            t_world=list(first.t), frames=tuple(frames_out),
            principal_point=sol.principal_point,
            camera_centre=tuple(float(x) for x in sol.camera_centre),
            distortion=(sol.distortion[0], sol.distortion[1]),
        ).save(args.output_camera_track)
        print(f"\nwrote camera track to {args.output_camera_track}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
