"""Camera-centre profile diagnostic CLI.

Reads detected painted lines + a bootstrap camera track, sweeps a 3-D
grid of candidate static camera centres, and prints the line-fitting RMS
as a function of C. The decisive readout is the mean RMS at the argmin
centre: sub-pixel means a single static camera CAN fit the detected
lines under the current lens model; ~4 px means it cannot.

Usage:
  .venv/bin/python3 scripts/profile_static_c.py \\
      output/camera/<shot>_detected_lines.json \\
      output/camera/<shot>_camera_track.json \\
      [output/camera/<shot>_anchors.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

# Allow `python scripts/profile_static_c.py ...` from anywhere — the
# editable install exposes `football_perspectives`, not `src`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.schemas.anchor import AnchorSet, LineObservation
from src.schemas.camera_track import CameraTrack
from src.utils.anchor_solver import _is_rich
from src.utils.static_c_profile import make_c_grid, profile_camera_centre
from src.utils.static_line_solver import solve_static_camera_from_lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("detected_lines", type=Path)
    parser.add_argument("bootstrap_camera", type=Path,
                        help="CameraTrack JSON providing per-frame (rvec, fx) seeds.")
    parser.add_argument("anchors", type=Path, nargs="?", default=None)
    parser.add_argument("--coarse-extent-m", type=float, default=7.5)
    parser.add_argument("--coarse-steps", type=int, default=7)
    parser.add_argument("--fine-extent-m", type=float, default=2.0)
    parser.add_argument("--fine-steps", type=int, default=5)
    args = parser.parse_args()

    with open(args.detected_lines) as f:
        data = json.load(f)
    frames = data["frames"]
    image_size = tuple(data["image_size"])

    per_frame_lines: dict[int, list[LineObservation]] = {}
    for fid_str, body in frames.items():
        lines = [
            LineObservation(
                name=ln["name"],
                image_segment=tuple(map(tuple, ln["image_segment"])),
                world_segment=tuple(map(tuple, ln["world_segment"])),
            )
            for ln in body["lines"]
        ]
        if len(lines) >= 2:
            per_frame_lines[int(fid_str)] = lines

    track = CameraTrack.load(args.bootstrap_camera)
    boot: dict[int, tuple[np.ndarray, float]] = {}
    seed_cs: list[np.ndarray] = []
    rich: set[int] = set()
    anchor_landmarks: dict[int, list] = {}
    if args.anchors is not None:
        aset = AnchorSet.load(args.anchors)
        rich = {a.frame for a in aset.anchors if _is_rich(a)}
        anchor_landmarks = {
            a.frame: list(a.landmarks) for a in aset.anchors if a.landmarks
        }
    for cf in track.frames:
        if cf.frame not in per_frame_lines or cf.t is None:
            continue
        R = np.array(cf.R)
        t = np.array(cf.t)
        rv, _ = cv2.Rodrigues(R)
        boot[cf.frame] = (rv.reshape(3), float(np.array(cf.K)[0, 0]))
        seed_cs.append(-R.T @ t)

    # Keep only frames that have both lines and a bootstrap camera.
    per_frame_lines = {f: per_frame_lines[f] for f in boot}
    if len(per_frame_lines) < 2:
        print("not enough frames with both detected lines and a bootstrap camera")
        return 1

    rich_cs = [
        -np.array(cf.R).T @ np.array(cf.t)
        for cf in track.frames
        if cf.frame in rich and cf.frame in boot and cf.t is not None
    ]
    c_center = np.median(np.stack(rich_cs if rich_cs else seed_cs), axis=0)
    cx, cy = (track.principal_point
              if track.principal_point is not None
              else (image_size[0] / 2.0, image_size[1] / 2.0))
    # The bootstrap track's distortion is usually the lens-from-anchor
    # estimate, which saturates to non-physical values on real clips (the
    # LM absorbing click noise — see the experiment note). The profile
    # holds the lens fixed, so seed it with zero distortion; the
    # static-camera bundle adjustment refines the real distortion.
    lens_seed = (float(cx), float(cy), 0.0, 0.0)

    print(f"loaded {len(per_frame_lines)} frames; C seed = {np.round(c_center, 3)}")
    print(f"lens seed (cx, cy, k1, k2) = {np.round(lens_seed, 4)}")

    coarse = profile_camera_centre(
        per_frame_lines, image_size,
        c_grid=make_c_grid(c_center, extent_m=args.coarse_extent_m,
                           n_steps=args.coarse_steps),
        lens_seed=lens_seed, per_frame_bootstrap=boot,
    )
    print(f"coarse argmin C = {np.round(coarse.argmin_c, 3)}  "
          f"mean RMS = {coarse.mean_rms.min():.3f} px")

    fine = profile_camera_centre(
        per_frame_lines, image_size,
        c_grid=make_c_grid(coarse.argmin_c, extent_m=args.fine_extent_m,
                           n_steps=args.fine_steps),
        lens_seed=lens_seed, per_frame_bootstrap=coarse.per_frame_seeds,
    )
    best = int(np.argmin(fine.mean_rms))
    print(f"\nfine argmin C = {np.round(fine.argmin_c, 3)}  "
          f"(profile, lens fixed): mean={fine.mean_rms[best]:.3f} "
          f"P95={fine.p95_rms[best]:.3f} max={fine.max_rms[best]:.3f} px")

    # The profile holds the lens fixed, so its RMS is only an upper bound.
    # Run the actual static-camera bundle adjustment (pinhole_k1k2, which
    # frees cx/cy/k1/k2) from the profile seed for the definitive floor.
    print("\nrunning the static-camera bundle adjustment (pinhole_k1k2) "
          "for the definitive floor...")
    sol = solve_static_camera_from_lines(
        per_frame_lines, image_size,
        c_seed=fine.argmin_c, lens_seed=lens_seed,
        per_frame_seeds=fine.per_frame_seeds,
        point_hints=anchor_landmarks or None,
        lens_model="pinhole_k1k2",
    )
    rms = np.array(
        [v for v in sol.per_frame_line_rms.values() if np.isfinite(v)]
    )
    print(f"  BA C = {np.round(sol.camera_centre, 3)}  "
          f"distortion = {np.round(sol.distortion, 4)}")
    print(f"  BA line RMS: mean={rms.mean():.3f} median={np.median(rms):.3f} "
          f"max={rms.max():.3f} P95={np.percentile(rms, 95):.3f} px")
    print(f"  frac <1px={(rms < 1.0).mean():.3f}  <2px={(rms < 2.0).mean():.3f}")
    print()
    if rms.mean() < 1.0:
        print("VERDICT: pinhole_k1k2 reaches sub-pixel static-C — keep "
              "line_extraction_lens_model: pinhole_k1k2.")
    else:
        print("VERDICT: pinhole_k1k2 BA floor is above 1 px — try "
              "line_extraction_lens_model: brown_conrady.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
