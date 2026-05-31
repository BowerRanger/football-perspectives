"""Measure ball-track accuracy at anchor frames.

For each clip we have user-placed ball anchors (the ground truth: a
clicked pixel + a state, and for player_touch a specific player+bone).
The *true* ball centre lies on the camera ray through the clicked pixel.
We measure, at every anchor frame:

  - reproj_px:   reprojection error of the emitted world_xyz vs the
                 clicked anchor pixel (pixels).
  - lateral_m:   perpendicular distance from the emitted world_xyz to
                 the clicked-pixel ray (metres). This is a depth-
                 independent lower bound on the 3D error: if it exceeds
                 0.10 m the ball is laterally off regardless of depth.
  - state:       the anchor state.

Run from repo root with the project venv active:
    python docs/superpowers/notes/ball-accuracy/measure_anchor_error.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from src.schemas.ball_anchor import BallAnchorSet  # noqa: E402
from src.schemas.ball_track import BallTrack  # noqa: E402
from src.schemas.camera_track import CameraTrack  # noqa: E402
from src.utils.camera_projection import project_world_to_image, undistort_pixel  # noqa: E402

CLIPS = [
    ("gberch", "output/ball/gberch_ball_anchors.json",
     "output/ball/gberch_ball_track.json",
     "output/camera/gberch_camera_track.json"),
    ("kroupi01", "output-kroupi/ball/kroupi01_ball_anchors.json",
     "output-kroupi/ball/kroupi01_ball_track.json",
     "output-kroupi/camera/kroupi01_camera_track.json"),
    ("origi01", "output-origi/ball/origi01_ball_anchors.json",
     "output-origi/ball/origi01_ball_track.json",
     "output-origi/camera/origi01_camera_track.json"),
]


def ray_from_pixel(uv, K, R, t, distortion):
    """Return (C, d_hat): camera centre and unit ray direction in world."""
    uv = np.asarray(uv, dtype=float)
    if distortion != (0.0, 0.0):
        uv = undistort_pixel(uv, K, distortion)
    C = -R.T @ t
    pixel_h = np.array([uv[0], uv[1], 1.0])
    d_cam = np.linalg.inv(K) @ pixel_h
    d_world = R.T @ d_cam
    d_hat = d_world / np.linalg.norm(d_world)
    return C, d_hat


def perp_distance(P, C, d_hat):
    """Perpendicular distance of point P from the ray (C, d_hat)."""
    v = P - C
    along = np.dot(v, d_hat)
    closest = C + along * d_hat
    return float(np.linalg.norm(P - closest)), float(along)


def main():
    for clip_id, anc_path, trk_path, cam_path in CLIPS:
        anc_p, trk_p, cam_p = Path(anc_path), Path(trk_path), Path(cam_path)
        if not (anc_p.exists() and trk_p.exists() and cam_p.exists()):
            print(f"\n### {clip_id}: MISSING files, skipping")
            continue
        anchors = BallAnchorSet.load(anc_p)
        track = BallTrack.load(trk_p)
        cam = CameraTrack.load(cam_p)
        distortion = cam.distortion
        per_K = {f.frame: np.array(f.K) for f in cam.frames}
        per_R = {f.frame: np.array(f.R) for f in cam.frames}
        t_fallback = np.array(cam.t_world)
        per_t = {f.frame: (np.array(f.t) if f.t is not None else t_fallback)
                 for f in cam.frames}
        frame_to_world = {f.frame: f.world_xyz for f in track.frames}
        frame_to_state = {f.frame: f.state for f in track.frames}

        print(f"\n{'='*70}\n### {clip_id}  ({len(anchors.anchors)} anchors, "
              f"{len(track.frames)} track frames, "
              f"{len(track.flight_segments)} flight segs)\n{'='*70}")
        rows = []
        for a in anchors.anchors:
            fi = a.frame
            if a.image_xy is None:
                continue  # off_screen_flight
            world = frame_to_world.get(fi)
            if fi not in per_K:
                rows.append((fi, a.state, None, None, "no-camera"))
                continue
            if world is None:
                rows.append((fi, a.state, None, None,
                             f"track={frame_to_state.get(fi,'?')}/None"))
                continue
            P = np.array(world, dtype=float)
            # reprojection error
            uv_proj = project_world_to_image(
                per_K[fi], per_R[fi], per_t[fi], distortion,
                P.reshape(1, 3))[0]
            reproj = float(np.linalg.norm(uv_proj - np.array(a.image_xy)))
            # lateral (perpendicular to ray)
            C, d_hat = ray_from_pixel(a.image_xy, per_K[fi], per_R[fi],
                                      per_t[fi], distortion)
            lateral, depth = perp_distance(P, C, d_hat)
            rows.append((fi, a.state, reproj, lateral,
                         f"z={P[2]:.2f} depth={depth:.1f}m "
                         f"track={frame_to_state.get(fi,'?')}"))

        # Per-state summary
        from collections import defaultdict
        by_state = defaultdict(list)
        print(f"{'frame':>6} {'state':<16} {'reproj_px':>9} {'lateral_m':>9}  detail")
        for fi, st, reproj, lat, detail in rows:
            rp = f"{reproj:9.1f}" if reproj is not None else f"{'--':>9}"
            lm = f"{lat:9.3f}" if lat is not None else f"{'--':>9}"
            flag = ""
            if lat is not None and lat > 0.10:
                flag = "  <<< >10cm"
            print(f"{fi:>6} {st:<16} {rp} {lm}  {detail}{flag}")
            if lat is not None:
                by_state[st].append((reproj, lat))
        print("\n  -- per-state median (reproj_px / lateral_m) --")
        for st in sorted(by_state):
            vals = by_state[st]
            reprojs = np.array([v[0] for v in vals])
            lats = np.array([v[1] for v in vals])
            n_bad = int(np.sum(lats > 0.10))
            print(f"    {st:<16} n={len(vals):<3} "
                  f"reproj med={np.median(reprojs):6.1f} max={reprojs.max():7.1f}  "
                  f"lateral med={np.median(lats):.3f} max={lats.max():.3f}  "
                  f">10cm: {n_bad}/{len(vals)}")
        all_lats = np.array([l for v in by_state.values() for (_, l) in v])
        if len(all_lats):
            print(f"\n  OVERALL lateral: median={np.median(all_lats):.3f}m "
                  f"mean={all_lats.mean():.3f}m max={all_lats.max():.3f}m  "
                  f">10cm: {int(np.sum(all_lats>0.10))}/{len(all_lats)}")


if __name__ == "__main__":
    main()
