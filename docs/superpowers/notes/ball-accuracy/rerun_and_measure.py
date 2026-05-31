"""Re-run the ball stage with current code (anchor-driven, no WASB) and
measure anchor accuracy.

The goal is *anchor* accuracy. Anchored frames are reconstructed from
anchors + camera + refined_poses, independent of WASB detections, so a
no-op detector gives the true current-code trajectory at anchored frames
in seconds. Writes the refreshed track to a temp path (never clobbers the
real output) and reports per-state lateral error against the clicked ray.

Run from repo root with the venv active:
    python docs/superpowers/notes/ball-accuracy/rerun_and_measure.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402

from src.stages.ball import BallStage  # noqa: E402
from src.utils.ball_detector import BallDetector  # noqa: E402
from src.schemas.ball_anchor import BallAnchorSet  # noqa: E402
from src.schemas.ball_track import BallTrack  # noqa: E402
from src.schemas.camera_track import CameraTrack  # noqa: E402
from src.utils.camera_projection import (  # noqa: E402
    project_world_to_image, undistort_pixel,
)


class NoopDetector(BallDetector):
    """Returns None for every frame — anchored frames still reconstruct."""

    def detect(self, frame):  # noqa: ANN001
        return None


CLIPS = [
    ("gberch", "output", "gberch"),
    ("kroupi01", "output-kroupi", "kroupi01"),
    ("origi01", "output-origi", "origi01"),
]


def ray_from_pixel(uv, K, R, t, distortion):
    uv = np.asarray(uv, dtype=float)
    if distortion != (0.0, 0.0):
        uv = undistort_pixel(uv, K, distortion)
    C = -R.T @ t
    d_world = R.T @ (np.linalg.inv(K) @ np.array([uv[0], uv[1], 1.0]))
    return C, d_world / np.linalg.norm(d_world)


def perp(P, C, d_hat):
    v = P - C
    along = float(np.dot(v, d_hat))
    return float(np.linalg.norm(P - (C + along * d_hat))), along


def measure(clip_id, out_dir, shot_id, track_path):
    anc = BallAnchorSet.load(Path(out_dir) / "ball" / f"{shot_id}_ball_anchors.json")
    track = BallTrack.load(track_path)
    cam = CameraTrack.load(Path(out_dir) / "camera" / f"{shot_id}_camera_track.json")
    dist = cam.distortion
    per_K = {f.frame: np.array(f.K) for f in cam.frames}
    per_R = {f.frame: np.array(f.R) for f in cam.frames}
    tfb = np.array(cam.t_world)
    per_t = {f.frame: (np.array(f.t) if f.t is not None else tfb) for f in cam.frames}
    f2w = {f.frame: f.world_xyz for f in track.frames}

    from collections import defaultdict
    by_state = defaultdict(list)
    worst = []
    for a in anc.anchors:
        if a.image_xy is None or a.frame not in per_K:
            continue
        w = f2w.get(a.frame)
        if w is None:
            by_state[a.state].append((None, None))
            continue
        P = np.array(w, dtype=float)
        uvp = project_world_to_image(per_K[a.frame], per_R[a.frame],
                                     per_t[a.frame], dist, P.reshape(1, 3))[0]
        reproj = float(np.linalg.norm(uvp - np.array(a.image_xy)))
        C, dh = ray_from_pixel(a.image_xy, per_K[a.frame], per_R[a.frame],
                               per_t[a.frame], dist)
        lat, _ = perp(P, C, dh)
        by_state[a.state].append((reproj, lat))
        worst.append((lat, a.frame, a.state, reproj))

    print(f"\n### {clip_id}  ({len(track.flight_segments)} flight segs)")
    all_lat = []
    for st in sorted(by_state):
        vals = [v for v in by_state[st] if v[1] is not None]
        nmiss = sum(1 for v in by_state[st] if v[1] is None)
        if not vals:
            print(f"    {st:<16} n={len(by_state[st])} all world=None ({nmiss} missing)")
            continue
        lats = np.array([v[1] for v in vals])
        rps = np.array([v[0] for v in vals])
        all_lat.extend(lats.tolist())
        print(f"    {st:<16} n={len(vals):<3} "
              f"lateral med={np.median(lats):.3f} max={lats.max():.3f}  "
              f"reproj med={np.median(rps):6.1f} max={rps.max():7.1f}  "
              f">10cm:{int((lats>0.10).sum())}/{len(lats)}"
              + (f"  (+{nmiss} None)" if nmiss else ""))
    all_lat = np.array(all_lat)
    if len(all_lat):
        print(f"    {'OVERALL':<16} median={np.median(all_lat):.3f} "
              f"mean={all_lat.mean():.3f} max={all_lat.max():.3f}  "
              f">10cm:{int((all_lat>0.10).sum())}/{len(all_lat)}")
    worst.sort(reverse=True)
    print("    worst 5:", [(round(l, 2), f, s) for l, f, s, _ in worst[:5]])


def main():
    cfg = yaml.safe_load(open(ROOT / "config" / "default.yaml"))
    tmp = Path(tempfile.mkdtemp(prefix="ball_rerun_"))
    print(f"(writing refreshed tracks under {tmp})")
    for clip_id, out_dir, shot_id in CLIPS:
        out_p = ROOT / out_dir
        cam = out_p / "camera" / f"{shot_id}_camera_track.json"
        clip = out_p / "shots" / f"{shot_id}.mp4"
        if not cam.exists() or not clip.exists():
            print(f"\n### {clip_id}: missing camera/clip, skipping")
            continue
        stage = BallStage(config=cfg, output_dir=out_p, ball_detector=NoopDetector())
        stage.shot_filter = shot_id
        track_out = tmp / f"{shot_id}_ball_track.json"
        try:
            stage._run_shot(shot_id, clip, cam, track_out, cfg["ball"], NoopDetector())
        except Exception as exc:
            import traceback
            print(f"\n### {clip_id}: re-run FAILED: {exc}")
            traceback.print_exc()
            continue
        measure(clip_id, out_dir, shot_id, track_out)


if __name__ == "__main__":
    main()
