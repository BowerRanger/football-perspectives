"""Characterize the three jitter artifacts in raw hmr_world tracks.

Run:  python docs/superpowers/notes/camera-investigation/characterize_jitter.py <output_dir> [...]

For each output dir, reports per-player:
  - frame gaps (missing-frame artifact #3)
  - per-frame XY speed distribution (fast-motion artifact #2)
and per-shot:
  - cross-player common-mode XY motion (camera-drift artifact #1)
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from src.schemas.smpl_world import SmplWorldTrack  # noqa: E402

CONF_MIN = 0.3


def load_shot_tracks(hmr_dir: Path) -> dict[str, list[SmplWorldTrack]]:
    by_shot: dict[str, list[SmplWorldTrack]] = defaultdict(list)
    for npz in sorted(hmr_dir.glob("*_smpl_world.npz")):
        try:
            tr = SmplWorldTrack.load(npz)
        except Exception as exc:
            print(f"  !! failed to load {npz.name}: {exc}")
            continue
        by_shot[tr.shot_id or npz.name].append(tr)
    return by_shot


def fps_of(output_dir: Path) -> float:
    for p in [output_dir / "camera" / "camera_track.json", *sorted((output_dir / "camera").glob("*_camera_track.json"))]:
        if p.exists():
            try:
                from src.schemas.camera_track import CameraTrack
                return float(CameraTrack.load(p).fps)
            except Exception:
                pass
    return 25.0


def analyze(output_dir: Path) -> None:
    hmr = output_dir / "hmr_world"
    if not hmr.exists():
        print(f"[{output_dir}] no hmr_world — skip")
        return
    fps = fps_of(output_dir)
    by_shot = load_shot_tracks(hmr)
    print(f"\n========== {output_dir}  (fps={fps}) ==========")
    for shot, tracks in by_shot.items():
        print(f"\n--- shot {shot}: {len(tracks)} players ---")
        # ----- per-player gaps + speed -----
        all_speeds = []
        total_gaps = 0
        total_gap_frames = 0
        big_jump_total = 0
        n_anchored_frames = 0
        for tr in tracks:
            frames = np.asarray(tr.frames, dtype=np.int64)
            rt = np.asarray(tr.root_t, dtype=np.float64)
            conf = np.asarray(tr.confidence, dtype=np.float64)
            if len(frames) < 2:
                continue
            # gaps: where consecutive frame indices differ by > 1
            df = np.diff(frames)
            gaps = df[df > 1] - 1
            total_gaps += int((df > 1).sum())
            total_gap_frames += int(gaps.sum()) if gaps.size else 0
            # speed only between consecutive (df==1), confident frames
            cons = df == 1
            d_xy = np.linalg.norm(np.diff(rt[:, :2], axis=0), axis=1)
            conf_pair = (conf[:-1] >= CONF_MIN) & (conf[1:] >= CONF_MIN)
            valid = cons & conf_pair
            sp = d_xy[valid] * fps  # m/s
            all_speeds.append(sp)
            big_jump_total += int((sp > 12.0).sum())  # >12 m/s is implausible
            n_anchored_frames += int((conf >= CONF_MIN).sum())
        speeds = np.concatenate(all_speeds) if all_speeds else np.zeros(0)
        if speeds.size:
            pct = np.percentile(speeds, [50, 90, 99, 100])
            print(f"  speed m/s: median={pct[0]:.2f} p90={pct[1]:.2f} "
                  f"p99={pct[2]:.2f} max={pct[3]:.2f}  "
                  f"(>12m/s frames: {big_jump_total})")
        print(f"  gaps: {total_gaps} gaps, {total_gap_frames} missing frames "
              f"across {n_anchored_frames} anchored frames")

        # ----- cross-player common-mode XY motion (camera drift) -----
        frame_xy: dict[int, list[np.ndarray]] = defaultdict(list)
        for tr in tracks:
            frames = np.asarray(tr.frames, dtype=np.int64)
            rt = np.asarray(tr.root_t, dtype=np.float64)
            conf = np.asarray(tr.confidence, dtype=np.float64)
            for i, f in enumerate(frames):
                if conf[i] >= CONF_MIN:
                    frame_xy[int(f)].append(rt[i, :2])
        # per-frame median position, then its frame-to-frame delta is
        # the common-mode (camera) motion estimate
        fs = sorted(frame_xy.keys())
        meds = {f: np.median(np.stack(frame_xy[f]), axis=0)
                for f in fs if len(frame_xy[f]) >= 3}
        med_fs = sorted(meds.keys())
        cm = []
        for a, b in zip(med_fs[:-1], med_fs[1:]):
            if b - a == 1:
                cm.append(np.linalg.norm(meds[b] - meds[a]) * fps)
        cm = np.asarray(cm)
        if cm.size:
            print(f"  common-mode (median-player) speed m/s: "
                  f"median={np.median(cm):.2f} p90={np.percentile(cm,90):.2f} "
                  f"max={cm.max():.2f}")


if __name__ == "__main__":
    dirs = sys.argv[1:] or ["output", "output-gberch", "output-kroupi"]
    base = Path(__file__).resolve().parents[4]
    for d in dirs:
        analyze(base / d)
