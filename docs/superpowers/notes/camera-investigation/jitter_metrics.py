"""Reusable jitter-metrics harness for the refined_poses work.

Computes, per output dir, the three target artifacts plus a smoothness
proxy, on either the raw hmr_world tracks or the refined_poses output.

Metrics (all on root_t XY, pitch-metres, per shot):
  drift_*   : cross-player common-mode speed (camera drift, artifact #1)
              = frame-to-frame delta of the cross-player MEDIAN position.
  speed_*   : per-player consecutive-frame speed m/s (artifact #2);
              implausible = fraction of frame-pairs over SPEED_MAX.
  gap_*     : missing-frame gaps within each player's anchored span
              (artifact #3).
  jerk_*    : per-player |2nd difference| of XY position * fps^2 — an
              acceleration proxy for visible high-frequency jitter; the
              single best scalar for "does it look smooth".

Usage:
    from jitter_metrics import metrics_for_dir
    m = metrics_for_dir(Path("output"), source="raw")     # hmr_world
    m = metrics_for_dir(Path("output"), source="refined")  # refined_poses
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(_ROOT))

from src.schemas.refined_pose import RefinedPose  # noqa: E402
from src.schemas.smpl_world import SmplWorldTrack  # noqa: E402

CONF_MIN = 0.3
SPEED_MAX = 12.0  # m/s; elite sprint ~10, so >12 frame-to-frame is implausible


def _fps(output_dir: Path) -> float:
    cam = output_dir / "camera"
    cands = [cam / "camera_track.json", *sorted(cam.glob("*_camera_track.json"))]
    for p in cands:
        if p.exists():
            try:
                from src.schemas.camera_track import CameraTrack
                return float(CameraTrack.load(p).fps)
            except Exception:
                pass
    return 25.0


def _pct(a: np.ndarray, q: list[float]) -> list[float]:
    if a.size == 0:
        return [0.0] * len(q)
    return [float(x) for x in np.percentile(a, q)]


def _load_tracks(output_dir: Path, source: str) -> dict[str, list]:
    """Return {shot_id: [tracks]} with a uniform (frames, root_t, conf)
    interface for both raw hmr_world and refined_poses output."""
    by_shot: dict[str, list] = defaultdict(list)
    if source == "raw":
        for npz in sorted((output_dir / "hmr_world").glob("*_smpl_world.npz")):
            try:
                tr = SmplWorldTrack.load(npz)
            except Exception:
                continue
            by_shot[tr.shot_id or npz.name].append(tr)
    elif source == "refined":
        for npz in sorted((output_dir / "refined_poses").glob("*_refined.npz")):
            try:
                rp = RefinedPose.load(npz)
            except Exception:
                continue
            # refined is on the reference timeline; group all together
            by_shot["__refined__"].append(rp)
    else:
        raise ValueError(source)
    return by_shot


def metrics_for_dir(output_dir: Path, source: str = "raw") -> dict:
    fps = _fps(output_dir)
    by_shot = _load_tracks(output_dir, source)
    shots_out = {}
    for shot, tracks in by_shot.items():
        speeds, jerks = [], []
        n_pairs = 0
        n_implausible = 0
        total_gaps = total_gap_frames = total_anchored = 0
        for tr in tracks:
            frames = np.asarray(tr.frames, dtype=np.int64)
            rt = np.asarray(tr.root_t, dtype=np.float64)[:, :2]
            conf = np.asarray(tr.confidence, dtype=np.float64)
            if len(frames) < 2:
                continue
            df = np.diff(frames)
            total_gaps += int((df > 1).sum())
            total_gap_frames += int((df[df > 1] - 1).sum()) if (df > 1).any() else 0
            total_anchored += int((conf >= CONF_MIN).sum())
            cons = df == 1
            cpair = (conf[:-1] >= CONF_MIN) & (conf[1:] >= CONF_MIN)
            valid = cons & cpair
            d = np.linalg.norm(np.diff(rt, axis=0), axis=1)
            sp = d[valid] * fps
            speeds.append(sp)
            n_pairs += int(valid.sum())
            n_implausible += int((sp > SPEED_MAX).sum())
            # jerk proxy: 2nd diff over runs of consecutive frames
            if len(rt) >= 3:
                a = np.linalg.norm(np.diff(rt, n=2, axis=0), axis=1) * fps * fps
                cons2 = (df[:-1] == 1) & (df[1:] == 1)
                jerks.append(a[cons2])
        speeds_all = np.concatenate(speeds) if speeds else np.zeros(0)
        jerks_all = np.concatenate(jerks) if jerks else np.zeros(0)

        # common-mode drift: cross-player median position per frame
        frame_xy: dict[int, list[np.ndarray]] = defaultdict(list)
        for tr in tracks:
            frames = np.asarray(tr.frames, dtype=np.int64)
            rt = np.asarray(tr.root_t, dtype=np.float64)[:, :2]
            conf = np.asarray(tr.confidence, dtype=np.float64)
            for i, f in enumerate(frames):
                if conf[i] >= CONF_MIN:
                    frame_xy[int(f)].append(rt[i])
        meds = {f: np.median(np.stack(v), axis=0)
                for f, v in frame_xy.items() if len(v) >= 3}
        mfs = sorted(meds)
        drift = np.asarray([
            np.linalg.norm(meds[b] - meds[a]) * fps
            for a, b in zip(mfs[:-1], mfs[1:]) if b - a == 1
        ])

        sp50, sp90, sp99, spmax = _pct(speeds_all, [50, 90, 99, 100])
        jk50, jk90, jkmax = _pct(jerks_all, [50, 90, 100])
        dr50, dr90, drmax = _pct(drift, [50, 90, 100])
        shots_out[shot] = {
            "n_players": len(tracks),
            "speed_m_s": {"p50": sp50, "p90": sp90, "p99": sp99, "max": spmax},
            "implausible_frac": (n_implausible / n_pairs) if n_pairs else 0.0,
            "implausible_n": n_implausible,
            "jerk_m_s2": {"p50": jk50, "p90": jk90, "max": jkmax},
            "drift_m_s": {"p50": dr50, "p90": dr90, "max": drmax},
            "gaps": total_gaps,
            "gap_frames": total_gap_frames,
            "anchored_frames": total_anchored,
        }
    return {"fps": fps, "shots": shots_out}


def _fmt(m: dict) -> str:
    lines = []
    for shot, s in m["shots"].items():
        lines.append(
            f"  {shot:>12}  n={s['n_players']:>2}  "
            f"speed[p50/p90/p99/max]={s['speed_m_s']['p50']:.1f}/"
            f"{s['speed_m_s']['p90']:.1f}/{s['speed_m_s']['p99']:.1f}/"
            f"{s['speed_m_s']['max']:.0f}  "
            f"impl={s['implausible_frac']*100:.1f}%({s['implausible_n']})  "
            f"jerk[p90]={s['jerk_m_s2']['p90']:.0f}  "
            f"drift[p50/p90/max]={s['drift_m_s']['p50']:.1f}/"
            f"{s['drift_m_s']['p90']:.1f}/{s['drift_m_s']['max']:.0f}  "
            f"gaps={s['gaps']}({s['gap_frames']}f)"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    dirs = sys.argv[1:] or ["output", "output-gberch", "output-kroupi"]
    for d in dirs:
        p = _ROOT / d
        for source in ("raw", "refined"):
            try:
                m = metrics_for_dir(p, source=source)
            except Exception as exc:
                print(f"{d} [{source}]: ERROR {exc}")
                continue
            print(f"\n== {d} [{source}] fps={m['fps']} ==")
            print(_fmt(m))
