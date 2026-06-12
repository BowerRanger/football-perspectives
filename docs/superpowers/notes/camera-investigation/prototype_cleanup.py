"""Prototype + sweep for per-player robust cleanup (gap-fill, Hampel
outlier rejection, physical velocity limit) on real hmr_world tracks.

Measures before/after jitter metrics across output dirs so we can dial
in a generic config before wiring it into the stage. Operates on root_t
XY (the artifact the user sees); the stage port will carry rotation/pose
along the same trusted/interpolated mask.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter

_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(_ROOT))
from src.schemas.smpl_world import SmplWorldTrack  # noqa: E402

CONF_MIN = 0.3


# ───────────────────────── cleanup primitives ──────────────────────────

def resample_uniform(frames: np.ndarray, xy: np.ndarray):
    """Dense XY over [f0, fN]; returns (dense_frames, dense_xy, present_mask)."""
    f0, fN = int(frames[0]), int(frames[-1])
    dense_f = np.arange(f0, fN + 1)
    present = np.zeros(len(dense_f), dtype=bool)
    idx = frames - f0
    present[idx] = True
    dense_xy = np.empty((len(dense_f), 2))
    for a in range(2):
        dense_xy[:, a] = np.interp(dense_f, frames, xy[:, a])
    return dense_f, dense_xy, present


def hampel_reject(xy: np.ndarray, present: np.ndarray, *, window: int,
                  k: float, abs_floor: float, v_max_step: float):
    """Flag outliers by local-median deviation (metres) OR there-and-back
    velocity spike (> v_max_step from BOTH neighbours). Returns trusted
    mask (True = keep)."""
    n = len(xy)
    half = max(1, window // 2)
    trusted = present.copy()
    dev = np.zeros(n)
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        win = xy[lo:hi]
        med = np.median(win, axis=0)
        d = np.linalg.norm(win - med, axis=1)
        mad = np.median(d)
        thresh = max(k * 1.4826 * mad, abs_floor)
        dev[i] = np.linalg.norm(xy[i] - med)
        if dev[i] > thresh:
            trusted[i] = False
    # there-and-back velocity spike: far from both neighbours
    for i in range(1, n - 1):
        if not trusted[i]:
            continue
        s_prev = np.linalg.norm(xy[i] - xy[i - 1])
        s_next = np.linalg.norm(xy[i + 1] - xy[i])
        back = np.linalg.norm(xy[i + 1] - xy[i - 1])
        if s_prev > v_max_step and s_next > v_max_step and back < min(s_prev, s_next):
            trusted[i] = False
    return trusted


def reinterp(xy: np.ndarray, trusted: np.ndarray) -> np.ndarray:
    if trusted.all() or not trusted.any():
        return xy.copy()
    out = xy.copy()
    ti = np.where(trusted)[0]
    grid = np.arange(len(xy))
    for a in range(2):
        out[:, a] = np.interp(grid, ti, xy[ti, a])
    return out


def velocity_limit(xy: np.ndarray, dmax: float) -> np.ndarray:
    """Forward/backward displacement-limited filter; bounds |Δ| ≤ dmax."""
    n = len(xy)
    if n < 2:
        return xy.copy()

    def fwd(seq):
        out = seq.copy()
        for i in range(1, n):
            step = out[i] - out[i - 1]
            d = np.linalg.norm(step)
            if d > dmax:
                out[i] = out[i - 1] + step * (dmax / d)
        return out

    f = fwd(xy)
    b = fwd(xy[::-1])[::-1]
    return 0.5 * (f + b)


def savgol_xy(xy: np.ndarray, window: int, poly: int) -> np.ndarray:
    n = len(xy)
    w = min(window, n - (1 - n % 2))
    if w < poly + 2:
        return xy
    return savgol_filter(xy, window_length=w, polyorder=poly, axis=0)


def clean_player(frames, xy, cfg):
    dense_f, dense_xy, present = resample_uniform(frames, xy)
    out = dense_xy
    if cfg.get("hampel"):
        trusted = hampel_reject(
            out, present, window=cfg["hampel_window"], k=cfg["hampel_k"],
            abs_floor=cfg["hampel_floor"], v_max_step=cfg["v_max_step"])
        out = reinterp(out, trusted)
    if cfg.get("velclamp"):
        out = velocity_limit(out, cfg["v_max_step"])
    if cfg.get("savgol"):
        out = savgol_xy(out, cfg["savgol_window"], cfg["savgol_poly"])
    return dense_f, out


# ───────────────────────── metrics ──────────────────────────

def player_metrics(dense_f, xy, fps):
    d = np.linalg.norm(np.diff(xy, axis=0), axis=1) * fps
    jerk = (np.linalg.norm(np.diff(xy, n=2, axis=0), axis=1) * fps * fps
            if len(xy) >= 3 else np.zeros(0))
    return d, jerk


def evaluate(output_dir: Path, cfg: dict, fps: float = 30.0):
    by_shot = defaultdict(list)
    for npz in sorted((output_dir / "hmr_world").glob("*_smpl_world.npz")):
        try:
            tr = SmplWorldTrack.load(npz)
        except Exception:
            continue
        by_shot[tr.shot_id or npz.name].append(tr)

    results = {}
    for shot, tracks in by_shot.items():
        speeds, jerks = [], []
        # common-mode after cleaning
        frame_xy = defaultdict(list)
        for tr in tracks:
            frames = np.asarray(tr.frames, np.int64)
            conf = np.asarray(tr.confidence, float)
            xy = np.asarray(tr.root_t, float)[:, :2]
            keep = conf >= CONF_MIN
            if keep.sum() < 2:
                continue
            frames, xy = frames[keep], xy[keep]
            # keep contiguous unique
            uf, ui = np.unique(frames, return_index=True)
            frames, xy = uf, xy[ui]
            df, cxy = clean_player(frames, xy, cfg)
            sp, jk = player_metrics(df, cxy, fps)
            speeds.append(sp)
            jerks.append(jk)
            for f, p in zip(df, cxy):
                frame_xy[int(f)].append(p)
        S = np.concatenate(speeds) if speeds else np.zeros(0)
        J = np.concatenate(jerks) if jerks else np.zeros(0)
        meds = {f: np.median(np.stack(v), axis=0)
                for f, v in frame_xy.items() if len(v) >= 3}
        mfs = sorted(meds)
        drift = np.asarray([np.linalg.norm(meds[b] - meds[a]) * fps
                            for a, b in zip(mfs[:-1], mfs[1:]) if b - a == 1])
        results[shot] = {
            "sp_p50": np.percentile(S, 50) if S.size else 0,
            "sp_p90": np.percentile(S, 90) if S.size else 0,
            "sp_p99": np.percentile(S, 99) if S.size else 0,
            "impl%": 100 * (S > 12).mean() if S.size else 0,
            "jerk_p90": np.percentile(J, 90) if J.size else 0,
            "drift_p90": np.percentile(drift, 90) if drift.size else 0,
            "drift_max": drift.max() if drift.size else 0,
        }
    return results


CONFIGS = {
    "0-raw": {},
    "1-savgol7": {"savgol": True, "savgol_window": 7, "savgol_poly": 2},
    "2-hampel+sav": {"hampel": True, "hampel_window": 13, "hampel_k": 3.0,
                     "hampel_floor": 0.6, "v_max_step": 0.4,
                     "savgol": True, "savgol_window": 7, "savgol_poly": 2},
    "3-hamp+vclamp+sav": {"hampel": True, "hampel_window": 13, "hampel_k": 3.0,
                          "hampel_floor": 0.6, "v_max_step": 0.4,
                          "velclamp": True,
                          "savgol": True, "savgol_window": 9, "savgol_poly": 2},
    "4-aggressive": {"hampel": True, "hampel_window": 15, "hampel_k": 2.5,
                     "hampel_floor": 0.45, "v_max_step": 0.37,
                     "velclamp": True,
                     "savgol": True, "savgol_window": 11, "savgol_poly": 2},
}

if __name__ == "__main__":
    dirs = sys.argv[1:] or ["output", "output-gberch", "output-kroupi"]
    for d in dirs:
        p = _ROOT / d
        print(f"\n########## {d} ##########")
        for name, cfg in CONFIGS.items():
            res = evaluate(p, cfg)
            for shot, m in res.items():
                print(f"  {name:18} {shot:>10}  "
                      f"sp[p50/p90/p99]={m['sp_p50']:4.1f}/{m['sp_p90']:4.1f}/{m['sp_p99']:5.1f}  "
                      f"impl={m['impl%']:4.1f}%  jerk_p90={m['jerk_p90']:6.0f}  "
                      f"drift[p90/max]={m['drift_p90']:5.1f}/{m['drift_max']:5.0f}")
