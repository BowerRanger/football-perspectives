"""Prototype: spin identifiability on a real flight segment, with and without
cross-replay triangulated 3D fixes.

Takes origi01 segment 9 (frames 454-488, flagged underconstrained at ~30 px in
the monocular solve) and fits:

  A. gravity-only parabola               (repo fit_parabola_to_image_observations)
  B. gravity+Magnus ODE, monocular       (repo fit_magnus_trajectory)
  C. gravity+Magnus ODE + triangulated   (custom LM: pixel residuals + 3D fix
     fixes from the origi02 replay        residuals from replay_triangulation)

Reports pixel residual, recovered v0 / omega, apex height and lateral Magnus
deflection for each, to answer: is spin identifiable, and what do the replay
fixes change?

Usage:
    .venv/bin/python prototypes/spin_fit_with_fixes.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "prototypes"))

from src.utils.bundle_adjust import (  # noqa: E402
    _integrate_magnus_positions,
    fit_magnus_trajectory,
    fit_parabola_to_image_observations,
)

import replay_triangulation as rt  # noqa: E402

SEG_START, SEG_END = 454, 488
FIX_WEIGHT_PX_PER_M = 30.0  # ~1 m of 3D fix error costs as much as 30 px
G = -9.81
DRAG_K = 0.005


def collect_segment_obs():
    cams = rt.load_cameras(rt.REF_SHOT)
    obs = rt.load_obs(rt.REF_SHOT)
    rows = []
    for f in range(SEG_START, SEG_END + 1):
        if f in obs and obs[f][1] >= rt.MIN_CONF and f in cams:
            uv = obs[f][0]
            K, R, t = cams[f]
            rows.append((f, uv, K, R, t))
    return rows


def collect_fixes():
    """Triangulated 3D fixes inside the segment, from the replay prototype."""
    cams_ref = rt.load_cameras(rt.REF_SHOT)
    cams_rep = rt.load_cameras(rt.REP_SHOT)
    obs_ref = rt.load_obs(rt.REF_SHOT)
    obs_rep = rt.load_obs(rt.REP_SHOT)
    _, rows = rt.evaluate(-144.0, obs_ref, obs_rep, cams_ref, cams_rep)
    return [(r["f_ref"], r["xyz"]) for r in rows
            if r["miss_m"] < 1.0 and SEG_START <= r["f_ref"] <= SEG_END]


def describe(label, p0, v0, omega, residual_px, fps, frames):
    dt = (np.array(frames) - frames[0]) / fps
    pts = _integrate_magnus_positions(p0, v0, omega, np.array([0.0, 0.0, G]), DRAG_K, dt)
    # lateral Magnus deflection: distance of endpoint from the no-spin trajectory
    pts_nospin = _integrate_magnus_positions(p0, v0, np.zeros(3), np.array([0.0, 0.0, G]), DRAG_K, dt)
    defl = float(np.linalg.norm(pts[-1] - pts_nospin[-1]))
    print(f"\n  [{label}]")
    print(f"    residual          : {residual_px:.2f} px")
    print(f"    launch speed      : {np.linalg.norm(v0):.1f} m/s   v0={np.round(v0, 1)}")
    print(f"    omega             : |w|={np.linalg.norm(omega):.1f} rad/s"
          f" ({np.linalg.norm(omega) / (2 * np.pi):.1f} rev/s)  axis={np.round(omega, 1)}")
    print(f"    apex height       : {pts[:, 2].max():.2f} m")
    print(f"    start->end        : {np.round(pts[0], 1)} -> {np.round(pts[-1], 1)}")
    print(f"    Magnus deflection : {defl:.2f} m over {dt[-1]:.2f} s")
    return pts


def fit_with_fixes(seg, fixes, fps, p0_seed, v0_seed, omega_seed):
    """LM over (p0, v0, omega): pixel reprojection + weighted 3D fix residuals."""
    from scipy.optimize import least_squares

    frames = [f for f, *_ in seg]
    dt = (np.array(frames) - frames[0]) / fps
    g_vec = np.array([0.0, 0.0, G])
    fix_idx = {f: i for i, (f, *_,) in enumerate(seg)}

    def residuals(params):
        p0, v0, omega = params[:3], params[3:6], params[6:9]
        pts = _integrate_magnus_positions(p0, v0, omega, g_vec, DRAG_K, dt)
        out = []
        for (f, uv, K, R, t), p in zip(seg, pts):
            cam = K @ (R @ p + t)
            if cam[2] <= 1e-6:
                out.extend([1e3, 1e3])
                continue
            out.extend([cam[0] / cam[2] - uv[0], cam[1] / cam[2] - uv[1]])
        for f, xyz in fixes:
            if f in fix_idx:
                out.extend(FIX_WEIGHT_PX_PER_M * (pts[fix_idx[f]] - xyz))
        return np.array(out)

    x0 = np.concatenate([p0_seed, v0_seed, omega_seed])
    res = least_squares(residuals, x0, max_nfev=400)
    p0, v0, omega = res.x[:3], res.x[3:6], res.x[6:9]
    # report PIXEL-only residual for comparability
    pix = residuals(res.x)[: 2 * len(seg)].reshape(-1, 2)
    rms = float(np.sqrt((pix**2).sum(axis=1)).mean())
    return p0, v0, omega, rms


def main():
    seg = collect_segment_obs()
    fixes = collect_fixes()
    frames = [f for f, *_ in seg]
    fps = 30.0
    print(f"segment frames {SEG_START}-{SEG_END}: {len(seg)} observations,"
          f" {len(fixes)} triangulated fixes at frames {[f for f, _ in fixes]}")

    obs = [(f, tuple(uv)) for f, uv, *_ in seg]
    Ks = [K for *_, K, R, t in [(f, uv, K, R, t) for f, uv, K, R, t in seg]]
    Ks = [s[2] for s in seg]
    Rs = [s[3] for s in seg]
    ts = [s[4] for s in seg]

    # A: parabola
    p0a, v0a, resa = fit_parabola_to_image_observations(obs, Ks=Ks, Rs=Rs, t_world=ts, fps=fps, g=G)
    describe("A parabola (gravity only)", p0a, v0a, np.zeros(3), resa, fps, frames)

    # B: Magnus, monocular
    p0b, v0b, omb, resb = fit_magnus_trajectory(
        obs, Ks=Ks, Rs=Rs, t_world=ts, fps=fps, g=G, drag_k_over_m=DRAG_K)
    describe("B Magnus ODE, monocular", p0b, v0b, omb, resb, fps, frames)

    # C: Magnus + replay fixes
    p0c, v0c, omc, resc = fit_with_fixes(seg, fixes, fps, p0b, v0b, omb)
    pts_c = describe("C Magnus ODE + replay 3D fixes", p0c, v0c, omc, resc, fps, frames)

    # how far do the fixes pull the trajectory from the monocular solution?
    dtv = (np.array(frames) - frames[0]) / fps
    pts_b = _integrate_magnus_positions(p0b, v0b, omb, np.array([0.0, 0.0, G]), DRAG_K, dtv)
    gap = np.linalg.norm(pts_c - pts_b, axis=1)
    print(f"\n  monocular vs fixed-constrained trajectory gap:"
          f" median {np.median(gap):.2f} m, max {gap.max():.2f} m")
    for f, xyz in fixes:
        i = frames.index(f)
        print(f"    fix at frame {f}: tri={np.round(xyz, 2)}  fitC={np.round(pts_c[i], 2)}"
              f"  |err|={np.linalg.norm(pts_c[i] - xyz):.2f} m")


if __name__ == "__main__":
    main()
