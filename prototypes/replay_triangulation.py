"""Prototype: cross-replay ball triangulation feasibility.

Treats two synced shots of the same highlight (origi01 live + origi02 replay)
as an asynchronous two-view rig. For every frame pair linked by the sync map,
triangulates the WASB ball detections from both views using the solved
per-frame cameras, and reports:

  - parallax angle between the two rays (conditioning)
  - closest-approach distance between rays (sync + detection + calib error)
  - triangulated 3D point vs the monocular solved track

Also scans integer offsets around the saved sync offset to test whether the
ball itself can refine the alignment (VisualSync-lite).

Usage:
    python prototypes/replay_triangulation.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent.parent / "output-origi"
REF_SHOT = "origi01"
REP_SHOT = "origi02"
SYNC_OFFSET = -142  # origi02 frame f  <->  origi01 frame f - offset
MIN_CONF = 0.3


def load_cameras(shot: str) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    track = json.loads((OUT / "camera" / f"{shot}_camera_track.json").read_text())
    cams = {}
    for fr in track["frames"]:
        K = np.array(fr["K"])
        R = np.array(fr["R"])
        t = np.array(fr["t"])
        cams[int(fr["frame"])] = (K, R, t)
    return cams


def load_obs(shot: str) -> dict[int, tuple[np.ndarray, float]]:
    data = json.loads((OUT / "ball" / f"{shot}_ball_observations.json").read_text())
    obs = {}
    for fr in data["frames"]:
        if fr.get("uv") and not fr.get("gap_fill"):
            obs[int(fr["frame"])] = (np.array(fr["uv"], dtype=float), float(fr.get("confidence", 0.0)))
    return obs


def load_track(shot: str) -> dict[int, np.ndarray]:
    data = json.loads((OUT / "ball" / f"{shot}_ball_track.json").read_text())
    out = {}
    for fr in data["frames"]:
        if fr.get("world_xyz"):
            out[int(fr["frame"])] = np.array(fr["world_xyz"], dtype=float)
    return out


def pixel_ray(K: np.ndarray, R: np.ndarray, t: np.ndarray, uv: np.ndarray):
    """World-space camera centre and unit ray direction for a pixel (OpenCV x_cam = R X + t)."""
    centre = -R.T @ t
    d_cam = np.linalg.inv(K) @ np.array([uv[0], uv[1], 1.0])
    d_world = R.T @ d_cam
    return centre, d_world / np.linalg.norm(d_world)


def triangulate(c1, d1, c2, d2):
    """Midpoint of the common perpendicular between two rays.

    Returns (point, miss_distance, parallax_deg, depth1, depth2).
    """
    cos = float(np.dot(d1, d2))
    parallax = float(np.degrees(np.arccos(np.clip(abs(cos), -1.0, 1.0))))
    # Solve [d1 -d2][s;u] = c2 - c1 in least squares
    A = np.stack([d1, -d2], axis=1)
    b = c2 - c1
    (s, u), *_ = np.linalg.lstsq(A, b, rcond=None)
    p1 = c1 + s * d1
    p2 = c2 + u * d2
    miss = float(np.linalg.norm(p1 - p2))
    return (p1 + p2) / 2.0, miss, parallax, float(s), float(u)


def interp_obs(obs, f: float, min_conf=MIN_CONF, max_span=3):
    """Observation at (possibly fractional) frame f, linearly interpolated
    between the nearest detections no more than max_span frames apart."""
    lo, hi = int(np.floor(f)), int(np.ceil(f))
    if lo == hi:
        rec = obs.get(lo)
        return rec[0] if rec and rec[1] >= min_conf else None
    a = next(((g, obs[g]) for g in range(lo, lo - max_span, -1) if g in obs), None)
    b = next(((g, obs[g]) for g in range(hi, hi + max_span) if g in obs), None)
    if not a or not b or a[1][1] < min_conf or b[1][1] < min_conf or b[0] - a[0] > max_span:
        return None
    w = (f - a[0]) / (b[0] - a[0])
    return (1 - w) * a[1][0] + w * b[1][0]


def pair_frames(offset: float, obs_ref, obs_rep, cams_ref, cams_rep, min_conf=MIN_CONF):
    pairs = []
    for f_rep, (uv_rep, conf_rep) in obs_rep.items():
        if conf_rep < min_conf or f_rep not in cams_rep:
            continue
        f_ref = f_rep - offset
        f_ref_int = int(round(f_ref))
        if f_ref_int not in cams_ref:
            continue
        uv_ref = interp_obs(obs_ref, f_ref, min_conf)
        if uv_ref is None:
            continue
        pairs.append((f_ref_int, f_rep, uv_ref, uv_rep))
    return pairs


def evaluate(offset: int, obs_ref, obs_rep, cams_ref, cams_rep, verbose=False, track_ref=None):
    pairs = pair_frames(offset, obs_ref, obs_rep, cams_ref, cams_rep)
    rows = []
    for f_ref, f_rep, uv_ref, uv_rep in pairs:
        c1, d1 = pixel_ray(*cams_ref[f_ref], uv_ref)
        c2, d2 = pixel_ray(*cams_rep[f_rep], uv_rep)
        point, miss, parallax, s, u = triangulate(c1, d1, c2, d2)
        if s <= 0 or u <= 0:  # behind a camera: bad pair
            continue
        rows.append({
            "f_ref": f_ref, "f_rep": f_rep, "xyz": point,
            "miss_m": miss, "parallax_deg": parallax,
        })
    if not rows:
        return None, []
    miss = np.array([r["miss_m"] for r in rows])
    par = np.array([r["parallax_deg"] for r in rows])
    summary = {
        "offset": offset,
        "n_pairs": len(rows),
        "median_miss_m": float(np.median(miss)),
        "p90_miss_m": float(np.percentile(miss, 90)),
        "median_parallax_deg": float(np.median(par)),
        "min_parallax_deg": float(par.min()),
        "max_parallax_deg": float(par.max()),
    }
    if verbose and track_ref is not None:
        deltas, heights = [], []
        for r in rows:
            if r["f_ref"] in track_ref:
                deltas.append(np.linalg.norm(r["xyz"] - track_ref[r["f_ref"]]))
                heights.append((r["f_ref"], r["xyz"][2], track_ref[r["f_ref"]][2]))
        if deltas:
            summary["n_vs_mono"] = len(deltas)
            summary["median_delta_vs_mono_m"] = float(np.median(deltas))
            summary["p90_delta_vs_mono_m"] = float(np.percentile(deltas, 90))
        if verbose:
            print("\n  frame-by-frame (every 5th pair): f_ref  z_tri  z_mono  miss_m  parallax")
            for f_ref, z_tri, z_mono in heights[::5]:
                row = next(r for r in rows if r["f_ref"] == f_ref)
                print(f"    {f_ref:4d}  {z_tri:6.2f}  {z_mono:6.2f}  {row['miss_m']:6.2f}  {row['parallax_deg']:6.1f}")
    return summary, rows


def load_flight_segments(shot: str):
    data = json.loads((OUT / "ball" / f"{shot}_ball_track.json").read_text())
    return [(s["id"], s["frame_range"]) for s in data.get("flight_segments", [])]


def main():
    cams_ref = load_cameras(REF_SHOT)
    cams_rep = load_cameras(REP_SHOT)
    obs_ref = load_obs(REF_SHOT)
    obs_rep = load_obs(REP_SHOT)
    track_ref = load_track(REF_SHOT)

    print(f"{REF_SHOT}: {len(obs_ref)} detections, {len(cams_ref)} cams")
    print(f"{REP_SHOT}: {len(obs_rep)} detections, {len(cams_rep)} cams")

    # camera-centre separation: how far apart are the two rigs?
    c_ref = -cams_ref[0][1].T @ cams_ref[0][2]
    c_rep = -cams_rep[0][1].T @ cams_rep[0][2]
    print(f"camera centres: {REF_SHOT}={np.round(c_ref, 1)}  {REP_SHOT}={np.round(c_rep, 1)}"
          f"  separation={np.linalg.norm(c_ref - c_rep):.1f} m")

    print("\n=== sub-frame offset scan (VisualSync-lite, median ray miss) ===")
    best = (None, np.inf)
    for off in np.arange(SYNC_OFFSET - 4, SYNC_OFFSET + 4.01, 0.25):
        s, _ = evaluate(float(off), obs_ref, obs_rep, cams_ref, cams_rep)
        if s and s["n_pairs"] >= 8 and s["median_miss_m"] < best[1]:
            best = (float(off), s["median_miss_m"])
        if s and abs(off - round(off)) < 1e-9:
            marker = " <-- saved" if int(round(off)) == SYNC_OFFSET else ""
            print(f"  offset {off:8.2f}: n={s['n_pairs']:3d}  median_miss={s['median_miss_m']:.3f} m"
                  f"  p90={s['p90_miss_m']:.3f} m{marker}")
    print(f"  best sub-frame offset: {best[0]:.2f} (median miss {best[1]:.3f} m)")

    print(f"\n=== triangulation at refined offset {best[0]:.2f} ===")
    summary, rows = evaluate(best[0], obs_ref, obs_rep, cams_ref, cams_rep,
                             verbose=True, track_ref=track_ref)
    print(json.dumps(summary, indent=2))

    inliers = [r for r in rows if r["miss_m"] < 1.0]
    print(f"\n=== inliers (ray miss < 1 m): {len(inliers)}/{len(rows)} ===")
    deltas = [float(np.linalg.norm(r["xyz"] - track_ref[r["f_ref"]]))
              for r in inliers if r["f_ref"] in track_ref]
    if deltas:
        print(f"  vs monocular solve: median {np.median(deltas):.2f} m,"
              f" p90 {np.percentile(deltas, 90):.2f} m (n={len(deltas)})")
    print("  f_ref   x      y      z   | z_mono  miss_m  parallax")
    for r in inliers:
        zm = track_ref.get(r["f_ref"], [np.nan] * 3)[2]
        x, y, z = r["xyz"]
        print(f"  {r['f_ref']:5d} {x:6.1f} {y:6.1f} {z:6.2f} | {zm:6.2f}  {r['miss_m']:6.2f}  {r['parallax_deg']:6.1f}")

    print("\n=== coverage per origi01 flight segment ===")
    for seg_id, (f0, f1) in load_flight_segments(REF_SHOT):
        fixes = [r for r in inliers if f0 <= r["f_ref"] <= f1]
        print(f"  segment {seg_id} frames {f0}-{f1}: {len(fixes)} triangulated fixes")


if __name__ == "__main__":
    main()
