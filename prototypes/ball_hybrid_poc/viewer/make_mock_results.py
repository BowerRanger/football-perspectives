"""Generate plausible mock ``results.json`` fixtures for the ball hybrid-extraction
PoC viewer, so the viewer can be built and demoed before the engine ICs land
real per-clip results.

This script does NOT read any real pipeline output. It is a pure synthetic
generator: a physically-plausible truth arc (gravity + quadratic drag), a
"current" method track that is noisy/jittery and drifts in depth (the axis
least observable from the broadcast camera), and a "hybrid" method track that
stays close to truth. Shapes follow ``prototypes/ball_hybrid_poc/CONTRACT.md``
exactly (the ``Results`` JSON), so ``build_viewer.py`` and the real engine
output are interchangeable inputs to the viewer.

Usage:
    .venv311/bin/python prototypes/ball_hybrid_poc/viewer/make_mock_results.py \
        --out-dir prototypes/ball_hybrid_poc/viewer/mock_data
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0
BALL_RADIUS = 0.11
G = 9.81


# --------------------------------------------------------------------------
# Minimal pinhole camera math (self-contained; no dependency on src/, per the
# PoC contract's "never seed/derive from src.utils.ball_* output" spirit —
# this is throwaway mock geometry, not the pipeline's camera model).
# --------------------------------------------------------------------------

def _normalize(v: list[float]) -> list[float]:
    n = math.sqrt(sum(c * c for c in v))
    if n < 1e-9:
        return [0.0, 0.0, 0.0]
    return [c / n for c in v]


def _sub(a, b):
    return [a[i] - b[i] for i in range(3)]


def _cross(a, b):
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]


def _dot(a, b):
    return sum(a[i] * b[i] for i in range(3))


class SimpleCamera:
    """A fixed pinhole camera used only to compute mock reprojection-error
    metrics (px). Not the pipeline's real camera model."""

    def __init__(self, eye, target, fov_deg, image_size, world_up=(0.0, 0.0, 1.0)):
        self.eye = list(eye)
        forward = _normalize(_sub(target, eye))
        right = _normalize(_cross(forward, list(world_up)))
        true_up = _cross(right, forward)
        self.R = [right, true_up, [-c for c in forward]]
        self.fov_deg = fov_deg
        self.image_size = image_size

    def project(self, xyz):
        rel = _sub(xyz, self.eye)
        cam = [_dot(row, rel) for row in self.R]
        if cam[2] >= -1e-6:
            return None  # behind or at the camera plane
        f = self.image_size[1] / (2 * math.tan(math.radians(self.fov_deg) / 2))
        x = -cam[0] / cam[2] * f + self.image_size[0] / 2
        y = cam[1] / cam[2] * f + self.image_size[1] / 2
        return [x, y]


def _px_error(cam: SimpleCamera, a, b) -> float | None:
    pa, pb = cam.project(a), cam.project(b)
    if pa is None or pb is None:
        return None
    return math.hypot(pa[0] - pb[0], pa[1] - pb[1])


# --------------------------------------------------------------------------
# Truth arc synthesis: gravity + quadratic drag, explicit Euler integration.
# --------------------------------------------------------------------------

def _simulate_arc(
    launch_xyz, launch_vel, fps, drag_cd=0.25, ground_z=0.0, max_frames=260,
    bounce_restitution=None,
):
    """Integrate a lobbed/driven ball under gravity + quadratic drag.
    Returns list of (x, y, z, state) tuples and the frame index of any bounce.
    """
    dt = 1.0 / fps
    rho_cdA_over_m = drag_cd * 0.02  # lumped drag coefficient, tuned for plausible decel
    x, y, z = launch_xyz
    vx, vy, vz = launch_vel
    frames = []
    bounce_frame = None
    airborne = True
    for i in range(max_frames):
        state = "air" if airborne and z > ground_z + 1e-3 else "ground"
        frames.append((x, y, max(z, ground_z), state))
        speed = math.sqrt(vx * vx + vy * vy + vz * vz)
        ax = -rho_cdA_over_m * speed * vx
        ay = -rho_cdA_over_m * speed * vy
        az = -G - rho_cdA_over_m * speed * vz
        vx += ax * dt
        vy += ay * dt
        vz += az * dt
        x += vx * dt
        y += vy * dt
        z += vz * dt
        if z <= ground_z and airborne and vz < 0:
            if bounce_restitution is not None and i > 3:
                bounce_frame = i + 1
                z = ground_z
                vz = -vz * bounce_restitution
                vx *= 0.7
                vy *= 0.7
                if abs(vz) < 0.6:
                    airborne = False
            else:
                z = ground_z
                vz = 0.0
                vx *= 0.15
                vy *= 0.15
                airborne = False
        if not airborne and speed < 0.08:
            # settle: pad remaining frames at rest
            frames.extend([(x, y, ground_z, "ground")] * (max_frames - len(frames)))
            break
    return frames[:max_frames], bounce_frame


def _build_scenario(name: str, rng: random.Random, fps: int, kind: str) -> dict[str, Any]:
    """Build one TruthTrack + current/hybrid Tracks + per-frame errors + metrics."""
    cx = rng.uniform(30, 75)
    cy = rng.uniform(20, 48)

    if kind == "lob_pass":
        launch = (cx - 18, cy - 6, 0.05)
        vel = (14.0, 4.5, 9.0)
        raw, bounce_frame = _simulate_arc(launch, vel, fps, drag_cd=0.22, max_frames=180)
    elif kind == "driven_shot":
        launch = (cx - 22, cy, 0.11)
        vel = (24.0, rng.uniform(-1.5, 1.5), 1.2)
        raw, bounce_frame = _simulate_arc(launch, vel, fps, drag_cd=0.30, max_frames=90)
    else:  # bouncing_clearance
        launch = (cx - 10, cy - 10, 0.05)
        vel = (11.0, 9.0, 11.0)
        raw, bounce_frame = _simulate_arc(
            launch, vel, fps, drag_cd=0.24, max_frames=220, bounce_restitution=0.55
        )

    n = len(raw)
    truth_frames = []
    for i, (x, y, z, state) in enumerate(raw):
        st = state
        if bounce_frame is not None and i == bounce_frame:
            st = "contact"
        truth_frames.append({"frame": i, "xyz": [x, y, z], "state": st})

    events = [{"frame": 0, "kind": "touch", "player_id": "P001", "bone": "right_foot",
               "xyz": list(raw[0][:3])}]
    if bounce_frame is not None:
        events.append({"frame": bounce_frame, "kind": "bounce", "xyz": list(raw[bounce_frame][:3])})
    last_air = n - 1
    for i in range(n - 1, -1, -1):
        if raw[i][3] == "air":
            last_air = i
            break
    if kind == "driven_shot":
        events.append({"frame": min(last_air + 1, n - 1), "kind": "net",
                        "xyz": list(raw[min(last_air + 1, n - 1)][:3])})
    events.append({"frame": n - 1, "kind": "rest", "xyz": list(raw[-1][:3])})

    seed_anchor_frames = sorted({0, n // 3, (2 * n) // 3, n - 1} | (
        {bounce_frame} if bounce_frame is not None else set()
    ))

    truth = {
        "clip_id": None,  # filled by caller
        "scenario": name,
        "fps": fps,
        "frames": truth_frames,
        "events": events,
        "seed_anchor_frames": sorted(seed_anchor_frames),
        "physics": {"drag_cd": 0.25, "magnus": 0.0, "restitution": 0.55 if bounce_frame else None},
    }

    # "current": noisy depth (drifts along the least-observable axis for a
    # touchline-side broadcast camera, here approximated as +/-y drift) plus
    # per-frame jitter on all axes, occasional short gaps.
    depth_bias_phase = rng.uniform(0, math.tau)
    current_frames = []
    hybrid_frames = []
    for i, (x, y, z, state) in enumerate(raw):
        gap = (i % 37 == 18) and state == "air"
        if gap:
            current_frames.append({"frame": i, "xyz": None, "mode": "faithful", "conf": 0.0})
        else:
            depth_drift = 1.8 * math.sin(i / 14.0 + depth_bias_phase) * (1.0 if state == "air" else 0.3)
            jx = rng.gauss(0, 0.05)
            jy = rng.gauss(0, 0.05) + depth_drift
            jz = rng.gauss(0, 0.08) if state == "air" else rng.gauss(0, 0.02)
            cz = max(z + jz, 0.0 if state != "ground" else BALL_RADIUS * rng.uniform(0.3, 1.6))
            current_frames.append({
                "frame": i,
                "xyz": [x + jx, y + jy, cz],
                "mode": "faithful",
                "conf": round(rng.uniform(0.55, 0.95), 3),
            })

        hx = x + rng.gauss(0, 0.015)
        hy = y + rng.gauss(0, 0.015)
        hz = max(z + rng.gauss(0, 0.02), 0.0)
        hmode = "simulated" if state == "air" else ("anchor" if i in seed_anchor_frames else "faithful")
        hybrid_frames.append({
            "frame": i, "xyz": [hx, hy, hz], "mode": hmode,
            "conf": round(rng.uniform(0.75, 0.98), 3),
        })

    tracks = {
        "current": {"clip_id": None, "method": "current", "frames": current_frames},
        "hybrid": {"clip_id": None, "method": "hybrid", "frames": hybrid_frames},
    }

    broadcast_cam = SimpleCamera(
        eye=[cx, -18.0, 14.0], target=[cx, cy, 1.0], fov_deg=42.0, image_size=[1920, 1080],
    )
    side_cam = SimpleCamera(
        eye=[cx, -30.0, 5.0], target=[cx, cy, 1.0], fov_deg=50.0, image_size=[1920, 1080],
    )

    per_frame_err: dict[str, list[float | None]] = {"current": [], "hybrid": []}
    metrics: dict[str, dict[str, Any]] = {}
    for method, mframes in tracks.items():
        errs = []
        bpx_errs = []
        spx_errs = []
        ground_floats = []
        for tf, mf in zip(truth_frames, mframes["frames"]):
            if mf["xyz"] is None:
                errs.append(None)
                continue
            tx = tf["xyz"]
            mx = mf["xyz"]
            e = math.sqrt(sum((tx[k] - mx[k]) ** 2 for k in range(3)))
            errs.append(e)
            bpx = _px_error(broadcast_cam, tx, mx)
            spx = _px_error(side_cam, tx, mx)
            if bpx is not None:
                bpx_errs.append(bpx)
            if spx is not None:
                spx_errs.append(spx)
            if tf["state"] == "ground":
                ground_floats.append(abs(mx[2] - BALL_RADIUS))
        per_frame_err[method] = errs
        valid = sorted(e for e in errs if e is not None)
        contact_gaps = []
        for ev in events:
            if ev["kind"] in ("touch", "bounce", "net", "post"):
                mf = mframes["frames"][ev["frame"]]
                if mf["xyz"] is not None:
                    contact_gaps.append(math.sqrt(sum(
                        (ev["xyz"][k] - mf["xyz"][k]) ** 2 for k in range(3)
                    )))
        metrics[method] = {
            "p50": _percentile(valid, 0.50),
            "p95": _percentile(valid, 0.95),
            "max": max(valid) if valid else None,
            "pct_le_20cm": (sum(1 for e in valid if e <= 0.20) / len(valid)) if valid else None,
            "contact_gap": (sum(contact_gaps) / len(contact_gaps)) if contact_gaps else None,
            "ground_float_sink": (sum(ground_floats) / len(ground_floats)) if ground_floats else None,
            "broadcast_px_error": (sum(bpx_errs) / len(bpx_errs)) if bpx_errs else None,
            "side_px_error": (sum(spx_errs) / len(spx_errs)) if spx_errs else None,
            "naturalness_violations": rng.randint(0, 2) if method == "hybrid" else rng.randint(2, 9),
        }

    return {
        "name": name,
        "truth": truth,
        "tracks": tracks,
        "metrics": metrics,
        "per_frame_err": per_frame_err,
        "broadcast_cam_eye": broadcast_cam.eye,
    }


def _percentile(sorted_vals: list[float], q: float) -> float | None:
    if not sorted_vals:
        return None
    idx = min(len(sorted_vals) - 1, max(0, int(round(q * (len(sorted_vals) - 1)))))
    return sorted_vals[idx]


def build_clip_results(clip_id: str, fps: int, image_size, seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    scenario_kinds = [
        ("lob_pass", "lob_pass"),
        ("driven_shot", "driven_shot"),
        ("bouncing_clearance", "bouncing_clearance"),
    ]

    scenarios: dict[str, Any] = {}
    camera_eyes = []
    for name, kind in scenario_kinds:
        built = _build_scenario(name, rng, fps, kind)
        built["truth"]["clip_id"] = clip_id
        built["tracks"]["current"]["clip_id"] = clip_id
        built["tracks"]["hybrid"]["clip_id"] = clip_id
        scenarios[name] = {
            "truth": built["truth"],
            "tracks": built["tracks"],
            "metrics": built["metrics"],
            "per_frame_err": built["per_frame_err"],
        }
        camera_eyes.append(built["broadcast_cam_eye"])

    # Real-footage section: a longer noisy/hybrid track over real frame
    # numbers with held-out anchors + (for one clip) cross-replay fixes.
    n_real = 300
    real_launch = (20.0, 30.0, 0.08)
    real_vel = (12.0, 6.0, 7.5)
    raw, bounce_frame = _simulate_arc(real_launch, real_vel, fps, drag_cd=0.24, max_frames=n_real)
    real_current = []
    real_hybrid = []
    for i, (x, y, z, state) in enumerate(raw):
        depth_drift = 1.5 * math.sin(i / 16.0) * (1.0 if state == "air" else 0.2)
        real_current.append({
            "frame": i,
            "xyz": [x + rng.gauss(0, 0.05), y + rng.gauss(0, 0.05) + depth_drift,
                    max(z + rng.gauss(0, 0.08), 0.0)],
            "mode": "faithful",
            "conf": round(rng.uniform(0.5, 0.9), 3),
        })
        hmode = "simulated" if state == "air" else "faithful"
        real_hybrid.append({
            "frame": i,
            "xyz": [x + rng.gauss(0, 0.02), y + rng.gauss(0, 0.02), max(z + rng.gauss(0, 0.03), 0.0)],
            "mode": hmode,
            "conf": round(rng.uniform(0.7, 0.97), 3),
        })

    anchors_heldout = [
        {"frame": f, "xyz_gt": list(raw[f][:3])}
        for f in sorted(rng.sample(range(n_real), k=6))
    ]
    fixes = []
    if clip_id == "origi01":
        fixes = [
            {"frame": f, "xyz": [raw[f][0] + rng.gauss(0, 0.03), raw[f][1] + rng.gauss(0, 0.03),
                                  max(raw[f][2] + rng.gauss(0, 0.03), 0.0)]}
            for f in sorted(rng.sample(range(n_real), k=8))
        ]

    def _err_at(frames_track, samples, xyz_key):
        errs = []
        for s in samples:
            mf = frames_track[s["frame"]]
            if mf["xyz"] is None:
                continue
            errs.append(math.sqrt(sum((mf["xyz"][k] - s[xyz_key][k]) ** 2 for k in range(3))))
        return errs

    real_metrics = {}
    for method, mframes in (("current", real_current), ("hybrid", real_hybrid)):
        anchor_errs = _err_at(mframes, anchors_heldout, "xyz_gt")
        fix_errs = _err_at(mframes, fixes, "xyz") if fixes else []
        valid_all = [
            math.sqrt(sum((mframes[i]["xyz"][k] - raw[i][k]) ** 2 for k in range(3)))
            for i in range(n_real) if mframes[i]["xyz"] is not None
        ]
        sorted_all = sorted(valid_all)
        real_metrics[method] = {
            "p50": _percentile(sorted_all, 0.50),
            "p95": _percentile(sorted_all, 0.95),
            "max": max(sorted_all) if sorted_all else None,
            "pct_le_20cm": (sum(1 for e in sorted_all if e <= 0.20) / len(sorted_all)) if sorted_all else None,
            "contact_gap": None,
            "ground_float_sink": None,
            "broadcast_px_error": None,
            "side_px_error": None,
            "naturalness_violations": rng.randint(0, 2) if method == "hybrid" else rng.randint(3, 10),
            "anchor_heldout_err_m": (sum(anchor_errs) / len(anchor_errs)) if anchor_errs else None,
            "fix_err_m": (sum(fix_errs) / len(fix_errs)) if fix_errs else None,
        }

    real = {
        "tracks": {
            "current": {"clip_id": clip_id, "method": "current", "frames": real_current},
            "hybrid": {"clip_id": clip_id, "method": "hybrid", "frames": real_hybrid},
        },
        "metrics": real_metrics,
        "anchors_heldout": anchors_heldout,
        "fixes": fixes,
    }

    max_frames = max(
        [len(s["truth"]["frames"]) for s in scenarios.values()] + [n_real]
    )
    avg_eye = [sum(e[k] for e in camera_eyes) / len(camera_eyes) for k in range(3)]
    camera = {"centre_xyz_per_frame": [list(avg_eye) for _ in range(max_frames)]}

    players = {
        "frames": [
            {
                "frame": i,
                "P001": [avg_eye[0] - 12 + 0.02 * i, avg_eye[1] - 4],
                "P002": [avg_eye[0] - 6 + 0.015 * i, avg_eye[1] + 8],
            }
            for i in range(0, max_frames, 2)
        ]
    }

    return {
        "clip_id": clip_id,
        "fps": fps,
        "image_size": list(image_size),
        "pitch": {"length": PITCH_LENGTH, "width": PITCH_WIDTH},
        "camera": camera,
        "scenarios": scenarios,
        "real": real,
        "players": players,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default="prototypes/ball_hybrid_poc/viewer/mock_data")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    clips = [
        ("gberch", 30, (1920, 1080), 42),
        ("origi01", 25, (1920, 1080), 7),
    ]
    for clip_id, fps, image_size, seed in clips:
        results = build_clip_results(clip_id, fps, image_size, seed)
        clip_dir = out_dir / clip_id
        clip_dir.mkdir(parents=True, exist_ok=True)
        out_path = clip_dir / "results.json"
        out_path.write_text(json.dumps(results, indent=2))
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
