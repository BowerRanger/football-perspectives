"""Re-run the reworked ball stage on the real clips and compare against
the committed (pre-rework) tracks. Writes new artifacts to a temp dir —
the existing outputs are never touched.

Usage: .venv/bin/python scripts/validate_ball_rework.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.schemas.ball_track import BallTrack  # noqa: E402
from src.stages.ball import BallStage, _build_detector  # noqa: E402

CLIPS = [
    ("output-kroupi", "kroupi01"),
    ("output-origi", "origi01"),
    ("output-origi", "origi02"),
]


def track_metrics(track: BallTrack) -> dict:
    states = [f.state for f in track.frames]
    jumps = []
    prev = None
    for f in track.frames:
        w = f.world_xyz
        if w is not None and prev is not None and f.frame == prev[0] + 1:
            jumps.append(float(np.linalg.norm(np.array(w) - np.array(prev[1]))))
        prev = (f.frame, w) if w is not None else None
    residuals = [s.fit_residual_px for s in track.flight_segments]
    return {
        "frames": len(track.frames),
        "missing": states.count("missing"),
        "flight": states.count("flight"),
        "grounded": states.count("grounded"),
        "segments": len(track.flight_segments),
        "max_seg_residual_px": round(max(residuals), 1) if residuals else None,
        "mean_seg_residual_px": (
            round(float(np.mean(residuals)), 1) if residuals else None
        ),
        "max_jump_m": round(max(jumps), 2) if jumps else None,
        "jumps_over_2m": sum(1 for j in jumps if j > 2.0),
    }


def main() -> None:
    cfg = yaml.safe_load((ROOT / "config" / "default.yaml").read_text())
    detector = _build_detector(cfg["ball"])
    tmp = Path(tempfile.mkdtemp(prefix="ball_rework_"))
    print(f"new artifacts under: {tmp}", flush=True)
    summary = {}
    for out_dir, shot_id in CLIPS:
        out = ROOT / out_dir
        cam_p = out / "camera" / f"{shot_id}_camera_track.json"
        clip = out / "shots" / f"{shot_id}.mp4"
        old_p = out / "ball" / f"{shot_id}_ball_track.json"
        if not (cam_p.exists() and clip.exists()):
            print(f"skip {shot_id}: missing inputs", flush=True)
            continue
        stage = BallStage(config=cfg, output_dir=out, ball_detector=detector)
        new_p = tmp / f"{shot_id}_ball_track.json"
        print(f"running {shot_id} ...", flush=True)
        stage._run_shot(shot_id, clip, cam_p, new_p, cfg["ball"], detector)
        entry = {"new": track_metrics(BallTrack.load(new_p))}
        if old_p.exists():
            entry["old"] = track_metrics(BallTrack.load(old_p))
        diag = json.loads(
            new_p.with_name(f"{shot_id}_ball_diag.json").read_text()
        )
        entry["new"]["anchors"] = diag.get("anchors", {})
        entry["new"]["underconstrained"] = len(
            diag.get("underconstrained_spans", [])
        )
        entry["new"]["flagged_bounces"] = sum(
            1 for b in diag.get("bounces", []) if b.get("flagged")
        )
        gaps = [g["gap_m"] for g in diag.get("contact_gaps", [])]
        entry["new"]["max_contact_gap_m"] = (
            round(max(gaps), 2) if gaps else None
        )
        summary[shot_id] = entry
        print(json.dumps({shot_id: entry}, indent=1), flush=True)
    (tmp / "summary.json").write_text(json.dumps(summary, indent=2))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
