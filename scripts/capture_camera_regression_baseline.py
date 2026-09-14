"""Capture (or re-capture) a golden-clip baseline for the camera
regression gate (tests/test_camera_regression.py).

Re-solves the camera stage N times from scratch (PnLCalib on MPS is
nondeterministic — the across-run spread becomes the gate's noise
allowance), scores each run against the shot's manual anchors, and
writes the committed fixture dir:

    tests/regression/camera/<shot>/
        anchors.json    frozen manual anchor set (pseudo-ground-truth)
        shot.json       manifest entry + fps for rebuilding a solve dir
        baseline.json   aggregated metrics + spread + tolerances

Usage:
  .venv311/bin/python scripts/capture_camera_regression_baseline.py \
      --output output/ --shot gberch --runs 3

Re-run after an intentional camera improvement and commit the
baseline.json diff — rebaselining is always visible in review.
"""

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.pipeline.config import load_config  # noqa: E402
from src.pipeline.runner import run_pipeline  # noqa: E402
from src.utils.anchor_click_eval import score_track  # noqa: E402
from src.utils.camera_regression import (  # noqa: E402
    aggregate_runs,
    build_solve_dir,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT,
            capture_output=True, text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True,
                        help="source output dir (e.g. output/)")
    parser.add_argument("--shot", required=True,
                        help="shot id from the output dir's manifest")
    parser.add_argument("--runs", type=int, default=3,
                        help="solve repetitions for spread measurement")
    parser.add_argument("--fixture-root", type=Path,
                        default=REPO_ROOT / "tests" / "regression" / "camera")
    args = parser.parse_args()

    manifest = json.loads(
        (args.output / "shots" / "shots_manifest.json").read_text())
    entries = [s for s in manifest["shots"] if s["id"] == args.shot]
    if not entries:
        sys.exit(f"shot {args.shot!r} not in {args.output}/shots manifest")
    shot_fixture = {"fps": manifest["fps"], "shot": entries[0]}

    anchors_path = args.output / "camera" / f"{args.shot}_anchors.json"
    if not anchors_path.exists():
        sys.exit(f"no manual anchors at {anchors_path}")
    anchors = json.loads(anchors_path.read_text())

    clip_path = (args.output / entries[0]["clip_file"]).resolve()
    if not clip_path.exists():
        sys.exit(f"clip missing: {clip_path}")

    config = load_config()
    run_metrics = []
    for i in range(args.runs):
        with tempfile.TemporaryDirectory(prefix="cam_regression_") as tmp:
            work = Path(tmp)
            build_solve_dir(work, shot_fixture, anchors, clip_path)
            run_pipeline(output_dir=work, stages="camera",
                         from_stage=None, config=config)
            track_path = work / "camera" / f"{args.shot}_camera_track.json"
            if not track_path.exists():
                sys.exit(f"run {i + 1}: camera stage produced no track")
            metrics = score_track(anchors,
                                  json.loads(track_path.read_text()))
        if metrics["med_px"] is None:
            sys.exit(f"run {i + 1}: no anchor clicks were scorable")
        print(f"run {i + 1}/{args.runs}: med {metrics['med_px']:.2f}px  "
              f"p90 {metrics['p90_px']:.2f}px  "
              f"covered {metrics['anchor_frames_covered']}"
              f"/{metrics['anchor_frames_total']}  "
              f"conf {metrics['mean_confidence']:.3f}")
        run_metrics.append(metrics)

    metrics, spread = aggregate_runs(run_metrics)
    baseline = {
        "clip_id": args.shot,
        "clip_file": str(clip_path.relative_to(REPO_ROOT)),
        "clip_sha256": _sha256(clip_path),
        "captured_at": datetime.now(timezone.utc).isoformat(
            timespec="seconds"),
        "git_commit": _git_commit(),
        "runs": args.runs,
        "metrics": metrics,
        "spread": spread,
        "tolerances": {"med_rel": 0.15, "p90_rel": 0.20,
                       "confidence_abs": 0.05},
    }

    fixture_dir = args.fixture_root / args.shot
    fixture_dir.mkdir(parents=True, exist_ok=True)
    (fixture_dir / "anchors.json").write_text(json.dumps(anchors, indent=1))
    (fixture_dir / "shot.json").write_text(
        json.dumps(shot_fixture, indent=1))
    (fixture_dir / "baseline.json").write_text(
        json.dumps(baseline, indent=1))
    print(f"\nbaseline written to {fixture_dir}")
    print(f"  med {metrics['med_px']:.2f}px (spread "
          f"{spread['med_px']:.2f})  p90 {metrics['p90_px']:.2f}px "
          f"(spread {spread['p90_px']:.2f})")


if __name__ == "__main__":
    main()
