"""Capture (or re-capture) a golden-clip baseline for the ball-stage
regression gate (``tests/test_ball_regression.py``).

Builds (once) the frozen ``mismatch``-scenario synthetic truth/evidence
from the clip's CURRENT manual anchors, then re-runs the real ball stage
N times with the shipped ``ball.trajectory`` (``config/default.yaml``;
override with ``--trajectory``) — against that frozen synthetic evidence
AND the real detector's 2-fold held-out anchor evaluation —
to measure run-to-run spread (the gate's nondeterminism allowance; the
ball stage should be near-deterministic once its detection cache is
warm, but this is measured, not assumed). Writes the committed fixture
dir:

    tests/regression/ball/<clip>/
        anchors.json          frozen manual anchor set (pseudo-ground-truth)
        shot.json             shot id/fps + clip video relpath + sha256
        truth_mismatch.json   frozen synthetic 3-D ground truth
        synth_obs_mismatch.json  frozen synthetic detector stream
        baseline.json         aggregated metrics + spread + tolerances

Usage:
    .venv311/bin/python scripts/capture_ball_regression_baseline.py \\
        --clip gberch --runs 3

Re-run after an intentional ball-stage improvement and commit the
baseline.json diff (and, if the anchor set changed, the other fixture
files too) — rebaselining is always visible in review.
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.utils import ball_bench_metrics as BM  # noqa: E402
from src.utils import ball_bench_regression as BRG  # noqa: E402
from src.utils import ball_bench_runner as BR  # noqa: E402
from src.utils import ball_bench_synth as BS  # noqa: E402
from src.utils import ball_bench_truth as BT  # noqa: E402
from src.utils.ball_bench_clip import load_clip  # noqa: E402
from src.utils.ball_bench_types import save_json  # noqa: E402

SCENARIO = "mismatch"
_SYNTH_KEYS = ("p50", "p95", "pct_le_20cm", "ground_float_sink",
               "naturalness_violations_minus_truth")
_REAL_KEYS = ("p50", "p95")


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
    parser.add_argument("--clip", required=True, help="clip id (see "
                        "src.utils.ball_bench_clip.CLIPS)")
    parser.add_argument("--runs", type=int, default=3,
                        help="repetitions for spread measurement")
    parser.add_argument("--fixture-root", type=Path,
                        default=REPO_ROOT / "tests" / "regression" / "ball")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trajectory", default=BR.shipped_trajectory(),
                        help="ball.trajectory to baseline (default: the "
                        "shipped value in config/default.yaml)")
    args = parser.parse_args()
    trajectory = args.trajectory

    clip_ctx = load_clip(args.clip)
    video_relpath = str(
        clip_ctx.video_path.relative_to(clip_ctx.output_dir))
    if not clip_ctx.video_path.exists():
        sys.exit(f"clip video missing: {clip_ctx.video_path}")

    print(f"== {args.clip}: building frozen '{SCENARIO}' truth/evidence "
          f"from {len(clip_ctx.anchors.anchors)} current manual anchors ==")
    truth = BT.build_truth(clip_ctx, SCENARIO, seed=args.seed)
    synth = BS.make_synth_run(clip_ctx, truth, SCENARIO, seed=args.seed)
    side_cam = BM.build_side_camera([f.xyz for f in truth.frames])

    # The clip-level scratch dir (shared with ad hoc run_ball_bench.py
    # runs against this clip) so the det_cache and anchor-fold split are
    # reused/warmed across --runs AND across separate invocations of this
    # script (per CLAUDE.md: budget the wall-clock; confirm cache hits in
    # logs). The det_cache and real_split_foldN.json entries are
    # content/anchor-keyed, so sharing them with other runs against the
    # same clip's current anchors is safe.
    bench_root = BR.BENCH_OUT_ROOT / args.clip
    det_cache = BR.det_cache_path(args.clip, bench_root)
    hit = BR.dry_run_cache_hit_rate(clip_ctx, det_cache)
    print(f"   det_cache pre-flight: {hit['detect_cache_hit_rate']:.1%} "
          f"({hit['detect_cache_hits']}/{hit['total_frames']} frames, "
          f"{hit['cache_detect_entries']} cached entries)")

    synth_runs: list[dict] = []
    real_runs: list[dict] = []
    for i in range(args.runs):
        print(f"-- run {i + 1}/{args.runs} --")
        track = BR.run_synthetic(clip_ctx, synth, SCENARIO, trajectory)
        flat, _detail = BM.compute_scenario_metrics(
            clip_ctx, track, truth, side_camera=side_cam)
        synth_runs.append(flat)
        print(f"   synth: p50={flat['p50']:.4f} p95={flat['p95']:.4f} "
              f"pct_le_20cm={flat['pct_le_20cm']:.3f} "
              f"float_sink={flat['ground_float_sink']:.4f} "
              f"nat_delta={flat['naturalness_violations_minus_truth']}")

        combined_errs: list[float] = []
        for fold in range(BR.N_FOLDS):
            real_track = BR.run_real(clip_ctx, trajectory, fold=fold,
                                     det_cache=det_cache, bench_root=bench_root)
            held = BR.anchor_heldout_error(clip_ctx, real_track, fold,
                                           bench_root=bench_root)
            combined_errs.extend(held["errs"])
        real_flat = {
            "p50": float(np.percentile(combined_errs, 50)) if combined_errs else None,
            "p95": float(np.percentile(combined_errs, 95)) if combined_errs else None,
            "n": len(combined_errs),
        }
        real_runs.append(real_flat)
        print(f"   real: p50={real_flat['p50']} p95={real_flat['p95']} "
              f"n={real_flat['n']}")

    synth_metrics, synth_spread = BRG.aggregate_runs(synth_runs, _SYNTH_KEYS)
    real_metrics, real_spread = BRG.aggregate_runs(real_runs, _REAL_KEYS)
    real_metrics["n"] = min(r["n"] for r in real_runs)

    baseline = {
        "clip_id": args.clip,
        "scenario": SCENARIO,
        "trajectory": trajectory,
        "captured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "runs": args.runs,
        "synth_metrics": synth_metrics,
        "synth_spread": synth_spread,
        "real_metrics": real_metrics,
        "real_spread": real_spread,
        "tolerances": {
            "pct_le_20cm_abs": 0.05,
            "p95_rel": 0.20,
            "ground_float_sink_rel": 0.20,
            "naturalness_violations_abs": 2.0,
            "real_p50_rel": 0.20,
        },
    }

    fixture_dir = args.fixture_root / args.clip
    fixture_dir.mkdir(parents=True, exist_ok=True)
    clip_ctx.anchors.save(fixture_dir / "anchors.json")
    save_json(fixture_dir / "shot.json", {
        "clip_id": args.clip, "shot_id": clip_ctx.shot_id,
        "fps": clip_ctx.fps, "video_relpath": video_relpath,
        "video_sha256": _sha256(clip_ctx.video_path),
    })
    save_json(fixture_dir / "truth_mismatch.json", truth)
    save_json(fixture_dir / "synth_obs_mismatch.json", synth)
    save_json(fixture_dir / "baseline.json", baseline)

    print(f"\nbaseline written to {fixture_dir}")
    print(f"  synth p50={synth_metrics['p50']:.4f}m "
          f"(spread {synth_spread['p50']:.4f}) "
          f"pct_le_20cm={synth_metrics['pct_le_20cm']:.3f} "
          f"(spread {synth_spread['pct_le_20cm']:.3f})")
    print(f"  real  p50={real_metrics['p50']}m "
          f"(spread {real_spread['p50']}) n={real_metrics['n']}")


if __name__ == "__main__":
    main()
