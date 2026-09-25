"""CLI: run the ball-stage regression bench and assemble per-clip results.

Runs the real, unmodified ``BallStage`` against synthetic-truth scenarios
(``base``/``mismatch``/``sparse``/``hidden``) and, optionally, the real
detector's 2-fold held-out anchor evaluation, for one or more
``ball.trajectory`` values (``reference`` today; ``hybrid`` once IC-A's
``ball.trajectory: reference|hybrid`` config switch lands — see
``src/utils/ball_bench_runner.py``'s ``apply_trajectory``).

Truth/synthetic-evidence artifacts and the real-detector cache are cached
per clip (``$M/output-ball-poc/<clip>/{truth,synth_obs}_<scenario>.json``,
``det_cache.json``) and reused across runs; per-run outputs (dense tracks,
metrics, ``results.json``, ``summary.md``) land under
``$M/output-ball-poc/<clip>/runs/<tag>/`` so different ``--trajectory``/
``--set`` combinations never clobber each other.

Usage:
    .venv311/bin/python scripts/run_ball_bench.py \\
        --clips gberch,origi01,kroupi01,s013 \\
        --scenarios base,mismatch,sparse,hidden \\
        --trajectory reference --folds 2 --tag smoke

    .venv311/bin/python scripts/run_ball_bench.py \\
        --clips gberch --scenarios mismatch --trajectory reference,hybrid \\
        --folds 0 --set ball.physics.roll_friction=0.5 --tag ablation1
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import ball_bench_metrics as BM  # noqa: E402
from src.utils import ball_bench_runner as BR  # noqa: E402
from src.utils import ball_bench_synth as BS  # noqa: E402
from src.utils import ball_bench_truth as BT  # noqa: E402
from src.utils.ball_bench_clip import CLIPS, ClipContext, load_clip  # noqa: E402
from src.utils.ball_bench_types import (  # noqa: E402
    SynthRun,
    TruthTrack,
    load_json,
    save_json,
    validate_results,
)

logger = logging.getLogger(__name__)

DEFAULT_CLIPS = tuple(CLIPS)
DEFAULT_SCENARIOS = ("base", "mismatch", "sparse", "hidden")
BASE_SCENARIOS = ("base", "mismatch", "sparse")


# ---------------------------------------------------------------------------
# truth / synth cache (per-clip, independent of --trajectory / --set / --tag)
# ---------------------------------------------------------------------------

def _load_or_build_truth(clip_ctx: ClipContext, scenario: str,
                          clip_dir: Path, seed: int) -> TruthTrack:
    if scenario == BS.HIDDEN_SCENARIO:
        mismatch_truth = _load_or_build_truth(
            clip_ctx, BS.SOURCE_SCENARIO, clip_dir, seed)
        mismatch_synth = _load_or_build_synth(
            clip_ctx, mismatch_truth, BS.SOURCE_SCENARIO, clip_dir, seed)
        hidden_truth, hidden_synth = BS.derive_hidden_scenario(
            mismatch_truth, mismatch_synth)
        save_json(clip_dir / f"truth_{BS.HIDDEN_SCENARIO}.json", hidden_truth)
        save_json(clip_dir / f"synth_obs_{BS.HIDDEN_SCENARIO}.json", hidden_synth)
        return hidden_truth
    path = clip_dir / f"truth_{scenario}.json"
    if path.exists():
        return TruthTrack.from_json(load_json(path))
    truth = BT.build_truth(clip_ctx, scenario, seed=seed)
    save_json(path, truth)
    return truth


def _load_or_build_synth(clip_ctx: ClipContext, truth: TruthTrack,
                          scenario: str, clip_dir: Path, seed: int) -> SynthRun:
    path = clip_dir / f"synth_obs_{scenario}.json"
    if path.exists():
        return SynthRun.from_json(load_json(path))
    if scenario == BS.HIDDEN_SCENARIO:
        # _load_or_build_truth already wrote both files for "hidden".
        return SynthRun.from_json(load_json(path))
    synth = BS.make_synth_run(clip_ctx, truth, scenario, seed=seed)
    save_json(path, synth)
    return synth


# ---------------------------------------------------------------------------
# scenario processing
# ---------------------------------------------------------------------------

def process_scenario(clip_ctx: ClipContext, clip_dir: Path, run_dir: Path,
                      scenario: str, trajectories: list[str],
                      overrides: list[str], seed: int) -> dict[str, Any]:
    truth = _load_or_build_truth(clip_ctx, scenario, clip_dir, seed)
    synth = _load_or_build_synth(clip_ctx, truth, scenario, clip_dir, seed)
    side_cam = BM.build_side_camera([f.xyz for f in truth.frames])

    tracks: dict[str, Any] = {}
    flat_metrics: dict[str, Any] = {}
    detail_metrics: dict[str, Any] = {}
    per_frame_err: dict[str, Any] = {}

    for trajectory in trajectories:
        t0 = time.time()
        track = BR.run_synthetic(clip_ctx, synth, scenario, trajectory,
                                  overrides=overrides)
        elapsed = time.time() - t0
        save_json(run_dir / f"track_{trajectory}_{scenario}.json", track)
        flat, detail = BM.compute_scenario_metrics(
            clip_ctx, track, truth, side_camera=side_cam)
        tracks[trajectory] = track.to_json()
        flat_metrics[trajectory] = flat
        detail_metrics[trajectory] = detail
        per_frame_err[trajectory] = BM.per_frame_error(track, truth)
        logger.info("%s/%s/%s: p50=%.3fm pct_le_20cm=%s (%.1fs)",
                     clip_ctx.clip_id, scenario, trajectory,
                     flat["p50"] or -1.0, flat["pct_le_20cm"], elapsed)

    return {
        "truth": truth.to_json(),
        "tracks": tracks,
        "metrics": flat_metrics,
        "metrics_detail": detail_metrics,
        "per_frame_err": per_frame_err,
        "side_camera": side_cam.to_json(),
    }


# ---------------------------------------------------------------------------
# real-footage held-out evaluation
# ---------------------------------------------------------------------------

def process_real(clip_ctx: ClipContext, clip_dir: Path, run_dir: Path,
                  trajectories: list[str], folds: int,
                  overrides: list[str]) -> dict[str, Any]:
    if folds == 0:
        return {"tracks": {}, "metrics": {},
                "_status": "skipped (--folds 0)"}

    det_cache = BR.det_cache_path(clip_ctx.clip_id, BR.BENCH_OUT_ROOT)
    hit_rate = BR.dry_run_cache_hit_rate(clip_ctx, det_cache)
    logger.info("%s: det_cache pre-flight hit rate %.1f%% (%d/%d frames, "
                "%d cached detect entries)", clip_ctx.clip_id,
                100.0 * hit_rate["detect_cache_hit_rate"],
                hit_rate["detect_cache_hits"], hit_rate["total_frames"],
                hit_rate["cache_detect_entries"])

    tracks: dict[str, Any] = {}
    metrics_flat: dict[str, Any] = {}
    fold_detail: dict[str, Any] = {}

    for trajectory in trajectories:
        combined_errs: list[float] = []
        per_fold: dict[str, Any] = {}
        for fold in range(BR.N_FOLDS):
            t0 = time.time()
            track = BR.run_real(clip_ctx, trajectory, fold=fold,
                                 overrides=overrides, det_cache=det_cache,
                                 bench_root=BR.BENCH_OUT_ROOT)
            elapsed = time.time() - t0
            save_json(run_dir / f"track_{trajectory}_real_fold{fold}.json",
                      track)
            held = BR.anchor_heldout_error(clip_ctx, track, fold,
                                            bench_root=BR.BENCH_OUT_ROOT)
            combined_errs.extend(held["errs"])
            per_fold[f"fold{fold}"] = {
                "n": held["n"], "p50": held["p50"], "p95": held["p95"],
                "elapsed_s": elapsed,
            }
            logger.info("%s/%s/real/fold%d: n=%d p50=%s (%.1fs)",
                         clip_ctx.clip_id, trajectory, fold, held["n"],
                         held["p50"], elapsed)
            if fold == 0:
                tracks[trajectory] = track.to_json()

        p50 = float(np.percentile(combined_errs, 50)) if combined_errs else None
        p95 = float(np.percentile(combined_errs, 95)) if combined_errs else None
        metrics_flat[trajectory] = {
            "anchor_heldout_err_m": p50,
            "anchor_heldout_err_p95_m": p95,
            "anchor_heldout_n": len(combined_errs),
        }
        fold_detail[trajectory] = per_fold

    return {
        "tracks": tracks,
        "metrics": metrics_flat,
        "_fold_detail": fold_detail,
        "det_cache_preflight": hit_rate,
    }


# ---------------------------------------------------------------------------
# per-clip assembly
# ---------------------------------------------------------------------------

def process_clip(clip_id: str, scenario_names: list[str],
                  trajectories: list[str], folds: int, overrides: list[str],
                  tag: str, seed: int) -> dict[str, Any]:
    clip_ctx = load_clip(clip_id)
    clip_dir = BR.BENCH_OUT_ROOT / clip_id
    clip_dir.mkdir(parents=True, exist_ok=True)
    run_dir = clip_dir / "runs" / tag
    run_dir.mkdir(parents=True, exist_ok=True)

    scenarios = {
        name: process_scenario(clip_ctx, clip_dir, run_dir, name,
                                trajectories, overrides, seed)
        for name in scenario_names
    }
    real = process_real(clip_ctx, clip_dir, run_dir, trajectories, folds,
                         overrides)

    results = {
        "clip_id": clip_id,
        "fps": clip_ctx.fps,
        "image_size": list(clip_ctx.image_size),
        "pitch": {"length": 105.0, "width": 68.0},
        "camera": {"centre_xyz_per_frame": clip_ctx.camera_centres()},
        "scenarios": scenarios,
        "real": real,
        "run": {"tag": tag, "trajectories": trajectories, "folds": folds,
                 "overrides": overrides, "scenario_names": scenario_names,
                 "seed": seed},
    }
    problems = validate_results(results)
    if problems:
        results["_validate_results"] = problems
    save_json(run_dir / "results.json", results)
    return results


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------

def _fmt(v, nd=3) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def build_summary_md(clip_id: str, results: dict) -> str:
    lines = [f"# Ball bench — {clip_id}", "",
             f"tag={results['run']['tag']} "
             f"trajectories={results['run']['trajectories']} "
             f"folds={results['run']['folds']}", "",
             "## Synthetic (scenario x trajectory)", "",
             "| scenario | trajectory | p50 (m) | p95 (m) | %<=20cm | "
             "contact gap (m) | float/sink (m) | nat. viol. Δ |",
             "|---|---|---|---|---|---|---|---|"]
    for scen_name, sc in results.get("scenarios", {}).items():
        for method, mm in sc.get("metrics", {}).items():
            lines.append(
                f"| {scen_name} | {method} | {_fmt(mm.get('p50'))} | "
                f"{_fmt(mm.get('p95'))} | {_fmt(mm.get('pct_le_20cm'))} | "
                f"{_fmt(mm.get('contact_gap'))} | "
                f"{_fmt(mm.get('ground_float_sink'))} | "
                f"{_fmt(mm.get('naturalness_violations_minus_truth'), 0)} |")

    lines += ["", "## Real footage (2-fold held-out anchor error)", "",
              "| trajectory | held-out p50 (m) | held-out p95 (m) | n |",
              "|---|---|---|---|"]
    for method, mm in results.get("real", {}).get("metrics", {}).items():
        lines.append(
            f"| {method} | {_fmt(mm.get('anchor_heldout_err_m'))} | "
            f"{_fmt(mm.get('anchor_heldout_err_p95_m'))} | "
            f"{mm.get('anchor_heldout_n', 0)} |")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clips", default=",".join(DEFAULT_CLIPS))
    ap.add_argument("--scenarios", default=",".join(DEFAULT_SCENARIOS))
    ap.add_argument("--trajectory", default=BR.DEFAULT_TRAJECTORY,
                     help="comma-separated ball.trajectory values, e.g. "
                          "'reference' or 'reference,hybrid'")
    ap.add_argument("--folds", type=int, choices=(0, 2), default=2,
                     help="0 = skip the real-footage held-out evaluation; "
                          "2 = run both folds of the 2-fold split")
    ap.add_argument("--set", dest="overrides", action="append", default=[],
                     help="config override 'dotted.key=value' (repeatable)")
    ap.add_argument("--tag", required=True, help="run tag (output subdir)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    clips = [c.strip() for c in args.clips.split(",") if c.strip()]
    scenarios = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    trajectories = [t.strip() for t in args.trajectory.split(",") if t.strip()]

    for clip_id in clips:
        print(f"=== {clip_id} (tag={args.tag}) ===")
        results = process_clip(clip_id, scenarios, trajectories, args.folds,
                                args.overrides, args.tag, args.seed)
        summary_md = build_summary_md(clip_id, results)
        run_dir = BR.BENCH_OUT_ROOT / clip_id / "runs" / args.tag
        (run_dir / "summary.md").write_text(summary_md)
        print(summary_md)
        print(f"wrote {run_dir / 'results.json'} and {run_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
