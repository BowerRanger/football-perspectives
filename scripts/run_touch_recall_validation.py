"""Two-config touch-recall validation for one shot (Phase 1, spec
docs/superpowers/specs/2026-07-02-ball-stage-improvement-design.md §4.1).

Runs the ball stage twice on an existing output directory — body-kinematics
touch proposer disabled, then enabled — snapshots the auto-anchor sidecar
after each run under the names ball_touch_recall_report.py documents, and
prints the break-only / proposer-only / union recall table against the
shot's manual anchors (pseudo-ground-truth). Requires the shot's upstream
artifacts (camera track, refined_poses/hmr_world) and the real detector,
so the stage runs need the GPU box.

Usage (gberch example):
    python scripts/run_touch_recall_validation.py \
        --output output-gberch --shot gberch

    # re-print the table from existing snapshots, no GPU needed:
    python scripts/run_touch_recall_validation.py \
        --output output-gberch --shot gberch --report-only
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.ball_touch_recall_report import (  # noqa: E402
    _print_table,
    proposer_only_touches,
    recall_table,
)
from src.utils.ball_touch_recall import touches_from_anchor_set  # noqa: E402


def with_kinematic_toggle(cfg: dict, enabled: bool) -> dict:
    """New deep-copied config with ``ball.kinematic_touch.enabled`` set."""
    out = copy.deepcopy(cfg)
    out.setdefault("ball", {}).setdefault("kinematic_touch", {})["enabled"] = enabled
    return out


def snapshot_auto_anchors(ball_dir: Path, shot_id: str, label: str) -> Path:
    """Copy ``<shot>_ball_anchors_auto.json`` to the labelled snapshot the
    recall report consumes (``..._auto_<label>.json``)."""
    src = ball_dir / f"{shot_id}_ball_anchors_auto.json"
    if not src.exists():
        raise FileNotFoundError(
            f"{src} missing — did the ball stage run produce auto anchors?")
    dst = ball_dir / f"{shot_id}_ball_anchors_auto_{label}.json"
    shutil.copyfile(src, dst)
    return dst


def _run_ball_stage(output_dir: Path, shot_id: str, cfg: dict) -> None:
    # Import inside the run path: the stage pulls detector deps (torch/WASB)
    # that the pure helpers and their tests must not require.
    from src.stages.ball import BallStage

    stage = BallStage(config=cfg, output_dir=output_dir)
    stage.shot_filter = shot_id  # only re-detect/solve the shot under test
    stage.run()


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", required=True, type=Path,
                    help="pipeline output dir (e.g. output-gberch)")
    ap.add_argument("--shot", required=True, help="shot id (e.g. gberch)")
    ap.add_argument("--config", type=Path, default=None,
                    help="optional YAML override merged with defaults")
    ap.add_argument("--report-only", action="store_true",
                    help="skip the stage runs; score existing snapshots")
    args = ap.parse_args()

    ball_dir = args.output / "ball"
    manual_path = ball_dir / f"{args.shot}_ball_anchors.json"
    if not manual_path.exists():
        ap.error(f"no manual anchors at {manual_path} — nothing to score against")

    if not args.report_only:
        from src.pipeline.config import load_config

        cfg = load_config(args.config)
        print("run 1/2: ball stage with kinematic_touch DISABLED (break-only)")
        _run_ball_stage(args.output, args.shot, with_kinematic_toggle(cfg, False))
        snapshot_auto_anchors(ball_dir, args.shot, "break_only")
        print("run 2/2: ball stage with kinematic_touch ENABLED (union)")
        _run_ball_stage(args.output, args.shot, with_kinematic_toggle(cfg, True))
        snapshot_auto_anchors(ball_dir, args.shot, "union")

    manual = touches_from_anchor_set(manual_path)
    break_only = touches_from_anchor_set(
        ball_dir / f"{args.shot}_ball_anchors_auto_break_only.json")
    union = touches_from_anchor_set(
        ball_dir / f"{args.shot}_ball_anchors_auto_union.json")
    proposer_only = proposer_only_touches(break_only, union)
    table = recall_table(manual, break_only, proposer_only, union)
    _print_table(table)
    report_path = ball_dir / f"{args.shot}_touch_recall.json"
    report_path.write_text(json.dumps(table, indent=2))
    print(f"written {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
