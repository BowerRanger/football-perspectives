"""Derive the ``hidden`` synthetic scenario from ``mismatch``.

Same truth trajectory as ``mismatch`` (every event still happens), but the
methods only receive the fold-0 half of the operator anchors
(``src.utils.ball_eval.split_anchors``, the same stratified split the real
held-out check uses). Events whose anchor was withheld must be discovered
from evidence — the situation real footage is always in, and the one the
``base``/``mismatch``/``sparse`` scenarios never exercise (their truth only
changes behaviour at anchor frames).

Usage:
    python -m prototypes.ball_hybrid_poc.make_hidden_scenario --clips gberch,origi01
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

from src.utils import ball_eval as BE

from .ctx import M
from .run_current import _anchor_from_dict
from .types import SynthRun, TruthTrack, load_json, save_json

OUT_ROOT = Path(M) / "output-ball-poc"
SOURCE = "mismatch"
SCENARIO = "hidden"


def make_hidden(clip_id: str) -> tuple[TruthTrack, SynthRun]:
    clip_dir = OUT_ROOT / clip_id
    truth = TruthTrack.from_json(load_json(clip_dir / f"truth_{SOURCE}.json"))
    run = SynthRun.from_json(load_json(clip_dir / f"synth_obs_{SOURCE}.json"))

    anchors = [_anchor_from_dict(a) for a in run.anchors]
    kept, _held = BE.split_anchors(anchors, fold=0, n_folds=2)
    kept_frames = {a.frame for a in kept}
    kept_dicts = tuple(a for a in run.anchors if int(a["frame"]) in kept_frames)

    hidden_truth = dataclasses.replace(truth, scenario=SCENARIO)
    hidden_run = dataclasses.replace(
        run, scenario=SCENARIO, anchors=kept_dicts,
        noise_model={**run.noise_model, "derived_from": SOURCE,
                     "anchors_kept": len(kept_dicts),
                     "anchors_withheld": len(run.anchors) - len(kept_dicts)},
    )
    save_json(clip_dir / f"truth_{SCENARIO}.json", hidden_truth)
    save_json(clip_dir / f"synth_obs_{SCENARIO}.json", hidden_run)
    return hidden_truth, hidden_run


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clips", default="gberch,origi01,kroupi01,s013")
    args = ap.parse_args()
    for clip in args.clips.split(","):
        _t, run = make_hidden(clip)
        print(f"{clip}: kept {run.noise_model['anchors_kept']} anchors, "
              f"withheld {run.noise_model['anchors_withheld']}")


if __name__ == "__main__":
    main()
