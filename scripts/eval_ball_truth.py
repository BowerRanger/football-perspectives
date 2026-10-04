"""Score a pipeline ball track against Ball Studio truth.

Usage:
  .venv311/bin/python scripts/eval_ball_truth.py --output OUT --group G \
      [--shot SHOT] [--json report.json]

Compares ``OUT/ball/<shot>_ball_track.json`` (mapped onto the group's
reference timeline) with ``OUT/ball_truth/<G>_ball_truth_dense.json``.
Without ``--shot`` every group member that has a ball track is scored.
Metrics: src/utils/ball_truth_eval.py docstring.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.schemas import ball_truth as bt  # noqa: E402
from src.schemas.ball_track import BallTrack  # noqa: E402
from src.utils import ball_truth_eval as ev  # noqa: E402
from src.web.ball_studio import StudioData  # noqa: E402


def _pipeline_events(out: Path, shot: str, offset: int) -> list[dict]:
    rows = []
    for suffix in ("_ball_anchors_auto.json", "_ball_anchors.json"):
        p = out / "ball" / f"{shot}{suffix}"
        if not p.exists():
            continue
        for a in json.loads(p.read_text()).get("anchors", []):
            rows.append({"frame": int(a["frame"]) - offset, "state": a.get("state", "")})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--group", required=True)
    ap.add_argument("--shot", default=None)
    ap.add_argument("--json", type=Path, default=None, help="write the report here")
    args = ap.parse_args()

    out = args.output
    dense = bt.load_dense(out, args.group)
    if dense is None:
        print(f"no dense truth at {bt.dense_path(out, args.group)} - save the group in Ball Studio first",
              file=sys.stderr)
        return 2
    data = StudioData(out, None)
    g = data.group(args.group)
    ctx = data.context(g)
    cameras = {sid: (lambda sf, sid=sid: ctx.camera(sid, sf)) for sid, _, _ in g.members}
    offsets = {sid: off for sid, off, _ in g.members}

    report: dict = {"group": args.group, "outcome": dense.get("outcome"), "shots": {}}
    for sid, off, _ in g.members:
        if args.shot and sid != args.shot:
            continue
        p = out / "ball" / f"{sid}_ball_track.json"
        if not p.exists():
            continue
        tr = BallTrack.load(p)
        pipe = ev.pipeline_by_ref(
            [f.frame - off for f in tr.frames], [f.world_xyz for f in tr.frames])
        report["shots"][sid] = ev.evaluate(
            dense, pipe, pipeline_events=_pipeline_events(out, sid, off),
            cameras=cameras, offsets=offsets)

    text = json.dumps(report, indent=2)
    print(text)
    if args.json:
        args.json.write_text(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
