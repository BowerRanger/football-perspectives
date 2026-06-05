"""Inspect a camera track produced with auto_anchors (zero manual clicks)
and print confidence + quality-report camera diagnostics for comparison
against the manual baseline (~0.95 px mean line RMS on gberch).

  python scripts/eval_auto_anchor.py --output ./output-autotest --shot s1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.schemas.camera_track import CameraTrack


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--shot", default="s1")
    args = ap.parse_args()
    out = Path(args.output)
    track = CameraTrack.load(out / "camera" / f"{args.shot}_camera_track.json")
    confs = [f.confidence for f in track.frames]
    print(f"frames={len(track.frames)} mean_conf={sum(confs)/len(confs):.3f}")
    qr = out / "quality_report.json"
    if qr.exists():
        print(json.dumps(json.loads(qr.read_text()).get("camera", {}), indent=2))


if __name__ == "__main__":
    main()
