"""Build the WASB fine-tune corpus from annotated clips (spec §4.3 step 1).

For each --pairs OUTPUT_DIR:CLIP_ID this extracts every clip frame to
<corpus>/frames/<clip>/{fid:05d}.png and writes <corpus>/annos/<clip>.xml
containing the operator's gold anchor pixels UNION solved-track weak labels
within ±window of each gold frame (gold wins on collision). A manifest
records the train/holdout split.

Usage:
    python scripts/build_finetune_corpus.py \
        --pairs output:gberch output-origi:origi01 \
                output-kroupi:kroupi01 output-japan:s013 \
        --corpus-root output/ball_finetune_corpus \
        --holdout kroupi01
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from src.schemas.ball_anchor import BallAnchorSet  # noqa: E402
from src.schemas.ball_track import BallTrack  # noqa: E402
from src.schemas.camera_track import CameraTrack  # noqa: E402
from src.utils.ball_finetune_export import extract_frames  # noqa: E402
from src.utils.ball_weak_labels import (  # noqa: E402
    labels_to_cvat_xml,
    merge_labels,
    weak_labels_from_track,
)


def build_clip_entry(
    output_dir: Path,
    clip_id: str,
    corpus_root: Path,
    *,
    window: int,
    min_conf: float,
    skip_frames: bool = False,
) -> dict:
    """Frames + merged gold∪weak XML for one clip; returns the manifest entry."""
    anchors = BallAnchorSet.load(
        output_dir / "ball" / f"{clip_id}_ball_anchors.json")
    gold = {
        a.frame: (float(a.image_xy[0]), float(a.image_xy[1]))
        for a in anchors.anchors if a.image_xy is not None
    }

    camera = CameraTrack.load(
        output_dir / "camera" / f"{clip_id}_camera_track.json")
    per_frame_K = {f.frame: np.array(f.K) for f in camera.frames}
    per_frame_R = {f.frame: np.array(f.R) for f in camera.frames}
    t_world = np.array(camera.t_world)
    per_frame_t = {
        f.frame: (np.array(f.t) if f.t is not None else t_world)
        for f in camera.frames
    }
    track = BallTrack.load(output_dir / "ball" / f"{clip_id}_ball_track.json")
    weak = weak_labels_from_track(
        track,
        per_frame_K=per_frame_K, per_frame_R=per_frame_R,
        per_frame_t=per_frame_t, distortion=camera.distortion,
        image_size=camera.image_size, gold_frames=set(gold),
        window=window, min_conf=min_conf,
    )
    merged = merge_labels(gold, weak)

    anno_path = corpus_root / "annos" / f"{clip_id}.xml"
    anno_path.parent.mkdir(parents=True, exist_ok=True)
    anno_path.write_text(labels_to_cvat_xml(clip_id, merged))

    frames_dir = corpus_root / "frames" / clip_id
    if skip_frames and frames_dir.exists():
        n_frames = len(list(frames_dir.glob("*.png")))
    else:
        n_frames = extract_frames(
            output_dir / "shots" / f"{clip_id}.mp4", frames_dir)

    return {
        "clip_id": clip_id,
        "n_gold": len(gold),
        "n_weak": len(weak),
        "n_frames": n_frames,
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", nargs="+", required=True,
                    metavar="OUTPUT_DIR:CLIP_ID")
    ap.add_argument("--corpus-root", type=Path, required=True)
    ap.add_argument("--holdout", nargs="*", default=[])
    ap.add_argument("--weak-window", type=int, default=20)
    ap.add_argument("--weak-min-conf", type=float, default=0.5)
    ap.add_argument("--skip-frames", action="store_true",
                    help="reuse already-extracted frames")
    args = ap.parse_args()

    clips: dict[str, dict] = {}
    for pair in args.pairs:
        out_dir, _, clip_id = pair.partition(":")
        if not clip_id:
            ap.error(f"--pairs entries must be OUTPUT_DIR:CLIP_ID; got {pair!r}")
        entry = build_clip_entry(
            Path(out_dir), clip_id, args.corpus_root,
            window=args.weak_window, min_conf=args.weak_min_conf,
            skip_frames=args.skip_frames,
        )
        clips[clip_id] = entry
        print(f"{clip_id}: gold={entry['n_gold']} weak={entry['n_weak']} "
              f"frames={entry['n_frames']}")

    unknown = [h for h in args.holdout if h not in clips]
    if unknown:
        ap.error(f"--holdout clip(s) not in --pairs: {unknown}")
    manifest = {
        "clips": clips,
        "holdout": list(args.holdout),
        "train": [c for c in clips if c not in set(args.holdout)],
    }
    (args.corpus_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2))
    print(f"manifest: train={manifest['train']} holdout={manifest['holdout']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
