#!/usr/bin/env python3
"""Build a JobManifest for a single (shot, player) pair from local outputs.

Usage:
  python scripts/build_test_manifest.py \\
      --output-dir ./output \\
      --shot-id shot_03 \\
      --player-id p17 \\
      --manifest-out /tmp/manifest.json

The resulting manifest can be fed to ``recon.py batch-handler`` (no
Docker) or to ``docker run ... <image>`` to verify the container works
end-to-end without AWS.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow ``python scripts/build_test_manifest.py`` from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.cloud.manifest import JobManifest  # noqa: E402
from src.pipeline.config import load_config  # noqa: E402
from src.schemas.tracks import TracksResult  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("./output"))
    parser.add_argument("--shot-id", required=True)
    parser.add_argument("--player-id", required=True)
    parser.add_argument(
        "--manifest-out",
        type=Path,
        required=True,
        help="Where to write the JobManifest JSON.",
    )
    parser.add_argument(
        "--output-prefix",
        default=None,
        help=(
            "output_prefix to embed in the manifest. Default: "
            "''file://{manifest_out_dir}/handler-out''."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional config override; otherwise uses config/default.yaml.",
    )
    args = parser.parse_args()

    tracks_path = args.output_dir / "tracks" / f"{args.shot_id}_tracks.json"
    camera_path = args.output_dir / "camera" / f"{args.shot_id}_camera_track.json"
    video_path = args.output_dir / "shots" / f"{args.shot_id}.mp4"
    for p in (tracks_path, camera_path, video_path):
        if not p.exists():
            print(f"missing input: {p}", file=sys.stderr)
            return 2

    tr = TracksResult.load(tracks_path)
    track = next(
        (t for t in tr.tracks if t.player_id == args.player_id),
        None,
    )
    if track is None:
        ids = sorted({t.player_id for t in tr.tracks if t.player_id})
        print(
            f"player_id {args.player_id!r} not found in {tracks_path}; "
            f"available: {ids}",
            file=sys.stderr,
        )
        return 2

    track_frames = tuple(
        (int(f.frame), tuple(int(x) for x in f.bbox))
        for f in track.frames
    )
    if not track_frames:
        print(f"player {args.player_id} has no frames", file=sys.stderr)
        return 2

    cfg = load_config(args.config)
    hmr_cfg = {
        k: v for k, v in cfg.get("hmr_world", {}).items()
        if k not in ("batch", "runner")
    }

    output_prefix = args.output_prefix or f"file://{args.manifest_out.parent.resolve()}/handler-out"
    manifest = JobManifest(
        run_id="local-test",
        shot_id=args.shot_id,
        player_id=args.player_id,
        video_uri=f"file://{video_path.resolve()}",
        camera_track_uri=f"file://{camera_path.resolve()}",
        track_frames=track_frames,
        hmr_world_cfg=hmr_cfg,
        output_prefix=output_prefix,
    )
    args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_out.write_text(manifest.to_json())
    print(f"wrote manifest: {args.manifest_out}")
    print(f"  shot_id    = {manifest.shot_id}")
    print(f"  player_id  = {manifest.player_id}")
    print(f"  n_frames   = {len(manifest.track_frames)}")
    print(f"  video_uri  = {manifest.video_uri}")
    print(f"  output     = {manifest.output_prefix}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
