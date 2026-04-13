#!/usr/bin/env python
"""One-off: extract ball tracks from existing shot clips and merge into tracks files.

The tracking stage was originally configured to *exclude* ball
detections from ByteTrack, so the existing
``output/tracks/<shot>_tracks.json`` files have no ball track even
though they contain players, goalkeepers, and referees.  Re-running
the full tracking stage to fix this would re-detect every player on
every frame, which is slow.

This script does only the ball-detection half of the work:

1. Iterate every shot clip listed in the manifest.
2. Run a YOLO ball detector on every frame.
3. Build a single ball ``Track`` per shot, with one ``TrackFrame``
   per detection (highest-confidence ball per frame).
4. Load the existing tracks JSON, remove any existing ball tracks
   (clean re-run), append the new ball track, save.

The downstream triangulation stage's ``_gather_ball_pixels_per_shot``
picks the longest ball track per shot, so a single high-quality
ball track is exactly what it expects.

Usage::

    .venv311/bin/python extract_ball_tracks.py --output output

Optional flags::

    --ball-model yolov8n.pt    # smaller/faster, default
    --ball-model yolov8x.pt    # larger/slower, may be more accurate
    --confidence 0.2           # ball-class confidence threshold
"""

from __future__ import annotations

import sys
from pathlib import Path

import click
import cv2

from src.schemas.shots import ShotsManifest
from src.schemas.tracks import Track, TrackFrame, TracksResult


# COCO "sports ball" class — used by the standard YOLOv8 weights.
_COCO_SPORTS_BALL = 32


def _detect_best_ball(model, frame, confidence_threshold: float) -> tuple[list[float], float] | None:
    """Run a YOLO model on a frame and return the highest-confidence ball bbox.

    Returns ``([x1, y1, x2, y2], conf)`` or ``None`` when no ball detection
    clears ``confidence_threshold``.
    """
    results = model(frame, verbose=False)[0]
    best: tuple[list[float], float] | None = None
    for box in results.boxes:
        cls_id = int(box.cls)
        conf = float(box.conf)
        if cls_id != _COCO_SPORTS_BALL or conf < confidence_threshold:
            continue
        if best is None or conf > best[1]:
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
            best = ([x1, y1, x2, y2], conf)
    return best


@click.command()
@click.option("--output", type=click.Path(file_okay=False), default="output",
              help="Pipeline output directory (where shots/, tracks/ live).")
@click.option("--ball-model", default="yolov8n.pt",
              help="YOLO weights to use for ball detection (default: yolov8n.pt).")
@click.option("--confidence", default=0.2,
              help="Minimum confidence for a ball detection to be kept.")
@click.option("--shots", default=None,
              help="Comma-separated subset of shot IDs to process (default: all).")
@click.option("--progress-every", default=200,
              help="Print a progress line every N frames within a shot.")
def main(output: str, ball_model: str, confidence: float,
         shots: str | None, progress_every: int) -> None:
    """Extract ball tracks from existing shot clips and merge them in."""
    out = Path(output)
    shots_dir = out / "shots"
    tracks_dir = out / "tracks"

    if not shots_dir.exists():
        click.echo(f"error: {shots_dir} does not exist", err=True)
        sys.exit(1)
    if not tracks_dir.exists():
        click.echo(f"error: {tracks_dir} does not exist — run tracking first", err=True)
        sys.exit(1)

    manifest = ShotsManifest.load_or_infer(shots_dir, persist=False)
    target_ids: set[str] | None = (
        {s.strip() for s in shots.split(",")} if shots else None
    )

    click.echo(f"loading {ball_model}…")
    from ultralytics import YOLO  # lazy: keeps non-ball CLI fast
    model = YOLO(ball_model)

    total_detected = 0
    total_frames = 0
    for shot in manifest.shots:
        if target_ids is not None and shot.id not in target_ids:
            continue
        clip_path = out / shot.clip_file
        tracks_path = tracks_dir / f"{shot.id}_tracks.json"
        if not clip_path.exists():
            click.echo(f"  - {shot.id}: no clip at {clip_path}, skipping")
            continue
        if not tracks_path.exists():
            click.echo(f"  - {shot.id}: no tracks file at {tracks_path}, skipping")
            continue

        cap = cv2.VideoCapture(str(clip_path))
        if not cap.isOpened():
            click.echo(f"  - {shot.id}: failed to open clip")
            continue

        n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        click.echo(f"  - {shot.id}: scanning {n_total} frames for ball…")

        ball_frames: list[TrackFrame] = []
        try:
            for fi in range(n_total):
                ok, frame = cap.read()
                if not ok:
                    break
                best = _detect_best_ball(model, frame, confidence)
                if best is not None:
                    bbox, conf = best
                    ball_frames.append(TrackFrame(
                        frame=fi,
                        bbox=bbox,
                        confidence=conf,
                        pitch_position=None,
                    ))
                if fi > 0 and fi % progress_every == 0:
                    click.echo(
                        f"     frame {fi}/{n_total}: {len(ball_frames)} balls so far"
                    )
        finally:
            cap.release()

        if not ball_frames:
            click.echo(f"     no ball detections — skipping merge")
            continue

        # Load existing tracks, drop any existing ball tracks, append new one.
        tr = TracksResult.load(tracks_path)
        before = len(tr.tracks)
        tr.tracks = [t for t in tr.tracks if t.class_name != "ball"]
        removed = before - len(tr.tracks)
        new_track = Track(
            track_id="BALL",
            class_name="ball",
            team="",
            frames=ball_frames,
        )
        tr.tracks.append(new_track)
        tr.save(tracks_path)
        click.echo(
            f"     saved {len(ball_frames)} ball detections "
            f"({removed} stale ball track(s) replaced)"
        )
        total_detected += len(ball_frames)
        total_frames += n_total

    if total_frames > 0:
        click.echo(
            f"\ndone. {total_detected} ball detections across "
            f"{total_frames} frames ({100*total_detected/total_frames:.1f}% hit rate)."
        )
        click.echo(
            "next: re-run triangulation to pick up the new ball tracks:\n"
            "  .venv311/bin/python recon.py run --output output --from-stage triangulation"
        )


if __name__ == "__main__":
    main()
