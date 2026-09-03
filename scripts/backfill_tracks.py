"""Apply the backward track-extension pass (``src.utils.track_backfill``)
to an EXISTING ``tracks/<shot>_tracks.json`` WITHOUT re-running the
tracking stage — ball-stage campaign Workstream 4.

Motivating case: on gberch, P008 (Hato)'s track starts at frame 62
though the player is visible earlier, leaving a manual ball-touch
anchor at frame 56 unattributable (no FK exists there). Re-running the
tracking stage would mint new track ids and destroy operator-assigned
``player_id``/``player_name`` annotations, so this script instead walks
backward from each eligible track's ORIGINAL first frame and prepends
matched detections, never touching track ids, frame ordering, or
annotations on any track it doesn't extend.

Usage:
    .venv311/bin/python scripts/backfill_tracks.py \\
        --output output --shot gberch
    # in place by default: backs up the PRISTINE original to
    # tracks/<shot>_tracks.json.bak once (never overwritten on later
    # runs), then overwrites tracks/<shot>_tracks.json.

    .venv311/bin/python scripts/backfill_tracks.py \\
        --output output --shot gberch --player P008
    # only attempt the named player's track(s).

    .venv311/bin/python scripts/backfill_tracks.py \\
        --output output --shot gberch --dry-run
    # report what WOULD change, writes nothing (no .bak either).

Config knobs (``tracking.backfill.*`` in config/default.yaml) are used
as defaults; a handful are overridable on the command line for one-off
tuning without editing the config file.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline.config import load_config  # noqa: E402
from src.schemas.shots import ShotsManifest  # noqa: E402
from src.schemas.tracks import Track, TracksResult  # noqa: E402
from src.utils.player_detector import PlayerDetector, YOLOPlayerDetector  # noqa: E402
from src.utils.track_backfill import (  # noqa: E402
    BackfillConfig,
    VideoFrameSource,
    backfill_tracks_result,
)

_BAK_SUFFIX = ".bak"


def _backup_original(tracks_path: Path) -> Path | None:
    """One-time safety copy: ``<name>_tracks.json.bak``. Returns the
    backup path when a NEW backup was written, ``None`` when one
    already existed (and was therefore left untouched)."""
    bak_path = tracks_path.with_name(tracks_path.name + _BAK_SUFFIX)
    if bak_path.exists():
        return None
    shutil.copyfile(tracks_path, bak_path)
    return bak_path


def _resolve_clip_path(output_dir: Path, shot_id: str) -> Path:
    manifest_path = output_dir / "shots" / "shots_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"No shots manifest at {manifest_path}")
    manifest = ShotsManifest.load(manifest_path)
    for shot in manifest.shots:
        if shot.id == shot_id:
            return output_dir / shot.clip_file
    raise ValueError(f"Shot {shot_id!r} not found in {manifest_path}")


def _make_select(player_ids: set[str] | None, track_ids: set[str] | None):
    def _select(track: Track) -> bool:
        if player_ids is not None and track.player_id not in player_ids:
            return False
        if track_ids is not None and track.track_id not in track_ids:
            return False
        return True
    return _select


def _build_yolo_detector(tracking_cfg: dict) -> YOLOPlayerDetector:
    sahi_cfg = tracking_cfg.get("sahi", {}) or {}
    return YOLOPlayerDetector(
        model_name=tracking_cfg.get("player_model", "yolov8x.pt"),
        confidence=tracking_cfg.get("confidence_threshold", 0.3),
        iou_threshold=float(tracking_cfg.get("iou_threshold", 0.85)),
        imgsz=int(tracking_cfg.get("imgsz", 1280)),
        sahi_enabled=bool(sahi_cfg.get("enabled", False)),
        sahi_tile_size=int(sahi_cfg.get("tile_size", 960)),
        sahi_overlap_ratio=float(sahi_cfg.get("overlap_ratio", 0.25)),
        sahi_nms_iou_threshold=float(sahi_cfg.get("nms_iou_threshold", 0.5)),
    )


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--output", required=True, help="pipeline output directory")
    ap.add_argument("--shot", required=True, help="shot id, e.g. gberch")
    ap.add_argument(
        "--player", default=None,
        help="comma-separated player_id(s) to restrict to (default: all tracks)",
    )
    ap.add_argument(
        "--track-id", default=None,
        help="comma-separated track_id(s) to restrict to, e.g. T008 (default: all tracks)",
    )
    ap.add_argument(
        "--dry-run", action="store_true",
        help="report what would change; write nothing (no .bak either)",
    )
    ap.add_argument("--min-late-start-frames", type=int, default=None)
    ap.add_argument("--min-iou", type=float, default=None)
    ap.add_argument("--patience", type=int, default=None)
    ap.add_argument("--max-backfill-frames", type=int, default=None)
    ap.add_argument(
        "--use-appearance-gate", action="store_true", default=None,
        help="also gate acceptance on an HSV-histogram appearance distance",
    )
    return ap


def run(args: argparse.Namespace, detector: PlayerDetector | None = None) -> int:
    """Execute one backfill pass. ``detector`` is injectable so tests
    exercise the full CLI orchestration (arg handling, selection,
    .bak safety, in-place write) with a ``FakePlayerDetector`` instead
    of paying for real YOLO inference / a model download."""
    output_dir = Path(args.output)
    tracks_path = output_dir / "tracks" / f"{args.shot}_tracks.json"
    if not tracks_path.exists():
        print(f"[backfill_tracks] no tracks file at {tracks_path}")
        return 1

    cfg = load_config()
    raw_backfill_cfg = dict(cfg.get("tracking", {}).get("backfill", {}) or {})
    for key, cli_val in (
        ("min_late_start_frames", args.min_late_start_frames),
        ("min_iou", args.min_iou),
        ("patience", args.patience),
        ("max_backfill_frames", args.max_backfill_frames),
        ("use_appearance_gate", args.use_appearance_gate),
    ):
        if cli_val is not None:
            raw_backfill_cfg[key] = cli_val
    backfill_cfg = BackfillConfig.from_dict(raw_backfill_cfg)

    player_ids = (
        {p.strip() for p in args.player.split(",") if p.strip()} if args.player else None
    )
    track_ids = (
        {t.strip() for t in args.track_id.split(",") if t.strip()} if args.track_id else None
    )
    select = _make_select(player_ids, track_ids)

    if detector is None:
        detector = _build_yolo_detector(cfg.get("tracking", {}))

    clip_path = _resolve_clip_path(output_dir, args.shot)
    tracks_result = TracksResult.load(tracks_path)
    frame_source = VideoFrameSource(clip_path)
    try:
        new_result, reports = backfill_tracks_result(
            tracks_result, frame_source, detector, backfill_cfg, select=select
        )
    finally:
        frame_source.close()

    n_extended = 0
    for r in reports:
        if r.frames_added > 0:
            n_extended += 1
            print(
                f"  {r.track_id} ({r.player_id or 'unassigned'}): "
                f"frame {r.original_start_frame} -> {r.new_start_frame} "
                f"(+{r.frames_added} frames, stop={r.stop_reason})"
            )
        else:
            print(
                f"  {r.track_id} ({r.player_id or 'unassigned'}): "
                f"unchanged (stop={r.stop_reason})"
            )

    print(
        f"[backfill_tracks] {args.shot}: {len(reports)} track(s) selected, "
        f"{n_extended} extended"
    )

    if args.dry_run:
        print("[backfill_tracks] --dry-run: no files written")
        return 0

    if n_extended == 0:
        print("[backfill_tracks] nothing to write")
        return 0

    bak = _backup_original(tracks_path)
    if bak is not None:
        print(f"[backfill_tracks] backed up {tracks_path.name} -> {bak.name}")
    else:
        print(f"[backfill_tracks] {tracks_path.name}.bak already exists, left untouched")

    new_result.save(tracks_path)
    print(f"[backfill_tracks] wrote {tracks_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
