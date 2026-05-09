"""Stage 1: Prepare shots — copy pre-trimmed clips into ``shots/``.

The user manually trims clips in CapCut (or similar) and provides them
either as a single ``--input clip.mp4`` (single-shot) or as a directory
``--input clips/`` (multi-shot). This stage copies the clip(s) into
``output/shots/`` and writes a flat manifest. No automatic scene
segmentation is performed.

Re-running the stage merges new clips into the existing manifest rather
than overwriting it. Clips already present (matching ``shot_id`` and
destination filename) are left alone, so the dashboard's "Add Shots"
upload + "Continue" buttons can incrementally grow the shot list
without wiping per-shot artefacts produced by later stages.

When ``video_path`` is omitted the stage scans ``output/shots/`` for any
``.mp4`` files not yet recorded in the manifest and registers them — this
is the path the dashboard uses after writing uploaded clips directly
into ``shots/``.

When called against an existing single-shot output dir (legacy artefacts
at ``output/camera/anchors.json`` etc., no shot prefix), it migrates
those files in-place to the per-shot naming the rest of the pipeline
expects. Idempotent.

See spec section 5.1 and ``docs/superpowers/specs/2026-05-08-multi-shot-plumbing-design.md``.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import cv2

from src.pipeline.base import BaseStage
from src.schemas.shots import Shot, ShotsManifest, _sanitise_shot_id

logger = logging.getLogger(__name__)


def _video_metadata(clip_path: Path) -> tuple[float, int]:
    """Return ``(fps, frame_count)`` for the clip; (0.0, 0) on failure."""
    cap = cv2.VideoCapture(str(clip_path))
    try:
        if not cap.isOpened():
            return 0.0, 0
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        return fps, frames
    finally:
        cap.release()


def _migrate_legacy_artefacts(output_dir: Path, shot_id: str) -> None:
    """Rename legacy single-shot artefacts to per-shot naming.

    Idempotent — files that don't exist are skipped silently. If the
    per-shot variant already exists, the legacy file is left in place
    (the per-shot file wins; caller can clean up manually).
    """
    legacy_pairs = [
        (output_dir / "camera" / "anchors.json",
         output_dir / "camera" / f"{shot_id}_anchors.json"),
        (output_dir / "camera" / "camera_track.json",
         output_dir / "camera" / f"{shot_id}_camera_track.json"),
        (output_dir / "ball" / "ball_track.json",
         output_dir / "ball" / f"{shot_id}_ball_track.json"),
        (output_dir / "export" / "gltf" / "scene.glb",
         output_dir / "export" / "gltf" / f"{shot_id}_scene.glb"),
        (output_dir / "export" / "gltf" / "scene_metadata.json",
         output_dir / "export" / "gltf" / f"{shot_id}_scene_metadata.json"),
    ]
    migrated: list[str] = []
    for legacy, new in legacy_pairs:
        if not legacy.exists():
            continue
        if new.exists():
            continue
        legacy.rename(new)
        migrated.append(legacy.name)
    if migrated:
        logger.info(
            "[prepare_shots] migrated legacy single-shot artefacts to "
            "per-shot layout under shot_id=%s: %s",
            shot_id, ", ".join(migrated),
        )


def _build_shot(shot_id: str, clip_dest: Path, output_dir: Path) -> tuple[Shot, float, int]:
    """Probe ``clip_dest`` and return ``(shot, fps, frame_count)``."""
    fps, frame_count = _video_metadata(clip_dest)
    if frame_count <= 0:
        logger.warning(
            "prepare_shots: cv2 reported 0 frames for %s — manifest "
            "entry written but downstream stages may fail.",
            clip_dest,
        )
    effective_fps = fps if fps > 0 else 25.0
    end_frame = max(0, frame_count - 1)
    shot = Shot(
        id=shot_id,
        start_frame=0,
        end_frame=end_frame,
        start_time=0.0,
        end_time=(end_frame + 1) / effective_fps if frame_count > 0 else 0.0,
        clip_file=str(clip_dest.relative_to(output_dir)),
    )
    return shot, effective_fps, frame_count


class PrepareShotsStage(BaseStage):
    name = "prepare_shots"

    def __init__(
        self,
        config: dict,
        output_dir: Path,
        video_path: Path | None = None,
        **_: object,
    ) -> None:
        super().__init__(config, output_dir)
        self.video_path = video_path

    def is_complete(self) -> bool:
        return (self.output_dir / "shots" / "shots_manifest.json").exists()

    def run(self) -> None:
        shots_dir = self.output_dir / "shots"
        shots_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = shots_dir / "shots_manifest.json"

        existing = (
            ShotsManifest.load(manifest_path)
            if manifest_path.exists()
            else ShotsManifest(source_file="", fps=25.0, total_frames=0, shots=[])
        )
        known_ids = {s.id for s in existing.shots}

        # Resolve the input. ``video_path`` is optional so the dashboard's
        # "Continue" button can rescan ``shots/`` for clips uploaded out-
        # of-band without forcing the operator to re-pick them.
        clip_files: list[Path] = []
        if self.video_path is not None:
            clip_src = Path(self.video_path).resolve()
            if not clip_src.exists():
                raise FileNotFoundError(f"Input not found: {clip_src}")
            if clip_src.is_dir():
                clip_files = sorted(clip_src.glob("*.mp4"))
                if not clip_files and not existing.shots:
                    raise FileNotFoundError(f"no .mp4 files in {clip_src}")
            else:
                clip_files = [clip_src]

        # Single-input → also migrate any legacy single-shot artefacts to
        # per-shot naming under the resulting shot_id.
        if len(clip_files) == 1:
            legacy_shot_id = _sanitise_shot_id(clip_files[0].stem)
            _migrate_legacy_artefacts(self.output_dir, legacy_shot_id)

        new_shots: list[Shot] = []
        seen_new: set[str] = set()
        fps_observed = existing.fps if existing.shots else 25.0
        added_frames = 0

        for clip_path in clip_files:
            shot_id = _sanitise_shot_id(clip_path.stem)
            if shot_id in seen_new:
                raise ValueError(
                    f"duplicate shot_id {shot_id!r} from {clip_path}; "
                    "rename one of the input clips so their stems differ "
                    "after sanitisation"
                )
            seen_new.add(shot_id)

            clip_dest = shots_dir / f"{shot_id}.mp4"
            try:
                same_file = clip_dest.exists() and clip_dest.samefile(clip_path)
            except FileNotFoundError:
                same_file = False
            if not same_file:
                shutil.copy2(clip_path, clip_dest)

            if shot_id in known_ids:
                logger.info(
                    "prepare_shots: skipping already-registered shot %s",
                    shot_id,
                )
                continue

            shot, effective_fps, frame_count = _build_shot(
                shot_id, clip_dest, self.output_dir,
            )
            new_shots.append(shot)
            fps_observed = effective_fps
            added_frames += frame_count

        # Pick up any clips already in shots/ that aren't in the manifest
        # — covers the "files uploaded directly into shots/ via the
        # dashboard's Add Shots button" flow as well as manual drops.
        for clip_path in sorted(shots_dir.glob("*.mp4")):
            shot_id = _sanitise_shot_id(clip_path.stem)
            if shot_id in known_ids or shot_id in seen_new:
                continue
            shot, effective_fps, frame_count = _build_shot(
                shot_id, clip_path, self.output_dir,
            )
            new_shots.append(shot)
            seen_new.add(shot_id)
            fps_observed = effective_fps
            added_frames += frame_count

        if not new_shots and not existing.shots:
            raise ValueError(
                "prepare_shots: no clips to register — pass --input "
                "<clip.mp4 or dir> or upload clips via the dashboard."
            )

        manifest = ShotsManifest(
            source_file=(
                str(self.video_path.resolve())
                if self.video_path is not None
                else existing.source_file
            ),
            fps=fps_observed,
            total_frames=existing.total_frames + added_frames,
            shots=existing.shots + new_shots,
        )
        manifest.save(manifest_path)
        if new_shots:
            logger.info(
                "prepare_shots: added %d shot(s) (%s); total now %d",
                len(new_shots),
                ", ".join(s.id for s in new_shots),
                len(manifest.shots),
            )
        else:
            logger.info(
                "prepare_shots: manifest unchanged (%d shot(s) already registered)",
                len(manifest.shots),
            )
