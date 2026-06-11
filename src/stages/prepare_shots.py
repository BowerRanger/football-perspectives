"""Stage 1: Prepare shots — copy pre-trimmed clips OR split a full reel.

Two modes (``prepare_shots.mode``: ``auto`` | ``copy`` | ``split``):

**Copy** (the original behaviour): the user manually trims clips and
provides them either as a single ``--input clip.mp4`` or as a directory
``--input clips/``. Clips are copied into ``output/shots/`` and merged
into a flat manifest. Re-running merges rather than overwrites, so the
dashboard's "Add Shots" upload can grow the shot list incrementally;
omitting ``video_path`` rescans ``shots/`` for unregistered clips.
Legacy single-shot artefacts are migrated to per-shot naming.

**Split** (highlights ingestion, 2026-06-11 design): a single long
input (e.g. a full match-highlights reel) is automatically

1. split into shots (PySceneDetect, ``src/utils/shot_split.py``),
2. classified per shot (pitch ratio / fades / zoom-invariant motion
   rate, ``src/utils/shot_features.py``) — reaction and transition
   shots are marked ``excluded`` (still extracted, so the dashboard's
   dropped tray can preview + restore them),
3. grouped into highlight events (``src/utils/highlight_grouping.py``),
4. extracted frame-accurately (slow-mo replays retimed to real time),
5. auto-aligned within each group (motion-profile NCC,
   ``src/utils/shot_alignment.py``) into ``shots/sync_map.json``.

``auto`` picks split for a single video at least
``split.min_input_duration_s`` long, copy otherwise.

See ``docs/superpowers/specs/2026-06-11-highlights-ingestion-design.md``
and ``docs/superpowers/specs/2026-05-08-multi-shot-plumbing-design.md``.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
from dataclasses import asdict, replace
from pathlib import Path

import cv2

from src.pipeline.base import BaseStage
from src.schemas.shots import (
    HighlightGroup,
    Shot,
    ShotsManifest,
    _sanitise_shot_id,
)

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


def _next_sequential_id(existing_ids: set[str], prefix: str) -> int:
    """First free index for ids shaped ``{prefix}{NNN}`` (1-based).

    Zero-padding is the caller's concern when formatting the id.
    """
    pattern = re.compile(rf"^{re.escape(prefix)}(\d+)$")
    taken = [int(m.group(1)) for sid in existing_ids
             if (m := pattern.match(sid))]
    return max(taken, default=0) + 1


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

    # ── Mode dispatch ─────────────────────────────────────────────────

    def run(self) -> None:
        cfg = self.config.get("prepare_shots", {})
        mode = cfg.get("mode", "auto")
        if mode not in ("auto", "copy", "split"):
            raise ValueError(
                f"prepare_shots.mode must be auto|copy|split, got {mode!r}"
            )

        if mode != "copy" and self.video_path is not None:
            src = Path(self.video_path).resolve()
            if src.exists() and src.is_file():
                if mode == "split":
                    self._run_split_mode(src)
                    return
                # auto: split only when the input is long enough to be a
                # reel rather than a hand-trimmed clip.
                min_dur = float(
                    cfg.get("split", {}).get("min_input_duration_s", 90)
                )
                fps, frames = _video_metadata(src)
                duration_s = frames / fps if fps > 0 else 0.0
                if duration_s >= min_dur:
                    self._run_split_mode(src)
                    return
        self._run_copy_mode()

    # ── Split mode (highlights reel ingestion) ────────────────────────

    def _run_split_mode(self, src: Path) -> None:
        from src.utils.ffmpeg import extract_clip_reencode, extract_thumbnail
        from src.utils.highlight_grouping import GroupingInput, group_shots
        from src.utils.shot_alignment import align_group
        from src.utils.shot_features import (
            compute_span_features,
            estimate_speed_factors,
        )
        from src.utils.shot_split import detect_spans, merge_short_spans

        cfg = self.config.get("prepare_shots", {})
        split_cfg = cfg.get("split", {})
        classify_cfg = cfg.get("classify", {})
        group_cfg = cfg.get("group", {})
        align_cfg = cfg.get("align", {})

        shots_dir = self.output_dir / "shots"
        shots_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = shots_dir / "shots_manifest.json"

        existing = (
            ShotsManifest.load(manifest_path)
            if manifest_path.exists()
            else ShotsManifest(source_file="", fps=25.0, total_frames=0)
        )
        if existing.shots and existing.source_file == str(src):
            logger.info(
                "prepare_shots[split]: %s already ingested (%d shots) — "
                "skipping. Use --clean (or a fresh output dir) to re-split.",
                src.name, len(existing.shots),
            )
            return

        fps, total_frames = _video_metadata(src)
        if fps <= 0 or total_frames <= 0:
            raise RuntimeError(
                f"prepare_shots[split]: unreadable input {src} "
                f"(fps={fps}, frames={total_frames})"
            )

        # 1. Shot boundaries.
        logger.info("prepare_shots[split]: detecting shots in %s …", src.name)
        spike_cfg = split_cfg.get("spike_rescue", {})
        dissolve_cfg = split_cfg.get("dissolve_split", {})
        spans = detect_spans(
            src,
            detector=split_cfg.get("detector", "adaptive"),
            threshold=float(split_cfg.get("threshold", 27.0)),
            adaptive_threshold=float(split_cfg.get("adaptive_threshold", 3.0)),
            min_scene_len_frames=int(split_cfg.get("min_scene_len_frames", 13)),
            adaptive_min_content_val=float(
                split_cfg.get("adaptive_min_content_val", 15.0)
            ),
            spike_rescue=bool(spike_cfg.get("enabled", True)),
            spike_z_min=float(spike_cfg.get("z_min", 4.0)),
            spike_abs_min=float(spike_cfg.get("abs_min", 18.0)),
            spike_window_frames=int(spike_cfg.get("window_frames", 25)),
            dissolve_split=bool(dissolve_cfg.get("enabled", False)),
            dissolve_uniformity_min=float(
                dissolve_cfg.get("uniformity_min", 10.0)
            ),
            dissolve_flow_max=float(dissolve_cfg.get("flow_max", 1.25)),
            dissolve_min_run_frames=int(
                dissolve_cfg.get("min_run_frames", 5)
            ),
        )
        spans = merge_short_spans(
            spans,
            max_short_duration_s=float(
                split_cfg.get("merge_short_shots_max_duration_s", 1.2)
            ),
            max_gap_s=float(split_cfg.get("merge_max_gap_s", 0.08)),
        )
        min_shot_s = float(split_cfg.get("min_shot_duration_s", 1.0))
        spans = [s for s in spans if s.duration_s >= min_shot_s]
        if not spans:
            raise ValueError(
                "prepare_shots[split]: no shots detected — try "
                "prepare_shots.split.detector: content or a lower threshold."
            )
        logger.info("prepare_shots[split]: %d shots after hygiene", len(spans))

        # 2. Features + classification + speed factors.
        replay_min = float(classify_cfg.get("replay_min_speed_factor", 1.25))
        person_cfg = classify_cfg.get("person_check", {})
        person_height_fn = None
        if bool(person_cfg.get("enabled", False)):
            from src.utils.shot_features import make_yolo_person_height_fn

            try:
                person_height_fn = make_yolo_person_height_fn(
                    str(person_cfg.get("model", "yolov8n.pt")),
                    confidence=float(person_cfg.get("confidence", 0.35)),
                )
                logger.info(
                    "prepare_shots[split]: person check enabled (%s)",
                    person_cfg.get("model", "yolov8n.pt"),
                )
            except Exception as exc:
                logger.warning(
                    "prepare_shots[split]: person check unavailable (%s) — "
                    "close-up classification disabled for this run", exc,
                )
        features = compute_span_features(
            src, spans,
            person_height_fn=person_height_fn,
            closeup_max_person_height=float(
                classify_cfg.get("closeup_max_person_height", 0.5)
            ),
            sample_points=list(
                classify_cfg.get("sample_points", [0.15, 0.3, 0.5, 0.7, 0.85])
            ),
            reaction_max_median_pitch_ratio=float(
                classify_cfg.get("reaction_max_median_pitch_ratio", 0.12)
            ),
            reaction_max_peak_pitch_ratio=float(
                classify_cfg.get("reaction_max_peak_pitch_ratio", 0.20)
            ),
            fade_black_frame_threshold=float(
                classify_cfg.get("fade_black_frame_threshold", 0.18)
            ),
            fade_min_brightness_range=float(
                classify_cfg.get("fade_min_brightness_range", 0.25)
            ),
            transition_max_duration_s=float(
                classify_cfg.get("transition_max_duration_s", 2.0)
            ),
            wide_min_pitch_ratio=float(
                classify_cfg.get("wide_min_pitch_ratio", 0.40)
            ),
            tight_max_pitch_ratio=float(
                classify_cfg.get("tight_max_pitch_ratio", 0.22)
            ),
        )
        features = estimate_speed_factors(
            features, replay_min_speed_factor=replay_min,
        )

        # 3. Extract clips + thumbnails, build Shot rows.
        existing_ids = {s.id for s in existing.shots}
        next_idx = _next_sequential_id(existing_ids, "s")
        thumbs_dir = shots_dir / "thumbs"
        new_shots: list[Shot] = []
        kept_features = []  # (shot_id, ShotFeatures) for grouping/sidecar
        added_frames = 0
        for f in features:
            sid = f"s{next_idx:03d}"
            next_idx += 1
            dest = shots_dir / f"{sid}.mp4"
            # Retime only confident replays (or clearly sped-up spans):
            # live shots carry ±0.2 estimation noise and must not be
            # resampled on it.
            retime = f.speed_factor >= replay_min or f.speed_factor <= 0.8
            try:
                extract_clip_reencode(
                    src, dest, f.span.start_s, f.span.end_s, fps=fps,
                    speed_factor=f.speed_factor if retime else 1.0,
                )
            except subprocess.CalledProcessError as exc:
                logger.warning(
                    "prepare_shots[split]: extraction failed for %s "
                    "(%.1f–%.1fs): %s — skipping shot",
                    sid, f.span.start_s, f.span.end_s, exc,
                )
                continue
            try:
                extract_thumbnail(
                    src, thumbs_dir / f"{sid}.jpg",
                    (f.span.start_s + f.span.end_s) / 2.0,
                )
            except subprocess.CalledProcessError as exc:
                logger.warning(
                    "prepare_shots[split]: thumbnail failed for %s: %s",
                    sid, exc,
                )

            shot, _, frame_count = _build_shot(sid, dest, self.output_dir)
            shot = replace(
                shot,
                speed_factor=f.speed_factor,
                kind=f.kind,
                excluded=f.kind != "gameplay",
                exclude_reason="" if f.kind == "gameplay" else f.kind,
                source_start_s=f.span.start_s,
                source_end_s=f.span.end_s,
            )
            new_shots.append(shot)
            kept_features.append((sid, f))
            added_frames += frame_count
        logger.info(
            "prepare_shots[split]: extracted %d clips (%d excluded as "
            "reaction/transition)",
            len(new_shots), sum(1 for s in new_shots if s.excluded),
        )

        # 4. Group into highlights.
        grouping_inputs = [
            GroupingInput(
                shot_id=sid,
                kind=f.kind,
                scale=f.scale,
                speed_factor=f.speed_factor,
                source_start_s=f.span.start_s,
                source_end_s=f.span.end_s,
            )
            for sid, f in kept_features
        ]
        grouped = group_shots(
            grouping_inputs,
            gap_boundary_s=float(group_cfg.get("gap_boundary_s", 5.0)),
            replay_min_speed_factor=replay_min,
        )
        group_idx = _next_sequential_id({g.id for g in existing.groups}, "g")
        new_groups: list[HighlightGroup] = []
        group_id_by_shot: dict[str, str] = {}
        reference_by_group: dict[str, str] = {}
        for g in grouped:
            idx = group_idx
            group_idx += 1
            gid = f"g{idx:02d}"
            new_groups.append(HighlightGroup(
                id=gid,
                label=f"Highlight {idx}",
                shot_ids=list(g.shot_ids),
                boundary_rule=g.boundary_rule,
                boundary_confidence=g.boundary_confidence,
            ))
            reference_by_group[gid] = g.reference_shot
            for sid in g.shot_ids:
                group_id_by_shot[sid] = gid
        new_shots = [
            replace(s, group_id=group_id_by_shot.get(s.id, ""))
            for s in new_shots
        ]
        logger.info(
            "prepare_shots[split]: %d highlight group(s): %s",
            len(new_groups),
            ", ".join(f"{g.id}[{len(g.shot_ids)}]" for g in new_groups),
        )

        # 5. Auto-align each multi-shot group.
        if bool(align_cfg.get("enabled", True)):
            self._auto_align_groups(
                new_groups, reference_by_group, new_shots, fps, align_cfg,
            )

        # 6. Persist manifest + features sidecar.
        manifest = ShotsManifest(
            source_file=str(src),
            fps=existing.fps if existing.shots else fps,
            total_frames=existing.total_frames + added_frames,
            shots=existing.shots + new_shots,
            groups=existing.groups + new_groups,
            match=existing.match,
        )
        manifest.save(manifest_path)
        self._write_features_sidecar(shots_dir, kept_features)
        logger.info(
            "prepare_shots[split]: manifest written — %d shot(s), "
            "%d group(s)", len(manifest.shots), len(manifest.groups),
        )

    def _auto_align_groups(
        self,
        groups: list[HighlightGroup],
        reference_by_group: dict[str, str],
        shots: list[Shot],
        fps: float,
        align_cfg: dict,
    ) -> None:
        from src.schemas.sync_map import Alignment, SyncMap
        from src.utils.shot_alignment import align_group

        sync_path = self.output_dir / "shots" / "sync_map.json"
        sync_map = SyncMap.load(sync_path) if sync_path.exists() else SyncMap()
        clip_by_id = {s.id: self.output_dir / s.clip_file for s in shots}

        for group in groups:
            members = [sid for sid in group.shot_ids if sid in clip_by_id]
            if len(members) < 2:
                continue
            reference = reference_by_group.get(group.id) or members[0]
            logger.info(
                "prepare_shots[split]: aligning %s (%d shots, ref %s)",
                group.id, len(members), reference,
            )
            results = align_group(
                {sid: clip_by_id[sid] for sid in members},
                reference_id=reference,
                width_px=int(align_cfg.get("curve_width_px", 192)),
                smooth_sigma=float(align_cfg.get("smooth_sigma_frames", 2.0)),
                min_overlap_frames=max(
                    2, int(float(align_cfg.get("min_overlap_s", 1.0)) * fps),
                ),
                min_confidence=float(align_cfg.get("min_confidence", 0.5)),
            )
            saved = sync_map.group(group.id)
            for sid, result in results.items():
                if saved is not None:
                    prior = next(
                        (a for a in saved.alignments if a.shot_id == sid),
                        None,
                    )
                    if prior is not None and prior.method == "manual":
                        continue  # operator-tuned offsets always win
                sync_map = sync_map.with_group_alignment(
                    group.id, reference, Alignment(
                        shot_id=sid,
                        frame_offset=result.frame_offset,
                        method=result.method,
                        confidence=result.confidence,
                    ),
                )
        sync_map.save(sync_path)

    def _write_features_sidecar(self, shots_dir: Path, kept_features) -> None:
        """Per-shot diagnostics for the dashboard's badges/tooltips and
        the quality report. Merges over an existing sidecar so split
        runs into a dir with prior shots keep their rows."""
        sidecar_path = shots_dir / "shot_features.json"
        data: dict = {}
        if sidecar_path.exists():
            try:
                data = json.loads(sidecar_path.read_text())
            except Exception:
                logger.warning(
                    "prepare_shots[split]: unreadable %s — rewriting",
                    sidecar_path,
                )
        for sid, f in kept_features:
            data[sid] = {
                "pitch_ratio_median": round(f.pitch_ratio_median, 4),
                "pitch_ratio_peak": round(f.pitch_ratio_peak, 4),
                "brightness_min": round(f.brightness_min, 4),
                "brightness_range": round(f.brightness_range, 4),
                "motion_rate": round(f.motion_rate, 6),
                "max_person_height": round(f.max_person_height, 3),
                "speed_factor": round(f.speed_factor, 3),
                "kind": f.kind,
                "scale": f.scale,
            }
        sidecar_path.write_text(json.dumps(data, indent=2))

    # ── Copy mode (pre-trimmed clips; original behaviour) ─────────────

    def _run_copy_mode(self) -> None:
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
            groups=existing.groups,
            match=existing.match,
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
