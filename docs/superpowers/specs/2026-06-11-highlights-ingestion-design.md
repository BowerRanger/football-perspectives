# Highlights-Reel Ingestion Design

**Date:** 2026-06-11
**Status:** Approved (autonomous run — review requested post-hoc)
**Branch:** `worktree-highlights-ingestion` (worktree off `origin/main`)

## Goal

Feed a full highlights video (e.g. the 7-minute Liverpool 4-0 Barcelona reel in
`test-media/`) straight into the pipeline. `prepare_shots` must:

1. split the reel into individual shots,
2. group shots that cover the same highlight (live action + its replays from
   other angles),
3. drop "reaction" shots (crowd, bench, manager close-ups) — players only,
4. auto-align shots within a highlight group on a shared timeline,

and the dashboard must let the operator review groups, discard shots or whole
groups, and inspect/correct the temporal alignment on a timeline — all
non-destructively.

## Current state

- `prepare_shots` only copies pre-trimmed clips into `output/shots/` and merges
  a flat `shots_manifest.json` (`Shot` already carries `speed_factor`).
- A **deleted** stage (`src/stages/segmentation.py`, recoverable at
  `262d08a~1`) already did PySceneDetect splitting, pitch-ratio reaction
  filtering, fade-transition detection, short-span merging and ffmpeg
  extraction. `scenedetect[opencv]` is still a dependency.
- The **deleted** old `prepare_shots.py` had zoom-invariant speed-factor
  estimation (LK optical flow normalised by Sobel gradient density).
- `SyncMap` (`output/shots/sync_map.json`) stores manual per-shot frame
  offsets against a single global reference; the dashboard's Shot Sync editor
  (side-by-side scrub videos + draggable timeline) edits it. No pipeline stage
  consumes it yet.
- Downstream stages iterate `manifest.shots` directly.
- Dashboard is a vanilla-JS single file (`index.html`, ~5.8k lines) with an
  upload→background-job flow (`/api/shots/upload`, `Job` threads, log
  streaming, `attachToJob`).

## Approaches considered

**Shot splitting** — (a) PySceneDetect `AdaptiveDetector` *(chosen: dependency
already present, proven deleted-stage code to resurrect; adaptive variant
tolerates broadcast pans/zooms better than plain content threshold)*;
(b) TransNetV2 deep shot detector (better on dissolves, but a new torch model
+ weights for marginal gain); (c) hand-rolled HSV histogram differencing
(reinventing scenedetect).

**Reaction filtering** — (a) pitch-ratio heuristics on sampled frames
*(chosen: deleted stage proved the approach; crowd/bench/manager shots have
low green fraction)*; (b) YOLO person-scale signals (already a dep, but adds
model load/GPU time for little gain over (a) — revisit if (a) misclassifies
on the test reel); (c) CLIP zero-shot classification (heavy new dependency).
Close-ups of *players on the pitch* stay (grass behind them passes the pitch
test) — the operator can discard them in the UI.

**Highlight grouping** — (a) rule-based boundaries on the ordered kept shots
*(chosen — see rules below; transparent, tunable, operator-correctable)*;
(b) visual-similarity clustering (different angles of the same event don't
look alike — weak signal); (c) scoreboard-clock OCR (strong signal but a new
OCR dependency, and replays hide the clock — exactly where grouping matters).

**Temporal alignment** — (a) motion-energy profile cross-correlation after
speed normalisation *(chosen: viewpoint-tolerant 1-D signal, cheap, honest
confidence score)*; (b) align shot ends only (replays usually end just after
the key moment — used as the low-confidence fallback); (c) audio
cross-correlation (useless here: a reel's audio is continuous commentary, it
does not repeat with replays); (d) tracking-based alignment à la the old
two-pass sync graph (most accurate, but needs stage-2+ outputs; noted as a
future refinement that would *update* the same SyncMap).

## Stage design

`prepare_shots` gains a **split mode**. Mode resolution
(`prepare_shots.mode: auto | copy | split`, default `auto`): split when the
input is a single video whose duration ≥ `split.min_input_duration_s`
(default 90 s); otherwise copy (existing behaviour, untouched — including
directory inputs, dashboard "Add Shots" merging and legacy migration).

Split-mode pipeline (new pure modules, stage orchestrates):

1. **Detect spans** — `src/utils/shot_split.py`: PySceneDetect
   (`AdaptiveDetector` default, `ContentDetector` fallback via config) →
   `ShotSpan(start_frame, end_frame, start_s, end_s)`; merge sub-second
   false cuts (recovered `_merge_adjacent_short_spans` logic); drop spans
   shorter than `min_shot_duration_s` (default 1.0 s).
2. **Extract features** — `src/utils/shot_features.py`: sample ≤5 frames per
   span at fixed fractions; compute median/peak pitch ratio (HSV green mask),
   brightness min/range (fade detection), zoom-invariant motion rate
   (LK flow magnitude ÷ Sobel gradient density, recovered logic). Reference
   motion rate = median rate over wide gameplay spans → per-span
   `speed_factor = rate_ref / rate_span`. Classify:
   - `kind`: `transition` (fade/black) | `reaction` (low pitch ratio) |
     `gameplay` (everything else),
   - `scale`: `wide` | `medium` | `tight` (pitch-ratio bands; UI badge +
     grouping signal),
   - `is_replay`: `speed_factor ≥ replay_min_speed_factor` (default 1.25).
3. **Group** — `src/utils/highlight_grouping.py`: walk kept (gameplay) shots
   in reel order; start a new group before shot *i* when any rule fires:
   - **R1** a dropped `transition` shot sits between *i−1* and *i*
     (confidence 0.9),
   - **R2** source-time gap from dropped spans > `gap_boundary_s`
     (default 5 s; confidence 0.6),
   - **R3** shot *i* is wide + real-time AND the current group already
     contains a replay (confidence 0.75) — "replays finished, back to live".
   Groups get ids `g01…`, labels `Highlight N`, and a reference shot (first
   wide real-time member, else longest member).
4. **Extract clips** — new `extract_clip_reencode()` in `src/utils/ffmpeg.py`
   (libx264 CRF 18, preset fast, AAC audio kept for the sync editor;
   frame-accurate, unlike the keyframe-snapped stream-copy `extract_clip`).
   Slow-motion shots with `|speed_factor − 1| > 0.15` are retimed to real
   time (`setpts=PTS/sf`, `-r fps`) so every stored clip is real-time — the
   sync editor, GVHMR and the camera stage all assume that. Shot ids
   `s001…`; thumbnails to `shots/thumbs/<id>.jpg` (existing
   `extract_thumbnail`).
5. **Align within groups** — `src/utils/shot_alignment.py`: per extracted
   clip, motion-energy curve = mean |gray frame-diff| at ~192 px width,
   Gaussian-smoothed; normalised cross-correlation against the group
   reference over all lags with ≥1 s overlap. Offset uses the existing sign
   convention (`frame_offset = frame_in_shot − frame_in_reference`).
   `confidence = max(0, peak NCC)`; `method = "motion_profile"` when
   confidence ≥ 0.5, else `"low_confidence"` with an align-ends fallback
   offset. Auto results never overwrite saved `manual` alignments.
6. **Write artefacts** — `shots_manifest.json` (shots + groups),
   `sync_map.json` (v2, group-scoped), `shots/shot_features.json`
   (per-shot diagnostics sidecar: pitch ratio, motion rate, speed factor,
   scale, NCC, boundary rule), thumbnails.

Reaction/transition shots are **still extracted** but marked excluded — the
dashboard's dropped tray needs the preview, and restore must be instant.
Idempotency: when the manifest already contains shots for this
`source_file`, split mode is a no-op (re-ingest via `--clean` or a new file).

## Schema changes

`Shot` (additive, defaults keep old manifests loading):
`kind: str = "gameplay"`, `excluded: bool = False`, `exclude_reason: str = ""`,
`group_id: str = ""`, `source_start_s: float = -1.0`,
`source_end_s: float = -1.0`.

`ShotsManifest`: new `groups: list[HighlightGroup]`
(`id`, `label`, `shot_ids`, `boundary_rule`, `boundary_confidence`) and an
`active_shots()` helper (non-excluded). **Every** downstream stage that
iterates `manifest.shots` switches to `active_shots()` (tracking, camera,
hmr_world, refined_poses, ball, export, plus server shot listings).
Manually-added clips keep `group_id = ""` (rendered as "Ungrouped").

`SyncMap` v2: `{version: 2, groups: [{group_id, reference_shot,
alignments: [Alignment…]}]}`. Loader migrates v1 (flat
`reference_shot`/`alignments`) into a single `group_id = ""` entry.
`motion_profile` joins the valid methods. Offsets remain meaningful only
within a group.

## Config (`config/default.yaml`)

```yaml
prepare_shots:
  expected_format: mp4
  output_fps: null
  mode: auto                      # auto | copy | split
  split:
    min_input_duration_s: 90
    detector: adaptive            # adaptive | content
    threshold: 27.0               # content detector
    adaptive_threshold: 3.0
    min_scene_len_frames: 13
    min_shot_duration_s: 1.0
    merge_max_gap_s: 0.08
    merge_short_shots_max_duration_s: 1.2
  classify:
    sample_points: [0.15, 0.3, 0.5, 0.7, 0.85]
    reaction_max_median_pitch_ratio: 0.12
    reaction_max_peak_pitch_ratio: 0.20
    wide_min_pitch_ratio: 0.40
    tight_max_pitch_ratio: 0.22
    fade_black_frame_threshold: 0.18
    fade_min_brightness_range: 0.25
    replay_min_speed_factor: 1.25
    speed_normalise_threshold: 0.15
  group:
    gap_boundary_s: 5.0
  align:
    enabled: true
    curve_width_px: 192
    smooth_sigma_frames: 2.0
    min_overlap_s: 1.0
    min_confidence: 0.5
```

## Server API

- `POST /api/shots/upload-reel` — single video upload → saved to
  `output/source/<sanitised>.mp4` → spawns a `prepare_shots` job.
  `RunRequest` gains optional `input_path`, validated to live under
  `output/source/`; `_run_job` passes it through as `video_path`.
- `PATCH /api/shots/bulk` — `{updates: [{shot_id, excluded?, exclude_reason?,
  group_id?}]}` under the existing manifest lock. Covers single discard,
  restore, move-between-groups, merge (move all), split (move tail to a new
  id) and group discard. The server reconciles `manifest.groups` afterwards:
  drops emptied groups, creates records for new ids (next free `gNN`), keeps
  labels/order stable, and prunes sync-map entries of moved shots.
- `POST /api/sync/auto` — `{group_id, force?: false}` recomputes alignment
  for one group; `manual` entries survive unless `force`.
- `GET /api/shots/features` — serves the diagnostics sidecar.
- `GET /api/shots/{id}/thumb` — thumbnail (on-demand generation fallback for
  legacy clips).
- `GET/POST /api/sync` — upgraded to the v2 group payload (legacy files
  migrate on read).
- `GET /api/output/shots` — now manifest-aware: lists *active* shots so
  excluded clips disappear from tracking/camera/viewer dropdowns (glob
  fallback when no manifest exists). The prepare-shots panel reads the full
  manifest instead.

## Dashboard UX

The prepare-shots panel moves out of `index.html` into
`src/web/static/js/prepare_shots_panel.js` (plain script include — keeps the
no-framework idiom while respecting file-size limits; `index.html` keeps a
thin call-through). Layout, top to bottom:

1. **Ingest card** — drag-and-drop / file-pick a full reel →
   `upload-reel` → job progress via existing `attachToJob` log streaming;
   "Add Shots" multi-clip upload stays alongside.
2. **Highlight groups board** — one card per group in reel order: header
   (`Highlight 1 · 4 shots · 0:12–0:44`, boundary-confidence chip, discard-
   group ✕, "Re-align" button), horizontal shot-tile strip. Each tile:
   thumbnail (hover = video preview), id, duration, badges (`WIDE/TIGHT`,
   `REPLAY ×1.8`, alignment method/confidence), discard ✕, and
   drag-to-another-group (HTML5 DnD) with ◀/▶ move buttons as the
   keyboard-accessible fallback; "⋮" menu: *make reference*, *split group
   here*, *move to new group*. Ungrouped shots render in an "Ungrouped" card.
3. **Dropped tray** — collapsed list of excluded shots (reason badge:
   `reaction`, `transition`, `manual`) with thumbnails and *Restore*.
4. **Group sync editor** — the existing two-video scrub + draggable timeline
   + nudge/lock/play-both controls, now **scoped to a selected group** (group
   tabs above it). Per-shot rows show method + confidence
   (`auto 0.82` / `manual`); editing an offset marks it `manual`. Save posts
   the v2 payload for that group only.

All mutations go through `PATCH /api/shots/bulk` and re-render
optimistically with toast feedback, consistent with the existing dark theme
and helpers (`makePanel`, `makeToolbarBtn`, …).

## Quality report

`quality_report.json` gains a `prepare_shots` section: shot/drop counts by
reason, group count, per-group alignment confidence, list of low-confidence
groups (the "needs operator attention" surface).

## Error handling

- Unreadable input / zero scenes detected → stage error with actionable
  message (suggest `detector: content`, lower threshold).
- ffmpeg extraction failure for one span → warn, skip that shot, continue
  (matches old stage).
- Speed estimation degenerate (static scene, <2 frames) → `speed_factor 1.0`.
- Alignment failure (curve too short, no overlap) → `low_confidence` +
  align-ends fallback; never blocks the stage.
- Bulk PATCH validates shot ids and group-id shape (`^[A-Za-z0-9_-]{1,32}$|^$`)
  and rejects unknown ids with 400 before writing anything.

## Testing

- **Unit** (no GPU, synthetic fixtures): scenedetect wrapper on a generated
  cut video (cv2 `VideoWriter`, solid-colour segments); pitch/brightness/
  motion features on synthetic green vs crowd-textured frames; speed factor
  on frame-duplicated slow-mo; grouping rules as pure-function table tests;
  NCC alignment on shifted synthetic curves and shifted synthetic clips;
  schema round-trips + v1→v2 sync migration + legacy manifest load;
  `extract_clip_reencode` (ffmpeg available locally).
- **Integration**: full stage run on a small synthetic reel (cuts, green
  "gameplay", dark "crowd", slow-mo replay segment) asserting manifest
  groups/exclusions/sync map; idempotent re-run; FastAPI TestClient coverage
  of every new/changed endpoint (mirroring `test_web_api.py` patterns).
- **E2E (manual)**: Liverpool reel through split mode; compare grouping
  against the hand-trimmed `origi01–04` ground truth; dashboard review on
  port 8001.

## Out of scope

Scoreboard OCR, deep replay detectors, audio alignment, tracking-based
alignment refinement (future stage updates the same SyncMap), cross-group
dedup, automatic multi-shot pose convergence.
