# Multi-shot pipeline plumbing

**Date**: 2026-05-08
**Status**: Approved — moving to implementation plan

## Goals

1. `recon.py run --input <path>` accepts either a single .mp4 or a directory of .mp4s. Directory → one shot per file.
2. The camera, ball, and export stages process each shot independently, producing per-shot artefacts.
3. The hmr_world stage routes each player track through *its shot's* camera track.
4. The anchor editor's clip-select dropdown drives where anchors are saved/loaded — currently it switches the editor's video but anchors are shared across clips, which is the wiring bug this work fixes.
5. The 3D viewer can select which shot's scene to load.
6. Replace the anchor editor's "← Dashboard" link with a "Rerun camera tracking" button that auto-saves anchors and re-runs the camera stage for the selected shot only.

## Non-goals (this work)

- Cross-shot player re-identification. A player visible in multiple shots gets a different `player_id` per shot. (Followup spec.)
- Cross-shot HMR fusion / merging poses across shots. Each shot's reconstruction is independent. (Followup spec.)
- Automatic scene segmentation. The user keeps trimming clips manually in CapCut and dropping them into the input directory.
- A single combined GLB stitching all shots. One GLB per shot — viewer picks.

## Architecture

### File layout

```
output/
├── shots/
│   ├── shots_manifest.json
│   ├── {shot_id_a}.mp4
│   ├── {shot_id_b}.mp4
│   └── …
├── tracks/
│   ├── {shot_id_a}_tracks.json     (already per-shot)
│   └── {shot_id_b}_tracks.json
├── camera/
│   ├── {shot_id_a}_anchors.json
│   ├── {shot_id_a}_camera_track.json
│   ├── {shot_id_b}_anchors.json
│   ├── {shot_id_b}_camera_track.json
│   └── debug/                       (per-anchor overlays — debug-camera flag)
├── hmr_world/
│   ├── {player_id}_smpl_world.npz   (player_id is per-shot; SmplWorldTrack
│   ├── {player_id}_kp2d.json         carries shot_id internally)
│   └── …
├── ball/
│   ├── {shot_id_a}_ball_track.json
│   └── {shot_id_b}_ball_track.json
└── export/
    ├── gltf/
    │   ├── {shot_id_a}_scene.glb
    │   ├── {shot_id_a}_scene_metadata.json
    │   ├── {shot_id_b}_scene.glb
    │   └── {shot_id_b}_scene_metadata.json
    └── fbx/
        ├── {shot_id_a}/
        └── {shot_id_b}/
```

### Key invariants

- `shot_id` = bare filename stem of the input clip, sanitised (alphanumeric + `_-`, max 64 chars).
- One `shot_id` links shot videos, anchors, camera tracks, ball tracks, and GLBs throughout the tree. Dashboard URLs and APIs use this same id.
- A stage's `is_complete` returns `True` only when *every* shot in the manifest has its expected output. Adding a new shot mid-project re-runs downstream stages for that shot only.

### Schema fields touched

All already exist on the dataclasses; this work just enforces them as routing keys:

- `AnchorSet.clip_id` → equals `shot_id`.
- `CameraTrack.clip_id` → equals `shot_id`.
- `SmplWorldTrack.shot_id` → equals `shot_id`.
- `BallTrack.clip_id` → equals `shot_id`.

## Components

### CLI / `prepare_shots`

```python
clip_src = Path(self.video_path).resolve()
if clip_src.is_dir():
    clip_files = sorted(clip_src.glob("*.mp4"))   # depth-1 only
    if not clip_files:
        raise FileNotFoundError(f"no .mp4 files in {clip_src}")
else:
    clip_files = [clip_src]

for clip in clip_files:
    shot_id = _sanitise_shot_id(clip.stem)
    # … existing per-clip copy + manifest entry …

# manifest.shots becomes a list with N entries instead of always 1.
```

**Validation**:
- Duplicate `shot_id`s across the input directory raise an error.
- Clips with zero frames still write a manifest entry but log a warning (matches current single-clip behaviour).
- `_sanitise_shot_id`: strip non-alphanumeric except `_-`, trim to 64 chars.

**Legacy migration**, run once at the start of `prepare_shots` if any of the legacy single-shot artefacts (`output/camera/anchors.json`, `output/camera/camera_track.json`, `output/ball/ball_track.json`, `output/export/gltf/scene.glb`) exists:
- Read `shots/shots_manifest.json` to find the existing single shot's id; fall back to `"clip"` if no manifest yet.
- Rename legacy files to their new per-shot names.
- Log `[prepare_shots] migrated legacy single-shot artefacts to per-shot layout under shot_id=…`.
- Idempotent — running twice is a no-op.
- If legacy artefacts exist for *different* unspecified shot_ids, refuse the migration and ask the user to resolve manually.

### `tracking`

Already iterates `manifest.shots` and writes `{shot_id}_tracks.json` per shot. **No code change.**

### `camera`

```python
class CameraStage(BaseStage):
    def is_complete(self) -> bool:
        manifest = ShotsManifest.load(...)
        return all(
            (self.output_dir / "camera" / f"{shot.id}_camera_track.json").exists()
            for shot in manifest.shots
        )

    def run(self) -> None:
        manifest = ShotsManifest.load(...)
        for shot in manifest.shots:
            self._run_shot(shot)

    def _run_shot(self, shot: Shot) -> None:
        anchors_path = self.output_dir / "camera" / f"{shot.id}_anchors.json"
        if not anchors_path.exists():
            logger.warning(
                "skipping shot %s — no anchors at %s. Open the anchor editor "
                "and place keyframes before re-running camera.",
                shot.id, anchors_path,
            )
            return
        # ...existing single-shot logic, parameterised on shot.id...
        track.save(self.output_dir / "camera" / f"{shot.id}_camera_track.json")
```

A shot without anchors is **skipped with a warning**, not a hard fail — partial pipelines (some shots anchored, others not) need to remain runnable. Quality report flags missing-anchor shots.

### `hmr_world`

Player-track collection across all shots is already correct. Add per-shot camera lookup:

```python
manifest = ShotsManifest.load(...)
camera_tracks_by_shot: dict[str, CameraTrack] = {}
for shot in manifest.shots:
    p = self.output_dir / "camera" / f"{shot.id}_camera_track.json"
    if p.exists():
        camera_tracks_by_shot[shot.id] = CameraTrack.load(p)
    else:
        logger.warning("hmr_world skipping shot %s — no camera track", shot.id)
```

When processing a player track, look up its shot's camera via the existing `group_shot[player_id]` mapping. Build per-shot `per_frame_K`, `per_frame_R`, `per_frame_t`, `distortion` from the right CameraTrack.

**Player-id collisions across shots**: ByteTrack numbers tracks 1..N within each clip, so `track_id=3` in shot A collides with `track_id=3` in shot B. Resolution: prefix unannotated tracks with the shot id —

```python
pid = track.player_id or f"{shot_id}_T{track.track_id}"
```

Once cross-shot re-id lands later, the user reassigns `player_name` in the dashboard and a stable `player_id` follows.

### `ball`

Same pattern as camera. Per-shot `_ball_track.json`. Reads its shot's camera_track for ground projection.

### `export`

Emits one GLB per shot:

```python
for shot in manifest.shots:
    cam = ... load shot's camera_track ...
    players = [p for p in all_players if p.shot_id == shot.id]
    ball = ... load shot's ball_track if present ...
    glb_bytes, metadata = build_glb(SceneBundle(...))
    (gltf_dir / f"{shot.id}_scene.glb").write_bytes(glb_bytes)
    (gltf_dir / f"{shot.id}_scene_metadata.json").write_text(json.dumps(metadata, indent=2))
```

`SmplWorldTrack` already has a `shot_id` field, so the player → shot grouping is just a filter.

### Dashboard — `anchor_editor.html`

The clip-select dropdown is already present; what changes is what it drives:
- Switching the dropdown loads `/anchors/{shot_id}` (new endpoint — replaces singular `/anchors`).
- Saving POSTs to `/anchors/{shot_id}`.
- The "← Dashboard" link at line 231 is replaced with a **"Rerun camera tracking"** button. Click flow:
  1. Auto-save current anchors via `POST /anchors/{shot_id}`.
  2. POST `/api/run-shot` with `{stage: "camera", shot_id}`.
  3. Show a small inline log panel with live tail.
  4. On success, button toggles to "Done — Open viewer for {shot_id}" linking to `/viewer?shot={shot_id}`.
  5. On failure, log shows the error and button reverts to "Rerun camera tracking".

A small "← Dashboard" link in the top-right corner provides escape navigation, so the user isn't trapped in the editor.

### Dashboard — `index.html`

Two surgical updates:
- The shot picker that's already present in the Tracking panel filters which shot's tracks/HMR/ball previews are shown in the panels below.
- A new "Multi-shot status" panel summarises per-shot completion:

```
Shot           Anchors  Camera   HMR     Ball    Export
match_1        ✓ 11     ✓        ✓ 23p   ✓       ✓
match_2        ✓ 8      ⚠ stale  ✗       ✗       ✗
half_2_open    ✗        ✗        ✗       ✗       ✗
```

`⚠ stale` flags when the camera track exists but its `mtime` predates the anchors' `mtime` — anchors edited since the camera was last solved.

### Dashboard — `viewer.html`

Add a shot picker to the controls bar, next to `Bones: OFF`:
- `<select id="shot-select">` listing all shots from `/api/output/shots`.
- On change, fetches `{shot_id}_scene.glb` instead of legacy `scene.glb`.
- Default selection: shot from URL query (`?shot=...`) if present, else first shot in manifest.

Browser URL kept in sync via `history.replaceState({}, '', '?shot=...')`.

### Server endpoints (`server.py`)

Three new endpoints + adjustments to two existing ones:

```
GET  /anchors/{shot_id}        → AnchorSet for that shot, or empty stub
POST /anchors/{shot_id}        → save AnchorSet to {shot_id}_anchors.json
POST /api/run-shot             → body: {stage: "camera", shot_id: "..."}
                                 Wipes that shot's stage artefacts and runs
                                 only that stage for only that shot.
```

The legacy `/anchors` singular endpoints stay for one release; they redirect to `/anchors/{first_shot_id}` and log a deprecation warning so existing clients keep working.

`POST /api/run-shot` reuses the existing job runner (same `Job` dataclass, `_run_job` helper) with two extra params: `target_stage` and `target_shot_id`. The runner calls `run_pipeline(stages=target_stage, shot_filter=target_shot_id)` after `run_pipeline` is updated to accept a shot-filter; each stage's `run()` becomes shot-aware via the existing `manifest.shots` iteration but skips shots not matching the filter.

## Validation, testing, edge cases

### Unit tests

| Test | Module | Asserts |
|---|---|---|
| `test_prepare_shots_directory_input` | `tests/test_prepare_shots.py` (new) | Tmp dir with 3 .mp4s yields a manifest with 3 shots, ordered by stem. Files copied; shot_ids match stems. |
| `test_prepare_shots_legacy_migration` | same | Pre-existing `output/camera/anchors.json` (no shot prefix) gets renamed to `{shot_id}_anchors.json` on first run; same for camera_track, ball_track, scene.glb. Idempotent. |
| `test_prepare_shots_duplicate_stems_raises` | same | Directory with two clips with the same stem raises a clear error. |
| `test_prepare_shots_sanitises_shot_id` | same | Clip named `"my clip [v2].mp4"` produces a sanitised shot_id without spaces or brackets. |
| `test_camera_stage_skips_shots_without_anchors` | `tests/test_camera_stage.py` | Manifest with 3 shots, anchors only for 2 → 2 camera_tracks written, 1 warning logged, no exception. |
| `test_hmr_world_routes_per_shot_camera` | `tests/test_hmr_world_stage.py` | Tracks across 2 shots, each with its own camera. Each player's foot anchor uses the right shot's K/R/t. |
| `test_export_emits_per_shot_glb` | `tests/test_runner.py` | 2 shots → 2 `{shot_id}_scene.glb` files, each containing only that shot's players + ball + camera. |
| `test_player_id_prefixed_with_shot_when_unannotated` | tracking-side | `track_id=3` in shot A and `track_id=3` in shot B don't collide; `pid = "{shot_id}_T3"` for each. |
| `test_run_shot_endpoint_filters_to_one_shot` | `tests/test_web_api.py` | `POST /api/run-shot` with `{stage: "camera", shot_id: "x"}` runs the camera stage and only that shot's `_camera_track.json` appears. |
| `test_anchors_per_shot_endpoint` | `tests/test_web_api.py` | `GET /anchors/x` returns `{shot_id}_anchors.json`; legacy `GET /anchors` redirects to first shot's. |

### Integration test

End-to-end: synthesise two minimal shots (re-use the existing `synthetic_clip` fixture, vary the yaw between the two), run the whole pipeline, assert two `scene.glb` files, two camera_tracks, and that switching shots in the viewer route loads the right one (server-side: `GET /api/output/shots` returns both).

### Edge cases handled

- Manifest exists but all shots have empty anchors → camera stage no-ops with warnings; downstream stages skip those shots.
- Adding a shot mid-project (drop one more .mp4 into the input dir, re-run prepare_shots) → only that shot's downstream artefacts are missing, the rest stay cached.
- A shot's anchors are edited after camera solved → stale camera flagged in the dashboard status panel.
- Legacy single-shot output dir → migrated once, transparently.
- Singular `/anchors` endpoint hit by an outdated client → redirects to first shot's per-shot endpoint, deprecation warning logged.

### Out of scope for validation

- Cross-shot consistency of recovered camera centres. Each shot is independent; if two shots are from the same physical camera, no test asserts their `camera_centre`s match — that's the next phase.
- Player identity across shots — designed for the followup re-id work.
