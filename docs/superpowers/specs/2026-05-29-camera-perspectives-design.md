# Camera Perspectives — Design

**Date:** 2026-05-29
**Status:** Approved (design); pending implementation plan

## Problem

The pipeline exports a single animated broadcast camera into the FBX/glTF output.
For UE5 Sequencer editorial we want additional, pre-populated camera angles so an
editor can drop ready-made perspective tracks onto the timeline instead of hand-rigging
them. Specifically: per-player **first-person POV** and **over-the-shoulder (OTS)**
cameras, derived from the reconstruction we already have.

Fixed angles (tactical top-down, corner, sideline) are intentionally **out of scope** —
those are easier to author directly in UE5 and the user prefers to keep them there.

## Goals

- Generate, per selected player, a **POV** camera and an **OTS** camera (always paired).
- Let the user choose **which players** get cameras **per shot**, from the web viewer's
  Export panel (not just a config file).
- Bake each generated camera into the FBX output (matching the broadcast camera path) and
  emit matching glTF camera nodes for web-viewer parity.
- Surface every camera in `ue_manifest.json` so the UE5 ingest can bind them.

## Non-Goals (YAGNI)

- Fixed-angle cameras (tactical/corner/sideline) — authored UE-side.
- Live POV/OTS preview in the browser viewer (no Blender) — possible follow-up.
- Auto-triggering the export job when a selection is saved — user re-runs export manually.
- Heuristic "ball carrier" auto-selection — selection is explicit.
- Extra POV jitter smoothing beyond what the head joint provides — add only if footage demands it.

## Decisions

| # | Decision | Rationale |
|---|----------|-----------|
| D1 | POV + OTS generated as a pair for each selected player | Matches user's editorial need; simplest mental model. |
| D2 | Player selection is **per shot**, persisted to disk | Player IDs differ per shot; UE manifest is already per-shot. |
| D3 | Selection edited in the web Export panel; applied on manual export re-run | Reuses existing `/api/run` flow; no new job-spawn path. |
| D4 | Cameras baked into FBX **and** emitted as `track_json` | Mirrors broadcast camera; UE EUW load path prefers `track_json`, FBX kept for parity. |
| D5 | Generated cameras reuse the existing `CameraTrack` schema | No new schema; Blender + glTF builders already understand it. |
| D6 | New `cameras: list[NamedCameraEntry]` field on the UE manifest; existing `camera` field unchanged | Backwards compatible — existing UE code reading `camera` (broadcast) keeps working. |

## Architecture

```
[Export panel checkboxes]
        │ PUT /api/export/camera-selection?shot={id}
        ▼
output/export/{shot_id}_camera_selection.json
        │
        │ user clicks Run → recon.py run --stages export
        ▼
ExportStage._generate_virtual_cameras()
  reads: selection JSON, per-shot SmplWorldTracks, BallTrack, config
  calls: src/utils/virtual_cameras.py  (pure functions, no I/O)
  writes: output/camera/{shot}_{player}_{pov|ots}_camera_track.json
        │
        ├─────────────────────────────┬───────────────────────────────┐
        ▼                             ▼                                 ▼
_export_gltf()              _export_fbx() (Blender)          write_ue_manifest()
  KHR_cameras node per rig    one FBX per rig under            cameras: [
                              output/export/fbx/                 {name:"broadcast", ...},
                                                                 {name:"P003_pov",  ...},
                                                                 {name:"P003_ots",  ...},
                                                               ]
```

### Component boundaries

- **`src/utils/virtual_cameras.py`** — pure math. Input: a `SmplWorldTrack`, optional
  `BallTrack`, a rig spec, and config; output: a `CameraTrack`. No file I/O, fully
  unit-testable.
- **`ExportStage`** (`src/stages/export.py`) — orchestration: read selection, call the
  generator, write per-rig `CameraTrack` files, then run the existing glTF/FBX/manifest steps.
- **`scripts/blender_export_fbx.py`** — generalise the current single "broadcast_camera"
  branch into a loop over all `*_camera_track.json` files for the shot (broadcast + rigs),
  naming each FBX after the file stem.
- **Web layer** (`src/web/server.py` + `src/web/static/index.html`) — selection CRUD and the
  picker UI. Never invokes generation directly; only writes the selection file.

## Camera Math (`virtual_cameras.py`)

Uses existing `src/utils/smpl_skeleton.py` and `src/utils/smpl_pitch_transform.py` for
forward kinematics into the pitch frame.

- **POV**: world position = SMPL head-joint world transform (FK from `root_R`, `root_t`,
  `thetas`, `betas`); orientation = head-joint rotation (camera looks where the head faces).
  Per-frame fallback to `root_t + fixed head offset` with `root_R` when head FK is unavailable
  for that frame.
- **OTS**: position = head pose translated by a configurable offset expressed in the head
  frame (default `forward -0.4 m`, `up +0.3 m`); orientation = look-at the **ball** world
  position when available. During short ball-occlusion gaps (≤
  `ball_target_max_occlusion_frames`), hold the last good target; if the ball is absent
  entirely, look along the player's forward direction.
- **Intrinsics**: share `image_size` with the broadcast camera; horizontal FOV per rig from
  config (`pov_fov_deg`, `ots_fov_deg`), converted to the `K` matrix the `CameraTrack`
  schema expects.
- **Confidence**: emit per-frame confidence using the same convention as the broadcast
  camera; frames that fell back to root-only POV are marked lower-confidence.

## Data Schemas

### Selection file — `output/export/{shot_id}_camera_selection.json`

```json
{
  "shot_id": "shot_01",
  "selections": [
    {"player_id": "P003", "rigs": ["pov", "ots"]},
    {"player_id": "P012", "rigs": ["pov", "ots"]}
  ]
}
```

`rigs` is a list (not a pair of booleans) to allow per-rig opt-out later without a schema
break. Valid rig values: `"pov"`, `"ots"`.

### UE manifest addition

New field `cameras: list[NamedCameraEntry]` where `NamedCameraEntry` carries:
`name` (e.g. `"broadcast"`, `"P003_pov"`, `"P003_ots"`), `track_json` (relative path),
`fbx` (relative path, may be empty when Blender unavailable), `image_size`, `frame_range`.
The broadcast camera is included in this list **and** retained in the existing scalar
`camera` field for backwards compatibility.

### Config — `config/default.yaml`

```yaml
export:
  virtual_cameras:
    pov_fov_deg: 75.0
    ots_fov_deg: 60.0
    ots_offset_m: {forward: -0.4, up: 0.3, right: 0.0}
    ball_target_max_occlusion_frames: 10
```

## New Endpoints (`src/web/server.py`)

| Method | Path | Behaviour |
|--------|------|-----------|
| `GET` | `/api/export/available-players?shot={id}` | Returns `player_id`s with SMPL data for the shot, plus display names from `players.json`. Populates the checkbox list. |
| `GET` | `/api/export/camera-selection?shot={id}` | Returns the selection JSON; `{"shot_id": id, "selections": []}` when no file exists. |
| `PUT` | `/api/export/camera-selection?shot={id}` | Validates each `player_id` exists in the shot and each rig is known; writes the file atomically. |

### Export panel UI (`index.html`)

In `renderExport`, add a "Perspective cameras" block: one row per available player with two
checkboxes (POV, OTS). On save (PUT), show a banner: *"Selection saved. Re-run the Export
stage to generate the FBX cameras."* No live preview in v1.

## Error Handling

- Missing selection file → no virtual cameras generated; broadcast camera unchanged.
- `player_id` in selection but no SMPL track in the shot → log warning, skip that entry.
- Head-FK failure on some frames → per-frame fallback to root + fixed offset; mark lower confidence.
- Ball missing entirely → OTS look-at uses player forward direction; log once per rig.
- Blender unavailable → existing behaviour: skip FBX, keep glTF + `track_json` + manifest.

## Testing

- **Unit** `tests/test_virtual_cameras.py` — POV R/t at a synthetic head pose; OTS look-at-ball
  orientation; OTS fallback when ball missing; POV head-FK fallback to root when `thetas` absent.
- **Unit** `tests/test_camera_selection_schema.py` — selection JSON round-trip; validation
  rejects unknown rig types.
- **Integration** `tests/test_export_virtual_cameras.py` — fixture shot with two players + a
  selection emits the expected per-rig `CameraTrack` files, a `ue_manifest.json` with the right
  `cameras` list, and glTF metadata with the extra camera nodes.
- **API** `tests/test_web_api_camera_selection.py` — GET/PUT round-trip; rejects unknown shot;
  rejects unknown player.
- **Blender** — no new Blender-dependent test; add a unit check that the camera-track glob picks
  up rig files in addition to the broadcast. FBX path stays covered by existing smoke runs.

## Open Questions

None blocking. FOV defaults and the OTS offset are first guesses to be tuned against real
footage after the first export.
