# UE5 Base — Pitch Level, Player BP, Reconstruction Ingest

**Date:** 2026-05-08
**Status:** Draft (pending user review)
**Scope:** Establish the Unreal Engine 5.7 project base for `football-perspectives`. Extend the pipeline export to produce UE-ready skeletal FBX, set up SMPL→UE5 Mannequin retargeting in UE, build the pitch level + generic player Blueprint, and add an editor utility that ingests a clip into a per-clip `LevelSequence` containing one mannequin per player, the ball, and the broadcast camera.

## 1. Background

The Python pipeline (`recon.py`) reconstructs per-player SMPL animation, ball trajectory, and camera pose from a single broadcast clip and writes them to `output/`. The end goal stated in `docs/football-reconstruction-pipeline-design.md` is to recreate the play in UE5 with a virtual camera matching the real camera, players animated as mannequins (and eventually MetaHumans) driven by the SMPL output, and the ball as a rigid body.

The current export stage (`src/stages/export.py`, `scripts/blender_export_fbx.py`) is intentionally minimal: player FBX files contain a single-bone armature with **root translation only** — no joint rotations. Importing those today produces T-pose mannequins sliding around the pitch. The web viewer renders capsules, not animated bodies. This spec closes that gap.

## 2. Goals

1. Player FBX exports contain the **canonical 24-joint SMPL skeleton with per-joint animation** (parent-relative joint rotations + root translation).
2. A sibling UE5.7 project contains:
   - A pitch level (`L_PitchMaster`) with a FAB-marketplace pitch asset placed at world origin.
   - A generic `BP_PlayerActor` using `SK_Mannequin` (UE5 Manny/Quinn).
   - SMPL→UE5 Mannequin IK Rigs and IK Retargeter, authored once and checked in.
3. An editor utility (`EUW_LoadReconstruction`) ingests a pipeline output directory and authors a per-clip `LevelSequence_<clip>` containing player tracks, a ball track, and a `CineCameraActor` driven by the broadcast camera.
4. Hitting Play on the imported sequence reproduces the broadcast frame: mannequins moving on the pitch, ball trajectory, broadcast camera view.

### Non-goals (v1)

- MetaHuman support (designed-for, not built-for).
- Skinned SMPL mesh in the FBX (UE consumes skeleton + animation only).
- Runtime ingestion or packaged-build loading of reconstructions.
- Team detection / jersey assignment from the pipeline (manual in UE).
- Rendering output / cinematic post-process polish.
- Automated UE-side asset tests.

## 3. Architecture

Three layers connected by a versioned manifest contract.

```
┌─────────────────────────────┐    ┌─────────────────────────────┐
│ football-perspectives       │    │ football-perspectives-ue    │
│ (this repo, Python)         │    │ (sibling repo, UE5.7)       │
│                             │    │                             │
│ recon.py run …              │    │ Editor: EUW_LoadReconstruction
│  → output/export/fbx/*.fbx  ├────┤  reads manifest             │
│  → output/export/ue_manifest│    │  imports + retargets        │
│    .json                    │    │  authors LS_<clip>          │
└─────────────────────────────┘    └─────────────────────────────┘
        producer                            consumer
```

- **Pipeline side** owns the FBX format and the manifest schema.
- **UE side** owns the level, BP, retarget assets, and ingest tooling.
- The two never share code; they only share the manifest contract documented in §6.

## 4. Pipeline-side changes

### 4.1 Full SMPL skeleton in player FBX

`scripts/blender_export_fbx.py` currently keyframes only `arm.location` (root translation). The upgrade builds a real 24-joint SMPL armature and bakes per-joint rotations alongside the root translation.

**Joint hierarchy (24 SMPL joints):**

```
pelvis ─┬─ l_hip ── l_knee ── l_ankle ── l_foot
        ├─ r_hip ── r_knee ── r_ankle ── r_foot
        └─ spine1 ── spine2 ── spine3 ─┬─ neck ── head
                                       ├─ l_collar ── l_shoulder ── l_elbow ── l_wrist ── l_hand
                                       └─ r_collar ── r_shoulder ── r_elbow ── r_wrist ── r_hand
```

**Rest pose:** joint rest positions come from SMPL's `J_regressor @ V_template` (mean-pose template), available via the GVHMR submodule's bundled SMPL body model. Rest positions are computed once and cached.

**Per-frame data:** for each frame `f` of `*_smpl_world.npz`:
- `global_orient[f]` (3,) — pelvis root rotation in axis-angle.
- `body_pose[f]` (23, 3) — non-root joint rotations in axis-angle, **parent-relative** (this is SMPL's native convention).
- `root_t[f]` (3,) — pelvis world translation in pitch metres.

**Encoding into FBX:**
- Root: keyframe `arm.location = root_t[f]` and pelvis bone rotation = `axis_angle_to_quat(global_orient[f])`.
- Other 23 joints: keyframe pose-bone rotation = `axis_angle_to_quat(body_pose[f, j])`.
- These are FBX-standard parent-relative joint rotations. Blender is the FBX serializer; the rotation semantics are not Blender-specific.
- No skinned mesh is added — the FBX contains the armature + animation only.

**Why Blender, not direct FBX SDK or glTF:** the project already runs Blender headless for FBX export; `bpy.ops.export_scene.fbx` is well-tested and idiomatic. Direct FBX SDK Python bindings are awkward to install. glTF skeletal import works in UE5 but the FBX path is more thoroughly trodden for MoCap.

### 4.2 Manifest schema

A new artefact `output/export/ue_manifest.json` is written at the end of the export stage. It is the contract between the pipeline and the UE editor utility.

```json
{
  "schema_version": 1,
  "clip_name": "<derived from input filename, sanitised for use as folder name>",
  "fps": 30,
  "frame_range": [0, 149],
  "pitch": {"length_m": 105.0, "width_m": 68.0},
  "players": [
    {
      "player_id": "P001",
      "fbx": "fbx/P001.fbx",
      "frame_range": [0, 149],
      "world_bbox": {"min": [-12.3, -8.1, 0.0], "max": [10.8, 6.5, 1.95]}
    }
  ],
  "ball":   {"fbx": "fbx/ball.fbx",   "frame_range": [12, 78]},
  "camera": {"fbx": "fbx/camera.fbx", "image_size": [1920, 1080], "frame_range": [0, 149]}
}
```

- All paths in the manifest are **relative to `output/export/`**.
- `world_bbox` lets the UE side warn if anything escapes pitch bounds.
- `ball` and `camera` are optional. `players` is non-empty (otherwise the export stage fails earlier).

**Implementation:** `src/schemas/ue_manifest.py` defines a `UeManifest` dataclass with `load`/`save` methods and validation. The export stage instantiates it after writing FBX files and only includes entries for FBX files that were successfully written. The manifest is written **iff at least one player FBX was written successfully**; if zero players succeeded (e.g. Blender missing), no manifest file is produced and the UE side reports "no manifest at `<path>`".

### 4.3 Coordinate frames

Pipeline coordinate convention (per `CLAUDE.md`):
- `+X` along the nearside touchline.
- `+Y` across the pitch toward the far touchline.
- `+Z` up; ground plane at `z = 0`.
- Units: metres.
- FIFA standard pitch 105m × 68m, origin at pitch centre.

Existing FBX export uses `axis_forward="-Y", axis_up="Z", global_scale=1.0`. Under those flags, a pitch-frame `(x, y, z)` lands directly into UE world coordinates without sign flips, given UE's left-handed `+X forward, +Y right, +Z up` world. A player at pitch `(0, 0, 0)` lands at UE world `(0, 0, 0)` (pitch centre, on the ground).

The exact axis interpretation depends on Blender's source convention (right-handed, X-right, Y-forward, Z-up) being remapped at FBX export to UE's left-handed world; the FBX axis flags above encode that conversion. Pipeline unit test P-T2 asserts a synthetic `root_t` round-trips through the FBX export within 1mm — that test, not the prose, is the authoritative spec for the conversion.

### 4.4 Pipeline tests

| ID | Type | What |
|---|---|---|
| P-T1 | Unit | `tests/test_ue_manifest.py` — `UeManifest` validates required fields, rejects bad `schema_version`, rejects non-finite numbers, accepts optional ball/camera. |
| P-T2 | Unit (`-m fbx`) | `tests/test_blender_export_smpl_skeleton.py` — synthetic 5-frame `SmplWorldTrack`; run Blender export headless; reopen FBX in a follow-up Blender step; assert (a) 24 bones with correct parenting, (b) keyframe count == frame count, (c) root translation round-trips within 1mm, (d) a chosen joint's quaternion round-trips within 1e-4. |
| P-T3 | Integration (`-m fbx`) | Full export run on a fixture clip; assert manifest validates and references all written FBX files. |
| P-T4 | Smoke | Manifest writer test that bypasses Blender (uses fake FBX paths) — runs in normal CI. |

## 5. UE5-side components

Project lives in sibling repo `football-perspectives-ue/`. The UE binaries (`Binaries/`, `Intermediate/`, `Saved/`, `DerivedDataCache/`) are gitignored; `.uproject`, `Content/`, `Config/` are tracked.

### 5.1 Project layout

```
Content/FootballPerspectives/
├── Maps/
│   └── L_PitchMaster.umap
├── Pitch/                               # FAB asset placed here (manual)
├── Skeleton/
│   ├── SK_SMPL.uasset                   # imported from a reference SMPL FBX
│   ├── IK_SMPL.uasset
│   ├── IK_Mannequin.uasset
│   └── IKR_SMPL_to_Mannequin.uasset
├── Player/
│   ├── BP_PlayerActor.uasset
│   ├── BP_BallActor.uasset
│   └── M_PlayerJersey.uasset
├── Reconstructions/                     # editor utility writes here
│   └── <clip_name>/
│       ├── Players/                     # imported per-track AnimSequences (SMPL)
│       ├── PlayersRetargeted/           # retargeted AnimSequences (SK_Mannequin)
│       ├── Ball/
│       ├── Camera/
│       └── LS_<clip_name>.uasset        # the per-clip LevelSequence
├── Editor/
│   ├── EUW_LoadReconstruction.uasset
│   └── DA_FootballPerspectivesSettings.uasset   # holds path to pipeline output dir
```

### 5.2 BP_PlayerActor

- Components: root `SceneComponent`; `SkeletalMeshComponent` with `SK_Mannequin` (UE5 Manny/Quinn) and anim mode "Use Animation Asset".
- Instance variables (Sequencer-exposed): `TeamColor` (FLinearColor), `Number` (int), `PlayerId` (FName, read-only label).
- `M_PlayerJersey` reads `TeamColor` and `Number` via dynamic material instance set in the construction script.
- No movement logic in the BP — root motion comes entirely from the AnimSequence.

### 5.3 SMPL → UE5 Mannequin retargeting

One-time setup, checked into the UE repo:

1. **Import reference SMPL skeleton.** Pick any clip's `P001.fbx`, import as Skeletal Mesh into `Content/FootballPerspectives/Skeleton/`. UE creates `SK_SMPL`. (No mesh — UE handles armature-only FBX as a skeleton.)
2. **Author `IK_SMPL`** (IK Rig over `SK_SMPL`) with chains:
   - `Root` — pelvis
   - `Spine` — pelvis → spine1 → spine2 → spine3 → neck → head
   - `LeftArm` — l_collar → l_shoulder → l_elbow → l_wrist
   - `RightArm` — mirror
   - `LeftLeg` — l_hip → l_knee → l_ankle → l_foot
   - `RightLeg` — mirror
3. **Author `IK_Mannequin`** (IK Rig over `SK_Mannequin`) with the same chain names mapped to UE bone names: `pelvis`, `spine_01`..`spine_05` + `neck_01` + `head`, `clavicle_l`/`upperarm_l`/`lowerarm_l`/`hand_l`, etc.
4. **`IKR_SMPL_to_Mannequin`** (IK Retargeter): source = `IK_SMPL`, target = `IK_Mannequin`. Auto-aligns chains by name. Manual tuning: pelvis height offset and per-chain rotation offsets via the retarget pose, until a known reference frame produces an upright mannequin with feet at `z ≈ 0`.

Acceptance: a known clip's reference frame, retargeted, places the mannequin on the pitch with feet within ±5cm of `z = 0`.

### 5.4 EUW_LoadReconstruction (Editor Utility)

UI: a single "Load Reconstruction" button + a path field defaulting from `DA_FootballPerspectivesSettings`. Implementation is **Editor Utility Widget shell with a Python backend** (UE5 Editor Scripting Utilities + Python plugin). LevelSequence authoring and asset import are far more concise in `unreal.LevelSequence` Python than in Blueprint.

Per click, the Python script:

1. Reads `<output_dir>/export/ue_manifest.json`. Validates `schema_version`, fps, presence of all referenced FBX files. Aborts with a clear message on failure.
2. Reads `clip_name` from the manifest. Wipes `Content/FootballPerspectives/Reconstructions/<clip_name>/` if it already exists (idempotent re-import).
3. For each player FBX:
   - Imports as Skeletal Animation onto `SK_SMPL` → `Players/<player_id>_anim`.
   - Runs `IKR_SMPL_to_Mannequin` to produce `PlayersRetargeted/<player_id>_anim_retargeted` (target skeleton `SK_Mannequin`).
4. Imports `ball.fbx` as a transform-only animation; imports `camera.fbx` as a Camera Anim Sequence.
5. Creates `LS_<clip_name>` (`LevelSequence`):
   - Sets fps and frame range from the manifest.
   - For each player: adds a `BP_PlayerActor` **Spawnable** to the sequence, binds its `SkeletalMeshComponent.AnimationData.AnimToPlay` to the retargeted anim, sets `PlayerId`, leaves `TeamColor`/`Number` blank for v1. Spawnables (not Possessables) are correct here — these actors only exist while the sequence plays.
   - Adds a `BP_BallActor` Spawnable bound to the ball anim's root transform. `BP_BallActor` is a simple BP with a sphere static mesh component (radius 0.11m).
   - Adds a `CineCameraActor` Spawnable bound to the camera anim. Focal length comes from the camera FBX (already converted from `K`/`image_size` at export).
6. Assigns `LS_<clip_name>` to a `LevelSequenceActor` named `LSA_ActiveClip` already present in `L_PitchMaster`, replacing whatever was there.
7. Prints a summary report: player count, frame range, any `world_bbox` warnings.

### 5.5 L_PitchMaster

Saved level containing:
- The FAB pitch asset (manually placed at world origin; touchline aligned to `+X`).
- Default lighting and sky.
- One `LevelSequenceActor` named `LSA_ActiveClip`, sequence asset initially unset.
- Optional `BP_DebugOverlay` (off by default) that draws pitch-coordinate axes at origin for alignment verification.

### 5.6 Why Editor Utility, not runtime spawner

LevelSequence assets are edit-time constructs; building them at runtime fights the engine. Authoring at edit time means scrubbing, tweaking, and rendering work without the pipeline running. Runtime spawning could be added later if headless batch ingestion is needed.

## 6. Manifest contract

The manifest is the single source of truth between the two repos. Both sides version it via `schema_version`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `schema_version` | int | yes | Currently `1`. Bumped on breaking changes; UE side rejects unknown versions. |
| `clip_name` | string | yes | Folder-name-safe. Becomes UE asset prefix and Reconstructions subfolder. |
| `fps` | float | yes | > 0. |
| `frame_range` | `[int, int]` | yes | Inclusive, in source clip frame indices. |
| `pitch.length_m` | float | yes | FIFA default 105. |
| `pitch.width_m` | float | yes | FIFA default 68. |
| `players` | array | yes, non-empty | One entry per successfully exported track. |
| `players[i].player_id` | string | yes | Matches pipeline `PXXX` id. |
| `players[i].fbx` | string | yes | Relative to `output/export/`. |
| `players[i].frame_range` | `[int, int]` | yes | Per-track range, may be a sub-range of clip. |
| `players[i].world_bbox` | object | yes | `{min: [x,y,z], max: [x,y,z]}` in metres. |
| `ball` | object | optional | Omitted if `ball.fbx` failed to write. |
| `camera` | object | optional | Omitted if `camera.fbx` failed to write. |

UE-side validation is described in §7.

## 7. Error handling, idempotency, edge cases

**Manifest validation (UE side, before any asset mutation):**
- Reject if `schema_version != 1` → "pipeline version mismatch — re-export required".
- Reject if any referenced FBX path is missing on disk.
- Reject if `fps`, `frame_range`, or any `world_bbox` field is missing/non-finite.
- Warn (don't reject) if any player `world_bbox` exceeds `[-60, 60] × [-40, 40] × [-1, 5]` metres — slightly larger than the 105×68 pitch to allow for keepers behind the goal line; values outside this typically signal a pitch-frame solve failure upstream.
- Warn if the player count differs from the previous import for the same `clip_name`.

**Idempotency:**
- All clip-specific assets live under `Content/FootballPerspectives/Reconstructions/<clip_name>/`. The editor utility deletes that folder before re-importing.
- Shared assets (`SK_SMPL`, `IK_*`, `IKR_*`, `BP_PlayerActor`, `M_PlayerJersey`) are never touched by the utility.

**Frame-range mismatches:**
- Per-track `frame_range` may be a sub-range of clip `frame_range`. Each Sequencer section is set to its track's range; players outside their range simply don't render.
- Zero-frame player FBX → log warning, skip, don't abort.

**Missing optional inputs:**
- No `ball` block → skip ball track.
- No `camera` block → fall back to default editor camera and warn.
- Empty `players` → fail loudly.

**Pipeline-side failure modes:**
- One player FBX fails → existing stage logs and continues; manifest only lists successful FBX files; UE side sees fewer players than expected and warns.
- Blender missing → no manifest written; existing "Blender not found" warning surfaces; UE side reports "no manifest at `<path>`".

**Coordinate-frame regressions:**
- Pipeline test asserts `root_t` round-trips through FBX export within 1mm.
- UE side relies on manual acceptance ("feet at z≈0, players inside touchlines"); `BP_DebugOverlay` aids this.

**Retarget pose drift:**
- A-pose tilt or floor offset → IKR retarget-pose tuning, fixed once and checked in, not a per-clip concern.

## 8. Testing strategy

### 8.1 Pipeline (automated)

- **P-T1 — Manifest schema unit test** — see §4.4.
- **P-T2 — SMPL→FBX joint encoding** (`-m fbx`) — see §4.4.
- **P-T3 — Full export integration** (`-m fbx`) — see §4.4.
- **P-T4 — Manifest smoke** (no Blender) — see §4.4.

Coverage target: 80%+ on new pipeline code (per repo testing rules).

### 8.2 UE5 (manual, documented in spec)

- **U-T1 — Reference clip acceptance.** Run pipeline on a known short clip → "Load Reconstruction" → open `LS_<clip_name>` → hit Play. Pass criteria:
  - All player mannequins visible and inside the touchlines for the entire clip.
  - Player feet within ±5cm of pitch surface at standing-still frames.
  - Broadcast camera view matches a reference screenshot from the same frame in the web viewer.
  - Ball trajectory intersects the goal area on a known goal-frame clip.
- **U-T2 — Idempotency.** Re-run "Load Reconstruction" on the same clip → no asset duplication; sequence still plays.
- **U-T3 — Negative paths.** Run with manifest missing `ball`, with a corrupted FBX, with a missing player file → editor utility surfaces useful errors and leaves the project in a recoverable state.

### 8.3 Out of scope for v1

- Automated UE-side asset tests (would require a UE commandlet harness).
- Visual regression on rendered output.
- Performance / scaling beyond ~22 players.

## 9. Acceptance criteria

The spec is satisfied when:

1. `recon.py run` on a fixture clip produces `output/export/fbx/PXXX.fbx` with a 24-bone SMPL skeleton + per-joint animation, plus `output/export/ue_manifest.json` validating against the schema.
2. The pipeline tests P-T1..P-T4 pass.
3. The UE5 project at `../football-perspectives-ue/` has `L_PitchMaster`, `BP_PlayerActor`, the IK Rigs + Retargeter, and `EUW_LoadReconstruction` checked in.
4. Running `EUW_LoadReconstruction` on the fixture clip's `output/` directory produces a `LS_<clip_name>` whose Play action shows mannequins running on the pitch under the broadcast camera, satisfying U-T1.
5. U-T2 and U-T3 pass.

## 10. Follow-up work (explicitly deferred)

- Skinned SMPL mesh in player FBX for visual debugging (small extension to §4.1).
- MetaHuman target skeleton (additional IK Rig + Retargeter; pipeline unchanged).
- Runtime ingestion / packaged-build loading.
- Pipeline-side team detection and jersey colour assignment.
- Sequencer-driven cinematic render output and post-process polish.
- Performance work for high-player-count clips.
- Camera distortion correction (UE camera takes linear K only; current pipeline already does).
