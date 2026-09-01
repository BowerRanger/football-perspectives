# Blender Render Stage — Design

- **Date:** 2026-09-01
- **Status:** Approved (design reviewed in-session; spec pending user review)
- **Goal:** Close the "last mile" from pipeline output to shareable video: a new
  `render` pipeline stage that turns each shot's reconstruction into finished
  **toon/cel-shaded MP4s** from multiple cameras (broadcast, player-POV,
  over-the-shoulder, tactical drone), fully headless via Blender, in 16:9 with
  an optional 9:16 variant per camera.
- **Decisions locked with the user (2026-09-01):** toon/cel-shaded art
  direction; SMPL mesh bodies for players; 16:9 primary + 9:16 variant.
- **Builds on:** the 2026-09-01 last-mile research report
  (https://claude.ai/code/artifact/dca426fd-bb6e-42b7-837f-f639850abe19;
  memory `last-mile-content-research`):
  Blender-headless + stylized was selected as the main content channel; UE5 is
  demoted to a hero-shot channel. Key precedents: Beyond Sports-style stylized
  virtual replays; `scripts/blender_export_fbx.py` (existing headless Blender
  scene building); `src/utils/virtual_cameras.py` (existing POV/OTS tracks).

---

## 1. Why a stage, and why Blender

The UE5 + MetaHuman path renders only inside a live, stateful, crash-prone
editor (5.8.0 Mac MRQ camera-cut SIGSEGV unfixed in 5.8.1/5.8.2; 5.8 is the
last planned UE5 release). Blender's Python runs headless —
`blender -b -P script.py` is the exact mechanism the export stage already uses
— so production rendering becomes a deterministic, re-runnable, testable
pipeline stage with no live-session bridge. The toon direction is the
industry-validated choice for reconstruction-driven sports content
(NFL×Nickelodeon / Simpsons alt-casts): stylization masks reconstruction noise
instead of amplifying it and requires no likeness rights.

Integration shapes considered and rejected: extending the export stage
(conflates interchange artifacts with look-dev iteration cadence) and a
standalone `scripts/` tool (renders are a first-class product output and belong
in stage/manifest/quality-report plumbing).

## 2. Stage contract

New stage **#8: `render`** (after `export`; independently re-runnable via
`--stages render` / `--from-stage render`).

| | |
|---|---|
| **Reads** | `refined_poses/PXXX_refined.npz` (fallback `hmr_world/*_smpl_world.npz`), `ball/<shot>_ball_track.json`, `camera/camera_track.json`, `shots/shots_manifest.json` + `sync_map.json`. Virtual-camera tracks are recomputed in-stage via `src/utils/virtual_cameras.py` — no dependency on `export/` having run |
| **Writes** | `output/render/<shot>/<camera>.mp4` (+ `<camera>_9x16.mp4` when enabled), optional `output/render/<shot>/aov/` EXR passes, optional `--save-blend` scene file, render section in `quality_report.json` |
| **Iterates** | `manifest.active_shots()` like every other stage |

Execution mirrors `ExportStage._export_fbx`: the stage module
(`src/stages/render.py`) is a thin orchestrator that resolves
`render.blender_path` (shared default with `export.blender_path`), shells to
`blender --background --python scripts/blender_render_scene.py -- --output-dir …
--shot … --camera …`, and degrades gracefully (warn + skip) when Blender is
absent — same posture as FBX export.

## 3. Shared artifact readers

`scripts/blender_export_fbx.py` already contains the artifact-reading logic a
renderer needs (refined-NPZ iteration keyed by player with sync-map offset
translation, ball track parsing, camera track parsing). That logic moves to a
new **`src/utils/blender_scene_io.py`** (bpy-free, plain numpy/json) imported by
both Blender scripts. `iter_player_fbx_entries` and friends keep their behavior
and existing tests; the FBX exporter becomes a consumer. This is the only
refactor of existing code in scope.

## 4. Scene assembly (Blender-side, `scripts/blender_render_scene.py`)

Structured like the FBX exporter: CLI entry point + `main` + helpers importable
without `bpy` (testable under plain pytest).

**Fully procedural in v1 — zero binary assets in the repo.**

- **Pitch & lines:** ground plane + line/circle/box curves generated from the
  existing `src/utils/pitch.py` geometry (105×68 FIFA standard). Toon grass =
  two-tone mow stripes via a procedural material, no textures.
- **Stadium:** a procedural low-detail bowl (stands ring + simple roof band)
  plus gradient world sky. Under the toon direction this is a legitimate look,
  not a placeholder. Marketplace stadium assets are a later, optional upgrade.
- **Players:** the SMPL template mesh skinned (LBS weights from the SMPL model
  release) onto the 24-joint armatures built exactly as the FBX exporter builds
  them (canonical rest pose; `thetas` on pose bones; `root_R`/`root_t` on the
  armature object — `thetas[0]` ignored per the repo convention). SMPL model
  files are a **gitignored asset** (same policy as model weights) with a
  documented fetch path; when absent, a **procedural capsule-limb body**
  (skinned primitive per bone) renders instead — the same graceful-fallback
  pattern as the detector stack. Licensing note: SMPL commercial use requires a
  Meshcapade license; before monetization either license it or flip the
  default to the capsule-limb body / a marketplace rig.
- **Kits:** rest-pose z-band material zones (shirt / shorts / socks / skin),
  mirroring the UE `M_FootballKit` scheme. Colors resolve from a new
  `render.teams` config block (per-team shirt/shorts/socks + a player→team map
  with a heuristic default), so future kits are config-only.
- **Ball:** UV sphere driven by f-curves baked from the dense ball track;
  rolling rotation via a driver on horizontal displacement (yaw to travel
  direction, pitch += distance/radius) — the same motion-derived scheme as
  `BP_BallActor`, but keyable freely since Blender has no
  float-track-only crash constraint.
- **Toon look:** EEVEE; Shader-to-RGB + constant color ramp materials;
  inverted-hull outlines (Solidify modifier, flipped normals, emission black)
  on players and ball; compositor edge/AO pass for environment lines. A blob
  ground-shadow (shrinkwrapped dark disc or contact-shadow trick) anchors
  players and hides residual foot noise.
- **Lighting:** one sun + world gradient tuned for the cel ramps. Day look
  only in v1.

## 5. Cameras

All cameras convert through the existing `camera_math` conventions
(OpenCV K/R/t → Blender camera with per-frame rotation + focal keys).

| Camera id | Source | Notes |
|---|---|---|
| `broadcast` | `camera/camera_track.json` | The calibrated real camera; also the validation view |
| `pov:<pid>` | `virtual_cameras.build_pov_track` | Existing export-stage data |
| `ots:<pid>` | `virtual_cameras.build_ots_track` | Existing export-stage data |
| `drone` | **new** preset in `virtual_cameras.py` | Smoothed look-at track over the action centroid (players+ball bbox), elevated; a `look_at_view` composition — small addition |

**9:16 variant:** same camera, portrait render resolution with adjusted sensor
fit and a per-preset framing tweak (tighter target framing for POV/drone).
Enabled per camera via config; off for `broadcast` by default (broadcast
framing doesn't survive vertical).

## 6. Configuration (`render:` in `config/default.yaml`)

```yaml
render:
  enabled: true
  blender_path: blender          # falls back to export.blender_path
  cameras: [broadcast, drone]    # pov:<pid>/ots:<pid> opt-in per run
  resolution: [1920, 1080]
  vertical_variant: false        # per-run flag; per-camera override map
  fps: null                      # null = shot fps
  style:                        # look tokens — iteration is config-only
    palette: default             # named palette presets
    ramp_steps: 3
    outline_width_m: 0.02
    grass_stripes: 8
  teams:
    defaults:
      home: {shirt: "#c0392b", shorts: "#ffffff", socks: "#c0392b"}
      away: {shirt: "#2980b9", shorts: "#2c3e50", socks: "#2980b9"}
    by_player: {}                # e.g. {P003: away} — overrides the heuristic split
  aov_passes: false              # depth + normal + cryptomatte EXRs
  save_blend: false
  samples: 16                    # EEVEE taa samples
```

`aov_passes` is the deliberate hook for the AI-polish path and the
LoRA-training-pair farming bet from the research report: Blender emits
depth/normal/Cryptomatte natively, so the flag costs nothing and accumulates
`(untextured render, broadcast frame)` pairs as a side effect of normal
operation when enabled with the `broadcast` camera.

## 7. Testing

- **Unit (default suite):** `tests/test_blender_scene_io.py` (moved readers —
  existing FBX-exporter reader tests migrate here), kit z-band assignment,
  camera conversion round-trips, drone-track composition, teams-config
  resolution. All bpy-free.
- **Blender-gated (`fbx` marker — semantics are "needs Blender on PATH"):**
  single-frame low-res smoke render per camera type asserting output exists,
  correct resolution, and coarse pixel statistics (non-empty, grass-green
  dominant) — never exact pixels (GPU nondeterminism).
- **E2E (manual/GPU-box-free — Blender runs fine on this Mac):** 2-second
  gberch excerpt through `--stages render`, broadcast + drone, eyeball review.
- Perf gate in week 1: benchmark EEVEE s/frame on this Mac on the pinned
  Blender LTS before building out the look (research flagged 4.2-era Apple
  Silicon regressions; pin a recent LTS and verify 2–10 s/frame at 1080p).

## 8. Look-dev workflow

Production is headless only. Interactive look-dev: run the stage with
`save_blend: true`, open the emitted `.blend` (GUI or Blender MCP session) to
iterate on ramps/outlines/palette, then translate the chosen look back into
`render.style` tokens / script constants. The MCP is never in the production
path.

## 9. Out of scope (v1)

Crowd systems; marketplace stadium/character assets; cloth; scoreboard &
graphics overlays (that's the separate broadcast-overlay channel); multi-shot
highlight-group stitching (per-shot renders only; groups compose later via
ffmpeg concat); the AI polish pass itself (only the AOV hook ships); night
lighting; vertical-specific graphic layouts; audio.

## 10. Risks & mitigations

| Risk | Mitigation |
|---|---|
| EEVEE perf regression on Apple Silicon | Pin recent Blender LTS; benchmark before look-dev (week-1 gate) |
| SMPL commercial licensing | Gitignored asset + capsule-limb fallback; license or swap rig before monetization |
| Toon outlines amplify foot/pose noise | Blob shadows; refined_poses smoothing already handles the worst; outline width tunable to zero |
| Art direction doesn't land | All look parameters are config tokens; a palette/ramp change is a re-run, not a rebuild |
| Blender version drift vs FBX export | One shared `blender_path` resolution + a startup version assert in both scripts |

## 11. Effort

1–3 weeks to first publishable clip (per the research estimate): week 1 = stage
skeleton + shared readers + procedural scene + broadcast camera + perf gate;
week 2 = toon look-dev + players/kits + drone & POV cameras; week 3 = 9:16,
AOV hook, quality-report wiring, polish.
