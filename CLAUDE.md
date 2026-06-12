# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A Python CLI tool (`recon.py`) that reconstructs 3D player animations and ball trajectories from a single broadcast football camera. It takes a manually-trimmed clip **or a full highlights reel**, runs a 7-stage ML pipeline, and exports glTF (for a browser viewer) and FBX (for Unreal Engine 5).

The full technical design is in `docs/football-reconstruction-pipeline-design.md`.

## Commands

```bash
# Run the full pipeline
python recon.py run --input clip.mp4 --output ./output/

# Re-run only the camera stage (after editing anchors)
python recon.py run --input clip.mp4 --output ./output/ --from-stage camera

# Run a subset of stages by name
python recon.py run --input clip.mp4 --output ./output/ --stages tracking,camera,hmr_world

# Wipe legacy output dirs from earlier pipeline versions
python recon.py run --input clip.mp4 --output ./output/ --clean

# Ingest a full highlights reel (auto split/classify/group/align —
# inputs ≥ prepare_shots.split.min_input_duration_s take this path)
python recon.py run --input highlights.mp4 --output ./output/ --stages prepare_shots

# Web dashboard (anchor editor + 3D viewer); --port for a second instance
python recon.py serve --output ./output/ --port 8001
```

## Pipeline Architecture

The pipeline has 7 sequential stages. Each stage reads from previous stage outputs in `output/` and writes its own subdirectory. Stages are independently re-runnable.

| # | Stage | Input | Output |
|---|-------|-------|--------|
| 1 | `prepare_shots` | trimmed clip(s) or full highlights reel | `shots/*.mp4` + manifest (+ groups, `sync_map.json`, `shot_features.json`, thumbs) |
| 2 | `tracking` | shots | `tracks/PXXX_track.json` + `tracks/ball_track.json` |
| 3 | `camera` | shots + anchors | `camera/camera_track.json` + debug |
| 4 | `hmr_world` | tracks + camera | `hmr_world/PXXX_smpl_world.npz` + `hmr_world/PXXX_kp2d.json` |
| 5 | `refined_poses` | hmr_world + sync_map | `refined_poses/PXXX_refined.npz` + summary |
| 6 | `ball` | shots + camera + refined_poses/hmr_world (+ manual anchors) | `ball/<shot>_ball_track.json` + `_ball_anchors_auto.json` + `_ball_observations.json` + `_ball_keyframes.json` + `_ball_diag.json` |
| 7 | `export` | refined_poses + ball + camera | `export/gltf/scene.glb` + `export/fbx/` |

The 2D pose stage was collapsed into `hmr_world` (decision D15): GVHMR runs ViTPose internally on each player crop, so `hmr_world` consumes those keypoints directly for foot anchoring and writes them as a `PXXX_kp2d.json` side-output for the dashboard overlay.

## Key Design Decisions

**Pitch coordinate system**: The football pitch is the ground plane (z=0), FIFA standard 105m × 68m. The x axis runs along the nearside touchline; y points across the pitch toward the far touchline; z is up. All 3D positions are in pitch-metres.

**Single camera per clip**: One broadcast camera, manually trimmed to a single uninterrupted shot. The camera body is assumed fixed (broadcast pan-tilt-zoom rig), so translation `t` is solved once and held constant; only `R` and focal length vary per frame.

**Highlights ingestion** (`prepare_shots.mode: auto|copy|split`): a single input ≥ `split.min_input_duration_s` (default 90 s) is auto-split with PySceneDetect **plus a frame-diff spike-rescue pass** (recovers hard cuts the adaptive detector loses inside continuous fast action — see `src/utils/shot_split.py`), then each shot is classified from sampled frames (pitch-green ratio → reaction shots, brightness → fade transitions, YOLO person dominance → player close-ups/celebrations, zoom-invariant motion rate → slow-mo replays, which are retimed to real time at extraction). Reaction/transition shots stay in the manifest but are `excluded` (every stage iterates `manifest.active_shots()`); the dashboard's dropped tray restores them. Contiguous gameplay shots are grouped into highlight events (rules: transition between shots / source-time gap / wide live shot after a replay) and each group is auto-aligned by motion-energy NCC into the group-scoped `shots/sync_map.json` (operator `manual` offsets always win). Review UX lives in the dashboard's Prepare Shots panel (`src/web/static/js/prepare_shots_panel.js`): groups board with drag-to-regroup, dropped tray, and a per-group sync timeline.

**Camera tracking**: Keyframe-anchored. The user marks pitch landmarks on a sparse set of keyframes via the web anchor editor; the camera stage solves anchor frames first, then propagates between them with bidirectional optical-flow feature tracking and a smoother. Per-frame confidence is reported so uncertain spans surface as candidates for additional anchors.

**Player skeletal animation**: GVHMR (SIGGRAPH Asia 2024) runs per track on the cropped player. Output SMPL parameters are transformed into the pitch frame using the per-frame camera and foot-anchored against the ground plane. Foot anchoring uses GVHMR's internal ViTPose-Huge ankle keypoints when visible and falls back through bounded occlusion windows.

**Ball**: Automatic, physically-correct piecewise solve. WASB detections + IMM smoothing produce the pixel track; velocity breaks are classified into events (player touches via SMPL FK contact joints from refined_poses, bounces, goal-frame impacts via goal geometry) and become auto-anchors — same `BallAnchorSet` schema as the manual editor, persisted to `<shot>_ball_anchors_auto.json`, manual anchors always win. Resolved anchors become trajectory nodes; segments between nodes are physical primitives: endpoint-exact rolling with a friction cap, or gravity arcs through both nodes (two anchored knots + gravity fully determine monocular depth), with Magnus refinement, bounce-restitution checks and split-and-retry at velocity breaks. Spans the solver cannot explain are flagged in `quality_report.json` as the cue for a manual anchor.

**Export**: glTF for the web viewer (capsule-mesh players in v1, swappable later) and FBX via Blender headless for UE5 retargeting. UE5 convention: scale 1.0m, forward -Y, up Z.

**Keypoint format**: COCO 17 keypoints (nose through right_ankle) for 2D pose. Confidence threshold 0.3 is the cutoff below which a keypoint is treated as occluded.

## ML Models Used

- **Player detection + tracking**: YOLOv8x + ByteTrack (via `supervision`)
- **HMR + 2D pose**: GVHMR (SIGGRAPH Asia 2024) — vendored under `third_party/gvhmr`. Bundles ViTPose-Huge for COCO-17 keypoints (used internally for HMR and reused for foot anchoring + dashboard overlay).
- **Ball detection**: WASB-SBDT HRNet (vendored at `third_party/wasb_sbdt`); YOLOv8 (`yolov8n.pt`) as fallback

## Configuration

`config/default.yaml` controls all tunable parameters per stage. Key values to know:

- `camera.anchor_max_reprojection_px: 4.0` — anchor frames whose solver reprojection exceeds this are flagged
- `hmr_world.foot_anchor_max_occlusion_frames: 10` — maximum gap during which the last anchored foot position is held
- `ball.auto_anchors.*` — automatic event/anchor thresholds (touch radius, goal-impact tolerances, grounded sampling); `ball.physics.*` — solver gates (rolling friction/speed caps, restitution envelope)
- `ball.second_pass.*` — corridor-gated second detection pass over evidence gaps (on by default; re-decodes gap spans, so the ball stage costs roughly one extra detector pass on gap-heavy clips). Accepted frames carry `source="second_pass"` in the observations sidecar and never mint auto-anchors; per-shot coverage lands in `detection_coverage` in the diag sidecar and quality report.

The ankle-confidence cutoff for foot anchoring (formerly `pose_2d.min_confidence: 0.3`) is now a constant `_ANKLE_CONF_MIN = 0.3` inside `src/stages/hmr_world.py`.

## External Dependencies

Beyond Python packages, the pipeline requires:
- **FFmpeg** (clip handling)
- **GVHMR submodule + checkpoint** (`third_party/gvhmr/inputs/checkpoints/gvhmr/gvhmr_siga24_release.ckpt`)
- **Blender ≥ 3.6** (headless, only for FBX export): `snap install blender --classic`

GPU: strongly recommended for `hmr_world` (GVHMR); 8GB VRAM minimum, 12GB+ recommended for concurrent ViTPose + YOLOv8x.

## Browser Dashboard and Viewer

`python recon.py serve --output ./output/` starts a FastAPI dashboard. Static assets live in `src/web/static/` and are served alongside read-only API endpoints:

- `/` (`index.html`) — pipeline dashboard with stage status and the anchor editor link.
- `/anchor-editor` (`anchor_editor.html`) — place pitch landmarks on keyframes; the camera stage propagates between them.
- `/viewer` (`viewer.html`) — 3D viewer that loads `export/gltf/scene.glb`, with playback controls, orbit camera, and a confidence timeline highlighting frames where camera or HMR are uncertain.

## Quality Report

`output/quality_report.json` is generated at the end — per-stage diagnostics aggregated from each stage (anchor reprojection, camera confidence, HMR foot-anchor coverage, ball flight segments, export status). Check this first when debugging reconstruction quality.

## Unreal Engine Integration

The UE5 project lives at `/Users/joebower/workplace/FootballPerspectives 5.8/`. Editor Python lives in `Content/Python/football_perspectives/`. The `unreal-mcp` bridge runs on `http://127.0.0.1:8123/mcp`.

## UE5 Crash Recovery Protocol

When any `unreal-mcp` tool call fails with a connection error or timeout, follow these steps **without waiting for the user**:

### 1. Confirm crash
```bash
pgrep -f "UnrealEditor.app/Contents/MacOS/UnrealEditor"
```
No output = editor is down.

### 2. Read the crash log
```bash
tail -150 ~/Library/Logs/Unreal\ Engine/FootballPerspectivesEditor/FootballPerspectives.log
```
Scan for `Assertion failed`, `Fatal error`, `Error:`, or a call stack ending with `LogExit: Executing StaticShutdownAfterError`.

### 3. Diagnose and fix

| Crash signature | Cause | Fix |
|---|---|---|
| `ACineCameraActor::Tick` → `GetObjectDataFromId` | Stale material packed refs in CineCameraActor spawnable from prior material recompile | Already fixed (switched to CameraActor); re-run Load Reconstruction |
| `SDetailsView::PostSetObject` → `FObjectPropertyNode` → `GetObjectDataFromId` | Clicking a Sequencer track whose spawnable template has stale object refs | Delete the Level Sequence `.uasset` from disk (e.g. `Content/Reconstructions/gberch/LS_gberch.uasset`), relaunch, then re-run Load Reconstruction |
| `SWidget::Paint` / `SOverlay::OnPaint` crash loop | Editor restores session with stale Sequencer open — Slate paint crashes on stale spawnable | **Delete the Level Sequence `.uasset` from disk before relaunching** — editor starts clean without the stale Sequencer state; then re-run Load Reconstruction |
| `IsCreatedByConstructionScript` in `DestroySpawnedObject` | Stale spawnable template during save — LS was open | Close LS before saving; re-run Load Reconstruction |
| `RerunConstructionScripts` re-entrancy during startup | `/tmp/pyexec_job.py` left a live material-edit job from previous session | Write no-op: `echo 'import pathlib,json; pathlib.Path("/tmp/pyexec_out.json").write_text(json.dumps({"noop":True}))' > /tmp/pyexec_job.py` |
| `Array index out of bounds` in `LeaderPoseComponent` | Kit part mesh using `MeshComp` (Mannequin skeleton) as leader — bone count mismatch | Clear `LeaderPoseComponent` on all part components in BP_PlayerActor |

If the crash is from a Python script run via `load_reconstruction` or `build_sequence`, the traceback will appear above the callstack in the log.

### 4. Relaunch
After applying any fix (or immediately if no fix is needed):
```bash
bash "/Users/joebower/workplace/FootballPerspectives 5.8/Scripts/ue-rebuild-reattach.sh" --continue
```
Always pass `--continue` so the script arms the auto-continue helper before reloading VS Code — this resumes all open Claude panels automatically after the window reloads. Run it via Bash with `run_in_background=true` and a 360 s timeout — it takes ~2 min on a cold start.

**Before relaunching, always write a no-op to `/tmp/pyexec_job.py`** to prevent BP_PyExec from running a stale job on startup:
```bash
python3 -c "import pathlib,json; pathlib.Path('/tmp/pyexec_job.py').write_text('import pathlib,json\npathlib.Path(\"/tmp/pyexec_out.json\").write_text(json.dumps({\"noop\":True}))\n')"
```
