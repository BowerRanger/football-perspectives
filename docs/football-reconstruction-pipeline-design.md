# Football Reconstruction Pipeline — Design Reference

Implementation reference for the single-camera pipeline. The
authoritative spec is the brainstorming-skill output at
`docs/superpowers/specs/2026-05-04-broadcast-mono-pipeline-design.md`;
this file is kept in sync with it.

---

# Broadcast Single-Camera Pipeline — Design

**Date:** 2026-05-04
**Status:** Approved (pending implementation plan)
**Supersedes:** the 8-stage multi-view triangulation pipeline described in `docs/football-reconstruction-pipeline-design.md`.

## 1. Goals and scope

Replace the current pipeline with a focused single-camera reconstruction system that produces three artefacts per clip:

1. **Per-frame camera pose** (K, R, t) in pitch metres, suitable for driving a virtual camera in UE5/Blender.
2. **Per-player skeletal animation** as SMPL parameters in the pitch frame, exportable to glTF (web viewer) and FBX (UE5 retarget).
3. **Per-frame ball position** in pitch metres, with full 3D reconstruction during flight.

### Input assumptions (locked for v1)

- Single camera per clip. Multi-angle fusion is a possible future layer; not in scope.
- Manually trimmed clip (CapCut or equivalent). No automatic shot segmentation.
- Broadcast wide angle: camera body fixed (effectively a pan-tilt-zoom head). Pan up to ~60° per clip is supported. Translating cameras (steadicam) are out of scope; the pipeline detects this via low confidence and the user re-shoots / skips.
- FIFA-standard 105 × 68 m pitch geometry. Custom pitch dimensions deferred to a future enhancement (see `docs/FEATURE_IDEAS.md`).

### Output target

Pitch-relative reconstruction. The end use is recreating the play in UE5 with a virtual camera matching the real camera, players animated as MetaHumans driven by the SMPL output, and the ball as a rigid body following the recovered trajectory.

### Pragmatic posture

Manual keyframe annotation is a first-class input, not a fallback. Where a manual + propagation approach gives a working clip in minutes and a fully automated approach has been chronically flaky, the manual approach wins.

## 2. Pipeline shape

Seven stages, single mode, no `pipeline.mode` switch.

| # | Stage | Purpose |
|---|-------|---------|
| 1 | `prepare_shots` | Treats input as one already-trimmed clip; no auto-segmentation. |
| 2 | `tracking` | YOLOv8x + ByteTrack for players, YOLOv8 ball model for ball. Manual dedupe UI kept. |
| 3 | `camera` | **NEW.** Per-frame K, R, t in pitch metres via keyframe-anchored propagation. |
| 4 | `pose_2d` | ViTPose, COCO 17. Used for foot anchoring and overlays; not for 3D joint positions. |
| 5 | `hmr_world` | **NEW.** GVHMR per track → SMPL params transformed into pitch frame, foot-anchored. |
| 6 | `ball` | **NEW.** Ground-projection on grounded frames, parabolic 3D fit on flight segments. |
| 7 | `export` | glTF for web viewer + FBX via Blender headless for UE5. |

### Stages and code deleted from the repository

Source files, tests, schemas, and runner wiring all go:

- `src/stages/`: `segmentation.py`, `sync.py`, `matching.py`, `triangulation.py`, `smpl_fitting.py`, the legacy `calibration.py`, the legacy `hmr.py`.
- `src/utils/`: `tvcalib_calibrator.py`, `neural_calibrator.py`, `calibration_propagation.py`, `calibration_debug.py`, plus any helper that is only used by the removed stages.
- `src/schemas/`: legacy multi-backend calibration block, sync-related fields on `tracks.py`, segmentation metadata on `shots.py`, per-shot-island fields on `hmr_result.py`.
- Tests: legacy `test_calibration.py`, sync portions of `test_tracking.py`, `test_patch_sync.py`, `test_web_calibration_variants.py`, segmentation portions of `test_prepare_shots.py`, legacy-mode portions of `test_runner.py`, legacy `test_hmr*` (rewritten for `hmr_world`), legacy parts of `test_export.py`. `test_pose.py` is renamed to `test_pose_2d.py` and retained — the underlying ViTPose path is unchanged.
- `third_party/tvcalib` submodule and the `.gitmodules` entry.

### Code retained

- `src/utils/bundle_adjust.py` — parabolic LM is reused by `ball`.
- `src/utils/ffmpeg.py` — generic.
- `src/utils/gvhmr_estimator.py`, `gvhmr_register.py`, `third_party/gvhmr_shims/` — `hmr_world` uses GVHMR.
- `third_party/gvhmr` submodule.
- The temporal-smoothing helpers from `triangulation.py` (Savgol, foot-ground snap) — extracted into a small util used by `hmr_world`.

### Output layout

```
output/
├── shots/                  # single trimmed clip + manifest
├── tracks/                 # ByteTrack output (players + ball)
├── camera/                 # NEW: per-frame K, R, t + anchor metadata
│   ├── anchors.json        # user-annotated keyframes
│   ├── camera_track.json   # per-frame K, R, t with confidence
│   └── debug/              # overlay frames for visual inspection
├── pose_2d/                # ViTPose output
├── hmr_world/              # NEW: per-player SMPL params in pitch frame
│   ├── PXXX_smpl_world.npz
│   └── debug/              # overlay GIFs for visual inspection
├── ball/                   # NEW: per-frame ball position + flight segments
│   └── ball_track.json
├── export/
│   ├── gltf/               # web viewer
│   └── fbx/                # UE5
└── quality_report.json     # extended with per-frame camera/anchor confidence
```

### Web viewer cleanup

`src/web/`:

- `index.html`: drop the triangulation panel, sync diagnostic UI, multi-shot/per-shot-island rendering, and the calibration-backend comparison split-pane. Repurpose the existing manual landmark tool as the **anchor editor**.
- `viewer.html`: drop per-shot islands and the "no calibration" fallback. The viewer always renders pitch-registered scenes.
- `server.py`: drop `/sync/*`, `/triangulation/*`, `/calibration/compare`. Add `/anchors`, `/camera/track`, `/hmr_world/preview`, `/ball/preview`.
- New panels: anchor editor, camera-track confidence timeline, ball-flight inspector.

## 3. Camera tracking (`camera` stage)

Three components: anchor solver, frame-to-frame propagator, bidirectional smoother. Plus a web UI driving anchor placement.

### Anchor data model

An anchor is a user-annotated keyframe with pitch-landmark correspondences:

```jsonc
// output/camera/anchors.json
{
  "clip_id": "play_037",
  "image_size": [1920, 1080],
  "anchors": [
    {
      "frame": 0,
      "landmarks": [
        {"name": "near_left_corner",   "image_xy": [412, 904], "world_xyz": [0, 0, 0]},
        {"name": "near_right_corner",  "image_xy": [1612, 901],"world_xyz": [105, 0, 0]},
        {"name": "far_left_corner",    "image_xy": [610, 312], "world_xyz": [0, 68, 0]},
        {"name": "halfway_near",       "image_xy": [965, 870], "world_xyz": [52.5, 0, 0]},
        {"name": "left_goal_crossbar_left", "image_xy": [430, 845], "world_xyz": [0, 30.34, 2.44]}
      ]
    }
  ]
}
```

The pitch landmark catalogue (FIFA standard) **must include non-coplanar points** (crossbar at z = 2.44, corner flags at z = 1.5). Purely-on-pitch landmarks are degenerate for K recovery; non-coplanar anchors are what make K solvable from a single anchor without ambiguity.

### Anchor solver

- **First anchor** of a clip: full 3×4 projection-matrix DLT from ≥6 landmarks, then RQ decomposition into (K, R, t). K parameterised as `fx = fy, principal point at image centre, no skew` (1 unknown). Total unknowns 7; 6 non-coplanar points give 12 equations. Well-constrained.
- **Subsequent anchors**: t inherited from the first anchor (camera body fixed). Solve only (fx, R) with ≥4 points.
- Reprojection-residual gate: subsequent anchor's residual > `anchor_max_reprojection_px` (default 4 px) flags either bad annotation *or* a clip where t isn't actually fixed. Viewer surfaces; user re-anchors or skips clip.

### Frame-to-frame propagator

Between anchors, per-frame (K, R) is propagated by image-based feature tracking:

1. Feature detection (SuperPoint or ORB) on stable image regions. Mask out: central pitch turf, tracked player bboxes, the ball.
2. Rolling KLT/LK from frame N → N+1, with re-detection when feature count drops below `redetect_threshold` (handles long pans where features continuously enter/leave the FOV).
3. RANSAC homography H from frame N → N+1, gated by inlier ratio ≥ `ransac_inlier_min_ratio` (default 0.4).
4. Decomposition: assuming pure rotation + zoom (which `t` fixed implies for a far scene), `H ≈ K_{N+1} · ΔR · K_N⁻¹`. With (K_N, R_N) known, solve ΔR and the K_{N+1}/K_N zoom ratio. Apply to get (K_{N+1}, R_{N+1}).

This solves only the *relative* per-frame change. Drift accumulates over time and is bounded by the smoother below.

### Bidirectional smoother

For each pair of adjacent anchors (A at frame `i`, B at frame `j`), the propagator runs forward from A and backward from B over `[i, j]`. Per-frame fused estimate:

```
w(t) = (j - t) / (j - i)
K_t  = w(t) · K_t^fwd + (1 - w(t)) · K_t^bwd
R_t  = SLERP(R_t^fwd, R_t^bwd, 1 - w(t))
```

Guarantees: at `t = i`, only forward contributes (matches anchor A exactly); at `t = j`, only backward (matches anchor B exactly). Maximum drift is bounded by the *minimum* of forward and backward error at the midpoint, empirically ~half of one-way error.

### Per-frame confidence

Scalar in [0, 1]:

- Inlier ratio in homography RANSAC (drops on motion blur / occlusion).
- Forward/backward disagreement in (R, K).
- Pitch-line consistency: project FIFA pitch lines through (K_t, R_t, t) and measure pixel distance to detected pitch lines. PnLCalib's *line head* is reused as a soft check (not as a calibration source). Confidence drops when projected lines miss painted lines by > `pitch_line_consistency_max_px` (default 5 px).

### Web UI: anchor editor

Repurposed manual landmark tool:

- Top: video player with overlay (projected pitch lines, anchored landmarks, optional camera frustum).
- Middle: pitch landmark catalogue palette.
- Bottom: per-frame confidence timeline with red bands for low-confidence regions. Click → seek → "add anchor here" → click 4–8 landmarks → that anchor span re-runs.

Re-running a single anchor span is fast (< 5 s) because tracking, pose, HMR are pre-computed; only the camera span between two anchors recomputes.

### Output format

```jsonc
// output/camera/camera_track.json
{
  "clip_id": "play_037",
  "fps": 30.0,
  "image_size": [1920, 1080],
  "t_world": [55.4, -12.7, 22.3],   // fixed for the clip
  "frames": [
    {
      "frame": 0,
      "K": [[1820.0, 0, 960.0], [0, 1820.0, 540.0], [0, 0, 1]],
      "R": [[...], [...], [...]],
      "confidence": 1.0,
      "is_anchor": true
    }
  ]
}
```

### Edge cases

- **Feature-poor frames**: anchor density goes up, viewer flags. No silent failure.
- **Rolling shutter**: out of scope; flag and skip.
- **Custom pitch dimensions**: deferred to future enhancement; v1 assumes 105 × 68 m.

## 4. Player skeletal animation (`hmr_world` stage)

Produces SMPL params per player **in the pitch frame**, foot-anchored on the pitch plane.

### Inputs and output

**In:** `tracks/PXXX_track.json`, `pose_2d/PXXX_pose.json`, `camera/camera_track.json`.

**Out:**

```jsonc
// output/hmr_world/PXXX_smpl_world.npz
{
  "player_id": "P037",
  "frames":     [...],          // global frame indices
  "betas":      [10],           // shape, constant per player across clip
  "thetas":     [N, 24, 3],     // per-frame body pose (axis-angle), root excluded
  "root_R":     [N, 3, 3],      // per-frame root orientation IN PITCH FRAME
  "root_t":     [N, 3],         // per-frame world translation in pitch metres
  "confidence": [N]
}
```

This is the canonical internal format. Both glTF (web) and FBX (UE5) export read this.

### Per-track HMR

Run GVHMR on the tracked bbox crop sequence. Existing `gvhmr_estimator.py` + shims kept. β is taken as the **median** over the track (shape constant per player; GVHMR over-estimates per-frame variance). GVHMR's SimpleVO output (its monocular camera estimate) is **discarded** — we have a real camera now.

### Coordinate-frame transform — the upside-down fix

GVHMR emits root rotations in its internal SMPL world (z-up, character facing -y). Pitch frame is FIFA-standard (z-up, x along nearside touchline, y toward far side). Per frame:

```
R_smpl_to_pitch_static = fixed 3x3 rotation, computed once
R_world_root_pitch_t   = R_t^T · R_smpl_to_pitch_static · root_R_t^cam
```

where `R_t` is from `camera_track.json` (world→camera rotation, standard OpenCV extrinsic convention; `R_t^T` is therefore camera→world, which is what we apply to a quantity in the camera frame to express it in pitch metres). This is the regression-pinned fix for the upside-down problem: currently the SMPL-world transform is applied but never composed with the camera, so when the camera tilts down at the pitch the player rotates into the ground plane. The exact composition order is pinned by `test_hmr_world_frame_transform.py`.

### Foot anchoring (world translation)

GVHMR's monocular world translation is unreliable. Replace it:

1. From `pose_2d`, take the midpoint of left+right ankle keypoints (image coords).
2. From SMPL output (now in pitch frame), compute foot position relative to the SMPL root via forward kinematics.
3. Cast a ray from camera centre through the ankle-midpoint pixel and intersect with z = 0.05 m.
4. World translation `root_t_t = ray_intersection - foot_offset_in_root_frame`.

Ankle-occlusion fallback: when ankle confidence < 0.3, propagate translation forward from previous frame using image-space velocity projected to pitch metres via the camera. Both ankles occluded for > `foot_anchor_max_occlusion_frames` (default 10) → frame flagged low-confidence.

### Temporal smoothing

- **β**: collapse to single per-track median over high-confidence frames.
- **θ**: Savitzky-Golay (window 11, order 2) per joint axis-angle channel.
- **root_t**: Savgol on x, y; z snapped toward 0 with the existing `ground_snap_velocity` logic.
- **root_R**: SLERP-based smoothing (window 5).

### Compute

- GVHMR runs on bbox-crop tracks (not whole frames). Cuts work substantially compared to current `hmr.py`.
- Sequence cap of 120 frames per GVHMR call to avoid the known MPS kernel assertion; sequences stitched at the boundary using the bidirectional smoother's logic.
- Target: minutes per clip on GPU; CPU is unworkable.

### Confidence

Per-frame, per-player scalar in [0, 1]:

- HMR per-joint confidence (lowest joint as bottleneck).
- Foot-anchor agreement (ankle ray vs SMPL feet reprojection).
- Bbox tracking confidence.

Drives a per-player overlay GIF in `output/hmr_world/debug/PXXX_overlay.gif` and a per-frame heatmap in the viewer.

### Out of scope for v1

- Goalkeeper diving / extreme horizontal poses (GVHMR struggles; flagged via confidence).
- Cross-player physical constraints (no two players at the same pitch point). Future enhancement.

## 5. Ball reconstruction (`ball` stage)

Most algorithmic work already exists; this stage rewires it onto the new camera track.

### Inputs and output

**In:** `tracks/ball_track.json`, `camera/camera_track.json`.

**Out:**

```jsonc
// output/ball/ball_track.json
{
  "clip_id": "play_037",
  "fps": 30.0,
  "frames": [
    {"frame": 0, "world_xyz": [52.5, 34.0, 0.11], "state": "grounded", "confidence": 0.92}
  ],
  "flight_segments": [
    {
      "id": 3,
      "frame_range": [82, 119],
      "parabola": {"v0": [12.4, -8.1, 9.7], "p0": [...], "g": -9.81},
      "fit_residual_px": 1.4
    }
  ]
}
```

`state` is `grounded` | `flight` | `occluded` | `missing`.

### Algorithm (per frame)

1. No detection → `state: missing`, skip.
2. Provisional grounded estimate: ray from camera through detection centre, intersect with z = `ball_radius_m` (default 0.11).
3. Pixel velocity vs previous frame.

### Flight segmentation

Contiguous frames with pixel velocity ≥ `flight_px_velocity` (default 25) form candidate flight segments. Segments shorter than `min_flight_frames` (default 4) or longer than `max_flight_frames` (default 60) are demoted to grounded.

### Parabolic fit per flight segment

1. Seed from grounded estimates at segment start/end (existing heuristic).
2. LM on `(p0, v0)`; gravity fixed at -9.81 along z. Per-frame predicted position is reprojected through `(K_t, R_t, t_world)` and image-space residual minimised.
3. Accept iff mean residual < `flight_max_residual_px` (default 5) **and** trajectory within plausibility bounds (z ∈ [0, 50 m], horizontal speed ≤ 40 m/s). Else fall back to grounded.
4. For accepted segments, fill per-frame `world_xyz` with parabola evaluation; set `state: flight`.

LM and seeding code from `bundle_adjust.py` reused.

### Occlusion handling

Detection missing for ≤ `max_occlusion_frames` (default 8) inside an active flight segment → interpolate world position along the parabola. Outside flight, missing frames stay `missing`.

### Single-camera caveat

A ball moving directly toward / away from the camera, in flight, is depth-degenerate. The fit residual gate catches this; failed fits fall back to grounded estimates.

## 6. Configuration and data flow

### `config/default.yaml` (post-cleanup)

```yaml
pitch:
  length_m: 105.0
  width_m: 68.0
  goal_height_m: 2.44
  corner_flag_height_m: 1.5

prepare_shots:
  expected_format: mp4
  output_fps: null

tracking:
  player_model: yolov8x.pt
  ball_model: yolov8n.pt
  confidence_threshold: 0.3
  team_classifier: none
  default_team_label: unknown
  progress_every_frames: 150

camera:
  first_anchor_min_landmarks: 6
  subsequent_anchor_min_landmarks: 4
  anchor_max_reprojection_px: 4.0
  feature_detector: superpoint
  max_features_per_frame: 1000
  ransac_inlier_min_ratio: 0.4
  redetect_threshold: 200
  enable_bidirectional: true
  pitch_line_consistency_max_px: 5.0
  forward_backward_disagreement_warn_deg: 0.5

pose_2d:
  model_config: td-hm_ViTPose-small_8xb64-210e_coco-256x192
  checkpoint: null
  device: auto
  min_confidence: 0.3
  smooth_sigma: 2.0

hmr_world:
  device: auto
  checkpoint: third_party/gvhmr/inputs/checkpoints/gvhmr/gvhmr_siga24_release.ckpt
  batch_size: 16
  max_sequence_length: 120
  min_track_frames: 10
  beta_aggregation: median
  theta_savgol_window: 11
  theta_savgol_order: 2
  ground_snap_velocity: 0.1
  foot_anchor_max_occlusion_frames: 10

ball:
  flight_px_velocity: 25.0
  min_flight_frames: 4
  max_flight_frames: 60
  flight_max_residual_px: 5.0
  max_occlusion_frames: 8
  ball_radius_m: 0.11
  plausibility:
    z_max_m: 50.0
    horizontal_speed_max_m_s: 40.0

export:
  gltf_enabled: true
  fbx_enabled: true
  blender_path: blender
  ue5:
    forward_axis: -Y
    up_axis: Z
    scale: 1.0
```

### Stage data flow

```
   input.mp4
       │
       ▼
  prepare_shots ──▶ shots/clip.mp4 + manifest
       │
       ▼
    tracking ─────▶ tracks/PXXX_track.json + tracks/ball_track.json
       │
       ▼
     camera ◀───── (user) anchors.json   (web UI loop)
       │       └─▶ camera/camera_track.json + debug/
       ▼
    pose_2d ─────▶ pose_2d/PXXX_pose.json
       │
       ▼
   hmr_world ───▶ hmr_world/PXXX_smpl_world.npz + debug/
       │
       ▼
      ball ─────▶ ball/ball_track.json
       │
       ▼
    export ─────▶ export/gltf/scene.glb + scene_metadata.json
                  export/fbx/PXXX.fbx + ball.fbx + camera.fbx
       │
       ▼
quality_report.json
```

The `camera` stage is the only stage with a manual artefact in its inputs. The viewer drives the iteration loop; downstream stages re-run automatically when their inputs change via the existing stage-cache invalidation.

### CLI changes

```bash
# End-to-end
python recon.py --input clip.mp4 --output ./output/

# Re-run only the camera stage (after editing anchors)
python recon.py --output ./output/ --from-stage camera

# Web viewer (anchor editor + 3D viewer + per-stage debug)
python recon.py --output ./output/ --viewer

# Wipe legacy artefacts (calibration/, triangulation/, smpl/, sync/, matching/) before running
python recon.py --input clip.mp4 --output ./output/ --clean
```

**Removed:** `--stages 1,2,3` (numeric stage selection — names only now), legacy mode toggles, `--stages segmentation,calibration,sync,matching,triangulation,smpl_fitting` (those stage names no longer exist).
**Added:** `--clean` (one-time migration helper).
**Retained:** `--input`, `--output`, `--from-stage`, `--config`, `--device`, `--viewer`.

### Migration of existing `output/`

Detect legacy layout (presence of `calibration/`, `triangulation/`, `smpl/` etc.), print a single warning, refuse to mix old and new artefacts. User runs on a fresh dir or passes `--clean` to wipe legacy directories. No auto-migration of legacy calibration outputs.

### Quality report

`quality_report.json` extended:

```jsonc
{
  "camera": {
    "anchor_count": 5,
    "mean_anchor_residual_px": 1.8,
    "low_confidence_frame_count": 12,
    "low_confidence_frame_ranges": [[145, 152], [284, 288]]
  },
  "hmr_world": {
    "tracked_players": 22,
    "mean_per_player_confidence": 0.81,
    "low_confidence_players": ["P017"]
  },
  "ball": {
    "grounded_frames": 247,
    "flight_segments": 3,
    "missing_frames": 14,
    "mean_flight_fit_residual_px": 1.4
  }
}
```

## 7. Error handling, testing, documentation

### Error handling philosophy

1. **Stage cannot run** (missing model, bad config, missing input). Fail fast with one-line actionable error. No stack traces in user-facing path.
2. **Stage runs but produces low-confidence output**. Never silently substitute "good enough" data. Write the output, mark `confidence` low, surface in `quality_report.json` and viewer. User decides next step.
3. **Stage runs but produces visibly wrong output**. Caught by post-stage debug renders (overlays, projected pitch lines, anchor reprojection plots). Viewer opens to the most recently completed stage's debug view.

Stages **never** return partial state without writing it. A crashed `hmr_world` writes whatever players completed plus `partial: true` on the rest. Protects resume-from-stage workflow.

### Testing

Adheres to the global 80%+ coverage requirement (unit + integration + E2E).

**Unit tests** — small and many:

- `test_camera_anchor_solver.py`: synthetic landmark sets → solved (K, R, t) within tolerance; degenerate coplanar set → graceful failure with explanatory error.
- `test_camera_propagator.py`: synthetic two-frame homographies with known ΔR + zoom → recovered within 0.1° / 1%.
- `test_camera_bidirectional.py`: with known ground-truth pan + injected per-frame error, smoother bounds final error ≤ 50% of one-way.
- `test_hmr_world_frame_transform.py`: synthetic SMPL pose + known camera → reprojects to expected pitch coordinates; **specifically pins the upside-down regression**.
- `test_hmr_world_foot_anchor.py`: synthetic ankle keypoints + known camera → ray intersection matches expected pitch coordinate.
- `test_ball_grounded.py`: synthetic ball pixel + known camera → expected pitch x/y at z = 0.11.
- `test_ball_flight.py`: synthetic parabolic 3D arc reprojected to image, then recovered → recovered v0 within 0.5 m/s, residual < 1 px.

**Integration tests** — fewer, full stage interactions:

- `test_camera_stage.py`: synthetic clip with known ground-truth camera trajectory → camera stage end-to-end recovers (K, R, t) within tolerance for ≥ 95% of frames.
- `test_hmr_world_stage.py`: pre-computed `tracks/`, `pose_2d/`, `camera/` fixtures → `hmr_world/` output matches reference.
- `test_runner.py`: full pipeline on small synthetic clip → all expected output files present, `quality_report.json` populated.

**E2E** — gated by `pytest -m e2e`:

- 5-second real broadcast clip, 3 anchors, full pipeline, checks `camera_track.json` complete, `hmr_world/` produced for ≥ 1 player, `ball/ball_track.json` present, `export/gltf/scene.glb` opens, `export/fbx/` non-empty.

**Test fixtures**: fresh `tests/fixtures/` with synthetic clip generator (renders calibrated SMPL avatar walking on a pitch into a pre-known camera trajectory). Replaces real-clip dependency in most tests, makes suite fast and deterministic.

**Not tested in default run**: GPU-only model inference paths (GVHMR weights aren't checked into the repo). Exercised by E2E and the optional `pytest -m gpu` marker.

### Web viewer test coverage

- API-level integration tests for `/anchors`, `/camera/track`, `/hmr_world/preview`, `/ball/preview`.
- Snapshot tests for anchor editor state at known confidence inputs.

### Documentation deliverables

Three docs rewritten:

1. **`README.md`** — full rewrite. New content: project description; requirements (Python 3.11+, GVHMR submodule + checkpoint, FFmpeg, optional Blender for FBX); install (no PnLCalib, no SMPL model needed); CLI usage with new stage names; output layout; web dashboard description with anchor editor as headline feature. All references to segmentation/calibration/sync/triangulation/smpl_fitting/numeric stage aliases removed.
2. **`docs/football-reconstruction-pipeline-design.md`** — full rewrite. The current 36 KB document describes the obsolete 8-stage pipeline. Replace with a description of this new pipeline including diagrams for the camera-tracking math.
3. **`CLAUDE.md`** — replace the "Pipeline Architecture" table and "Key Design Decisions" section with the new pipeline. Keep project overview header. Update "External Dependencies" (remove PnLCalib, add GVHMR).

Three docs deleted:

- `docs/sync-approaches-diagnostic.md` — about an obsolete stage.
- `docs/2026-04-04-macos-dependency-handoff.md` — about PnLCalib/MMPose dependency woes; obsolete.
- `docs/open-questions-2026-04-13.md` — its decisions are baked into this spec.

### Migration / cleanup checklist (one-time, executed as the final implementation step)

- [ ] Delete listed source files, tests, schemas (Section 2).
- [ ] Remove `third_party/tvcalib` submodule and `.gitmodules` entry.
- [ ] Drop legacy config blocks.
- [ ] Drop legacy CLI flags / numeric aliases.
- [ ] Update README, design doc, CLAUDE.md.
- [ ] Delete obsolete docs.
- [ ] Ensure `pytest` and `pytest -m e2e` both pass on the new codebase before merge.
- [ ] Manual run on a real clip end-to-end, with screenshots in the PR description.

## 8. Open questions

None blocking — all v1 decisions are locked in this spec. Future enhancements (custom pitch dimensions, multi-angle fusion, cross-player constraints, learned flight classifier) are tracked in `docs/FEATURE_IDEAS.md`.
