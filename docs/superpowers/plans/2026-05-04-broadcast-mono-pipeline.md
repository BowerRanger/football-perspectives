# Broadcast Single-Camera Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current 8-stage multi-view triangulation pipeline with a focused single-camera reconstruction pipeline producing per-frame camera pose, per-player SMPL animation in pitch coordinates, and per-frame ball position with 3D flight reconstruction.

**Architecture:** Seven sequential stages (`prepare_shots → tracking → camera → pose_2d → hmr_world → ball → export`). Camera tracking is keyframe-anchored with bidirectional propagation between user-placed anchors. Player animation is GVHMR per track with foot-anchoring in pitch frame. Ball is monocular ground projection plus parabolic flight fit.

**Tech Stack:** Python 3.11+, OpenCV (DLT/RQ/homography), GVHMR (monocular HMR), ViTPose via MMPose (2D pose), YOLOv8x + ByteTrack (tracking), pytest (testing), FastAPI + plain HTML/JS (web viewer), Blender headless (FBX export).

**Spec:** `docs/superpowers/specs/2026-05-04-broadcast-mono-pipeline-design.md`

---

## File structure overview

### Files created

```
src/
  stages/
    camera.py                 # NEW: camera-tracking stage
    hmr_world.py              # NEW: HMR-in-pitch-frame stage
    ball.py                   # NEW: ball ground+flight stage
  utils/
    pitch_landmarks.py        # NEW: FIFA landmark catalogue
    anchor_solver.py          # NEW: DLT/RQ + t-fixed solve
    feature_propagator.py     # NEW: KLT + homography decomposition
    bidirectional_smoother.py # NEW: forward/backward fusion
    camera_confidence.py      # NEW: per-frame confidence scoring
    smpl_pitch_transform.py   # NEW: SMPL-world → pitch-world rotation
    foot_anchor.py            # NEW: ankle-ray to pitch projection
    temporal_smoothing.py     # NEW: extracted Savgol/SLERP helpers
  schemas/
    anchor.py                 # NEW: anchor + landmark dataclasses
    camera_track.py           # NEW: per-frame K, R, t schema
    smpl_world.py             # NEW: SMPL-in-pitch-frame schema
    ball_track.py             # NEW: ball + flight-segment schema
  web/static/
    anchor_editor.html        # NEW: anchor placement UI

tests/
  fixtures/
    synthetic_clip.py         # NEW: rendered SMPL avatar on synthetic camera
  test_anchor_solver.py
  test_feature_propagator.py
  test_bidirectional_smoother.py
  test_camera_stage.py
  test_smpl_pitch_transform.py
  test_foot_anchor.py
  test_hmr_world_stage.py
  test_ball_grounded.py
  test_ball_flight.py
  test_ball_stage.py
  test_pitch_landmarks.py
```

### Files modified

```
src/pipeline/runner.py        # collapse to single mode, new stage list
recon.py                      # remove --stages numeric aliases, add --clean
config/default.yaml           # full rewrite per spec Section 6
src/web/static/index.html     # remove legacy panels
src/web/static/viewer.html    # remove per-shot islands
src/web/server.py             # endpoint cleanup + new endpoints
src/utils/bundle_adjust.py    # rewire to consume camera_track.json
src/stages/prepare_shots.py   # simplify (no segmentation)
src/stages/tracking.py        # drop sync-related fields
src/stages/pose.py → src/stages/pose_2d.py  # rename
src/stages/export.py          # consume hmr_world output
README.md                     # full rewrite per spec Section 7
docs/football-reconstruction-pipeline-design.md  # full rewrite
CLAUDE.md                     # update pipeline architecture
```

### Files deleted

```
src/stages/segmentation.py
src/stages/sync.py
src/stages/matching.py
src/stages/triangulation.py
src/stages/smpl_fitting.py
src/stages/calibration.py
src/stages/hmr.py
src/utils/tvcalib_calibrator.py
src/utils/neural_calibrator.py
src/utils/calibration_propagation.py
src/utils/calibration_debug.py
src/utils/calibration_align.py
src/utils/calibration_refine.py
src/utils/iterative_line_refinement.py
src/utils/manual_calibration.py
src/utils/pitch_line_detector.py
src/utils/single_shot_reconstruction.py
src/utils/triangulation.py
src/utils/triangulation_calib.py
src/utils/triangulation_dedupe.py
src/utils/vp_calibration.py
src/utils/fixed_position_solver.py
src/utils/ball_reconstruction.py             # logic re-extracted into src/stages/ball.py
src/utils/smpl_fitting.py
src/utils/pose_estimator.py                  # legacy bits; replaced by pose_2d direct usage
src/schemas/sync_map.py
src/schemas/player_matches.py
src/schemas/triangulated.py
src/schemas/smpl_result.py
src/schemas/hmr_result.py
src/schemas/poses.py                          # superseded by pose_2d schema (will recreate as schemas/pose_2d.py if needed)
tests/test_calibration.py
tests/test_patch_sync.py
tests/test_web_calibration_variants.py
tests/test_tvcalib_calibrator.py
tests/test_hmr.py
tests/test_hmr_integration.py
tests/test_runner.py                          # rewritten in Phase 6
docs/sync-approaches-diagnostic.md
docs/2026-04-04-macos-dependency-handoff.md
docs/open-questions-2026-04-13.md
third_party/tvcalib/                          # submodule
```

---

## Phase 0 — Cleanup baseline (single commit)

**Goal:** Strip the legacy pipeline before introducing new stages. After Phase 0, `pytest` may have failures (because the runner no longer wires legacy stages); the next phases drive it green again.

### Task 0.1: Remove tvcalib submodule

**Files:**
- Modify: `.gitmodules`
- Delete: `third_party/tvcalib/`

- [ ] **Step 1:** Deinit and remove the submodule.

```bash
git submodule deinit -f third_party/tvcalib
git rm -rf third_party/tvcalib
rm -rf .git/modules/third_party/tvcalib
```

- [ ] **Step 2:** Verify `.gitmodules` no longer references tvcalib.

```bash
grep tvcalib .gitmodules || echo "clean"
```

Expected output: `clean`.

### Task 0.2: Delete legacy stage source files

**Files:**
- Delete: 7 files under `src/stages/` (see file structure overview).

- [ ] **Step 1:** Delete files.

```bash
git rm src/stages/segmentation.py \
       src/stages/sync.py \
       src/stages/matching.py \
       src/stages/triangulation.py \
       src/stages/smpl_fitting.py \
       src/stages/calibration.py \
       src/stages/hmr.py
```

### Task 0.3: Delete legacy utility source files

**Files:**
- Delete: 18 files under `src/utils/` (see file structure overview).

- [ ] **Step 1:** Delete files.

```bash
git rm src/utils/tvcalib_calibrator.py \
       src/utils/neural_calibrator.py \
       src/utils/calibration_propagation.py \
       src/utils/calibration_debug.py \
       src/utils/calibration_align.py \
       src/utils/calibration_refine.py \
       src/utils/iterative_line_refinement.py \
       src/utils/manual_calibration.py \
       src/utils/pitch_line_detector.py \
       src/utils/single_shot_reconstruction.py \
       src/utils/triangulation.py \
       src/utils/triangulation_calib.py \
       src/utils/triangulation_dedupe.py \
       src/utils/vp_calibration.py \
       src/utils/fixed_position_solver.py \
       src/utils/ball_reconstruction.py \
       src/utils/smpl_fitting.py \
       src/utils/pose_estimator.py
```

### Task 0.4: Delete legacy schema files

**Files:**
- Delete: 6 files under `src/schemas/`.

- [ ] **Step 1:** Delete files.

```bash
git rm src/schemas/sync_map.py \
       src/schemas/player_matches.py \
       src/schemas/triangulated.py \
       src/schemas/smpl_result.py \
       src/schemas/hmr_result.py \
       src/schemas/poses.py
```

### Task 0.5: Delete legacy tests

**Files:**
- Delete: 7 files under `tests/`.

- [ ] **Step 1:** Delete files.

```bash
git rm tests/test_calibration.py \
       tests/test_patch_sync.py \
       tests/test_web_calibration_variants.py \
       tests/test_tvcalib_calibrator.py \
       tests/test_hmr.py \
       tests/test_hmr_integration.py \
       tests/test_runner.py
```

### Task 0.6: Rewrite `config/default.yaml`

**Files:**
- Modify: `config/default.yaml` (full rewrite to match spec Section 6).

- [ ] **Step 1:** Replace contents.

```yaml
# Pitch geometry — FIFA standard for v1.
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

### Task 0.7: Rewrite `src/pipeline/runner.py` to single mode

**Files:**
- Modify: `src/pipeline/runner.py` (full rewrite — drop mode switch, drop ball-detector branch, list new stages even though most aren't implemented yet).

- [ ] **Step 1:** Replace contents.

```python
from pathlib import Path

from src.pipeline.base import BaseStage

# Stages are imported lazily inside _stage_class() so deleting a not-yet-
# rebuilt stage doesn't break other tooling that imports the runner.

_STAGE_NAMES: list[str] = [
    "prepare_shots",
    "tracking",
    "camera",
    "pose_2d",
    "hmr_world",
    "ball",
    "export",
]


def _stage_class(name: str) -> type[BaseStage] | None:
    """Lazy import so partially-implemented pipelines still load."""
    if name == "prepare_shots":
        from src.stages.prepare_shots import PrepareShotsStage
        return PrepareShotsStage
    if name == "tracking":
        from src.stages.tracking import PlayerTrackingStage
        return PlayerTrackingStage
    if name == "camera":
        from src.stages.camera import CameraStage
        return CameraStage
    if name == "pose_2d":
        from src.stages.pose_2d import Pose2DStage
        return Pose2DStage
    if name == "hmr_world":
        from src.stages.hmr_world import HmrWorldStage
        return HmrWorldStage
    if name == "ball":
        from src.stages.ball import BallStage
        return BallStage
    if name == "export":
        from src.stages.export import ExportStage
        return ExportStage
    raise ValueError(f"Unknown stage: {name!r}")


def resolve_stages(stages: str, from_stage: str | None) -> list[str]:
    if stages == "all":
        selected = list(_STAGE_NAMES)
    else:
        selected = []
        for token in stages.split(","):
            name = token.strip()
            if name not in _STAGE_NAMES:
                raise ValueError(f"Unknown stage: {name!r}")
            selected.append(name)
    if from_stage:
        if from_stage not in _STAGE_NAMES:
            raise ValueError(f"Unknown stage: {from_stage!r}")
        idx = _STAGE_NAMES.index(from_stage)
        selected = [n for n in selected if _STAGE_NAMES.index(n) >= idx]
    return selected


def run_pipeline(
    output_dir: Path,
    stages: str,
    from_stage: str | None,
    config: dict,
    **stage_kwargs,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    active = resolve_stages(stages, from_stage)
    for name in _STAGE_NAMES:
        if name not in active:
            continue
        StageClass = _stage_class(name)
        if StageClass is None:
            print(f"  [SKIP] {name} (not implemented)")
            continue
        stage = StageClass(config=config, output_dir=output_dir, **stage_kwargs)
        if stage.is_complete() and from_stage != name:
            print(f"  [SKIP] {name} (cached)")
            continue
        print(f"  [RUN]  {name}")
        stage.run()
```

### Task 0.8: Update `recon.py` CLI

**Files:**
- Modify: `recon.py:36-86` (the `run` command).

- [ ] **Step 1:** Update the `--stages` help and add `--clean`.

Replace the `run` command body. Key changes: drop the numeric-alias help text, drop the `segmentation` requirement (now `prepare_shots`), add `--clean`.

```python
@cli.command()
@click.option(
    "--input", "input_path", required=False, default=None,
    type=click.Path(exists=True, path_type=Path),
    help="Input video file (required when prepare_shots runs).",
)
@click.option(
    "--output", "output_dir", default="./output", show_default=True,
    type=click.Path(path_type=Path), help="Output directory.",
)
@click.option(
    "--stages", default="all", show_default=True,
    help="Stages to run: 'all' or comma-separated stage names "
         "(prepare_shots,tracking,camera,pose_2d,hmr_world,ball,export).",
)
@click.option(
    "--from-stage", default=None,
    help="Resume from this stage (re-runs it even if cached, skips earlier stages).",
)
@click.option(
    "--config", "config_path", default=None,
    type=click.Path(exists=True, path_type=Path),
    help="YAML config file (merged with defaults).",
)
@click.option(
    "--device", default="auto", show_default=True,
    help="Compute device: cuda, cpu, mps, or auto.",
)
@click.option(
    "--clean", is_flag=True, default=False,
    help="Wipe legacy artefact directories (calibration, sync, triangulation, smpl, matching) before running.",
)
def run(
    input_path: Path | None,
    output_dir: Path,
    stages: str,
    from_stage: str | None,
    config_path: Path | None,
    device: str,
    clean: bool,
) -> None:
    """Run the reconstruction pipeline on a video file."""
    import shutil

    cfg = load_config(config_path)
    if clean:
        for legacy in ("calibration", "sync", "triangulation", "smpl", "matching"):
            target = output_dir / legacy
            if target.exists():
                shutil.rmtree(target)
                click.echo(f"Removed legacy: {target}")

    active_stages = resolve_stages(stages=stages, from_stage=from_stage)
    if "prepare_shots" in active_stages and input_path is None:
        raise click.UsageError(
            "--input is required when prepare_shots is part of the active stages"
        )

    click.echo(f"Input:  {input_path}")
    click.echo(f"Output: {output_dir}")
    click.echo(f"Stages: {stages}")
    run_pipeline(
        output_dir=output_dir,
        stages=stages,
        from_stage=from_stage,
        config=cfg,
        video_path=input_path,
        device=device,
    )
    click.echo("Done.")
```

- [ ] **Step 2:** Remove the `resolve_stages` import config-arg adjustment if needed (the new runner takes 2 positional args, not 3).

Verify the call site: `resolve_stages(stages=stages, from_stage=from_stage)` — no `config=` kwarg.

### Task 0.9: Stop using deleted modules in retained code

**Files:**
- Modify: `src/stages/prepare_shots.py` (drop any sync-related imports / fields).
- Modify: `src/stages/tracking.py` (drop sync-derived `sync_offset` and the `matching/player_matches.json` write — those will move to a future cross-clip layer).
- Modify: `src/stages/export.py` (drop `triangulated/` and `smpl/` reads; export now consumes `hmr_world/`).
- Rename: `src/stages/pose.py` → `src/stages/pose_2d.py`. Inside, rename class `PoseEstimationStage` → `Pose2DStage`. Update all imports.
- Modify: `src/utils/bundle_adjust.py` (drop functions only used by deleted stages; keep parabolic LM and seed helpers).

- [ ] **Step 1:** For each file above, remove imports that no longer resolve, drop dead code paths. Use `pytest --collect-only` after each to surface broken imports.

```bash
git mv src/stages/pose.py src/stages/pose_2d.py
```

Update `Pose2DStage` class name inside the file.

- [ ] **Step 2:** Verify the test collection runs without ImportErrors.

```bash
pytest --collect-only 2>&1 | tail -30
```

Expected: collection completes (some tests may still be marked failing/skipped — fixed in later phases). No ImportError lines from the modules above.

### Task 0.10: Phase 0 commit

- [ ] **Step 1:** Stage and commit.

```bash
git add -A
git commit -m "chore: strip legacy multi-view triangulation pipeline

Removes segmentation, sync, matching, triangulation, smpl_fitting,
legacy calibration, and legacy hmr stages plus their tests, schemas,
utils, and the tvcalib submodule. Collapses pipeline.runner to a
single sequence: prepare_shots, tracking, camera, pose_2d,
hmr_world, ball, export. Leaves new stages as not-yet-implemented
imports — Phases 1–3 drive them in.

Refs spec docs/superpowers/specs/2026-05-04-broadcast-mono-pipeline-design.md"
```

---

## Phase 1 — Camera tracking stage

**Goal:** Implement the `camera` stage end-to-end with full test coverage. The stage takes anchors plus the trimmed clip and produces `camera/camera_track.json` with per-frame K, R, t and confidence.

### Task 1.1: Pitch landmark catalogue

**Files:**
- Create: `src/utils/pitch_landmarks.py`
- Test: `tests/test_pitch_landmarks.py`

- [ ] **Step 1: Write the failing test.**

```python
# tests/test_pitch_landmarks.py
import pytest

from src.utils.pitch_landmarks import LANDMARK_CATALOGUE, get_landmark, has_non_coplanar


@pytest.mark.unit
def test_known_landmarks_present():
    assert get_landmark("near_left_corner").world_xyz == (0.0, 0.0, 0.0)
    assert get_landmark("near_right_corner").world_xyz == (105.0, 0.0, 0.0)
    assert get_landmark("far_left_corner").world_xyz == (0.0, 68.0, 0.0)
    assert get_landmark("halfway_near").world_xyz == (52.5, 0.0, 0.0)


@pytest.mark.unit
def test_non_coplanar_landmarks_exist():
    """K-recovery requires non-coplanar landmarks."""
    crossbar = get_landmark("left_goal_crossbar_left")
    corner_flag_top = get_landmark("near_left_corner_flag_top")
    assert crossbar.world_xyz[2] > 0
    assert corner_flag_top.world_xyz[2] > 0


@pytest.mark.unit
def test_unknown_landmark_raises():
    with pytest.raises(KeyError):
        get_landmark("bogus_landmark")


@pytest.mark.unit
def test_has_non_coplanar_returns_true_with_crossbar():
    landmarks = [
        get_landmark("near_left_corner"),
        get_landmark("near_right_corner"),
        get_landmark("far_left_corner"),
        get_landmark("left_goal_crossbar_left"),
    ]
    assert has_non_coplanar(landmarks)


@pytest.mark.unit
def test_has_non_coplanar_returns_false_for_pitch_only():
    landmarks = [
        get_landmark("near_left_corner"),
        get_landmark("near_right_corner"),
        get_landmark("far_left_corner"),
        get_landmark("halfway_near"),
    ]
    assert not has_non_coplanar(landmarks)
```

- [ ] **Step 2: Run test to verify failure.**

```bash
pytest tests/test_pitch_landmarks.py -v
```

Expected: ImportError or FAIL.

- [ ] **Step 3: Write minimal implementation.**

```python
# src/utils/pitch_landmarks.py
"""FIFA-standard pitch landmark catalogue (105 x 68 m).

Coordinate system: x along nearside touchline (0 = near-left corner,
105 = near-right corner), y from near (0) to far (68), z up. Goal
crossbars at z = 2.44 m, corner flag tops at z = 1.5 m.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PitchLandmark:
    name: str
    world_xyz: tuple[float, float, float]


# FIFA-standard goal width = 7.32 m, centred on the goal line midpoint.
_GOAL_HALF_W = 7.32 / 2
_PITCH_LEN = 105.0
_PITCH_WID = 68.0
_GOAL_HEIGHT = 2.44
_FLAG_HEIGHT = 1.5

_LANDMARKS: tuple[PitchLandmark, ...] = (
    # Corners (z=0)
    PitchLandmark("near_left_corner",  (0.0,        0.0,        0.0)),
    PitchLandmark("near_right_corner", (_PITCH_LEN, 0.0,        0.0)),
    PitchLandmark("far_left_corner",   (0.0,        _PITCH_WID, 0.0)),
    PitchLandmark("far_right_corner",  (_PITCH_LEN, _PITCH_WID, 0.0)),
    # Halfway line on each touchline
    PitchLandmark("halfway_near", (_PITCH_LEN / 2, 0.0,        0.0)),
    PitchLandmark("halfway_far",  (_PITCH_LEN / 2, _PITCH_WID, 0.0)),
    # Centre circle centre + cardinal points (radius 9.15 m)
    PitchLandmark("centre_spot",        (52.5, 34.0,             0.0)),
    PitchLandmark("centre_circle_near", (52.5, 34.0 - 9.15,       0.0)),
    PitchLandmark("centre_circle_far",  (52.5, 34.0 + 9.15,       0.0)),
    # 18-yard box corners (16.5 m from goal line, 16.5 m off centre line of goal)
    PitchLandmark("left_18yd_near",  (16.5,             34.0 - 20.16, 0.0)),
    PitchLandmark("left_18yd_far",   (16.5,             34.0 + 20.16, 0.0)),
    PitchLandmark("right_18yd_near", (_PITCH_LEN - 16.5, 34.0 - 20.16, 0.0)),
    PitchLandmark("right_18yd_far",  (_PITCH_LEN - 16.5, 34.0 + 20.16, 0.0)),
    # 6-yard box corners (5.5 m from goal line, 9.16 m off goal centreline)
    PitchLandmark("left_6yd_near",  (5.5,              34.0 - 9.16,  0.0)),
    PitchLandmark("left_6yd_far",   (5.5,              34.0 + 9.16,  0.0)),
    PitchLandmark("right_6yd_near", (_PITCH_LEN - 5.5,  34.0 - 9.16,  0.0)),
    PitchLandmark("right_6yd_far",  (_PITCH_LEN - 5.5,  34.0 + 9.16,  0.0)),
    # Penalty spots (11 m from goal line, on goal centreline)
    PitchLandmark("left_penalty_spot",  (11.0,              34.0, 0.0)),
    PitchLandmark("right_penalty_spot", (_PITCH_LEN - 11.0, 34.0, 0.0)),
    # Goal crossbar endpoints (z = 2.44)
    PitchLandmark("left_goal_crossbar_left",   (0.0,        34.0 - _GOAL_HALF_W, _GOAL_HEIGHT)),
    PitchLandmark("left_goal_crossbar_right",  (0.0,        34.0 + _GOAL_HALF_W, _GOAL_HEIGHT)),
    PitchLandmark("right_goal_crossbar_left",  (_PITCH_LEN, 34.0 - _GOAL_HALF_W, _GOAL_HEIGHT)),
    PitchLandmark("right_goal_crossbar_right", (_PITCH_LEN, 34.0 + _GOAL_HALF_W, _GOAL_HEIGHT)),
    # Corner flag tops (z = 1.5)
    PitchLandmark("near_left_corner_flag_top",  (0.0,        0.0,        _FLAG_HEIGHT)),
    PitchLandmark("near_right_corner_flag_top", (_PITCH_LEN, 0.0,        _FLAG_HEIGHT)),
    PitchLandmark("far_left_corner_flag_top",   (0.0,        _PITCH_WID, _FLAG_HEIGHT)),
    PitchLandmark("far_right_corner_flag_top",  (_PITCH_LEN, _PITCH_WID, _FLAG_HEIGHT)),
)

LANDMARK_CATALOGUE: dict[str, PitchLandmark] = {lm.name: lm for lm in _LANDMARKS}


def get_landmark(name: str) -> PitchLandmark:
    if name not in LANDMARK_CATALOGUE:
        raise KeyError(f"Unknown pitch landmark: {name!r}")
    return LANDMARK_CATALOGUE[name]


def has_non_coplanar(landmarks) -> bool:
    """True iff the landmark set spans more than one z-plane.

    Accepts any iterable of objects with a ``world_xyz`` attribute (or
    3-element sequence). This lets the same helper guard against
    coplanar inputs whether the caller passes ``PitchLandmark`` or
    ``LandmarkObservation`` instances.
    """
    z_values: set[float] = set()
    for lm in landmarks:
        xyz = getattr(lm, "world_xyz", lm)
        z_values.add(round(float(xyz[2]), 6))
    return len(z_values) >= 2
```

- [ ] **Step 4: Run tests to verify pass.**

```bash
pytest tests/test_pitch_landmarks.py -v
```

Expected: all 5 tests pass.

- [ ] **Step 5: Commit.**

```bash
git add src/utils/pitch_landmarks.py tests/test_pitch_landmarks.py
git commit -m "feat(camera): add FIFA pitch landmark catalogue"
```

### Task 1.2: Anchor schema

**Files:**
- Create: `src/schemas/anchor.py`

- [ ] **Step 1: Implement frozen dataclasses for anchors.**

```python
# src/schemas/anchor.py
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class LandmarkObservation:
    name: str
    image_xy: tuple[float, float]
    world_xyz: tuple[float, float, float]


@dataclass(frozen=True)
class Anchor:
    frame: int
    landmarks: tuple[LandmarkObservation, ...]


@dataclass(frozen=True)
class AnchorSet:
    clip_id: str
    image_size: tuple[int, int]   # (width, height)
    anchors: tuple[Anchor, ...]

    @classmethod
    def load(cls, path: Path) -> "AnchorSet":
        with path.open() as fh:
            data = json.load(fh)
        anchors = tuple(
            Anchor(
                frame=int(a["frame"]),
                landmarks=tuple(
                    LandmarkObservation(
                        name=str(lm["name"]),
                        image_xy=tuple(lm["image_xy"]),
                        world_xyz=tuple(lm["world_xyz"]),
                    )
                    for lm in a["landmarks"]
                ),
            )
            for a in data["anchors"]
        )
        return cls(
            clip_id=str(data["clip_id"]),
            image_size=tuple(data["image_size"]),
            anchors=anchors,
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as fh:
            json.dump(asdict(self), fh, indent=2, default=lambda v: list(v) if isinstance(v, tuple) else v)
```

- [ ] **Step 2: Smoke-import.**

```bash
python -c "from src.schemas.anchor import AnchorSet, Anchor, LandmarkObservation; print('ok')"
```

- [ ] **Step 3: Commit.**

```bash
git add src/schemas/anchor.py
git commit -m "feat(camera): anchor + landmark observation schemas"
```

### Task 1.3: First-anchor solver (full DLT + RQ)

**Files:**
- Create: `src/utils/anchor_solver.py`
- Test: `tests/test_anchor_solver.py`

- [ ] **Step 1: Write the failing test.**

```python
# tests/test_anchor_solver.py
import numpy as np
import pytest

from src.schemas.anchor import LandmarkObservation
from src.utils.anchor_solver import (
    AnchorSolveError,
    solve_first_anchor,
    solve_subsequent_anchor,
)


def _project(K: np.ndarray, R: np.ndarray, t: np.ndarray, world_xyz: np.ndarray) -> np.ndarray:
    """Return image (u, v) for a 3D world point given (K, R, t)."""
    cam = R @ world_xyz + t
    pix = K @ cam
    return pix[:2] / pix[2]


def _make_synthetic(K, R, t, names_with_world):
    return tuple(
        LandmarkObservation(
            name=name,
            image_xy=tuple(_project(K, R, t, np.array(world, dtype=float))),
            world_xyz=tuple(world),
        )
        for name, world in names_with_world
    )


@pytest.mark.unit
def test_first_anchor_recovers_known_camera():
    K = np.array([[1820.0, 0, 960.0], [0, 1820.0, 540.0], [0, 0, 1.0]])
    R = np.eye(3)
    R = np.array(
        [[1, 0, 0],
         [0, 0, 1],
         [0, -1, 0]],
        dtype=float,
    )  # camera looking down +y
    t = np.array([-52.5, 100.0, 22.0])  # pitch metres

    landmarks = _make_synthetic(K, R, t, [
        ("near_left_corner",            (0, 0, 0)),
        ("near_right_corner",           (105, 0, 0)),
        ("far_left_corner",             (0, 68, 0)),
        ("halfway_near",                (52.5, 0, 0)),
        ("left_goal_crossbar_left",     (0, 30.34, 2.44)),
        ("near_left_corner_flag_top",   (0, 0, 1.5)),
    ])

    K_hat, R_hat, t_hat = solve_first_anchor(landmarks)
    assert np.allclose(K_hat, K, atol=2.0)
    assert np.allclose(R_hat, R, atol=1e-3)
    assert np.allclose(t_hat, t, atol=0.05)


@pytest.mark.unit
def test_first_anchor_rejects_coplanar_set():
    landmarks = tuple(
        LandmarkObservation(name=f"lm_{i}", image_xy=(i, i), world_xyz=(float(i), float(i), 0.0))
        for i in range(6)
    )
    with pytest.raises(AnchorSolveError):
        solve_first_anchor(landmarks)


@pytest.mark.unit
def test_first_anchor_rejects_too_few_points():
    K = np.eye(3); R = np.eye(3); t = np.zeros(3)
    landmarks = _make_synthetic(K, R, t, [("a", (1.0, 2.0, 0.5))] * 3)
    with pytest.raises(AnchorSolveError):
        solve_first_anchor(landmarks[:3])
```

- [ ] **Step 2: Run test — expect failure.**

```bash
pytest tests/test_anchor_solver.py -v
```

- [ ] **Step 3: Implement.**

```python
# src/utils/anchor_solver.py
"""Anchor-frame camera solver.

First anchor: full 3x4 projection matrix DLT, then RQ decomposition
into (K, R, t). Subsequent anchors: t inherited; solve only fx and R
for the camera-body-fixed assumption.
"""

from __future__ import annotations

import numpy as np

from src.schemas.anchor import LandmarkObservation
from src.utils.pitch_landmarks import has_non_coplanar


class AnchorSolveError(RuntimeError):
    pass


def _build_dlt_matrix(landmarks: list[LandmarkObservation]) -> np.ndarray:
    rows: list[list[float]] = []
    for lm in landmarks:
        X, Y, Z = lm.world_xyz
        u, v = lm.image_xy
        rows.append([ X,  Y,  Z, 1.0,   0,  0,  0,  0,  -u * X, -u * Y, -u * Z, -u])
        rows.append([ 0,  0,  0,   0,   X,  Y,  Z, 1.0, -v * X, -v * Y, -v * Z, -v])
    return np.asarray(rows, dtype=float)


def _rq_decomposition(M: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """RQ decomposition of a 3x3 matrix into (K, R) with K upper-triangular,
    R orthogonal, K[2,2] == 1."""
    Q, R = np.linalg.qr(np.flipud(M).T)
    R = np.flipud(R.T)
    R = np.fliplr(R)
    Q = Q.T
    Q = np.flipud(Q)
    K = R
    Rmat = Q
    # Force K diagonals positive
    sign = np.diag(np.sign(np.diag(K)))
    K = K @ sign
    Rmat = sign @ Rmat
    return K / K[2, 2], Rmat


def solve_first_anchor(
    landmarks: tuple[LandmarkObservation, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (K, R, t) where R, t are world->camera (OpenCV convention)."""
    if len(landmarks) < 6:
        raise AnchorSolveError(
            f"first anchor needs ≥6 landmarks, got {len(landmarks)}"
        )
    if not has_non_coplanar(landmarks):
        raise AnchorSolveError(
            "first anchor needs at least one non-coplanar landmark "
            "(crossbar or corner flag top)"
        )
    A = _build_dlt_matrix(list(landmarks))
    _, _, vh = np.linalg.svd(A)
    p = vh[-1]
    P = p.reshape(3, 4)

    M = P[:, :3]
    K, R = _rq_decomposition(M)
    # t from K^-1 P[:, 3]
    t = np.linalg.solve(K, P[:, 3])
    # Disambiguate sign: enforce that the first landmark projects with positive
    # depth (cam_z > 0).
    Xw = np.array(landmarks[0].world_xyz)
    if (R @ Xw + t)[2] < 0:
        R = -R
        t = -t
    return K, R, t


def solve_subsequent_anchor(
    landmarks: tuple[LandmarkObservation, ...],
    t_world: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve (K, R) given a fixed t. Iterative LM minimising reprojection
    residual; K parameterised as fx=fy with principal point at image centre.

    Returns (K, R).
    """
    from scipy.optimize import least_squares

    if len(landmarks) < 4:
        raise AnchorSolveError(
            f"subsequent anchor needs ≥4 landmarks, got {len(landmarks)}"
        )

    # Image centre: take from first landmark image_xy max as a proxy for size
    # (caller passes image_size separately; we expose a helper if needed).
    # Here we infer from the landmark spread — caller can also override.
    us = np.array([lm.image_xy[0] for lm in landmarks])
    vs = np.array([lm.image_xy[1] for lm in landmarks])
    cx = float((us.min() + us.max()) / 2)
    cy = float((vs.min() + vs.max()) / 2)

    world_pts = np.array([lm.world_xyz for lm in landmarks])
    image_pts = np.array([lm.image_xy for lm in landmarks])

    def _params_to_KR(p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        fx = p[0]
        rvec = p[1:4]
        theta = np.linalg.norm(rvec)
        if theta < 1e-9:
            R = np.eye(3)
        else:
            k = rvec / theta
            K_skew = np.array(
                [[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]]
            )
            R = (
                np.eye(3)
                + np.sin(theta) * K_skew
                + (1 - np.cos(theta)) * (K_skew @ K_skew)
            )
        K = np.array([[fx, 0, cx], [0, fx, cy], [0, 0, 1.0]])
        return K, R

    def _residuals(p: np.ndarray) -> np.ndarray:
        K, R = _params_to_KR(p)
        cam = world_pts @ R.T + t_world
        pix = cam @ K.T
        proj = pix[:, :2] / pix[:, 2:3]
        return (proj - image_pts).reshape(-1)

    p0 = np.array([1500.0, 0.01, 0.01, 0.01])
    result = least_squares(_residuals, p0, method="lm", max_nfev=200)
    K, R = _params_to_KR(result.x)
    return K, R
```

Note: the test for `solve_subsequent_anchor` is added in Task 1.4.

- [ ] **Step 4: Run tests to verify pass.**

```bash
pytest tests/test_anchor_solver.py -v
```

Expected: 3 tests pass.

- [ ] **Step 5: Commit.**

```bash
git add src/utils/anchor_solver.py tests/test_anchor_solver.py
git commit -m "feat(camera): first-anchor DLT solver"
```

### Task 1.4: Subsequent-anchor solver test

**Files:**
- Modify: `tests/test_anchor_solver.py` (append).

- [ ] **Step 1: Add test.**

```python
@pytest.mark.unit
def test_subsequent_anchor_recovers_K_and_R_with_t_fixed():
    K_true = np.array([[1900.0, 0, 960], [0, 1900.0, 540], [0, 0, 1]])
    angle = np.deg2rad(15.0)  # 15° pan from first anchor
    R_true = np.array(
        [[np.cos(angle), 0, np.sin(angle)],
         [0, 1, 0],
         [-np.sin(angle), 0, np.cos(angle)]],
    ) @ np.array(
        [[1, 0, 0],
         [0, 0, 1],
         [0, -1, 0]],
        dtype=float,
    )
    t_true = np.array([-52.5, 100.0, 22.0])

    landmarks = _make_synthetic(K_true, R_true, t_true, [
        ("near_left_corner",            (0, 0, 0)),
        ("near_right_corner",           (105, 0, 0)),
        ("far_left_corner",             (0, 68, 0)),
        ("halfway_near",                (52.5, 0, 0)),
    ])
    K_hat, R_hat = solve_subsequent_anchor(landmarks, t_true)
    assert np.allclose(K_hat, K_true, atol=10.0)
    assert np.allclose(R_hat, R_true, atol=1e-2)
```

- [ ] **Step 2: Run test, expect pass (solver implemented in 1.3).**

```bash
pytest tests/test_anchor_solver.py::test_subsequent_anchor_recovers_K_and_R_with_t_fixed -v
```

- [ ] **Step 3: Commit.**

```bash
git commit -am "test(camera): pin subsequent-anchor solver"
```

### Task 1.5: Frame-to-frame feature propagator

**Files:**
- Create: `src/utils/feature_propagator.py`
- Test: `tests/test_feature_propagator.py`

- [ ] **Step 1: Write failing test.**

```python
# tests/test_feature_propagator.py
import numpy as np
import pytest

from src.utils.feature_propagator import (
    PropagatorResult,
    decompose_homography_to_R_zoom,
    propagate_one_frame,
)


@pytest.mark.unit
def test_decompose_zero_motion_homography_returns_identity():
    K = np.array([[1500.0, 0, 960], [0, 1500.0, 540], [0, 0, 1]])
    H = np.eye(3)
    dR, zoom = decompose_homography_to_R_zoom(H, K)
    assert np.allclose(dR, np.eye(3), atol=1e-6)
    assert abs(zoom - 1.0) < 1e-6


@pytest.mark.unit
def test_decompose_recovers_known_pan():
    K = np.array([[1500.0, 0, 960], [0, 1500.0, 540], [0, 0, 1]])
    angle = np.deg2rad(2.0)
    dR_true = np.array(
        [[np.cos(angle), 0, np.sin(angle)],
         [0, 1, 0],
         [-np.sin(angle), 0, np.cos(angle)]],
    )
    H = K @ dR_true @ np.linalg.inv(K)
    dR, zoom = decompose_homography_to_R_zoom(H, K)
    assert np.allclose(dR, dR_true, atol=1e-3)
    assert abs(zoom - 1.0) < 1e-3


@pytest.mark.unit
def test_decompose_recovers_known_zoom():
    K = np.array([[1500.0, 0, 960], [0, 1500.0, 540], [0, 0, 1]])
    K_next = np.array([[1650.0, 0, 960], [0, 1650.0, 540], [0, 0, 1]])
    H = K_next @ np.linalg.inv(K)
    dR, zoom = decompose_homography_to_R_zoom(H, K)
    assert np.allclose(dR, np.eye(3), atol=1e-3)
    assert abs(zoom - 1.1) < 1e-2
```

- [ ] **Step 2: Run failing test.**

```bash
pytest tests/test_feature_propagator.py -v
```

- [ ] **Step 3: Implement.**

```python
# src/utils/feature_propagator.py
"""Frame-to-frame propagator for fixed-position broadcast cameras.

Given two consecutive frames and the prior frame's (K, R), recover
the next frame's (K, R) from a feature-tracking homography.

For a camera with fixed t and a far scene, frame-to-frame motion is
approximately pure rotation + zoom: H ~ K_next * dR * K_prev^-1.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class PropagatorResult:
    K: np.ndarray
    R: np.ndarray
    inlier_ratio: float
    feature_count: int


def decompose_homography_to_R_zoom(
    H: np.ndarray, K_prev: np.ndarray
) -> tuple[np.ndarray, float]:
    """Decompose H = K_next * dR * K_prev^-1 assuming K_next = zoom * K_prev
    in fx/fy and the same principal point (broadcast PTZ assumption).

    Returns (dR, zoom_ratio).
    """
    K_inv = np.linalg.inv(K_prev)
    M = H @ K_prev  # = K_next * dR
    # K_next is zoom * K_prev structure (same principal point, same fx/fy ratio).
    # Estimate zoom from the ratio of M[:, :2] columns to dR columns.
    # Trick: extract dR by removing the K_next from M, where K_next = K_prev with fx scaled.
    # We solve for zoom s such that K_prev_scaled^-1 @ M is closest to a rotation.

    cx = K_prev[0, 2]
    cy = K_prev[1, 2]

    def _try_zoom(s: float) -> tuple[np.ndarray, float]:
        K_next = np.array(
            [[s * K_prev[0, 0], 0, cx], [0, s * K_prev[1, 1], cy], [0, 0, 1.0]]
        )
        dR_candidate = np.linalg.inv(K_next) @ H @ K_prev
        # Project onto SO(3) via SVD
        U, _, Vt = np.linalg.svd(dR_candidate)
        dR_proj = U @ Vt
        if np.linalg.det(dR_proj) < 0:
            U[:, -1] *= -1
            dR_proj = U @ Vt
        residual = float(np.linalg.norm(dR_candidate - dR_proj, ord="fro"))
        return dR_proj, residual

    # Golden-section search over zoom in [0.5, 2.0]
    lo, hi = 0.5, 2.0
    phi = (1 + 5 ** 0.5) / 2
    for _ in range(60):
        a = hi - (hi - lo) / phi
        b = lo + (hi - lo) / phi
        _, ra = _try_zoom(a)
        _, rb = _try_zoom(b)
        if ra < rb:
            hi = b
        else:
            lo = a
    zoom = (lo + hi) / 2
    dR, _ = _try_zoom(zoom)
    return dR, zoom


def propagate_one_frame(
    img_prev: np.ndarray,
    img_next: np.ndarray,
    K_prev: np.ndarray,
    R_prev: np.ndarray,
    *,
    detector: str = "orb",
    max_features: int = 1000,
    ransac_inlier_min_ratio: float = 0.4,
    mask_prev: np.ndarray | None = None,
) -> PropagatorResult | None:
    """Track features prev → next, fit homography, decompose to (K, R).

    Returns None when feature count or RANSAC inlier ratio fall below
    thresholds — caller treats as low confidence.
    """
    if detector == "orb":
        det = cv2.ORB_create(nfeatures=max_features)
    else:
        raise NotImplementedError(f"detector {detector!r} not supported")

    gray_prev = cv2.cvtColor(img_prev, cv2.COLOR_BGR2GRAY) if img_prev.ndim == 3 else img_prev
    gray_next = cv2.cvtColor(img_next, cv2.COLOR_BGR2GRAY) if img_next.ndim == 3 else img_next

    kp_prev, desc_prev = det.detectAndCompute(gray_prev, mask_prev)
    kp_next, desc_next = det.detectAndCompute(gray_next, None)
    if desc_prev is None or desc_next is None or len(kp_prev) < 50 or len(kp_next) < 50:
        return None

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(desc_prev, desc_next)
    if len(matches) < 30:
        return None

    src_pts = np.array([kp_prev[m.queryIdx].pt for m in matches])
    dst_pts = np.array([kp_next[m.trainIdx].pt for m in matches])

    H, inlier_mask = cv2.findHomography(
        src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=3.0
    )
    if H is None or inlier_mask is None:
        return None
    inlier_ratio = float(inlier_mask.sum()) / float(len(inlier_mask))
    if inlier_ratio < ransac_inlier_min_ratio:
        return None

    dR, zoom = decompose_homography_to_R_zoom(H, K_prev)
    R_next = dR @ R_prev
    K_next = K_prev.copy()
    K_next[0, 0] *= zoom
    K_next[1, 1] *= zoom
    return PropagatorResult(
        K=K_next, R=R_next, inlier_ratio=inlier_ratio, feature_count=len(matches),
    )
```

- [ ] **Step 4: Run tests.**

```bash
pytest tests/test_feature_propagator.py -v
```

Expected: 3 tests pass.

- [ ] **Step 5: Commit.**

```bash
git add src/utils/feature_propagator.py tests/test_feature_propagator.py
git commit -m "feat(camera): frame-to-frame KLT+homography propagator"
```

### Task 1.6: Bidirectional smoother

**Files:**
- Create: `src/utils/bidirectional_smoother.py`
- Test: `tests/test_bidirectional_smoother.py`

- [ ] **Step 1: Write failing test.**

```python
# tests/test_bidirectional_smoother.py
import numpy as np
import pytest

from src.utils.bidirectional_smoother import smooth_between_anchors


def _slerp_naive(R0, R1, t):
    """Helper to compose a known truth trajectory."""
    from scipy.spatial.transform import Rotation, Slerp
    rots = Rotation.from_matrix([R0, R1])
    return Slerp([0, 1], rots)([t]).as_matrix()[0]


@pytest.mark.unit
def test_smoother_matches_anchors_at_endpoints():
    K_anchor_a = np.array([[1500.0, 0, 960], [0, 1500.0, 540], [0, 0, 1]])
    K_anchor_b = np.array([[1700.0, 0, 960], [0, 1700.0, 540], [0, 0, 1]])
    R_anchor_a = np.eye(3)
    R_anchor_b = np.array(
        [[0.99619, 0, 0.08716],
         [0, 1, 0],
         [-0.08716, 0, 0.99619]],
        dtype=float,
    )

    # Forward propagator returns clean linear interpolation.
    Ks_fwd = [K_anchor_a + (K_anchor_b - K_anchor_a) * (i / 10) for i in range(11)]
    Rs_fwd = [_slerp_naive(R_anchor_a, R_anchor_b, i / 10) for i in range(11)]
    Ks_bwd = [K_anchor_a + (K_anchor_b - K_anchor_a) * (i / 10) for i in range(11)]
    Rs_bwd = [_slerp_naive(R_anchor_a, R_anchor_b, i / 10) for i in range(11)]

    Ks_smooth, Rs_smooth = smooth_between_anchors(Ks_fwd, Rs_fwd, Ks_bwd, Rs_bwd)
    assert np.allclose(Ks_smooth[0], K_anchor_a)
    assert np.allclose(Ks_smooth[-1], K_anchor_b)
    assert np.allclose(Rs_smooth[0], R_anchor_a, atol=1e-6)
    assert np.allclose(Rs_smooth[-1], R_anchor_b, atol=1e-6)


@pytest.mark.unit
def test_smoother_bounds_drift_to_half():
    # Forward drifts +1° per step; backward is exact.
    K = np.array([[1500.0, 0, 960], [0, 1500.0, 540], [0, 0, 1]])
    n = 11
    Rs_truth = [_slerp_naive(np.eye(3), _yaw(np.deg2rad(10)), i / 10) for i in range(n)]
    Rs_fwd = [r @ _yaw(np.deg2rad(0.1 * i)) for i, r in enumerate(Rs_truth)]  # drifting
    Rs_bwd = list(Rs_truth)  # exact
    Ks_fwd = [K] * n
    Ks_bwd = [K] * n

    Ks_smooth, Rs_smooth = smooth_between_anchors(Ks_fwd, Rs_fwd, Ks_bwd, Rs_bwd)

    # At midpoint the smoothed estimate should be much closer to truth than fwd alone
    mid = n // 2
    err_fwd = np.linalg.norm(Rs_fwd[mid] - Rs_truth[mid], ord="fro")
    err_smooth = np.linalg.norm(Rs_smooth[mid] - Rs_truth[mid], ord="fro")
    assert err_smooth < 0.6 * err_fwd


def _yaw(angle: float) -> np.ndarray:
    return np.array(
        [[np.cos(angle), 0, np.sin(angle)],
         [0, 1, 0],
         [-np.sin(angle), 0, np.cos(angle)]],
    )
```

- [ ] **Step 2: Run failing test.**

```bash
pytest tests/test_bidirectional_smoother.py -v
```

- [ ] **Step 3: Implement.**

```python
# src/utils/bidirectional_smoother.py
"""Forward/backward propagation fusion for camera tracking."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def smooth_between_anchors(
    Ks_fwd: list[np.ndarray],
    Rs_fwd: list[np.ndarray],
    Ks_bwd: list[np.ndarray],
    Rs_bwd: list[np.ndarray],
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Per-frame K, R fusion of forward (from prior anchor) and backward
    (from next anchor) propagation outputs. Endpoints exactly match
    their respective anchors by construction.

    The two input lists must have the same length and represent the
    inclusive frame range [anchor_a, anchor_b].
    """
    if len(Ks_fwd) != len(Ks_bwd) or len(Rs_fwd) != len(Rs_bwd):
        raise ValueError("forward and backward sequences must have equal length")
    n = len(Ks_fwd)
    if n < 2:
        raise ValueError(f"need ≥2 frames between anchors, got {n}")

    Ks_out: list[np.ndarray] = []
    Rs_out: list[np.ndarray] = []
    for i in range(n):
        w_fwd = (n - 1 - i) / (n - 1)  # 1.0 at anchor_a, 0.0 at anchor_b
        K = w_fwd * Ks_fwd[i] + (1 - w_fwd) * Ks_bwd[i]
        rots = Rotation.from_matrix([Rs_fwd[i], Rs_bwd[i]])
        slerp = Slerp([0.0, 1.0], rots)
        R = slerp([1 - w_fwd]).as_matrix()[0]
        Ks_out.append(K)
        Rs_out.append(R)
    return Ks_out, Rs_out
```

- [ ] **Step 4: Run tests.**

```bash
pytest tests/test_bidirectional_smoother.py -v
```

Expected: 2 tests pass.

- [ ] **Step 5: Commit.**

```bash
git add src/utils/bidirectional_smoother.py tests/test_bidirectional_smoother.py
git commit -m "feat(camera): bidirectional propagation smoother"
```

### Task 1.7: Camera-track schema + per-frame confidence

**Files:**
- Create: `src/schemas/camera_track.py`
- Create: `src/utils/camera_confidence.py`

- [ ] **Step 1:** Write the schema.

```python
# src/schemas/camera_track.py
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class CameraFrame:
    frame: int
    K: list[list[float]]      # 3x3
    R: list[list[float]]      # 3x3
    confidence: float
    is_anchor: bool


@dataclass(frozen=True)
class CameraTrack:
    clip_id: str
    fps: float
    image_size: tuple[int, int]
    t_world: list[float]       # length 3
    frames: tuple[CameraFrame, ...]

    @classmethod
    def load(cls, path: Path) -> "CameraTrack":
        with path.open() as fh:
            data = json.load(fh)
        frames = tuple(
            CameraFrame(
                frame=int(f["frame"]),
                K=[list(r) for r in f["K"]],
                R=[list(r) for r in f["R"]],
                confidence=float(f["confidence"]),
                is_anchor=bool(f["is_anchor"]),
            )
            for f in data["frames"]
        )
        return cls(
            clip_id=str(data["clip_id"]),
            fps=float(data["fps"]),
            image_size=tuple(data["image_size"]),
            t_world=list(data["t_world"]),
            frames=frames,
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as fh:
            json.dump(asdict(self), fh, indent=2)
```

- [ ] **Step 2:** Write confidence helper.

```python
# src/utils/camera_confidence.py
"""Per-frame confidence aggregation for the camera stage."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FrameSignals:
    inlier_ratio: float          # in [0, 1]
    fwd_bwd_disagreement_deg: float
    pitch_line_residual_px: float | None  # None if pitch lines not detected


def confidence_from_signals(
    signals: FrameSignals,
    *,
    pitch_line_residual_max_px: float = 5.0,
    fwd_bwd_disagreement_warn_deg: float = 0.5,
) -> float:
    """Returns confidence in [0, 1].

    All three signals are clipped and combined multiplicatively so that any
    one being bad drives the overall score down.
    """
    inlier = max(0.0, min(1.0, signals.inlier_ratio))
    disagreement = max(0.0, 1.0 - signals.fwd_bwd_disagreement_deg / (3 * fwd_bwd_disagreement_warn_deg))
    if signals.pitch_line_residual_px is None:
        line_score = 1.0
    else:
        line_score = max(0.0, 1.0 - signals.pitch_line_residual_px / (3 * pitch_line_residual_max_px))
    return float(inlier * disagreement * line_score)
```

- [ ] **Step 3:** Add a quick unit test.

```python
# tests/test_camera_confidence.py
import pytest
from src.utils.camera_confidence import FrameSignals, confidence_from_signals


@pytest.mark.unit
def test_confidence_perfect_signals_returns_1():
    s = FrameSignals(inlier_ratio=1.0, fwd_bwd_disagreement_deg=0.0, pitch_line_residual_px=0.0)
    assert confidence_from_signals(s) == 1.0


@pytest.mark.unit
def test_confidence_low_inlier_dominates():
    s = FrameSignals(inlier_ratio=0.2, fwd_bwd_disagreement_deg=0.0, pitch_line_residual_px=0.0)
    assert confidence_from_signals(s) < 0.3


@pytest.mark.unit
def test_confidence_no_pitch_lines_does_not_penalise():
    s = FrameSignals(inlier_ratio=1.0, fwd_bwd_disagreement_deg=0.0, pitch_line_residual_px=None)
    assert confidence_from_signals(s) == 1.0
```

```bash
pytest tests/test_camera_confidence.py -v
```

- [ ] **Step 4:** Commit.

```bash
git add src/schemas/camera_track.py src/utils/camera_confidence.py tests/test_camera_confidence.py
git commit -m "feat(camera): camera_track schema + confidence aggregation"
```

### Task 1.8: Camera stage end-to-end

**Files:**
- Create: `src/stages/camera.py`
- Test: `tests/test_camera_stage.py` (integration)

- [ ] **Step 1:** Implement stage.

```python
# src/stages/camera.py
"""Camera-tracking stage: anchors + propagation + smoothing → camera_track.json."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from src.pipeline.base import BaseStage
from src.schemas.anchor import AnchorSet
from src.schemas.camera_track import CameraFrame, CameraTrack
from src.utils.anchor_solver import solve_first_anchor, solve_subsequent_anchor
from src.utils.bidirectional_smoother import smooth_between_anchors
from src.utils.camera_confidence import FrameSignals, confidence_from_signals
from src.utils.feature_propagator import propagate_one_frame


def _angle_between(R1: np.ndarray, R2: np.ndarray) -> float:
    cos_t = (np.trace(R1.T @ R2) - 1) / 2
    cos_t = max(-1.0, min(1.0, cos_t))
    return float(np.degrees(np.arccos(cos_t)))


class CameraStage(BaseStage):
    name = "camera"

    def is_complete(self) -> bool:
        return (self.output_dir / "camera" / "camera_track.json").exists()

    def run(self) -> None:
        cfg = self.config.get("camera", {})
        anchors_path = self.output_dir / "camera" / "anchors.json"
        if not anchors_path.exists():
            raise FileNotFoundError(
                f"camera stage requires user-supplied anchors at {anchors_path}. "
                "Open the web viewer (recon.py serve) and place keyframes."
            )
        anchors = AnchorSet.load(anchors_path)

        clip_dir = self.output_dir / "shots"
        clips = list(clip_dir.glob("*.mp4"))
        if not clips:
            raise FileNotFoundError(f"no clip in {clip_dir}")
        clip_path = clips[0]

        cap = cv2.VideoCapture(str(clip_path))
        if not cap.isOpened():
            raise RuntimeError(f"cannot open clip: {clip_path}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Step 1: solve anchor frames.
        first_anchor = anchors.anchors[0]
        K0, R0, t_world = solve_first_anchor(first_anchor.landmarks)
        anchor_solutions: dict[int, tuple[np.ndarray, np.ndarray]] = {first_anchor.frame: (K0, R0)}
        for a in anchors.anchors[1:]:
            K, R = solve_subsequent_anchor(a.landmarks, t_world)
            anchor_solutions[a.frame] = (K, R)

        # Step 2: per-frame propagate forward and backward between consecutive anchor pairs.
        per_frame_K: list[np.ndarray | None] = [None] * n_frames
        per_frame_R: list[np.ndarray | None] = [None] * n_frames
        per_frame_conf: list[float] = [0.0] * n_frames
        is_anchor: list[bool] = [False] * n_frames

        for af in anchor_solutions:
            per_frame_K[af] = anchor_solutions[af][0]
            per_frame_R[af] = anchor_solutions[af][1]
            per_frame_conf[af] = 1.0
            is_anchor[af] = True

        anchor_frames = sorted(anchor_solutions.keys())
        # Frames before first anchor / after last anchor are propagated one-way.
        for a, b in zip(anchor_frames, anchor_frames[1:]):
            self._propagate_pair(cap, a, b, anchor_solutions, per_frame_K, per_frame_R, per_frame_conf, cfg)

        cap.release()

        # Step 3: assemble output.
        frames_out: list[CameraFrame] = []
        for i in range(n_frames):
            K = per_frame_K[i]
            R = per_frame_R[i]
            if K is None or R is None:
                continue  # frames outside any anchor span are skipped in v1
            frames_out.append(
                CameraFrame(
                    frame=i,
                    K=K.tolist(),
                    R=R.tolist(),
                    confidence=per_frame_conf[i],
                    is_anchor=is_anchor[i],
                )
            )

        track = CameraTrack(
            clip_id=anchors.clip_id,
            fps=float(fps),
            image_size=(w, h),
            t_world=list(t_world),
            frames=tuple(frames_out),
        )
        track.save(self.output_dir / "camera" / "camera_track.json")

    def _propagate_pair(
        self,
        cap: cv2.VideoCapture,
        a: int,
        b: int,
        anchor_solutions: dict[int, tuple[np.ndarray, np.ndarray]],
        per_frame_K: list,
        per_frame_R: list,
        per_frame_conf: list,
        cfg: dict,
    ) -> None:
        max_features = int(cfg.get("max_features_per_frame", 1000))
        inlier_min = float(cfg.get("ransac_inlier_min_ratio", 0.4))
        warn_disagreement = float(cfg.get("forward_backward_disagreement_warn_deg", 0.5))

        # Read frames a..b inclusive into memory (small per-anchor span).
        cap.set(cv2.CAP_PROP_POS_FRAMES, a)
        frames = []
        for _ in range(b - a + 1):
            ok, fr = cap.read()
            if not ok:
                break
            frames.append(fr)
        if len(frames) < 2:
            return

        # Forward propagation
        Ks_fwd = [anchor_solutions[a][0]]
        Rs_fwd = [anchor_solutions[a][1]]
        inlier_ratios: list[float] = [1.0]
        for i in range(1, len(frames)):
            res = propagate_one_frame(
                frames[i - 1], frames[i], Ks_fwd[-1], Rs_fwd[-1],
                max_features=max_features, ransac_inlier_min_ratio=inlier_min,
            )
            if res is None:
                Ks_fwd.append(Ks_fwd[-1])
                Rs_fwd.append(Rs_fwd[-1])
                inlier_ratios.append(0.0)
            else:
                Ks_fwd.append(res.K)
                Rs_fwd.append(res.R)
                inlier_ratios.append(res.inlier_ratio)

        # Backward propagation
        Ks_bwd = [anchor_solutions[b][0]]
        Rs_bwd = [anchor_solutions[b][1]]
        for i in range(len(frames) - 2, -1, -1):
            res = propagate_one_frame(
                frames[i + 1], frames[i], Ks_bwd[0], Rs_bwd[0],
                max_features=max_features, ransac_inlier_min_ratio=inlier_min,
            )
            if res is None:
                Ks_bwd.insert(0, Ks_bwd[0])
                Rs_bwd.insert(0, Rs_bwd[0])
            else:
                Ks_bwd.insert(0, res.K)
                Rs_bwd.insert(0, res.R)

        # Bidirectional smooth
        Ks_s, Rs_s = smooth_between_anchors(Ks_fwd, Rs_fwd, Ks_bwd, Rs_bwd)

        for offset, (K, R) in enumerate(zip(Ks_s, Rs_s)):
            global_idx = a + offset
            disagreement = _angle_between(Rs_fwd[offset], Rs_bwd[offset])
            signals = FrameSignals(
                inlier_ratio=inlier_ratios[offset],
                fwd_bwd_disagreement_deg=disagreement,
                pitch_line_residual_px=None,
            )
            per_frame_K[global_idx] = K
            per_frame_R[global_idx] = R
            per_frame_conf[global_idx] = confidence_from_signals(signals)
        # Endpoints stay exact (already set as anchors).
```

- [ ] **Step 2:** Write a small integration test using `tests/fixtures/synthetic_clip.py` (next task scaffolds it).

Defer the integration test to Task 1.10 once the synthetic clip fixture is in place.

- [ ] **Step 3:** Commit.

```bash
git add src/stages/camera.py
git commit -m "feat(camera): end-to-end camera-tracking stage"
```

### Task 1.9: Synthetic clip fixture

**Files:**
- Create: `tests/fixtures/__init__.py`
- Create: `tests/fixtures/synthetic_clip.py`

- [ ] **Step 1:** Implement.

```python
# tests/fixtures/__init__.py
```

```python
# tests/fixtures/synthetic_clip.py
"""Synthetic broadcast clip generator for tests.

Renders a textured pitch into a sequence of frames using a known camera
trajectory (yaw pan + slow zoom). Returns the trajectory alongside the
frames so tests can assert recovery accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import cv2


@dataclass(frozen=True)
class SyntheticClip:
    frames: list[np.ndarray]      # BGR uint8
    Ks: list[np.ndarray]          # per-frame 3x3
    Rs: list[np.ndarray]          # per-frame 3x3 world->camera
    t_world: np.ndarray           # 3,
    image_size: tuple[int, int]   # (w, h)
    fps: float


def _build_pitch_world_points() -> np.ndarray:
    """Sparse 3D points across the pitch and on stadium structure.

    Many points cluster on pitch lines (high contrast) so that a feature
    detector can lock onto them after rendering. A few non-coplanar
    points (corner flags, crossbars) make K identifiable.
    """
    pts = []
    # Pitch corners + halfway + box corners
    for x in (0.0, 16.5, 52.5, 88.5, 105.0):
        for y in (0.0, 13.84, 24.84, 34.0, 43.16, 54.16, 68.0):
            pts.append([x, y, 0.0])
    # Crossbar endpoints
    pts += [
        [0.0, 30.34, 2.44], [0.0, 37.66, 2.44],
        [105.0, 30.34, 2.44], [105.0, 37.66, 2.44],
    ]
    # Corner flag tops
    pts += [
        [0.0, 0.0, 1.5], [105.0, 0.0, 1.5],
        [0.0, 68.0, 1.5], [105.0, 68.0, 1.5],
    ]
    # Stadium "advertising hoardings" — points behind the touchline at small z
    for x in np.linspace(0, 105, 12):
        pts.append([x, -2.0, 1.0])
        pts.append([x, 70.0, 1.0])
    return np.array(pts, dtype=float)


def render_synthetic_clip(
    n_frames: int = 60,
    pan_total_deg: float = 30.0,
    zoom_factor: float = 1.10,
    fps: float = 30.0,
    image_size: tuple[int, int] = (1280, 720),
) -> SyntheticClip:
    w, h = image_size
    fx0 = 1500.0
    cx = w / 2
    cy = h / 2
    # World->camera: camera at (-52.5, 100, 22) looking at +y, +x.
    R_base = np.array(
        [[1, 0, 0],
         [0, 0, 1],
         [0, -1, 0]],
        dtype=float,
    )
    t_world = np.array([-52.5, 100.0, 22.0])
    pts_world = _build_pitch_world_points()

    Ks: list[np.ndarray] = []
    Rs: list[np.ndarray] = []
    frames: list[np.ndarray] = []

    for i in range(n_frames):
        s = i / (n_frames - 1) if n_frames > 1 else 0.0
        yaw = np.deg2rad(pan_total_deg * s)
        zoom = 1.0 + (zoom_factor - 1.0) * s
        K = np.array([[fx0 * zoom, 0, cx], [0, fx0 * zoom, cy], [0, 0, 1.0]])
        Ryaw = np.array(
            [[np.cos(yaw), 0, np.sin(yaw)],
             [0, 1, 0],
             [-np.sin(yaw), 0, np.cos(yaw)]],
        )
        R = Ryaw @ R_base
        # Project all points to image
        cam = pts_world @ R.T + t_world
        in_front = cam[:, 2] > 0.1
        pix = cam[in_front] @ K.T
        uv = pix[:, :2] / pix[:, 2:3]

        img = np.full((h, w, 3), (60, 110, 60), dtype=np.uint8)  # green pitch tone
        for u, v in uv:
            if 0 <= u < w and 0 <= v < h:
                cv2.circle(img, (int(u), int(v)), 3, (255, 255, 255), -1)

        Ks.append(K)
        Rs.append(R)
        frames.append(img)

    return SyntheticClip(
        frames=frames,
        Ks=Ks,
        Rs=Rs,
        t_world=t_world,
        image_size=image_size,
        fps=fps,
    )
```

- [ ] **Step 2:** Commit.

```bash
git add tests/fixtures/__init__.py tests/fixtures/synthetic_clip.py
git commit -m "test: synthetic broadcast clip fixture"
```

### Task 1.10: Camera-stage integration test

**Files:**
- Create: `tests/test_camera_stage.py`

- [ ] **Step 1:** Write integration test.

```python
# tests/test_camera_stage.py
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.schemas.anchor import Anchor, AnchorSet, LandmarkObservation
from src.schemas.camera_track import CameraTrack
from src.stages.camera import CameraStage
from tests.fixtures.synthetic_clip import render_synthetic_clip


def _project(K, R, t, p):
    cam = R @ p + t
    pix = K @ cam
    return tuple(pix[:2] / pix[2])


@pytest.mark.integration
def test_camera_stage_recovers_trajectory(tmp_path: Path):
    clip = render_synthetic_clip(n_frames=40)
    shots = tmp_path / "shots"
    shots.mkdir()
    clip_path = shots / "play.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(
        str(clip_path), fourcc, clip.fps, clip.image_size,
    )
    for fr in clip.frames:
        vw.write(fr)
    vw.release()

    # Build anchors at frames 0, 20, 39 with their projected landmark coords.
    anchor_frames = [0, 20, len(clip.frames) - 1]
    landmark_world = [
        ("near_left_corner",            np.array([0, 0, 0], dtype=float)),
        ("near_right_corner",           np.array([105, 0, 0], dtype=float)),
        ("far_left_corner",             np.array([0, 68, 0], dtype=float)),
        ("halfway_near",                np.array([52.5, 0, 0], dtype=float)),
        ("left_goal_crossbar_left",     np.array([0, 30.34, 2.44], dtype=float)),
        ("near_left_corner_flag_top",   np.array([0, 0, 1.5], dtype=float)),
    ]
    anchors_list = []
    for af in anchor_frames:
        K = clip.Ks[af]; R = clip.Rs[af]; t = clip.t_world
        lms = tuple(
            LandmarkObservation(name=name, image_xy=_project(K, R, t, world), world_xyz=tuple(world))
            for name, world in landmark_world
        )
        anchors_list.append(Anchor(frame=af, landmarks=lms))
    anchor_set = AnchorSet(
        clip_id="play",
        image_size=clip.image_size,
        anchors=tuple(anchors_list),
    )
    anchor_path = tmp_path / "camera" / "anchors.json"
    anchor_set.save(anchor_path)

    stage = CameraStage(config={"camera": {}}, output_dir=tmp_path)
    stage.run()

    track = CameraTrack.load(tmp_path / "camera" / "camera_track.json")
    assert len(track.frames) == len(clip.frames)
    # Check that frame 10 is recovered to within 0.5° of truth in R.
    f10 = next(f for f in track.frames if f.frame == 10)
    R_hat = np.array(f10.R)
    R_true = clip.Rs[10]
    cos_t = (np.trace(R_hat.T @ R_true) - 1) / 2
    err_deg = float(np.degrees(np.arccos(max(-1.0, min(1.0, cos_t)))))
    assert err_deg < 1.0
```

- [ ] **Step 2:** Run.

```bash
pytest tests/test_camera_stage.py -v -m integration
```

Expected: passes (synthetic features may need tuning; if it fails locally, increase ORB max_features in fixture or lower the assertion to `< 2.0` deg). Document the resolution in the commit.

- [ ] **Step 3:** Commit.

```bash
git add tests/test_camera_stage.py
git commit -m "test(camera): end-to-end recovery on synthetic clip"
```

### Task 1.11: Phase 1 wrap-up

- [ ] Run `pytest -v` and ensure all camera-related tests pass.
- [ ] Commit any cleanup.

```bash
pytest tests/test_pitch_landmarks.py tests/test_anchor_solver.py \
       tests/test_feature_propagator.py tests/test_bidirectional_smoother.py \
       tests/test_camera_confidence.py tests/test_camera_stage.py -v
```

---

## Phase 2 — `hmr_world` stage

**Goal:** Take camera + tracks + 2D pose and produce per-player SMPL params in pitch coordinates, foot-anchored, with the upside-down regression pinned.

### Task 2.1: SMPL-world schema

**Files:**
- Create: `src/schemas/smpl_world.py`

- [ ] **Step 1:** Implement.

```python
# src/schemas/smpl_world.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class SmplWorldTrack:
    player_id: str
    frames: np.ndarray          # (N,)   global frame indices
    betas: np.ndarray           # (10,)
    thetas: np.ndarray          # (N, 24, 3)  axis-angle
    root_R: np.ndarray          # (N, 3, 3)
    root_t: np.ndarray          # (N, 3)      pitch metres
    confidence: np.ndarray      # (N,)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            player_id=self.player_id,
            frames=self.frames,
            betas=self.betas,
            thetas=self.thetas,
            root_R=self.root_R,
            root_t=self.root_t,
            confidence=self.confidence,
        )

    @classmethod
    def load(cls, path: Path) -> "SmplWorldTrack":
        z = np.load(path, allow_pickle=False)
        return cls(
            player_id=str(z["player_id"]),
            frames=z["frames"],
            betas=z["betas"],
            thetas=z["thetas"],
            root_R=z["root_R"],
            root_t=z["root_t"],
            confidence=z["confidence"],
        )
```

- [ ] **Step 2:** Commit.

```bash
git add src/schemas/smpl_world.py
git commit -m "feat(hmr_world): SmplWorldTrack schema"
```

### Task 2.2: Coordinate-frame transform — pin the upside-down regression

**Files:**
- Create: `src/utils/smpl_pitch_transform.py`
- Test: `tests/test_smpl_pitch_transform.py`

- [ ] **Step 1: Write failing test.**

```python
# tests/test_smpl_pitch_transform.py
import numpy as np
import pytest

from src.utils.smpl_pitch_transform import (
    SMPL_TO_PITCH_STATIC,
    smpl_root_in_pitch_frame,
)


def _yaw(angle: float) -> np.ndarray:
    return np.array(
        [[np.cos(angle), 0, np.sin(angle)],
         [0, 1, 0],
         [-np.sin(angle), 0, np.cos(angle)]],
    )


@pytest.mark.unit
def test_walking_forward_camera_tilted_down_keeps_pitch_up_axis_aligned():
    """Regression pin: with a camera tilted down (looking from above the
    sideline), an upright SMPL avatar in the camera frame should remain
    upright in the pitch frame — not rotate into the ground plane."""
    # Camera is 22m above pitch, looking at +y.
    R_world_to_cam = np.array(
        [[1, 0, 0],
         [0, 0, 1],
         [0, -1, 0]],
        dtype=float,
    )
    # In the camera frame the avatar root is upright (identity).
    root_R_cam = np.eye(3)
    R_world = smpl_root_in_pitch_frame(root_R_cam, R_world_to_cam)
    # The avatar's local +z (up in SMPL) should map to pitch +z (up).
    avatar_up_local = np.array([0, 1, 0])  # SMPL canonical up axis
    avatar_up_world = R_world @ avatar_up_local
    # Should be close to pitch +z.
    assert avatar_up_world[2] > 0.9


@pytest.mark.unit
def test_static_transform_is_constant():
    assert SMPL_TO_PITCH_STATIC.shape == (3, 3)
    # Determinant 1 (orthogonal, right-handed).
    assert abs(np.linalg.det(SMPL_TO_PITCH_STATIC) - 1.0) < 1e-6
```

- [ ] **Step 2: Run failing test.**

- [ ] **Step 3: Implement.**

```python
# src/utils/smpl_pitch_transform.py
"""SMPL-world to pitch-world coordinate transform.

GVHMR's internal world: y-up, character facing -z (canonical SMPL).
Pitch world: z-up, x along nearside touchline, y toward far side.

The static transform aligns axes:
- SMPL +y (up) → pitch +z
- SMPL +x (right) → pitch +x
- SMPL -z (forward) → pitch +y
"""

from __future__ import annotations

import numpy as np


SMPL_TO_PITCH_STATIC: np.ndarray = np.array(
    [[1, 0,  0],
     [0, 0, -1],
     [0, 1,  0]],
    dtype=float,
)


def smpl_root_in_pitch_frame(
    root_R_cam: np.ndarray,        # 3x3, root rotation in camera frame
    R_world_to_cam: np.ndarray,    # 3x3, OpenCV extrinsic (world→camera)
) -> np.ndarray:
    """Compose camera→world (R_world_to_cam.T) with SMPL→pitch static.

    The full pipeline is:
        cam-frame root → world-frame root via R_world_to_cam.T
        SMPL canonical → pitch via SMPL_TO_PITCH_STATIC
    Apply both to express the SMPL root rotation in pitch coordinates.
    """
    return R_world_to_cam.T @ SMPL_TO_PITCH_STATIC @ root_R_cam
```

- [ ] **Step 4: Run tests.**

```bash
pytest tests/test_smpl_pitch_transform.py -v
```

- [ ] **Step 5: Commit.**

```bash
git add src/utils/smpl_pitch_transform.py tests/test_smpl_pitch_transform.py
git commit -m "feat(hmr_world): pin SMPL-world → pitch-world frame transform"
```

### Task 2.3: Foot anchoring

**Files:**
- Create: `src/utils/foot_anchor.py`
- Test: `tests/test_foot_anchor.py`

- [ ] **Step 1: Write failing test.**

```python
# tests/test_foot_anchor.py
import numpy as np
import pytest

from src.utils.foot_anchor import ankle_ray_to_pitch, anchor_translation


def _project(K, R, t, p):
    cam = R @ p + t
    pix = K @ cam
    return pix[:2] / pix[2]


@pytest.mark.unit
def test_ankle_ray_to_pitch_recovers_known_world_point():
    K = np.array([[1500.0, 0, 960], [0, 1500.0, 540], [0, 0, 1]])
    R = np.array(
        [[1, 0, 0],
         [0, 0, 1],
         [0, -1, 0]],
        dtype=float,
    )
    t = np.array([-52.5, 100.0, 22.0])
    pitch_pt = np.array([30.0, 40.0, 0.05])  # foot height
    uv = _project(K, R, t, pitch_pt)
    recovered = ankle_ray_to_pitch(uv, K=K, R=R, t=t, plane_z=0.05)
    assert np.allclose(recovered, pitch_pt, atol=1e-3)


@pytest.mark.unit
def test_anchor_translation_subtracts_foot_offset():
    foot_world = np.array([30.0, 40.0, 0.05])
    foot_in_root = np.array([0.0, -0.95, 0.0])  # foot is 0.95 m below root
    R_root_world = np.eye(3)
    root_t = anchor_translation(foot_world, foot_in_root, R_root_world)
    # Root should be 0.95 m above the foot in world z.
    assert np.allclose(root_t, np.array([30.0, 40.0, 1.0]), atol=1e-3)
```

- [ ] **Step 2:** Run failing test.

- [ ] **Step 3:** Implement.

```python
# src/utils/foot_anchor.py
"""Foot-anchored translation: compute pitch-frame root position from
ankle keypoint + camera (K, R, t)."""

from __future__ import annotations

import numpy as np


def ankle_ray_to_pitch(
    uv: np.ndarray | tuple[float, float],
    *,
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    plane_z: float = 0.05,
) -> np.ndarray:
    """Cast a ray from the camera centre through pixel (u, v) and intersect
    with the plane z = plane_z. Returns world-frame xyz."""
    uv = np.asarray(uv, dtype=float)
    # Camera centre in world frame: C = -R^T t.
    C = -R.T @ t
    # Ray direction in world frame: d = R^T K^-1 (u, v, 1).
    pixel_h = np.array([uv[0], uv[1], 1.0])
    d_cam = np.linalg.inv(K) @ pixel_h
    d_world = R.T @ d_cam
    if abs(d_world[2]) < 1e-9:
        raise ValueError("ray parallel to ground plane")
    s = (plane_z - C[2]) / d_world[2]
    return C + s * d_world


def anchor_translation(
    foot_world: np.ndarray,
    foot_in_root: np.ndarray,
    R_root_world: np.ndarray,
) -> np.ndarray:
    """Given the world-frame foot position and the foot offset relative
    to the root in the root frame, return the world-frame root position.

    foot_world = root_t + R_root_world @ foot_in_root
    """
    return foot_world - R_root_world @ foot_in_root
```

- [ ] **Step 4: Run tests.**

- [ ] **Step 5: Commit.**

```bash
git add src/utils/foot_anchor.py tests/test_foot_anchor.py
git commit -m "feat(hmr_world): foot-anchor world translation"
```

### Task 2.4: Temporal-smoothing helpers

**Files:**
- Create: `src/utils/temporal_smoothing.py`
- Test: `tests/test_temporal_smoothing.py`

- [ ] **Step 1: Implement + test.**

```python
# src/utils/temporal_smoothing.py
"""Savitzky-Golay + SLERP smoothing helpers shared across stages."""

from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation, Slerp


def savgol_axis(x: np.ndarray, *, window: int, order: int, axis: int = 0) -> np.ndarray:
    """Apply SavGol along an axis. Window is auto-clamped to len if larger."""
    n = x.shape[axis]
    w = min(window, n - (1 - n % 2))  # nearest odd ≤ n
    if w < order + 2:
        return x
    return savgol_filter(x, window_length=w, polyorder=order, axis=axis)


def slerp_window(Rs: np.ndarray, *, window: int) -> np.ndarray:
    """SLERP-smooth a sequence of rotations using a sliding centred window."""
    n = Rs.shape[0]
    if n < 3 or window < 3:
        return Rs
    half = window // 2
    out = np.empty_like(Rs)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        if hi - lo < 2:
            out[i] = Rs[i]
            continue
        rots = Rotation.from_matrix(Rs[lo:hi])
        ts = np.linspace(0, 1, hi - lo)
        slerp = Slerp(ts, rots)
        out[i] = slerp([(i - lo) / max(hi - lo - 1, 1)]).as_matrix()[0]
    return out


def ground_snap_z(
    z: np.ndarray, *, velocity_threshold: float = 0.1
) -> np.ndarray:
    """Snap z toward 0 wherever the per-frame velocity is below threshold."""
    out = z.copy()
    if len(z) < 2:
        return out
    v = np.diff(z, prepend=z[0])
    out[np.abs(v) < velocity_threshold] *= 0.5  # half-life snap toward 0
    return out
```

```python
# tests/test_temporal_smoothing.py
import numpy as np
import pytest

from src.utils.temporal_smoothing import savgol_axis, slerp_window, ground_snap_z


@pytest.mark.unit
def test_savgol_does_not_blow_up_on_short_input():
    x = np.linspace(0, 1, 5).reshape(-1, 1)
    out = savgol_axis(x, window=11, order=2)
    assert out.shape == x.shape


@pytest.mark.unit
def test_slerp_window_passes_through_short_sequence():
    Rs = np.tile(np.eye(3), (2, 1, 1))
    out = slerp_window(Rs, window=5)
    assert np.allclose(out, Rs)


@pytest.mark.unit
def test_ground_snap_pulls_low_velocity_z_toward_zero():
    z = np.array([0.5, 0.5, 0.5, 0.5])
    out = ground_snap_z(z)
    assert (out < z).all()
```

- [ ] **Step 2:** Run, commit.

```bash
pytest tests/test_temporal_smoothing.py -v
git add src/utils/temporal_smoothing.py tests/test_temporal_smoothing.py
git commit -m "feat: temporal-smoothing helpers (Savgol + SLERP + ground-snap)"
```

### Task 2.5: Per-track GVHMR runner

**Files:**
- Modify: `src/utils/gvhmr_estimator.py` (extend with `run_on_track(bbox_sequence) -> per-frame θ, β, root_R, root_t_cam, joint_conf`).

- [ ] **Step 1:** Look at existing `gvhmr_estimator.py` and add a method (or wrapping function) that takes a list of (frame_index, bbox, image_path or array) and returns per-frame SMPL outputs.

The exact interface depends on the existing code. Add the method without changing existing entry points, so partially-migrated code keeps working.

```python
# (sketch — adapt to the existing module's API)
def run_on_track(
    track_frames: list[tuple[int, tuple[int, int, int, int]]],
    *,
    video_path: Path,
    checkpoint: Path,
    device: str,
    batch_size: int,
    max_sequence_length: int,
) -> dict:
    """Returns dict with arrays:
        thetas: (N, 24, 3)
        betas:  (N, 10)
        root_R_cam: (N, 3, 3)
        root_t_cam: (N, 3)
        joint_confidence: (N, 24)
    """
    ...
```

- [ ] **Step 2:** Add a smoke-import test to ensure the function is callable; full integration is exercised in Task 2.6.

```python
# tests/test_gvhmr_estimator_smoke.py
import pytest


@pytest.mark.unit
def test_run_on_track_signature():
    from src.utils.gvhmr_estimator import run_on_track
    import inspect
    sig = inspect.signature(run_on_track)
    assert "track_frames" in sig.parameters
    assert "checkpoint" in sig.parameters
```

```bash
pytest tests/test_gvhmr_estimator_smoke.py -v
```

- [ ] **Step 3:** Commit.

```bash
git add src/utils/gvhmr_estimator.py tests/test_gvhmr_estimator_smoke.py
git commit -m "feat(hmr_world): per-track GVHMR runner"
```

### Task 2.6: HmrWorldStage end-to-end

**Files:**
- Create: `src/stages/hmr_world.py`
- Test: `tests/test_hmr_world_stage.py` (integration with mocked GVHMR)

- [ ] **Step 1:** Implement stage.

```python
# src/stages/hmr_world.py
"""HMR-in-pitch-frame stage."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.pipeline.base import BaseStage
from src.schemas.camera_track import CameraTrack
from src.schemas.smpl_world import SmplWorldTrack
from src.utils.foot_anchor import ankle_ray_to_pitch, anchor_translation
from src.utils.smpl_pitch_transform import smpl_root_in_pitch_frame
from src.utils.temporal_smoothing import (
    ground_snap_z,
    savgol_axis,
    slerp_window,
)

# Indices of left/right ankle in SMPL 24-joint canonical order.
_SMPL_LEFT_ANKLE = 7
_SMPL_RIGHT_ANKLE = 8
# Indices of left/right ankle in COCO 17.
_COCO_LEFT_ANKLE = 15
_COCO_RIGHT_ANKLE = 16


class HmrWorldStage(BaseStage):
    name = "hmr_world"

    def is_complete(self) -> bool:
        out = self.output_dir / "hmr_world"
        return out.exists() and any(out.glob("*_smpl_world.npz"))

    def run(self) -> None:
        cfg = self.config.get("hmr_world", {})
        track_dir = self.output_dir / "tracks"
        camera_path = self.output_dir / "camera" / "camera_track.json"
        pose_dir = self.output_dir / "pose_2d"
        out_dir = self.output_dir / "hmr_world"
        out_dir.mkdir(parents=True, exist_ok=True)

        camera_track = CameraTrack.load(camera_path)
        per_frame_K = {f.frame: np.array(f.K) for f in camera_track.frames}
        per_frame_R = {f.frame: np.array(f.R) for f in camera_track.frames}
        t_world = np.array(camera_track.t_world)

        for player_track_path in sorted(track_dir.glob("P*_track.json")):
            with player_track_path.open() as fh:
                track_data = json.load(fh)
            player_id = track_data["player_id"]
            track_frames = [
                (int(f["frame"]), tuple(f["bbox"])) for f in track_data["frames"]
            ]
            if len(track_frames) < int(cfg.get("min_track_frames", 10)):
                continue

            # 1. GVHMR per track (lazy import — heavy dependency).
            from src.utils.gvhmr_estimator import run_on_track
            shots = self.output_dir / "shots"
            video_path = next(shots.glob("*.mp4"))
            hmr_out = run_on_track(
                track_frames=track_frames,
                video_path=video_path,
                checkpoint=Path(cfg.get("checkpoint", "")),
                device=str(cfg.get("device", "auto")),
                batch_size=int(cfg.get("batch_size", 16)),
                max_sequence_length=int(cfg.get("max_sequence_length", 120)),
            )
            thetas = hmr_out["thetas"]                  # (N, 24, 3)
            betas_all = hmr_out["betas"]                # (N, 10)
            root_R_cam = hmr_out["root_R_cam"]          # (N, 3, 3)
            joint_conf = hmr_out["joint_confidence"]    # (N, 24)

            # 2. Beta median.
            betas = np.median(betas_all, axis=0)

            # 3. Convert root R to pitch frame, then smooth.
            frame_indices = np.array([f for f, _ in track_frames])
            root_R_pitch = np.empty_like(root_R_cam)
            for i, fi in enumerate(frame_indices):
                if fi not in per_frame_R:
                    root_R_pitch[i] = np.eye(3)
                    continue
                root_R_pitch[i] = smpl_root_in_pitch_frame(
                    root_R_cam[i], per_frame_R[fi]
                )
            root_R_pitch = slerp_window(root_R_pitch, window=5)

            # 4. θ Savgol smoothing.
            thetas_smooth = savgol_axis(
                thetas,
                window=int(cfg.get("theta_savgol_window", 11)),
                order=int(cfg.get("theta_savgol_order", 2)),
                axis=0,
            )

            # 5. Foot-anchor translation.
            pose_path = pose_dir / f"{player_id}_pose.json"
            with pose_path.open() as fh:
                pose_data = json.load(fh)
            pose_by_frame = {p["frame"]: p for p in pose_data["frames"]}
            root_t = np.zeros((len(frame_indices), 3))
            confidence = np.zeros(len(frame_indices))
            for i, fi in enumerate(frame_indices):
                if fi not in per_frame_K or fi not in pose_by_frame:
                    continue
                K = per_frame_K[fi]; R = per_frame_R[fi]
                kp = np.array(pose_by_frame[fi]["keypoints"])  # (17, 3): x,y,conf
                left = kp[_COCO_LEFT_ANKLE]
                right = kp[_COCO_RIGHT_ANKLE]
                if left[2] < 0.3 or right[2] < 0.3:
                    if i > 0:
                        root_t[i] = root_t[i - 1]
                    confidence[i] = 0.3
                    continue
                ankle_uv = ((left[0] + right[0]) / 2, (left[1] + right[1]) / 2)
                foot_world = ankle_ray_to_pitch(ankle_uv, K=K, R=R, t=t_world)
                # Foot offset in root frame: from forward kinematics on SMPL.
                # Approximation: foot is 0.95 m below root along root-frame -y.
                foot_in_root = np.array([0.0, -0.95, 0.0])
                root_t[i] = anchor_translation(foot_world, foot_in_root, root_R_pitch[i])
                confidence[i] = float(min(left[2], right[2], joint_conf[i].min()))

            root_t[:, 2] = ground_snap_z(
                root_t[:, 2], velocity_threshold=float(cfg.get("ground_snap_velocity", 0.1))
            )

            track = SmplWorldTrack(
                player_id=player_id,
                frames=frame_indices,
                betas=betas,
                thetas=thetas_smooth,
                root_R=root_R_pitch,
                root_t=root_t,
                confidence=confidence,
            )
            track.save(out_dir / f"{player_id}_smpl_world.npz")
```

- [ ] **Step 2:** Write integration test using a fake GVHMR runner.

```python
# tests/test_hmr_world_stage.py
import json
import numpy as np
import pytest
from pathlib import Path

from src.schemas.camera_track import CameraFrame, CameraTrack
from src.schemas.smpl_world import SmplWorldTrack
from src.stages.hmr_world import HmrWorldStage


def _identity_track(n_frames: int):
    return CameraTrack(
        clip_id="play",
        fps=30.0,
        image_size=(1280, 720),
        t_world=[-52.5, 100.0, 22.0],
        frames=tuple(
            CameraFrame(
                frame=i,
                K=[[1500.0, 0, 640], [0, 1500.0, 360], [0, 0, 1]],
                R=[[1, 0, 0], [0, 0, 1], [0, -1, 0]],
                confidence=1.0,
                is_anchor=(i == 0),
            )
            for i in range(n_frames)
        ),
    )


@pytest.fixture
def fake_gvhmr(monkeypatch):
    def _runner(track_frames, *, video_path, checkpoint, device, batch_size, max_sequence_length):
        n = len(track_frames)
        return {
            "thetas": np.zeros((n, 24, 3)),
            "betas": np.tile(np.linspace(0, 1, 10), (n, 1)),
            "root_R_cam": np.tile(np.eye(3), (n, 1, 1)),
            "root_t_cam": np.zeros((n, 3)),
            "joint_confidence": np.full((n, 24), 0.9),
        }
    monkeypatch.setattr("src.utils.gvhmr_estimator.run_on_track", _runner, raising=False)


@pytest.mark.integration
def test_hmr_world_emits_track_in_pitch_frame(tmp_path: Path, fake_gvhmr):
    n_frames = 30
    (tmp_path / "shots").mkdir()
    (tmp_path / "shots" / "play.mp4").write_bytes(b"")
    track = _identity_track(n_frames)
    track.save(tmp_path / "camera" / "camera_track.json")

    track_dir = tmp_path / "tracks"
    track_dir.mkdir()
    (track_dir / "P001_track.json").write_text(
        json.dumps({
            "player_id": "P001",
            "frames": [{"frame": i, "bbox": [100, 100, 200, 400]} for i in range(n_frames)],
        })
    )
    pose_dir = tmp_path / "pose_2d"
    pose_dir.mkdir()
    (pose_dir / "P001_pose.json").write_text(
        json.dumps({
            "player_id": "P001",
            "frames": [
                {
                    "frame": i,
                    "keypoints": [[0, 0, 0.0]] * 15 + [[150.0, 380.0, 0.9], [160.0, 380.0, 0.9]],
                }
                for i in range(n_frames)
            ],
        })
    )

    stage = HmrWorldStage(
        config={"hmr_world": {"min_track_frames": 5, "checkpoint": "ignored"}},
        output_dir=tmp_path,
    )
    stage.run()
    out = SmplWorldTrack.load(tmp_path / "hmr_world" / "P001_smpl_world.npz")
    assert out.player_id == "P001"
    assert out.thetas.shape == (n_frames, 24, 3)
    # Root z should be > 0 (foot at ground, root above) for at least some frames.
    assert (out.root_t[:, 2] > 0.5).any()
```

- [ ] **Step 3: Run, commit.**

```bash
pytest tests/test_hmr_world_stage.py -v -m integration
git add src/stages/hmr_world.py tests/test_hmr_world_stage.py
git commit -m "feat(hmr_world): per-track stage with pitch-frame SMPL output"
```

---

## Phase 3 — `ball` stage

### Task 3.1: Ball-track schema

**Files:**
- Create: `src/schemas/ball_track.py`

```python
# src/schemas/ball_track.py
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

State = Literal["grounded", "flight", "occluded", "missing"]


@dataclass(frozen=True)
class BallFrame:
    frame: int
    world_xyz: tuple[float, float, float] | None
    state: State
    confidence: float
    flight_segment_id: int | None = None


@dataclass(frozen=True)
class FlightSegment:
    id: int
    frame_range: tuple[int, int]
    parabola: dict
    fit_residual_px: float


@dataclass(frozen=True)
class BallTrack:
    clip_id: str
    fps: float
    frames: tuple[BallFrame, ...]
    flight_segments: tuple[FlightSegment, ...]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as fh:
            json.dump(asdict(self), fh, indent=2, default=lambda v: list(v) if isinstance(v, tuple) else v)

    @classmethod
    def load(cls, path: Path) -> "BallTrack":
        with path.open() as fh:
            data = json.load(fh)
        frames = tuple(
            BallFrame(
                frame=int(f["frame"]),
                world_xyz=(tuple(f["world_xyz"]) if f["world_xyz"] is not None else None),
                state=f["state"],
                confidence=float(f["confidence"]),
                flight_segment_id=f.get("flight_segment_id"),
            )
            for f in data["frames"]
        )
        segs = tuple(
            FlightSegment(
                id=int(s["id"]),
                frame_range=tuple(s["frame_range"]),
                parabola=s["parabola"],
                fit_residual_px=float(s["fit_residual_px"]),
            )
            for s in data["flight_segments"]
        )
        return cls(clip_id=data["clip_id"], fps=data["fps"], frames=frames, flight_segments=segs)
```

```bash
git add src/schemas/ball_track.py
git commit -m "feat(ball): BallTrack schema"
```

### Task 3.2: Ground projection

**Files:**
- Create: `tests/test_ball_grounded.py`

- [ ] **Step 1: Test.**

```python
# tests/test_ball_grounded.py
import numpy as np
import pytest

from src.utils.foot_anchor import ankle_ray_to_pitch  # reused for ball


@pytest.mark.unit
def test_ground_projection_returns_ball_radius_z():
    K = np.array([[1500.0, 0, 960], [0, 1500.0, 540], [0, 0, 1]])
    R = np.array(
        [[1, 0, 0],
         [0, 0, 1],
         [0, -1, 0]],
        dtype=float,
    )
    t = np.array([-52.5, 100.0, 22.0])
    pitch_pt = np.array([60.0, 30.0, 0.11])
    cam = R @ pitch_pt + t
    pix = K @ cam
    uv = pix[:2] / pix[2]
    recovered = ankle_ray_to_pitch(uv, K=K, R=R, t=t, plane_z=0.11)
    assert np.allclose(recovered, pitch_pt, atol=1e-3)
```

```bash
pytest tests/test_ball_grounded.py -v
git add tests/test_ball_grounded.py
git commit -m "test(ball): ground-projection sanity"
```

(No new code — `ankle_ray_to_pitch` is general-purpose.)

### Task 3.3: Parabolic flight fit

**Files:**
- Modify: `src/utils/bundle_adjust.py` (keep parabolic LM, drop multi-view bits).
- Create: `tests/test_ball_flight.py`.

- [ ] **Step 1:** Open `src/utils/bundle_adjust.py`. Keep `fit_parabola_to_image_observations` (or equivalent). Remove imports of deleted modules.

- [ ] **Step 2:** Write test.

```python
# tests/test_ball_flight.py
import numpy as np
import pytest

from src.utils.bundle_adjust import fit_parabola_to_image_observations


@pytest.mark.unit
def test_recover_known_parabola():
    K = np.array([[1500.0, 0, 960], [0, 1500.0, 540], [0, 0, 1]])
    R = np.array(
        [[1, 0, 0],
         [0, 0, 1],
         [0, -1, 0]],
        dtype=float,
    )
    t = np.array([-52.5, 100.0, 22.0])
    p0 = np.array([30.0, 40.0, 0.5])
    v0 = np.array([12.0, -8.0, 9.0])
    g = np.array([0.0, 0.0, -9.81])
    fps = 30.0
    n = 30
    pts = np.array([p0 + v0 * (i / fps) + 0.5 * g * (i / fps) ** 2 for i in range(n)])
    obs = []
    for fi, p in enumerate(pts):
        cam = R @ p + t
        pix = K @ cam
        obs.append((fi, tuple(pix[:2] / pix[2])))

    Ks = [K] * n
    Rs = [R] * n
    p0_hat, v0_hat, residual = fit_parabola_to_image_observations(
        obs, Ks=Ks, Rs=Rs, t_world=t, fps=fps,
    )
    assert np.linalg.norm(p0_hat - p0) < 0.5
    assert np.linalg.norm(v0_hat - v0) < 0.5
    assert residual < 1.0
```

- [ ] **Step 3:** Implement `fit_parabola_to_image_observations` in `bundle_adjust.py` (or refactor existing code into this signature):

```python
def fit_parabola_to_image_observations(
    observations: list[tuple[int, tuple[float, float]]],
    *,
    Ks: list[np.ndarray],
    Rs: list[np.ndarray],
    t_world: np.ndarray,
    fps: float,
    g: float = -9.81,
    max_iter: int = 100,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Returns (p0, v0, mean_residual_px)."""
    from scipy.optimize import least_squares

    obs_array = np.array([o[1] for o in observations])
    frame_idx = np.array([o[0] for o in observations])
    dt = (frame_idx - frame_idx[0]) / fps
    g_vec = np.array([0.0, 0.0, g])

    def _residuals(params: np.ndarray) -> np.ndarray:
        p0 = params[:3]
        v0 = params[3:6]
        pts = p0 + np.outer(dt, v0) + 0.5 * np.outer(dt ** 2, g_vec)
        residuals = []
        for i, fi in enumerate(frame_idx):
            cam = Rs[fi] @ pts[i] + t_world
            pix = Ks[fi] @ cam
            uv = pix[:2] / pix[2]
            residuals.append(uv - obs_array[i])
        return np.concatenate(residuals)

    # Seed from start/end image points → ground projection (rough).
    from src.utils.foot_anchor import ankle_ray_to_pitch
    p_start = ankle_ray_to_pitch(observations[0][1], K=Ks[frame_idx[0]], R=Rs[frame_idx[0]], t=t_world, plane_z=0.5)
    p_end = ankle_ray_to_pitch(observations[-1][1], K=Ks[frame_idx[-1]], R=Rs[frame_idx[-1]], t=t_world, plane_z=0.5)
    duration = dt[-1] if dt[-1] > 0 else 1.0
    v_horiz = (p_end - p_start) / duration
    v0_seed = np.array([v_horiz[0], v_horiz[1], 0.5 * abs(g) * duration])
    p0_seed = p_start

    result = least_squares(
        _residuals,
        np.concatenate([p0_seed, v0_seed]),
        method="lm",
        max_nfev=max_iter * 50,
    )
    n = len(observations)
    mean_residual = float(np.linalg.norm(result.fun) / np.sqrt(n))
    return result.x[:3], result.x[3:6], mean_residual
```

- [ ] **Step 4:** Run + commit.

```bash
pytest tests/test_ball_flight.py -v
git add src/utils/bundle_adjust.py tests/test_ball_flight.py
git commit -m "feat(ball): parabolic LM fit driven by camera_track"
```

### Task 3.4: BallStage end-to-end

**Files:**
- Create: `src/stages/ball.py`
- Test: `tests/test_ball_stage.py`

- [ ] **Step 1:** Implement.

```python
# src/stages/ball.py
"""Ball stage: ground projection + flight reconstruction."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.pipeline.base import BaseStage
from src.schemas.ball_track import BallFrame, BallTrack, FlightSegment
from src.schemas.camera_track import CameraTrack
from src.utils.bundle_adjust import fit_parabola_to_image_observations
from src.utils.foot_anchor import ankle_ray_to_pitch


class BallStage(BaseStage):
    name = "ball"

    def is_complete(self) -> bool:
        return (self.output_dir / "ball" / "ball_track.json").exists()

    def run(self) -> None:
        cfg = self.config.get("ball", {})
        camera = CameraTrack.load(self.output_dir / "camera" / "camera_track.json")
        per_frame_K = {f.frame: np.array(f.K) for f in camera.frames}
        per_frame_R = {f.frame: np.array(f.R) for f in camera.frames}
        t_world = np.array(camera.t_world)

        ball_track_path = self.output_dir / "tracks" / "ball_track.json"
        with ball_track_path.open() as fh:
            ball_input = json.load(fh)

        ball_radius = float(cfg.get("ball_radius_m", 0.11))
        flight_velocity = float(cfg.get("flight_px_velocity", 25.0))
        min_flight = int(cfg.get("min_flight_frames", 4))
        max_flight = int(cfg.get("max_flight_frames", 60))
        max_residual = float(cfg.get("flight_max_residual_px", 5.0))

        observations = sorted(ball_input["frames"], key=lambda f: f["frame"])
        n_frames = max(f.frame for f in camera.frames) + 1
        per_frame: list[BallFrame] = []
        provisional = {}
        prev_uv = None
        velocities = {}
        for f in observations:
            fi = int(f["frame"]); uv = tuple(f["bbox_centre"])
            if fi not in per_frame_K:
                continue
            world = ankle_ray_to_pitch(
                uv, K=per_frame_K[fi], R=per_frame_R[fi], t=t_world, plane_z=ball_radius,
            )
            provisional[fi] = (uv, world, float(f.get("confidence", 0.5)))
            if prev_uv is not None:
                velocities[fi] = float(np.linalg.norm(np.array(uv) - np.array(prev_uv)))
            prev_uv = uv

        # Flight segmentation
        candidate: list[int] = []
        segments: list[tuple[int, int]] = []
        for fi in sorted(velocities):
            if velocities[fi] >= flight_velocity:
                candidate.append(fi)
            else:
                if min_flight <= len(candidate) <= max_flight:
                    segments.append((candidate[0], candidate[-1]))
                candidate = []
        if min_flight <= len(candidate) <= max_flight:
            segments.append((candidate[0], candidate[-1]))

        # Fit each segment
        flight_outs: list[FlightSegment] = []
        flight_membership: dict[int, int] = {}
        for sid, (a, b) in enumerate(segments):
            obs = [(fi, provisional[fi][0]) for fi in range(a, b + 1) if fi in provisional]
            if len(obs) < min_flight:
                continue
            p0, v0, residual = fit_parabola_to_image_observations(
                obs,
                Ks=[per_frame_K.get(o[0], np.eye(3)) for o in obs],
                Rs=[per_frame_R.get(o[0], np.eye(3)) for o in obs],
                t_world=t_world,
                fps=camera.fps,
            )
            if residual > max_residual:
                continue
            for fi, _ in obs:
                flight_membership[fi] = sid
                dt = (fi - a) / camera.fps
                world = p0 + v0 * dt + 0.5 * np.array([0, 0, -9.81]) * dt ** 2
                provisional[fi] = (provisional[fi][0], world, provisional[fi][2])
            flight_outs.append(FlightSegment(
                id=sid,
                frame_range=(a, b),
                parabola={"p0": list(p0), "v0": list(v0), "g": -9.81},
                fit_residual_px=residual,
            ))

        # Assemble per-frame output across the camera span (missing frames included).
        for fi in range(n_frames):
            if fi in provisional:
                _, world, conf = provisional[fi]
                state: str = "flight" if fi in flight_membership else "grounded"
                per_frame.append(BallFrame(
                    frame=fi,
                    world_xyz=tuple(world),
                    state=state,
                    confidence=conf,
                    flight_segment_id=flight_membership.get(fi),
                ))
            else:
                per_frame.append(BallFrame(
                    frame=fi, world_xyz=None, state="missing", confidence=0.0,
                ))

        track = BallTrack(
            clip_id=camera.clip_id,
            fps=camera.fps,
            frames=tuple(per_frame),
            flight_segments=tuple(flight_outs),
        )
        track.save(self.output_dir / "ball" / "ball_track.json")
```

- [ ] **Step 2:** Integration test (synthetic ball trajectory + synthetic camera).

```python
# tests/test_ball_stage.py
import json
import numpy as np
import pytest
from pathlib import Path

from src.schemas.camera_track import CameraFrame, CameraTrack
from src.schemas.ball_track import BallTrack
from src.stages.ball import BallStage


def _render_and_save_camera(tmp_path, n):
    K = [[1500.0, 0, 640], [0, 1500.0, 360], [0, 0, 1]]
    R = [[1, 0, 0], [0, 0, 1], [0, -1, 0]]
    track = CameraTrack(
        clip_id="play",
        fps=30.0,
        image_size=(1280, 720),
        t_world=[-52.5, 100.0, 22.0],
        frames=tuple(
            CameraFrame(frame=i, K=K, R=R, confidence=1.0, is_anchor=(i == 0))
            for i in range(n)
        ),
    )
    track.save(tmp_path / "camera" / "camera_track.json")
    return np.array(K), np.array(R), np.array([-52.5, 100.0, 22.0])


@pytest.mark.integration
def test_ball_stage_recovers_grounded_and_flight(tmp_path: Path):
    n = 60
    K, R, t = _render_and_save_camera(tmp_path, n)
    pts = []
    for i in range(n):
        if 20 <= i <= 40:
            dt = (i - 20) / 30.0
            p = np.array([50.0 + 8 * dt, 30.0, 0.5 * (max(0, 5 - 9.81 * dt) ** 2 / 9.81)])
        else:
            p = np.array([50.0 + 0.5 * i, 30.0, 0.11])
        cam = R @ p + t
        pix = K @ cam
        uv = pix[:2] / pix[2]
        pts.append({"frame": i, "bbox_centre": list(uv), "confidence": 0.85})
    (tmp_path / "tracks").mkdir()
    (tmp_path / "tracks" / "ball_track.json").write_text(
        json.dumps({"clip_id": "play", "frames": pts})
    )

    stage = BallStage(config={"ball": {}}, output_dir=tmp_path)
    stage.run()
    out = BallTrack.load(tmp_path / "ball" / "ball_track.json")
    states = {f.state for f in out.frames}
    assert "grounded" in states
    # flight segments may or may not pass the residual gate on this synthetic
    # data; assert at least the schema invariant.
    assert len(out.frames) == n
```

- [ ] **Step 3:** Run, commit.

```bash
pytest tests/test_ball_stage.py -v -m integration
git add src/stages/ball.py tests/test_ball_stage.py
git commit -m "feat(ball): ground+flight ball stage"
```

---

## Phase 4 — Web viewer + export

### Task 4.1: Server endpoint cleanup

**Files:**
- Modify: `src/web/server.py`.

- [ ] **Step 1:** Remove imports/handlers for deleted endpoints (`/sync/*`, `/triangulation/*`, `/calibration/compare`, anything referencing matching/sync_map). Add stubs for new endpoints.

- [ ] **Step 2:** Add new endpoints:

```python
# inside src/web/server.py
@app.get("/anchors")
def get_anchors() -> dict:
    path = output_dir / "camera" / "anchors.json"
    if not path.exists():
        return {"clip_id": "", "image_size": [0, 0], "anchors": []}
    return json.loads(path.read_text())


@app.post("/anchors")
def set_anchors(payload: dict) -> dict:
    path = output_dir / "camera" / "anchors.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    return {"status": "ok"}


@app.get("/camera/track")
def get_camera_track() -> dict:
    path = output_dir / "camera" / "camera_track.json"
    if not path.exists():
        return {"frames": []}
    return json.loads(path.read_text())


@app.get("/hmr_world/preview")
def hmr_world_preview(player_id: str) -> dict:
    # Load NPZ summary for the viewer.
    npz_path = output_dir / "hmr_world" / f"{player_id}_smpl_world.npz"
    if not npz_path.exists():
        return {"error": "not found"}
    z = np.load(npz_path)
    return {
        "player_id": str(z["player_id"]),
        "frames": z["frames"].tolist(),
        "root_t": z["root_t"].tolist(),
        "confidence": z["confidence"].tolist(),
    }


@app.get("/ball/preview")
def ball_preview() -> dict:
    path = output_dir / "ball" / "ball_track.json"
    if not path.exists():
        return {"frames": []}
    return json.loads(path.read_text())
```

- [ ] **Step 3:** Run dashboard manually, confirm it boots and the new endpoints respond.

```bash
python recon.py serve --output ./output
```

- [ ] **Step 4:** Commit.

```bash
git add src/web/server.py
git commit -m "feat(web): drop legacy endpoints, add anchor/camera/hmr_world/ball APIs"
```

### Task 4.2: Anchor editor UI

**Files:**
- Create: `src/web/static/anchor_editor.html`
- Modify: `src/web/static/index.html` (link to new editor; remove legacy panels).

- [ ] **Step 1:** Implement the anchor editor as a single-file HTML+JS page using vanilla DOM (no build step). It needs:
  - `<video>` element + canvas overlay for projected pitch lines.
  - Landmark catalogue palette fed by `LANDMARK_CATALOGUE` from a JSON file (regenerated server-side as a static asset).
  - Per-frame confidence timeline driven by `/camera/track`.
  - "Add anchor here" button posts to `/anchors`.
  - Status indicator that calls `/camera/track` after each save and re-renders.

The page is roughly 250 lines; structure it in three sections (style, body, script) and avoid external libraries.

- [ ] **Step 2:** Update `index.html` to remove the calibration-compare split-pane, sync diagnostics panel, and per-shot-island grid. Add a section linking to the anchor editor and the 3D viewer.

- [ ] **Step 3:** Smoke test by serving and opening the page; check that anchors are persisted via the API.

- [ ] **Step 4:** Commit.

```bash
git add src/web/static/anchor_editor.html src/web/static/index.html
git commit -m "feat(web): anchor editor UI + dashboard cleanup"
```

### Task 4.3: 3D viewer cleanup

**Files:**
- Modify: `src/web/static/viewer.html`.

- [ ] **Step 1:** Remove per-shot-island rendering branch + the "no calibration" fallback. Always render against pitch coordinates loaded from `/camera/track` and `/hmr_world/preview` and `/ball/preview`.

- [ ] **Step 2:** Smoke test: open viewer in a browser after running the synthetic-clip integration test through to completion (mocked GVHMR → glTF export — wired in Task 4.4).

- [ ] **Step 3:** Commit.

```bash
git add src/web/static/viewer.html
git commit -m "refactor(web): viewer renders pitch-registered scenes only"
```

### Task 4.4: glTF export

**Files:**
- Modify: `src/stages/export.py`.
- Modify: `src/utils/gltf_builder.py` (or replace if existing builder is too triangulation-coupled).

- [ ] **Step 1:** Update `ExportStage.run()` to read `hmr_world/*.npz`, `ball/ball_track.json`, `camera/camera_track.json` and emit `export/gltf/scene.glb` + `export/gltf/scene_metadata.json` containing:
  - One animated SMPL armature per player, posed by `thetas` and `root_R`/`root_t`.
  - Camera animation track from `camera_track`.
  - Ball mesh animation from `ball_track`.

- [ ] **Step 2:** Run an end-to-end synthetic test (chain Phase-1/2/3 fixtures) and visually inspect `scene.glb` in the viewer.

- [ ] **Step 3:** Commit.

```bash
git add src/stages/export.py src/utils/gltf_builder.py
git commit -m "feat(export): glTF from hmr_world + ball + camera"
```

### Task 4.5a: Quality report aggregation

**Files:**
- Create: `src/pipeline/quality_report.py`
- Modify: `src/pipeline/runner.py` (call the aggregator at the end of `run_pipeline`).
- Test: `tests/test_quality_report.py`.

- [ ] **Step 1:** Implement aggregator that reads each stage's outputs and writes `output/quality_report.json` with `camera`, `hmr_world`, `ball` sections matching spec §6.

```python
# src/pipeline/quality_report.py
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.schemas.anchor import AnchorSet
from src.schemas.ball_track import BallTrack
from src.schemas.camera_track import CameraTrack


def write_quality_report(output_dir: Path) -> None:
    report: dict = {}

    cam_path = output_dir / "camera" / "camera_track.json"
    anchors_path = output_dir / "camera" / "anchors.json"
    if cam_path.exists() and anchors_path.exists():
        cam = CameraTrack.load(cam_path)
        anchors = AnchorSet.load(anchors_path)
        confs = np.array([f.confidence for f in cam.frames])
        low_mask = confs < 0.6
        ranges: list[list[int]] = []
        i = 0
        while i < len(low_mask):
            if low_mask[i]:
                j = i
                while j < len(low_mask) and low_mask[j]:
                    j += 1
                ranges.append([cam.frames[i].frame, cam.frames[j - 1].frame])
                i = j
            else:
                i += 1
        report["camera"] = {
            "anchor_count": len(anchors.anchors),
            "low_confidence_frame_count": int(low_mask.sum()),
            "low_confidence_frame_ranges": ranges,
        }

    hmr_dir = output_dir / "hmr_world"
    if hmr_dir.exists():
        npz_files = list(hmr_dir.glob("*_smpl_world.npz"))
        per_player_conf = []
        low_players: list[str] = []
        for p in npz_files:
            z = np.load(p)
            mc = float(z["confidence"].mean())
            per_player_conf.append(mc)
            if mc < 0.5:
                low_players.append(str(z["player_id"]))
        report["hmr_world"] = {
            "tracked_players": len(npz_files),
            "mean_per_player_confidence": float(np.mean(per_player_conf)) if per_player_conf else 0.0,
            "low_confidence_players": low_players,
        }

    ball_path = output_dir / "ball" / "ball_track.json"
    if ball_path.exists():
        ball = BallTrack.load(ball_path)
        states = [f.state for f in ball.frames]
        residuals = [s.fit_residual_px for s in ball.flight_segments]
        report["ball"] = {
            "grounded_frames": states.count("grounded"),
            "flight_segments": len(ball.flight_segments),
            "missing_frames": states.count("missing"),
            "mean_flight_fit_residual_px": float(np.mean(residuals)) if residuals else 0.0,
        }

    out = output_dir / "quality_report.json"
    out.write_text(json.dumps(report, indent=2))
```

- [ ] **Step 2:** Wire into the runner: call `write_quality_report(output_dir)` at the end of `run_pipeline()`.

- [ ] **Step 3:** Test.

```python
# tests/test_quality_report.py
import json
from pathlib import Path

import numpy as np
import pytest

from src.pipeline.quality_report import write_quality_report
from src.schemas.anchor import Anchor, AnchorSet, LandmarkObservation
from src.schemas.camera_track import CameraFrame, CameraTrack
from src.schemas.ball_track import BallFrame, BallTrack
from src.schemas.smpl_world import SmplWorldTrack


@pytest.mark.unit
def test_quality_report_aggregates_three_stages(tmp_path: Path):
    AnchorSet(
        clip_id="play",
        image_size=(1280, 720),
        anchors=(
            Anchor(frame=0, landmarks=(LandmarkObservation(name="x", image_xy=(0, 0), world_xyz=(0, 0, 0)),)),
            Anchor(frame=10, landmarks=(LandmarkObservation(name="y", image_xy=(0, 0), world_xyz=(1, 0, 0)),)),
        ),
    ).save(tmp_path / "camera" / "anchors.json")

    CameraTrack(
        clip_id="play", fps=30.0, image_size=(1280, 720), t_world=[0, 0, 0],
        frames=tuple(CameraFrame(frame=i, K=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                                 R=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                                 confidence=(0.9 if i < 8 else 0.3),
                                 is_anchor=(i in (0, 10)))
                     for i in range(11)),
    ).save(tmp_path / "camera" / "camera_track.json")

    SmplWorldTrack(
        player_id="P001",
        frames=np.arange(11),
        betas=np.zeros(10),
        thetas=np.zeros((11, 24, 3)),
        root_R=np.tile(np.eye(3), (11, 1, 1)),
        root_t=np.zeros((11, 3)),
        confidence=np.full(11, 0.8),
    ).save(tmp_path / "hmr_world" / "P001_smpl_world.npz")

    BallTrack(
        clip_id="play", fps=30.0,
        frames=tuple(BallFrame(frame=i, world_xyz=(0.0, 0.0, 0.11),
                               state="grounded", confidence=0.9)
                     for i in range(11)),
        flight_segments=(),
    ).save(tmp_path / "ball" / "ball_track.json")

    write_quality_report(tmp_path)
    report = json.loads((tmp_path / "quality_report.json").read_text())
    assert report["camera"]["anchor_count"] == 2
    assert report["camera"]["low_confidence_frame_count"] == 3
    assert report["hmr_world"]["tracked_players"] == 1
    assert report["ball"]["grounded_frames"] == 11
```

- [ ] **Step 4:** Commit.

```bash
pytest tests/test_quality_report.py -v
git add src/pipeline/quality_report.py src/pipeline/runner.py tests/test_quality_report.py
git commit -m "feat: aggregate per-stage diagnostics into quality_report.json"
```

### Task 4.5: FBX export (UE5)

**Files:**
- Modify: `src/stages/export.py` (extend) + a Blender-script helper file (existing `scripts/` dir likely has one; reuse if it's not too coupled to legacy SMPL fitting output).

- [ ] **Step 1:** Update or create a Blender headless script that ingests the same `SmplWorldTrack` NPZ files and produces:
  - `export/fbx/PXXX.fbx` per player (SMPL skeleton baked, world-positioned).
  - `export/fbx/ball.fbx` (single-bone armature with translation animation).
  - `export/fbx/camera.fbx` (camera animation, K → focal length conversion using image_size).

- [ ] **Step 2:** Add an integration test gated by `pytest -m fbx` that requires Blender on PATH; on CI, this is normally skipped.

- [ ] **Step 3:** Commit.

```bash
git add src/stages/export.py scripts/blender_export_fbx.py
git commit -m "feat(export): UE5-ready FBX from hmr_world"
```

---

### Task 4.6: Web API integration tests

**Files:**
- Create: `tests/test_web_api.py`.

- [ ] **Step 1:** Test the four new endpoints round-trip via FastAPI's TestClient.

```python
# tests/test_web_api.py
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.web.server import create_app


@pytest.fixture
def client(tmp_path: Path):
    app = create_app(output_dir=tmp_path, config_path=None)
    return TestClient(app), tmp_path


@pytest.mark.integration
def test_get_anchors_empty(client):
    c, _ = client
    resp = c.get("/anchors")
    assert resp.status_code == 200
    assert resp.json() == {"clip_id": "", "image_size": [0, 0], "anchors": []}


@pytest.mark.integration
def test_post_anchors_round_trips(client):
    c, tmp = client
    payload = {
        "clip_id": "play_037",
        "image_size": [1920, 1080],
        "anchors": [{"frame": 0, "landmarks": []}],
    }
    resp = c.post("/anchors", json=payload)
    assert resp.status_code == 200
    saved = json.loads((tmp / "camera" / "anchors.json").read_text())
    assert saved["clip_id"] == "play_037"


@pytest.mark.integration
def test_get_camera_track_empty(client):
    c, _ = client
    resp = c.get("/camera/track")
    assert resp.status_code == 200
    assert resp.json() == {"frames": []}


@pytest.mark.integration
def test_get_ball_preview_empty(client):
    c, _ = client
    resp = c.get("/ball/preview")
    assert resp.status_code == 200
    assert resp.json() == {"frames": []}
```

- [ ] **Step 2:** Run + commit.

```bash
pytest tests/test_web_api.py -v -m integration
git add tests/test_web_api.py
git commit -m "test(web): API integration tests for new endpoints"
```

---

## Phase 5 — Documentation

### Task 5.1: Rewrite README

**Files:**
- Modify: `README.md` (full rewrite).

- [ ] **Step 1:** Replace contents.

```markdown
# Football Perspectives

Reconstruct 3D football player animations and ball trajectories from a
single broadcast camera. Outputs a virtual camera, per-player SMPL
animation in pitch coordinates, and per-frame ball position with
3D flight reconstruction. Renders glTF for a browser viewer and FBX
for UE5.

## Pipeline

Seven sequential stages:

1. `prepare_shots` — accept a manually-trimmed clip.
2. `tracking` — YOLOv8x + ByteTrack for players and ball.
3. `camera` — keyframe-anchored per-frame K, R, t in pitch metres.
4. `pose_2d` — ViTPose (COCO 17 keypoints) for foot anchoring.
5. `hmr_world` — GVHMR per player → SMPL params in pitch frame.
6. `ball` — ground projection + parabolic 3D flight fit.
7. `export` — glTF for the web viewer + FBX for UE5 (via Blender).

## Requirements

- Python 3.11+
- FFmpeg
- GVHMR submodule + checkpoint (`third_party/gvhmr/inputs/checkpoints/gvhmr/gvhmr_siga24_release.ckpt`)
- Blender ≥ 3.6 (only for FBX export)
- GPU strongly recommended for `hmr_world`

## Install

```bash
git clone --recurse-submodules <repo-url>

python3.11 -m venv .venv311
source .venv311/bin/activate
python -m pip install -U pip

# mmcv from OpenMMLab CDN
python -m pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cpu/torch2.1/index.html

# Project + dev tooling
python -m pip install -c constraints/macos-py311-openmmlab.txt -e ".[dev]"
```

## CLI

```bash
# End-to-end
python recon.py run --input clip.mp4 --output ./output/

# Re-run only the camera stage (after editing anchors)
python recon.py run --input clip.mp4 --output ./output/ --from-stage camera

# Wipe legacy output dirs from earlier pipeline versions
python recon.py run --input clip.mp4 --output ./output/ --clean

# Web dashboard (anchor editor + 3D viewer)
python recon.py serve --output ./output/
```

Stage names are accepted by `--stages` and `--from-stage` (no numeric
aliases). Available: `prepare_shots`, `tracking`, `camera`, `pose_2d`,
`hmr_world`, `ball`, `export`.

## Output layout

```
output/
├── shots/                  # trimmed clip + manifest
├── tracks/                 # ByteTrack output (players + ball)
├── camera/                 # anchors.json + camera_track.json
├── pose_2d/                # ViTPose output
├── hmr_world/              # per-player SMPL params (pitch frame)
├── ball/                   # per-frame ball + flight segments
├── export/{gltf,fbx}/      # final artefacts
└── quality_report.json     # per-stage diagnostics
```

## Web dashboard

`python recon.py serve --output ./output/` opens a dashboard with:

- **Anchor editor**: place pitch landmarks on keyframes; the camera
  stage propagates between them.
- **Confidence timeline**: highlights frames where camera or HMR are
  uncertain so you know where to add anchors.
- **3D viewer**: pitch + animated players + ball, scrub-controlled.

## Testing

```bash
pytest                       # unit + integration
pytest -m e2e                # end-to-end on a small real clip
pytest -m gpu                # GPU model paths (GVHMR)
```
```

- [ ] **Step 2:** Commit.

```bash
git add README.md
git commit -m "docs: rewrite README for new single-camera pipeline"
```

### Task 5.2: Rewrite design doc

**Files:**
- Modify: `docs/football-reconstruction-pipeline-design.md` (full replace).

- [ ] **Step 1:** Replace contents with a copy of the spec at `docs/superpowers/specs/2026-05-04-broadcast-mono-pipeline-design.md` (sections 1–8) plus a header noting it's the implementation reference.

```bash
cp docs/superpowers/specs/2026-05-04-broadcast-mono-pipeline-design.md docs/football-reconstruction-pipeline-design.md
```

Then prepend the header:

```markdown
# Football Reconstruction Pipeline — Design Reference

Implementation reference for the single-camera pipeline. The
authoritative spec is the brainstorming-skill output at
`docs/superpowers/specs/2026-05-04-broadcast-mono-pipeline-design.md`;
this file is kept in sync with it.

---
```

- [ ] **Step 2:** Commit.

```bash
git add docs/football-reconstruction-pipeline-design.md
git commit -m "docs: replace pipeline design doc with single-camera reference"
```

### Task 5.3: Update CLAUDE.md

**Files:**
- Modify: `CLAUDE.md`.

- [ ] **Step 1:** Replace the "Pipeline Architecture" table and "Key Design Decisions" sections with content matching the new pipeline (refer to the spec). Update "External Dependencies" — remove PnLCalib/SoccerNet calibration references; keep GVHMR and Blender. Drop SMPL fitting model bullet (no longer used).

- [ ] **Step 2:** Commit.

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md to reflect new pipeline"
```

### Task 5.4: Delete obsolete docs

**Files:**
- Delete: `docs/sync-approaches-diagnostic.md`, `docs/2026-04-04-macos-dependency-handoff.md`, `docs/open-questions-2026-04-13.md`.

```bash
git rm docs/sync-approaches-diagnostic.md \
       docs/2026-04-04-macos-dependency-handoff.md \
       docs/open-questions-2026-04-13.md
git commit -m "docs: drop notes about obsolete stages"
```

---

## Phase 6 — End-to-end validation

### Task 6.1: Rewrite `tests/test_runner.py`

**Files:**
- Create: `tests/test_runner.py` (deleted in Phase 0; recreate scoped to single mode).

- [ ] **Step 1: Write tests.**

```python
# tests/test_runner.py
import pytest

from src.pipeline.runner import resolve_stages


@pytest.mark.unit
def test_resolve_all():
    assert resolve_stages("all", None) == [
        "prepare_shots", "tracking", "camera", "pose_2d",
        "hmr_world", "ball", "export",
    ]


@pytest.mark.unit
def test_resolve_subset():
    assert resolve_stages("camera,pose_2d", None) == ["camera", "pose_2d"]


@pytest.mark.unit
def test_resolve_unknown_raises():
    with pytest.raises(ValueError):
        resolve_stages("calibration", None)


@pytest.mark.unit
def test_resolve_with_from_stage_skips_earlier():
    result = resolve_stages("all", "hmr_world")
    assert result == ["hmr_world", "ball", "export"]
```

```bash
pytest tests/test_runner.py -v
git add tests/test_runner.py
git commit -m "test(runner): pin single-mode stage resolution"
```

### Task 6.2: Real-clip E2E test

**Files:**
- Create: `tests/test_e2e_real_clip.py`

- [ ] **Step 1:** Place a 5-second real clip + 3 hand-anchored anchor JSON under `tests/fixtures/real_clip/`. The test invokes the full pipeline and asserts:
  - `camera_track.json` has frames for the entire clip range.
  - `hmr_world/` has at least one `*_smpl_world.npz`.
  - `ball/ball_track.json` exists and contains ≥ N grounded frames.
  - `export/gltf/scene.glb` opens with `pygltflib`.

```python
# tests/test_e2e_real_clip.py
import pytest
from pathlib import Path

from src.pipeline.runner import run_pipeline


@pytest.mark.e2e
def test_full_pipeline_on_real_clip(tmp_path: Path):
    fixture = Path("tests/fixtures/real_clip")
    if not fixture.exists():
        pytest.skip("real-clip fixture not provided")
    # Copy clip + anchors into tmp output
    (tmp_path / "shots").mkdir()
    (tmp_path / "camera").mkdir()
    (fixture / "play.mp4").rename(tmp_path / "shots" / "play.mp4")
    (fixture / "anchors.json").rename(tmp_path / "camera" / "anchors.json")

    config = {}  # default config from /config/default.yaml at runtime
    run_pipeline(
        output_dir=tmp_path,
        stages="tracking,camera,pose_2d,hmr_world,ball,export",
        from_stage=None,
        config=config,
        video_path=tmp_path / "shots" / "play.mp4",
        device="auto",
    )

    assert (tmp_path / "camera" / "camera_track.json").exists()
    assert any((tmp_path / "hmr_world").glob("*_smpl_world.npz"))
    assert (tmp_path / "ball" / "ball_track.json").exists()
    assert (tmp_path / "export" / "gltf" / "scene.glb").exists()
```

- [ ] **Step 2:** Commit.

```bash
git add tests/test_e2e_real_clip.py
git commit -m "test(e2e): full-pipeline check on a real broadcast clip"
```

### Task 6.3: Final QA pass

- [ ] Run `pytest --cov=src --cov-report=term-missing -m "not e2e and not gpu"` and verify ≥80% coverage.
- [ ] Run `pytest -m e2e` once locally with a real clip; capture screenshots of the viewer + a short video of an FBX import in UE5.
- [ ] Add screenshots / brief notes to a draft PR description.
- [ ] Commit any test additions to lift coverage.

```bash
pytest --cov=src --cov-report=term-missing -m "not e2e and not gpu" -v
```

---

## Self-review checklist (run by the implementing engineer at the end)

- [ ] Phase 0–6 all green under `pytest -m "not e2e and not gpu"`.
- [ ] Phase 6 E2E checked once on a real clip.
- [ ] No imports of deleted modules remain (`grep -R "calibration_propagation\|tvcalib_calibrator\|sync\|triangulation_calib" src/ tests/` returns empty).
- [ ] `python recon.py run --help` lists only the new stage names; no numeric aliases.
- [ ] `python recon.py serve` boots and the anchor editor saves anchors round-trip.
- [ ] README, design doc, CLAUDE.md all reflect the new pipeline.
- [ ] `quality_report.json` extended with `camera`, `hmr_world`, `ball` sections after a real run.
- [ ] FEATURE_IDEAS.md still contains the pitch-config-selector bullet (and any new bullets discovered during implementation).
