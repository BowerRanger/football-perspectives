# Football Perspectives

A Python CLI for reconstructing 3D football player motion from broadcast footage.

## What this repo currently supports

The full 8-stage pipeline is implemented:

- Stage 1: `segmentation` — shot boundary detection
- Stage 2: `calibration` — neural camera parameter estimation via PnLCalib
- Stage 3: `tracking` — player detection, tracking, team labels, and cross-view identity matching
- Stage 4: `pose` — 2D pose estimation (ViTPose via MMPose)
- Stage 5: `sync` — temporal alignment across camera views
- Stage 6: `triangulation` — multi-view 3D joint triangulation
- Stage 7: `smpl_fitting` — SMPL body model fitting to 3D joints
- Stage 8: `export` — glTF export for the 3D viewer

## Requirements

- Python 3.11+
- FFmpeg (recommended for video workflows)

## Install

Clone with submodules (PnLCalib is pulled as a submodule under `third_party/PnLCalib`):

```bash
# Fresh clone
git clone --recurse-submodules <repo-url>

# Or, if you already cloned without --recurse-submodules
git submodule update --init --recursive
```

Then the Python environment:

```bash
python3.11 -m venv .venv311
source .venv311/bin/activate
python -m pip install -U pip

# mmcv does not publish standard PyPI wheels — install from OpenMMLab CDN first
python -m pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cpu/torch2.1/index.html

# Install everything else under the pinned constraints
python -m pip install -c constraints/macos-py311-openmmlab.txt -e .
# optional dev tooling
python -m pip install -c constraints/macos-py311-openmmlab.txt -e ".[dev]"

# PnLCalib runtime dependencies (not on PyPI as a package)
python -m pip install shapely lsq-ellipse
```

PnLCalib model weights (SV_kp, SV_lines, ~253 MB each) download automatically on first use of the calibration stage. They are cached at `data/models/pnlcalib/`. To pre-download manually, see the calibration stage documentation.

Notes:

- **Python 3.11 is required** for Stage 4 pose estimation. The OpenMMLab stack (`mmcv`, `xtcocotools`) does not publish pre-built wheels for Python 3.12+ on macOS arm64. Source builds fail without a full C++ / Fortran toolchain.
- `mmcv` must be installed from the OpenMMLab CDN (see above). Running `pip install mmcv` without the `-f` flag causes pip to fall back to a source build, which fails on a standard macOS setup.
- Install with the constraints file to avoid resolver drift across `numpy`, `opencv-python`, `mmcv`, `mmpose`, and `mmdet`.
- Avoid unconstrained upgrades of OpenMMLab packages in-place (for example `pip install -U mmcv`) because that can break compatibility.
- The default pose config uses an MMPose model alias plus an optional explicit checkpoint path under `pose_estimation` in [config/default.yaml](config/default.yaml).

Quick health checks after install:

```bash
python -m pip check
python -c "import numpy, cv2, torch, mmcv, mmdet, mmpose; print('imports ok')"
python -c "import xtcocotools._mask; print('xtcocotools ok')"
```

## CLI usage

Show top-level help:

```bash
python recon.py --help
```

Show help for the run command:

```bash
python recon.py run --help
```

Run all currently implemented stages:

```bash
python recon.py run \
  --input test-media/match_replay.mp4 \
  --output ./output \
  --stages all
```

Run only selected stages by number:

```bash
python recon.py run \
  --input test-media/match_replay.mp4 \
  --output ./output \
  --stages 1,2,3,4,5,6
```

Run only selected stages by name:

```bash
python recon.py run \
  --input test-media/match_replay.mp4 \
  --output ./output \
  --stages segmentation,calibration,tracking,pose,sync,matching
```

Resume from a stage (forces that stage to re-run and skips earlier stages):

```bash
python recon.py run \
  --input test-media/match_replay.mp4 \
  --output ./output \
  --stages all \
  --from-stage calibration
```

Use a custom config merged onto defaults:

```bash
python recon.py run \
  --input test-media/match_replay.mp4 \
  --output ./output \
  --config config/default.yaml
```

Select compute device for stage-level inference. Stage 4 pose estimation consumes this flag directly:

```bash
python recon.py run \
  --input test-media/match_replay.mp4 \
  --output ./output \
  --device auto
```

Device resolution:

- `auto` prefers `cuda:0`, then `mps`, then `cpu`
- an explicit CLI value overrides `pose_estimation.device` in [config/default.yaml](config/default.yaml)
- the config value is used when the CLI stays on `auto`

## Prepared shots workflow (skip segmentation)

If you already have pre-cut clips (for example `origi01.mp4`, `origi02.mp4`) and want to test calibration/sync directly, you can run from Stage 2 onward.

1. Put prepared clips in `output/shots/`:

```text
output/
  shots/
    origi01.mp4
    origi02.mp4
    origi03.mp4
```

2. Run calibration + sync (or more downstream stages):

```bash
python recon.py run \
  --input test-media/origi-vs-barcelona.mp4 \
  --output ./output \
  --stages calibration,tracking,pose,sync
```

3. Continue with downstream stages if needed:

```bash
python recon.py run \
  --input test-media/origi-vs-barcelona.mp4 \
  --output ./output \
  --stages tracking,pose,matching
```

Notes:

- Stages 2 and 3 now infer `shots/shots_manifest.json` automatically when it is missing, based on clips present in `output/shots/`.
- Stages 2 and 5 infer `shots/shots_manifest.json` automatically when it is missing, based on clips present in `output/shots/`.
- The inferred `shot.id` is the clip filename stem (for example `origi01.mp4` -> `origi01`).
- Prepared clips must share a common FPS. Mixed-FPS clip sets are rejected to avoid bad timing alignment.
- The CLI currently requires `--input` even when Stage 1 is not selected; for Stage 2+ runs it is not used by calibration/sync logic.
- Calibration runs PnLCalib on each keyframe and fuses results into a single static camera position per shot. No manual landmark annotation is required.

## Stage aliases

These numeric aliases are supported in `--stages` and `--from-stage`:

- `1` -> `segmentation`
- `2` -> `calibration`
- `3` -> `tracking`
- `4` -> `pose`
- `5` -> `sync`
- `6` -> `triangulation`
- `7` -> `smpl_fitting`
- `8` -> `export`

Migration note:

- Numeric aliases were remapped to match execution order. If you previously used `--stages 1,2,3` for `segmentation,calibration,sync`, switch to named stages or use `--stages 1,2,5`.

## Stage summaries (imports/exports)

- Stage 1 `segmentation`: Detects camera-cut shot boundaries, excludes fade-style transition spans, and keeps only shots where the ball appears.
  Imports: input video (`--input`).
  Exports: `shots/shots_manifest.json`, `shots/shot_XXX.mp4`.
- Stage 2 `calibration`: Runs PnLCalib on sampled keyframes to recover per-frame camera parameters, then fuses them into a single static camera position per shot (median world position, per-frame rotation + focal length refit).
  Imports: `shots/shots_manifest.json`, shot clips (`shots/shot_XXX.mp4`).
  Exports: `calibration/shot_XXX_calibration.json`.
- Stage 3 `tracking`: Detects and tracks players across frames with team labels, pitch projection, and cross-view player identity matching. Auto-assigns global player IDs (P001, P002...) when sync data is available. Player names can be edited in the web dashboard.
  Imports: `shots/shots_manifest.json`, shot clips, optional `calibration/`, optional `sync/sync_map.json`.
  Exports: `tracks/shot_XXX_tracks.json`, `matching/player_matches.json`.
- Stage 4 `pose`: Runs MMPose top-down inference on each tracked player crop, remaps keypoints into frame coordinates, and applies temporal smoothing. Carries forward player_id and player_name from tracking.
  Imports: `shots/shots_manifest.json`, shot clips, `tracks/shot_XXX_tracks.json`.
  Exports: `poses/shot_XXX_poses.json`.
- Stage 5 `sync`: Aligns shots in time using hybrid evidence from ball trajectories and player-motion signals.
  Imports: `shots/shots_manifest.json`, shot clips, optional calibration files, optional `tracks/shot_XXX_tracks.json`.
  Exports: `sync/sync_map.json`.
- Stage 6 `triangulation`: Triangulates 2D keypoints from multiple views into 3D joint positions using weighted DLT with RANSAC outlier rejection. Post-processes with Savitzky-Golay temporal smoothing, bone length enforcement, and foot-ground contact snapping.
  Imports: `matching/player_matches.json`, `sync/sync_map.json`, `calibration/shot_XXX_calibration.json`, `poses/shot_XXX_poses.json`.
  Exports: `triangulated/PXXX_3d_joints.npz`.
- Stage 7 `smpl_fitting`: Fits SMPL body model parameters (pose, shape, translation) to triangulated 3D joints. Uses SMPLify-style optimization when the `smplx` package and SMPL model file are available; falls back to a lightweight approximation otherwise.
  Imports: `triangulated/PXXX_3d_joints.npz`.
  Exports: `smpl/PXXX_smpl.npz`.
- Stage 8 `export`: Exports SMPL animations as glTF 2.0 (.glb) for the 3D viewer. FBX export (for UE5) is optional and requires Blender.
  Imports: `smpl/PXXX_smpl.npz`, `matching/player_matches.json`.
  Exports: `export/gltf/scene.glb`, `export/gltf/scene_metadata.json`.

## Output structure

After running, outputs are written under your `--output` directory, for example:

- `shots/shots_manifest.json`
- `calibration/shot_XXX_calibration.json`
- `sync/sync_map.json`
- `tracks/shot_XXX_tracks.json`
- `poses/shot_XXX_poses.json`
- `matching/player_matches.json`
- `triangulated/PXXX_3d_joints.npz`
- `smpl/PXXX_smpl.npz`
- `export/gltf/scene.glb`
- `export/gltf/scene_metadata.json`
- `export/export_result.json`

## Web dashboard and 3D viewer

Launch the pipeline dashboard:

```bash
python recon.py serve --output ./output
```

The dashboard (http://localhost:8000) shows:

- Stage completion status in the sidebar
- Per-stage output inspection (calibration errors, tracking overlays, pose skeletons, sync offsets, matching pitch map, triangulated 3D skeletons)
- Pipeline job execution with real-time log streaming

The standalone 3D viewer is at http://localhost:8000/viewer after running stages 7-9. Features:

- Three.js rendering of player skeletons on a pitch model
- Playback controls: play/pause, timeline scrub, speed (0.25x–2x)
- Camera presets: broadcast, tactical (top-down), behind-goal
- Orbit camera controls
- Skeleton wireframe toggle

## SMPL model setup (optional, for Stage 8)

Stage 8 uses the SMPL body model. Without it, the stage falls back to a lightweight approximation that derives translation from hip positions.

For full SMPL fitting:

1. Register at https://smpl.is.tue.mpg.de/ and download `SMPL_NEUTRAL.pkl`
2. Place at `data/smpl/SMPL_NEUTRAL.pkl` (or set `smpl_fitting.model_path` in config)
3. Install `smplx`: `pip install smplx`

## Testing

Run test suite:

```bash
pytest
```
