# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A Python CLI tool (`recon.py`) that reconstructs 3D player animations from broadcast football footage. It takes single or multi-angle video, runs a 9-stage ML pipeline, and exports FBX (for Unreal Engine 5) and glTF (for a browser viewer).

The full technical design is in `docs/football-reconstruction-pipeline-design.md`.

## Commands

```bash
# Run the full pipeline
python recon.py --input match_replay.mp4 --output ./output/ --stages all

# Resume from a specific stage (uses cached outputs from earlier stages)
python recon.py --input match_replay.mp4 --output ./output/ --from-stage triangulation

# Run only specific stages
python recon.py --input match_replay.mp4 --output ./output/ --stages 1,2,3

# Launch browser viewer on existing output
python recon.py --output ./output/ --viewer

# Custom config
python recon.py --input match_replay.mp4 --output ./output/ --config config/smooth_heavy.yaml
```

## Pipeline Architecture

The pipeline has 9 sequential stages. Each stage reads from previous stage outputs in `output/` and writes its own subdirectory. Stages are independently re-runnable.

| Stage | Name | Input | Output |
|-------|------|-------|--------|
| 1 | Shot Segmentation | video file | `shots/shots_manifest.json` + per-shot `.mp4` clips |
| 2 | Camera Calibration | shots | `calibration/shot_XXX_calibration.json` |
| 3 | Temporal Synchronisation | shots + calibration | `sync/sync_map.json` |
| 4 | Player Detection & Tracking | shots | `tracks/shot_XXX_tracks.json` |
| 5 | 2D Pose Estimation | shots + tracks | `poses/shot_XXX_poses.json` |
| 6 | Cross-View Player Matching | tracks + sync | `matching/player_matches.json` |
| 7 | 3D Triangulation | poses + calibration + matches | `triangulated/PXXX_3d_joints.npz` |
| 8 | SMPL Fitting | triangulated joints | `smpl/PXXX_smpl.npz` |
| 9 | Export | SMPL params | `export/fbx/`, `export/gltf/` |

## Key Design Decisions

**Pitch coordinate system**: The football pitch is the ground plane (z=0), 105m × 68m (FIFA standard). All 3D positions are in pitch-metres. Camera calibration maps pixel space → pitch space.

**Keypoint format**: COCO 17 keypoints throughout (nose through right_ankle). Confidence threshold 0.3 is the cutoff below which a keypoint is treated as occluded.

**Frame offset convention** (sync): `frame_in_reference = frame_in_shot + offset`. A `frame_offset` of -47 means shot_003's frame 0 = shot_001's frame 47.

**Triangulation**: Weighted DLT — each 2D observation is weighted by ViTPose confidence. Requires ≥2 views; falls back to monocular for single-view frames.

**SMPL fitting**: Shape parameters (β) are shared across all frames for a given player. Pose parameters (θ) are per-frame, initialised from the previous frame. Uses VPoser as the pose prior.

**FBX export bridge**: Goes through Blender headless (`bpy`) — SMPL params → Blender armature → FBX. UE5 convention: scale 1.0m, forward -Y, up Z.

## ML Models Used

- **Shot segmentation**: PySceneDetect `ContentDetector(threshold=30.0)`
- **Camera calibration**: SoccerNet Camera Calibration model (pitch line segmentation + OpenCV `solvePnP`)
- **Ball detection**: YOLOv8 specialised ball model (`yolov8_ball_custom`)
- **Player detection**: YOLOv8x fine-tuned on football data (classes: player, goalkeeper, referee, ball)
- **Tracking**: ByteTrack (via `supervision`)
- **Team classification**: SigLIP/CLIP embeddings + K-means (k=3)
- **Pose estimation**: ViTPose-Large (via MMPose or HuggingFace)
- **SMPL body model**: `smplx` library (neutral model, 10 shape params, 72 pose params, 24 joints)

## Configuration

`config/default.yaml` controls all tunable parameters per stage. Key values to know:

- `calibration.max_reprojection_error: 15.0` — shots above this are flagged unreliable
- `pose_estimation.min_confidence: 0.3` — below this, keypoints are excluded from triangulation
- `triangulation.min_views: 2` — minimum cameras for DLT; single-view falls back to monocular
- `smpl_fitting.lambda_ground: 10.0` — high weight to prevent foot-ground penetration

## External Dependencies

Beyond Python packages, the pipeline requires:
- **Blender ≥ 3.6** (headless, for FBX export): `snap install blender --classic`
- **FFmpeg** (shot extraction)
- **SMPL model files**: download from smpl.is.tue.mpg.de (requires free registration)

GPU: minimum 8GB VRAM; recommended 12GB+ for ViTPose-Large + YOLOv8x concurrent.

## Browser Viewer

`output/viewer/index.html` — single standalone HTML file (no build step), loads `export/gltf/scene.glb` + `scene_metadata.json`. Features: playback controls, orbit camera, player selection, joint confidence heatmap, preset camera angles.

## Quality Report

`output/quality_report.json` is generated at the end — per-stage confidence metrics, failed shots, low-confidence joints. Check this first when debugging reconstruction quality.
