# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A Python CLI tool (`recon.py`) that reconstructs 3D player animations and ball trajectories from a single broadcast football camera. It takes a manually-trimmed clip, runs a 7-stage ML pipeline, and exports glTF (for a browser viewer) and FBX (for Unreal Engine 5).

The full technical design is in `docs/football-reconstruction-pipeline-design.md`.

## Commands

```bash
# Run the full pipeline
python recon.py run --input clip.mp4 --output ./output/

# Re-run only the camera stage (after editing anchors)
python recon.py run --input clip.mp4 --output ./output/ --from-stage camera

# Run a subset of stages by name
python recon.py run --input clip.mp4 --output ./output/ --stages tracking,camera,pose_2d

# Wipe legacy output dirs from earlier pipeline versions
python recon.py run --input clip.mp4 --output ./output/ --clean

# Web dashboard (anchor editor + 3D viewer)
python recon.py serve --output ./output/
```

## Pipeline Architecture

The pipeline has 7 sequential stages. Each stage reads from previous stage outputs in `output/` and writes its own subdirectory. Stages are independently re-runnable.

| # | Stage | Input | Output |
|---|-------|-------|--------|
| 1 | `prepare_shots` | trimmed clip | `shots/clip.mp4` + manifest |
| 2 | `tracking` | shots | `tracks/PXXX_track.json` + `tracks/ball_track.json` |
| 3 | `camera` | shots + anchors | `camera/camera_track.json` + debug |
| 4 | `pose_2d` | shots + tracks | `pose_2d/PXXX_pose.json` |
| 5 | `hmr_world` | tracks + pose_2d + camera | `hmr_world/PXXX_smpl_world.npz` |
| 6 | `ball` | tracks + camera | `ball/ball_track.json` |
| 7 | `export` | hmr_world + ball + camera | `export/gltf/scene.glb` + `export/fbx/` |

## Key Design Decisions

**Pitch coordinate system**: The football pitch is the ground plane (z=0), FIFA standard 105m × 68m. The x axis runs along the nearside touchline; y points across the pitch toward the far touchline; z is up. All 3D positions are in pitch-metres.

**Single camera per clip**: One broadcast camera, manually trimmed to a single uninterrupted shot. The camera body is assumed fixed (broadcast pan-tilt-zoom rig), so translation `t` is solved once and held constant; only `R` and focal length vary per frame.

**Camera tracking**: Keyframe-anchored. The user marks pitch landmarks on a sparse set of keyframes via the web anchor editor; the camera stage solves anchor frames first, then propagates between them with bidirectional optical-flow feature tracking and a smoother. Per-frame confidence is reported so uncertain spans surface as candidates for additional anchors.

**Player skeletal animation**: GVHMR (SIGGRAPH Asia 2024) runs per track on the cropped player. Output SMPL parameters are transformed into the pitch frame using the per-frame camera and foot-anchored against the ground plane. Foot anchoring uses ViTPose ankle keypoints when visible and falls back through bounded occlusion windows.

**Ball**: Monocular ground projection while the ball is rolling; parabolic 3D fit on flight segments detected from pixel velocity spikes in the ball track.

**Export**: glTF for the web viewer (capsule-mesh players in v1, swappable later) and FBX via Blender headless for UE5 retargeting. UE5 convention: scale 1.0m, forward -Y, up Z.

**Keypoint format**: COCO 17 keypoints (nose through right_ankle) for 2D pose. Confidence threshold 0.3 is the cutoff below which a keypoint is treated as occluded.

## ML Models Used

- **Player detection + tracking**: YOLOv8x + ByteTrack (via `supervision`)
- **Pose 2D**: ViTPose (COCO 17 keypoints, via MMPose)
- **HMR**: GVHMR (SIGGRAPH Asia 2024) — vendored under `third_party/gvhmr`
- **Ball detection**: YOLOv8 ball variant (`yolov8n.pt`)

## Configuration

`config/default.yaml` controls all tunable parameters per stage. Key values to know:

- `camera.anchor_max_reprojection_px: 4.0` — anchor frames whose solver reprojection exceeds this are flagged
- `pose_2d.min_confidence: 0.3` — below this, keypoints are excluded from foot anchoring
- `hmr_world.foot_anchor_max_occlusion_frames: 10` — maximum gap during which the last anchored foot position is held
- `ball.flight_px_velocity: 25.0` — pixel-velocity threshold above which the ball is treated as airborne

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
