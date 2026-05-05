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
