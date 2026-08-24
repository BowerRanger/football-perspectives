# Football Perspectives

Reconstruct 3D football player animations and ball trajectories from a
single broadcast camera. Outputs a virtual camera, per-player SMPL
animation in pitch coordinates, and per-frame ball position with
3D flight reconstruction. Renders glTF for a browser viewer and FBX
for UE5.

## Pipeline

Seven sequential stages:

1. `prepare_shots` — copy a trimmed clip, or auto-split/classify/group/align
   a full highlights reel.
2. `tracking` — YOLOv8x + ByteTrack for players; WASB HRNet for the ball.
3. `camera` — keyframe-anchored per-frame K, R, t in pitch metres.
4. `hmr_world` — GVHMR per player → SMPL params in pitch frame
   (runs ViTPose internally; 2D keypoints written as a side-output).
5. `refined_poses` — per-player translation cleanup (gap-fill, outlier
   rejection, velocity limiting) across shots.
6. `ball` — event-based auto-anchors (touches, bounces, goal impacts) +
   physically-correct piecewise trajectory solve.
7. `export` — glTF for the web viewer + FBX for UE5 (via Blender).

## Requirements

- Python 3.11+
- FFmpeg
- GVHMR submodule + checkpoint (`third_party/gvhmr/inputs/checkpoints/gvhmr/gvhmr_siga24_release.ckpt`)
- WASB ball-detector checkpoint (`third_party/wasb_sbdt/pretrained_weights/wasb_soccer_finetuned_v1.pth.tar`;
  weights are gitignored — regenerate with `scripts/build_finetune_corpus.py` +
  `scripts/finetune_wasb.py`, or set `ball.wasb.checkpoint` to the stock
  `wasb_soccer_best.pth.tar`)
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
aliases). Available: `prepare_shots`, `tracking`, `camera`,
`hmr_world`, `refined_poses`, `ball`, `export`.

## Output layout

```
output/
├── shots/                  # per-shot clips + manifest (+ sync_map.json for grouped highlights)
├── tracks/                 # ByteTrack output (players + ball)
├── camera/                 # anchors.json + camera_track.json
├── hmr_world/              # per-player SMPL params (pitch frame) + kp2d side-outputs
├── refined_poses/          # cleaned per-player translations + summary
├── ball/                   # per-shot ball track, observations, auto-anchors, keyframes, diag
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
pytest                       # unit + integration (e2e/fbx skipped by default)
pytest -m e2e                # end-to-end on a small real clip (needs fixtures/GPU)
pytest -m fbx                # FBX serialisation (needs blender on PATH)
```
