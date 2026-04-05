# Football Perspectives

A Python CLI for reconstructing 3D football player motion from broadcast footage.

## What this repo currently supports

The pipeline architecture is designed for 9 stages, but this codebase currently implements and wires up:

- Stage 1: `segmentation`
- Stage 2: `calibration`
- Stage 3: `sync`
- Stage 4: `tracking`
- Stage 5: `pose`
- Stage 6: `matching`

## Requirements

- Python 3.11+
- FFmpeg (recommended for video workflows)

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
# optional dev tooling
pip install -e ".[dev]"
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
  --stages segmentation,calibration,sync,tracking,pose,matching
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

Select compute device (flag is available in CLI and reserved for stage-level usage):

```bash
python recon.py run \
  --input test-media/match_replay.mp4 \
  --output ./output \
  --device auto
```

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
  --stages calibration,sync
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
- The inferred `shot.id` is the clip filename stem (for example `origi01.mp4` -> `origi01`).
- Prepared clips must share a common FPS. Mixed-FPS clip sets are rejected to avoid bad timing alignment.
- The CLI currently requires `--input` even when Stage 1 is not selected; for Stage 2+ runs it is not used by calibration/sync logic.

## Stage aliases

These numeric aliases are supported in `--stages` and `--from-stage`:

- `1` -> `segmentation`
- `2` -> `calibration`
- `3` -> `sync`
- `4` -> `tracking`
- `5` -> `pose`
- `6` -> `matching`

## Stage summaries (imports/exports)

- Stage 1 `segmentation`: Detects camera-cut shot boundaries, excludes fade-style transition spans, and keeps only shots where the ball appears.
  Imports: input video (`--input`).
  Exports: `shots/shots_manifest.json`, `shots/shot_XXX.mp4`.
- Stage 2 `calibration`: Estimates camera parameters for each shot from pitch landmarks.
  Imports: `shots/shots_manifest.json`, shot clips (`shots/shot_XXX.mp4`).
  Exports: `calibration/shot_XXX_calibration.json`.
- Stage 3 `sync`: Aligns shots in time using ball-trajectory cross-correlation.
  Imports: `shots/shots_manifest.json`, shot clips, optional calibration files.
  Exports: `sync/sync_map.json`.
- Stage 4 `tracking`: Detects and tracks players across frames, with team labels and optional pitch projection.
  Imports: `shots/shots_manifest.json`, shot clips, optional `calibration/shot_XXX_calibration.json`.
  Exports: `tracks/shot_XXX_tracks.json`.
- Stage 5 `pose`: Estimates 2D keypoints per tracked player and applies temporal smoothing.
  Imports: `shots/shots_manifest.json`, shot clips, `tracks/shot_XXX_tracks.json`.
  Exports: `poses/shot_XXX_poses.json`.
- Stage 6 `matching`: Matches player tracks across synchronized camera views and assigns global player IDs.
  Imports: `shots/shots_manifest.json`, `sync/sync_map.json`, `tracks/shot_XXX_tracks.json`.
  Exports: `matching/player_matches.json`.

## Output structure

After running, outputs are written under your `--output` directory, for example:

- `shots/shots_manifest.json`
- `calibration/shot_XXX_calibration.json`
- `sync/sync_map.json`
- `tracks/shot_XXX_tracks.json`
- `poses/shot_XXX_poses.json`
- `matching/player_matches.json`

## Testing

Run test suite:

```bash
pytest
```
