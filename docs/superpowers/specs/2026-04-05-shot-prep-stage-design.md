# Shot Preparation Stage Design

**Date:** 2026-04-05
**Status:** Approved

## Overview

Insert a new `prepare_shots` stage (Stage 2) between segmentation and calibration. It handles two post-processing responsibilities that currently have no clean home:

1. **Speed normalisation** — detect slow-motion or speed-up in clips relative to the reference clip (first shot) and retime them to real-time speed.
2. **Duplicate frame removal** — remove broadcast freeze frames (already implemented in `deduplicate_clip`; moved here from the segmentation stage).

Pre-segmented clips (dropped by the user into `output/shots/`) enter the pipeline at this stage via `--from-stage prepare-shots`, bypassing PySceneDetect entirely.

## Stage Structure

| # | Name | Notes |
|---|------|-------|
| 1 | `segmentation` | Unchanged |
| 2 | `prepare_shots` | **New** |
| 3 | `calibration` | Renumbered (was 2) |
| 4 | `tracking` | Renumbered (was 3) |
| 5 | `pose` | Renumbered (was 4) |
| 6 | `sync` | Renumbered (was 5) |
| 7 | `matching` | Renumbered (was 6) |

`shot_segmentation.remove_duplicate_frames` is deprecated; a warning is logged if present. The canonical location is `prepare_shots.remove_duplicate_frames`.

## Pre-Segmented Clips Entry Point

`ShotPrepStage.run()` calls `ShotsManifest.load_or_infer()` to obtain the manifest. If no `shots_manifest.json` exists, `infer_from_clips` discovers all `.mp4`/`.mov`/`.avi`/`.mkv` files in `output/shots/` and builds a manifest automatically. This means a user can drop pre-cut clips into `output/shots/` and run:

```bash
python recon.py run --input dummy.mp4 --output ./output --from-stage prepare-shots
```

The `--input` flag on `recon.py run` is currently `required=True`. When starting from Stage 2 with pre-segmented clips, there is no source video. The CLI must be updated to make `--input` optional when `--from-stage` resolves to a stage after segmentation (i.e., segmentation is not in the active stage set). If `--input` is omitted and segmentation would run, the CLI raises a clear error.

## Speed Detection

**Function:** `_estimate_speed_factor(ref_clip, shot_clip, n_samples, min_flow_magnitude) → float`

### Algorithm

1. Sample `n_samples` evenly-spaced consecutive frame pairs from each clip independently.
2. For each pair: detect Shi-Tomasi corners on frame A; track to frame B with Lucas-Kanade optical flow; compute mean Euclidean magnitude of successfully tracked point displacements.
3. Average magnitudes across all sampled pairs → `ref_avg`, `shot_avg`.
4. If `shot_avg < min_flow_magnitude` (static scene guard — players standing still, crowd reaction, black frames): return `1.0` to avoid division by near-zero.
5. Return `speed_factor = ref_avg / shot_avg`.

### Interpretation

- `speed_factor > 1.0`: shot has less inter-frame motion than reference → it is slower (slow-motion replay). Example: origi03 at ~1.76× slow-motion would yield `speed_factor ≈ 1.76`.
- `speed_factor < 1.0`: shot has more inter-frame motion than reference → it is faster.
- `speed_factor ≈ 1.0`: shot is real-time, no retiming needed.

### Normalisation

If `abs(speed_factor - 1.0) > speed_factor_threshold` (default 0.15):

- Re-encode clip in-place: `ffmpeg -vf setpts=PTS/{speed_factor} -r {fps} -c:v libx264 -crf 18 -preset fast -an`
- Read the new clip's frame count from cv2 post-encode.
- Update `Shot.end_frame = start_frame + new_frames - 1`
- Update `Shot.end_time = start_time + new_frames / fps`
- Set `Shot.speed_factor = speed_factor`

The reference clip (first shot) is never retimed; its `speed_factor` remains `1.0`.

## Schema Changes

### `Shot` dataclass (`src/schemas/shots.py`)

```python
@dataclass
class Shot:
    id: str
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    clip_file: str
    speed_factor: float = 1.0   # detected and applied speed factor; 1.0 = real-time
```

`ShotsManifest.load()` already filters to known fields, so existing manifests without `speed_factor` load cleanly via the dataclass default.

## Configuration

```yaml
prepare_shots:
  speed_detection_samples: 10        # frame pairs sampled per clip for optical flow
  speed_factor_threshold: 0.15       # normalise only if factor deviates >15% from 1.0
  min_flow_magnitude: 0.5            # skip speed detection if scene is too static (pixels/frame)
  normalise_speed: true              # set false to detect but not retime
  remove_duplicate_frames: true      # broadcast freeze-frame removal (moved from shot_segmentation)
```

## New and Modified Files

| File | Action |
|------|--------|
| `src/stages/prepare_shots.py` | **New** — `ShotPrepStage`, `_estimate_speed_factor()` |
| `src/utils/ffmpeg.py` | **Modify** — add `retime_clip(path, speed_factor, fps)` |
| `src/schemas/shots.py` | **Modify** — add `speed_factor: float = 1.0` to `Shot` |
| `src/pipeline/runner.py` | **Modify** — insert stage at position 2, update aliases 2–7 |
| `config/default.yaml` | **Modify** — add `prepare_shots` block; deprecate `shot_segmentation.remove_duplicate_frames` |
| `tests/test_prepare_shots.py` | **New** — unit + integration tests |
| `tests/test_runner.py` | **Modify** — update alias assertions (2→`prepare_shots`, etc.) |

## Tests (`tests/test_prepare_shots.py`)

| Test | What it verifies |
|------|-----------------|
| `test_speed_factor_identical_clips_returns_one` | Same clip used as ref and shot → factor ≈ 1.0 |
| `test_speed_factor_slow_clip_returns_gt_one` | Shot with half the inter-frame motion → factor ≈ 2.0 |
| `test_speed_factor_static_scene_returns_one` | Near-zero flow in shot → static guard returns 1.0 |
| `test_retime_clip_shortens_slow_clip` | 2× slow clip retimed → new duration ≈ half original |
| `test_retime_clip_noop_when_factor_within_threshold` | Factor 1.05 → clip unchanged |
| `test_stage_skips_reference_clip` | First shot speed_factor stays 1.0, never retimed |
| `test_stage_normalises_slow_shot` | Second clip low flow → retimed, manifest speed_factor updated |
| `test_stage_infers_manifest_when_missing` | Clips in `shots/` with no manifest → stage runs cleanly |
| `test_stage_updates_shot_end_time_after_retime` | `end_time` reflects new duration after retime |
| `test_stage_is_complete_after_run` | `is_complete()` returns True after successful run |

## Error Handling

- If optical flow tracking yields zero successfully tracked points on all sampled pairs, treat as static scene and return `speed_factor = 1.0`.
- If FFmpeg retime fails, log a warning and keep the original clip unchanged (do not crash the stage).
- If a clip has fewer frames than `2 * n_samples`, reduce `n_samples` to `max(1, total_frames // 2)`.
