# PhysPT physics-prior experiment results

**Status, 8 September 2026: benchmark complete on three configurations — PhysPT is rejected both as a post-pass and as a replacement refiner.** The released [PhysPT](https://github.com/zhangy76/PhysPT) model (CVPR 2024) was applied to all 22 `gberch` players three ways: as a post-pass on the preserved original animation, as a post-pass on the **current pipeline animation** (the September extraction/refinement rework, which is what `output/refined_poses/` holds today), and as the **only refinement** — run directly on the raw HMR extraction with none of our cleanup, smoothing, or foot locking. In every configuration it improves some smoothness measures but moves the body away from the image evidence on **every** player and introduces ground penetration and joint-limit violations the input did not have; as a sole refiner it is worse than the current pipeline on nearly every axis. Open the [interactive before/after review](../output/physpt_experiment/index.html) — a comparison selector at the top switches between all three configurations, each with synchronized skeleton playback, matching rendered clips, and per-player measurements.

This follows the [7 September review](animation-extraction-review-2026-09-07.md)'s recommendation to benchmark a learned physics prior (implementation-order item 4). The direct extraction/refinement rework was separately [rejected on visual review](animation-extraction-results.md); its artifacts remain the current pipeline output.

## What was run

- **Model**: the released pretrained PhysPT network, unmodified (checkpoint SHA-256 `20660b9e…`, author repo at `40d86992`), running on MPS in `.venv311`.
- **Inputs**: three configurations, refined independently —
  - current pipeline animation `output/refined_poses/` → `output/physpt_experiment/physpt_current/` (the review page's default);
  - raw HMR extraction `output/hmr_world/*_smpl_world.npz` (converted to per-player tracks in `physpt_experiment/hmr_input/`) → `output/physpt_experiment/physpt_only/` — PhysPT as the only refinement;
  - preserved original animation `output/animation_comparison/before/refined_poses/` → `output/physpt_experiment/physpt/`.
  Per-track input SHA-256 digests are recorded in each `*_physpt.json` sidecar and verified by the comparison (`--derived-after`, with `--derived-input` naming the staging directory when the input is not the before animation itself).
- **Camera fixed**: the author's CLIFF preprocessing and GlobalTrajPredictor are bypassed; our calibrated world-space motion feeds the transformer directly.
- **Time step**: PhysPT is a 20 fps model. Motion is resampled 30 → 20 fps (SLERP for rotations, linear for translation), refined in overlapping 16-sample windows, then resampled back. `src/utils/physpt_adapter.py` also converts to PhysPT's interleaved 6D rotation columns (round-trip verified against the author's decoder).
- **Trajectory stitching**: absolute per-window XY origins are arbitrary, so stitching integrates each window's central XY *increments* from one starting anchor per contiguous tracking run (the author's demo does the same). Height is the model's absolute prediction. SMPL `transl` is offset from the shaped rest pelvis, distinct from our world pelvis `root_t`.
- **Coverage**: all 29 contiguous runs across 22 players processed in each pass (none below the 16-sample window minimum).

## Results — current pipeline vs + PhysPT post-pass (8,257 identical samples)

Both versions are scored against the **same original 2D keypoints and candidate-contact labels**; these are reconstruction evidence, not ground truth. PhysPT was not given the keypoints or contact labels, so reprojection measures how far its prior pulls the body from the image.

| Measure | Current | + PhysPT |
| --- | ---: | ---: |
| Root turns over 90° in one frame | 0 | 0 |
| Largest root turn per frame | 22.46° | 20.91° |
| Largest joint turn per frame | 29.83° | 14.69° |
| Mean root acceleration | 5.06 m/s² | 3.81 m/s² |
| Peak root acceleration | 66.61 m/s² | 72.79 m/s² |
| Mean body reprojection error | 3.80 px | 11.36 px |
| Peak foot speed | 16.88 m/s | 12.21 m/s |
| Mean foot speed inside original candidate stances | 0.05 m/s | 0.77 m/s |
| Mean foot speed at original contact boundaries | 1.88 m/s | 1.74 m/s |
| Configured joint-limit violations (joint × frame) | 0 | 288 |
| Foot-proxy penetrating frames | 0 | 768 |

The current animation is already smooth, so PhysPT has little dynamics left to fix: it trims joint steps and mean accelerations slightly, but **trebles mean reprojection error** (worse on all 22 players; worst P010 4.00 → 25.82 px), makes peak root acceleration worse on 9 of 22 players (P021 36.77 → 72.79 m/s²), multiplies in-stance foot sliding ~14×, and introduces 768 foot-proxy penetrating frames and 288 joint-limit violations where the current output has zero. Integrated XY drift per contiguous run averages 0.42 m (max 2.00 m).

## Results — current pipeline vs PhysPT only (raw HMR in, same protocol)

Measurements at `output/physpt_experiment/only/comparison.json`.

| Measure | Current | PhysPT only |
| --- | ---: | ---: |
| Largest root turn per frame | 22.46° | 31.54° |
| Largest joint turn per frame | 29.83° | 21.50° |
| Mean root acceleration | 5.06 m/s² | 6.79 m/s² |
| Peak root acceleration | 66.61 m/s² | 276.17 m/s² |
| Mean body reprojection error | 3.80 px | 20.07 px |
| Peak foot speed | 16.88 m/s | 22.34 m/s |
| Mean foot speed inside original candidate stances | 0.05 m/s | 0.94 m/s |
| Configured joint-limit violations (joint × frame) | 0 | 448 |
| Foot-proxy penetrating frames | 0 | 3,494 |

PhysPT alone cannot replace the hand-built refinement: reprojection is worse on all 22 players (worst P007 4.73 → 78.34 px), peak root acceleration is worse on 21 of 22 (P021 spikes to 276 m/s²), and 42% of all samples penetrate the foot-proxy ground plane. Its only aggregate win is per-frame joint-step smoothness. The learned prior smooths locally but neither removes the raw extraction's trajectory noise nor respects our pitch plane or the image evidence.

## Results — preserved original vs PhysPT only (same protocol)

Measurements at `output/physpt_experiment/original_vs_only/comparison.json` (`?variant=original_vs_only` on the review page). This answers "what if we had replaced the whole refinement with just PhysPT" against the pre-September animation.

| Measure | Original | PhysPT only |
| --- | ---: | ---: |
| Root turns over 90° in one frame | 9 | 0 |
| Largest root turn per frame | 176.96° | 31.54° |
| Mean root acceleration | 26.51 m/s² | 6.79 m/s² |
| Peak root acceleration | 133.06 m/s² | 276.17 m/s² |
| Mean body reprojection error | 8.50 px | 20.07 px |
| Peak foot speed | 31.79 m/s | 22.34 m/s |
| Mean foot speed inside original candidate stances | 0.56 m/s | 1.00 m/s |
| Mean foot speed at original contact boundaries | 3.53 m/s | 1.89 m/s |
| Configured joint-limit violations (joint × frame) | 554 | 448 |
| Foot-proxy penetrating frames | 0 | 3,494 |

Against the rough original animation, PhysPT-only wins convincingly on rotation continuity (every >90° flip gone, largest root turn 177° → 32°) and mean dynamics — but the trade is the same in kind: reprojection more than doubles, peak accelerations spike higher than the original's worst, in-stance sliding nearly doubles, and 42% of samples penetrate the ground. It reads as smooth but detached from the footage.

## Gated takeover — PhysPT only on flagged spans (the first configuration that wins)

`scripts/experiment_physpt_hybrid.py` (+ `src/utils/physpt_hybrid.py`, 4 unit tests) keeps the current animation authoritative and splices in the PhysPT post-pass only where a detector flags the current motion as unrealistic: root acceleration above 35 m/s², root rotation step above 15°/frame, or mean keypoint confidence below 0.25 (occlusion). Flags dilate ±4 frames and merge into spans — **262 of 8,257 frames (3.2%) across 10 players**. Within each span the PhysPT trajectory is re-anchored to match the current animation at both edges (killing its drift by construction), Savitzky–Golay smoothed against its 20 fps jitter, and eased over 6 frames. Every span is then **verified per channel**: a takeover that worsens the span's own peak acceleration (translation) or peak rotation step (rotations) beyond 1.05× is rejected and that channel keeps the current animation. 20 channel-takeovers were accepted, 16 rejected. Results (`?variant=hybrid`, data in `hybrid/`):

| Measure | Current | Hybrid |
| --- | ---: | ---: |
| Largest root turn per frame | 22.46° | 20.91° |
| Largest joint turn per frame | 29.83° | 24.42° |
| Peak root acceleration | 66.61 m/s² | 58.76 m/s² |
| Peak foot speed | 16.88 m/s | 12.78 m/s |
| Mean body reprojection error | 3.799 px | 3.800 px |
| Configured joint-limit violations (joint × frame) | 0 | 14 |
| Foot-proxy penetrating frames | 0 | 31 |

Every player is equal-or-better on peak root turn, and the worst offenders improve substantially — P019's largest root turn drops 19.92° → 12.07° with its reprojection *also* improving (4.00 → 3.94 px, because the flagged frames were occlusion spans with weak evidence anyway); P005 21.01° → 18.27°; P013 18.94° → 15.75°. The 12 players with no flagged spans are bit-identical. Remaining blemish: 14 joint-limit and 31 penetration frames appear inside accepted spans (0.5% of samples) — the natural next step is adding both checks to the per-span verification gate.

This is the shape in which PhysPT earns a place: **local, evidence-gated, endpoint-pinned, and verified** — never global. Cost is unchanged (the PhysPT pass it splices from already exists; the splice itself takes seconds).

**Promoted to the pipeline default (8 September 2026).** The gated takeover now runs as the `refined_poses` stage's true final pass — `refined_poses.physpt_takeover` in `config/default.yaml`, implemented by `src/utils/physpt_refiner.py` + `src/utils/physpt_hybrid.py` with the experiment scripts as thin CLIs over the same code. The in-stage version additionally re-runs the raise-only penetration guard over spliced tracks, closing the 31-penetrating-frame blemish measured here. It skips with one warning when the PhysPT checkout/assets/torch are unavailable.

## Cross-clip check — kroupi01, unstable camera (current vs PhysPT only)

`kroupi01` has the least stable camera track of the eval clips (mean confidence 0.67; 19 of 156 frames below 0.5). PhysPT-only was run on its raw HMR extraction (9 players, 13 runs, one below the window minimum; 34 s model time) and scored against the current kroupi animation on 1,294 identical samples (`output/physpt_experiment/kroupi/`, `?variant=kroupi` on the review page):

| Measure | Current | PhysPT only |
| --- | ---: | ---: |
| Largest root turn per frame | 78.60° | 29.03° |
| Mean root acceleration | 30.98 m/s² | 16.59 m/s² |
| Peak root acceleration | 100.44 m/s² | **1,128.36 m/s²** |
| Mean body reprojection error | 17.64 px | **101.86 px** |
| Peak foot speed | 21.04 m/s | 34.29 m/s |
| Foot-proxy penetrating frames | 0 | 919 (71%) |

The shakier camera amplifies every PhysPT-only failure mode: P002's mean reprojection reaches 618 px (the player effectively leaves the image) and P001's peak acceleration hits 1,128 m/s². Rotation smoothness still improves — the prior does what it always does — but the output is unusable. The camera-noise sensitivity makes sense: PhysPT sees only the world-space trajectories, and an unstable camera solve injects exactly the kind of correlated root noise its increment-integrating trajectory has no way to correct against evidence it never sees.

## Runtime

Refining all 22 players (8,257 samples, 5,090 windows) with PhysPT takes **≈4 minutes wall-clock** on this Mac (MPS, ≈10.6 s per player; ≈233 s of model time, and the three runs' wall-clocks were 234–239 s including model load). The current pipeline's constrained joint refinement of the same 22 players takes **33.6 s** on the same machine (2 Torch CPU threads, from `output/animation_comparison/refinement_timings.json`) — so "just PhysPT" is roughly **7× slower** than the refinement it would replace. Both are small next to the HMR extraction itself, so runtime is not the deciding factor either way; quality is.

## Results — preserved original vs PhysPT (earlier pass, same protocol)

Data archived at `output/physpt_experiment/vs_original/comparison.json`.

| Measure | Original | PhysPT |
| --- | ---: | ---: |
| Root turns over 90° in one frame | 9 | 0 |
| Largest root turn per frame | 176.96° | 76.44° |
| Mean root acceleration | 26.51 m/s² | 7.15 m/s² |
| Peak root acceleration | 133.06 m/s² | 209.54 m/s² |
| Mean body reprojection error | 8.50 px | 19.41 px |
| Mean foot speed inside original candidate stances | 0.56 m/s | 1.04 m/s |
| Mean foot speed at original contact boundaries | 3.53 m/s | 1.91 m/s |
| Configured joint-limit violations (joint × frame) | 554 | 382 |
| Foot-proxy penetrating frames | 0 | 222 |

On the rough original animation the physics prior earns its keep on dynamics — every catastrophic root flip disappears — but the same image-fidelity and ground-contact losses appear (reprojection worse on all 22 players, new penetration, doubled in-stance sliding, 0.74 m mean XY drift).

## Interpretation

- **Consistent pattern across both baselines**: PhysPT's learned prior smooths whatever it is given, and the smoother the input, the more the remaining change is pure evidence loss. On the current animation the trade is clearly bad — small smoothness gains against a 3× reprojection cost, new penetration, and new joint-limit violations.
- **It is not anchored to our evidence**: with no reprojection term, the refined body drifts off the observed pixels everywhere, and its own learned ground/contact model conflicts with our pitch plane. The trajectory is only relatively correct — XY increments integrate to metre-scale drift over a 14-second track.
- **It cannot replace the pipeline either**: run directly on the raw HMR extraction, PhysPT-only loses to the current pipeline on almost every measure — the physics prior is not a substitute for evidence-anchored cleanup, trajectory reconstruction, and foot locking.
- **Practical conclusion**: PhysPT as a drop-in post-pass or standalone refiner is rejected in all three configurations. Its plausible value is as a *prior inside* a constrained fixed-camera optimization (the review's item-5 architecture) — e.g. its refined dynamics as a regularization target alongside confidence-weighted reprojection, our contact labels, and the calibrated camera — rather than as the final word on pose.

These settings were evaluated on this clip only, and both baselines are themselves reconstructions. No motion-capture reference exists here.

## Artifacts and reproduction

- `output/physpt_experiment/physpt_current/refined_poses/` — PhysPT applied to the current animation (+ `*_physpt.json` provenance sidecars); `physpt_only/refined_poses/` — applied to the raw HMR extraction (staged input in `hmr_input/`); `physpt/refined_poses/` — applied to the preserved original.
- `output/physpt_experiment/{physpt_current,physpt_only,physpt}/experiment.json` — checkpoint/commit digests, per-run windows, timings, endpoint drift.
- `output/physpt_experiment/comparison.json` / `motion.js` / `index.html` — current-baseline measurements and the interactive review page (`?variant=only`, `?variant=original`, `?variant=original_vs_only` select the other configurations); `only/`, `vs_original/`, and `original_vs_only/` hold their measurements.
- `output/physpt_experiment/{physpt_current,physpt_only,physpt}/review_render/gberch/` — matching rendered clips (same review cameras/settings as the preserved pairs). The PhysPT-only variants add two extra review cameras — `focus_p007` (frames 312–372, its worst reprojection) and `focus_p021` (frames 283–343, its worst acceleration spike) — with matching current-side clips in `current_render/gberch/` and original-side clips in `original_side/review_render/gberch/`. Every variant also opens with a `wide_orbit` pair (frames 250–400): a 120° arc at 20 m radius / 8 m height around the smoothed action centroid (`build_orbit_track`, 60° FOV), rendered identically for all five animation sides so whole-squad position and motion differences are visible at once.
- `third_party/PhysPT` — author checkout; assets under `assets/checkpoint/`, `assets/data/` (downloads preserved in `output/physpt_experiment/downloads/`).

```sh
# Refine all 22 current-pipeline players (≈4 min on this Mac, MPS)
.venv311/bin/python scripts/experiment_physpt.py \
  --input output --output output/physpt_experiment/physpt_current \
  --players P001,...,P022 --device mps

# Score and regenerate the review data
.venv311/bin/python scripts/compare_animation.py \
  --before output \
  --after output/physpt_experiment/physpt_current \
  --output output/physpt_experiment --derived-after

# PhysPT-only variant: stage raw HMR as per-player tracks, refine, then score
# with --input/--after physpt_experiment/{hmr_input,physpt_only} and
# --output output/physpt_experiment/only --derived-input output/physpt_experiment/hmr_input

# Re-render a matching review clip
blender --background --python scripts/blender_render_scene.py -- \
  --output-dir output/physpt_experiment/physpt_current --shot gberch \
  --cameras focus_p005 --render-root review_render \
  --width 960 --height 540 --samples 8 --frame-start 225 --frame-end 285
```

## Validation

`tests/test_physpt_adapter.py` (author-decoder 6D round-trip, real-speed 30→20 fps resampling, velocity-based stitching with tail coverage), `tests/test_pose_temporal.py`, and `tests/test_animation_quality.py` pass — 13 tests. The comparison's derived-after mode verifies each PhysPT track's recorded input digest against the exact before-file being scored, so stale or mismatched runs are skipped rather than silently compared. Review-render style parity with the preserved clips was verified frame-by-frame before reuse.
