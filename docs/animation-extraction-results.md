# Animation extraction results

**Status, 8 September: rejected after visual review.** The user observed worse root drift/sliding across the players they checked. The numerical improvements below did not establish better animation and should not be treated as an accepted result. A separate PhysPT experiment evaluated physical support and trajectory behavior against the preserved original animation — see [physpt-experiment-results.md](physpt-experiment-results.md).

The HMR_World and Refine Poses changes were evaluated against the saved `gberch` animation for all 22 players. Open the [interactive before/after review](../output/animation_comparison/index.html) for synchronized skeleton playback, the original broadcast, matching rendered clips, and per-player measurements.

The comparison uses 8,257 identical frame/player samples at 30 fps. Motion derivatives exclude gaps in the original video timeline. Both versions are scored against the **same original 2D keypoints and candidate-contact labels**; these are reconstruction evidence, not ground truth. Every original comparison sample is retained.

| Measure | Before | After |
| --- | ---: | ---: |
| Root turns over 90° in one frame | 9 | 0 |
| Largest root turn per frame | 176.96° | 22.46° |
| Largest joint turn per frame | 38.29° | 29.83° |
| Mean root acceleration | 26.51 m/s² | 5.06 m/s² |
| Peak root acceleration | 133.06 m/s² | 66.61 m/s² |
| Mean body reprojection error | 8.50 px | 3.80 px |
| Peak foot speed | 31.79 m/s | 16.88 m/s |
| Mean foot speed at original contact boundaries | 3.53 m/s | 1.75 m/s |
| Largest joint step at original contact boundaries | 30.66° | 16.21° |
| Configured joint-limit violations (joint × frame) | 554 | 0 |
| Foot-proxy penetrating frames | 0 | 0 |

All 22 players improve on their individual peak root turn, peak root acceleration, and mean body reprojection error. Mean root acceleration falls 80.9%; mean reprojection error falls 55.3%. These derivatives measure reconstructed pelvis motion, not center-of-mass forces.

The fixed original candidate-contact coverage is **17.89%** in both versions. Mean foot speed inside those same candidate spans falls **0.562 → 0.441 m/s**. Accepted-contact coverage changes **15.16% → 19.68%**; mean speed inside each version’s accepted spans changes **0.121 → 0.049 m/s**. The fixed-candidate measurements are the fairer comparison because acceptance can change.

### Ablation

| Configuration | Mean reprojection | Mean root acceleration | Mean original-boundary foot speed |
| --- | ---: | ---: | ---: |
| Corrected extraction/rotations + previous foot IK | 6.5989 px | 26.17 m/s² | 3.01 m/s |
| Joint refinement, residual compensation off (selected) | 3.7994 px | 5.06 m/s² | 1.75 m/s |
| Joint refinement, residual compensation on | 3.7986 px | 4.99 m/s² | 1.74 m/s |

The residual pass provides only a 0.0008 px difference in average reprojection and small motion-metric differences. It is disabled because this clip’s calibrated camera is stable and the player-consensus correction adds no material image-fit benefit. The selected solver uses 160 iterations and angular-acceleration weight 8; refinement of all 22 players took 33.6 seconds on this machine with two Torch CPU threads. Initial tuning used this same clip, so these are development results.


## What changed

1. **Corrected the rotation convention at inference.** Our PyTorch3D compatibility shim assembled the 6D rotation basis as columns; its inverse extracted rows. The official [PyTorch3D implementation](https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/rotation_conversions.html#rotation_6d_to_matrix) assembles rows. Correcting the shim fixes root orientation as well as local pose. The compensating blanket negation of local rotations was removed after verifying asymmetric poses through upstream SMPL FK and pipeline FK. All 22 tracks were freshly extracted with the corrected shim.
2. **Made pose filtering operate on rotations.** Quaternion filtering now replaces the no-op SLERP evaluation. Joint smoothing and short-gap interpolation operate on rotations, avoiding the axis-angle ±π discontinuity. Robust root filtering handles isolated flips; a configurable angular-speed guard bounds initial reconstruction jumps. The lean threshold is continuous, and the empirical lean corrections are disabled with the convention fixed.
3. **Preserved time and inference context.** Missing observations split inference into contiguous runs. The 120-frame model windows overlap by 24 frames, selecting predictions with the most surrounding context. Contacts and motion derivatives also respect gaps. Image features, raw predictions, stationary probabilities, global trajectory, window origins, and extraction provenance are retained for subsequent experiments.
4. **Fit root and body pose together with the camera fixed.** The new solver combines confidence-weighted body reprojection, pose/translation priors, temporal acceleration penalties, foot contacts, ground clearance, and bounds on knee, elbow, and ankle rotations. Bone lengths remain constant for each fitted body shape. Elbow flexion uses a coordinate chart that can pass 90° without a representation switch. Final contacts are verified after the clearance guard.
5. **Fixed reappearance initialization.** When a contiguous track begins with uncertain ankles, its first valid anchor initializes the preceding uncertain positions within that run. This prevents zeros at the pitch origin from contaminating smoothing and speed limiting. Observation confidence remains low on those frames.
6. **Made evaluation and rendering consistent.** Candidate contacts and accepted contacts are reported separately, including touchdown/liftoff transitions. The Blender mesh and skeleton now use the same fitted body shape as refinement. Both review variants use this corrected shape handling and identical review cameras, so the visual comparison reflects their saved pose differences. Refinement cache checks reject changed HMR inputs or configuration.

## What the result establishes—and what remains

This is constrained kinematic fitting, not a dynamics simulation or a guarantee of physically correct motion. The limits cover knees, elbows, and ankles; they do not prove correct shoulder/hip motion or prevent body self-intersection. Ground clearance is a foot-joint proxy, not full mesh collision checking. The speed bound is a reconstruction setting, not a universal anatomical threshold.

Frames without contact evidence are not forced to a support height by the new solver; a synthetic airborne regression verifies that behavior. However, the upstream ankle-plane carrier can still underestimate real jump height. GVHMR's global trajectory and stationary probabilities are now saved, but are not yet used as a calibrated flight or center-of-mass prior. Source-video inspection of P005 includes small preparatory hops, so this limitation matters.

The original candidate-contact coverage is held fixed when comparing scores. Accepted-contact coverage is reported separately; acceptance thresholds differ from the old solver, so increased acceptance alone does not establish better contact detection. There are no independently annotated contact labels or motion-capture reference here. These settings have been evaluated on this clip, not held-out running, jumping, diving, kicking, and occlusion sequences.

The constrained finale currently applies to players with a single contributing shot, which covers all 22 players here. Multi-shot fusion retains its existing path. Long missing intervals remain missing in the saved reconstruction and in the comparison viewer; downstream render/export interpolation across absent intervals is a separate remaining limitation.

Residual heading ambiguity can still remain even when a sudden half-turn is removed. The P005 hop/turn interval should be judged against the source video; smoothness alone does not establish correct facing. P019 still has the largest local-joint step (29.83° in one frame) and root acceleration (66.61 m/s²), so further action-specific validation remains useful.

PhysPT has not been integrated or benchmarked in this pass. The saved raw predictions, fixed-camera comparison, and contact-transition metrics provide a baseline for that discussion and experiment.

## Artifacts and reproduction

- `output/animation_comparison/before/`: preserved original HMR and refined animation.
- `output/animation_comparison/after/`: selected updated HMR and refined animation, plus matching review renders.
- `output/animation_comparison/metrics/`: refinement ablations on the same HMR and camera inputs.
- `output/animation_comparison/comparison.json`: per-player measurements and weighted aggregate means/maxima.
- `output/animation_comparison/after/hmr_features/`: cached extraction features.
- `output/animation_comparison/after/hmr_world/*_raw.npz`: unfiltered predictions and retained upstream signals.

The selected HMR/refinement artifacts have also been copied into the current `output/hmr_world/` and `output/refined_poses/` directories. Both stages report current outputs. Original saved renders and FBX files remain from their earlier runs; the four review MP4s are the fresh matched comparisons. A link to this review has been added to `output/stadium_review/index.html`.

Reproduce the measurements without repeating extraction:

```sh
.venv311/bin/python scripts/compare_animation.py \
  --before output/animation_comparison/before \
  --after output/animation_comparison/after \
  --output output/animation_comparison
```

Re-run the refinement ablations using the saved fresh HMR:

```sh
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv311/bin/python \
  scripts/refine_animation_variants.py
```

The expensive fresh extraction completed for all 22 tracks. Subsequent refinement experiments reuse those predictions. Original calibrated camera data was verified byte-for-byte against its pre-change SHA-256 digest.

## Validation

The final focused Python suite passed **260 tests**, with six GPU-only tests skipped because Metal is unavailable inside the sandbox and four dependency warnings. The actual 22-player extraction ran with Metal access outside the sandbox. Blender validation separately passed **3 tests**: full animated FBX skeleton export, a player-render smoke test, and Blender joint positions matching pipeline FK.

Regression coverage includes rotation roundtrips, asymmetric upstream FK parity, isolated flips, the ±π branch, continuous lean handling, real-time gaps and overlap, reappearance with uncertain ankles, elbow flexion through 90°, finite optimization gradients, fixed bone lengths, joint bounds, final ground clearance, preservation of unsupported airborne height, contact-boundary spikes, and stale refinement-cache detection. The generated review page was opened and visually checked in isolated headless Chrome. Matching 61-frame render pairs were produced at 960 × 540, 30 fps, with eight render samples.
