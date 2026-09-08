# Motion-snap experiments — recovering the sharp movements

**Status, 8 September 2026: three fixes implemented and benchmarked; the compound passes every gate and improves reprojection.** Follow-up to the floaty-animation diagnosis (players read as too smooth / slower than reality). Open the [interactive before/after review](../output/snap_experiments/index.html) — a selector switches between each fix and the compounded result, with synchronized skeleton playback and matched render pairs on the clip's sharpest actions (P006's goal strike at frame 371, P019's sharp turn).

The **baseline** ("Current") is the shipping animation: the September refinement with the gated PhysPT takeover applied (promoted into `output/refined_poses/`). The diagnosis measured it at a **0.880 sharp-move speed ratio** (projected limb speed vs observed keypoint speed on the top-decile fastest observed frames; 1.0 = matches the video) with high-frequency limb detail 3.5× smoother than the pre-September animation.

## The fixes

**A — speed-adaptive refinement weights** (`refined_poses.kinematic_refinement.speed_adaptive`, code default OFF). Per-frame observed keypoint speed (confidence ≥ 0.5) modulates the joint refinement: on fast frames the three acceleration priors scale down to `acc_scale` (0.25) and the reprojection term is boosted up to `reprojection_boost` (2×), ramping linearly between `obs_speed_lo_px` (6) and `obs_speed_hi_px` (14). Quiet frames keep full smoothing, so noise suppression survives where there is no evidence of fast motion.

**B — lighter smoothing knobs**: `joint_angular_acc_weight` 8 → 3 and the final rotation `savgol_window` 7 → 5. No code change — pure config.

**C — procedural end-effector motion** (`refined_poses.end_effectors`, code default OFF; `src/utils/end_effector_motion.py`). GVHMR leaves SMPL hand joints frozen (measured 0.00°/frame) and toes near-rigid because COCO-17 evidence ends at wrists/ankles. This pass adds a relaxed hand curl replacing the mannequin-flat zero pose, follow-through (a lagged, decaying echo of wrist angular velocity — fast swings whip through the hand, clamped at 25°), and contact-driven toe roll (plantarflex easing out after lift-off, height-gated against pitch penetration; dorsiflex before touchdown). Hand-curl direction verified visually in renders.

## Results — 22 players, scored against the same observed keypoints

| | sharp ratio | HF limb p95 | knee p99 | foot p99 | mean reproj | flips >90° | penetration |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Current | 0.880 | 1.88 cm | 12.27° | 8.58 m/s | 3.801 px | 0 | 0 |
| A speed-adaptive | 0.899 | 3.35 cm | 14.17° | 9.50 m/s | **3.587 px** | 0 | 0 |
| B lighter smoothing | 0.884 | 1.99 cm | 13.85° | 8.74 m/s | 3.762 px | 0 | 0 |
| C end effectors | 0.880 | 1.85 cm | 12.41° | 8.58 m/s | 3.799 px | 0 | 0 |
| **Compound (A+B+C+takeover)** | **0.901** | **3.44 cm** | **14.85°** | 9.20 m/s | **3.587 px** | 0 | 0 |

- **A is the workhorse**: it recovers most of the recoverable sharpness (+16% knee peak turn, +78% high-frequency limb detail) while *improving* image fit — reprojection drops 3.80 → 3.59 px because the evidence is finally allowed to win on fast frames. This confirms the diagnosis: the uniform acceleration prior was fighting the keypoints exactly where motion was sharpest.
- **B alone is a blunter, smaller win** (+13% knee peak, +6% HF) — it cannot tell signal from noise, so it must stay conservative.
- **C is metric-neutral by design** — its change is visual (hands and toe roll; watch the render pairs). On the strike, P006's largest joint turn goes 16.6° → 22.6° and peak foot speed 11.7 → 13.7 m/s under the compound.
- **Gates hold everywhere**: zero >90° flips, largest per-frame root turn stays ≈22°, zero foot-proxy penetration, and the compound's joint-limit count (17, inside PhysPT takeover spans) matches the known takeover blemish.
- The remaining gap to 1.0 on the sharp ratio (~0.10) is **GVHMR's own attenuation** — the raw estimator already under-shoots the fastest observed motion (0.897 sharp ratio); no downstream reweighting can recover what the estimator never produced. That ceiling would need estimator-side work.

## Artifacts and reproduction

- `output/snap_experiments/{A_speed_adaptive,B_lighter_smoothing,C_end_effectors,compound}/` — variant stage outputs (A/B/C run with `physpt_takeover` off to isolate each fix; compound keeps the shipped takeover ON).
- `output/snap_experiments/cmp_<variant>/` — same-evidence comparisons; `snap_*.json` — snap metrics; `index.html` — the review page (template `scripts/snap_review.html`).
- `output/snap_experiments/before_render/` + `<variant>/review_render/` — matched clips (P006 strike frames 341–401, P019 turn frames 48–108), all rendered from identical cameras after the takeover promotion so the before side is the true current animation.

```sh
# Score any animation's motion snap against the observed keypoints
.venv311/bin/python scripts/eval_motion_snap.py --animation <dir> --reference output

# Re-run a variant: load config/default.yaml, merge the overrides listed above
# into refined_poses, and run RefinedPosesStage into a scratch dir with
# hmr_world/camera/shots/tracks/ball symlinked from output/.
```

## Validation

`tests/test_end_effector_motion.py` (relaxed curl + untouched body joints, follow-through echo, span-edge-only toe roll) and `tests/test_kinematic_refinement.py` pass; the refined_poses suite (103 tests) passes with the new hooks. All temporal metrics in `eval_motion_snap.py` respect tracking gaps (computed within contiguous frame runs only).

**Promoted to the pipeline default (8 September 2026)**: `config/default.yaml` now ships `speed_adaptive` and `end_effectors` enabled, `joint_angular_acc_weight: 3.0`, and `savgol_window: 5` — the compound configuration. Code defaults remain OFF for both new blocks, so bare test configs and light environments are unaffected. `output/refined_poses/` was regenerated with the promoted defaults and verified against the compound variant.
