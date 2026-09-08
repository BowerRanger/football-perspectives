Review of HMR_World and Refine Poses — 7 September 2026
=====================================================

The first priority is rotation continuity. The saved refined animation still contains almost instantaneous half-turns, and the function intended to smooth root rotations is effectively a no-op. Foot contact cleanup is useful, but it cannot compensate for these orientation errors. After fixing the temporal integration, the larger improvement should be a joint pose-and-root refinement with anatomical and contact constraints, using the existing calibrated camera as fixed input.

This review inspected the implementation, evaluated all 22 current `gberch` player tracks (8,257 refined samples, 30 fps), and ran 193 focused tests. It does not change pipeline code, configuration, camera tracks, or existing animation outputs. Fresh measurements are saved in `output/animation_review/quality.json` and `output/animation_review/events.json`. These are measurements of the current NPZ files, not a new GVHMR extraction or a visual certification of the existing Blender renders.

The current pipeline
--------------------

HMR_World uses GVHMR with calibrated intrinsics and camera rotations, processes tracks in 120-sample chunks, negates local body rotations, aggregates shape, converts root orientation to pitch coordinates, smooths pose, applies a lean correction, and reconstructs root translation from ankle rays plus detected stance pins.

Refine Poses rejects root orientation outliers, reduces lean, adjusts ground height, trims unanchored ends, cleans translation, subtracts cross-player motion residuals, smooths the tracks, runs foot IK, and finally raises the body to clear a foot-based ground proxy. The current 22 tracks each contribute one shot, so all qualify for the foot-lock finale.

Useful foundations already exist: fixed per-player bone lengths from shape-adjusted FK; calibrated camera conditioning; contact hysteresis and sidecars; gap-aware smoothing in Refine Poses; robust translation cleanup; and two-pass rejection of unreliable IK spans. Preserve these while correcting the following problems.

Priority findings
-----------------

**1. Root rotation smoothing does not smooth; severe flips survive refinement. Confirmed defect and observed output problem.**

[`slerp_window`](../src/utils/temporal_smoothing.py#L77) constructs a SLERP through every input rotation and evaluates it at the original sample time. It therefore returns that sample's original rotation. Both HMR_World and Refine Poses call it. A seven-frame reproduction containing one 30° spike retains the full spike; maximum matrix change is approximately 2.3e-17.

Measured geodesic root changes between consecutive real video frames in the saved refined tracks:

| Player | Frames | Root rotation change in 1/30 second |
| --- | --- | ---: |
| P020 | 53 → 54 | 176.96° |
| P005 | 249 → 250 | 164.28° |
| P017 | 426 → 427 | 156.57° |
| P008 | 83 → 84 | 107.33° |
| P021 | 311 → 312 | 93.03° |

These are rotation-matrix distances, so quaternion sign changes or axis-angle wrapping cannot explain them away. `_reject_root_R_outliers` is insufficient on these outputs. It also requires anchors on both sides, leaving track edges vulnerable.

Replace the pose-stage call sites with actual rotation filtering, then add robust sequence-level orientation recovery for flips, with joint/image evidence and angular velocity/acceleration costs. A quaternion filter such as the existing `quat_savgol` can provide an initial baseline, but averaging a 180° ambiguity alone is not a recovery strategy. Keep edits scoped to pose call sites so shared camera smoothing behavior remains unchanged.

**2. Pose representation and lean handling can introduce motion errors. Confirmed numerical risks; the sign convention still needs verification.**

HMR_World [negates all local body axis-angle vectors](../src/stages/hmr_world.py#L914). Its own comment says the underlying convention mismatch has not been isolated. This inverts each local rotation; it is not a general coordinate-basis conversion. Verify the same asymmetric pose through upstream SMPL FK, pipeline FK, and the rendered skeleton, using matching joints/rest frames and reprojection. Do not simply remove the negation based on this review: a compensating convention elsewhere could make that change worse.

Both stages smooth axis-angle components as ordinary scalars, and Refine Poses interpolates those components across short gaps. Equivalent representations near +π and −π can average toward a completely different pose. In a synthetic +179°/−179° sequence, the current nine-frame Savitzky–Golay filter introduces up to 133.28° of angular error. This demonstrates the method's failure mode; it does not prove that this branch crossing explains the current clip. Use rotations on SO(3) for every joint, including gap fill and future chunk blending.

[`_reduce_root_lean`](../src/stages/refined_poses.py#L168) applies a 70% correction up to 30° of lean, then abruptly applies none above 30°. A synthetic lean change from 29.9° to 30.1° becomes a 21.13° orientation change. The correction also pivots around a fixed rest-pose ankle approximation. Replace this hard threshold with a continuous, evidence-weighted prior; sprint lean, kicks, dives, and falls must remain possible. The upstream 5° lean correction should be included in the same ablation.

**3. GVHMR loses temporal continuity at chunk boundaries and tracking gaps. Confirmed integration problem.**

[`run_on_track`](../src/utils/gvhmr_estimator.py#L1241) runs disjoint 120-sample chunks and concatenates their results. There is no overlap, shared boundary context, or rotational blending. It also selects only detected video frames before calling the temporal model, without supplying their original time gaps. HMR smoothing and contact-speed calculations likewise operate on packed rows. Later gap-aware cleanup cannot undo an upstream temporal model that interpreted a long absence as one timestep.

Six current HMR tracks have gaps. P017's largest adjacent frame-number difference is 131 (4.37 seconds at 30 fps); P009 and P011 have differences of 117 and 115. These must not be presented to the model as ordinary adjacent frames.

Split inference and contact processing into contiguous runs at long gaps. Handle short gaps explicitly on a regular time grid, retain observation masks, and make duration thresholds frame-rate aware. Run overlapping inference windows within each run and blend local/root rotations on SO(3), or retain the best-supported central predictions. Cache extracted image features/keypoints separately so window experiments do not repeat the expensive extraction. Test boundary jumps separately from normal motion.

**4. The contact metric does not fully capture animation continuity. Confirmed evaluation gap.**

The current configuration sets [`edge_ease_frames: 0`](../config/default.yaml#L1324). The accompanying comment explicitly notes that stepping into a contact span is excluded from the stance skating metric. The unrestricted foot-speed metric does cover consecutive-frame transitions, but it is not broken down by contact state, and no joint-rotation transition metric is reported.

The evaluator prefers the solver's accepted contact set. That is useful for evaluating successful locks, but makes it insufficient as an overall contact-quality score: rejecting a difficult span removes it from that score. Current resolved contact coverage ranges from 2.3% to 35.2% of each player's frames. This is the fraction with at least one accepted contact, not a measured flight ratio. The stage summary records 142 locked spans, 10 skipped, and 20 unresolved.

Report raw candidate contacts, accepted contacts, coverage, and independently annotated contacts separately. Measure touchdown/liftoff position and angular continuity, not only motion inside accepted spans. Plan contact transitions over neighboring swing frames while maintaining the planted interval; simply increasing in-span easing trades a snap for sliding. Test against annotated source-video intervals so fewer accepted contacts cannot masquerade as better reconstruction.

**5. Sequential root/pose corrections do not enforce whole-body physical consistency. Confirmed architectural limitation.**

The final foot IK has a limit on the amount it can change a joint, not an anatomical joint-limit model. There is no general constraint against knee hyperextension, implausible joint twist, self-intersection, or motion inconsistent with support. Smooth rotations alone can still describe an impossible pose.

The carrier translation is reconstructed from the two ankle pixels intersecting a fixed-height plane, including outside confirmed contact. A swing or airborne ankle does not lie on that plane, so its ray intersection can move substantially even with a perfect camera. The local-pose-dependent offset cannot recover flight height from that assumption. GVHMR's recovered global trajectory and stationary-joint probabilities are not retained as downstream priors. Its body IK is already used: the vendored upstream pipeline writes its postprocessed body pose to both the camera and global parameter outputs, so upstream IK is not wholly being bypassed.

Refine Poses then separately changes orientation, ground height, XY motion, local rotations, and root height again. The penetration guard runs **after** the lock is verified and can invalidate its 3D pin residual; the persisted resolved set is not reverified after that raise. Its 2.5 cm clearance also differs from the 2 cm target foot height. The saved summary reports 7,802 of 8,257 samples raised, with a maximum raise of 28.44 cm. This is substantial reliance on the final correction, although the count also includes small target/clearance adjustments.

The current evaluation reports zero penetrating frames under its foot-joint/sole-proxy tolerance for every refined player. This is not mesh collision verification. Final per-player root-acceleration maxima still range from 71.48 to 133.06 m/s²; peak foot speeds range from 7.92 to 31.79 m/s. These are review signals, not universal biomechanical failure thresholds: pelvis acceleration differs from center-of-mass acceleration, and fast kicking feet need action-specific interpretation.

Build a short-window optimization over root translation, root/local rotations, and contact state with the camera fixed. Combine confidence-weighted image reprojection, a prior toward the recovered pose, geodesic temporal regularization, anatomically defined joint limits, ground nonpenetration, and stationary contact constraints. Preserve shape/bone lengths. Allow explicit stance, swing, and flight states; use an appropriate center-of-mass flight prior rather than forcing the pelvis itself to follow a strict ballistic path. Model heel/toe support and validate sole geometry as the solver matures. Evaluate every constraint on the final output instead of assuming a preceding pass's guarantees survived.

**6. Residual camera compensation remains active inside Refine Poses. Ablation recommended.**

The user reports that the camera is stable. The saved summary's direct common-motion jitter pass made zero corrections, but its residual-consensus pass changed all 429 evaluated frames, with a mean offset of 4.82 cm and maximum 19.23 cm. That does not establish that the corrections are wrong: common residual error can remain. It does mean that camera stability should not be inferred from this player-based statistic.

Compare Refine Poses with this residual pass disabled, holding calibrated camera data fixed. Judge source reprojection and motion continuity together. If it no longer contributes, remove or evidence-gate it; coordinated football movement should not automatically become a camera correction.

Recommended implementation order
--------------------------------

| Order | Work | Evidence required before promotion |
| --- | --- | --- |
| 1 | Capture regression clips/events; fix no-op pose rotation smoothing, hard lean transition, and SO(3) interpolation; verify pose convention. | The five measured flips addressed without flattening genuine turns; synthetic branch/lean tests pass; source reprojection retained. |
| 2 | Preserve real time through inference/contact detection; overlap GVHMR windows; persist raw predictions, stationary probabilities, and provenance. | No artificial seam or gap jumps; compare identical observed frames and fresh caches. |
| 3 | Add contact-boundary metrics and final-output verification; coordinate ground clearance with foot locking; ablate residual-consensus corrections. | Lower transition spikes with stable independently measured stance coverage; final contact/clearance constraints checked together. |
| 4 | Introduce constrained pose/root refinement and benchmark a learned physics prior. | Improvements across running, turning, kicking, heading/jumping, goalkeeper motion, occlusion, and distant players on held-out clips. |

GVHMR remains a sensible starting point: its method already predicts stationary-joint probabilities and root velocity, which can inform our fixed-camera refinement. The first experiment should retain and use those signals, not replace the camera solve. See the [authors' method description](https://zju3dv.github.io/gvhmr/).

For the later physics-prior experiment, **PhysPT** is particularly relevant because its paper describes applying physics-aware refinement to existing kinematic predictions. This is a candidate benchmark, not a demonstrated improvement on our football footage. Its public demo uses a CLIFF preprocessing path and documents a Python 3.7 environment, so integration would need an adapter and an isolated environment. Sources: [CVPR paper](https://openaccess.thecvf.com/content/CVPR2024/html/Zhang_PhysPT_Physics-aware_Pretrained_Transformer_for_Estimating_Human_Dynamics_from_Monocular_CVPR_2024_paper.html), [author implementation](https://github.com/zhangy76/PhysPT).

**WHAM** is a useful comparison for contact-aware trajectory recovery, but switching estimators would not by itself fix our rotation conversion, interpolation, or contact-transition problems. Use it as a benchmark if needed after the integration defects are corrected. See the [authors' project](https://wham.is.tue.mpg.de/index.html).

Validation performed
--------------------

`scripts/eval_foot_quality.py --output output --stage both` evaluated the 22 current players. Additional NumPy/SciPy diagnostics measured root/joint angular changes, foot-speed event frames, and tracking gaps, excluding nonconsecutive frame pairs from motion derivatives. The synthetic checks reproduced the SLERP no-op, axis-angle branch-cut error, and lean-threshold discontinuity.

The existing focused test selection passed: **193 passed**, with four dependency deprecation warnings. Selection: `test_hmr_world_stage`, `test_refined_poses_stage`, `test_refined_poses_cleanup`, `test_refined_poses_jitter`, `test_foot_lock`, `test_foot_contact`, `test_foot_quality`, and `test_gvhmr_estimator_camera_r`. Passing these tests does not validate physical plausibility; the measured flips show why output-level motion regressions are needed.
