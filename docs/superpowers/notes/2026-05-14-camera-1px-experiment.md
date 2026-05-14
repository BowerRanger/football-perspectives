# Camera tracking — sub-pixel residual experiment

Goal: every anchor frame in `output/camera/gberch_anchors.json` (Anfield, gberch clip) at **<1 px mean** and **<3 px max** per-landmark reprojection deviance, with clip-wide camera body motion staying **<2 m**.

Constraints:
- No new third-party dependencies (any considered ones are documented under "Deferred").
- Existing tests in `tests/test_anchor_solver.py` and `tests/test_camera_stage.py` stay green.
- Minimal additions to `config/default.yaml` (one new key max; per-clip overrides preferred).

## Harness

`scripts/measure_anchor_residuals.py` loads the anchors and runs the same `solve_anchors_jointly` + `refine_with_shared_translation` pipeline the camera stage runs, then reports per-anchor mean and max landmark pixel deviance, the locked camera centre, and the delta from previous run.

Run with: `.venv/bin/python3 scripts/measure_anchor_residuals.py output/camera/gberch_anchors.json`.

Optional flags:
- `--no-relock` — skip the static-camera relock; show solo-solve metrics only.
- `--lens-prior` — recover (cx, cy, k1, k2) via lens-from-anchor first.

## Approaches

Per rich anchor: `mean / max` landmark-deviance in pixels. **bold** when the anchor meets the <1 px mean & <3 px max gate. Body motion is the max-pairwise rich-anchor C distance.

| # | Variant | f0 | f45 | f205 | f282 | f371 | Body | Notes |
|---|---|---|---|---|---|---|---|---|
| 0a | Baseline — joint + static-camera relock | 4.31 / 9.79 | 4.09 / 7.85 | 13.34 / 32.12 | 5.40 / 9.27 | 8.83 / 18.71 | 0 (locked) | f205 see-saw |
| 0b | Baseline — solo solves only (no relock) | 4.24 / 9.24 | 3.76 / 8.54 | 3.35 / 8.78 | 4.68 / 10.26 | 4.90 / 11.19 | 5.37 m | Solo Cs disagree by 5 m — too much |
| 1 | + Anfield in stadium config (101 × 68) | 4.31 / 9.79 | 4.09 / 7.85 | 13.34 / 32.12 | 5.40 / 9.27 | 8.83 / 18.71 | 0 | No change to camera-stage residuals — all this clip's landmarks are goal-relative (x ≤ 16.5), insensitive to length. Anfield config feeds the ball/viewer stages instead. |
| 2 | + Joint multi-anchor lens prior (no relock) | 2.38 / 6.11 | 2.38 / 5.61 | 3.46 / 6.79 | 2.89 / 6.04 | 3.82 / 8.75 | 2.54 m | New `_estimate_lens_jointly`: shared (cx, cy, k1, k2) + per-anchor (rvec, tvec, fx) joint LM. Beats single-anchor estimator's 2× gate via the looser 1.3× joint gate. Recovers cx=958, cy=543, k1=+0.41 (saturated), k2=+0.50 (saturated) — distortion is absorbing non-lens noise, but it still tightens solos. |
| 3 | + Joint lens prior + static-camera relock | 2.43 / 6.17 | 2.64 / 5.00 | 13.98 / 30.98 | 3.01 / 6.25 | 7.52 / 13.82 | 0 (locked) | Relock forces shared C → f205 (whose solo wants C 1.5 m from rich centroid) pays the worst penalty. Other anchors benefit from the joint lens prior. |
| 4 | + Joint lens prior + bounded-motion relock (2 m L2 budget) | 2.33 / 6.51 | 2.34 / 5.92 | 6.18 / 12.24 | 2.94 / 6.18 | 6.65 / 16.28 | ≤2 m | New `refine_with_bounded_motion`: each anchor's C is within `max_motion_m` L2 of a shared centre. f205 stays at 2m bound, so still pays a residual penalty (it wants 2.4 m); others stay clean. Max anchor L2-motion 1.75 m, pairwise estimated ≤3.5 m. |
| FLOOR | Per-anchor 11-DOF (rvec, tvec, fx, cx, cy, k1, k2) — every param free per anchor | 2.07 / 6.79 | 2.18 / 5.30 | 2.90 / 6.92 | 2.82 / 6.00 | 3.72 / 8.30 | n/a (each anchor independent) | **This is the click-noise floor.** No model can fit these clicks below 2 px mean / 5 px max. k1, k2 saturate at ±1 on most anchors, meaning even maximum lens-DOF can't capture the residual structure. |

### Per-landmark systematic residual diagnostic (after row 2, joint lens prior)

Highest systematic per-landmark residuals (mean across anchors, no-distortion pinhole — used to spot consistent click/catalogue errors):

| landmark | mean ‖du, dv‖ | sd du | sd dv | comment |
|---|---|---|---|---|
| `left_6yd_goal_near` | 6.86 | 1.83 | 0.75 | Tight std → systematic click or world_xyz bias |
| `left_6yd_far` | 5.87 | 3.14 | 1.06 | High variance — likely click noise |
| `left_goal_crossbar_left` | 5.05 | 2.87 | 1.14 | Mixed |
| `centre_line_left_18yd_intersect` | 4.67 | 4.57 | 0.69 | Painted-line crossing, click ambiguity |
| `left_18yd_goal_far` | 3.81 | 3.75 | 1.32 | Variable |
| `centre_line_left_goal_intersect` | 3.10 | 1.92 | 0.63 | Tight std → systematic |

(See `scripts/measure_anchor_residuals.py` plus the ad-hoc diagnostics in the conversation log for full data.)

## Why <1 px mean / <3 px max wasn't reached

The per-anchor 11-DOF fit (every camera parameter free per anchor, no static-camera, no shared lens) is the absolute lower bound any model can hit against this anchor data — it gives each anchor maximum freedom and a perfect fit is then limited only by what's in the clicks themselves. That floor is **2–3 px mean / 5–8 px max** across all 5 rich anchors. Distortion parameters saturate at the ±1 bound on 4 of 5 anchors at that point, which is the LM trying to absorb non-lens noise into the only remaining knobs — a strong sign the model is no longer the bottleneck.

Two things would lift the floor:

1. **Sub-pixel click refinement.** The user-facing anchor editor offers no zoom UI or sub-pixel placement. Mouse-click placement on a 1920×1080 broadcast frame typically has 1–3 px noise per click, and several of the high-residual landmarks (`left_6yd_goal_near`, `centre_line_left_D_intersect`) consistently show 5–7 px shifts across anchors with sub-2 px standard deviation — that looks like a *systematic* click bias (probably clicking the inner vs outer edge of a painted line, where the painted line itself is ~12 cm wide and projects to several pixels at far-side landmarks).

2. **Line-segment fitting instead of point landmarks for painted features.** A line endpoint has the same click-noise issue, but the *line direction* fitted from many points along the line is much more accurate than a single point click. The current solver supports line correspondences but the existing anchor JSON only uses a handful per anchor; making them the dominant constraint (and demoting point landmarks to ambiguous-but-coarse hints) would likely reach <1 px.

Neither is reachable inside this session without UI/editor work or re-clicking the anchors with sub-pixel care.

## What got delivered

- **`config/stadiums.yaml`**: added `anfield` entry (101 × 68 m). Extended `StadiumConfig` with optional `pitch_length_m` / `pitch_width_m` so future per-clip dimensions can flow through to ball / viewer stages.
- **`src/utils/anchor_solver.py`**: new `_estimate_lens_jointly` that fits shared (cx, cy, k1, k2) jointly across all rich anchors with per-anchor (rvec, tvec, fx). Multi-seed grid handles the shallow cost basin near image centre. Used by the camera stage automatically when the single-anchor estimator rejects.
- **`src/utils/anchor_solver.py`**: new `refine_with_bounded_motion` — soft static-camera relock with a configurable max L2 motion budget per anchor. Not wired into the camera stage by default; available for callers (e.g. per-clip override configs) that know the camera body moves a small fixed amount.
- **`src/stages/camera.py`**: chains joint → single-anchor lens estimators with the existing `lens_from_anchor: true` flag controlling the whole pipeline.
- **`scripts/measure_anchor_residuals.py`**: reproducible harness for all measurements in this doc. Supports `--lens-prior`, `--lens-prior-joint`, `--no-relock`, `--motion-budget-m`, and `--drop-landmarks <substring>` for ad-hoc cleanup experiments.
- **Tests**: `test_joint_lens_estimator_recovers_pp_and_distortion` and `test_joint_lens_estimator_returns_none_on_single_anchor` added to `tests/test_anchor_solver.py`. Existing tests stay green (26 passed locally; full repo 219 passed excluding 1 pre-existing unrelated Blender snapshot test).

## Deferred (would need significant work or new dependencies)

- **Sub-pixel click placement in the anchor editor.** Would require zoom UI + (optionally) edge-detection on the underlying broadcast frame. Likely the most impactful single change. No new dep needed for zoom UI; edge-snap could use OpenCV (already a dep) but the algorithm needs care.
- **Replace point-landmarks-on-painted-lines with line-segment fitting.** Anchor editor would expose a "draw line along painted feature" tool; user marks 2 endpoints along the touchline, the solver fits the full line direction with sub-pixel accuracy. Schema change + UI work + solver weighting change.
- **OpenCV `CALIB_RATIONAL_MODEL` (8 distortion coefficients) or full Brown-Conrady (k1, k2, p1, p2, k3).** Tested empirically — adding tangential (p1, p2) and k3 gets the joint LM to 2.88 px (from 2.96 with k1+k2 only) but pushes (cx, cy) to bound corners, confirming the LM is overfitting click noise rather than recovering real optics.
- **Per-clip ground-truth via court-line detection from the video frames.** Image-processing extraction of painted-line positions would bypass click noise entirely. Wide scope — not in this session.

## Decisions log

- **Mow-line landmarks**: user removed them from `output/camera/gberch_anchors.json` between sessions. Confirmed correct — `near_mow_line_*` and `far_mow_line_*` in `src/utils/pitch_landmarks.py` use hardcoded y=17.5 / y=50.5 that don't match Anfield's actual grass stripes. Not removed from the catalogue (out of scope) but they should be deprecated or stadium-derived; flagged for follow-up.
- **Anfield pitch length**: 101 × 68 m added to `config/stadiums.yaml`, but the camera-stage residuals on this clip are insensitive to it (all landmarks goal-relative, x ≤ 16.5). The dimensions feed downstream stages (ball, web viewer) where pitch length matters.
- **Static-camera assumption**: kept on by default. Bounded-motion variant added but not wired into the camera stage default flow — needs a per-clip override config to enable, which can land when a real "≤ 2 m motion" clip is encountered. The synthetic and gberch data both have solo-C disagreements driven by click noise rather than real motion, so bounded-motion doesn't help for those.
- **Distortion sign + magnitude**: the recovered (k1, k2) on gberch is (+0.41, +0.50) — k2 saturated. This is non-physical for a real broadcast lens (negative k1 / barrel is the broadcast default). Treated as the LM absorbing model error into the available knobs. The estimator returns these values anyway because residuals do drop, but they should NOT be interpreted as the lens's true distortion; they're a regression coefficient.

# Phase 2 — Image-processing line extraction

User confirmed direction: replace point-landmark clicks with detected painted-line constraints. Detect on every frame; success metric is line-fitting RMS.

## Detector design

`src/utils/line_detector.py` — for each frame + bootstrap camera + line catalogue:
1. Project each world line to image space (with current `(K, R, t, distortion)`).
2. Clip to the image rectangle, walk along it at `sample_step_px` intervals.
3. At each cross-section: build a 1-D intensity profile across `search_strip_px` either side of the projected line.
4. Convolve the profile with a **bright-ridge template** (narrow positive lobe, dark flanks) — kernel responds to a 3-px-wide painted line surrounded by grass.
5. Locate the response peak; sub-pixel offset via parabolic interpolation of the 3 samples around the peak.
6. Reject cross-sections where the centreline lands off-grass (HSV-green-mask check around the candidate point).
7. RANSAC-line-fit the surviving sub-pixel centreline samples to reject occlusions / players crossing the line.
8. Project the predicted endpoints onto the fitted line → refined image-space line segment.

Per-line confidence = fraction of cross-sections producing RANSAC inliers. Configurable thresholds: `min_gradient` (paint-vs-grass contrast), `min_confidence`, `min_paint_width_px / max_paint_width_px`. Defaults are set so the detector survives moderate shadow without picking up grass texture.

## Approach trace

| # | Variant | Mean line RMS | Other |
|---|---|---|---|
| L1 | First detector: white-on-grass centroid centroids | 9.4 px | Saturates at 9 px because wide-white regions (mowing stripes) are picked up alongside the painted line |
| L2 | + Sobel-gradient edge-pair detector | 0.83 px on 2 lines | Sub-pixel achieved for narrow strips but only 2 lines fit; under-determined camera (other lines outside strip) |
| L3 | + Bright-ridge template + RANSAC + sub-pixel centroid | 0.99 px mean across 414/429 frames (every-frame run); 95th %ile = 1.99 px | 5 lines avg per frame, 96 % frame coverage |

## Per-frame solve metrics (414/429 frames)

```
Line-fitting RMS (per-frame solver, point hints @ 0.3 weight):
  mean    = 0.995 px
  median  = 0.942 px
  std     = 0.608
  min     = 0.000
  max     = 3.040
  P90     = 1.835
  P95     = 1.989
  P99     = 2.245
  frac <1px = 59.2%
  frac <2px = 94.9%
  frac <3px = 99.8%
```

**Mean meets the <1 px target**, and 99.8 % of frames meet <3 px max. Worst frame at 3.04 px.

## Caveat — body-motion drift

Per-frame solves are independent, so each frame finds the camera that locally best fits its 2–7 detected lines. With only that many constraints per frame, the under-determined dimensions of the (R, t, fx) parameterisation produce **16 m spread in camera centres across the clip**. That's not real camera motion — the broadcast camera is static — it's the LM's local-minimum freedom.

Two paths to enforce the static-camera contract:
- **Global solve with shared C** (`scripts/global_solve_from_lines.py`): 414 frames jointly solved with one C, per-frame (rvec, fx). 2 905 params vs 4 320 residuals — well-determined.
- **Temporal smoothing**: post-hoc median / kalman across per-frame (R, t) — easier to implement, may inflate per-frame line residuals.

The global solve is in progress; updates below.

## Sanity check (point landmarks)

With per-frame line-derived cameras, the point landmarks reproject at 6–9 px mean / 17–24 px max on rich anchor frames. That's WORSE than the lens-prior solve from Phase 1 — but it's consistent with our finding that the point clicks themselves carry 2–3 px noise plus systematic-bias on some landmarks. The line-derived cameras follow the painted lines, which are the ground truth.

## Global static-C solve attempts

Goal: enforce body motion <2 m while keeping line RMS sub-pixel.

| Variant | Lens seed | C seed | Line RMS mean | <1px frac | Body motion |
|---|---|---|---|---|---|
| Static-C, lens free, 800 nfev | (0,0,0,0) | rich median | 4.49 | 5.6 % | 0 (locked) |
| Static-C, lens free, point hint @ 0.2 | (0,0,0,0) | rich median | 4.49 | 5.6 % | 0 |
| Static-C, lens **fixed** to Phase-1 prior | (958, 543, +0.41, +0.50) | rich median | 4.22 | 15.9 % | 0 |
| Per-frame independent line solve | n/a | n/a | **0.99** | **59.2 %** | 16.3 m ❌ |

The pattern across all static-C variants: line RMS plateaus around 4 px, with the cost surface preferring **zero distortion** (k1, k2 → 0) when free, even though the per-frame solves prefer **saturated distortion** (k1=+0.41, k2=+0.50). The two LM regimes find different optima because per-frame fits are locally over-determined within their own clicks but globally under-determined across frames.

## What this means

- The image-processing line detector is **sub-pixel per frame** (median 0.94, P95 1.99). The detector itself meets the gate.
- The remaining residual is not detector noise — it's structural inconsistency between per-frame line observations under a single static-camera + radial-distortion model.
- Per-frame line-derived camera centres span 11 × 11 × 3 m. That's not real body motion (the broadcast camera is static); it's the local-minimum freedom of an under-determined per-frame solve. With only 2–7 detected lines per frame, a 1-D dimension of the camera-pose manifold is unconstrained, and the LM can slide along it freely.
- Forcing static C exposes the per-frame line biases. Either:
  - The lens model is insufficient (tangential distortion p1/p2 or higher-order radial k3 needed)
  - The detector has frame-dependent systematic bias (lighting, shadow, near-vs-far-side line width)
  - The world-line catalogue has tiny inaccuracies

Neither of those is solvable by changing the line detector. Image-processing line extraction has plateaued.

## Decisions / wrap-up

- Per-frame line-derived cameras in `output/camera/gberch_detected_lines.json` give **0.99 px mean line RMS across 414/429 frames** — the line-fitting gate the user defined.
- Static-C enforcement under the current model breaks the gate (~4 px mean RMS) — the body-motion constraint cannot be enforced without a richer model.
- Honest recommendation: keep both outputs. Use per-frame line-derived cameras for downstream stages that benefit from sub-pixel per-frame accuracy (hmr_world's foot anchoring, ball ground projection). Use the strict static-C for downstream code that demands a single camera body (3D viewer pitch overlay).

# Phase 3 — Sub-pixel click placement (recommended next step)

Status: per the user's instruction ("If you exhaust image-processing and don't achieve within 1 px results, then move on to sub-pixel click placement features"), this is the next thing to build. Sketch only — not implemented in this session.

## Why it's the right next step

The point landmarks have 2–3 px placement noise per click plus systematic bias on certain landmarks (`left_6yd_goal_near` consistently -6.83 px in du across anchors). That's the **floor** that no camera model can fit below.

Sub-pixel click placement attacks that floor directly: the user clicks approximately, and the editor refines the click to sub-pixel using local image processing. This combines well with the existing line detector — both share the ridge-filter + edge-pair sub-pixel math.

## Implementation sketch

1. **New endpoint** `POST /anchor/snap` in `src/web/server.py`: takes `{frame, click_xy, hint: "line_endpoint" | "point_landmark", landmark_name}` and returns refined `xy`.
2. **Snap logic**:
   - `line_endpoint`: use the existing `line_detector` ridge-filter logic in a small local window. Find the painted line nearest the click; project the click onto it. (Reuses `_sample_centreline_offset`.)
   - `point_landmark` for line intersections (e.g. `left_6yd_goal_near`): detect the two lines that pass through this landmark, compute their intersection. The two lines come from the catalogue per landmark name (e.g. for `left_6yd_goal_near` they're `left_goal_line` and `left_6yd_near_edge`).
   - `point_landmark` for non-intersection points (corner-flag-top, penalty-spot): fall back to local corner detection (Harris / cv2.goodFeaturesToTrack).
3. **Editor UX** in `src/web/static/anchor_editor.html`:
   - On click, immediately call `/anchor/snap`. Show a small loading spinner.
   - Display the refined position with a small confidence indicator.
   - Let the user override if the snap is obviously wrong.
4. **Bulk re-snap pass** for existing clicks: a "Refine all clicks on this anchor" button that runs every click through `/anchor/snap` and updates the JSON. Lets users upgrade legacy hand-placed anchors without re-clicking each one.

Expected impact: should drop click noise from 2–3 px to ≤0.5 px, lifting the noise floor enough for the existing lens-prior + bounded-motion solver to reach <1 px point residuals.

## Phase 3 — sub-pixel click placement (built)

### What landed

- `src/utils/click_snap.py` — local feature snap. Given a click, scans 18 orientations through a 60×60 px window for bright-ridge responses, then picks the best snap mode based on what's detected:
  - 2 roughly-perpendicular lines → snap to intersection.
  - 1 line → project click onto it.
  - 0 lines → return click unchanged.
- `POST /api/anchor/snap` endpoint in `src/web/server.py` — body `{shot_id, frame, click: [x, y], mode}`, returns `{xy, snapped, mode_used, confidence}`.
- Anchor editor (`src/web/static/anchor_editor.html`) calls `/api/anchor/snap` on every landmark or line-endpoint click. New "snap to lines" toggle in the toolbar (on by default).
- 2 integration tests in `tests/test_web_api.py` covering input validation and the no-feature fallback path.

### Empirical impact on gberch

Snapping all 87 point clicks across the 5 rich anchors moved 66 of them by **mean 2.30 px** — confirming the diagnosed 2–3 px click-noise floor.

Re-running the camera solve with snapped clicks:

| Anchor | Before snap (mean / max) | After snap (mean / max) |
|---|---|---|
| f0   | 4.31 / 9.79 | 2.33 / 5.91 |
| f45  | 4.09 / 7.85 | 2.84 / 5.64 |
| f205 | 13.34 / 32.12 | 14.90 / 33.02 |
| f282 | 5.40 / 9.27 | 3.06 / 6.73 |
| f371 | 8.83 / 18.71 | 8.22 / 17.04 |

Most anchors improved (f0, f45, f282 by ~40 %). **f205 didn't improve** — its issue is structural (the static-camera relock forces a shared C that doesn't fit f205's clicks regardless of how accurately they're placed). f205's see-saw needs the bounded-motion approach or richer lens model, not sub-pixel snap.

### Floor we're now at

- Click-noise floor: 2–3 px → ~1 px after snap (snap reduces but doesn't eliminate; detector itself has ~1 px bias).
- Lens-model floor: ~1–2 px (k1, k2 saturate at ±0.5 absorbing non-radial residuals — likely tangential + zoom-dependent distortion).
- Static-camera floor on f205-style frames: ~12 px (when the rich-anchor solo Cs disagree by 2–3 m, locking C to a single value forces one anchor to absorb a large residual).

The fundamental gate of <1 px max deviance across all anchor frames is **not reachable** without either:

1. A richer lens model (CALIB_RATIONAL_MODEL with k1..k6 + p1, p2, possibly per-zoom-level coefficients), OR
2. Allowing the camera body to truly move (relaxing static-C), OR
3. Both.

The sub-pixel snap is the right plumbing to support whatever direction comes next — it lowers the click-noise contribution so the remaining residual is model error, not data noise.

# Phase 4 — Static-camera solve from detected lines

User direction: use the painted-line detector **and** produce a static-camera result. Acceptance gate (confirmed with the user): a single fixed camera centre — zero body motion — is non-negotiable; line RMS is driven as low as it goes under that lock and **reported, not gated**. Deliverable: a production camera-stage path, validated on gberch.

Design: `docs/superpowers/specs/2026-05-14-static-camera-line-solve-design.md`. Plan: `docs/superpowers/plans/2026-05-14-static-camera-line-solve.md`.

## What landed

- **`src/utils/static_line_solver.py`** — static-camera bundle adjustment. Shared `(cx, cy, distortion, C)` + per-frame `(rvec, fx)`; `t = -R @ C`, no per-frame motion budget. `pinhole_k1k2` and `brown_conrady` lens models.
- **`src/utils/static_c_profile.py`** — C-profile diagnostic: sweep candidate camera centres on a 3-D grid, solving each frame's `(rvec, fx)` with `C` pinned; the argmin `C` (and the per-frame seeds at it) seed the BA. Subsamples frames for the grid sweep — the line-RMS-vs-C surface is smooth.
- **`src/utils/line_camera_refine.detect_lines_for_frames`** — detect-all-frames helper.
- **`src/stages/camera.py` `_refine_with_static_line_solve`** — detect → C-profile → BA → iterative re-detect, behind `line_extraction` + `static_camera`. The legacy independent per-frame path stays for `static_camera: false`. Output track carries a single `camera_centre` and the line-solve's principal point / distortion.
- `config/default.yaml`: `camera.line_extraction_lens_model` (one new key).
- `scripts/global_solve_from_lines.py`, `iterative_global_solve.py` rewired onto the shared solver; new `scripts/profile_static_c.py` CLI.
- Tests: `test_static_line_solver.py`, `test_static_c_profile.py`, `test_detect_lines_for_frames.py`, `test_camera_stage_static_line.py`. `test_anchor_solver.py` / `test_camera_stage.py` stay green.

## Two optimisation findings

**Sparse trust-region is broken for this problem structure.** scipy `least_squares` auto-selects the `lsmr` iterative trust-region solver whenever `jac_sparsity` is set, and it fails to converge on the block-arrow (shared + per-frame) structure — on *clean synthetic data* it hit `max_nfev` at multi-pixel error even at 30 frames. Fix: compute the grouped sparse finite-difference Jacobian via `approx_derivative(..., sparsity=spar)`, return it **dense**, and use `tr_solver='exact'`. Cheap Jacobian + working solver — 300 synthetic frames solve in 54 s to ~0 error.

**C-profile warm-start drift.** The first gberch profile reported a 14 px static-C floor. Cause: the profile warm-started each grid cell's inner LMs from the *previous cell's* solution; the meshgrid sweep starts at a far corner, so by the time it reached good cells the seed state was corrupted garbage the bounded-`nfev` inner LM couldn't recover from. Each cell now seeds from the stable per-frame bootstrap. (Also fixed: the profile seeded its fixed lens from the anchor solve's saturated `k1=0.44, k2=0.5` — now seeds zero distortion, the BA refines it.)

## Result on gberch (429 frames)

Production camera stage, `static_camera + line_extraction`, `pinhole_k1k2`:

| stage | mean | median | max | <1px | body motion |
|---|---|---|---|---|---|
| C-profile (lens fixed at 0 distortion) | 1.81 | — | — | — | 0 |
| **static-C BA — round 1 (kept)** | **0.949** | 0.814 | 6.902 | 74 % | **6.4e-14 m** |
| static-C BA — round 2 | 1.276 | 0.916 | 6.512 | 58 % | 0 |
| static-C BA — round 3 | 1.289 | 0.922 | 6.564 | 58 % | 0 |

**The static-camera gate is met:** one fixed camera centre `C ≈ (47.67, −29.24, 14.59)`, body motion `6.4e-14 m` — machine epsilon, i.e. zero by construction. Line RMS under that lock is **mean 0.949 px / median 0.814 px**, 74 % of frames sub-pixel — far below the ~4 px the Phase 3 "Global static-C solve attempts" table plateaued at.

Why this beats Phase 3's ~4 px: Phase 3 seeded `C` from the median of a 16 m-spread point cloud (a meaningless centroid), ran ~800 nfev for ~2900 params on the broken `lsmr` solver, and never re-detected. Phase 4 seeds `C` from the C-profile argmin, uses the dense `tr_solver='exact'` path, and frees the lens. The 16 m per-frame `C` spread was depth ambiguity from clustered coplanar constraints — tying frames with differing pan/tilt under one shared `C` resolves it.

Cross-check via `scripts/global_solve_from_lines.py` (seeds `C` from the rich-anchor median, no profile, single BA pass) reproduces the result exactly — mean 0.949 / median 0.814 / max 6.902 px, same locked `C`. On that solve the clicked point landmarks reproject at 4–12 px mean on the rich anchor frames — consistent with Phase 2's finding that the painted lines are ground truth and the manual clicks carry the noise.

## Iterative re-detection regresses — round 1 wins

The plan's iterative loop (re-detect lines under the coherent static-C cameras, re-solve, repeat) was expected to remove per-frame bootstrap bias. On gberch it does the opposite: round 1 (lines detected under the *propagated per-frame* cameras) is best at 0.949 px; rounds 2–3, re-detecting under the *shared static-C* camera, degrade to ~1.28 px. The shared camera is a per-frame compromise — it reprojects each frame's catalogue lines slightly off, so the detector searches strips slightly off and picks up *more* lines (2221 → 2489) but *noisier* ones. `_refine_with_static_line_solve` keeps the best round by mean RMS, so extra rounds are safe (never worse) but slower. `camera.line_extraction_static_rounds` (default 3) is a per-clip tunable — gberch would be just as good and ~3× faster at 1.

## Lens model — brown_conrady not warranted

The C-profile checkpoint suggested trying `brown_conrady` (its fixed-lens RMS read >1 px), but an actual BA pass shows `pinhole_k1k2` already reaches sub-pixel. On the Phase 2/3 detected-line set: `pinhole_k1k2` mean 0.765 px vs `brown_conrady` mean 0.693 px — a ~0.07 px gain, and `brown_conrady`'s `k3` saturates at the ±0.5 bound (the LM absorbing non-radial noise — exactly the overfitting Phase 1's FLOOR experiment flagged). Config default stays `line_extraction_lens_model: pinhole_k1k2`; `brown_conrady` remains available behind the key. Zoom-dependent distortion remains the documented, deferred next rung.

## Caveats

- The recovered lens still shows the saturated-distortion pattern from earlier phases: gberch's `pinhole_k1k2` BA lands `k2` at the +0.5 bound with the principal point pulled ~40–85 px off image centre. The line RMS is good (0.95 px) but, as in Phase 1–3, the lens parameters are a regression fit absorbing residual structure, not a recovered physical lens.
- `max` line RMS is 6.9 px — a tail of badly-fit frames (occlusion, players on lines, motion blur). Median 0.81 px shows the bulk is clean; the tail surfaces in the per-frame confidence timeline.
- The static-C camera stage run takes ~15 min on gberch (429 frames, 3 re-detect rounds). Dominated by line detection × rounds and the C-profile grid sweep; `line_extraction_static_rounds: 1` roughly thirds it.
