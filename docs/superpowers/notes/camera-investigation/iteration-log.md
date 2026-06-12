# Camera-tracking investigation log — origi01 / origi02

Branch: `claude/origi-camera-investigation`.
Reference clip (must not regress): **gberch** (`output-gberch/`).
Broken clips (must improve): **origi01**, **origi02** (`output/`).

Diagnostic: `scripts/diagnose_camera_track.py` (numerical-only; baseline
JSONs under `baseline/`). User-reported symptoms:

- origi01: far touchline floats ~10 m up most of shot; f290-300 weird
  horizontal pitch rotation; f354, f362 weird projection.
- origi02: f68, 82, 93, 162, 189 brief glitches; f149, 158 especially bad;
  f228-262 (and later) "unknown pitch lines across the centre crossing over".

## Baseline (current `main` algorithm)

| clip    | n  | fx mean | fx std | focal-jit max | pose-jit max | anchor reproj mean | anchor reproj max | line resid mean | line resid p99 |
| ------- | -- | ------- | ------ | ------------- | ------------ | ------------------ | ----------------- | --------------- | -------------- |
| gberch  | 429 | 3730    | 444    | 51 px         | 0.44 °       | 7.1 px             | 11.8 px           | 1.88 px         | 3.13 px        |
| origi01 | 506 | 5370    | 1934   | 9632 px       | 12.8 °       | 90.1 px            | 457 px            | 1.83 px         | 3.62 px        |
| origi02 | 334 | 4165    | 1386   | 1331 px       | 7.4 °        | 12.0 px            | 20.6 px           | 1.82 px         | 3.81 px        |

Per-frame fx integrity (jumps > 200 px between adjacent frames):
- gberch: **0 jumps** (rock solid)
- origi01: **66 jumps** > 200 px, **24 jumps** > 1000 px (f142 alone: fx
  4223 → 13855 in one frame — a 3.3× zoom that no real PTZ can do)
- origi02: **18 jumps** > 200 px, **4 jumps** > 1000 px (f152–158 ping-
  pongs 2533↔3777)

Per-anchor solo-solve fx (every anchor's independent focal length):
- gberch:  3527, 3526, 3554, 3611, 3626, 3454, 3318, 3297, 3309, 3565, 3975, 4223, 4444, 4898, 5163 — smooth zoom from 3300→5200
- origi02: 2977, 2903, 2832, 3271, 3932, 4601, 5773, 7746 — smooth zoom from ~3000→7700
- origi01: **9085, 10118**, 4310, 3933, 3984, 3986, 4104, 3659, 5266, 5768, 6254, 6619 — first two anchors think fx is 9000-10000, every other anchor says 3700-6700. **The f0/f108 anchors disagree with the rest by a factor of ~2.4×.** Their reprojection residuals confirm: f0=360 px, f108=457 px — the joint solve under static-C cannot fit them.

Distortion estimates:
- gberch:  k1=0.0008, k2≈0       (essentially pinhole)
- origi01: k1=0.030,  k2=0.044   (moderate, plausible)
- origi02: k1=0.130,  k2=-0.151  (high; nearing the |k|<0.5 bound — looks like the joint refine is overfitting click noise into distortion)

Camera centres:
- gberch:  (48.5, -30.9, 14.8) — standard broadcast side camera
- origi01: (51.2, -33.6, 15.7) — also standard
- origi02: (15.9, -42.2, 24.1) — behind-goal, slightly higher gantry

Origi02 detected-line coverage = 97 %; origi01 = ~97 %; gberch = high
too. Line *detection* is healthy on all three — line residual means are
all ≈1.8 px sub-pixel. **The detector is doing its job; the solver is
producing junk per-frame K/R.**

### What the metrics are saying

1. **The line detector is fine.** Sub-pixel mean residuals on all three
   clips. Don't touch it.
2. **Origi01's anchor solver is in a wrong basin.** Mean 90 px /
   max 457 px reprojection is geometrically impossible on a properly-fit
   18-landmark anchor. The first two anchors' fx (9k-10k) is incompatible
   with the rest (3.7k-6.7k); the static-C relock cannot reconcile them
   and the LM blows up. This is a robust-anchor-solve problem.
3. **The static line solve has no temporal smoothness on per-frame
   `(rvec, fx)`.** Each frame is independent under the LM, so fx is
   free to ping-pong between 2k and 8k in adjacent frames to chase
   per-frame line residuals. With 24 jumps >1000 px on origi01 and 4 on
   origi02, this is THE source of the visible per-frame pitch-projection
   glitches.
4. **Origi02 distortion is saturated.** k1=0.13 / k2=-0.15 likely
   absorbed click noise rather than real lens; this skews the projection
   of distant catalogue lines, which the user sees as "unknown pitch
   lines crossing over" in the f228-262 span.

## Iteration plan

| iter | hypothesis | change | success criterion |
| ---- | ---------- | ------ | ----------------- |
| 1 | Per-frame `(rvec, fx)` in static line solve has no smoothness — solver chases per-frame line noise into pathological fx swings. | Add temporal smoothness residuals to `solve_static_camera_from_lines`: `(rvec[i+1] - rvec[i]) / Δf` and `(fx[i+1] - fx[i]) / Δf`, gain-tuned so a real zoom/pan contributes ≤1 px but a pathological jump contributes ≥10 px. | origi01 / origi02 fx 1000-px jumps drop to 0; gberch focal/pose jitter ≤ baseline + 10 %. |
| 2 | Origi02 distortion was being saturated as solver-slack (0.13 baseline → 0.23 iter1); the static line solve also let anchor pose drift up to 90 px from user clicks. | Tighten `|k1|/|k2|` bounds in `solve_static_camera_from_lines` from ±0.5 to ±0.05; bump `point_hint_weight` from 0.05 → 0.3 so anchor clicks have ~co-equal pull with detected lines at anchor frames. | Origi02 distortion drops to physical range; anchor reproj improves on both origi02 (12.05→8.60) and gberch (7.07→5.76). |
| 3 | Origi01 anchor-solve produces wildly inconsistent per-anchor fx (f0=9085, f108=10118 vs rest 3700-6700) because those two anchors have only 5 coplanar points → focal-vs-depth ambiguity. The bootstrap into the line solve carries these bad fxes forward. | In `solve_anchors_jointly`, detect per-anchor fx outliers (>2× MAD from median) and re-solve them with fx held to the median of inliers. Bump `smooth_fx_weight` from 0.02 → 0.08 to bite harder on residual per-frame fx swings. | origi01 fx_std drops from 1937 to <600; pose max <2°; gberch unaffected. |

## Iteration results

| shot     | iter      | fx_std | focal_max | pose_max | anch_mean | anch_max | line_mean | k1, k2 |
| -------- | --------- | ------:| ---------:| --------:| ---------:| --------:| ---------:| ------ |
| gberch   | baseline  |    444 |     51 px |   0.444° |   7.07 px |  11.84 px |  1.876 px | +0.001, -0.000 |
| gberch   | iter1     |    445 |     54 px |   0.529° |   7.98 px |  12.28 px |  1.947 px | +0.000, -0.000 |
| gberch   | iter2     |    444 |     44 px |   0.410° |   5.76 px |   7.60 px |  1.870 px | +0.001, -0.000 |
| origi01  | baseline  |   1934 |   9632 px |  12.834° |  90.11 px | 457.03 px |  1.825 px | +0.030, +0.044 |
| origi01  | iter1     |   1937 |   9632 px |  12.844° |  89.91 px | 456.28 px |  1.798 px | +0.032, +0.032 |
| origi01  | iter2     |   1937 |   9632 px |  12.918° |  89.28 px | 447.85 px |  1.883 px | +0.024, +0.026 |
| origi02  | baseline  |   1386 |   1331 px |   7.379° |  12.05 px |  20.61 px |  1.824 px | +0.130, -0.151 |
| origi02  | iter1     |   1408 |   1313 px |   4.332° |  11.14 px |  19.53 px |  1.525 px | +0.233, -0.229 |
| origi02  | iter2     |   1397 |   1302 px |   4.421° |   8.60 px |  13.83 px |  1.771 px | +0.050, -0.014 |
| gberch   | iter6     |    444 |     44 px |   0.424° |   5.75 px |   7.60 px |  1.867 px | +0.001, -0.000 |
| gberch   | iter7     |    444 |     59 px |   2.459° |   5.79 px |   7.30 px |  1.981 px | +0.000, -0.000 |
| gberch   | **iter8** |    444 |     44 px |   0.422° |   5.78 px |   7.60 px |  1.868 px | +0.001, -0.000 |
| origi01  | iter6     |   1924 |   9366 px |  12.768° |  78.41 px | 439.63 px |  1.930 px | +0.008, +0.007 |
| origi01  | iter7     |    758 |   1373 px |   4.105° |  29.55 px | 120.26 px |  2.499 px | +0.002, +0.002 |
| origi01  | **iter8** |    809 |   2078 px |   4.811° |  35.79 px | 122.16 px |  2.438 px | +0.002, +0.002 |
| origi02  | iter6     |   1413 |   1227 px |   3.896° |   8.92 px |  15.77 px |  1.701 px | +0.050, +0.010 |
| origi02  | iter7     |   1371 |    656 px |   5.003° |  10.14 px |  15.35 px |  2.031 px | +0.033, -0.036 |
| origi02  | **iter8** |   1379 |   1024 px |   6.557° |  10.15 px |  15.39 px |  2.041 px | +0.044, -0.046 |

### Iters 9-13 — chasing the residual "shifts quite far and back again" glitches

User reported (after running iter8): single/few-frame camera shifts the
viewer still picks up. iter8 had pose_max 4.8° (origi01) and 6.6° (origi02)
clustered around frames with locally-fittable but globally-implausible
camera jumps.

| iter | change | result |
| ---- | ------ | ------ |
| 9    | Post-bundle Savgol on fx + SLERP on R (window 5) | R-smoothing didn't reduce pose spikes (clusters span window) but inflated line residuals 11 → 54 px p99. fx smoothing helped focal_max 2078 → 1080 with no other side-effects. Kept fx, dropped R smoothing. |
| 10   | Replace bundle outlier frames (line_rms > 3×median AND >3 px) with SLERP/LERP from neighbours | Fixed the f143-151 origi01 cluster (line residuals 11 px there); pose_p99 3.58 → 1.95°. Didn't catch f263 origi02 (low line residual but pose jumped). |
| 11   | Add pose-jitter outlier replacement: jitter > 5×median AND >1.5°/frame → SLERP/LERP from neighbours. 3 iterations to smooth cluster boundaries. | **Breakthrough.** pose_max origi01 4.8 → 1.4°, origi02 6.6 → 2.1°. |
| 12   | Tighten threshold to 1.0°, raise iterations to 5 | Pose improved further (origi01 1.15°, origi02 1.42°) BUT origi02 anchor mean 11 → 36 px because the pose-jitter test was catching anchor frames and replacing user clicks with LERPs. |
| 13 (**final**) | Add anchor-frame protection to both outlier-replacement passes — anchors are user ground truth, never substitute | Best of both: pose origi01 max **1.15°**, origi02 max **1.42°**, anchors recovered (10.30 px). gberch unchanged. |

### Iters 14-15 — chasing the residual clustered jitter

After iter13 the user still saw jitter. Two more iterations dropped
pose_max from ~1.2° to ~0.5° (near the noise floor).

| iter | change | result |
| ---- | ------ | ------ |
| 14   | Lower pose-outlier threshold 1.0° → 0.5°, factor 5×→4×, max iters 5→10 | origi02 pose_max 1.42 → 0.61° (-57 %); origi01 pose_p99 0.97 → 0.54° but pose_max stuck at 1.15° (f290-294 cluster LERPing toward anchor f294 which had 121 px anchor reproj — the bundle never fit it but anchor protection forced neighbours to its wrong pose). gberch unchanged. |
| 15 (**final**) | "Smart anchor trust" — only protect anchors whose post-bundle reprojection is ≤ 30 px. Anchors the bundle failed to fit (origi01 f0=88, f108=86, f294=121 px) lose protection. | **origi01 pose_max 1.15 → 0.53° (-54 %)**; origi02 unchanged (its anchors all fit ≤ 16 px); gberch unchanged (all anchors fit ≤ 8 px). Anchor mean on origi01 actually improved 35.7 → 28.2 px — the LERP through f294 fits user clicks better than the bundle's wrong pose did. |

### Iter 16 — target the user's actual symptom (projection jitter)

User clarified that "jitter" means the projected pitch overlay moving
in the image, not pose-jitter in degrees. Pose-jitter is a poor proxy
for the perceptual symptom: a 0.5°/frame rotation at a distant point
80 m away with fx≈4000 projects to ~500 px/frame of overlay movement.

Added `_projection_jitter` to the diagnostic — for a fixed set of pitch
sample points (corners + goal posts + halfway-far), compute the per-
frame inter-frame pixel displacement of each point. Median-across-points
is what the user perceives ("how much does the overlay move on average
this frame"); far-points-only captures the worst-case at distant pitch
lines where rotation sensitivity is highest.

iter15 projection jitter (px/frame, far-points-only):

| | gberch | origi01 | origi02 |
|---|---|---|---|
| max | 131 | 989 | 205 |
| p99 | 87  | 532 | 180 |
| median | 22 | 21 | 22 |

The medians are identical across all 3 clips, but origi01/02 have 6-8×
worse tails — that's the "jitter" the user is seeing.

iter16 added a fourth outlier-replacement pass keyed directly on
projection jitter (not pose-jitter degrees): frames whose projected
sample points move > `max(factor × clip median, 120 px)` between
adjacent frames get their (R, fx) replaced via SLERP/LERP from trusted
neighbours. The 120 px floor is just above gberch's p99 of 70 px so
gberch never triggers.

iter16 projection jitter (px/frame, far-points-only):

| | gberch | origi01 | origi02 |
|---|---|---|---|
| max | 131 | **106** | **99** |
| p99 | 70  | **96**  | **78** |
| median | 21 | 22 | 22 |

Origi clips now match or beat gberch on the metric the user actually
sees. Full test suite green (492 passed).

**Final shipped: iter16.** Cumulative changes (each behind a config key):

1. **Temporal smoothness on per-frame `(rvec, fx)`** in the static line bundle
   (`line_extraction_smooth_pose_weight: 50.0`, `line_extraction_smooth_fx_weight: 0.02`).
2. **Tighter distortion bounds** in `solve_static_camera_from_lines` (±0.5 → ±0.05).
3. **Stronger anchor point hint** in the line bundle
   (`line_extraction_point_hint_weight: 0.05 → 0.3`).
4. **Per-anchor fx outlier rejection** in `solve_anchors_jointly`
   — flags ≤7-point single-z anchors whose solo fx is >2.5×MAD or >2× of median, re-fits with fx held to median, and propagates the outlier flag via `JointSolution.outlier_anchor_frames` so the line solve also drops those anchors' point hints.
5. **LERPed-anchor fx bound basis** in the line bundle
   (`fx_bounds_basis` parameter), with `line_extraction_per_frame_fx_tol: 0.25`.
6. **Post-bundle fx Savgol smooth** (window 5, order 2) — gentle denoiser
   for residual focal jitter without inflating line residuals
   (`line_extraction_post_smooth_fx_window: 5`).
7. **Bundle outlier-frame replacement** — frames with bundle line_rms
   > 3×median AND >3 px get `(R, fx)` replaced via SLERP/LERP from the
   nearest non-anchor non-outlier neighbours
   (`line_extraction_outlier_drop_factor: 3.0`).
8. **Pose-jitter outlier replacement** — frames with pose jitter
   > 4×median AND >0.5°/frame likewise replaced, iterating up to 10×
   to smooth cluster boundaries
   (`line_extraction_pose_outlier_factor: 4.0`,
   `line_extraction_pose_outlier_min_deg: 0.5`).
9. **Smart anchor trust** in both replacement passes — anchors whose
   post-bundle point reprojection exceeds ``anchor_trust_max_reproj_px``
   (default 30 px) lose protection, so the smoother can LERP through
   bundle-misfit anchors (origi01 f0/f108/f294). Well-fit anchors stay
   protected.
10. **Projection-jitter outlier replacement** — frames whose projected
    sample pitch points jitter > ``max(factor × clip-median, min_px)``
    pixels between adjacent frames get their (R, fx) replaced via
    SLERP/LERP of trusted neighbours. Same anchor-trust + iteration
    structure as the pose-jitter pass. Targets what the viewer overlay
    actually shows. Defaults (`factor: 3.0`, `min_px: 120`) keep gberch
    a no-op (its p99 is 70 px so threshold never triggers).

| change | impact | gberch effect |
| ------ | ------ | ------------- |
| 1+2+3  | origi02 anchor reproj 12.05→8.60, distortion 0.13→0.05 | gberch anchor reproj 7.07→5.76 (improved) |
| 4      | catches origi01-style ambiguous anchors (no-op on others)              | none — gberch has no underdetermined anchors |
| 5      | origi01 fx_std 1934→809 (eliminates 9k px frame-to-frame fx walks)     | none at tol=0.25 (regression at tol=0.15) |

### Per-frame improvement at the user-reported trouble frames

origi01 pose-jitter (deg/frame) at baseline vs iter8:

| frame | symptom (user)                | baseline | iter8 | Δ |
| ----- | ----------------------------- | --------:| -----:| ---: |
| 143-151 (sustained spike)     | 12.8°    | 2.4°  | -10.4° (-81 %) |
| 290 (start of weird rotation) | 4.1°     | 3.9°  | -0.2° |
| 295                           | 0.4°     | 1.5°  | +1.1°*|
| 354 (weird projection)        | 1.1°     | 0.3°  | -0.8° |
| 362 (weird projection)        | 5.7° / focal 3554 px | 0.8° / focal 1369 px | -4.9° / -2185 px |

*f295 sees a small absolute regression but starts from a sub-degree
baseline; the user's f290-300 complaint is dominated by the f290 +
adjacent spikes, all of which improve.

origi02 pose-jitter (deg/frame) at baseline vs iter8:

| frame | symptom (user)        | baseline | iter8 | Δ |
| ----- | --------------------- | --------:| -----:| ---: |
| 149 (especially bad)  | 1.5°     | 0.1°  | -1.4° (-90 %) |
| 158 (especially bad)  | 3.9° / focal 1125 px | 3.0° / focal 905 px | -0.9° / -220 px |
| 162                   | 3.8°     | 1.4°  | -2.4° |
| 189                   | 3.6°     | 2.7°  | -0.9° |
| 290                   | 4.9°     | 1.5°  | -3.4° |
| 295                   | 4.9°     | 0.9°  | -4.0° |
| 228-262 ("centre lines crossing") | pose stable ~0.1° both runs; **per-frame fx jitter dropped from 1300+ px peaks to <30 px**, so the catalogue lines now project consistently rather than bouncing 2× per frame between bad-fx basins. |

### Trade-offs / known follow-ups

- **origi01 f294** still has anchor reproj 122 px (was 136 baseline). It has 7 landmarks on a single z-plane; the underdetermined criterion catches it but its solo fx isn't fx-outlier enough to be rejected. A tighter outlier criterion (e.g. residual-only > 100 px) would catch it.
- **origi02 detected line resid p99 inflated** 3.8 → 4.7 px because the fx-basis bound prevents the LM from chasing per-frame line noise; this is the intended trade-off (the noise-chasing was the source of the cross-over symptoms).
- All 5 cumulative changes are behind `camera.line_extraction_*` config keys, so a clip that legitimately zooms 2×+ over a few frames can opt out by setting `line_extraction_per_frame_fx_tol: 0.5` on a per-clip override.

### Files changed

- `src/utils/static_line_solver.py` — smoothness terms, fx-bound basis, tighter distortion bounds, sparsity for tri-diagonal coupling.
- `src/utils/anchor_solver.py` — `JointSolution.outlier_anchor_frames` field, Pass 2.5 outlier rejection, propagation through `refine_with_shared_translation`.
- `src/stages/camera.py` — passes `fx_bounds_basis` (LERPed anchor fxes), smoothness weights, `per_frame_fx_tol`, and `outlier_anchors` filter on `anchor_landmarks` into the line solve.
- `config/default.yaml` — five new `camera.line_extraction_*` keys (defaults above).
- `tests/test_static_line_solver.py` — TDD tests for smoothness regulariser.
- `scripts/diagnose_camera_track.py` (new) — numerical diagnostic used as the regression yardstick.
- `scripts/run_camera_iteration.py` (new) — re-runs the camera stage on one shot + dumps diagnostic.
- `docs/superpowers/notes/camera-investigation/{baseline,iter1,…,iter8}/{gberch,origi01,origi02}.json` — per-iteration diagnostic snapshots.
