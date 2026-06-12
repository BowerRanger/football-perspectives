# Refined-poses jitter reduction — design

Date: 2026-05-30
Status: draft (goal-driven, autonomous)

## Goal

Reduce the jittery output the user sees in the reconstructed scene by
operating in the **refined_poses** stage (faster to iterate than re-running
GVHMR). Three artifacts to eliminate:

1. **All players shifting on the pitch simultaneously** — camera-tracking
   artifact (common-mode drift).
2. **Players moving abnormally quickly over frames** — per-player tracking /
   monocular-depth artifact (teleports + high-frequency wobble).
3. **Missing frames for players** — gaps; interpolate skeletal motion.

Must be a *generic* approach that works across many clips. Validated against
`output` (origi01/origi02), `output-gberch`, `output-kroupi`.

## Findings from the data (raw hmr_world, 30 fps)

Measured with `docs/superpowers/notes/camera-investigation/jitter_metrics.py`.

| dir | shot | speed p50/p90/p99/max (m/s) | implausible >12 m/s | jerk p90 | drift p90/max (m/s) |
|-----|------|------|------|------|------|
| output | origi01 | 5.0/57/1093/2575 | 19% | 1811 | 95/2023 |
| output | origi02 | 7.0/65/168/816 | 33% | 2309 | 94/209 |
| gberch | gberch | 2.4/6/16/185 | 2% | 163 | 37/105 |
| kroupi | kroupi01 | 8.2/48/165/280 | 37% | 1871 | 117/466 |

The **current refined output** still shows 18–29 % implausible frames,
p99 speeds of 93–123 m/s and jerk p90 ~1700 — the existing passes barely
dent the dominant artifact.

Inspecting a single player (origi01 P001) shows the structure:
- a **transient high-frequency wobble** (~0.5–1 m/frame oscillation around a
  fixed point) at clip start,
- a **catastrophic single excursion** (a 19 m jump over 6 frames),
- riding on top of a **mostly-clean** trajectory (0.5–4.7 m/s).

### Why the existing passes don't fix it

- `_reject_root_R_outliers` handles **rotation** flips only — there is **no
  per-player translation outlier rejection**.
- `_apply_jitter_correction` / `_apply_residual_consensus_correction` remove
  only **common-mode** (cross-player) motion. Per-player teleports survive,
  and worse, they *contaminate* the cross-player median, weakening the
  common-mode estimate (artifact #1).
- Savgol/SLERP **smooth** but do not **reject**: a single-frame 85 m teleport
  is fitted by the polynomial and smeared across the window rather than
  removed.

## Approach

Add a **per-player robust cleanup** that runs in `_clean_single_track`,
**before** the cross-player consensus passes and the final Savgol/SLERP
smoothing. Ordering is the core idea: reject outliers and fill gaps first, so
the consensus median and the polynomial smoother both operate on a clean,
uniformly-sampled signal.

New per-player steps (XY-and-Z translation; rotation/pose follow the same
trusted/interpolated mask):

1. **Resample onto a uniform per-frame grid** across the player's anchored
   span. Missing frames (gaps) are interpolated: linear for `root_t` and
   `thetas`, SLERP for `root_R`. Interpolated frames get a reduced
   confidence so downstream consensus de-weights them. (Artifact #3.)

2. **Robust translation outlier rejection (Hampel, in metres).** For each
   frame compute a local robust centre (median over a ~0.4 s window) and the
   MAD; flag frames deviating by more than `max(k·1.4826·MAD, abs_floor_m)`.
   Also flag frames whose implied speed from *both* neighbours exceeds a
   physical cap (≈ `v_max`), i.e. a there-and-back spike. Flagged frames are
   re-interpolated from the nearest trusted neighbours. (Artifact #2, spikes.)

3. **Physical velocity limit** on the de-spiked `root_t`: a forward/backward
   pass that caps frame-to-frame displacement at `v_max / fps`, bounding any
   residual wobble's apparent speed without flattening genuine sprints.
   (Artifact #2, residual wobble.)

All parameters are in **physical units** (m/s, metres, seconds) tied to the
pitch coordinate system, so the same config generalises across clips and
frame rates. Everything is config-gated under `refined_poses.cleanup` with an
`enabled` flag and a passthrough default for tests.

Then the existing pipeline runs unchanged on the cleaned tracks:
4. cross-player Δ-consensus + residual-consensus (artifact #1, now cleaner),
5. per-player Savgol/SLERP smoothing (now polishing, not fighting outliers).

## Parameter selection

Empirically swept on all four dirs with the metrics harness before wiring
into the stage. Targets: cut implausible-frame fraction by >10×, jerk p90 by
>3×, and reduce drift p90, **without** flattening the clean regions (the
gberch scene, already calm, must not get worse — guards against
over-smoothing).

## Testing

- New unit tests (TDD) for each primitive: gap resample, Hampel rejection,
  velocity clamp — synthetic tracks with known spikes/gaps/wobble.
- Preserve genuine fast motion (a real sprint must survive the velocity
  limit and Hampel).
- Stage-level integration test: a track with an injected teleport + gap
  arrives clean in the saved RefinedPose; summary records counts.
- Existing `test_refined_poses_jitter.py` must stay green.
- Before/after metrics on all four dirs reported in the summary.

## Risks

- Over-smoothing genuine fast motion → guard with physical `v_max` and the
  gberch no-regression check.
- Interpolating long gaps can glide a player in a straight line; cap the
  max gap length that gets filled (longer gaps left as keyframe holds for
  the glTF LINEAR sampler).

## Implementation (as built)

New primitives in `src/stages/refined_poses.py`:
- `_velocity_limit_xy` — forward/backward displacement limiter (physical cap).
- `_hampel_outlier_mask` — local-median outlier detection in metres.
- `_slerp_fill` — SLERP rotations onto a dense grid.
- `_clean_player_translation` — densify short gaps + Hampel reject +
  velocity limit, carrying root_R/thetas/confidence. Runs in
  `_clean_single_track` **before** the cross-player consensus and the
  final smoother.
- `_clean_refined_translation` — re-applies the same cleanup to assembled
  **multi-shot** tracks, bounding the per-frame teleports introduced by the
  highest-confidence-pick assembly between disagreeing camera solves.

Config block `refined_poses.cleanup` (default-on in `config/default.yaml`,
default-**off** when the block is absent so existing minimal-config callers
keep legacy behaviour). Tests: `tests/test_refined_poses_cleanup.py`
(11 cases) + existing suites stay green (51 refined-poses tests, 29 consumer
tests).

## Results (measured on the four target outputs, 30 fps)

Before = the refined output prior to this change; After = with cleanup on.

| artifact | metric | output | gberch | kroupi |
|----------|--------|--------|--------|--------|
| #1 common-mode shift | max m/s | 2023 → **8** | 6 → **2.7** | 147 → **11** |
| #1 common-mode shift | p90 m/s | 55 → **6.2** | 2.6 → **2.0** | 41 → **9.5** |
| #2 implausible (>12 m/s) | % of frame-pairs | 17.8 → **1.0** | 1.3 → **0.3** | 28.6 → **2.6** |
| #2 jerk (accel proxy) | p90 m/s² | 1694 → **258** | 89 → **76** | 810 → **108** |
| #2 per-player speed | p99 m/s | 123 → **12** | 13 → **10** | 93 → **13** |
| #3 missing frames | gap-fill | 285 filled | long gaps split | 28 filled |

The already-calm gberch scene keeps its median motion (p50 2.2 m/s
unchanged) — only the noise tail is trimmed, confirming genuine motion is
preserved and the pass is not over-smoothing. The residual common-mode p50
of ~3–4 m/s on `output`/`kroupi` is real coordinated play motion (camera
pan / players tracking the ball), correctly retained.

Measurement tooling: `docs/superpowers/notes/camera-investigation/`
(`jitter_metrics.py`, `characterize_jitter.py`, `prototype_cleanup.py`).

To re-validate locally (the refined_poses math needs only numpy + scipy,
no torch/ML deps) — pinned to the project versions:

```bash
python3.11 -m venv .venv            # ignored by .gitignore
.venv/bin/pip install "numpy<2.0,>=1.26" "scipy<1.14,>=1.11" pyyaml pytest
.venv/bin/python -m pytest tests/test_refined_poses_cleanup.py \
    tests/test_refined_poses_stage.py tests/test_refined_poses_jitter.py
# re-run the stage on an existing output dir, then measure:
.venv/bin/python docs/superpowers/notes/camera-investigation/jitter_metrics.py
```
