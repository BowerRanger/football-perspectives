# Static-camera solve from detected painted lines — design

## Context

`docs/superpowers/notes/2026-05-14-camera-1px-experiment.md` (Phases 2–3)
built an image-processing painted-line detector that hits **0.99 px mean
line RMS** per frame, but left an unresolved tension:

- **Per-frame independent line solves**: 0.99 px RMS, but camera centres
  spread **16 m** across the clip — not real body motion, just the
  local-minimum freedom of an under-determined per-frame solve.
- **Global static-C solve** (`scripts/global_solve_from_lines.py`): locks
  one camera body but line RMS plateaus at **~4 px**.

The note concluded "image-processing line extraction has plateaued" and
listed three suspects for the 4 px static-C floor: insufficient lens
model, frame-dependent detector bias, world-line catalogue error.

The `gberch` clip (Anfield) detects **7 distinct painted lines, all in
the left penalty box** (`left_18yd_*`, `left_6yd_*`, `left_goal_line`) —
two families of parallel lines, all coplanar and clustered in one corner.
That clustering is the most likely cause of the 16 m per-frame spread:
the camera centre is weakly determined along the viewing/depth direction
when every constraint sits in one region of the ground plane. That is
**depth ambiguity**, not detector noise — which means a static-C solve
*should* be able to pin C by tying together frames with differing
pan/tilt, *if* the clip has enough pan/tilt spread.

Crucially, the note's 4 px static-C result is **not a clean proof of a
model limit**: that solve seeded C from the *median of the 16 m-spread
point cloud* (a meaningless centroid), ran only 800 `nfev` for 2905
parameters, and never re-detected lines under coherent cameras.

## Goal & acceptance gate

Use the painted-line detector **and** produce a **static-camera** result.

**Acceptance gate: static C is non-negotiable.** The output camera track
has a single fixed camera centre `C` (zero body motion) across the whole
clip. Line RMS is driven as low as the model and optimisation can reach
*under that lock*, and is **reported** (quality report + experiment
note) — it is not itself a hard gate. When static-C and sub-pixel line
RMS conflict, static-C wins.

**Deliverable: production path + validation, in one go.** A static-C
line solve becomes the actual camera-stage output (behind the existing
`camera.line_extraction` flag, when `camera.static_camera` is true),
validated on the `gberch` clip within this same work.

## Approach — optimisation-first escalation

The note never isolated whether the 4 px static-C floor is an
*optimisation* failure or a *model* limit. This approach front-loads a
cheap diagnostic that answers that definitively and, either way, hands
the solver the correct seed. The lens model is escalated only if the
diagnostic proves the simple model insufficient.

Rejected alternatives:

- **Richer lens model directly** (skip the diagnostic, go straight to
  Brown-Conrady). If the 4 px was actually a bad seed — likely, given the
  median-of-spread seed — extra lens DOF just lets the LM overfit (the
  Phase-1 "FLOOR" failure mode on click data) without fixing the cause.
- **Iterative detect↔solve loop only.** Useful, but on its own it does
  not help if the seed is bad or the lens model is genuinely
  insufficient. It is a *component* of this approach (Step 3), not a
  standalone strategy.

## Architecture & module boundaries

Three new pure, independently-testable units, plus camera-stage
orchestration. Guiding boundary: detection, diagnosis, and solving never
import each other or the camera stage.

### New: `src/utils/static_line_solver.py`

The reusable static-C bundle adjustment.

```
solve_static_camera_from_lines(
    per_frame_lines: dict[int, list[LineObservation]],
    image_size: tuple[int, int],
    *,
    c_seed: np.ndarray,
    lens_seed,
    point_hints: dict[int, list[LandmarkObservation]] | None = None,
    lens_model: Literal["pinhole_k1k2", "brown_conrady"] = "pinhole_k1k2",
    per_frame_seeds: dict[int, tuple[np.ndarray, float]] | None = None,
    max_nfev: int | None = None,
) -> StaticCameraSolution
```

- Shared params: `(cx, cy, distortion…, Cx, Cy, Cz)`. Per-frame params:
  `(rvec, fx)`. `t_i = -R_i @ C`.
- **No `dC` parameter.** C is strictly one 3-vector — not a bounded
  per-frame perturbation, gone entirely.
- `StaticCameraSolution` carries the single `camera_centre`,
  `principal_point`, `distortion`, per-frame `(K, R, t)`, and per-frame
  line RMS.
- This is `scripts/global_solve_from_lines.py`'s core (residual fn +
  sparse Jacobian + LM call), extracted, generalised over `lens_model`,
  and tested.
- The `brown_conrady` model needs a **distortion-aware line residual**.
  `anchor_solver._line_residuals` is *not* distortion-aware; the
  distortion-aware variant lives in this module rather than mutating the
  shared anchor-solver helper.

### New: `src/utils/static_c_profile.py`

The C-profile diagnostic — both seed-finder and honesty check.

```
profile_camera_centre(
    per_frame_lines: dict[int, list[LineObservation]],
    image_size: tuple[int, int],
    *,
    c_grid,
    lens_seed,
    per_frame_bootstrap: dict[int, tuple[np.ndarray, float]],
) -> CProfileResult
```

For each grid C: solve every frame's `(rvec, fx)` independently with C
pinned and lens fixed to `lens_seed`, seeded from the propagated
bootstrap; record per-frame line RMS. Aggregate to mean / P95 / max as a
function of C. `CProfileResult` carries the full surface, the argmin C
(minimising mean per-frame RMS), and the per-frame `(rvec, fx)` at the
argmin (reused as the solver seed, not re-solved).

### New: `scripts/profile_static_c.py`

Thin CLI over `profile_camera_centre` for experiment-note runs — prints
the line-RMS-vs-C surface.

### New: `line_camera_refine.detect_lines_for_frames(...)`

```
detect_lines_for_frames(
    frames_bgr: dict[int, np.ndarray],
    cameras: dict[int, dict],
    distortion,
    detector_cfg,
) -> dict[int, list[LineObservation]]
```

Detection-orchestration helper — the loop currently inlined in
`scripts/iterative_global_solve.py`'s `detect_all` — so the camera stage
and scripts share one detect-all-frames path.

### Refactored scripts

`scripts/global_solve_from_lines.py` and `scripts/iterative_global_solve.py`
are rewired to call the three new modules instead of their duplicated
inline cost functions. They become thin scripts.

### Modified: `src/stages/camera.py`

`_run_shot`'s `line_extraction` branch becomes:

- `static_camera: true` → new `_refine_with_static_line_solve` (Steps
  0–3 below).
- `static_camera: false` → existing `_refine_with_line_extraction`
  (independent per-frame solves), kept for moving rigs.

### Reused from `anchor_solver.py`

`_make_K`, `_is_rich`. (`_line_residuals` is reused for the
`pinhole_k1k2` model only — see the distortion note above.)

## The algorithm

Five steps. Step 1 is a **decision gate**: its result determines whether
Step 4 is built in this work or deferred.

### Step 0 — Detect

Using the propagated per-frame cameras (camera-stage propagation step) as
detection bootstraps, run `detect_lines_for_frames` → `per_frame_lines`.
Frames with <2 usable detections are excluded from the solve (they keep
their propagated camera).

### Step 1 — C-profile diagnostic

Coarse-then-fine 3-D grid around the rich-anchor median C: coarse ≈7³ at
~2.5 m spanning ±7.5 m, then refine ≈5³ at ~0.5 m around the coarse
argmin. At each grid C, solve every frame's `(rvec, fx)` independently
with C pinned and lens fixed to the seed; record per-frame line RMS;
aggregate to mean / P95 / max.

Output: the **argmin C** (minimising mean per-frame RMS) and the
per-frame `(rvec, fx)` at that C.

**Decisive readout — mean line RMS at the best C:**

- <1 px → sub-pixel static-C is reachable, `pinhole_k1k2` suffices,
  Step 4 is **deferred**.
- ≈4 px → genuine model floor, Step 4 is **built**.

### Step 2 — Static-C bundle adjustment

`solve_static_camera_from_lines`, seeded from the profile argmin (C *and*
per-frame `(rvec, fx)`). Shared `(cx, cy, distortion, C)` + per-frame
`(rvec, fx)`; `t_i = -R_i @ C`.

Differences from the note's `global_solve_from_lines.py` that plateaued
at 4 px:

- C seeded from the **profile argmin**, not the median of the 16 m
  spread.
- **`dC` removed entirely** — not bounded, gone.
- **Well-converged** — `max_nfev` scaled to parameter count, tight
  tolerances.
- Point-landmark hints stay as an *optional, very-low-weight* basin
  regulariser. The line `world_segment`s fix the gauge; the points only
  catch gross basin errors. Weight is `cfg`-overridable, default low.

### Step 3 — Iterative re-detection

The Step-0 lines were detected under biased per-frame bootstraps.
Re-detect under the coherent Step-2 static-C cameras → re-run Step 2
(seeded from the previous solution, no re-profile) → repeat 2–3× or until
line RMS stops improving. This attacks the note's suspect #2
(frame-dependent detector bias).

Frames are re-read from the video each iteration — avoids a ~2.5 GB
in-memory spike for ~430 frames; detection dominates runtime anyway.

### Step 4 — Lens-model escalation (gated by Step 1)

*Only if* the Step-1 floor is >1 px: switch `lens_model` to
`brown_conrady` (k1, k2, p1, p2, k3 — shared, still static-C, using the
distortion-aware line residual), re-run Steps 1–3.

If *still* >1 px, the documented next rung is **zoom-dependent
distortion** (distortion coefficients as a low-order function of `fx`) —
**deferred to a follow-up** unless gberch validation forces it. The
`lens_model` parameter is the hook; the zoom-dependent variant is not
built speculatively.

Escalation is **config-driven** (`line_extraction_lens_model`), not
automatic — the gberch validation picks the right default.

### Always produced

Per the acceptance gate, the static-C track is produced regardless: C is
locked at the argmin, and whatever line RMS results is reported, not
gated on.

## Camera-stage wiring

`_refine_with_static_line_solve` orchestrates Steps 0–3, then writes
per-frame `(K, R, t)` back into the `per_frame_*` arrays.

**One-C consistency.** Static-C is the gate, so *every* frame in the
track must share the solved C — including frames the line solve skipped
(<2 detections) and frames only propagation covered. After the solve:
for any non-line-solved frame with a propagated `R`, set
`t = -R @ C_line`. The track's existing `camera_centre` field carries
the single line-derived C.

Per-frame confidence still derives from line RMS, exactly as the current
`_refine_with_line_extraction` does. The `detected_lines.json` debug
side-output (dashboard overlay) is still written.

## Configuration

**One new key** in `config/default.yaml` under `camera:`:

```yaml
  line_extraction_lens_model: pinhole_k1k2   # or brown_conrady — see Phase 4 note
```

C-grid extent/resolution, iterative-round count, and point-hint weight
are `cfg.get()` reads with hardcoded defaults — per-clip overridable, but
not written into `default.yaml`. Honors the experiment note's "one new
key max; per-clip overrides preferred" constraint.

## Data flow

```
propagated cameras
  → detect_lines_for_frames            (Step 0)
  → profile_camera_centre              (Step 1) → argmin C + per-frame seeds
  → solve_static_camera_from_lines     (Step 2)
  → [re-detect → solve]×N              (Step 3)
  → StaticCameraSolution               (single C, per-frame R/fx)
  → write back to per_frame_{K,R,t}; skipped frames get t = -R @ C_line
  → CameraTrack (single camera_centre) + <shot>_detected_lines.json
```

## Error handling

- <2 lines on a frame → excluded from the solve, keeps propagated `R`,
  `t` re-derived from `C_line`.
- C-profile finds no sub-pixel C → still returns the argmin; the track is
  produced at that C; the floor is reported honestly. Not a hard fail —
  this matches the acceptance gate.
- Detection finds nothing on all frames, or `scipy.least_squares` raises
  → log and fall back to the propagated track unchanged. The pipeline
  does not crash.

## Testing

- `static_line_solver` — synthetic: known C + per-frame `(R, fx)`,
  project catalogue lines with noise, assert C recovered within tolerance
  and sub-pixel RMS. Both lens models. Assert exactly one C in the
  output.
- `static_c_profile` — synthetic: the profile argmin lands at the true C;
  coarse→fine refinement converges.
- `detect_lines_for_frames` — light orchestration test over synthetic
  frames.
- Camera stage — with `line_extraction: true, static_camera: true`,
  assert the output track has one `camera_centre` and every frame
  satisfies `-R.T @ t == C` within tolerance.
- Existing `tests/test_anchor_solver.py` and `tests/test_camera_stage.py`
  stay green. 80%+ coverage on new modules.

## Validation — the deliverable

Run the new path on the `gberch` clip. Report:

- Line RMS distribution: mean / median / P95 / max.
- The single recovered camera centre C (body motion 0 by construction).
- Which lens model was needed.

Add a **Phase 4** section to
`docs/superpowers/notes/2026-05-14-camera-1px-experiment.md`: the
C-profile surface, the lens model used, and the final static-C line RMS.

## Constraints

- No new third-party dependencies. All lens models use OpenCV, already a
  dependency.
- Existing `tests/test_anchor_solver.py` and `tests/test_camera_stage.py`
  stay green.
- One new `config/default.yaml` key max; per-clip overrides preferred for
  everything else.

## Deferred

- **Zoom-dependent distortion** (distortion as a function of `fx`) — the
  escalation rung past `brown_conrady`. Built only if gberch validation
  proves `brown_conrady` insufficient.
- **World-line catalogue accuracy review** — the note's suspect #3. Out
  of scope here; if Step 3's iterative re-detection does not close the
  floor, catalogue error becomes the leading remaining suspect and is
  flagged for follow-up.
