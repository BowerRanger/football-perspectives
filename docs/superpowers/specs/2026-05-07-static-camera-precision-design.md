# Static-camera lock + projected-pitch-line precision

**Date**: 2026-05-07
**Status**: Draft — awaiting user review

## Goals

1. **Hard-lock the camera body** when `static_camera=true`. Every output frame
   (anchor and inter-anchor) must satisfy `-R^T @ t == C_locked` exactly. No
   silent fallback to a "camera-moves" solution. Only rotation and zoom may
   vary per frame.
2. **Reduce worst-case projected-vs-real pitch-line gap from ~2 m to <0.3 m
   (~1 ft)** as judged on visual playback over the whole clip.

## Non-goals

- Per-frame line refinement on every frame (kept as a possible Phase 4 if
  Phase 3 doesn't land us under 0.3 m).
- Tangential lens distortion (`p1`, `p2`). Broadcast lenses are dominantly
  radial; tangential adds risk of overfit on a sparse anchor set.
- Re-engineering the inter-anchor feature propagator. The stage currently
  bypasses ORB propagation in favour of SLERP/LERP; that stays.
- Robustness for non-FIFA pitch geometry, snow-obscured lines, or
  `static_camera=false` clips. The strict-lock work is gated behind that flag,
  so non-static behaviour is unchanged.

## Background

The camera stage solves a joint bundle adjustment over user-placed anchor
frames (`src/utils/anchor_solver.py`). When `static_camera=true` (the default),
`refine_with_shared_translation` re-fits every anchor with a shared world-frame
camera centre `C_locked`. Anchor frames satisfy `-R^T @ t == C_locked` by
construction.

Two issues today, both relevant to the goals:

1. **Inter-anchor `t` is LERP'd while `R` is SLERP'd** (`src/stages/camera.py`).
   The LERP'd `t` does not satisfy the locked-centre constraint for the SLERP'd
   `R`, so the camera body wanders between anchors even when `static_camera=true`.
2. **The relock has a silent fallback** that returns the un-relocked solution
   if its mean residual is more than 10× worse. That path lets translation
   re-enter without the user noticing.

Independently, the user reports a roughly constant ~2 m worst-case gap between
projected and real pitch lines on every frame (anchor and inter-anchor alike).
A constant-across-frames gap points at anchor-solver precision rather than
interpolation drift. The two strongest unmodelled error sources:

- **No lens distortion**: `cv2.solvePnP` is called with a zero distortion
  vector. Broadcast lenses have ~1–3 % radial barrel; uncorrected, this is a
  systemic 1–3 m projection bias on a 105 m pitch.
- **Down-weighted line residuals**: `_LINE_RESIDUAL_WEIGHT = 0.2`. Pitch lines
  are arguably the most precise primitive available, but were down-weighted
  empirically because raw lines were dominating LM under bad clicks. A robust
  loss (Huber) is the principled fix.

## Phasing

Three staged PRs, each independently shippable:

| Phase | Scope | Expected gain |
|---|---|---|
| 1. Strict body lock | Make `static_camera=true` truly strict. Fix inter-anchor `t = -R_slerp @ C_locked`. Add `camera_centre` to `CameraTrack`. | Eliminates camera-body wander; correctness fix that sets up Phases 2/3. |
| 2. Lens distortion + solver hygiene | Add `(k1, k2)` clip-shared distortion to joint solver. Restore line weight to 1.0 with Huber loss. Tighten `anchor_max_reprojection_px` 10 → 2. Distortion stored on `CameraTrack`; downstream consumers updated. | Removes the systemic 1–3 m radial-distortion bias. Probably the biggest single win. |
| 3. Anchor-frame line polish | Auto-detect pitch lines (LSD) at each anchor frame, match to projected world lines, per-anchor LM on `(R, fx, k1, k2)`. | Sub-pixel anchor pose; pushes toward <0.3 m. |

## Architecture

```
src/
├── stages/camera.py
│   ├── Phase 1: replace LERP(t) with t_inter = -R_slerp @ C_locked
│   ├── Phase 1: fail loudly when relock can't honour C
│   └── Phase 3: orchestrate anchor-frame line polish before interpolation
│
├── utils/anchor_solver.py
│   ├── Phase 2: add (k1, k2) as 2 shared params to JointSolution + joint LM
│   ├── Phase 2: solvePnP/projectPoints calls take the distortion vector
│   ├── Phase 2: Huber loss on residuals; line weight → 1.0
│   └── Phase 1: refine_with_shared_translation no-fallback mode
│
├── utils/line_detector.py                        ← NEW (Phase 3)
│   ├── detect_pitch_line_segments(frame_bgr) using cv2 LSD
│   ├── filter to white-on-green (rough HSV mask) to reject crowd/graphics
│   └── return list[ImageLineSegment]
│
├── utils/anchor_line_polish.py                   ← NEW (Phase 3)
│   ├── For one anchor frame: project FIFA line catalogue with current pose
│   ├── Match detected segments → projected lines (midpoint distance + angle)
│   ├── LM on (rvec, fx, k1_in, k2_in) minimising perp-distance residuals
│   └── Returns refined (K, R, t) for that anchor; k1/k2 stay clip-shared
│
├── utils/pitch_lines.py                          ← NEW (Phase 3)
│   └── PITCH_LINES_FIFA: list[WorldLineSegment]
│       — touchlines, halfway, 18yd boxes, 6yd boxes, penalty arcs (sampled),
│         centre circle (sampled). Curves stored as polylines.
│
├── utils/camera_projection.py                    ← NEW (Phase 2)
│   └── project_world_to_image(K, R, t, distortion, world_points) → (N, 2)
│       — single source of truth for world→image projection.
│
├── schemas/camera_track.py
│   ├── + distortion: tuple[float, float]              # (k1, k2), default (0,0)
│   └── + camera_centre: tuple[float, float, float] | None
│
├── stages/hmr_world.py
│   └── foot-anchor ray-cast: undistort ankle pixels before back-projecting
│
├── stages/ball.py
│   └── ground-projection: undistort ball pixel observations
│
└── web/static/viewer.html, anchor_editor.html
    ├── viewer pitch overlay: applyRadialDistortion(uv, k1, k2, cx, cy, fx)
    │   on each projected pitch-line vertex before drawing
    └── anchor editor: stretch goal — sub-pixel snap to detected line intersections
```

### Module boundaries

- `line_detector` is a pure function `(BGR image) → list[ImageLineSegment]`.
  No knowledge of camera model. Unit-testable on a single dumped frame.
- `anchor_line_polish` takes `(JointSolution, anchor index, frame BGR,
  pitch-line catalogue, image_size)` and returns updated per-anchor `(K, R, t)`.
  Does not touch other anchors or interpolation. Returns `None` when too few
  matches to constrain the LM; caller keeps the joint-solve result.
- `camera_projection.project_world_to_image` is the single chokepoint for
  world→image projection. All downstream code (`hmr_world`, `ball`, viewer
  helpers, quality report) routes through it.
- `CameraTrack.distortion` defaults to `(0.0, 0.0)` so legacy saved tracks
  load unchanged.

## Phase 1 — Strict body lock

### Behaviour change

In `src/utils/anchor_solver.py::refine_with_shared_translation`:

- Remove the silent fallback at the 10× threshold. Replace with a logged ERROR
  identifying the worst-offending anchors, but **continue using the relocked
  solution**. Failure to converge becomes a visible problem the user fixes
  with more anchors, not a silent relaxation of the constraint.
- Persist `camera_centre = C_locked` on `JointSolution` (new field).

In `src/stages/camera.py`, the inter-anchor inner loop changes from:

```python
per_frame_K[idx] = (1.0 - lerp_w) * K_a + lerp_w * K_b
per_frame_R[idx] = slerp([lerp_w]).as_matrix()[0]
per_frame_t[idx] = (1.0 - lerp_w) * t_a + lerp_w * t_b   # ← drifts
```

to:

```python
per_frame_K[idx] = (1.0 - lerp_w) * K_a + lerp_w * K_b
per_frame_R[idx] = slerp([lerp_w]).as_matrix()[0]
if static_camera:
    per_frame_t[idx] = -per_frame_R[idx] @ C_locked
else:
    per_frame_t[idx] = (1.0 - lerp_w) * t_a + lerp_w * t_b
```

Anchor frames already have `t = -R_anchor @ C_locked` from the relock; no
change needed there.

### Schema

`CameraTrack` gains:

```python
camera_centre: tuple[float, float, float] | None  # None when static_camera=false
```

This is the canonical "where is the camera body" field and lets downstream
code assert the invariant cheaply.

### Risk

The 10× fallback existed for clips with collinear / far-side-only anchors that
produce a high-residual relock. With strict mode, those clips fail more
visibly. Mitigation: the error message names the anchors and tells the user
what to add (z-diversity, off-axis landmarks). The current code already logs
this — we elevate from warning to actionable error and keep going with the
relocked solution.

### Validation

- Unit test `test_static_camera_invariant`: synthetic 3-anchor clip with
  `static_camera=true`; every frame's `-R^T @ t` within 1e-6 of `C_locked`.
- Unit test `test_no_silent_relock_fallback`: catastrophic relock returns the
  relocked (not original) solution and emits an ERROR log.
- Unit test `test_inter_anchor_t_locked` (in `tests/stages/test_camera.py`):
  inter-anchor frame's `-R^T @ t == C_locked` exactly.
- Quality report adds `camera.body_drift_max_m`. With Phase 1 in place, must
  be `0.0` by construction.

## Phase 2 — Lens distortion + solver hygiene

### 2.1 Add `(k1, k2)` to the joint solver

`src/utils/anchor_solver.py`:

- Param vector grows by 2 globals (still clip-shared):
  `[tx, ty, tz, cx, cy, k1, k2, rvec_0(3), fx_0, rvec_1(3), fx_1, ...]`
- `_GLOBALS` becomes 7.
- Residual switches from manual `K @ cam` projection to `cv2.projectPoints`
  with the current `(k1, k2)`. Slower per call but the LM does ~hundreds of
  evaluations; total cost remains <1 s.
- `cv2.solvePnP` seed calls take the current `(k1, k2)` (zeros on the very
  first iteration; subsequently the joint estimate).
- Initial `(k1, k2) = (0.0, 0.0)`. **Bound-clipped to `|k1| < 0.5`,
  `|k2| < 0.5` after each LM step.** Real broadcast lenses sit comfortably in
  `|k1| < 0.2`; tight bounds prevent the LM compensating other errors via
  runaway distortion.
- `_solve_one_anchor_full` and `_solve_anchor_with_C_fixed` accept `(k1, k2)`
  as inputs and treat them as fixed during solo solves; refinement of `(k1, k2)`
  happens only at the joint level so they stay clip-shared.

### 2.2 Robust loss on residuals

- Drop `_LINE_RESIDUAL_WEIGHT = 0.2`. Lines run at full weight.
- Use `least_squares(loss="huber", f_scale=2.0)` on the joint LM. Huber gives
  quadratic loss for small residuals, linear for large — bad clicks/lines get
  capped automatically without poisoning the rest.
- Drop the `use_lines_on_seed` heuristic in `_refine_seed_pose`. With Huber,
  lines are safe to include unconditionally on the seed solve.

### 2.3 Tighten anchor-residual threshold

- `config/default.yaml`: `anchor_max_reprojection_px: 10.0 → 2.0`.
- This is a soft threshold — anchors above it get `confidence=0.5` and a
  warning, not a hard fail. Consistent with current behaviour.

### 2.4 Schema + downstream

`CameraTrack` gains `distortion: tuple[float, float]` (default `(0.0, 0.0)`).

New helper `src/utils/camera_projection.py`:

```python
def project_world_to_image(
    K: np.ndarray, R: np.ndarray, t: np.ndarray,
    distortion: tuple[float, float],
    world_points: np.ndarray,        # (N, 3)
) -> np.ndarray:                      # (N, 2)
    """Single source of truth for world→image projection across the codebase."""
```

Callers updated:

- `hmr_world.py` foot-anchor ray-cast: `cv2.undistortPoints` on the ankle
  pixel with `(k1, k2)` before back-projecting through the inverse `K`.
  Otherwise we anchor the foot to the wrong pitch point.
- `ball.py` ground projection: same undistort step on the ball pixel.
- `export.py`: glTF/FBX cameras stay linear (UE5/Three.js don't accept k1/k2
  natively). Distortion is applied to the *overlay* in the viewer, not baked
  into the exported camera. Means: video plays raw; pitch overlay is
  distorted in JS to match.
- `src/web/static/viewer.html`: small JS helper
  `applyRadialDistortion(uv, k1, k2, cx, cy, fx)` applied to each projected
  pitch-line vertex before drawing.
- `src/web/static/anchor_editor.html`: same distortion applied if a track
  already exists for the clip (so reopening an editor with a finished solve
  shows the corrected overlay).

### Risk

- **Tight `(k1, k2)` bounds are critical.** Without them, the LM can inflate
  distortion to compensate for a bad anchor click, hiding the real error.
- **Easy to forget downstream undistort steps.** If `hmr_world` or `ball`
  don't undistort, Phase 2 *introduces* a fresh ~30 cm bias in foot/ball
  positions. The new `camera_projection` helper plus a search-and-replace
  audit covers this.
- `cv2.projectPoints` doesn't expose a clean Jacobian for the sparse-Jacobian
  optimisation. Acceptable: ~10 anchors × ~10 obs = <500 residuals; LM fit
  takes <1 s either way.

### Validation

- Unit test `test_distortion_round_trip`: synthetic camera with `k1=0.1,
  k2=-0.05`; joint solver recovers within 0.01.
- Unit test `test_huber_dampens_outliers`: one bad landmark click 200 px off
  doesn't move the recovered `fx` by >1 %.
- Integration: re-run the saved checkpoint clip; expect
  `quality_report.camera.anchor_residual_mean_px < 2.0` and a visible drop in
  worst-frame gap on playback.

## Phase 3 — Anchor-frame line polish

### 3.1 New module `src/utils/line_detector.py`

```python
@dataclass(frozen=True)
class ImageLineSegment:
    p0: tuple[float, float]    # (u, v)
    p1: tuple[float, float]
    length_px: float
    angle_rad: float

def detect_pitch_line_segments(
    frame_bgr: np.ndarray,
    min_length_px: float = 40.0,
) -> list[ImageLineSegment]:
    """White line segments on green pitch via LSD restricted to a green-mask ROI."""
```

Internal pipeline:

1. HSV mask for "field green" (configurable range). Dilate ~5 px.
2. Within the green mask, threshold for "white" (V high, S low). Erode 1 px.
3. Run `cv2.ximgproc.createFastLineDetector()` (or
   `cv2.createLineSegmentDetector()` on older OpenCV) on the white mask.
4. Drop segments shorter than `min_length_px`. Drop segments wholly outside
   green-mask dilation.

Pure function — only OpenCV dependency. Unit-testable on a dumped frame from
any clip.

### 3.2 New module `src/utils/anchor_line_polish.py`

```python
def polish_anchor_against_lines(
    anchor_frame_bgr: np.ndarray,
    K: np.ndarray, R: np.ndarray, t: np.ndarray,
    distortion: tuple[float, float],
    pitch_lines: list[WorldLineSegment],
    image_size: tuple[int, int],
    C_locked: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """
    Refine (R, fx) for one anchor by matching detected pitch lines to the
    catalogue. Returns refined (K, R, t), or None if too few matches.
    t is rebuilt as -R @ C_locked to honour the static-camera invariant.
    """
```

Algorithm:

1. Detect image line segments via `line_detector`.
2. Project every world `pitch_lines` segment into the image with the current
   pose using `camera_projection.project_world_to_image`. Reject any whose
   projection is fully off-screen.
3. **Match**: for each detected segment, find the projected line whose
   perpendicular distance from the detected segment's midpoint is below
   ~10 px AND whose angle differs by <5°. Reject ambiguous matches (closest
   two within 30 % of each other in distance). Hungarian assignment is not
   needed because pitch lines are well-separated in image space when both
   midpoint distance AND angle gates apply.
4. **Build residuals**: for each matched pair, two residuals = perpendicular
   distance from each detected-segment endpoint to the projected world line.
   Reuses `_line_residuals` in `anchor_solver.py` with synthetic
   `LineObservation`s built from detected segments.
5. **LM** on `(rvec, fx)` with `t = -R @ C_locked` rebuilt inside the
   residual; `cx, cy, k1, k2` held at the clip-shared values. Huber loss.
6. **Sanity guard**: if post-LM mean residual is *worse* than pre-LM, return
   `None`. Caller keeps the original. Real failure mode (false matches
   dragging the pose) caught here.

Curved lines (centre circle, penalty arcs) are sampled into 8–16 short
straight segments at catalogue construction time. LSD detects curves as
multiple short collinear segments anyway, so the matching logic does not need
a curved-line code path.

### 3.3 New module `src/utils/pitch_lines.py`

```python
@dataclass(frozen=True)
class WorldLineSegment:
    a: tuple[float, float, float]
    b: tuple[float, float, float]
    name: str                          # for debug overlay labels

PITCH_LINES_FIFA: tuple[WorldLineSegment, ...] = (
    # touchlines, halfway, 18yd box edges, 6yd box edges, goal lines,
    # penalty arcs (8 segments each), centre circle (16 segments)
)
```

All coordinates in pitch-frame metres (z=0 ground plane).

### 3.4 Wiring into the camera stage

`src/stages/camera.py`, after the joint solve and relock, before inter-anchor
interpolation:

```python
if cfg.get("anchor_line_polish", True):
    for af in anchor_frames:
        frame_bgr = _read_frame(cap, af)
        K, R, t = anchor_solutions[af]
        polished = polish_anchor_against_lines(
            frame_bgr, K, R, t, distortion=(k1, k2),
            pitch_lines=PITCH_LINES_FIFA, image_size=(w, h),
            C_locked=C_locked,
        )
        if polished is not None:
            anchor_solutions[af] = polished
```

`(k1, k2)` stays clip-shared and is *not* updated per anchor in the polish
step — updating it per anchor would re-introduce the camera-body wander
problem at the focal-plane level.

Polish runs **after** `refine_with_shared_translation`. Overwriting an
anchor's `R` does not break the static-camera invariant because the polish
LM rebuilds `t = -R @ C_locked` inside its residual; the anchor's relationship
to the locked camera centre is preserved by construction.

### 3.5 Config additions

```yaml
camera:
  anchor_line_polish: true
  line_detection:
    min_segment_length_px: 40
    green_h_range: [35, 90]      # HSV hue
    green_s_min: 40
    green_v_min: 40
    white_v_min: 200
    white_s_max: 60
  line_match:
    max_midpoint_distance_px: 10
    max_angle_diff_deg: 5
```

### Risk

- LSD on broadcast frames returns segments from advertising boards,
  jumbotron, UI graphics, jerseys. Three-stage filter: green-mask gating →
  midpoint+angle match → sanity guard. If a clip still produces regressions,
  `anchor_line_polish: false` disables Phase 3 cleanly.
- Curved lines (centre circle, penalty arcs) are the trickiest matches. v1
  may need to disable polish for arcs and re-enable later if precision still
  short of target.
- Player occlusion of pitch lines is fine — LSD just doesn't see those
  segments. Matching tolerates partial line visibility (each line contributes
  only the segments LSD finds).

### Validation

- Unit test `test_lsd_on_pitch_frame`: dumped anchor frame produces ≥10
  segments, ≥4 unique world-line matches after gating.
- Unit test `test_polish_never_regresses`: synthetic anchor where polish
  would worsen residual returns `None`.
- Unit test `test_polish_recovers_R_perturbation`: synthetic anchor + 1°
  rotation perturbation; polish returns to within 0.05°.

## Validation, testing, success criteria

### Unit tests

| Test | Module |
|---|---|
| `test_static_camera_invariant` | `tests/utils/test_anchor_solver.py` |
| `test_no_silent_relock_fallback` | same |
| `test_distortion_round_trip` | same |
| `test_huber_dampens_outliers` | same |
| `test_inter_anchor_t_locked` | `tests/stages/test_camera.py` |
| `test_lsd_on_pitch_frame` | `tests/utils/test_line_detector.py` |
| `test_polish_never_regresses` | `tests/utils/test_anchor_line_polish.py` |
| `test_polish_recovers_R_perturbation` | same |

### Integration test

End-to-end test against the existing checkpointed clip
(`5c31eee CHECKPOINT: Good camera tracking`):

```python
def test_camera_stage_end_to_end_meets_precision_target(tmp_output):
    # Copy fixture clip + anchors into tmp_output, run camera stage with
    # phases 1+2+3 enabled.
    assert qr["camera"]["body_drift_max_m"] == 0.0           # Phase 1
    assert qr["camera"]["anchor_residual_mean_px"] < 2.0     # Phase 2
    assert qr["camera"]["worst_frame_pitch_gap_m"] < 0.3     # overall goal
```

### `worst_frame_pitch_gap_m` — definition

Per frame: project the FIFA pitch-line catalogue into the image with the
frame's `(K, R, t, distortion)`. For each projected line, find the nearest
detected line segment using the same gating as polish. Convert the
perpendicular pixel gap to metres on the pitch plane via inverse projection
at that pixel. The frame's `pitch_gap_m` is the max over matched lines.

Frames where line detection produces too few matches are excluded from the
metric and counted in `quality_report.camera.unmeasured_frames` for
honesty-of-coverage.

### Debug instrumentation

Written when `--debug-camera` flag is set, kept long-term (not just for dev):

- `output/camera/debug/anchor_<frame>_polish.jpg` — per anchor: detected
  segments green, projected red, matches cyan
- `output/camera/debug/worst_frame_<frame>_overlay.jpg` — auto-pick frame
  with largest `pitch_gap_m`, same overlay style
- `output/camera/debug/pitch_gap_timeline.png` — line plot of `pitch_gap_m`
  across all frames; anchors marked with vertical lines

### Success criteria

- **Phase 1**: `body_drift_max_m == 0.0`. Re-run saved clip, anchor frames
  look identical, inter-anchor frames now constrained to `C`.
- **Phase 2**: `anchor_residual_mean_px < 2.0`. Subjective: worst-frame gap
  visibly drops from ~2 m toward ~1 m on playback.
- **Phase 3**: `worst_frame_pitch_gap_m < 0.3` on the test clip. Subjective:
  every frame on playback has projected lines indistinguishable from real
  lines on a nearside touchline; far side may still have a small visible
  bias if the metric passes.

### Out of scope for validation

- Non-FIFA pitch geometry (training-ground pitches, unusual line markings).
- Heavy weather (snow obscuring lines).
- `static_camera=false` clips. Phase 1's strict mode is gated on the flag, so
  non-static clips are unchanged.
