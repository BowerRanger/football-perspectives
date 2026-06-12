# Ball Tracking Improvements

**Date:** 2026-05-11
**Status:** Draft — pending user review
**Scope:** `src/stages/ball.py`, `src/utils/ball_*.py`, `src/utils/bundle_adjust.py`,
ball-stage config, ball-stage tests

## Problem

The current ball pipeline produces three observable failures on `origi01`:

1. **Aerial passes projected as grounded.** Frames 101–191 (a diagonal switch
   of play) emit `state="grounded"` with `world_xyz.z = 0.11 m` for every
   frame. The IMM tracker's flight mode never trips, so the airborne ball is
   ground-projected and lands at y≈22–24 m on the far side of the pitch —
   visibly wrong on the top-down view. Per-frame detection confidences are
   healthy (0.4–0.7); this is a flight-classification failure, not a
   detection failure.
2. **Detection holes.** Frames 191–442 emit `state="missing"` for 195 of 252
   frames. The WASB detector returns `None` and the IMM has no observation
   to gate on. Only one short flight run (360–366) survives, and it is
   nonsensical.
3. **Implausible parabola/Magnus fits accepted.** Three of four flight
   segments on `origi01` are physically impossible:
   - seg 1: p0 = (-444, 962, -194) m
   - seg 3: p0 = (-5.7 × 10⁶, 9.4 × 10⁶, -2.2 × 10⁶) m at 0.11 px residual
   - seg 0: |v0| ≈ 254 m/s
   The config defines `plausibility.z_max_m` and
   `horizontal_speed_max_m_s`, but they are not enforced anywhere in
   `BallStage.run`.

The root structural issue is monocular depth ambiguity: a 6-parameter
parabola fit to ~6 short-baseline pixel observations is underconstrained,
so LM finds projection-feasible but geometry-absurd solutions. The
ground-projection fallback (`plane_z = ball_radius`) is wrong whenever the
ball is airborne.

## Goals

- On `origi01`'s diagonal switch (frames 101–191), the ball trajectory must
  follow a credible aerial arc with apex height in [3, 30] m, not a
  ground-skim at z = 0.11 m.
- No flight segment in `output/ball/*_ball_track.json` may contain
  per-frame world positions outside the pitch by more than the configured
  margin, with z outside [-1, `z_max_m`], or instantaneous speed above
  `horizontal_speed_max_m_s + 5 m/s`.
- Frames where the WASB detector misses but the IMM has a confident prior
  and a recent template should be bridgeable up to 8 frames without
  emitting `state="missing"`.
- Existing test suite (`tests/test_ball*.py`) continues to pass; new tests
  enforce the above.

## Non-goals

- No global joint-optimization rewrite (Approach B from brainstorm).
- No new detection backend. WASB stays, configured the same.
- No ball-shadow extraction.
- No change to glTF export schema. Outputs remain `BallFrame` and
  `FlightSegment` as defined in `src/schemas/ball_track.py`.
- No change to camera or hmr_world stages.

## Approach: layered fixes

Four layers stacked on the existing IMM + per-segment-fit pipeline. Each
layer is independently switchable via config and produces an
independently-testable unit.

### Layer 1 — Plausibility filter on fitted segments

**Where:** new module `src/utils/ball_plausibility.py`, called from
`BallStage._run_shot` immediately after every `fit_parabola_to_image_observations`
and `fit_magnus_trajectory` call.

**Function:**
```
is_plausible(p0, v0, omega_or_none, duration_s, fps, cfg) -> bool
```

Sample the integrated trajectory at ≥ 8 points over the segment duration
and check:
- `|x| ≤ pitch_length_m / 2 + pitch_margin_m`
- `|y| ≤ pitch_width_m / 2 + pitch_margin_m`
- `z ∈ [-1.0, z_max_m]`
- per-sample `|v(t)| ≤ horizontal_speed_max_m_s + 5.0`

On rejection: drop the segment. Per-frame `world_xyz` in `[a, b]` falls back
to whatever was there from ground projection (and if that is also outside
plausibility, the frame becomes `state="missing"`).

**Config (under `ball.plausibility`, mostly already present):**
```yaml
ball:
  plausibility:
    z_max_m: 50.0
    horizontal_speed_max_m_s: 40.0
    pitch_margin_m: 5.0   # NEW
```

Pitch dimensions are read from the existing top-level `pitch:` block.

### Layer 2 — Promote grounded runs whose ground projection is implausible

**Where:** new function in `src/utils/ball_plausibility.py`, called from
`BallStage._run_shot` after Layer 1 has filtered fitted segments and before
the final `BallFrame` write.

**Algorithm:**
1. Walk the emitted per-frame world positions in time.
2. Identify contiguous runs of `state="grounded"` of length ≥
   `flight_promotion.min_run_frames`.
3. For each run, compute:
   - max distance of any frame from the pitch interior (point-to-rect
     distance for an axis-aligned rectangle on the ground)
   - max inter-frame ground speed `‖xy(t+1) − xy(t)‖ · fps`
4. If `off_pitch_distance > off_pitch_margin_m` OR
   `max_ground_speed > max_ground_speed_m_s`, mark the run as a
   candidate flight.
5. Refit the run as a flight segment by calling
   `fit_parabola_to_image_observations` (and optionally Magnus) on the
   underlying observations.
6. Run Layer 1 plausibility on the new segment. If passes, replace the
   grounded states with `state="flight"`, set
   `flight_segment_id`, and append a new `FlightSegment`. If fails, set
   those frames to `state="missing"` with `world_xyz = None` — better an
   honest hole than a confidently wrong position.

**Config:**
```yaml
ball:
  flight_promotion:
    enabled: true
    min_run_frames: 6
    off_pitch_margin_m: 5.0
    max_ground_speed_m_s: 35.0
```

### Layer 3 — Kick-anchored fits

**Where:** new module `src/utils/ball_kick_anchor.py`; extends
`src/utils/bundle_adjust.py` with optional `p0_fixed: np.ndarray | None`
parameter on both `fit_parabola_to_image_observations` and
`fit_magnus_trajectory`.

**Inputs read:**
- `output/refined_poses/<shot>_refined.json` (preferred) — if absent,
  Layer 3 is silently disabled for that shot and a warning is logged.

**Algorithm per flight segment `[a, b]`:**
1. Build per-frame player foot positions (left + right ankle from
   `refined_poses`) projected from world to pixel using the per-frame
   camera.
2. At the seed frame `a`, find the nearest foot in **pixel** distance.
3. If that distance `≤ kick_anchor.max_pixel_distance_px` AND the next
   `kick_anchor.lookahead_frames` show a pixel-speed jump above
   `kick_anchor.min_pixel_acceleration_px_per_frame`, declare a kick.
4. Set `p0_fixed = (foot_x, foot_y, ball_radius_m)` and call
   `fit_parabola_to_image_observations(..., p0_fixed=p0_fixed)`.
5. The fit now optimizes only `v0` (3 params) instead of `(p0, v0)` (6
   params). Magnus refinement extends to fit only `(v0, ω)` (6 params)
   instead of `(p0, v0, ω)` (9 params).

**Why this helps:** depth along the camera ray at frame `a` is the
underconstrained dimension. Anchoring `p0` to a known 3D position (a foot
on the ground at z = ball_radius) collapses that ambiguity. The result is
a well-posed least-squares problem with substantially better noise
behaviour.

**Config:**
```yaml
ball:
  kick_anchor:
    enabled: true
    max_pixel_distance_px: 30
    lookahead_frames: 4
    min_pixel_acceleration_px_per_frame: 6.0
    foot_anchor_z_m: 0.11
```

**Stage ordering:** ball stage runs after refined_poses if available;
graceful degradation when not present.

### Layer 4 — Appearance bridging in WASB gaps

**Where:** new module `src/utils/ball_appearance_bridge.py`; called inside
the detection loop in `BallStage._run_shot` when `detector.detect(frame)`
returns `None`.

**Algorithm:**
1. Maintain a rolling template: a `template_size_px` × `template_size_px`
   crop of the most recent frame where the detector returned a high-
   confidence detection.
2. When WASB returns `None`:
   - Predict the next pixel position from the IMM tracker's blended
     prediction.
   - If `consecutive_misses ≤ max_gap_frames` and the template is fresh
     (`≤ template_max_age_frames` old):
     - Crop a `(2 * search_radius_px)`-wide window from the current frame
       around the predicted position.
     - Run `cv2.matchTemplate(..., cv2.TM_CCOEFF_NORMED)`.
     - If peak NCC `≥ min_ncc`, emit `(peak_uv, ncc * 0.5)` as a
       bridged detection. The 0.5 multiplier discounts bridged detections
       so the IMM weighs real WASB hits higher.
   - Otherwise return `None` (let the IMM gap-fill).
3. Update the template on every confident WASB detection
   (`confidence ≥ template_update_confidence`).

**Why capped at 8:** templates drift, especially across rapid scale or
lighting changes (the user's clip is a broadcast clip with pans). 8 frames
≈ 0.27 s at 30 fps — enough to bridge a partial-occlusion blip without
risking lock-on to a white sock or sideline marker.

**Config:**
```yaml
ball:
  appearance_bridge:
    enabled: true
    max_gap_frames: 8
    template_size_px: 32
    search_radius_px: 64
    min_ncc: 0.60
    template_max_age_frames: 30
    template_update_confidence: 0.5
```

## Data flow

```
WASB detect ──┬─> appearance bridge (Layer 4) ──> IMM tracker ──> steps[]
              │                                                      │
              └── confident detection updates template ◄─────────────┘
                                                                     │
                                                                     ▼
                                          per-frame ground projection
                                                                     │
                                                                     ▼
                                          flight runs (p_flight ≥ 0.5)
                                                                     │
                                                       refined_poses ┤
                                                                     ▼
                              kick anchor (Layer 3) ──> p0_fixed seed
                                                                     │
                                                                     ▼
                              parabola + Magnus fit
                                                                     │
                                                                     ▼
                              plausibility check (Layer 1)
                                                                     │
                                                                     ▼
                              per-frame world position assembly
                                                                     │
                                                                     ▼
                              ground-run promotion (Layer 2) ──> refit ──> Layer 1
                                                                     │
                                                                     ▼
                              BallTrack save
```

## File layout

| Path | Status | Purpose |
|---|---|---|
| `src/utils/ball_plausibility.py` | NEW | `is_plausible(...)`, `find_implausible_grounded_runs(...)` |
| `src/utils/ball_kick_anchor.py` | NEW | `find_kick_anchor(frame, segment_a, foot_positions, ball_uv)` |
| `src/utils/ball_appearance_bridge.py` | NEW | `AppearanceBridge` class with `update_template(...)` / `try_bridge(...)` |
| `src/utils/bundle_adjust.py` | MODIFY | Add optional `p0_fixed` to `fit_parabola_to_image_observations` and `fit_magnus_trajectory` |
| `src/stages/ball.py` | MODIFY | Wire the four layers; optional refined_poses dependency |
| `config/default.yaml` | MODIFY | Add new config blocks under `ball:` |
| `tests/test_ball_plausibility.py` | NEW | Layer 1 + Layer 2 unit tests |
| `tests/test_ball_kick_anchor.py` | NEW | Layer 3 unit tests |
| `tests/test_ball_appearance_bridge.py` | NEW | Layer 4 unit tests |
| `tests/test_ball.py` | MODIFY | Integration scenario: synthetic detections that reproduce the diagonal-switch case |

## Testing strategy

**Unit, Layer 1**
- Plausible parabola (apex 10 m, 25 m/s) → True.
- p0 off-pitch by 1 m, margin 5 m → True.
- p0 off-pitch by 100 m → False.
- 250 m/s velocity → False.
- z = 10⁶ m → False.
- Empty/zero-duration segment → False (defensive).

**Unit, Layer 2**
- All-on-pitch grounded run, speed 5 m/s → not promoted.
- Grounded run with ground-projection at y = 50 m → promoted, refit
  attempted.
- Run where refit fails plausibility → frames set to `state="missing"`.

**Unit, Layer 3**
- `bundle_adjust.fit_parabola_to_image_observations(..., p0_fixed=p)`
  returns `p0 == p` and reduces residual on a synthetic kick scenario
  versus the unanchored fit.
- `find_kick_anchor`: foot within 20 px and pixel acceleration above
  threshold → anchor returned; foot 100 px away → None.

**Unit, Layer 4**
- White circle on green background, no detector hit, template from prior
  frame → bridge returns peak within 1 px of true center.
- Background with no ball-like patch → bridge returns None
  (NCC below threshold).
- Stale template (age > `template_max_age_frames`) → bridge returns None.

**Integration**
- `FakeBallDetector` produces a pattern that mirrors `origi01`:
  - Frames 0–60: solid grounded detections.
  - Frames 60–180: detections trace an aerial arc in pixel space (so the
    IMM will not classify flight on pixel kinematics alone).
  - Frames 180–250: detections become None (the messy stretch).
  - Frames 250–280: detections trace a high-arc final shot.
- With minimal canned camera and refined_poses fixtures, assert:
  - No `BallFrame.world_xyz.z > z_max_m` and none > 5 m off-pitch.
  - At least one flight segment in the 60–180 range with apex z ≥ 3 m.
  - No detection-hole frames in 180–188 emit `state="missing"`
    (appearance bridge takes 0–8 frames).
  - All `FlightSegment.parabola.p0` within plausible pitch bounds.

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Appearance bridge locks onto a referee's sleeve or sideline marker | NCC threshold 0.60, 8-frame cap, discount confidence 0.5×, IMM gating still applies |
| refined_poses output absent on first run | Layer 3 disabled with warning; Layers 1, 2, 4 still active |
| Layer 2 over-promotion creates new garbage flights | Layer 1 plausibility gates every promoted refit; failure → `state="missing"` not `state="flight"` |
| `p0_fixed` makes Magnus fit ill-conditioned (3-DOF v0 + 3-DOF ω is rank-deficient with short segments) | Anchored Magnus requires the existing `spin.min_flight_seconds` (≥ 0.5 s = 15 frames); shorter segments stay parabola-only |
| Pipeline order change (ball after refined_poses) breaks `--from-stage` / `--stages` invocations | Ball stage continues to declare only camera + shots as hard dependencies; refined_poses is read opportunistically |

## Rollout

1. Land Layer 1. Re-run `recon.py run --from-stage ball --input ... ` on
   `origi01`. Garbage flight segments disappear from `quality_report.json`
   and the dashboard. Lowest-risk change, immediate visual improvement.
2. Land Layer 2. The diagonal switch should now appear as a flight
   segment (or honest `missing` if the refit fails plausibility).
3. Land Layer 3. The diagonal switch's flight fit should now have a
   physically credible apex anchored to the player's foot.
4. Land Layer 4. The 191–442 stretch's detection holes shorten; very
   long gaps still surface as `missing`.

Each layer is committed and validated separately on `origi01` before the
next is enabled.

## Open questions

- None blocking. The user accepted refined_poses as the foot source, the
  conservative 8-frame bridge cap, and the strict off-pitch threshold in
  brainstorming.
