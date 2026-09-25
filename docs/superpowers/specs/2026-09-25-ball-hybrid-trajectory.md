# Ball hybrid trajectory layer

Status: integration complete on `ball-hybrid-integration`, config-gated OFF
by default (`ball.trajectory: reference`) pending the T6 4-clip benchmark.

## Goal

The reference ball solver (`src/utils/ball_piecewise_solver.py` + friends,
driven from `src/stages/ball.py`) is tuned for **touch-recall and anchor
accuracy**, not for what a Blender render exposes: a virtual side/drone
camera shows depth error, ground float/sink, and non-physical velocity
kinks the broadcast camera's own viewing angle hides. The hybrid
trajectory layer's goal is **visual realism** for `render`/`export`
consumers — an aspiration of ≤20 cm 3-D error on as many frames as
possible, judged from an arbitrary virtual camera angle, not just the
broadcast one. It is an additive alternative dense-track builder, not a
replacement for the event/anchor layer: touch attribution, keyframe
building and export schemas are byte-identical regardless of which
trajectory mode produced the dense per-frame track.

## Architecture

```
event layer (unchanged)              gated knots                per-span solve
─────────────────────────       ─────────────────────      ───────────────────────
manual anchors        ─────┐
                            ├──►  hard knots (depth_hard)  ┐
cross-replay fixes    ─────┘     [ball_replay_knots.py:    │
  (operator-wins +               physical-volume +          │
   physical-volume gated)        operator-wins gates]        │
                                                               ├─► per-span fit
auto events (touches, ─────►  gate_auto_events()            │    (roll OR flight,
  bounces, goal impacts,        [ball_hybrid_gating.py:      │     whichever fits
  same BallAnchorSet             kind whitelist, confidence   │     better) + dual-
  schema as manual)              floor (or cue-relaxed),      │     fit robust
                                 evidence consistency,         │     gating + optional
event cues (audio/net/          residual-improvement,          │    Hermite join at
  blur, opt-in) ─────────►      physical-plausibility)         │    non-event knots
  [relax confidence                                            │
   floor only — never                                          ├─► ray-faithful C1
   mint a knot]                                                 │   delta blend
                                                                 │   (pulls onto a
                                                                 │   confident
                                                                 │   detection's ray
                                                                 │   near evidence,
                                                                 │   decays smoothly)
                                                                 │
                                                                 └─► optional bounded
                                                                     2-dof Magnus
                                                                     spin refinement
                                                                     per flight span
```

Modules (`src/utils/ball_hybrid_*.py`, all ported from the
`prototypes/ball_hybrid_poc` spike with an explicit "no behavioural
change, tests cross-check bit-for-bit" contract for the pure-math ones):

- **`ball_hybrid_types.py`** — shared dataclasses. `Knot` is the central
  type: `depth_hard=True` means the full 3-D position is authoritative
  (ground/touch/bounce/net/goal/catch/fix — resolved via ground plane,
  joint-ray intersection, or goal geometry); `depth_hard=False` is a "ray
  knot" — only the pixel ray is authoritative, depth is left to the
  physics fit (`airborne_low/mid/high`, off-screen flight, or an event
  anchor that failed 3-D resolution). `HybridShotCtx` carries the
  per-frame camera (built in-memory from what `ball.py`'s `_solve_shot`
  already has — no file IO). `CueEvidence` and `SpinFit` are the IC-D/IC-E
  hand-off shapes.
- **`ball_hybrid_physics.py`** — pure physics: a size-11 ball under
  gravity + quadratic drag + optional bounded Magnus, RK4-integrated.
  `shoot_arc` solves the two-point boundary-value problem (launch
  velocity that sends the ball from knot A to knot B under the model) so
  a fitted arc is always endpoint-exact — physical realism is never
  traded against hitting both hard knots.
- **`ball_hybrid_blend.py`** — the delta-blend smoothing kernel: a
  Gaussian-family (not Laplace/exponential — no derivative corner at an
  evidence frame) weighted average of `faithful_point - P_phys` at each
  confident-detection frame, decaying to zero within a few halflives of
  the nearest evidence, never smoothed across a hard event frame. A
  second independent safety net (`clamp_delta_rate`) rate-limits delta's
  own frame-to-frame change even where evidence is dense/noisy.
- **`ball_hybrid_trajectory.py`** — the mechanics: `resolve_knots` (turns
  anchors/fixes into hard/ray knots), `solve_span` (per-span roll-vs-
  flight fit with robust gating and split-and-retry), `finalize_track`
  (Hermite smoothing at non-event knots → delta blend → exact re-snap at
  sharp knots), and `run_trajectory` — the single entry point `ball.py`
  and bench harnesses call, wrapping resolve → gate → finalize.
- **`ball_hybrid_gating.py`** — the auto-event acceptance policy (kind
  whitelist, confidence floor, evidence consistency, residual
  improvement, physical plausibility — see "Integration bugs found"
  below for gate 5's origin). Never mutates or overwrites the manual
  knot list, only proposes additions; nothing within
  `min_frame_gap_from_manual` frames of a manual knot is ever considered.
- **`ball_hybrid_spin.py`** — bounded 2-dof monocular Magnus fit per
  resolved flight span (see "Spin results" below).
- **`ball_replay_knots.py`** / **`ball_replay_review.py`** — cross-replay
  fixes → depth-hard knots (physical-volume + operator-wins gated) and
  review proposals for replay groups whose partner geometry left a shot
  with zero usable fixes.
- **`ball_cue_audio.py`** / **`ball_cue_net.py`** / **`ball_cue_blur.py`**
  / **`ball_cue_fusion.py`** / **`ball_cue_config.py`** — single-camera
  event-cue corroboration (see "Event-cue results" below).
- **`ball_bench_*.py`** (`clip`, `metrics`, `regression`, `runner`,
  `synth`, `truth`, `types`) — the promoted regression/benchmark harness
  (see "Benchmark methodology").

**Wiring** (`src/stages/ball.py`): `_run_hybrid_trajectory` builds a
`HybridShotCtx` from `_solve_shot`'s own per-frame `K`/`R`/`t`/distortion
(no file IO), filters `steps`/`sources` to real detector evidence sources
(`detector`, `second_pass`, `foot_guided`, `strike_window` — matching
`scripts/eval_ball_accuracy.py`'s `_DENSE_EVAL_SOURCES`; `_detect_loop`
writes the literal per-frame detector pixel into `TrackerStep.uv`, not
the IMM's smoothed estimate, so this is genuinely raw evidence), and
delegates to `run_trajectory` with `manual_by_frame`, `auto_by_frame` and
the cross-replay `fixes` map. It runs right after
`world_by_frame`/`state_by_frame` are computed from the reference solve
and **before** the C4 ray-faithful block, replacing only the frames the
hybrid layer covers — additive relative to the reference solve, so a
frame the hybrid layer doesn't reach keeps its reference value and frame
coverage can never regress. The whole call is wrapped in try/except:
hybrid is opt-in enrichment and a failure falls back silently to the
reference solve, recording the error in the diag sidecar.
`touch_attribution` / keyframe building / export schemas are unchanged
either way — the hybrid layer only replaces the dense per-frame track the
rest of the stage already consumed. `FlightSegment`s for
`ball_orientation.integrate_orientation` and `BallTrack.flight_segments`
are built from hybrid's own flight-span diagnostics (`p0`/`v0`/`g`, plus
`omega_world`/`rad_s` when spin was accepted) rather than reused from the
reference solve, so hybrid+spin flows through consistently.

## Config switches and defaults

`ball.trajectory: reference|hybrid` (default `reference`) — opt-in, zero
behaviour change for existing runs. `ball.hybrid.*` mirrors
`ball_hybrid_trajectory.DEFAULT_CFG` / `ball_hybrid_gating.DEFAULT_GATING_CFG`
so a bare `{}` override still gets sane code defaults.

Top-level `ball.hybrid` knobs: `cd`/`fit_cd`/`cd_bounds` (drag
coefficient, fitted when ≥5 detections else 0.25 default), `magnus`
(legacy unbounded Magnus flag — superseded by `spin.*` below, left off),
`blend_halflife_frames` (delta-blend decay, 5 frames), `faithful_conf_min`
(0.5 — confidence floor for a detection to pull the ray-faithful blend),
`inlier_px`/`robust_gate_max_iters`/`split_residual_factor`/
`max_splits_per_span` (per-span robust-fit gating), `roll_mu_max` (0.9),
`grounded_ray_weight` / `airborne_ray_weight` (20.0 / 2.0 — see the
origi01 fix below), `hermite_k_*` (non-event knot join stiffness),
`blend_max_delta_step_*` (delta-blend rate limiter).

- **`ball.hybrid.gating.*`** (`ball_hybrid_gating.py`) — the auto-event
  acceptance policy: `confidence_floor` (0.5) /
  `corroborated_confidence_floor` (0.3, applies only when ≥1 `CueEvidence`
  agrees), `corroboration_window_frames` (3), `consistency_window_frames`
  (5) / `consistency_max_px` (20.0), `residual_improve_frac` (0.0 — "must
  not make the bracketing span's fit worse", the PoC-validated gate;
  positive opts into a stricter "must demonstrably improve" policy),
  `min_frame_gap_from_manual` (2). Governs additions on top of manual
  anchors only — manual anchors are never touched.
- **`ball.hybrid.spin.*`** (`ball_hybrid_spin.py`, default
  `enabled: false`) — bounded 2-dof Magnus refinement per resolved flight
  span; `bounds` ±62.8319 rad/s (10 rev/s), `min_obs: 8`,
  `min_delta_bic: 6.0`, `min_resid_gain: 0.10`. Validated (0/50 false
  accepts across 4 clips; mismatch airborne ≤20cm 0.605→0.679 on
  kroupi01) but default OFF pending the T6 benchmark decision — cost is
  ~0.3-1.5 s per flight span.
- **`ball.hybrid.cues.*`** (`ball_cue_*.py`, default `enabled: false`) —
  single-camera event-cue corroboration (audio onset, net motion energy,
  motion-blur direction change) fused into `CueEvidence`, which **only**
  relaxes the auto-event gate's `confidence_floor` — a cue alone never
  mints a knot. Default OFF: the 2026-09-25 anti-overfit tuning round
  found no fusion policy beat auto-only by ≥0.05 held-out F1 in both
  cross-validation folds. Sub-blocks: `audio.*` (band-limited spectral
  flux + per-clip latency calibration), `blur.*` (angle-change /
  streak-length / speed-ratio thresholds — cross-fold **validated**),
  `fusion.*` (`policy: any2` default — the least-overfit of the three
  policies tried; `net_blur_combo` and `weighted` are opt-in
  alternatives).
- **`ball.hybrid.fixes.*`** (`ball_replay_knots.py`, default
  `enabled: true`) — cross-replay fixes → depth-hard knots, gated by
  physical volume + operator-wins; `tol_px`/`adjacent_frames`/`weight`
  tunable. A dropped fix is recorded in the diag sidecar
  (`hybrid_trajectory.fixes_dropped`) with its reason, never silently
  discarded.

## Benchmark methodology

`src/utils/ball_bench_*.py` + `scripts/run_ball_bench.py` (promoted from
the `prototypes/ball_hybrid_poc` spike, `tests/test_ball_regression.py`
as the committed regression gate).

- **Independent synthetic truth**, seeded ONLY from operator data
  (manual anchors + reconstructed players) — never from ball-stage
  output (`*_ball_track.json`/`*_ball_anchors_auto.json`/
  `*_ball_keyframes.json` are the thing under test and must not leak
  into truth). `ball_bench_truth.py`'s physics simulator is a
  **separate, non-shared implementation** from the pipeline's own
  solvers — `tests/test_ball_bench_truth.py` greps its imports to
  enforce this — so grading is never against a truth model that shares
  code (or, in `mismatch`/`sparse`, shares parameters) with the thing
  being measured.
- **Scenarios**: `base` (truth uses the solver's own drag/physics
  assumptions — an easy, self-consistent check), `mismatch` (truth uses
  randomised drag/drag-crisis/spin the solvers don't share — the
  realistic-noise scenario used by the regression gate), `sparse` (fewer
  seed anchors), `hidden` (only fold-0 half of operator anchors exposed
  to the method under test — events whose anchor was withheld must be
  discovered from evidence, the situation real footage is always in).
- **Real 2-fold held-out**: the real detector, real footage, with half
  the manual anchors withheld and graded against the withheld half.
- **origi01 replay-fix hold-out**: origi01's 31 cross-replay
  triangulated fixes (real absolute 3-D ground truth, sub-20cm campaign
  finding) are split in half — one half feeds the hybrid as fixed
  points, grading always uses the other half.
- Clips: gberch (`$M/output`), origi01 (`$M/output-origi-global`),
  kroupi01 (`$M/output-kroupi`), s013 (`$M/output-japan`).
- Regression gate: `pytest tests/test_ball_regression.py -m regression`
  re-runs the real ball stage against a frozen `mismatch`-scenario
  synthetic truth/evidence AND the real detector's 2-fold held-out
  anchor eval for each golden clip under `tests/regression/ball/<clip_id>/`,
  failing if accuracy regresses beyond the baseline's measured-spread
  tolerances. Capture/re-baseline via
  `scripts/capture_ball_regression_baseline.py --clip <id> --runs 3`.

## Spike results (`prototypes/ball_hybrid_poc`, throwaway-labelled)

Full write-up: `prototypes/ball_hybrid_poc/viewer/findings.html`.

- **Synthetic, all anchors given ("mismatch")**: hybrid beats the
  current/reference stage on every clip on ≤20cm-frames and p95 depth
  error (e.g. gberch 70%→88%, kroupi01 40%→77%, s013 22%→86%; p95 error
  drops by 1-2 orders of magnitude on gberch/kroupi01/origi01). Ground
  float/sink also improves or ties everywhere.
- **Half anchors withheld ("hidden")**: hybrid+auto-events beats the
  reference everywhere but falls well short of 20cm on most frames
  (gberch 49%→61%, origi01 23%→32%, kroupi01 16%→41%, s013 10%→17%). Main
  spike finding: **missing events, not the solver, are the accuracy
  ceiling** from a single camera.
- **Real footage (held-out manual anchors)**: median 3-D error,
  reference→hybrid+auto: gberch 0.158→0.114 m, s013 0.615→0.227 m,
  kroupi01 2.97→1.78 m — but **origi01 regressed** (0.36→1.70 m): its
  flight-heavy spans have many mid-air anchors that fix only the ray, not
  the depth — the root cause fixed in production by the airborne/grounded
  ray-weight split (see next section). On origi01's replay-triangulated
  positions (real absolute truth), error dropped 4.16→0.55 m — but that
  measures the value of replay angles, not the solver alone (the other
  half of those same fixes fed the fit). The anchors-only hybrid that
  wins the synthetic table is not a clean win on real footage — it needs
  the event layer.
- **Air drag**: helps consistently when the fitted Cd matches truth
  (kroupi01 78%→99% ≤20cm; origi01 p95 halves); mixed under
  drag/spin mismatch — a real gain needs spin fitting too (built in
  production as `ball_hybrid_spin.py`, see below).
- **Caveats** (spike-level, some resolved in production): synthetic runs
  handicap the reference stage by disabling its foot-guided-zoom/
  appearance-bridge re-detection passes; auto events helped real footage
  but hurt s013's synthetic runs at spike time (production's
  `ball_hybrid_gating.py` adds the stricter multi-gate policy this
  motivated); four clips, one seed per scenario — spike numbers, not a
  regression gate (the promoted `ball_bench_*` harness is the successor).

## Integration bugs found (production, 2026-09-25)

Found via an origi01-fold0 per-anchor error decomposition (reference vs
hybrid, sorted by delta) requested mid-integration — both confirmed via a
direct-call reconstruction against the real clip, not just unit tests
(`a8ea91a`):

1. **Gate read the wrong confidence field → 0 auto knots accepted.**
   `ball_hybrid_gating._candidate_conf` read `.score`/`.conf` (matching
   its own PoC-derived test fixture), but a real `auto_anchors` candidate
   is `src.schemas.ball_anchor.BallAnchor`, which carries its detector
   score as `.confidence`. Every real candidate was silently read as
   confidence 0.0, failing the 0.5 floor regardless of its true score
   (0.65-0.92 in real data). On origi01 fold0: 0/54 auto-anchor
   candidates accepted before the fix, 11/54 after — the hybrid
   trajectory had been built with **zero** real touch/bounce knots, only
   manual anchors + cross-replay fixes. Fixed by checking `.confidence`
   first, `.score`/`.conf` retained as fallbacks for duck-typed fixtures.
2. **Implausible-speed flight spans laundered by the roll fallback.** The
   assumption that a flight span's boundary-value `shoot_arc` solve
   "can't run away" (both ends are hard 3-D knots) breaks when either
   knot is a bad auto-generated candidate: after fix 1 started accepting
   real auto knots, several `player_touch` knots only 1-3 frames apart
   (bad SMPL-FK bone/player attribution) implied launch speeds of
   hundreds of m/s. `solve_span`'s flight branch now falls back to the
   roll model (reusing `MAX_LAUNCH_SPEED_M_S`, the same bound the
   free-end fit already enforces) whenever fitted launch speed exceeds
   it, tagging `info["fallback_from_flight"]=True`; gating's new gate 5
   ("physical plausibility") treats that fallback as an outright reject
   — a roll model can always find some low-residual straight-ish line
   between two points regardless of physical correctness, which would
   otherwise silently launder a candidate the residual-improvement gate
   alone can't catch (reprojection is only evaluated at evidence points,
   often sparse/absent between two closely-spaced touches).
3. **Airborne ray weight vs. depth (the origi01 held-out fix, T1c).**
   Root cause of the spike's origi01 real-footage regression: inside a
   flight span, ray-only (`depth_hard=False`) evidence — depth-ambiguous
   by construction — was weighted as heavily as a manual click
   (`anchor_fit_weight=20`) in the span's `shoot_arc` Cd fit and its
   robust-gating residual. Cd sets the arc's curvature along its entire
   length, so a single noisy ray anchor could warp mid-span **depth**
   substantially while 2-D reprojection (the only thing minimised) stayed
   near-perfect — reprojection error cannot see a depth-only error, since
   a ray anchor's "faithful" point sits on its ray at any depth. Fix:
   ray evidence inside a roll span is first projected onto the span's own
   known ground level (externally supplying depth — keeps the full
   `grounded_ray_weight`, 20.0), while ray evidence inside a flight span
   gets the much smaller `airborne_ray_weight` (2.0, comparable to one
   solid detector observation) — it can still nudge lateral shape and
   participate in outlier gating, but can no longer dictate the arc's
   depth.

Investigated but **not** fixed (documented dead-end, `a8ea91a`): some
origi01 fold0 held-out frames (e.g. frame 287, bracketed by a roll span
whose knots are internal-split points, not original anchors) still show
large hybrid errors after both fixes above. A hypothesis (forcing an
airborne evidence point onto the ground plane during a re-split) was
written, confirmed via bench diagnostics to never actually fire, and
reverted rather than shipped as dead code. Most likely mechanism:
instability in the flight branch's own split-and-retry producing an
internal "bounce" knot when the parent arc's Cd/v0 fit is itself
unstable (some origi01 spans show 200-800px residuals even before any
knot changes) — not investigated further given time budget; see "Open
items".

## Event-cue results

2-fold anti-overfit protocol (`scripts/tune_ball_event_cues.py`): tune on
{gberch, origi01} / held out on {kroupi01, s013}, and the reverse — every
threshold/policy pick scored on **both** held-out folds so an
overfit choice can't hide behind a single split.

- **Blur thresholds — validated.** Both fold directions independently
  converged on the same stricter thresholds (`angle_change_deg=50`,
  `min_streak_px=10`, `speed_ratio_threshold=2.0`), with raw-blur pooled
  F1 staying in a similar 0.29-0.34 band on both the tuning fold and its
  own held-out fold — a genuine, cross-validated improvement over the
  pre-tuning defaults. Adopted as the shipped default.
- **Weighted fusion — an overfitting trap.** The `weighted` policy won
  tuning-fold F1 in **both** fold directions, but was the worst or
  near-worst policy on **both** held-out folds (fold A held-out F1 0.31
  vs `any2`'s 0.36 and `net_blur_combo`'s 0.40; fold B held-out F1 0.20
  vs `any2`'s 0.33 and `net_blur_combo`'s 0.28). Picking a fusion policy
  by tuning-fold F1 alone would have shipped the worst-generalizing
  option — the clearest overfitting signal in the whole round.
- **Audio — inconclusive.** The tuning-fold-winning strict setting
  (`k_mad=4.0`, `min_rise_ratio=0.6`, pooled F1 0.47 on its own fold)
  collapsed to F1=0.00 held-out (zero raw onsets survived calibration —
  the stricter threshold starved an already-marginal clip's onset count
  below the ≥4-matched-anchors calibration floor). The reverse fold
  couldn't cross-check meaningfully: kroupi01 has only 4 total contact
  anchors (borderline for calibration under any setting) and s013 is
  retimed (audio structurally disabled). `DEFAULT_CUE_CFG` keeps
  band-limiting enabled (physically motivated on its own) at the milder
  of the two candidates tried (`k_mad=3.0`, `min_rise_ratio=0.4`) as a
  conservative, **not validated**, middle ground.
- **Net cue** — implemented (`ball_cue_net.py`, camera-compensated
  frame-difference spike inside the projected net volume,
  back-projected onto the net's 3D plane) but not included in the tuning
  round's scored fusion path in the same way as audio/blur; see "Open
  items" for its stage-wiring status.
- **Net/blur not yet wired into the stage.** `ball.hybrid.cues.enabled`
  is wired end-to-end for **audio only** as of T5 (`_hybrid_cue_
  corroboration` in `ball.py`) — net and blur both need a shared
  frame-decode pass with `_detect_loop`, which doesn't exist yet; adding
  a second independent decode path was explicitly out of scope for this
  integration round. `cues` stays opt-in (`enabled: false`) regardless:
  no fusion policy beat auto-only by ≥0.05 held-out F1 in either fold.

## Spin results

`ball_hybrid_spin.fit_span_spin` (IC-E) fits a **bounded 2-dof** Magnus
term per resolved flight span rather than a free 3-vector: an earlier
cross-replay-triangulation investigation found an unconstrained 3-dof (or
9-dof rigid-body) fit from a short monocular arc is ill-conditioned and
diverges (one run hit 512 km/s) — monocular depth and spin are jointly
degenerate, and a free-standing omega has a rifle-spin component along
the velocity direction that is completely unobservable (`omega × v == 0`
when parallel), giving the optimiser a direction to run away in for no
benefit. The 2 dof span exactly the two modes that *do* produce an
observable force (topspin/backspin, sidespin), bounded to ≤10 rev/s, with
both knots always staying exact (every trial omega re-solves launch
velocity via `shoot_arc`, spin is never traded against hitting the
knots). Accepted only when it clears both a BIC bar (`min_delta_bic`,
Kass & Raftery's "strong evidence" convention, default 6.0) and a minimum
fractional reprojection-RSS improvement (`min_resid_gain`, default 10%)
over the no-spin fit. Validated: 0/50 false accepts across 4 clips;
mismatch airborne ≤20cm accuracy improved 0.605→0.679 on kroupi01. Wired
end-to-end (T5, `013fa42`): `solve_span`'s flight branch records
`p0`/`v0`/`g` in its diagnostics, and `ball.py`'s `_hybrid_flight_segments`
builds real `FlightSegment`s (`spin_axis_world`/`spin_omega_rad_s`) so
`ball_orientation.integrate_orientation` picks up hybrid-fitted spin.
Default OFF (`ball.hybrid.spin.enabled: false`) pending the T6 benchmark
— cost is ~0.3-1.5s per flight span, and gating's own residual probe
forces spin off regardless of the caller's config (a cheap accept/reject
probe should never pay the spin-fit cost).

## Replay review proposals

Sub-20cm campaign finding (W6): s013's replay group g02 had 6/6 partner
cameras that were globally wrong — 3-6px per-frame reprojection residual,
but fixes landing 6-8m underground (error purely along the reference ray,
invisible to any lateral/pixel gate). `ball_cross_replay.py`'s pair-level
gates correctly reject that geometry, but the stage's diagnostics
previously went silent: a shot with replay partners in its group just
quietly got zero fixes, indistinguishable from "no partner had
overlapping evidence" — costing a ~30-minute manual anchor-editor
investigation to even find the cause (2026-08-20 operator handoff doc).
`ball_replay_review.py` turns "every partner gated out" into an
actionable diag/quality_report entry instead (no new gates — it only
reads metadata `ball.py`'s `_triangulate_groups` already assembles).
Wired T5 (`013fa42`): the previously-silent empty-fixes branch now
records `a_partners` metadata (with `"n_fixes": 0` when geometry passed
but every individual fix was filtered out, vs. `"rejected":
"implausible_geometry"` when the pair geometry itself failed), the
`productive` filter now correctly requires `n_inlier_fixes > 0`, and a
new `_record_nonproductive_b_side` helper gives the **partner** shot its
own review entry too (previously only the reference/A side ever got one).
Proposals land in each shot's `ball_diag.json` under
`cross_replay.review`, and surface in `quality_report.json`
automatically (`src/pipeline/quality_report.py`'s `_ball_shot_entry`
already copies the `cross_replay` block verbatim).

## Open items

- **origi01 flight split-and-retry bad bounce knots** — the remaining
  post-fix origi01 held-out regression (frame 287 and similar), most
  likely in how the flight branch's own split-and-retry mints an
  internal "bounce" knot when the parent arc's Cd/v0 fit is itself
  unstable (some spans show 200-800px residuals even before any knot
  changes). Not root-caused; see "Integration bugs found" for the
  dead-end already ruled out.
- **Audio noise-suppression re-tune** — `DEFAULT_CUE_CFG`'s audio
  settings are a conservative, unvalidated middle ground; re-tune
  specifically once more audio-eligible clips (real-time playback, ≥4
  contact anchors) are available for a meaningful held-out check.
- **Net/blur stage wiring** — needs a shared frame-decode pass with
  `_detect_loop` before those two cues can be wired into
  `_hybrid_cue_corroboration` alongside audio; not built in this
  integration round to avoid a second large, unvalidated decode-path
  addition.

## Default decision (T6)

**Decision: `ball.trajectory` stays `reference`; `hybrid` ships opt-in.** The flip rule
was "synthetic improves on all four clips AND real held-out p50 is no worse than
`reference` on any clip". Synthetic passed everywhere; real held-out did not.

Final code (commits through `a8ea91a`), real 2-fold held-out anchors pooled so each
anchor is graded once, cached WASB detections:

| clip | real held-out p50, reference → hybrid | p95 | synthetic `mismatch` %≤20 cm | synthetic `hidden` %≤20 cm |
|---|---|---|---|---|
| gberch (58) | 0.158 → **0.105 m** | 1.54 → 2.38 m | 0.70 → 0.75 | 0.49 → 0.61 |
| s013 (13) | 0.615 → **0.178 m** | 2.21 → 2.09 m | 0.22 → 0.60 | 0.10 → 0.29 |
| origi01 (59) | 0.361 → 0.559 m | 4.54 → 8.33 m | 0.58 → 0.76 | 0.23 → 0.33 |
| kroupi01 (12) | 2.97 → 3.54 m | 5.59 → 10.6 m | 0.40 → 0.77 | 0.16 → 0.36 |

Reference synthetic numbers for gberch/origi01 `mismatch`/`hidden` come from the same
harness (spike + regression-baseline captures); `hybrid` from `t6_*` / `ica_postfix` runs.

The hybrid wins every synthetic comparison and the real footage on gberch and s013. It
regresses on origi01 and kroupi01 — the flight-heavy, sparse-anchor clips. On origi01
the auto-event gate is not the cause (persisted diag, fold 0: 54 candidates, 2 accepted,
23 rejected on evidence consistency with the fold's fresh detections); the open lead is
how `reference` treats touch-dense spans compared with the hybrid's flight split-and-retry
(interior "bounce" knots on spans with 200-800 px residuals). Use `hybrid` per clip where
the anchors are dense and the play is ground-heavy; revisit the default after that fix.
