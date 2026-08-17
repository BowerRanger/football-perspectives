# Ball Sub-20 cm Accuracy Campaign — Design

- **Date:** 2026-08-17
- **Status:** Active (autonomous goal session — written for async review; sections
  land incrementally, each gated by the eval harness)
- **Goal (verbatim):** iterate on the ball tracking stage until the ball's 3-D
  location error is **< 20 cm at any given time**, with natural motion (no
  direction change without an intermediary player touch, ground touch,
  crossbar/post touch, or goal-net touch), different treatment for airborne vs
  ground balls, and **touch moments held to the tightest accuracy**.
- **Builds on:** [`2026-05-31-ball-anchor-10cm-accuracy-design.md`](2026-05-31-ball-anchor-10cm-accuracy-design.md)
  (anchored-frame lateral accuracy), [`2026-06-15-ball-touch-events-design.md`](2026-06-15-ball-touch-events-design.md)
  (events mode, the current default), [`2026-06-12-cross-replay-triangulation`](../plans/2026-06-12-cross-replay-triangulation.md).

---

## 1. Problem: where the error lives today

The default back-end (`ball.solver: events`) resolves sparse keyframes and
renders the dense track with the pure reference interpolator
(`ball_interpolate.py`). Measured against the shipped gberch/origi outputs:

1. **Airborne keyframes are height-bucket placeholders.** `_resolve_waypoint_world`
   resolves `airborne_low/mid/high` anchors via `state_to_height` → ray ∩ a fixed
   plane (z = 1.0 / 6.0 / 15.0 m). Every airborne keyframe on gberch sits at
   exactly z = 1.00 m. The 2026-05-31 physics result ("≥ 2 hard knots + gravity
   fully determine the arc, depth included") lives only in the piecewise/global
   solvers, which events mode never calls. Airborne depth error: metre-scale.
2. **Touch keyframes can resolve below the pitch.** Body-pinning uses the SMPL
   joint world position (plus a ball-radius offset toward the camera when a ball
   pixel exists). Foot joints dip under the ground plane (FK/foot-anchor
   residual), so gberch touch keyframes sit at z = −0.08…−0.01 m — physically
   impossible (ball centre ≥ 0.11 m) and 12–30 cm wrong at exactly the moments
   the goal prioritises.
3. **Between keyframes the renderer ignores all evidence.** `roll`, `carry`
   (and, by bug, `free_flight` with two known endpoints) render **constant-speed
   straight lines**; `ballistic` renders a plain two-endpoint parabola (no spin
   curvature). WASB detections, cross-replay 3-D fixes, and z-bucket hints — all
   of which the piecewise solver consumes — never reach `resolve_events`.
   Segment spans reach 25–80 frames on origi/kroupi, so mid-span error is
   decimetre-to-metre class (a decelerating roll deviates from constant-speed
   linear by (Δv·T)/8; a 2 s carry rendered linearly ignores the dribbler's
   curved path entirely).
4. **No measurement of any of this exists.** `tests/test_ball_anchor_accuracy.py`
   grades *anchored frames only*, *lateral only*. Nothing grades between-anchor
   frames, depth, or motion naturalness.

## 2. Ground truth inventory (what we can grade against)

| Source | Clips | Nature | Use |
|---|---|---|---|
| Manual ball anchors (59 gberch / 60 origi01 / 12 kroupi01 / 14 s013) | 4 clips | clicked pixel ray + state; ground states pin full 3-D (ray ∩ z=r); `player_touch` adds joint depth | **Hold-out**: withhold a subset, grade the auto track at withheld frames |
| Cross-replay 3-D fixes (`origi01_ball_fixes.json`: 31 inlier fixes, median ray-miss 0.20 m, parallax 27°) | origi01 (+partners) | triangulated world XYZ incl. airborne | independent full-3-D check, esp. airborne depth |
| Goal geometry / pitch landmarks | all | exact 3-D lines/planes | impact-frame checks |
| WASB detections (fine-tuned v1, weights local) | all (rerun locally) | pixel rays per frame | dense lateral error proxy on every detected frame |
| Physics itself | all | gravity, friction, restitution envelopes | naturalness validator (direction changes only at events) |

Manual anchors are operator input ("operator wins"), so with all anchors fed in,
anchored frames are pinned by construction. **"Any given time" is therefore
graded two ways:** (a) *hold-out* — run the stage with a subset of manual
anchors and measure at the withheld frames; (b) *full-run* — measure between
anchors against detections/fixes/physics.

## 3. Acceptance criteria (operationalising "< 20 cm at any given time")

Graded on the emitted dense `ball_track.json` for gberch, origi01, kroupi01,
s013 (origi02 auto-only as stretch). All measured by the new harness (§4); the
campaign is done when, on every clip:

- **A1 — Hold-out accuracy.** With alternating ~50 % of manual anchors withheld:
  at every withheld anchor frame, 3-D error < 0.20 m where full 3-D GT exists
  (ground-level states via ray ∩ z=r; `player_touch` via clicked ray + joint
  depth; `goal_impact` via goal geometry), and lateral ray distance < 0.20 m for
  airborne ray-only anchors.
- **A2 — Independent fixes.** Full run (all anchors): 3-D error at each
  cross-replay fix < max(0.20 m, that fix's own `ray_miss_m`).
- **A3 — Dense lateral.** Full run with the real detector: perpendicular
  distance from the emitted position to the cleaned detection ray, p95 < 0.20 m
  over all confidently-detected frames (outlier-rejected via the existing track
  cleaner).
- **A4 — Naturalness.** Zero violations: no horizontal direction change
  > 12°/frame at speed > 2 m/s outside ±2 frames of an event keyframe
  (touch/bounce/goal/rest boundary); flight-frame vertical acceleration within
  ±25 % of g; roll speed non-increasing beyond 15 % tolerance between events.
- **A5 — Touch physicality.** Every touch keyframe: z ≥ ball_radius − 0.02 m,
  ≤ 2.6 m (reachable), and ball-to-attributed-joint gap ≤ `contact_max_gap_m`.

A1/A2 are the honest "< 20 cm" gates; A3 bounds the frames without 3-D GT; A4/A5
encode the user's naturalness and touch-priority requirements. Baseline is
recorded first; gates ratchet from baseline to target so every change is
provably monotone.

## 4. W1 — Evaluation harness (build first, gate everything)

New `src/utils/ball_eval.py` (pure, numpy-only):
- ray/error primitives reused from the 10 cm harness (perp distance, along-ray
  depth, reprojection);
- `evaluate_track(track, camera, gt_bundle) -> BallEvalReport` computing
  A1–A5 metric tables (per-frame rows + per-clip summaries);
- `naturalness_violations(track, keyframe_set|events, fps, cfg)`;
- hold-out splitting (`split_anchors(anchors, k=2, offset)` — deterministic
  alternating split, stratified so every state class appears in both halves
  where counts allow).

New `scripts/eval_ball_accuracy.py` CLI:
- `--output <dir> --shot <id> [--holdout] [--detector noop|wasb] [--json out]` —
  reruns the ball stage in-process (no-op detector by default, WASB for A3),
  writes per-frame CSV + summary JSON + a markdown table; never clobbers real
  outputs (temp overlay dir with symlinked camera/refined_poses/shots and a
  filtered anchors file for hold-out runs — zero changes to the stage API).

Tests `tests/test_ball_eval.py` (unit, light venv): metric math on synthetic
tracks/cameras; split determinism/stratification; naturalness detector on
synthetic clean + violating tracks.

## 5. W2 — Touch contact geometry (priority moments)

In `ball_event_resolver.py`:
- **Ground clamp:** after body-pin (+ray refine + radius offset), clamp
  z ≥ ball_radius when the resolved z falls below it — implemented as a lift
  along the vertical (preserving the lateral ray solution's XY where a pixel
  exists, else raw joint XY). Ground-level touches (foot near ground, ball pixel
  present) reconcile with ray ∩ z=r exactly like `kick` does today.
- **No-pixel radius offset:** when no ball pixel exists, still offset the joint
  centre by ball_radius along the horizontal direction of the joint's velocity
  (ball sits in front of the striking limb), never returning a sub-ground z.
- Keyframe `depth_source` gains `player_bone_clamped` so the report can count
  how often the clamp fires (a proxy for foot-anchor error).

## 6. W3 — Airborne physics in events mode (different approach for air)

Group consecutive airborne keyframes between hard knots (touch/bounce/
goal_impact/grounded — `HARD_KNOT_STATES`) into **flight chains**. Per chain,
fit gravity arcs with the existing `fit_parabola_to_image_observations`
(supports hard knots, pixel-ray observations, z-range hinges, world fixes):
- knots = the bracketing hard keyframes' resolved 3-D (post-W2);
- observations = the airborne anchors' clicked pixels (+ accepted detections in
  the span when available) — rays, not fixed planes;
- z-bucket hints demoted to one-sided hinges (as in the 10 cm design C2);
- cross-replay `world_fixes` added with their weights when present.

Airborne keyframe `world_xyz` is re-emitted from the fitted arc (killing the
z = 1.00 flats); ballistic segment hints carry the fitted `p0/v0/g` so the
interpolator and UE render the *fitted* arc, not a re-derived two-endpoint one.
< 2-knot chains keep today's bucket behaviour and are flagged in the diag +
quality report (same under-constrained rule as the 10 cm design). Manual
airborne anchors stay ray-faithful: the C4 snap already guarantees lateral;
depth now comes from physics instead of the bucket plane.

## 7. W4 — Evidence-constrained segment rendering

Thread `steps` (observations), confidences, and fixes into the events path
(`resolve_events` gains optional evidence kwargs; stage passes what it already
has in scope):
- **roll:** where ≥ N confident in-span detections exist, render the span as an
  endpoint-pinned smoothed ground path (robust fit of ray ∩ z=r points,
  monotone-speed / friction-capped); else fall back to an endpoint-exact
  constant-deceleration profile (not constant speed). Straight-line is the
  degenerate case, so sparse-evidence clips lose nothing.
- **ballistic:** render the W3-fitted arc; optional Magnus refinement against
  in-span detections via the existing spin machinery when residuals warrant.
- **carry:** follow the owning player's foot/ground path (player context is
  already loaded in the stage; thread it to the renderer), offset to keep the
  ball at ground radius — replacing the linear Phase-1 interim as §10 always
  intended. Fall back to linear when FK is unavailable.
- **free_flight (bug fix):** two known endpoints ⇒ gravity arc, not linear.
- The derived track remains derived; keyframes + segments stay the
  authoritative, editable product. Segment `hints` carry enough (fitted arc,
  path samples) for UE to reproduce the same shape later (UE-side work is out
  of scope here).

## 8. W5 — Iterate on measured residuals

After W2–W4 land, the harness tells us what still exceeds 20 cm. Expected
next levers, only if measurement demands: missed-event splits from span
residuals (naturalness violations become split proposals), attribution fixes,
sub-frame event timing, camera-error floor investigation.

## 9. Invariants preserved

- Operator input always wins; manual-anchored frames stay pinned exactly.
- `ball.touch_attribution` count-preservation untouched.
- `piecewise`/`global` solvers untouched; events stays default.
- No vendored-code edits; no schema breaks (additive fields only).
- Auto passes never overwrite operator data.

## 10. Test & iteration environment

- Local (this Mac, `.venv311`): no-op-detector stage reruns (seconds, deterministic),
  fine-tuned WASB v1 inference for real-evidence runs (weights present; MPS),
  all unit/integration tests. GVHMR/camera inputs are reused as-is — nothing
  here needs the GPU box.
- Every workstream: TDD (test first), scoped `tests/test_ball_*` runs, then the
  harness acceptance run on all four clips, recorded in
  `docs/superpowers/notes/ball-accuracy/` (baseline vs after).

## 11. Rollout order

1. W1 harness + committed **baseline report** (the numbers we must beat).
2. W2 touch geometry (smallest diff, priority moments) → re-measure.
3. W3 airborne chains → re-measure (expect the largest single improvement).
4. W4 rendering (roll decel+evidence, carry follow, free-flight fix) → re-measure.
5. W5 residual-driven iteration until A1–A5 hold on all clips.
