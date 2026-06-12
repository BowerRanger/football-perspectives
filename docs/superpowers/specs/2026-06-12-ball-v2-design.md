# Ball v2 design: evidence booster, global mode-sequence solve, spin

Status: draft for review.
Scope agreed 2026-06-12: Ideas 1, 3, 4 from
[2026-06-12-ball-v2-ideas.md](2026-06-12-ball-v2-ideas.md), in that order.
Idea 2 (cross-replay triangulation) is parked until multi-shot support is
ready; Idea 5 (UE realism layer) and Idea 6 (learned lifting prior) deferred.

## Goals

- Raise 2D ball evidence density so detector-limited clips (origi02: 44 %
  coverage) become solvable, with zero new user input.
- Replace greedy event-classification + split-and-retry with a global
  mode-sequence search so multi-impact sequences (goalmouth scrambles,
  deflections) segment correctly.
- Make spin a first-class fitted state with physical bounds and bounce
  coupling, and export ball rotation (today: position only).

## Non-goals

- No multi-shot/replay fusion (parked with Idea 2).
- No WASB retraining/fine-tuning (optional later; inference-time only here).
- No learned action spotter (touch typing is geometric in this iteration).
- No UE-side net/keeper simulation work (Idea 5); we only guarantee the
  exported data carries what that layer will need.
- Manual anchors remain authoritative everywhere — nothing in this design
  may override an operator anchor.

---

## Phase 1 — Ball Evidence Booster

### Problem

WASB at production threshold yields sparse confident detections (origi01:
168/506 frames ≥ 0.3 conf; origi02: 148/334 with any uv). Gaps starve the
event detector, the auto-anchors, and the solver alike.

### Design

Two-pass detection inside `src/stages/ball.py`'s detect loop, plus a
forward–backward track smoother to predict where to look.

**1. Candidate API on the detector** (`src/utils/wasb_ball_detector.py`)
- New method `detect_candidates(frame_bgr, min_score, top_k) ->
  list[BallCandidate]` where `BallCandidate = (uv, score, blob_area_px)`.
  Implementation: same HRNet forward pass, lower heatmap threshold, return
  top-k connected-component centroids with scores instead of the single
  best-above-threshold blob. The existing `detect()` becomes a thin wrapper
  (top-1 at production threshold), so pass 1 behavior is unchanged.
- The YOLO fallback gets the same interface (top-k boxes by confidence).

**2. Forward–backward IMM smoothing** (`src/utils/ball_tracker.py`)
- Run the existing IMM forward as today, then a second IMM pass over the
  reversed observation sequence; fuse per-frame state estimates (covariance-
  weighted average of forward and backward means; standard two-filter
  smoother). Output per frame: smoothed uv prediction + covariance, used
  only for second-pass gating — the persisted track semantics stay causal-
  compatible (same fields, better values).

**3. Second pass over evidence gaps** (new `src/utils/ball_second_pass.py`)
- For every frame where pass 1 produced no accepted detection (missing or
  rejected as outlier): build a gating ellipse from the smoothed prediction
  covariance (inflated by `corridor_sigma`), call `detect_candidates` on
  that frame, keep candidates inside the ellipse, and re-score:
  `combined = candidate.score * exp(-0.5 * mahalanobis² / corridor_sigma²)`
  — the hard gate (mahalanobis ≤ corridor_sigma) does the rejection work;
  the exponent is sigma-normalised so the distance penalty stays mild and
  `accept_min` is tuned against the candidate's score, not its position.
- Accept the best candidate if `combined >= accept_min`. Accepted frames are
  appended to observations with `source: "second_pass"` and
  `confidence = combined` (always < pass-1 confidences by construction of
  `accept_min`).
- **Feedback-loop guard**: the smoother that defines corridors is built from
  pass-1 observations only — second-pass detections never widen or steer
  the corridor that admitted them, and the second pass runs exactly once
  (no iteration).
- **Zoom retry**: if no candidate clears `accept_min` and the predicted
  apparent ball size is below `zoom_min_ball_px`, crop `zoom_crop_px`
  around the prediction, upscale to the detector input size, re-run
  `detect_candidates`, and map centroids back through the crop transform.
  Same acceptance rule.

**4. Downstream weighting**
- The IMM/event/anchor pipeline re-runs over the merged observation set.
  `second_pass` observations participate in event detection and solving but
  are excluded from auto-anchor *generation* (gates in
  `ball_auto_anchor.py` already key on confidence; add an explicit
  source check) — they densify evidence, they don't mint constraints.

### Config (`config/default.yaml`, `ball.second_pass.*`)

```yaml
second_pass:
  enabled: true
  candidate_min_score: 0.05   # heatmap floor for detect_candidates
  top_k: 5
  corridor_sigma: 3.0         # gating ellipse inflation
  accept_min: 0.25            # combined-score acceptance gate
  zoom_min_ball_px: 8.0
  zoom_crop_px: 320
```

### Diagnostics

`_ball_diag.json` gains `detection_coverage: {pass1, pass2, total}` (fractions
of frames) and per-source observation counts; `quality_report.json` surfaces
coverage per shot so detector-limited clips are visible at a glance.

### Acceptance criteria

- origi02 total coverage ≥ 0.75 (from 0.44) with anchor-accuracy harness
  (`tests/test_ball_anchor_accuracy.py`) still green.
- kroupi01 / origi01: no segment-residual regressions, no new jumps > 2 m.
- Synthetic unit tests: corridor gating rejects a decoy blob outside the
  ellipse; feedback guard holds (second-pass output identical when run
  twice); zoom path maps coordinates back exactly.

---

## Phase 2 — Global mode-sequence solve

### Problem

Events are classified greedily and anchors are minted before the solver
runs; the solver then fits between fixed nodes and can only split-and-retry
locally. Real failure: origi01 fit one ballistic span 454–488 across
detected side-net impacts (451, 460), a bounce (465) and velocity breaks
(470, 475). The evidence existed; the architecture couldn't use it jointly.

### Design

A per-shot beam search over **timeline partitions**: which mode the ball is
in between candidate breakpoints, scored by how well bounded physics
primitives explain the pixel evidence. Greedy classification becomes
hypothesis scoring. New module `src/utils/ball_mode_search.py`; existing
primitives in `ball_physics.py` / `bundle_adjust.py` are the segment fitters.

**Modes** (per segment): `rolling`, `flight`, `possessed`, `stationary`,
`out_of_view`. Mode transitions occur at **breakpoint candidates**.

**Breakpoint candidates** come from the existing event detector
(`ball_auto_events.py`) run in a permissive profile: every velocity break,
touch, bounce, goal-impact candidate with its score — decisions deferred to
the search. Manual anchors are forced breakpoints.

**Hypothesis** = ordered list of (breakpoint frame, transition kind) +
per-segment mode. Scored as the sum of:
- **Segment fit residuals**: each segment fitted by its primitive —
  `rolling`: existing endpoint-exact roll fit; `flight`: parabola (Magnus
  refinement stays a post-pass, Phase 3 upgrades it); `possessed`: ball
  follows the possessing player's foot position from `refined_poses` FK
  with a soft pixel tether (weak/occluded pixel evidence expected) — the
  possessing player is searched over the 2 players nearest the ball's last
  confident pixel before the segment, each as its own hypothesis branch;
  `stationary`: constant position; `out_of_view`: no pixel cost, fixed
  per-frame penalty so it's never free.
- **Transition priors**: flight→rolling/flight requires bounce (restitution
  inside the envelope, else penalty), touch transitions require a player
  joint within `contact_max_gap_m`, goal impacts require ray–goal-geometry
  proximity; each prior reuses the existing gate math as a *cost*, not a
  filter.
- **Event agreement**: a transition coinciding with a high-score detected
  event earns a bonus; a transition with no event evidence pays
  `unexplained_break_penalty`; a high-score event with no transition pays
  `ignored_event_penalty`.
- **Complexity prior**: per-segment constant (BIC-style) so the search
  prefers fewer segments when residuals tie.

**Search**: left-to-right beam over breakpoint candidates (sorted by frame),
beam width `beam_width` (default 24). State = (last breakpoint, mode,
accumulated cost). Segment fits are memoized by (start, end, mode) — the
dominant cost — and bounded by `max_segment_fit_calls` as a safety valve.
Deterministic by construction (no randomness; ties broken by frame then
mode order).

**Hard constraints**: manual anchors must lie on the trajectory exactly as
today (`_resolve_anchor_world` unchanged); hypotheses violating an anchor
are pruned, not penalized. Auto-anchor *generation* for the dashboard
remains, but the solver consumes the winning hypothesis's transitions
directly; persisted `_ball_anchors_auto.json` is now derived from the
winning hypothesis (same schema — dashboard unchanged).

**Output compatibility**: winning hypothesis renders to the same
`BallTrack` (dense frames), `flight_segments`, keyframes and diag schemas.
`segments[].kind` gains values `possessed`/`stationary`/`out_of_view`
(additive). Diag gains `mode_search: {hypotheses_explored, beam_width,
winning_cost, runner_up_cost}` for tuning.

**Rollout**: `ball.solver: piecewise | global` config switch; `piecewise`
remains the default until validation passes, then flips. The piecewise code
path is kept for one release as the fallback.

### Config (`ball.mode_search.*`)

```yaml
mode_search:
  beam_width: 24
  segment_cost_constant: 6.0      # BIC-style per-segment penalty
  unexplained_break_penalty: 10.0
  ignored_event_penalty: 8.0
  out_of_view_frame_penalty: 1.5
  possessed_tether_px: 40.0
  max_segment_fit_calls: 20000
```

### Acceptance criteria

- origi01 454–488: winning hypothesis contains ≥ 3 segments with a net
  impact and a bounce transition (matches hand-read ground truth), no
  underconstrained flag, residual ≤ 8 px per accepted segment.
- origi01 201–282: the 40-velocity-break span resolves to ≤ 4 segments with
  residuals ≤ 8 px (currently 2 segments at 62–76 px).
- kroupi01: identical or better residuals than piecewise on every segment.
- All clips: anchor harness green; `--from-stage ball` runtime ≤ 3× current.
- Unit tests: synthetic multi-bounce + possession scenarios where the
  correct partition is known; determinism test (two runs, identical output);
  manual-anchor pruning test.

---

## Phase 3 — Spin as a coupled state + rotation export

### Problem

Spin today is preset-seeded Magnus refinement per isolated segment with a
200 rad/s cap (~32 rev/s — beyond any real kick), no coupling across
bounces, and no exported rotation: the ball never visibly spins in the
viewer or UE.

### Design

**1. Bounded per-segment spin** (`bundle_adjust.py`, solver call sites)
- `omega_mag_bound` becomes always-on at `spin.max_omega_rad_s: 95`
  (~15 rev/s, top of real free-kick range). Acceptance gates
  (`min_residual_improvement`, endpoint continuity) unchanged.

**2. Bounce coupling** (`ball_physics.py` + solver)
- New spin-aware bounce model:
  `v_out = bounce(v_in, ω_in; e, μ)` — normal restitution `e` plus
  tangential update from sliding/rolling friction `μ` and spin (standard
  rigid-sphere-on-plane impulse model); returns `(v_out, ω_out)`.
- At every flight→flight bounce node in the winning hypothesis, the two
  adjacent flight segments are refit **jointly**: parameters
  `(p0, v0, ω0, e, μ)` with `e ∈ [restitution_min, restitution_max]`,
  `μ ∈ [0, mu_max]`; residual = pixel reprojection over both segments.
  This is the identifiability mechanism the gray-box literature validates:
  curvature alone is weak, curvature + bounce kinematics is strong.
- Joint fit accepted on the same improvement gate; on rejection both
  segments keep their independent fits (today's behavior).

**3. Geometric touch typing → spin seed** (extends `ball_spin_presets.py`)
- At touch transitions, derive a seed from geometry already in hand:
  contact bone (foot/head/chest from FK), approach/exit velocity directions.
  Mapping: foot contact with large horizontal direction change → side-spin
  about z (sign from cross product of in/out velocities); lofted foot
  contact (exit elevation > 25°) → backspin seed; header → ω seed 0.
  Seeds feed the existing `omega_seed` path with the hinted improvement
  gate. Manual preset on an anchor still overrides any derived seed.

**4. Ball orientation integration + export**
- New `src/utils/ball_orientation.py`: integrate quaternion q(t) over the
  dense track — flight: q̇ from fitted ω (constant per segment);
  rolling/possessed/stationary: rolling-consistent ω = (v/r) about the
  horizontal axis perpendicular to travel (zero when stationary);
  transitions: ω changes discontinuously at nodes, orientation stays
  continuous. Output: per-frame unit quaternion appended to `BallFrame`
  (`quat_wxyz`, optional field — schema-additive).
- Export: glTF gains a rotation sampler on the ball node
  (`gltf_builder.py`); FBX export keys rotation in the Blender script;
  `_ball_keyframes.json` carries per-flight `omega_rad_s` (vector) so UE
  can later drive physically-correct curl. Existing UE `ball_motion.py`
  preset path keeps working (additive data only).

### Config changes (`ball.spin.*`, `ball.physics.*`)

```yaml
spin:
  max_omega_rad_s: 95          # was 200
  bounce_coupling: true
  mu_max: 0.7                  # tangential friction bound in bounce model
```

### Acceptance criteria

- Synthetic round-trips: generate drag+Magnus trajectories with known ω
  (5–12 rev/s) + spin-coupled bounces; joint fit recovers |ω| within 20 %
  and axis within 25° — and, run on spin-free synthetics, reports |ω| < 1
  rev/s (no spin hallucination).
- Real clips: every accepted spin fit satisfies bounds + improvement gates;
  no segment-residual regressions vs Phase 2 output.
- Viewer/UE: ball node carries rotation animation; rolling ball's contact
  point has near-zero slip velocity (computable check in the export test).

---

## Testing strategy

TDD throughout (write failing test → implement → refactor), consistent with
the existing `tests/test_ball_*.py` style: pure numpy/scipy, light-venv
runnable, detectors and video mocked behind the candidate API. Each phase
lands with: unit tests for new math (smoother fusion, bounce-spin impulse,
quaternion integration), scenario tests on synthetic trajectories with known
ground truth, the anchor-accuracy harness, and a real-clip validation run
(kroupi01, origi01, origi02 — same metrics tables as the auto-physics
validation) appended to this doc before each phase is called done.

## Risks

- **Phase 1 false positives**: corridor + combined-score gate + the
  feedback-loop guard; second-pass detections can't mint anchors. Residual
  risk visible in coverage-vs-residual diagnostics.
- **Phase 2 search cost**: memoized segment fits + beam cap +
  `max_segment_fit_calls`; worst case degrades to piecewise-quality answer,
  never hangs. Config flag keeps the old solver one switch away.
- **Phase 2 scoring weights need tuning**: weights are config, diag exposes
  winner/runner-up costs, and the acceptance clips are the tuning set.
- **Phase 3 spin hallucination**: bounds, improvement gates, and the
  spin-free synthetic test guard it.

## Phase ordering note

Phase 3's bounce coupling assumes Phase 2's segmentation (joint refits need
correct bounce nodes), so phases land 1 → 2 → 3. Phase 3's export work
(orientation integration, glTF/FBX rotation) has no Phase 2 dependency and
can be pulled forward if a visible win is wanted early.
