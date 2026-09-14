# Generic Moving-Camera (Spidercam/Wirecam) Support — Design

**Date:** 2026-09-09
**Goal:** The camera stage's `static_camera` contract assumes one shared world-frame
camera centre for the whole shot — correct for a fixed stadium PTZ rig, wrong for a
spidercam/wirecam replay angle that genuinely translates. `gberch-2` (`output/shots/
gberch-2.mp4`, 181 frames, alongside `gberch` in the main `output/` dir) is exactly
that: the camera flies over the pitch, translating ~10m during the shot while zooming
fx≈990→2680. Forcing a single centre onto it visibly misprojects frame 0. This adds a
tri-state `camera.static_camera: auto|true|false`, a data-driven consistency gate, and
a moving-centre fallback, while leaving `gberch` (a true static camera) unchanged.

## 1. Diagnosis (evidence, reproduced by `probe_static_consistency.py`)

Manual anchors on gberch-2 exist at frames 0, 144, 162, 180
(`output/camera/gberch-2_anchors.json`, 19/19/18/15 landmarks respectively, no lines).

**Solo solves** (per-anchor free R, t, fx via `_solve_one_anchor_full`) — each anchor
fits itself extremely well, but at mutually incompatible camera positions:

| Anchor | Solo residual | Solo camera centre C | Solo fx |
|---|---|---|---|
| f0 | 3.9px | (51.1, 22.2, 11.2) | 990 |
| f144 | 9.8px | (46.5, 13.1, 15.0) | 1304 |
| f180 | 3.5px | (44.8, 15.7, 15.1) | 2626 |
| f162 | degenerate | (115, 26, −252) | ~50 |

f0/f144/f180's centres are real spidercam positions 11–15m above the pitch, ~5-8m
apart — genuine rig translation, not click noise: an exhaustive grid search over
C ∈ x[20,85]×y[−70,40]×z[6,46] with per-anchor free (R, fx) bottoms out at ~24px
minimax across just the three good anchors; per-anchor lens-distortion freedom
doesn't rescue it either (f0: 44.6→40.3px at the locked C). Leave-one-out on f0's own
clicks finds no single bad click that explains it (37→33.5px best) — this is
translation, not a click problem.

**f162 is a separate, secondary bug**: exactly one broken click, `pnl_kp_24` (world
`(0, 13.8, 0)`). Dropping it collapses f162's residual 183px → 9.9px (LOO). Before this
fix, `_solve_one_anchor_full` returned that degenerate solo candidate (fx≈50, C 250m
off the pitch) anyway — every fx-multiplier retry it tried was *also* degenerate, and
the old code fell back to the (degenerate) first attempt rather than signalling
failure. That degenerate `t` entered the "rich" pool feeding the first joint pass's
`t_world` median, corrupting it (see the 1e9-residual relock explosion captured in
`output/logs/job_ade963d1.log`: `shared-C joint LM` walks the seed, the relock forces
one anchor's landmarks behind the camera, and the stage logs `ERROR: static-camera
relock produced mean residual 333333352.35 px`, then *ships that solution anyway* to
honour the (at-the-time-unconditional) static contract).

**Current behaviour** (`output/logs/job_7ba524b1.log`, latest, 4 anchors): the existing
outlier-trim loop already manages to drop f162 before the final relock, but the
remaining three GOOD anchors still don't share a centre — the static relock forces
`C=(45.16, 14.54, 15.27)`, and frame 0 re-solves at 32–37px residual (the user's
complaint: a visibly misprojected pitch overlay at frame 0). Frames 144–180 land near
the compromise C so they look fine; frames 1–143 interpolate between a wrong f0
boundary and a good f144, with focal dipping 875→654→1209 px along the way.

**Reference**: `gberch` (same `output/` dir) has 25 anchors, all consistent with one
C=(51, −33.5, 16) at ~5px mean residual — a true static broadcast camera. This design
must leave gberch's result equivalent.

## 2. Design

### A. Degenerate-solo hardening (`src/utils/anchor_solver.py`)

`_solve_one_anchor_full`'s fallback ladder (11 fx-multiplier attempts) used to seed
`candidate_best` from the caller's primary attempt regardless of degeneracy, so when
every alternative was *also* degenerate it silently returned that original degenerate
candidate. Fixed to track `candidate_best = None` and only ever assign a
`not _is_degenerate_solo(...)` candidate; returns `None` when nothing qualifies. The
hybrid solve's rich-anchor seeding loop (`solve_anchors_jointly`, Pass 1) now logs and
excludes on `None` — "anchor at frame N: solo solve degenerate, excluded from seeding"
— which is what stops f162 (or any single-bad-click anchor) from poisoning the
`t_world` median and the shared-centre LM's seed.

### B. `camera.static_camera: auto | true | false` — gate + moving fallback

New module `src/utils/camera_mode_gate.py` (kept separate from the already
~2000-line `anchor_solver.py`):

- `parse_static_camera_mode(value)` — tri-state parse. Legacy `True`/`False` bools map
  to `"static"`/`"moving"` (100% back-compat: no gate runs for either). Strings
  `"auto"`/`"true"`/`"static"`/`"false"`/`"moving"` (case-insensitive) map the same
  way; `None` (key absent) is `"auto"`. Anything else raises `ValueError`.
- `evaluate_static_gate(anchors, sol, relocked, residual_ratio=3.0,
  residual_floor_px=8.0)` — for every rich, non-degenerate anchor, compares its own
  free (solo) residual against its residual once C is clamped to the candidate shared
  centre (`relocked`, i.e. `refine_with_shared_translation`'s output on the same
  anchors — reused rather than recomputed). Static holds iff every such anchor stays
  under `max(residual_ratio × solo_residual, residual_floor_px)`. Reports the worst
  anchor and the implied centre spread (max distance from each anchor's own solo
  centre to the candidate shared centre) for the loud diagnostic log.
- `refine_with_bounded_motion` (already existed in `anchor_solver.py` as an unused
  building block) is the moving path: a joint LM with a shared *reference* centre and
  a bounded per-anchor `dC`, budget `max_motion_m = max(5.0, camera.motion.max_speed_m_s
  × anchor_span_seconds)` — generous enough to cover a real ~10-15m spidercam
  translation while still catching a genuinely-degenerate solo solve.

  **Two more real-clip bugs surfaced (and fixed) once this ran end-to-end against
  gberch-2 for the first time** — this function pre-existed but was never previously
  exercised on real data:
  1. Its C-seed was `np.mean` over every "rich" (≥6-landmark) anchor's own centre,
     unfiltered by degeneracy. Even though Task A stops a degenerate solo solve from
     entering Pass 1's seeding, f162 still comes out of Pass 3's joint distortion
     refine (which re-frees every anchor's pose with no degeneracy check of its own)
     at fx=57227, C≈(−1184, 1123, 733). A single such outlier in an *unfiltered mean*
     of 4 dragged the shared reference — and therefore every OTHER anchor's centre —
     to nonsense (observed: mean residual exploded 35.78 → 750000264 px). Fixed to
     `np.median` over rich anchors that ALSO pass `_is_degenerate_solo`.
  2. Even with a sane seed, the degenerate anchor's OWN per-anchor `fx` bound
     (`[0.5×, 2×]` its *own* incoming value) was still centred on that same garbage
     fx=57227, handing the LM a nonsense search range for that one anchor — residual
     exploded to ~1e18 px for it specifically (bounded per-anchor `dC` doesn't help
     when the fx range itself is wrong). Fixed by borrowing the nearest (by frame)
     non-degenerate qualifying anchor's `(rvec, fx)` as that anchor's seed AND bound
     centre instead (with `dC` starting at 0, since the shared C_seed is already
     sane) — it still fits poorly (it has a real one-bad-click data problem), just
     boundedly so (81.99px on the real clip, not 1e9 or 1e18).

  Both are covered by `test_bounded_motion_ignores_a_degenerate_rich_anchor_when_seeding`
  in `tests/test_camera_mode_gate.py`, which injects a synthetic degenerate rich
  anchor and asserts both the good anchors' recovery AND the degenerate anchor's own
  result stay bounded.

Stage wiring (`src/stages/camera.py`, `_run_shot`): `config_mode` resolves once. Under
`auto`, the gate runs after the hybrid solve; holding reuses the gate's own relock
candidate (today's static path, including the pre-existing outlier-trim loop) and
failing runs `refine_with_bounded_motion` instead — logged loudly either way with the
worst anchor and centre-spread numbers. Under explicit `true`/`false`, the gate never
runs at all: `true` is `refine_with_shared_translation` unconditionally (today's exact
path — regression-tested against `test_camera_stage_picks_later_anchor_as_primary_when_first_is_thin`
et al.), `false` is the *raw* joint solve, completely unchanged (deliberately **not**
routed through `refine_with_bounded_motion` — an operator who explicitly pins `false`
gets a predictable, unmodified model, not new machinery under their feet; the new
bounded-motion treatment is reserved for auto's own diagnosis).

The same `static_camera` boolean this produces also gates the pre-existing
`line_extraction` branch (`_refine_with_static_line_solve` vs
`_refine_with_line_extraction`) — no separate wiring needed, since that branch was
already keyed off one variable and `_refine_with_line_extraction` was already a
per-frame, no-shared-C solve (today's `static_camera=false` path). This is what
satisfies "the line-solve must not relock a single C for moving shots."

**Inter-anchor interpolation.** Previously, `C_locked` (used to rebuild
`t = -R @ C_locked` under SLERP'd R) was derived from `sol.camera_centre is not None`
— but the moving path's `refine_with_bounded_motion` *also* reports a `camera_centre`
(its shared reference point, not a per-frame invariant), so that check alone would
wrongly LERP a moving clip's per-frame t onto one fixed point. Fixed to gate on the
`static_camera` decision directly. For moving clips, each inter-anchor frame now
interpolates the **physical rig position** (`C_a`, `C_b`, each anchor's own recovered
centre) rather than LERPing `t` directly, then rebuilds `t = -R_slerp @ C_lerp` — this
tracks a smoothly-translating rig instead of conflating translation with whatever R
does between two anchors that don't share a centre. Confidence for these frames decays
from 0.7 at either anchor to a floor of 0.35 at the gap midpoint (vs. static's flat
0.7) — they have no independent observation of their own, just a smooth guess.

**Sidecar.** Per-frame `t` was already always populated (schema unchanged). Top-level
`camera_centre` is explicitly forced to `None` for moving shots regardless of what the
solver internals report, preserving `CameraTrack`'s documented contract exactly
(`camera_centre` non-None ⇒ every frame's `-R^T@t` equals it). `t_world` is `sol.t_world`
(a representative anchor's t) either way — already a "median/representative" value per
its existing docstring, no schema change needed.

**Consumer audit** (`grep -rn "t_world\|camera_centre"` across `src/`):
- `src/pipeline/quality_report.py`, `src/stages/ball.py`, `src/stages/hmr_world.py`,
  `src/web/server.py`, and the dashboard JS (`index.html`, `viewer.html`,
  `anchor_editor.html`, `ball_anchor_editor.html`) all already use the
  `f.t if f.t is not None else t_world` fallback pattern — safe as-is, since every
  frame present in a saved track always has a populated per-frame `t`.
  `quality_report.py`'s static-camera drift check is explicitly gated on
  `cam.camera_centre is not None`, so it correctly skips for moving shots once that
  field is `None`.
- **Bug found and fixed**: `src/utils/gltf_builder.py`'s *main* (broadcast) camera
  export hardcoded a single `t_world` translation and animated rotation only ("t_world
  is clip-shared in the spec") — silently wrong for a moving clip, where the exported
  glTF camera would sit frozen at one point for the whole shot. Fixed to animate
  translation too, using per-frame `-R^T@t` (falling back to `t_world` only when a
  frame's `t` is absent) — mirroring the sibling `extra_cameras` (POV/OTS virtual
  camera) path, which already did this correctly. No-op for a static clip (constant
  translation channel). See `tests/test_gltf_broadcast_camera_translation.py`.
- `index.html`'s overlaid pitch-map camera marker already recomputes `C = -R^T@t` per
  frame (`byFrame` map keyed by frame, redrawn on every `render(currentFrame)`) — the
  drawing code was already correct; only a stale comment claiming "position is
  constant per shot" needed fixing (now describes the actual per-frame recompute).

### C. Leave-one-out click triage (reporting only)

`anchors_needing_click_triage(per_anchor_residual_px, flag_threshold_px, neighbour_ratio=2.0)`
picks anchors whose residual under the chosen model both exceeds the flag threshold
and stands far above the *other* anchors' median (a uniformly-bad clip is a model
problem, not a click problem, so it's excluded). `find_loo_click_culprit(anchor,
solve_fn, min_collapse_ratio=5.0, accept_below_px=15.0)` drops each landmark in turn
and reports the one whose removal collapses the residual by ≥5× **and** leaves a
residual ≤15px — both gates matter: f162's LOO (183→9.9px, 18.5× collapse) clears
both; f0's LOO (37→33.5px best, 1.1× collapse) clears neither, correctly reporting no
false culprit for a genuine-translation anchor. Requires the baseline and every
leave-one-out re-solve to come from the *same* model (`solve_fn`) — passing a
differently-constrained baseline (e.g. a bounded-motion joint fit) against freely
re-solved candidates would "collapse" for reasons having nothing to do with any click,
purely because removing the joint constraint alone improves the fit; this was caught
during testing (`test_camera_stage_recovers_anchor_frames_exactly` regressed) and fixed
by always deriving the baseline from the same per-branch `solve_fn` closure, plus a
`baseline_residual_px <= accept_below_px: return None` guard (nothing to fix if the
anchor already fits acceptably with every click).

Findings are logged (`anchor f%d: click '%s' looks mislabeled (%.1f -> %.1fpx without
it) — reclick or delete in the anchor editor`) and written into the `_camera_summary.json`
sidecar's `click_triage` list. Flag-only — never drops or edits operator clicks.

### D. Anchor-gap surfacing

`find_weak_support_gaps(anchor_frames, per_frame_confidence, max_gap_frames=60,
weak_confidence_below=0.6)` — for moving-camera shots only, a between-anchor span
longer than `max_gap_frames` whose mean per-frame confidence stays below
`weak_confidence_below` (i.e. line-extraction, if enabled, didn't already rescue it to
high per-frame confidence) gets one warning line naming the span and suggesting an
anchor at its midpoint. Static shots don't need this — a locked C makes every
inter-anchor frame equally well-determined regardless of gap width.

### Camera summary sidecar

`{shot_id}_camera_summary.json` — new, per-shot: `config_mode` (as configured),
`mode` (`"static"`/`"moving"`, as decided), `gate` (holds/worst_frame/worst_solo_px/
worst_clamped_px/centre_spread_m/thresholds, or `null` when the gate didn't run),
`click_triage` (list), `weak_support_gaps` (list, moving only). `quality_report.py`'s
pre-existing camera section only ever read the legacy non-prefixed
`camera/camera_track.json` / `camera/anchors.json` (dead for the current multi-shot
layout — confirmed empty on the real `output/quality_report.json`), so it wasn't
touched; the summary sidecar is the additive, per-shot mechanism this design uses
instead.

## 3. Config (`config/default.yaml`)

```yaml
static_camera: auto            # was: true — auto runs the gate; true/false skip it
static_gate:
  residual_ratio: 3.0
  residual_floor_px: 8.0
motion:
  max_speed_m_s: 6.0
click_triage:
  min_collapse_ratio: 5.0
  accept_below_px: 15.0
moving_gap:
  max_gap_frames: 60
  weak_confidence_below: 0.6
```

## 4. Testing

- `tests/test_anchor_solver_moving_camera.py` — Task A only: the degenerate-return
  fix (direct + monkeypatched), and a hybrid-solve-level exclusion-from-seeding check.
- `tests/test_camera_mode_gate.py` — the new module: tri-state parse (incl. legacy
  bool), gate arithmetic (holds/fails, knobs actually change the outcome, trivial
  single/zero-rich-anchor edge cases), the two acceptance-criteria generators (a
  translating-C + zooming-fx trajectory that fails the gate and recovers every
  anchor's centre to ≤1m error via `refine_with_bounded_motion`; the same generator
  family with fixed C that passes the gate and locks one shared centre <5px), LOO
  triage (finds a planted culprit; correctly finds none for genuine-translation data;
  needs ≥2 landmarks), gap surfacing (flags a wide low-confidence span with the right
  midpoint; skips short or well-supported spans).
- `tests/test_camera_stage_moving_camera.py` — `CameraStage` integration: auto+moving
  produces `camera_centre=None` and recovers anchor centres ≤1m; auto+static matches
  today's invariant exactly; legacy `true`/`false` escape hatches; interior-gap frame
  confidence is honestly lower than an anchor's and lies between the two bracketing
  anchors' centres; gap-surfacing and click-triage warnings fire with the right
  labels/frames; the summary sidecar records mode + gate for both outcomes.
- `tests/test_gltf_broadcast_camera_translation.py` — the export-side fix: a moving
  clip's glTF camera animates translation (and the values track the per-frame centre);
  a static clip's stays constant (regression guard).

Two real regressions surfaced and were fixed during this work (not pre-existing —
introduced by the first draft of this feature, caught by the existing suite):
`test_camera_stage_recovers_anchor_frames_exactly` (the click-triage baseline mismatch
above) and `test_camera_stage_picks_later_anchor_as_primary_when_first_is_thin`
(explicit `static_camera=false` was wrongly routed through the new bounded-motion
solve instead of staying on the raw unconstrained joint solve).

Regression baseline: `tests/test_anchor_solver*.py tests/test_camera*.py tests/test_gltf*.py`
— 99 passed, 2 skipped (pre-existing, unrelated skips), no change in skip count.
Full default suite (`tests/`): 2157 passed, 1 failed (`test_ball_stage.py::
test_aerial_arc_promotes_grounded_run_to_flight` — pre-existing per CLAUDE.md, unrelated
to this change), 4 skipped.

## 5. Real-clip validation

Judge clip-level quality only via `scripts/eval_anchor_clicks.py ANCHORS.json
TRACK.json` (positional args) before/after — never a single-run dashboard render, since
PnLCalib-on-MPS parts of the pipeline are nondeterministic across runs. Both clips run
via `CameraStage(config=load_config(), output_dir=Path("output")).run()` directly
(scoped to one shot via `stage.shot_filter`) rather than the full `recon.py` CLI, to
avoid the pipeline runner's stage-cascade (`--from-stage camera` re-runs every later
stage too — `hmr_world` alone is 35-60 min; `--stages camera` alone is a no-op once the
stage is cached, per `CameraStage.is_complete()`).

**gberch-2** — `output/camera/gberch-2_camera_summary.json`: `"mode": "moving"`,
`"gate": {"holds": false, "worst_frame": 180, "worst_solo_px": 3.648,
"worst_clamped_px": 29.013, "centre_spread_m": 7.291}`, `"click_triage": [{"frame":
162, "culprit_name": "pnl_kp_24", "residual_with_px": 1000000000.0,
"residual_without_px": 9.38}]`. `eval_anchor_clicks.py`:

| Anchor | Before (med / max px) | After (med / max px) |
|---|---|---|
| f0 | 17.5 / 93.0 | **3.6 / 8.8** |
| f144 | 6.7 / 56.1 | 5.5 / 49.7 |
| f162 | 4.6 / 1346.3 | 5.2 / 1350.6 (unchanged — the one bad click's own error; only fixable by reclicking, which this feature deliberately never does automatically) |
| f180 | 7.3 / 16.0 | **3.6 / 11.7** |
| ALL | med 7.2, p90 45.9, max 1346.3 | med 4.6, **p90 10.0**, max 1350.6 |

Frame 0 (the user's original complaint) drops from 32-37px (the old forced-static
residual quoted in the diagnosis) to 3.6px median — comfortably under the "≤6px" bar.
144 and 180 both improve slightly rather than regress. f162's per-click stats are
unchanged because they're dominated by the one mislabeled click's own irreducible
error, exactly as expected — the triage warning naming it is the intended fix, not a
number this feature can move.

**gberch** — validated the gate decision and its underlying solver outputs directly
(`solve_anchors_jointly` → `refine_with_shared_translation` → `evaluate_static_gate` on
the real `output/camera/gberch_anchors.json`, 25 anchors) rather than waiting out the
full stage: gberch's `line_extraction` pass (cold-start sweep + PnLCalib bootstrap
trials + propagation over 429 frames) is unchanged, heavy, and — per the task's own
instruction — its PnLCalib-on-MPS parts are nondeterministic across runs and
explicitly not the right thing to judge by anyway; a `--from-stage camera` /
`--stages camera` full-pipeline attempt either cascades into every later stage (35-60
min for `hmr_world` alone) or no-ops on the completeness cache, so a direct solver-level
call is both faster and the more correct check. Result: `gate.holds=True`, worst anchor
f288 at solo 3.72px vs its own C-clamped 3.76px (a ~0.04px difference — gberch's
anchors are so mutually consistent that locking the shared centre barely moves
anything), centre spread 4.25m, `config_mode="auto"` resolving to the static path
exactly as `static_camera: true` always has. This matches the pre-existing evidence
(`refine_with_shared_translation`'s own log: "mean residual 4.87px (was 4.64px)" over
"25 rich+non-collinear anchors") — gberch is unchanged.

## 6. Addendum (same day): corridor-drift defect in the per-frame line solve

A coordinator follow-up review of the persisted gberch-2 track (independent of the
above validation) found 10/181 frames with per-frame implied centres wildly off any
plausible rig position, reported at HIGH confidence (0.95-1.00) — e.g. f105 at 132.0m
off the anchor-interpolated corridor, f140-142 at up to 212.6m off, `p99` corridor
distance 197.8m, and — critically — far (wrong) frames averaging confidence 0.99 vs
0.64 for near (correct) ones: confidence anti-correlated with correctness on exactly
the broken frames.

**Root cause**: `_refine_with_line_extraction` (the pre-existing "legacy independent
per-frame" line solve, used whenever `static_camera` resolves to moving — both
auto-diagnosed and explicit `false`) calls `refine_camera_from_lines`
(`src/utils/line_camera_refine.py`), whose per-frame LM bounded the raw OpenCV `tvec`
to ±300m per component. That bounds `t`, not the camera CENTRE (`-R^T@t`) — for a
frame with only 1-2 detected lines (4 residuals for a 7-DOF `rvec+tvec+fx` problem,
massively under-determined), the centre is free to wander to any point on the
degenerate solution manifold, and the ±300m `tvec` bound does nothing to stop it. This
function pre-existed but had never been exercised on real data before this feature —
gberch-2 previously took the static path unconditionally, so its per-frame line solve
ran through the (unaffected) `_refine_with_static_line_solve`/bundle-adjust machinery
instead.

**Fix** (`refine_camera_from_lines`, `_refine_with_line_extraction`):
- New optional `corridor_centre`/`max_corridor_deviation_m` (default 5.0m,
  `camera.motion.max_corridor_deviation_m`) params re-parameterise the per-frame LM as
  `(rvec, dC, fx)` with `t = -R @ (corridor_centre + dC)` and `|dC|₂ ≤
  max_corridor_deviation_m` (an L∞ box bound sized by `/√3` for the L2 guarantee,
  identical to `refine_with_bounded_motion`'s own conversion) — the centre is bounded
  BY CONSTRUCTION of the optimiser's box bounds (a scipy `trf` guarantee), not a
  post-hoc best-effort check. `_refine_with_line_extraction` passes the anchor-
  interpolated centre already sitting in `per_frame_R[idx]`/`per_frame_t[idx]` (Step
  2's LERP output, read BEFORE this pass overwrites it) for every non-anchor frame;
  anchor frames are exempt (their own click-based pose is authoritative, never a
  corridor/speed-constraint candidate).
- New optional `prev_centre`/`max_step_m` params additionally clamp the ACCEPTED
  pose's centre to within `max_step_m` (`camera.motion.max_speed_m_s / fps`) of
  `prev_centre` — derived FRESH each iteration from the immediately-preceding frame's
  CURRENT state (not a running "last successfully refined" variable), which naturally
  handles runs of failed detections: frame `idx-1`'s state is sane regardless of
  whether it came from its own refinement, an anchor, or an untouched LERP. Line RMS
  (and therefore confidence) is honestly recomputed at the clamped pose in both the
  corridor and speed branches — a frame that had to be pulled back into bounds can
  never report a falsely high confidence from its pre-clamp (spuriously better) fit.
- **A second gap found during real-clip re-validation** (not covered by the first
  fix): a frame whose OWN detection fails outright keeps its existing (untouched,
  never-speed-checked) pose. If its predecessor had legitimately drifted a few metres
  within its own corridor allowance (each individual step fine), silently falling back
  reproduced the same kind of frame-to-frame jump (observed 1.3-2.6m in a single 1/30s
  frame on the real clip, e.g. f179→f180: 77.87 m/s implied). Fixed by applying the
  identical speed clamp to the failed-detection fallback path too.

**Remaining, understood, exempt case**: 3 transitions directly INTO an anchor frame
(f143→f144, f161→f162, f179→f180) still exceed the per-frame speed budget (26-78 m/s
implied) — because the anchor's own pose comes from the click-based
`refine_with_bounded_motion` solve, which is authoritative and deliberately never
corridor/speed-constrained (matching every other exemption in this design). This is
qualitatively different from the original defect: every position involved is within
the corridor (not 100+m wrong), and the frames approaching the anchor already carry
LOW confidence (0.3, the line-extraction floor) rather than the falsely-high
confidence the defect was about — i.e., the system is already honest about not
trusting that stretch. Constraining it further would mean either weakening the
anchor's authoritative pose or a bidirectional smoothing pass neither this fix nor the
original diagnosis called for; flagged here for visibility rather than silently
accepted.

**Verification** (gberch-2, `output/camera/gberch-2_camera_track.json`,
`scripts/eval_moving_camera_corridor.py` — new, reusable diagnostic for this class
of defect):

| Metric | Before | After |
|---|---|---|
| Corridor distance p50 | 0.3m | 0.25m |
| Corridor distance p99 | 197.8m | 2.53m |
| Corridor distance max | 212.6m | 2.60m |
| Non-anchor frames >5m off corridor | 10 | 0 |
| Far-frame mean confidence | 0.99 | n/a (0 far frames) |
| Interior frame-to-frame speed violations | multiple (18.6-212.6m jumps) | 0 |
| Anchor-boundary transitions (documented exempt) | n/a | 3, all <2.6m corridor distance |

Anchor-frame `eval_anchor_clicks.py` numbers are bit-identical to the pre-corridor-fix
run (f0 3.6/8.8px, f144 5.5/49.7px, f162 5.2/1350.6px, f180 3.6/11.7px) — anchor poses
are untouched by this fix, exactly as designed. `gberch` was not regenerated
(`gberch_camera_track.json` mtime unchanged throughout); the static path
(`_refine_with_static_line_solve`) never calls `refine_camera_from_lines` and is
therefore structurally unaffected — confirmed by grep and by the unchanged scoped
regression suite (107 passed, 2 pre-existing skips).

New tests: `tests/test_line_camera_refine.py` (6 — corridor bound respected even when
the true camera is far outside it; doesn't disturb a well-determined in-bound fit;
speed clamp limits/no-ops correctly; line RMS honestly reflects the clamped pose;
legacy no-corridor behaviour preserved) and two additions to
`tests/test_camera_stage_moving_camera.py` (wiring: corridor/speed params threaded to
every non-anchor frame, withheld from anchor frames; the failed-detection-fallback
gap, reproduced then fixed).

