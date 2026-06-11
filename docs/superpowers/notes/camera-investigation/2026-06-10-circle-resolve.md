# Circle-aided re-solve — origi01 full-clip coverage + midfield accuracy

Goal (session 2026-06-10): accurate camera position/rotation/zoom/lens across
ENTIRE clips (gberch, kroupi01, origi01, origi02), in service of player
placement + HMR_World. Entering state: gberch/kroupi/origi02 full coverage;
origi01 65 % (176–505) with start uncovered.

## Diagnosis (origi01)

Honest-metric probe: reproject the MANUAL anchor clicks (user ground truth,
valid even where the manual *track* is junk) under the live auto track:

| anchors | clicks reproj med |
|---|---|
| f200–f294 (midfield) | **305–355 px** |
| f399–f505 (box end)  | 4–11 px |

The live track did not even fit its own PnLCalib anchors at f176–286
(306–348 px) while nailing f462 (5.8 px). Mechanism (confirmed in run logs):

1. **Stale-R inconsistency** — the static bundle locks C from the line-solved
   (box-end) frames only: C_bundle=(48.9,−31.4,15.1) vs anchor-stage
   C=(52.4,−36.4,16.2). Frames the bundle skipped (no/too-few straight lines —
   the whole midfield span 176–342) kept rotations solved under the OLD
   geometry while `t = −R·C_new` was re-derived ("One-C consistency" loop) →
   wholesale ~300 px projection shift. The circle-lens refinement repeated the
   same patch with a moved principal point.
2. **Start 0–175 blocked** — content is halfway line + far touchline + CENTRE
   CIRCLE (+ a parallel 18yd pair early on): ≤2 straight, mutually parallel
   lines. Cold-start required ≥3 straight lines (`cs_min_lines`); propagation
   had `propagate_circle: false` (an old line-RMS-metric verdict — the metric
   is blind to the circle and self-referential); and the propagation seed at
   the f176 boundary was itself ~300 px wrong (issue 1).
3. PnLCalib keypoints/anchors in the midfield were FINE — user clicks and
   PnLCalib agree; the track lost them after the bundle.

## Changes (all in src/stages/camera.py + config/default.yaml)

1. **Demote-and-re-solve** (`line_extraction_resolve_underlined`, default
   true): frames the bundle skipped are demoted to uncovered instead of
   keeping stale R; the propagation pass re-solves them against their own
   detected lines (+ circle) at the locked geometry.
2. **Circle-when-sparse**: propagation consults the centre circle ONLY when a
   frame has < `min_lines_per_frame` straight lines (so the historical
   "circle trades away straight-line fit" regression cannot recur);
   `line_extraction_propagate_circle` default flipped to true. Cold-start
   accepts ≥1 line + circle (was: ≥3 lines).
3. **Lens-limited acceptance for circle frames**
   (`line_extraction_propagate_circle_max_rms: 12.0`): circle-aided sparse
   frames live on the wide/midfield end where a still-imperfect global lens
   inflates the residual; the post-propagation circle-lens refinement then
   corrects the lens (same rationale as the cold-start's 12 px gate).
4. **Interior gap-fill**: frames nothing can re-solve are SLERP/LERP-filled
   between solved brackets (consistent with the final geometry), replacing
   stale poses. Runs again after the lens refinement.
5. **Post-lens-refinement re-solve**: frames the lens solve didn't include
   are re-solved against their stored lines+circle under the NEW lens
   (was: keep old R, patch pp/C — issue 1 again); failures are invalidated
   and gap-filled.
6. **Pinning**: circle-solved frames are pinned in pin-and-smooth (else the
   smoother drags the only real solves in a line-sparse span).
7. extend_coverage restricted to frames beyond the solved span (interior
   demoted frames are propagation's job — avoids pointless per-frame
   PnLCalib inference).

TDD: tests/test_camera_stage_circle_resolve.py — synthetic constant-pan clip,
anchored only at the box tail, start = halfway+circle only; asserts the start
is covered AND the centre-spot projection matches truth (<40 px).

## Iteration findings (chronological)

- **Run-to-run noise**: the PnLCalib-on-MPS bootstrap is nondeterministic
  enough to move the bundle's C/pp between runs (kroupi C moved 0.5 m, all
  metrics wobble). gberch (which never uses that bootstrap) reproduces
  byte-identically. Judge changes on the honest click metric, not single-run
  dashboard deltas.
- **Boundary poisoning (origi02)**: circle-aided propagation marching past
  the clean line boundary accumulates lens-limited drift (≤3°/step compounds)
  and walks fx; the cold-start sweep then fails from the poisoned reference
  frame and the whole start is lost. A/B with flags off reproduced Jun-8
  quality exactly → the fix is ORDERING: two-pass propagation
  (lines-only → cold-start from the clean boundary → circle pass fills what
  remains). The orientation SWEEP also stays line-only (a false circle lock
  under a far-off sweep candidate can out-score finding nothing);
  the circle is allowed in cascade-fill (adjacent seeds).
- **In-view gate**: hallucination theory disproved (origi02's start sees the
  circle at fraction 1.0) but the gate stays at ≥0.3 to block true
  out-of-view attempts (origi01 f400 = 0.0).
- **Seed fragility at islands**: at fx≈4400 the ±25 px line strip tolerates
  ~0.3° of seed error; where an anchor island meets a circle-solved frame the
  velocity extrapolation is poisoned by the regime disagreement →
  detect_circle finds nothing. Fixes: wider circle strip (50 px — the circle
  has no adjacent-parallel trap) + plain-neighbour-seed retry in the circle
  pass.

## Results (honest click metric — reprojection of hand-placed anchor clicks)

origi01 midfield (was the catastrophic span):

| anchor | baseline | final |
|---|---|---|
| f200 | 338 px | **31.9 px** |
| f230 | 333 px | **21.8 px** |
| f255 | 306 px | 56.4 px |
| f294 | 355 px | 98.5 px (these clicks were never fittable — 121 px under the manual solve itself) |
| f399–f505 (box) | 4–11 px | 6.6–10.5 px (unchanged) |
| ALL median | 305 px | **9.8 px** |

PnLCalib anchors f176–286: 306–348 px → 10.5–30.4 px (keypoint noise floor).
Held-out circle misfit: 26.4 px → **3.6 px**; the centre circle fits at
3.5 px median on 18 frames with NO lens refinement needed.

origi02 final: 100 % coverage, k1=0.223-0.238 across runs (matches the
validated 0.23), held-out circle 6-10 px, vs-manual centre 0.55-0.61 m, clicks
med 8.6 px (better than baseline). gberch: byte-identical. kroupi01: neutral
within bootstrap noise.

## Cold-start circle: the lens-limited gate

The sweep+circle acquires on every origi01 start frame but the refined MEDIAN
residual is 13-37 px — global lens error across the circle's arc (the start is
solved under the box-derived lens, same chicken-and-egg the cold-start design
anticipated). Gates: raw-rms gating was wrong twice over — fat-tail detector
outliers (players/D-arc in the 100 px acquisition strip) dominate the raw rms
of a CORRECT Huber-solved lock, so circle frames gate on the median
(`line_extraction_cold_start_circle_max_rms: 40`); and the 3° pull-guard must
relax to 6° for sweep/refine (coarse-grid candidates legitimately pull that
far toward truth; 3° stays for adjacent-seed cascade fills).

## User-reported issues (post-milestone) + next lever

1. **Cross-image streaks (origi02 f215+, viewer)** — DISPLAY bug: catalogue
   points beyond the distortion polynomial's monotonic radius (fold at
   r=1.10 normalized for k1=0.22) fold back INSIDE the image. Fixed with a
   fold-radius clip in anchor_editor.html (`distortionFoldR2`) and
   dump_overlay_frames.py. The solver was never affected.
2. **Far-field 1-2 m error (origi02 pre-box span; kroupi f123+)** — the far
   touchline is essentially never DETECTED (low contrast under the bright
   boards; a 1-2 m camera error puts it outside the ±25 px strip), so nothing
   pins the far field. Confirmed visually: kroupi f123's projected far
   touchline lands ON the advertising boards while detections sit on paint.
3. **origi01 f289-354 wobble** — three stacked causes, found by peeling:
   (a) circle-dominant partial-arc solves (arc 0.3-0.5 in view) wobble →
   require arc fraction >= 0.5 when < 2 straight lines; (b) near-PARALLEL
   sparse line pairs fit at ~0 rms while fx slides → angular-spread gate
   (>= 20 deg) on sparse lines-only solves, applied to the fallback path too
   (the first version leaked through it); (c) the killer: a 47-frame march of
   exactly-determined 2-line solves random-walks fx (4800 -> 2893) even with
   good spread and per-step bands — fixed with an ORIGIN-ANCHORED envelope:
   every march inherits the fx of the solid frame it started from and no
   solve may leave [0.75, 1.3]x of it. Result: the dead zone falls to smooth
   SLERP between the f286 island and the box bracket — steady drift instead
   of jumping. Per-frame ACCURACY there awaits the hoardings constraint
   (the boards are crisp in exactly those frames).

**Next lever — advertising hoardings as a static scene line (user idea)**:
board base = line parallel to the far touchline at unknown per-clip (offset d,
height h), solved once in the global bundle, then a per-frame constraint
wherever visible (= everywhere, incl. origi01's start and dead zone). Highest-
contrast feature in exactly the starved zone; also disambiguates the far
touchline (search below the solved board line). Prefer this over further
circle tuning — it addresses origi02 far field, kroupi tail, and origi01
start/dead-zone at once.

**origi01 result: 100 % coverage (506/506) for the first time.** Clicks:
box 6.6-10.5 px, midfield 14.5-22.5 px, start 30.8-158.9 px (lens-limited —
degrades toward f0 because the circle-LENS refinement silently skipped:
its ellipse detection band (50 px) can't find the ring under the lens-limited
start cameras, and/or too few ellipse frames intersect the >=2-straight-line
pfl gate). Closing that lens loop is the remaining lever for a fully accurate
start.

## 2026-06-11 — Advertising-hoardings static scene line (user idea, implemented)

`src/utils/hoarding_detector.py`: the LED board base modelled as `y = 68 + d,
z = 0` (one `d` per clip — with a static C the (offset, height) family is
projectively equivalent, so h is fixed at 0 and ONE parameter is solved).
Detection = signed step kernel (bright band above, grass below) + RANSAC-style
geometric consensus across perpendiculars (photometric gates alone are too
brittle at far field: floodlit grass desaturates below any absolute threshold).
Calibration = per-frame-median of independent per-frame d solves (immune to
per-frame camera wobble; a joint LSQ smears d into the bound).

ORDERING LESSON (cost: one poisoned origi02 run, clicks med 9.9 -> 82.6 px):
calibrating right after the bundle uses box-end cameras that barely see the
boards -> garbage d -> injecting that plane into propagation/cold-start drags
whole spans into a wrong basin. The board must be calibrated AFTER coverage
exists (post-propagation, on the ~10 covered frames with the highest far-
touchline in-view fraction) and applied only as (a) a GATED RE-SOLVE of
covered frames (pitch features + board jointly; accepted only within 2 deg /
12 px of the current camera — adjust, never replace) and (b) stored
`board_line` entries that the lens refinement / outlier passes consume.
gberch: no covered frame passes the visibility gate -> calibration skipped ->
untouched.

### Hoardings result (final)

- kroupi: board ACTIVE (d=3.44 m, spread 0.20 m) — 121 frames far-field
  re-solved, clicks med 7.0 -> 6.3 px, the f123+ projected far field sits on
  the paint (the user-reported symptom), jitter 0.79 deg.
- origi01/origi02: calibration ABSTAINS (d-spread 0.7-0.9 m > the 0.5 m
  trust gate — their far-field camera wobble is the very thing being
  measured); tracks unchanged at their best states. Unlock path: better
  cameras (or weighting calibration frames by anchor quality) shrinks the
  spread below the gate.
- gberch: visibility gate skips (no covered frame sees the board zone).

### Quality-first calibration selection (2026-06-11, follow-up)

Calibration frames are now chosen by CAMERA QUALITY among board-visible
frames (circle entries +2, straight lines up to +4, anchor island +1; need
score >= 2), not visibility alone. kroupi: d-spread 0.20 -> 0.07 m (applies,
clicks hold at 6.3 px). origi01: 0.77 m, origi02: 0.90 m — UNCHANGED by
selection, i.e. the spread is LENS-BOUND, not selection-bound: the board
lives in a different image zone (above the far touchline) than the circle
that pins their midfield, and the unrefined wide-field lens tilts between
the zones. Their unlock — and the single deepest remaining lever for the
camera stage — is the lens-loop robustification: feed circle-point (and
board) frames into solve_static_camera_from_lines' global lens refinement so
it no longer starves on the >=2-straight-line pfl gate (validate against
origi02's k~0.23, which must not regress).

### Lens-loop robustification (2026-06-11)

Stored circle POINTS (the ridge detections committed by cold-start /
propagation solves) now serve as both a second refinement trigger and a
refinement input: frames join the lens solve on >=2 straight lines OR their
stored circle points (the solver consumes them as weighted point residuals;
empty line lists are fine). Residual-budget balance is essential: origi01's
261 circle frames x 20 points outvoted its 174 line frames wholesale and
bent k1 to ~0 — the ring fit at 0.9 px held-out while the user's clicks got
WORSE (fit-vs-truth trap); circle_weight is now capped so the circle's total
residual count <= half the lines'.

Results: origi02 k 0.123 -> 0.147 (toward the validated 0.23), vs-manual
1.35 -> 1.09 m, jitter p95 0.40 -> 0.26, clicks tail p90 92 -> 73 px.
origi01: neutral — its lens is no longer the binding constraint when refined
(held-out ring 0.9-2.8 px); its START accuracy now varies 141-226 px at f0
ACROSS RUNS of the same code (cold-start / PnLCalib nondeterminism), which is
the next investigation: stabilise the start solve (e.g. seed the sweep from
several boundary references, or average sweeps) before any further lens or
board work on that clip.

## 2026-06-11 night — 1-foot push: audits, polish, and the origi01 triangulation

Workflow audit (5 parallel analysts) localized every >0.3 m error: dominant
mode = horizontal/pan at far field; origi01 start/gaps had ZERO stored
constraints in the final track (the outlier pass mass-rejected honest sparse
solves via BOTH criteria and deleted their entries), fx collapsing in anchor
gaps; properly line-solved frames pass everywhere.

Fixes landed: iterative circle-lens refinement with a broad-evidence stop;
GLOBAL POLISH (per-frame constraints + continuity priors, Gauss-Seidel,
strong frames pinned at >=3 lines); outlier pass de-fanged for polished
frames only (scoped rot-exemption after a 7 deg spike on origi02 showed the
pinned frames still need it) and median-stat for circle-bearing frames.

The C experiments (scripts/probe_c_hypothesis.py): the stored detections
PREFER the wrong C (self-confirming strip-search bias — med 2.98 px at the
biased C vs 5.16 at the click-true C). Anchor-stage C fits midfield clicks
(4-7 px) but degrades the box; bundle C the reverse. Anchor-point hints in
the lens refinement at sane weight cannot steer C against thousands of
biased line residuals.

Anchor-set triangulation (the decisive experiment):
- AUTO anchors: midfield 14-23 px, start 210-330 px, box 4-11 px
- MANUAL anchors: start 12-21 px, midfield 102-181 px, box 4-12 px, jitter 0.78
- AUGMENT (both): start 15-16 px, midfield 29-127 px, box 4-6 px
No static-C + single-k1k2 configuration fits all three spans. origi01 zooms
~1.5x through the clip; broadcast lens distortion is zoom-dependent — the
documented frontier is a zoom-parameterised distortion model
(k = k(fx), two extra globals) or relaxing the static-C assumption.

FINAL per-clip states: origi02 = manual anchors + full stack — clicks med
8.5 px / p90 19.7 px / max 46.5, circle held-out 3.6 px on 30 frames,
vs-manual centre 0.29 m (at or near the 1-foot bar nearly everywhere).
origi01 = manual anchors + full stack — start+box 4-21 px, jitter 0.78 deg,
midfield span (f169-294) remains ~2 m pending the zoom-distortion model.
kroupi (clicks 6.3 px med, jitter 0.75) and gberch (baseline) provably
untouched by every change in this round (gates skip both).

## 2026-06-11 (day 2) — the C/lens arbitration endgame

The decisive discovery chain: (1) C is near-unidentifiable along the view
axis from any single-azimuth evidence (solo box-anchor centres slide ~4.5 m
along one line); (2) detections are strip-searched around the current
cameras' projections and SELF-CONFIRM whatever C they were found under, so
the free profile+bundle walks C down a shallow valley on azimuth-poor clips;
(3) with C held, the compensation moves to (k1, fx) — k walked to 0.395 with
fx ~12% high and the wide-field anchors paid ~300 px; (4) a broken
auto-anchor relock used as a SEED poisons everything (kroupi's relock lands
inside the pitch).

Resolution (commits 87d16ee, 069f11f): keep the proven free search path for
every clip; ARBITRATE post-bundle among {free, held-to-anchor-consensus,
held + modest lens} on the only non-self-confirming evidence — the anchor
keypoints. Broken consensuses lose their own arbitration; degenerate line
sets lose to the consensus. Validated on all four regimes.

FINAL MATRIX (manual-click reprojection):
- origi01: med 8.5 px, p90 ~45-55 px, 100% cov — box 3.6-6.4 px (<=0.15 m),
  midfield 11-22 px (~0.25-0.5 m), f294 355 -> 30 px; f0/f134 confirmed BAD
  CLICKS by overlay (camera tracks the painted structures; clicks sit in
  open field). vs-manual centre 6.4 m -> 0.69 m.
- origi02: med 8.5 px / p90 19.7 / max 44.9; centre 0.29 m; 100% cov.
- kroupi: restored best (med 6.3 px, jitter 0.79); gberch byte-identical.
Remaining above-bar: origi01 start (~0.5-1 m) + midfield tail (~0.3-0.5 m),
origi02 f75 area (~0.5 m) — next lever remains zoom-dependent distortion
k(fx) (origi01 zooms 1.5x; one k1k2 cannot serve both ends exactly).
