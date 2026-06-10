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

**origi01 result: 100 % coverage (506/506) for the first time.** Clicks:
box 6.6-10.5 px, midfield 14.5-22.5 px, start 30.8-158.9 px (lens-limited —
degrades toward f0 because the circle-LENS refinement silently skipped:
its ellipse detection band (50 px) can't find the ring under the lens-limited
start cameras, and/or too few ellipse frames intersect the >=2-straight-line
pfl gate). Closing that lens loop is the remaining lever for a fully accurate
start.
