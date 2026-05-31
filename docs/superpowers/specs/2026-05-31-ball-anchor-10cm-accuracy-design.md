# Ball Anchor 10 cm Accuracy

**Date:** 2026-05-31
**Status:** Draft — pending user review
**Scope:** `src/stages/ball.py`, `src/utils/bundle_adjust.py`,
`src/utils/ball_detector.py` (size sidecar), ball-stage config, ball-stage
tests, plus a promoted validation harness.

## Problem

The goal is for the reconstructed ball to be within **10 cm** of its true
pitch position in 3D, **especially at anchor points where a player touches
the ball**. Anchors are the user's ground truth: a clicked pixel + a state
tag (and, for `player_touch`, a specific player + bone).

A depth-independent measurement (perpendicular distance from the emitted
`world_xyz` to the camera ray through the clicked pixel — the *true* ball
centre must lie on that ray) was run on the three clips with anchors,
re-running the ball stage with current code (anchor-driven; WASB detections
do not affect anchored frames):

| State | gberch | kroupi01 | origi01 | Verdict |
|---|---|---|---|---|
| grounded / kick / bounce / catch / goal_impact | 0.00 m | 0.00 m | 0.00 m | pinned exactly on the clicked ray |
| `player_touch` (ground-touch) | 0.00 m | 0.00 m | 0.00 m | exact (ray-cast at z = ball_radius) |
| `player_touch` (airborne, via SMPL bone) | — | — | up to **2.9 m** off-ray | bone XY drifts off the clicked pixel |
| `airborne_low/mid/high` | 0.3–0.8 m | 0.7–1.5 m | 0.5–5.0 m | fitted arc misses clicked pixels by 20–194 px |

Two root failures, sharing one cause:

1. **Airborne `player_touch` ignores the clicked pixel.** The resolver
   returns the raw SMPL bone world XYZ. Monocular HMR depth drift moves the
   bone XY up to ~2.9 m off the ray the user clicked (origi01 frames 282
   `head`, 310/338/440 `r_foot`). The clicked pixel — the most accurate
   lateral signal we have — is discarded.
2. **Airborne `airborne_low/mid/high` fits miss the clicked pixels.** The
   Phase-2 span fit force-pins `p0` (collapsing depth ambiguity by removing
   3 DOF) and adds coarse z-bucket hinges (1/6/15 m). With `p0` fixed and
   only `v0` free, a 3-DOF arc cannot pass through the user's clicked pixels,
   so the trajectory reprojects 20–194 px (0.1–5 m lateral) away.

**Shared cause:** the pipeline overrides the user's clicked pixel with a
model-derived position (drifting SMPL bone, or an over-pinned parabola) that
leaves the clicked ray.

A separate, non-issue clarification: **origi01's saved track
(`output-origi/ball/origi01_ball_track.json`, dated May 12) is stale** — it
predates the hard-knot pinning, which is why even its grounded/player_touch
anchors were metres off (56/59 anchors > 10 cm). Re-running with current code
drops it to 23/59 > 10 cm, with all remaining failures in the two categories
above. The stale file is not a code problem; it just needs regeneration.

## Goals

- **Contact + ground anchors ≤ 10 cm in 3D**, including airborne
  `player_touch`: the emitted ball lies on the clicked ray (lateral → 0)
  *and* at the correct depth (ground plane for ground states; the contacting
  player's depth for airborne touches).
- **Every anchored frame is laterally ≤ 10 cm** (on the clicked ray) —
  airborne states included. The user's click is authoritative for lateral
  position.
- **Airborne depth genuinely recovered, not punted.** For flight spans
  bracketed by ≥ 2 hard 3D knots, the arc (depth included) is determined
  exactly by gravity + the knots. Spans with < 2 hard knots are monocularly
  under-determined; the design surfaces them as a diagnostic so the user can
  add one bracketing anchor (which makes C2 resolve them exactly), with an
  optional coarse size prior as a best-effort smoothing aid.
- **No regression** on the paths that already measure 0.00 m (ground and
  ground-contact states).
- Existing `tests/test_ball*.py` continue to pass; a new validation harness
  enforces the 10 cm bar on the three real clips.

## Non-goals

- No whole-clip joint bundle adjustment rewrite (Approach C).
- No change to the IMM tracker, WASB detection backend (beyond an optional
  blob-radius side-output), camera stage, or hmr_world stage.
- No change to the `BallFrame` / `FlightSegment` schema or glTF export.
- No ball-shadow extraction.
- 10 cm *depth* for high balls far from camera with **no** player contact
  **and** no bracketing ground/goal knots is best-effort: such spans rely on
  the size prior, which degrades with distance. We do not guarantee 10 cm
  depth there, only ≤ 10 cm lateral.

## Approach (A): ray-faithful + physics-first fit

The user's clicked pixel is hard lateral ground truth; depth comes from
physics (gravity + hard 3D knots), the contacting player, or — only when
those cannot determine it — a ball apparent-size prior. Five components,
each independently testable and gated.

### Physics insight motivating the approach

A free-flight arc with gravity fixed has 6 DOF (`p0`, `v0`). Two hard 3D
knots (e.g. a kick on the ground and a bounce or `goal_impact`) supply 6
constraints, so the arc — **depth at every frame included** — is fully
determined. The clicked airborne pixels between the knots then become a
*consistency check*: "the arc reprojects onto every clicked pixel within
click noise (~2–3 px) while hitting the hard endpoints" is exactly the
objective that yields correct depth, and it is measurable with the harness.

### C1 — Ray-constrained anchor resolution

**Where:** `_resolve_anchor_world` in `src/stages/ball.py`.

For airborne `player_touch` (and any bone-resolved airborne contact), do not
return the raw bone world XYZ. Instead:

1. Compute the bone world position `B` via SMPL FK (as today).
2. Build the clicked-pixel ray `(C, d̂)` (camera centre + undistorted unit
   ray direction).
3. Project `B` onto the ray: `depth = (B − C)·d̂`; emit
   `world = C + depth·d̂`.

Result: lateral = the user's click (→ 0), depth = the player's depth along
the ray. The residual depth error is only the camera-direction component of
the limb extension (small; limb offsets are mostly vertical). When bone
lookup is unavailable, fall back to the existing ray-cast at the
`player_touch` height — already on-ray. Ground/kick/bounce/goal states are
unchanged (they already intersect the ray with a known plane/geometry).

### C2 — Physics-determined Phase-2 arc fit

**Where:** the Phase-2 span fitter in `BallStage._run_shot`
(`fit_parabola_to_image_observations` already supports `knot_frames`,
`z_range_frames`, `p0_fixed`).

Choose the fit conditioning by how many hard 3D knots the span contains
(hard knots = resolved `HARD_KNOT_STATES` anchors: kick, bounce,
goal_impact, grounded, ground-touch player_touch, header/volley/chest,
airborne player_touch via C1):

- **≥ `free_p0_min_hard_knots` (default 2) hard knots:** leave `p0` **free**
  (`p0_fixed=None`); pass all hard knots as `knot_frames`. Gravity + the
  knots determine the 6-DOF arc, depth included. Demote z-buckets to a light
  one-sided hinge (low `z_range_weight`) so they cannot fight the determined
  arc.
- **Exactly 1 hard knot:** pin `p0` to it (today's safe behavior); the span
  is flagged under-constrained by C3 and gets the optional coarse size prior.
- **0 hard knots:** pin `p0` to the first airborne bucket ray-cast (today);
  flagged under-constrained by C3.

Freeing `p0` is gated on ≥ 2 hard knots specifically to prevent the
historical "`p0` drifts metres along the ray" failure: with two knots it
cannot drift.

Note on observation weighting: the Phase-2 fit's observations are *all*
anchor pixels (no WASB), so within Phase-2 the clicks are already
authoritative and the lever is the knot conditioning above. The
IMM/promotion fits use WASB observations; rather than add per-observation
weighting there, anchored-frame **lateral** accuracy is guaranteed
end-to-end by C4's ray-faithful snap, and **depth** by C2's conditioning —
so no `anchor_pixel_weight` knob is needed.

The Phase-2 acceptance check keeps the existing looser sanity bounds
(finite, |v0| ≤ 100, |p0| ≤ 1000) so user-anchored trajectories are trusted.

### C3 — Under-constrained-span diagnostic (+ optional coarse size prior)

Planning surfaced a hard physical limit: at 60–95 m from camera the ball is
only ~5–7 px in diameter, so a sub-pixel apparent-radius error scales depth
by ±10–20 m. An apparent-size prior therefore **cannot** deliver 10 cm depth
for far airborne balls — the only monocular mechanism that recovers correct
airborne depth is C2's physics (gravity + 2 hard knots fully determine the
arc). So C3's primary, high-value job is a **diagnostic**, with the size
prior demoted to an optional coarse aid.

**C3a — Under-constrained-span diagnostic (primary).** During span
processing, count the hard 3D knots bracketing each flight span. When a span
has < 2 hard knots, record it in `output/quality_report.json` (ball section)
and log a warning: "flight span [a, b] has N<2 hard knots — depth is
monocularly under-determined; add a kick/bounce/goal_impact/grounded anchor
to bracket it." This is how airborne depth actually gets pushed to 10 cm: the
user adds one bracketing anchor and C2 then determines the arc exactly.

**C3b — Optional coarse size prior (secondary).** A soft residual in
`fit_parabola_to_image_observations`: a new
`size_depth_frames: dict[int, tuple[np.ndarray, np.ndarray, float]]` mapping
`{rel_idx: (R, t, D_est)}` plus a scalar `size_depth_weight`. The residual is
`size_depth_weight * ((R @ pos_k + t)[2] − D_est)` (camera-frame depth toward
`D_est = f·2r/d_px`). For the 10 cm-at-anchors goal, `d_px` is measured from
the clip image patch at the **clicked anchor pixel** (radial-profile around
the click) — **no detector contract change**. Applies only to < 2-knot
airborne spans; gated by `ball.size_depth_prior.enabled` (default `false`);
clamped to a plausible pixel-diameter range; graceful fallback to today's
bucket midpoint when size is unavailable or implausible. Because the prior is
coarse (metre-scale at distance) it is off by default and is a smoothing aid,
not a 10 cm mechanism.

A WASB heatmap blob-radius side-output (for unanchored-frame smoothing) is
**out of scope** here — it does not affect anchored-frame accuracy, which is
what the goal measures.

### C4 — Ray-faithfulness guarantee

**Where:** a final reconciliation pass in `BallStage._run_shot`, after all
fits and the final hard-knot pin.

For every **airborne-anchored** frame whose emitted `world_xyz` reprojects
farther than `ray_faithful_tolerance_px` (default 3 px) from the clicked
pixel, snap it onto the clicked ray, preserving the fitted along-ray depth:
`world ← C + ((world − C)·d̂)·d̂`. With C2 honoring the clicks this rarely
fires, but it makes "clicks are authoritative for lateral position" a hard
invariant for every anchored frame. Hard-knot states are already pinned and
on-ray, so this pass is a no-op for them.

### C5 — Validation harness + acceptance gate

**Where:** promote the offline harness
(`docs/superpowers/notes/ball-accuracy/`) into the repo as
`tests/test_ball_anchor_accuracy.py` — a pytest acceptance test plus a
`__main__` block that prints the per-state table for ad-hoc measurement.

It re-runs the anchor-driven reconstruction (no-op detector) on gberch,
kroupi01, origi01 and asserts:
- contact + ground anchors (grounded, kick, bounce, catch, goal_impact,
  player_touch) stay ≤ 0.10 m lateral **and** reproject ≤ tolerance;
- per-clip airborne lateral/reproj distributions improve below the current
  baseline (regression guard);
- no anchored frame exceeds the `ray_faithful_tolerance_px` reprojection
  after C4.

The harness is the gate every component (C1–C4) must pass before commit.

## Config (under `ball:`)

```yaml
ball:
  free_p0_min_hard_knots: 2          # free p0 when >= this many hard knots
  ray_faithful_tolerance_px: 3.0     # snap airborne-anchored frame if reproj exceeds
  min_hard_knots_warn: 2             # C3a: flag flight spans with fewer hard knots
  size_depth_prior:
    enabled: false                   # C3b: coarse aid, off by default (not a 10cm mechanism)
    weight: 20.0
    min_pixel_diameter: 3.0
    max_pixel_diameter: 40.0
```

Pitch dimensions and `ball_radius_m` are read from existing config.

## Data flow

```
anchors + camera + refined_poses
        │
        ▼
resolve hard knots  ── C1: airborne player_touch projected onto clicked ray
        │
        ▼
Phase-2 arc fit     ── C2: >=2 hard knots → free p0 (physics determines depth)
                       <2 hard knots → pinned p0 (+ optional C3b size prior)
                       C3a: flag <2-knot spans in quality_report
        │
        ▼
final hard-knot pin (existing) + C4 ray-faithful snap (airborne anchors)
        │
        ▼
BallTrack save + quality_report under-constrained-span diagnostics
        │
        ▼
C5 harness: per-state lateral + reproj acceptance on real clips
```

## File layout

| Path | Status | Purpose |
|---|---|---|
| `src/stages/ball.py` | MODIFY | C1 ray-constrained `_resolve_anchor_world`; C2 fit conditioning; C3a span-knot diagnostic; C4 ray-faithful pass |
| `src/utils/bundle_adjust.py` | MODIFY | C3b `size_depth_frames` soft residual (free-p0 + knot path already present) |
| `src/pipeline/quality_report.py` | MODIFY | C3a: record under-constrained flight spans in the ball section |
| `config/default.yaml` | MODIFY | new `ball:` keys above |
| `tests/test_ball_ray_constrained_touch.py` | NEW | C1 unit tests (bone→ray-depth projection) |
| `tests/test_bundle_adjust_free_p0_knots.py` | NEW | C2 unit tests (≥2 knots + free p0 recovers depth) |
| `tests/test_ball_size_depth_prior.py` | NEW | C3b unit tests (size→depth residual, gating, fallback) |
| `tests/test_ball_underconstrained_diag.py` | NEW | C3a unit tests (span-knot count + quality flag) |
| `tests/test_ball_ray_faithful.py` | NEW | C4 unit tests (snap onto ray, no-op for on-ray points) |
| `tests/test_ball_anchor_accuracy.py` | NEW | C5 acceptance test on gberch/kroupi01/origi01 |
| `docs/superpowers/notes/ball-accuracy/` | EXISTING | measurement scripts (source for the promoted harness) |

## Testing strategy

**Unit, C1** — bone at `(x, y, z)` off the clicked ray → resolver returns a
point on the ray at the bone's along-ray depth; reproj of the result equals
the clicked pixel; bone-unavailable falls back to on-ray height ray-cast.

**Unit, C2** — synthesise a parabola, project to pixels through a moving
camera, supply start + end as hard knots with `p0` free → recovered arc
matches truth (depth included) to < 0.10 m; with only 1 knot, `p0` stays
pinned (the selection rule picks free-vs-pinned by hard-knot count).

**Unit, C3a** — a span with 0/1/2 hard knots → diagnostic flags the 0- and
1-knot spans, not the 2-knot span; the flag appears in the ball quality
section.

**Unit, C3b** — known `d_px` → `D = f·2r/d_px` within tolerance; out-of-range
`d_px` ignored; prior only applied on < 2-knot spans; disabled flag (default)
→ no-op.

**Unit, C4** — off-ray point with reproj > tol → snapped onto ray (reproj →
0, depth preserved); on-ray point → unchanged.

**Integration / acceptance, C5** — on the three real clips: contact+ground
anchors ≤ 0.10 m lateral and reproj ≤ tol; airborne lateral/reproj strictly
below the recorded baseline; print the per-state table for the record.

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Freeing `p0` reintroduces metre-scale along-ray drift | Free `p0` only when ≥ 2 hard knots constrain the arc (`free_p0_min_hard_knots`); otherwise keep today's pinned behavior |
| Far airborne balls (~5–7 px) are monocularly depth-ambiguous | C2 determines depth exactly for ≥2-knot spans; C3a surfaces <2-knot spans so the user adds a bracketing anchor; C3b size prior is a coarse aid only (off by default) |
| Regression on currently-perfect ground/contact paths | C5 harness asserts those stay 0.00 m; C1 and C4 only alter airborne frames; existing `tests/test_ball*` stay green |
| Snapping (C4) creates frame-to-frame kinks at anchors | C2 makes the whole arc honor the clicks first, so C4 rarely fires; when it does, the corrected lateral is ≤ tol so the kink is sub-10 cm |

## Rollout

1. **C5 harness first** — land the measurement + acceptance test recording
   the current baseline, so every later change is gated.
2. **C1** — ray-constrain airborne `player_touch`. Re-run harness: the
   airborne-touch outliers (origi01 282/310/338/440) drop to ≤ 10 cm lateral.
3. **C2** — physics-first fit. Re-run: airborne spans bracketed by ≥ 2 hard
   knots reproject onto their clicked pixels within tolerance; depth
   recovered.
4. **C4** — ray-faithful guarantee. Re-run: no anchored frame exceeds the
   reprojection tolerance.
5. **C3a** — under-constrained-span diagnostic in `quality_report.json`.
   This is the lever that lets the user push remaining airborne depth to
   10 cm by adding a bracketing anchor (which C2 then resolves exactly).
6. **C3b** — optional coarse size prior, only built if there is appetite for
   smoothing <2-knot airborne spans; off by default.
7. Regenerate the stale `origi01` track (and any other clips) with the final
   code.

## Open questions

- None blocking. C3b's necessity is decided empirically by the C5 harness;
  the user pre-approved the gated design and the option to defer the size
  prior if physics (C2) plus the diagnostic (C3a) clear the bar.
