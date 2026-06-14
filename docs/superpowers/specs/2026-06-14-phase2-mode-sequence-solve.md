
---

## Phase 2 validation results (2026-06-14)

Implementation: commits `f34e8ce..05dba11` on `ball-auto-physics` (Tasks 0–8
+ four validation-driven fixes). Full ball/anchor/tracker suite: **447
passed, 2 skipped**. The global solver ships **opt-in** behind
`ball.solver: global`; piecewise remains the default and is unchanged
(Task-0 golden test byte-identical; default-path tests pinned).

### What was built and verified (unit + synthetic)

- `solve_modes` mirrors `solve_piecewise`'s signature and returns a
  byte-shape-identical `SolveResult` (states ⊆ the `BallFrame.State`
  literal; flight frames carry segment ids; possessed→grounded).
- Beam machinery is **correct, bounded, and deterministic**: ~1,200–2,400
  real fits per clip (well under the 20,000 budget), cost-quantized +
  partition-signature tie-break, whole-shot piecewise fallback on
  `BudgetExceeded`/any failure (test-pinned to match piecewise exactly).
- Three adversarial-review-caught defects fixed during the build:
  knot-contaminated flight residual (F5); a Critical scoring defect where
  the beam preferred *gapping out* an impact (OUT_OF_VIEW) over modeling a
  physically-correct flight→roll landing — fixed by physics-aware
  transition costs (restitution penalty only flight↔flight; vertical-kill
  not penalized at landings; OUT_OF_VIEW cost ∝ observed frames); and the
  stage seam passing greedy events instead of permissive candidates.

### Real-clip coverage (the honest finding)

| clip | piecewise cov | global cov | Δ |
|---|---|---|---|
| origi01 (60 manual anchors) | **0.97** | 0.51 | **−0.46** |
| kroupi01 (12 manual anchors) | **0.99** | 0.40 | **−0.59** |
| origi02 (0 anchors, 31 cross-replay fixes) | 0.25 | **0.35** | **+0.10** |

**The global solver improves the anchorless, cross-replay-fix-rich clip**
(origi02 +0.10 — the Phase-1.5 payoff finally appearing: the beam builds
grounded/flight geometry at fix-bearing frames the piecewise anchor-bracket
approach left missing). **But it regresses coverage badly on well-anchored
clips.** Root cause: piecewise draws most of its grounded coverage
(origi01: 273 frames) from its mature **open-span per-frame ground
ray-cast** fallback (pixel-exact, ≈0 residual). The global beam has no
faithful equivalent — its ROLLING mode requires a clean constant-decel fit,
so grounded-but-not-tidily-rolling spans lose to OUT_OF_VIEW. Two
validation-driven attempts to close this (thread resolved anchor worlds
into the beam; add a grounded ray-cast coverage mode) did not recover
parity — the grounded-mode attempt regressed further (guards over-cede) and
was **reverted** (`05dba11`). Anchor-world threading is architecturally
correct and retained (unit-tested), but on real clips the beam's
breakpoints rarely coincide with anchor frames, so it had little effect.

### Verdict

- **Piecewise stays the default** — it is excellent on anchored clips
  (0.97/0.99) and unaffected by this work.
- **The global solver is a correct, deterministic, opt-in alternative** that
  demonstrably helps anchorless multi-impact re-segmentation and consumes
  Phase-1.5 cross-replay fixes, but is **not yet a drop-in replacement**:
  reaching coverage parity on anchored clips requires faithfully porting
  piecewise's open-span grounded coverage (pixel-exact per-frame ray-casts
  with the off-pitch/gap-fill/flight guards) and tuning the grounded↔flight
  guard interactions on real footage — a larger effort than the one-mode
  addition attempted here. This is the concrete next step before global can
  be promoted to default.
- **Net for the operator now:** unchanged default behaviour, plus an opt-in
  solver worth enabling on detector-limited, lightly-anchored clips with
  replay coverage (the origi02 class), where it beats piecewise.

### Carried forward

1. Faithful open-span grounded coverage in the beam (the dominant parity
   gap) with carefully-tuned grounded↔flight guards.
2. Make resolved auto-anchor frames soft breakpoints so anchor depth
   reaches more segments (currently only manual-anchor frames coincide with
   breakpoints).
