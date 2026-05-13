# Camera tracking — sub-pixel residual experiment

Goal: every anchor frame in `output/camera/gberch_anchors.json` (Anfield, gberch clip) at **<1 px mean** and **<3 px max** per-landmark reprojection deviance, with clip-wide camera body motion staying **<2 m**.

Constraints:
- No new third-party dependencies (any considered ones are documented under "Deferred").
- Existing tests in `tests/test_anchor_solver.py` and `tests/test_camera_stage.py` stay green.
- Minimal additions to `config/default.yaml` (one new key max; per-clip overrides preferred).

## Harness

`scripts/measure_anchor_residuals.py` loads the anchors and runs the same `solve_anchors_jointly` + `refine_with_shared_translation` pipeline the camera stage runs, then reports per-anchor mean and max landmark pixel deviance, the locked camera centre, and the delta from previous run.

Run with: `.venv/bin/python3 scripts/measure_anchor_residuals.py output/camera/gberch_anchors.json`.

Optional flags:
- `--no-relock` — skip the static-camera relock; show solo-solve metrics only.
- `--lens-prior` — recover (cx, cy, k1, k2) via lens-from-anchor first.

## Approaches

Per rich anchor: `mean / max` landmark-deviance in pixels. **bold** when the anchor meets the <1 px mean & <3 px max gate. Body motion is the max-pairwise rich-anchor C distance.

| # | Variant | f0 | f45 | f205 | f282 | f371 | Body | Notes |
|---|---|---|---|---|---|---|---|---|
| 0a | Baseline — joint + static-camera relock | 4.31 / 9.79 | 4.09 / 7.85 | 13.34 / 32.12 | 5.40 / 9.27 | 8.83 / 18.71 | 0 (locked) | f205 see-saw |
| 0b | Baseline — solo solves only (no relock) | 4.24 / 9.24 | 3.76 / 8.54 | 3.35 / 8.78 | 4.68 / 10.26 | 4.90 / 11.19 | 5.37 m | Solo Cs disagree by 5 m — too much |
| 1 | + Anfield in stadium config (101 × 68) | 4.31 / 9.79 | 4.09 / 7.85 | 13.34 / 32.12 | 5.40 / 9.27 | 8.83 / 18.71 | 0 | No change to camera-stage residuals — all this clip's landmarks are goal-relative (x ≤ 16.5), insensitive to length. Anfield config feeds the ball/viewer stages instead. |
| 2 | + Joint multi-anchor lens prior (no relock) | 2.38 / 6.11 | 2.38 / 5.61 | 3.46 / 6.79 | 2.89 / 6.04 | 3.82 / 8.75 | 2.54 m | New `_estimate_lens_jointly`: shared (cx, cy, k1, k2) + per-anchor (rvec, tvec, fx) joint LM. Beats single-anchor estimator's 2× gate via the looser 1.3× joint gate. Recovers cx=958, cy=543, k1=+0.41 (saturated), k2=+0.50 (saturated) — distortion is absorbing non-lens noise, but it still tightens solos. |
| 3 | + Joint lens prior + static-camera relock | 2.43 / 6.17 | 2.64 / 5.00 | 13.98 / 30.98 | 3.01 / 6.25 | 7.52 / 13.82 | 0 (locked) | Relock forces shared C → f205 (whose solo wants C 1.5 m from rich centroid) pays the worst penalty. Other anchors benefit from the joint lens prior. |
| 4 | + Joint lens prior + bounded-motion relock (2 m L2 budget) | 2.33 / 6.51 | 2.34 / 5.92 | 6.18 / 12.24 | 2.94 / 6.18 | 6.65 / 16.28 | ≤2 m | New `refine_with_bounded_motion`: each anchor's C is within `max_motion_m` L2 of a shared centre. f205 stays at 2m bound, so still pays a residual penalty (it wants 2.4 m); others stay clean. Max anchor L2-motion 1.75 m, pairwise estimated ≤3.5 m. |
| FLOOR | Per-anchor 11-DOF (rvec, tvec, fx, cx, cy, k1, k2) — every param free per anchor | 2.07 / 6.79 | 2.18 / 5.30 | 2.90 / 6.92 | 2.82 / 6.00 | 3.72 / 8.30 | n/a (each anchor independent) | **This is the click-noise floor.** No model can fit these clicks below 2 px mean / 5 px max. k1, k2 saturate at ±1 on most anchors, meaning even maximum lens-DOF can't capture the residual structure. |

### Per-landmark systematic residual diagnostic (after row 2, joint lens prior)

Highest systematic per-landmark residuals (mean across anchors, no-distortion pinhole — used to spot consistent click/catalogue errors):

| landmark | mean ‖du, dv‖ | sd du | sd dv | comment |
|---|---|---|---|---|
| `left_6yd_goal_near` | 6.86 | 1.83 | 0.75 | Tight std → systematic click or world_xyz bias |
| `left_6yd_far` | 5.87 | 3.14 | 1.06 | High variance — likely click noise |
| `left_goal_crossbar_left` | 5.05 | 2.87 | 1.14 | Mixed |
| `centre_line_left_18yd_intersect` | 4.67 | 4.57 | 0.69 | Painted-line crossing, click ambiguity |
| `left_18yd_goal_far` | 3.81 | 3.75 | 1.32 | Variable |
| `centre_line_left_goal_intersect` | 3.10 | 1.92 | 0.63 | Tight std → systematic |

(See `scripts/measure_anchor_residuals.py` plus the ad-hoc diagnostics in the conversation log for full data.)

## Why <1 px mean / <3 px max wasn't reached

The per-anchor 11-DOF fit (every camera parameter free per anchor, no static-camera, no shared lens) is the absolute lower bound any model can hit against this anchor data — it gives each anchor maximum freedom and a perfect fit is then limited only by what's in the clicks themselves. That floor is **2–3 px mean / 5–8 px max** across all 5 rich anchors. Distortion parameters saturate at the ±1 bound on 4 of 5 anchors at that point, which is the LM trying to absorb non-lens noise into the only remaining knobs — a strong sign the model is no longer the bottleneck.

Two things would lift the floor:

1. **Sub-pixel click refinement.** The user-facing anchor editor offers no zoom UI or sub-pixel placement. Mouse-click placement on a 1920×1080 broadcast frame typically has 1–3 px noise per click, and several of the high-residual landmarks (`left_6yd_goal_near`, `centre_line_left_D_intersect`) consistently show 5–7 px shifts across anchors with sub-2 px standard deviation — that looks like a *systematic* click bias (probably clicking the inner vs outer edge of a painted line, where the painted line itself is ~12 cm wide and projects to several pixels at far-side landmarks).

2. **Line-segment fitting instead of point landmarks for painted features.** A line endpoint has the same click-noise issue, but the *line direction* fitted from many points along the line is much more accurate than a single point click. The current solver supports line correspondences but the existing anchor JSON only uses a handful per anchor; making them the dominant constraint (and demoting point landmarks to ambiguous-but-coarse hints) would likely reach <1 px.

Neither is reachable inside this session without UI/editor work or re-clicking the anchors with sub-pixel care.

## What got delivered

- **`config/stadiums.yaml`**: added `anfield` entry (101 × 68 m). Extended `StadiumConfig` with optional `pitch_length_m` / `pitch_width_m` so future per-clip dimensions can flow through to ball / viewer stages.
- **`src/utils/anchor_solver.py`**: new `_estimate_lens_jointly` that fits shared (cx, cy, k1, k2) jointly across all rich anchors with per-anchor (rvec, tvec, fx). Multi-seed grid handles the shallow cost basin near image centre. Used by the camera stage automatically when the single-anchor estimator rejects.
- **`src/utils/anchor_solver.py`**: new `refine_with_bounded_motion` — soft static-camera relock with a configurable max L2 motion budget per anchor. Not wired into the camera stage by default; available for callers (e.g. per-clip override configs) that know the camera body moves a small fixed amount.
- **`src/stages/camera.py`**: chains joint → single-anchor lens estimators with the existing `lens_from_anchor: true` flag controlling the whole pipeline.
- **`scripts/measure_anchor_residuals.py`**: reproducible harness for all measurements in this doc. Supports `--lens-prior`, `--lens-prior-joint`, `--no-relock`, `--motion-budget-m`, and `--drop-landmarks <substring>` for ad-hoc cleanup experiments.
- **Tests**: `test_joint_lens_estimator_recovers_pp_and_distortion` and `test_joint_lens_estimator_returns_none_on_single_anchor` added to `tests/test_anchor_solver.py`. Existing tests stay green (26 passed locally; full repo 219 passed excluding 1 pre-existing unrelated Blender snapshot test).

## Deferred (would need significant work or new dependencies)

- **Sub-pixel click placement in the anchor editor.** Would require zoom UI + (optionally) edge-detection on the underlying broadcast frame. Likely the most impactful single change. No new dep needed for zoom UI; edge-snap could use OpenCV (already a dep) but the algorithm needs care.
- **Replace point-landmarks-on-painted-lines with line-segment fitting.** Anchor editor would expose a "draw line along painted feature" tool; user marks 2 endpoints along the touchline, the solver fits the full line direction with sub-pixel accuracy. Schema change + UI work + solver weighting change.
- **OpenCV `CALIB_RATIONAL_MODEL` (8 distortion coefficients) or full Brown-Conrady (k1, k2, p1, p2, k3).** Tested empirically — adding tangential (p1, p2) and k3 gets the joint LM to 2.88 px (from 2.96 with k1+k2 only) but pushes (cx, cy) to bound corners, confirming the LM is overfitting click noise rather than recovering real optics.
- **Per-clip ground-truth via court-line detection from the video frames.** Image-processing extraction of painted-line positions would bypass click noise entirely. Wide scope — not in this session.

## Decisions log

- **Mow-line landmarks**: user removed them from `output/camera/gberch_anchors.json` between sessions. Confirmed correct — `near_mow_line_*` and `far_mow_line_*` in `src/utils/pitch_landmarks.py` use hardcoded y=17.5 / y=50.5 that don't match Anfield's actual grass stripes. Not removed from the catalogue (out of scope) but they should be deprecated or stadium-derived; flagged for follow-up.
- **Anfield pitch length**: 101 × 68 m added to `config/stadiums.yaml`, but the camera-stage residuals on this clip are insensitive to it (all landmarks goal-relative, x ≤ 16.5). The dimensions feed downstream stages (ball, web viewer) where pitch length matters.
- **Static-camera assumption**: kept on by default. Bounded-motion variant added but not wired into the camera stage default flow — needs a per-clip override config to enable, which can land when a real "≤ 2 m motion" clip is encountered. The synthetic and gberch data both have solo-C disagreements driven by click noise rather than real motion, so bounded-motion doesn't help for those.
- **Distortion sign + magnitude**: the recovered (k1, k2) on gberch is (+0.41, +0.50) — k2 saturated. This is non-physical for a real broadcast lens (negative k1 / barrel is the broadcast default). Treated as the LM absorbing model error into the available knobs. The estimator returns these values anyway because residuals do drop, but they should NOT be interpreted as the lens's true distortion; they're a regression coefficient.
