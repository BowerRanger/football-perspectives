# Foot-Contact-Aware Locomotion — Design

**Date:** 2026-09-02
**Goal:** Eliminate foot-floor clipping and unnatural foot sliding in reconstructed
player animations so the output reads as a near-perfect recreation of the real
movement. Test bed: `output/` (gberch — near-perfect camera tracking, players
P001–P003 have full `hmr_world` sidecars).

## 1. Problem & baseline evidence

Measured on gberch refined poses (FK with beta-adjusted rest joints, 25 fps):

| Metric | P001 | P002 | P003 | Target |
|---|---|---|---|---|
| Foot speed while foot low (<10 cm), mean m/s | 3.23 | 2.10 | 2.45 | < 0.3 in stance |
| Foot speed while low, p95 m/s | 7.8 | 6.1 | 5.5 | < 0.8 in stance |
| Lower-foot z p95 (m) | 0.030 | 0.025 | 0.031 | flight phases preserved (≫0.05 when sprinting) |
| % frames lower foot < 0 (joint) | 0.53 | 0.47 | 0.93 | ~0 (incl. sole offset) |
| Max joint penetration (cm) | 1.10 | 1.27 | 1.20 | 0 |

Two artifacts, one root cause each:

- **Sliding (the big one).** `hmr_world` re-derives `root_t` *every frame* from
  the **ankle-midpoint pixel** using a **canonical rest-pose** root→ankle offset
  (`_ANKLE_IN_ROOT`, lateral zeroed). Nothing in the pipeline knows which foot is
  planted, so no foot is ever stationary: FK feet inherit the root's full
  translation and skate at 2–3 m/s while "on the ground". Additionally, projecting
  the midpoint of one grounded + one airborne ankle to the ground plane injects a
  stride-frequency depth oscillation into `root_t`.
- **Clipping + glued-to-floor look.** `refined_poses._ground_snap` shifts
  `root_t.z` so the **lower foot joint sits at exactly 0.02 m on every frame**
  within 0.30 m of the ground — including genuine flight phases (running has
  aerial moments; p95 lower-foot z after snap is ~0.03 m — everything is
  flattened). Then `_smooth_track` Savgol-smooths `root_t` (z included) *after*
  the snap, re-introducing up to ~1.3 cm of joint penetration. The mesh sole
  extends below the foot joint, so a joint pinned at 2 cm still visibly clips.

## 2. Design overview

Four components. A is the measurement foundation; B–D are the fix.

```
hmr_world (extraction)                     refined_poses (cleanup)
─────────────────────                      ──────────────────────
GVHMR → thetas/root_R/kp2d                 flip-reject → lean-reduce
      │                                          │
  [B] per-foot contact detection            XY cleanup passes (Hampel /
      (ray-cast kp2d ankles + FK)           velocity / jitter consensus — as today)
      │                                          │
  [C] contact-aware anchoring:              temporal smoothing (root_R/root_t/thetas)
      stance-pinned posed-FK root solve          │
      + legacy anchor as carrier            [D] foot-lock finale (NEW, always LAST):
      │                                         contact-aware z + stance pin IK
  writes *_smpl_world.npz                       + penetration guard
      + *_foot_contacts.json (NEW)              (nothing smooths after this)
```

**[A] Eval harness** (`src/utils/foot_quality.py` + `scripts/eval_foot_quality.py`)
quantifies penetration, flight preservation, stance skate, and image fidelity
before/after every change.

**[B] Contact detection** (`src/utils/foot_contact.py`). Per-foot, camera-grounded:
ray-cast **each ankle pixel separately** (not the midpoint) to the ground plane
per frame → a world-track per foot. During true stance that ray-cast position is
physically stationary; during swing it sweeps at >2× body speed. Stance =
ray-cast world speed below threshold with hysteresis, min span length (≥4
frames), ankle confidence ≥ 0.3, and a pixel-noise-adaptive floor (convert
local px→m scale at the ground point so far/small players don't produce
spurious detections). Secondary gate: the stance foot must be the lower FK foot
(posed, root-relative). Output: per-foot spans + robust pin position (median of
in-span ray-casts). Degrades gracefully: no confident stance ⇒ no pin ⇒
behavior falls back to today's anchor.

**[C] Contact-aware anchoring** (`hmr_world`, mode `contact`, legacy mode
`ankle_mid` kept). Two changes to the root solve:

1. **Posed-FK offsets.** Replace the canonical `_ANKLE_IN_ROOT` with the actual
   per-frame FK root→ankle offset (beta-adjusted rest joints + smoothed thetas)
   of the anchoring foot. This alone fixes the float-vs-snap tug of war: the
   root lands so the *posed* foot touches the ground.
2. **Stance pinning via a correction channel.** Compute a dense per-frame
   carrier path `root_carrier(t)`: today's ankle-midpoint ray-cast anchor, but
   using the posed-FK mid-ankle offset from change 1 (dense, and never worse
   than baseline). On stance frames of foot F, compute
   `root_stance(t) = pin_F − root_R(t) @ fk_offset_F(t)`; during double support,
   average the two implied roots (confidence-weighted). Define
   `δ(t) = root_stance(t) − root_carrier(t)` on constrained frames, interpolate δ
   smoothly (monotone cubic) across unconstrained frames (flight, occlusion,
   keeper dives), clamp |δ| ≤ 0.5 m, and output
   `root_t = root_carrier + δ`. Flight arcs keep the carrier's posed-FK z; long
   no-contact spans decay δ toward 0 (pure carrier behavior).

`hmr_world` persists a sidecar `{shot}__{pid}_foot_contacts.json` (spans, pins,
per-frame contact flags) so downstream stages don't re-derive contacts from
scratch. A new `scripts/reanchor_hmr_world.py` re-runs ONLY the anchoring math
from saved sidecars (kp2d + camera + thetas/root_R in the npz) — this is what
makes extraction changes fully testable on this Mac without re-running GVHMR.

**[D] refined_poses foot-lock finale.** Reorder so nothing degrades foot
placement after it is enforced:

1. Replace the blanket `_ground_snap` with a **contact-aware z pass**: snap only
   frames that are in contact (sidecar mask when present, else FK-derived:
   foot low AND slow); airborne frames keep their FK-consistent z.
2. Run the existing XY cleanup + jitter consensus + temporal smoothing as today
   (they operate on root_t and remain valuable).
3. **Foot-lock IK finale (new, last pass).** For each stance span on the final
   smoothed track: re-pin the foot to the span median; apply per-frame root XY/Z
   micro-correction (low-passed) plus **two-bone leg IK** (hip–knee–ankle) so the
   stance foot lands exactly on its pin; ease in/out over ~3 frames at span
   edges; clamp IK joint deltas (≤10°) and residual correction (≤15 cm) — skip
   the pin if clamps would be exceeded (report it instead of mangling).
4. **Penetration guard (very last).** Per frame, if the lowest foot joint minus
   `sole_clearance_m` (default 0.025) dips below 0, raise `root_t.z` by the
   deficit using a windowed max + short ease so the correction doesn't jitter.
   Never lowers, never smoothed afterwards.

## 3. Interfaces

- `src/utils/foot_contact.py`
  - `detect_contacts(kp2d, K, R, t, thetas, root_R, betas, fps, cfg) -> FootContacts`
  - `FootContacts`: frozen dataclass; per-foot `spans: [(start_i, end_i, pin_xyz)]`,
    per-frame `in_contact: (N, 2) bool`, `quality: (N, 2) float`. JSON round-trip
    (`to_json`/`from_json`) for the sidecar.
- `src/utils/foot_lock.py`
  - `solve_root_with_pins(root_legacy, root_R, thetas, betas, contacts, cfg) -> root_t`  (used by [C])
  - `lock_feet_ik(thetas, root_R, root_t, betas, contacts, cfg) -> (thetas, root_t, stats)`  (used by [D])
  - `penetration_guard(thetas, root_R, root_t, betas, sole_clearance_m) -> (root_t, stats)`
- `src/utils/foot_quality.py`
  - `foot_quality_metrics(frames, betas, thetas, root_R, root_t, fps, contacts=None) -> dict`
    (penetration stats, lower-foot z distribution, stance skate, span drift,
    contact ratio; optional kp2d+camera → ankle reprojection px error)
- `scripts/eval_foot_quality.py --output <dir> [--players P001,…] [--json out]` —
  table + JSON; runs on both `hmr_world` and `refined_poses` artefacts.
- `scripts/reanchor_hmr_world.py --output <dir> [--shot gberch] [--in-place|--suffix]` —
  recompute contacts + root_t from saved sidecars; writes npz + contacts sidecar.

All new numerics are numpy/scipy only (refined_poses' light-venv contract holds).
No schema changes to `SmplWorldTrack`/`RefinedPose`; contacts travel as a sidecar.

## 4. Config additions (`config/default.yaml`)

```yaml
hmr_world:
  anchor_mode: contact        # contact | ankle_mid (legacy)
  contact:
    speed_enter_m_s: 0.6      # ray-cast world speed to enter stance
    speed_exit_m_s: 1.2       # hysteresis exit
    min_span_frames: 4
    max_pin_spread_m: 0.25    # span ray-cast spread gate (kicks/false stances)
    px_noise: 2.0             # assumed kp2d jitter, scaled by local px→m
    max_correction_m: 0.5     # |δ| clamp
    decay_s: 0.6              # δ decay toward legacy on long no-contact spans
refined_poses:
  ground_snap_target_z: 0.02  # existing behavior, now contact-gated
  foot_lock:
    enabled: true
    ik_max_joint_delta_deg: 10.0
    max_residual_correction_m: 0.15
    edge_ease_frames: 3
    sole_clearance_m: 0.025
```

## 5. Error handling & fallbacks

- **Occluded/low-conf ankles:** frames below conf 0.3 never enter stance; the δ
  channel interpolates across them (existing hold-last legacy behavior is the
  carrier). — flag, don't substitute.
- **No stance found (keeper dive, player on ground, tiny far player):** δ ≡ 0 ⇒
  pure carrier path (today's anchor with the posed-FK offset fix); exact
  bit-parity with today is available via `anchor_mode: ankle_mid`. The [D]
  guard still prevents clipping either way.
- **Kicks:** ball-strike deceleration can mimic a brief stance — filtered by
  `min_span_frames` + `max_pin_spread_m`; the ball stage's touch events are NOT
  consumed here (no circular dependency).
- **IK infeasible:** clamps exceeded ⇒ skip pin for that span, count in stats,
  surface in `refined_poses_summary.json` and `quality_report.json`.
- **Operator data:** none touched — no interaction with manual ball anchors or
  sync offsets.

## 6. Testing & validation

- **Unit (TDD, all local):** synthetic gait generator (analytic walk/run cycles
  with known stance spans) → exact-expectation tests for contact detection,
  pin solve, IK landing, penetration guard, and metric computation. Edge cases:
  double support, flight, occlusion holes, single-frame spikes, kick-like
  deceleration, IK clamp overflow.
- **Integration:** `tests/test_refined_poses_stage.py` additions — stage run on
  fixture npz with foot_lock on/off; sidecar round-trip; legacy `ankle_mid`
  mode bit-parity test on a fixture.
- **End-to-end on gberch (local):** `scripts/reanchor_hmr_world.py` +
  re-run refined_poses; `scripts/eval_foot_quality.py` before/after.
  Acceptance:
  - stance-foot skate mean < 0.3 m/s, p95 < 0.8 m/s (baseline 2.1–3.2 / 5.5–7.8)
  - sole-proxy penetration (joint z − sole_clearance < 0): < 0.5 % of frames, max < 1 cm
  - flight preserved: lower-foot z p95 ≥ 0.10 m for P001 (sprint phases)
  - ankle reprojection error vs kp2d within +10 % of baseline (stay faithful
    to image evidence)
  - full default pytest suite green (minus the two known-failing tests on main)
- **Cross-clip sanity:** run the eval harness read-only on `output-origi`,
  `output-japan`, `output-kroupi` where sidecars exist; no acceptance gates
  (different camera quality), report-only.
- **Deferred to GPU box:** fresh GVHMR runs with `anchor_mode: contact`, and
  ball-stage touch-recall revalidation (ball touch attribution consumes
  refined_poses FK — expect neutral-to-better, but must be re-measured).

## 7. Non-goals

- No GVHMR/vendored-code changes, no detector work, no camera-stage changes.
- No SMPL toe articulation or mesh-level (vertex) contact — joint + sole-offset
  proxy only.
- No multi-shot fusion redesign (the highest-confidence pick stays).
- No UE-side changes; export/render consume the same npz schema.

## 8. Risks

| Risk | Mitigation |
|---|---|
| kp2d noise on far players → false/missed stances | px-noise-adaptive thresholds; span gates; fallback to legacy anchor |
| IK visibly distorts legs | hard clamps + skip-and-report; ease at span edges |
| δ channel fights XY jitter-consensus passes | δ applied in hmr_world (before), final pin in refined_poses (after) — consensus passes see a consistent track and the finale re-enforces pins |
| Ball touch attribution shifts | flagged for GPU-box revalidation before merging to a release path |
| Flight-phase z now unconstrained by snap | posed-FK z via legacy carrier + δ interp; guard prevents penetration |
