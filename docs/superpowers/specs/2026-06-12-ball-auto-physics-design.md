# Ball stage rework: automatic, physically-correct ball tracks — design

Date: 2026-06-12
Status: approved for implementation (autonomous goal session)

## Problem

The ball stage output is not believable without heavy manual work, and is
imperfect even with it. Evidence from the two real reconstruction outputs:

- **origi02 (zero manual anchors — the automatic baseline)**: 96/334 frames
  missing, zero flight segments, 73 m and repeated ~15 m frame-to-frame
  teleports from raw ground projection of airborne balls.
- **origi01 (60 manual anchors)**: accepted flight fits with 33–242 px RMS
  reprojection residuals (one parabola stretched over an 81-frame multi-touch
  sequence), and a 44 m jump at a segment boundary.
- **kroupi01 (12 manual anchors)**: a 95 px-residual accepted fit and a 138 m
  teleport at the clip end.

### Root causes

1. **No automatic depth source.** A monocular pixel gives a ray; the stage
   resolves depth only via the ground plane (wrong when airborne) or manual
   anchors. The pipeline already computes automatic depth sources it never
   uses: player joint positions from `refined_poses` (contacts), goal-frame
   geometry, and gravity curvature across knot-bracketed arcs (C2).
2. **No event model.** Flight segmentation is by IMM pixel-velocity
   posterior. Multi-arc sequences (kick → header → bounce) merge into one
   "segment" that no single parabola can fit; slow lobs are missed entirely.
3. **No continuity.** Segments are fit independently; nothing ties the end
   of one to the start of the next → teleports.
4. **Ground motion is non-physical.** Per-frame ray-casts of smoothed pixels
   (jittery, horizon blow-ups) or a quadratic "bulge" interp between manual
   anchors. No rolling-deceleration model.
5. **Acceptance gates are inverted.** Automatic fits face a 5 px gate (fail →
   silently grounded); user-anchored Phase-2 spans are accepted at *any*
   residual (242 px observed) because "the user is trusted" — but the span
   itself was mis-segmented, so trust bakes the error in.

## Goals

- Physically correct ball track **with zero manual anchors** on typical
  highlight shots; within 10 cm of the real position wherever a depth source
  exists (contacts, ground, goal frame, knot-bracketed arcs).
- Player contacts believable: at a touch the ball meets the contacting limb
  (|ball − bone| ≤ ball radius + tolerance), using refined_poses FK.
- Flight believable, especially shots: one ballistic arc (gravity + drag,
  optional Magnus) per flight, position-continuous at both ends.
- Goal-frame / keeper impacts pinned to post/crossbar/net geometry or the
  keeper's limb.
- Anchors applied **automatically** (camera-stage analogy); manual anchors
  remain the override for tricky cases and always win.
- Quality report gains a `ball` section that surfaces what needs manual help.

## Non-goals

- No new ML models; reuse WASB + existing IMM/fit machinery.
- No dashboard UI for reviewing auto-anchors in this pass (the sidecar file
  is the interface; UI can follow).
- No UE/export schema changes: `BallTrack` and `ball_keyframes.json` formats
  are unchanged.

## Alternatives considered

- **A. Keep patching the current layers** (more gates, tuning): rejected —
  the failures are structural (no events, no continuity, no automatic
  depth); thresholds can't fix them.
- **B. One global bundle adjustment** over the whole shot (spline + contact
  complementarity): most elegant, but high-risk, hard to debug, and opaque
  to the operator. The chosen design is a factored version of it; B remains
  a possible future refinement *inside* the chosen architecture.
- **C. Event-graph piecewise-physical solve fed by auto-anchors (chosen)**:
  reuses the proven manual-anchor machinery (C1–C4 resolution, knot-bracketed
  fits, goal geometry) by generating the anchors automatically, then replaces
  the trajectory-assembly middle of the stage with a solver that enforces
  physics and continuity.

## Architecture

Per shot, the stage becomes six phases. A and the anchor-resolution helpers
survive from today; C, D, E are new; B generalizes `_BoneWorldLookup`.

```
A. Observe    WASB detect + appearance bridge + IMM smooth (existing).
              NEW: persist observations sidecar for re-solves & dashboard.
B. Player     FK contact joints (feet/knees/head/chest/hands) for every
   context    player from refined_poses (fallback hmr_world, sync-offset
              aware) → per-frame {player, bone, world_xyz, uv}.
C. Auto       Detect events from ball pixels + player context + goal
   events     geometry: touches (incl. keeper saves), bounces, goal
              impacts, stationary spans. Score & deduplicate.
D. Auto       Events + sampled high-confidence grounded detections →
   anchors    BallAnchor records, validated (consensus/plausibility),
              written to {shot}_ball_anchors_auto.json. Merged with
              manual anchors at solve time; manual always wins.
E. Physical   Event timeline → piecewise solve:
   solve        rolling segments (friction model, endpoint-exact),
                ballistic segments (parabola/Magnus, BOTH endpoints as
                knots → depth determined), bounce restitution checks,
                goal-impact pinning, per-frame confidence.
F. Report     Dense BallTrack + keyframes sidecar (unchanged formats),
              events debug sidecar, quality-report ball section.
```

### B — Player context (`src/utils/ball_player_context.py`)

- Loads every player track for the shot: `refined_poses/*_refined.npz`
  preferred (reference timeline; translate via `sync_map.offset_for`),
  `hmr_world/{shot}__*_smpl_world.npz` fallback — same precedence as the
  existing `_BoneWorldLookup`, which this module absorbs.
- Contact joints: `l_ankle, r_ankle, l_foot, r_foot, l_knee, r_knee, head,
  neck, l_wrist, r_wrist, l_hand, r_hand, pelvis` (hands matter for
  keepers/throw-ins). FK via `compute_joint_world` (thetas[0] ignored,
  root_R carries world orientation — see SMPL convention memory).
- Projects each joint to pixels with the per-frame camera; exposes
  `joints_near_pixel(frame, uv, radius_px)` and
  `joint_world(frame, player_id, bone)`.
- Pure data + lookups; one frozen dataclass per joint sample.

### C — Auto events (`src/utils/ball_auto_events.py`)

Inputs: IMM steps (pixel track + p_flight), raw detection confidences,
player context, camera, goal geometry, fps. Output: scored
`BallEvent(frame, kind, player_id, bone, goal_element, score, debug)` list.

Detectors (each pure, individually testable):

- **Touch**: ball pixel within `touch_max_px` of a projected joint **and**
  a pixel-velocity direction/magnitude break at that frame (angle change ≥
  `min_direction_change_deg` or speed delta ≥ `min_speed_change_px`,
  measured over ±`window` frames). Bone = nearest joint; ties prefer feet.
  Score from pixel distance, break sharpness, joint confidence.
  A keeper save is just a touch with a hand/arm bone.
- **Bounce**: vertical pixel-velocity sign flip while IMM says flight→
  flight or flight→grounded, no joint within `touch_max_px`, and the
  ground-projected position is plausible. (A bounce is a velocity break
  *without* a player.)
- **Goal impact**: the ball ray at the break frame passes within
  `goal_hit_tolerance_m` of a goal element (reuse `resolve_goal_impact_world`
  candidates' residuals) **and** the pixel speed drops/reverses sharply.
  Element = best-residual candidate.
- **Stationary**: runs of near-zero pixel velocity with high detection
  confidence (free kicks, kick-offs) → strong grounded evidence.

Conflict resolution: events within `merge_window` frames compete by score;
one survivor per window. Goal impact beats touch beats bounce on ties
(specific beats generic).

### D — Auto anchors (`src/utils/ball_auto_anchor.py`)

- Map events → `BallAnchor`: touch → `player_touch` (+`touch_type="shot"`
  when post-touch speed ≥ `shot_speed_px` toward a goal), bounce → `bounce`,
  goal impact → `goal_impact` + element, stationary span → `grounded` at the
  span midpoint.
- **Grounded sampling** (camera-keyframe analogy): in spans where IMM is
  confidently grounded and detection confidence ≥ `grounded_min_conf`, emit
  `grounded` anchors every `grounded_interval` frames.
- **Validation gates** before writing (camera MAD-consensus analogy):
  resolve every candidate to world via the existing resolver; reject
  candidates that are off-pitch, kinematically unreachable from their
  neighbours (speed cap), or whose touch joint is > `contact_max_gap_m`
  from the ball ray. Drop the whole auto set for a span if fewer than
  `min_anchors_per_span` survive (prevents garbage-in).
- Output `output/ball/{shot}_ball_anchors_auto.json` (existing
  `BallAnchorSet` schema — no schema change; provenance is the filename).
- **Merge policy** (in the stage): manual anchors win; any auto anchor
  within `suppress_radius_frames` of a manual anchor is dropped; the merged
  dict feeds the solver exactly like today's manual dict, so C1–C4
  resolution, keyframe sidecar, and the dashboard preview flow are unchanged.

### E — Physical solve (`src/utils/ball_piecewise_solver.py`)

Replaces the IMM flight-run fitting, ground promotion, ground-level
quadratic interp, and Phase-2 span fitting inside `_run_shot`.

- **Timeline**: merged anchors resolved to world (existing
  `_resolve_anchor_world`) become nodes. Between consecutive nodes a
  segment is classified **rolling** (both nodes ground-level and no flight
  evidence between) or **ballistic** (any airborne anchor, p_flight run, or
  an event pair that implies flight, e.g. kick→bounce).
- **Rolling segment**: constant-deceleration rolling model on the ground
  plane, fit to the ray-cast XY observations between nodes and constrained
  to pass through both endpoints exactly. Falls back to today's bounded
  quadratic when observations are too sparse. z = ball radius throughout.
- **Ballistic segment**: `fit_parabola_to_image_observations` with
  p0 = start node (or free with both nodes as knots when ≥ 2 hard knots —
  C2, now the *common* case because auto-touches bracket arcs), end node as
  a knot, in-between WASB pixels as observations, drag/Magnus refinement
  via the existing `_refine_with_magnus` (spin presets still honoured).
  Hard residual gate stays, but failure now triggers **split-and-retry**:
  insert the strongest unused event inside the span and refit the halves —
  never silently accept a 100+ px fit and never silently demote to
  grounded.
- **Continuity invariant**: every segment starts at its start node and ends
  at its end node by construction → no teleports anywhere a node exists.
- **Bounce nodes**: check restitution `v_out_z ≈ −e·v_in_z`, e ∈
  [`restitution_min`, `restitution_max`] (grass ≈ 0.5–0.85) and horizontal
  velocity continuity within tolerance; violations are flagged in the
  quality report (not silently "fixed").
- **Touch nodes**: position continuity; velocity may jump (impulse).
  Contact believability metric recorded: |ball − bone| at the frame.
- **Goal impact nodes**: position pinned to element geometry; incoming arc
  terminates there. Post/crossbar with a following segment → outgoing arc
  starts at the impact point (rebound); net → trajectory ends (drop handled
  as a short rolling/settle segment if detections continue).
- **Unanchored leading/trailing spans**: solved as today (IMM runs +
  plausibility) but with the nearest node as a boundary knot when adjacent.
- Output: dense per-frame world + state + confidence + `FlightSegment`s
  (schema unchanged), plus solver diagnostics.

### F — Reporting

- `quality_report.json` gains a `ball` section per shot: anchor counts
  (auto/manual by state), per-segment residuals, contact-gap stats, bounce
  restitution values, goal impacts, underconstrained spans (existing C3a),
  missing-frame fraction. This is the "where do I need to add a manual
  anchor" surface, mirroring camera confidence.
- `{shot}_ball_events.json` debug sidecar (events with scores and the
  evidence behind them) for the dashboard later.

## Stage ordering & compatibility

- `src/pipeline/runner.py` already runs `refined_poses` before `ball`; no
  ordering change. CLAUDE.md's stage table will be updated to show it.
- `BallTrack`, `ball_keyframes.json`, manual `{shot}_ball_anchors.json`,
  dashboard endpoints and preview flow: unchanged.
- `ball.py` shrinks: orchestration only (detect loop, merge anchors, call
  solver, emit artifacts); current in-file passes move into the new modules.

## Config (new keys under `ball:`)

```yaml
auto_anchors:
  enabled: true
  touch_max_px: 25.0
  min_direction_change_deg: 25.0
  min_speed_change_px: 4.0
  event_window_frames: 3
  merge_window_frames: 4
  shot_speed_px: 12.0
  grounded_interval: 25
  grounded_min_conf: 0.55
  contact_max_gap_m: 0.6
  suppress_radius_frames: 3
  goal_hit_tolerance_m: 0.5
physics:
  restitution_min: 0.5
  restitution_max: 0.85
  rolling_decel_max_m_s2: 6.0
  max_arc_seconds: 2.5
```

## Error handling

- Missing refined_poses/hmr_world → player context degrades to empty; auto
  touch/goal detection disabled with a logged warning; bounce/grounded
  anchors still generated (pure ball+camera evidence).
- Auto-anchor generation failure is non-fatal: stage falls back to manual
  anchors + unanchored solve (today's behaviour floor).
- Solver split-and-retry bounded (`max_splits_per_span = 3`); exhaustion →
  span flagged underconstrained, endpoints still honoured, interior linear
  along physics prior (never a silent high-residual fit).

## Testing

- TDD per module with synthetic ground truth (camera + analytic
  trajectories), mirroring `test_ball_flight.py` style:
  - player context: FK + sync-offset + fallback precedence;
  - auto events: pass, shot-on-goal, header chain, bounce, keeper save,
    goal impacts (post/bar/net), no-event rolling — precision *and* recall
    assertions;
  - auto anchors: mapping, gates, merge/suppression policy, sidecar IO;
  - solver: ≤ 10 cm RMS world error on synthetic scenes vs. analytic truth;
    continuity at every node; restitution flagging; split-and-retry.
- Existing suites are the regression contract: anchor-accuracy harness
  (3 clips, currently green) must stay green; ball schema/keyframe tests
  unchanged; `test_ball_stage_layered`/`_anchors` updated only where the new
  solver intentionally supersedes old assembly semantics (documented in the
  diff).
- Real-clip validation: re-run kroupi01/origi01/origi02 (torch 2.11 in
  venv → WASB runs locally). Acceptance: origi02 produces a continuous
  track with flight segments and no >2 m frame jumps; origi01/kroupi
  segment residuals ≤ 8 px; no teleports at segment boundaries; manual
  anchors still within 10 cm.
