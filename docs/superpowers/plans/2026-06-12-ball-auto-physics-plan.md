# Ball auto-physics rework — implementation plan

Spec: [2026-06-12-ball-auto-physics-design.md](../specs/2026-06-12-ball-auto-physics-design.md)
Branch: `ball-auto-physics`. Each task is TDD (test first), committed separately.

## Task 1 — Player context

`src/utils/ball_player_context.py` + `tests/test_ball_player_context.py`.
- `ContactJointSample(player_id, bone, world_xyz, uv, confidence)` frozen.
- `PlayerContext.load(output_dir, shot_id, camera)` — refined_poses first
  (sync-offset translation), hmr_world fallback; FK contact joints; project.
- API: `joints_at(frame)`, `joints_near_pixel(frame, uv, radius_px)`,
  `joint_world(frame, player_id, bone)`.
- Absorbs `_BoneWorldLookup` duties (stage keeps a thin adapter until Task 5).
- Tests: synthetic NPZ fixtures; precedence; sync offset; FK matches
  `compute_joint_world`; projection round-trip.

## Task 2 — Auto events

`src/utils/ball_auto_events.py` + `tests/test_ball_auto_events.py`.
- `BallEvent(frame, kind, score, player_id, bone, goal_element, debug)`.
- `detect_events(steps, confidences, player_ctx, cameras, goal_geometry, cfg)`
  → touch / bounce / goal_impact / stationary detectors + merge-window
  conflict resolution.
- Tests: synthetic pixel tracks over a known camera — pass chain, shot,
  header, bounce (no player near), keeper save (hand bone), post/net hits,
  rolling-only (no events). Assert detected frames ±2 and kinds; assert no
  spurious events on the rolling scene.

## Task 3 — Auto anchors

`src/utils/ball_auto_anchor.py` + `tests/test_ball_auto_anchor.py`.
- `events_to_anchors(...)`, grounded-span sampling, validation gates
  (off-pitch / reachability / contact-gap), `merge_anchors(manual, auto,
  suppress_radius_frames)`, sidecar write `{shot}_ball_anchors_auto.json`
  (BallAnchorSet schema as-is).
- Tests: mapping per event kind incl. touch_type="shot"; gate rejections;
  manual-wins merge + suppression; round-trip via BallAnchorSet.load.

## Task 4 — Piecewise solver

`src/utils/ball_piecewise_solver.py` + `tests/test_ball_piecewise_solver.py`.
- `solve(nodes, steps, cameras, cfg)` → per-frame world/state/conf +
  FlightSegments + diagnostics.
- Rolling model (endpoint-exact constant-decel LSQ; quadratic fallback),
  ballistic fit (reuse bundle_adjust + _refine_with_magnus; both endpoints
  as knots; split-and-retry on residual-gate failure), bounce restitution
  check, goal-impact pinning, continuity invariant.
- Tests: synthetic analytic scenes → ≤10 cm RMS vs truth; continuity at all
  nodes; restitution flag; split-and-retry recovers two-arc span; never
  emits a segment with residual > gate without an `underconstrained` flag.

## Task 5 — Stage rewiring

`src/stages/ball.py` slimmed: detect loop (+ NEW observations sidecar
`{shot}_ball_observations.json`) → PlayerContext → detect_events →
events_to_anchors → merge with manual → resolve nodes (existing
`_resolve_anchor_world`) → piecewise solver → emit track/keyframes/diag.
Removes: in-file flight-run fitting, ground promotion, quadratic interp,
Phase-2 span pass (superseded). Keeps: C4 ray-snap for manual airborne
anchors, keyframe sidecar, off_screen_flight semantics.
- Update `config/default.yaml` (auto_anchors, physics blocks).
- Update/justify affected stage-level tests (`test_ball_stage*`,
  `test_ball_grounded`, `test_ball_stage_layered`).

## Task 6 — Quality report

`src/pipeline/quality_report.py`: ball section (anchor counts auto/manual,
segment residuals, contact gaps, restitution, goal impacts, missing
fraction, underconstrained spans). Test in `test_quality_report.py`.

## Task 7 — Real-clip validation

- Re-run ball stage on output-kroupi/kroupi01 and output-origi/origi01+02
  with WASB (torch available). Compare before/after: jumps, residuals,
  missing fraction; verify acceptance bars from the spec.
- `tests/test_ball_anchor_accuracy.py` stays green (10 cm harness).
- Full `pytest` run; code review; docs (CLAUDE.md stage table).
