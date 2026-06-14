# Phase 3 — Spin + Rotation Export Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Make the ball carry physically-bounded spin and *visibly rotate/curl* in the viewer and UE — per the spec's Phase 3 (`docs/superpowers/specs/2026-06-12-ball-v2-design.md`). Today spin is preset-seeded Magnus with a 200 rad/s cap, no bounce coupling, and **no exported rotation at all** (the glTF/FBX ball node has position only).

**Architecture:** Five focused tasks, ordered so the visible wins land first and the riskiest (bounce coupling) last:
- **T1** bounded spin: lower the Magnus cap 200→95 rad/s (config + `SolverCfg`).
- **T2** orientation integration: new `src/utils/ball_orientation.py` integrates a per-frame unit quaternion over the dense track (flight from fitted ω; rolling/possessed from rolling-consistent ω=v/r; stationary zero); `BallFrame.quat_wxyz` optional field (schema-additive).
- **T3** rotation export: glTF rotation sampler on the ball node + FBX rotation keys + `omega_rad_s` per flight in `_ball_keyframes.json` (the headline visible deliverable).
- **T4** geometric touch-typing spin seeds (extends `ball_spin_presets.py`).
- **T5** bounce coupling: spin-aware bounce model + joint refit of bounce-adjacent flight segments (the spin-identifiability mechanism — highest risk).

**Tech stack:** numpy/scipy; `bundle_adjust.fit_magnus_trajectory` (has `omega_mag_bound`/`omega_seed`/`omega_axis_fixed`), `ball_physics` (`restitution`, `parabola_end_velocity`, `G_VEC`), `ball_spin_presets`, `gltf_builder`, the Blender FBX script. Determinism throughout. Additive everywhere — piecewise/global solve output unchanged except the new optional `quat_wxyz` and additional spin diag.

**Conventions:** TDD, frozen dataclasses, type annotations, files <800 lines. Tests `.venv311/bin/python -m pytest <files> -q`. Conventional commits, NO attribution footer. Each task: implementer → adversarial review.

**Key facts (mapped):**
- `BallFrame` (frozen, `src/schemas/ball_track.py:21`): `world_xyz, state, confidence, flight_segment_id`. Add `quat_wxyz: tuple[float,float,float,float] | None = None` (LAST, defaulted — backward-compatible). `State = Literal["grounded","flight","occluded","missing"]`.
- `SolverCfg.spin_max_omega_rad_s = 200.0` (`ball_piecewise_solver.py:130`); config `ball.spin.max_omega_rad_s: 200.0`.
- `fit_magnus_trajectory` returns `(p0, v0, omega, mean_resid)`; `omega_mag_bound` caps |ω|.
- FlightSegment.parabola dict carries `spin_axis_world`, `spin_omega_rad_s`, `spin_confidence` (already emitted, null when no spin).
- `gltf_builder`: `_rotmat_to_quat(R) -> (x,y,z,w)`; players already have rotation animation samplers (`gltf_builder.py:472-479`); the ball is a position-only sphere node. glTF quaternion order is **(x,y,z,w)**; BallFrame.quat_wxyz is **(w,x,y,z)** — convert at export.

---

### Task 1: Bounded per-segment spin (200 → 95 rad/s)

**Files:** Modify `config/default.yaml`, `src/utils/ball_piecewise_solver.py`; Test `tests/test_ball_spin_fit.py` (extend).

- [ ] **Step 1: Failing test** — extend `tests/test_ball_spin_fit.py`: a synthetic arc with an unphysically-high true ω (e.g. 150 rad/s) → the accepted fit's |ω| is clamped to ≤ 95 rad/s (assert the cap is enforced via the config/SolverCfg default). And a normal free-kick ω (~10 rev/s ≈ 63 rad/s) is recovered unclamped.
- [ ] **Step 2: FAIL** (current cap 200 allows 150). **Step 3: Implement** — `config/default.yaml` `ball.spin.max_omega_rad_s: 95.0` (update the comment: ~15 rev/s, top of real free-kick range), and `SolverCfg.spin_max_omega_rad_s` default → 95.0. Confirm the value threads to `fit_magnus_trajectory`'s `omega_mag_bound` at the solver call sites (it already reads `spin_max_omega_rad_s` — verify). **Step 4: PASS** + `tests/test_ball_spin_fit.py tests/test_ball_piecewise_solver.py` green. **Step 5: Commit** `feat(ball): bound recovered spin to 95 rad/s (real free-kick range)`.

---

### Task 2: Ball orientation integration

**Files:** Create `src/utils/ball_orientation.py`; Modify `src/schemas/ball_track.py`; Test `tests/test_ball_orientation.py`, `tests/test_ball_track_schema.py` (extend).

- [ ] **Step 1: Schema field** — add `quat_wxyz: tuple[float,float,float,float] | None = None` to `BallFrame` (last field, defaulted). Extend `tests/test_ball_track_schema.py` (or the relevant schema test): a BallFrame without quat round-trips (None); with quat round-trips exactly. Run → implement → green.
- [ ] **Step 2: Failing test** — `tests/test_ball_orientation.py`: 
  - `integrate_orientation(frames, flight_segments, fps, ball_radius) -> dict[int, quat_wxyz]`: 
  - (a) a pure flight segment with known ω → the per-frame quaternion advances by ω·dt about ω's axis (check the relative rotation between consecutive frames matches `axis-angle(ω, dt)` within tol; q0 = identity).
  - (b) a rolling segment moving at speed v on the ground → rolling-consistent ω has magnitude v/ball_radius about the horizontal axis perpendicular to travel (assert |ω_implied| ≈ v/r and the ball's contact-point slip ≈ 0).
  - (c) a stationary span → quaternion constant (ω=0).
  - (d) orientation is CONTINUOUS across a segment boundary even though ω changes discontinuously (no quaternion jump at the node).
  - (e) all quaternions are unit norm.
- [ ] **Step 3: Implement** `src/utils/ball_orientation.py`:
  - Walk frames in order; maintain a running unit quaternion `q` (start identity).
  - Per frame, determine ω: flight → the segment's fitted `spin_omega_rad_s * spin_axis_world` (from FlightSegment.parabola; zero if null); grounded/rolling → `ω = (v/r)` about `up × v_hat` (v from consecutive world positions); stationary/missing → carry q unchanged.
  - Advance `q ← normalize(quat_from_axis_angle(ω, dt) * q)`; store `quat_wxyz` per frame. Missing frames carry the last q (or None — pick None to mirror world_xyz, document).
  - Pure functions; numpy only; deterministic.
- [ ] **Step 4: PASS.** **Step 5: Commit** `feat(ball): per-frame ball orientation integration (quaternion)`.

---

### Task 3: Rotation export (glTF + FBX + keyframes)

**Files:** Modify `src/stages/ball.py` (populate quat on dense frames), `src/utils/gltf_builder.py`, the Blender FBX export script, `src/schemas/ball_keyframes.py` (or wherever keyframes are built); Test `tests/test_gltf_ball_rotation.py`, extend keyframe + stage tests.

- [ ] **Step 1: Stage populates quat** — in `_solve_shot`, after the dense `BallFrame` list is built, call `integrate_orientation(...)` and attach `quat_wxyz` to each frame (rebuild the frozen BallFrames with the quat). Extend a stage test to assert the emitted BallTrack frames carry unit quaternions on flight/grounded frames.
- [ ] **Step 2: glTF rotation sampler** — `gltf_builder.py`: the ball node currently animates translation only. Add a rotation animation sampler/channel driven by per-frame `quat_wxyz` (convert wxyz→xyz,w for glTF). Mirror the player rotation-sampler code (`gltf_builder.py:472-479`). Test `tests/test_gltf_ball_rotation.py`: build a glTF from a BallTrack with quats → the ball node has a rotation channel with one keyframe per frame, quaternions unit-norm, correct xyz,w order (a known wxyz maps to the expected xyz,w).
- [ ] **Step 3: FBX rotation** — the Blender headless export script keys ball rotation from the same quats (find the ball-export code in the Blender script; add rotation keys alongside the existing location keys). Since Blender runs headless, gate the test on Blender availability (skipif) OR assert the script passes the quats through to its job payload (a unit-level check on the data handed to Blender, not a full Blender run).
- [ ] **Step 4: keyframes omega** — `_ball_keyframes.json`: add per-flight `omega_rad_s` (the fitted ω vector) so UE can drive physically-correct curl. Extend the keyframe schema + builder + its test. Keep existing fields (UE `ball_motion.py` preset path keeps working — additive only).
- [ ] **Step 5:** Run `tests/test_gltf_ball_rotation.py tests/test_ball_keyframes_schema.py tests/test_blender_export_*.py tests/test_ball_stage.py -q` (Blender tests may skip). **Commit** `feat(ball): export ball rotation to glTF/FBX + omega in keyframes`.

---

### Task 4: Geometric touch-typing spin seeds

**Files:** Modify `src/utils/ball_spin_presets.py`, solver call sites in `ball_piecewise_solver.py`; Test `tests/test_ball_spin_presets.py` (extend).

- [ ] **Step 1: Failing test** — `tests/test_ball_spin_presets.py`: a new `derive_spin_seed(contact_bone, v_in, v_out) -> omega_seed | None`:
  - foot contact with large horizontal direction change → side-spin about world-z, sign = sign of `cross(v_in_xy, v_out_xy).z` (curl direction); assert axis ≈ ±z and magnitude in a sane seed range.
  - lofted foot contact (exit elevation > 25°) → backspin seed (ω about the horizontal axis ⟂ travel, sign giving backspin).
  - header (contact_bone "head") → ω seed 0.
  - no clear pattern → None.
- [ ] **Step 2: FAIL. Step 3: Implement** `derive_spin_seed` and wire it at touch transitions in the solver's Magnus-refinement path (feed as `omega_seed` with the hinted improvement gate `min_residual_improvement_with_hint`). A manual preset on an anchor still overrides any derived seed (preset path wins). **Step 4: PASS** + `tests/test_ball_spin_presets.py tests/test_ball_spin_fit.py` green. **Step 5: Commit** `feat(ball): geometric touch-typing spin seeds (foot curl/backspin, header zero)`.

---

### Task 5: Bounce coupling (spin-aware bounce + joint refit)

**Files:** Modify `src/utils/ball_physics.py`, `src/utils/ball_piecewise_solver.py`; Test `tests/test_ball_bounce_coupling.py`.

This is the spin-identifiability mechanism and the highest-risk task. If the joint refit proves to iterate without converging (as Phase-2's solver did), land it behind the `spin.bounce_coupling` flag DEFAULT FALSE and document, so the visible wins (T1-T4) ship regardless.

- [ ] **Step 1: Failing test** — `tests/test_ball_bounce_coupling.py`:
  - `bounce(v_in, omega_in, e, mu) -> (v_out, omega_out)` rigid-sphere-on-plane impulse model: normal restitution `e` flips/scales v_z; tangential update from sliding/rolling friction `mu` couples spin↔tangential velocity. Unit tests: pure-normal drop (ω=0) → v_out_z = -e·v_in_z, v_xy unchanged; backspin → tangential velocity reduced/reversed per the impulse model; topspin → tangential increased. (Use the standard tangential impulse model; cite it in a comment.)
  - `fit_coupled_bounce(seg_a_obs, seg_b_obs, cams, fps, cfg) -> (p0, v0, omega0, e, mu, residual)`: synthetic two-arc trajectory with a known spin-coupled bounce → joint fit recovers |ω| within 20% and axis within 25%, and `e`/`mu` within the bounds; on a spin-free synthetic, recovers |ω| < 1 rev/s (no hallucination).
- [ ] **Step 2: FAIL. Step 3: Implement**: `bounce()` in `ball_physics.py`; `fit_coupled_bounce` (LM over `(p0,v0,ω0,e,μ)`, `e∈[restitution_min,restitution_max]`, `μ∈[0,mu_max]`, residual = pixel reprojection over both arcs via the existing Magnus integrator). Wire into the solver: at flight→flight bounce nodes, attempt the joint refit; accept on the same improvement gate, else keep independent fits (today's behavior). Behind `spin.bounce_coupling` (config; default true per spec, but flip to false if real-clip validation shows instability). **Step 4: PASS** + full spin/solver suite green. **Step 5: Commit** `feat(ball): spin-coupled bounce model + joint refit of bounce-adjacent arcs`.

---

### Task 6: Validation

- [ ] **Step 1:** `.venv311/bin/python -m pytest tests/ -q -k "ball or anchor or tracker or spin or orientation or gltf or quality"` green.
- [ ] **Step 2: Synthetic round-trip acceptance** (spec): drag+Magnus trajectories with known ω (5–12 rev/s) + spin-coupled bounces → joint fit recovers |ω| within 20% / axis within 25%; spin-free synthetics report |ω| < 1 rev/s (no hallucination). (These are the T5 unit tests — confirm green.)
- [ ] **Step 3: Real clips** — re-run ball stage (default piecewise solver) on origi + kroupi; confirm: every accepted spin fit satisfies bounds (≤95) + improvement gates; no segment-residual regressions vs the Phase-1.5/2 piecewise output; BallTrack frames carry unit quaternions; a built glTF has a ball rotation channel.
- [ ] **Step 4: Slip check** (spec) — a rolling ball's contact-point slip velocity ≈ 0 (computable from the integrated orientation + world velocity); assert in the orientation/export test.
- [ ] **Step 5:** Append `## Phase 3 validation results` to the design spec; commit `docs: phase 3 spin + rotation export validation`.

---

## Risk register

| Risk | Mitigation | Task |
|---|---|---|
| Spin hallucination on noisy arcs | bound 95 rad/s + improvement gate + spin-free synthetic test asserts |ω|<1 rev/s | T1/T5/T6 |
| Schema break from quat field | optional defaulted field, round-trip test, additive | T2 |
| glTF/FBX quaternion order/handedness | explicit wxyz→xyz,w conversion + known-value test; mirror player sampler | T3 |
| Bounce-coupling joint fit instability (Phase-2-style iteration) | behind `spin.bounce_coupling` flag; T1-T4 ship independently; flip default false if real clips show instability | T5 |
| Orientation discontinuity at nodes | integrate q continuously (ω jumps, q doesn't); continuity test | T2 |
