# Phase 2 — Global Mode-Sequence Solve Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Replace the greedy event-classification + per-span dispatch in the ball solver with a left-to-right **beam search over timeline partitions** (modes: rolling, flight, possessed, stationary, out_of_view), so multi-impact sequences segment correctly and Phase-1.5 triangulated fixes become first-class flight evidence. Ships behind `ball.solver: piecewise|global`, piecewise remaining the default until validated.

**Architecture:** New `src/utils/ball_mode_search.py` exposes `solve_modes(...)` with the **exact** keyword signature and `SolveResult` return of `solve_piecewise` (the call-site swap is one line in `_solve_shot`). It reuses the existing fitters (`_fit_arc`, `_rolling_span`, `fit_parabola_to_image_observations`, `fit_rolling_segment`, `two_knot_arc`) via a real `_Solver` instance, and reuses the output assembler — which Task 0 first **extracts from the `solve()` closure into importable free functions** (the critique's F1/F3/F12/F15). Beam segments are fit **endpoint-free from interior evidence** (pixels + world_fixes + ground plane), with continuity enforced as a *scoring* term, not a hard endpoint knot — this keeps the fit cache soundly frame-keyed (resolves F7) while allowing free-floating breakpoints. Manual anchors hard-pin (pruning, not penalty).

**Tech stack:** numpy/scipy; reuses `ball_piecewise_solver`, `ball_physics`, `bundle_adjust`, `ball_player_context`, `ball_auto_events`. Light-venv testable; detectors/video never touched (solve pass only).

**Conventions:** TDD, frozen dataclasses, type annotations, files <800 lines. Tests: `.venv311/bin/python -m pytest <files> -q`. Conventional commits, NO attribution footer. Determinism: no `random`/`Date`; ties broken by quantized `(cost, frame, mode_enum)`.

**Critical design resolutions (from the adversarial design review — honor these):**
- **F1/F3/F12/F15 (Task 0):** the output assembler (`_commit_span`), node-authority block, restitution-flag block, and open-span leading/trailing commit are inline in `_Solver.solve()` (ball_piecewise_solver.py:1112-1255). Extract to importable free functions BEFORE building the renderer; gate with a golden test that piecewise output is byte-identical pre/post.
- **F12/F13:** `BallFrame.State = Literal["grounded","flight","occluded","missing"]` (schemas/ball_track.py:17). New modes map onto EXISTING states: `possessed→grounded`, `stationary→grounded`, `out_of_view→missing`. The additive part is `diagnostics["segments"][].kind` (free dict) only. Possessed never emits `flight`.
- **F2:** the global path does NOT call `_audit_auto_nodes` (which itself calls the greedy `_solve_span`). Manual anchors become hard breakpoints; auto-anchor frames + events become *candidate* breakpoints the beam ranks. This sidesteps the audit's self-reference entirely.
- **F7:** beam flight/rolling segments fit endpoint-free from interior evidence; continuity is a scoring term. Cache key = `(fa, fb, mode, player_id, fa_anchor, fb_anchor, wfixes_sig)` — anchor worlds are deterministic from the frame, so a bool suffices. Sound.
- **F8:** budget backstop = **whole-shot fallback to `solve_piecewise`**, never a half-LM/half-analytic beam result.
- **F5:** residual cost in **raw px** for cross-mode comparison; gates only flag `underconstrained`.

---

## File structure

| File | Action | Responsibility |
|---|---|---|
| `src/utils/ball_piecewise_solver.py` | Modify (T0) | Extract `commit_span`/`apply_node_authority`/`apply_restitution_flags`/open-span commit to free functions; extend kind→state map |
| `src/utils/ball_auto_events.py` | Modify (T1) | `detect_event_candidates` permissive profile (soft-NMS / top-K per window) |
| `src/utils/ball_mode_search.py` | Create (T2-T7) | Mode enum, dataclasses, cfg, fitters-reuse layer, scoring, beam loop, render |
| `src/stages/ball.py` | Modify (T8) | `ball.solver` seam; whole-shot piecewise fallback wiring |
| `config/default.yaml` | Modify (T8) | `ball.solver` + `ball.mode_search.*` |
| `src/pipeline/quality_report.py` | Modify (T7) | surface `mode_search` + `out_of_view_spans` diag |
| `tests/test_*` | Create | per-task (see tasks) |

---

### Task 0 (BLOCKING): Extract the output assembler to reusable free functions

The renderer must call the SAME assembly logic `solve_piecewise` uses, but it is inline in `_Solver.solve()`. Extract it with zero behavior change, proven by a golden test.

**Files:** Modify `src/utils/ball_piecewise_solver.py`; Test `tests/test_solver_commit_extraction.py`.

- [ ] **Step 1: Golden test FIRST (pin current piecewise output).** Create `tests/test_solver_commit_extraction.py`: build 3 synthetic scenarios with `tests/fixtures/ball_synthetic.py` (a rolling-only roll, a node-bracketed arc, an open-span trailing flight). For each, call `solve_piecewise(...)` and snapshot `world_by_frame` (rounded 6dp), `state_by_frame`, `[(s.frame_range, round(s.fit_residual_px,4)) for s in flight_segments]`, and `diagnostics["segments"]`. Store the snapshots inline as expected dicts. Run → PASS (this pins current behavior before refactor).

- [ ] **Step 2: Read `_Solver.solve()` (ball_piecewise_solver.py:1090-1255) fully.** Identify the inline blocks: `_commit_span` closure (~1125-1172, closes over `world_by_frame`, `state_by_frame`, `segments`, `diagnostics`, `arcs_by_bound`), the node-authority block (~1202-1224), the restitution-flag block (~1227-1248), and the leading/trailing open-span commit (~1184-1200, note the `n_frames` vs `n_frames-1` boundary — F15).

- [ ] **Step 3: Extract free functions.** Add module-level:

```python
@dataclass
class _CommitState:
    world_by_frame: dict[int, tuple[np.ndarray, float]]
    state_by_frame: dict[int, str]
    segments: list["FlightSegment"]
    diagnostics: dict
    arcs_by_bound: dict

# kind -> per-frame state. New Phase-2 kinds map onto existing BallFrame.State.
_KIND_STATE = {"rolling": "grounded", "ballistic": "flight",
               "possessed": "grounded", "stationary": "grounded",
               "out_of_view": "missing"}

def commit_span(st: _CommitState, outcome: "_SpanOutcome", fa: int, fb: int,
                conf: float, n_frames: int) -> None:
    """Assemble one span's outcome into the dense track + segments + diag.
    Identical logic to the former _Solver.solve() closure."""
    # ... moved body; for outcome.kind in _KIND_STATE use the mapped state,
    # except the legacy 'open' branch keeps its per-arc membership rule.
```

Move node-authority and restitution blocks to `apply_node_authority(st, nodes, ...)` and `apply_restitution_flags(st, ...)` likewise. `_Solver.solve()` now builds a `_CommitState` and calls these. Keep all signatures internal-stable.

- [ ] **Step 4: Run golden test → must still PASS byte-identical**, plus `tests/test_ball_piecewise_solver.py tests/test_ball_grounded.py tests/test_ball_flight.py tests/test_ball_piecewise_fixes.py tests/test_ball_stage.py -q`. Any diff = the extraction changed behavior; fix until identical.

- [ ] **Step 5: Commit** `refactor(ball): extract solver output assembler to reusable free functions`.

---

### Task 1: Permissive event candidates

**Files:** Modify `src/utils/ball_auto_events.py`; Test `tests/test_ball_event_candidates.py`.

Add `detect_event_candidates(...)` (additive — does not change `detect_events`): same detection, but a **soft-NMS / top-K-per-merge-window** policy instead of the greedy merge, returning every candidate `BallEvent` with its score (not just survivors). Add a `profile: str = "default"|"permissive"` knob lowering `min_event_score`/`min_speed_change_px` thresholds. Preserve the velocity-break feature fields. Boundary/synthetic frames are NOT events (F17).

- [ ] **Step 1: Failing test** — `tests/test_ball_event_candidates.py`: synthetic track with two touches 3 frames apart → `detect_event_candidates(profile="permissive")` returns BOTH (greedy `detect_events` merges them); a decoy low-score break is retained with its low score (not dropped); scores are in [0,1].
- [ ] **Step 2: FAIL.** **Step 3: Implement** (top-K per `merge_window_frames`, not full suppression). **Step 4:** run + `tests/test_ball_auto_events.py` green. **Step 5: Commit** `feat(ball): permissive event-candidate profile for mode search`.

---

### Task 2: Mode-search scaffolding

**Files:** Create `src/utils/ball_mode_search.py`; Test `tests/test_ball_mode_search_scaffold.py`.

Define `Mode(Enum)` (ROLLING=0, FLIGHT=1, POSSESSED=2, STATIONARY=3, OUT_OF_VIEW=4 — ordering is the tie-break key), frozen `Breakpoint`/`Segment`/`Hypothesis` dataclasses (per the design §1.1, with `Breakpoint.event_score=0` for synthetic boundaries — F17), `ModeSearchCfg` (frozen: `beam_width=8`, `segment_cost_constant=6.0`, `unexplained_break_penalty=10.0`, `ignored_event_penalty=8.0`, `out_of_view_frame_penalty=1.5`, `possessed_tether_px=40.0`, `velocity_discontinuity_weight=2.0`, `max_segment_fit_calls=20000`), and `_mode_search_cfg(cfg)` mapping `ball.mode_search.*`.

- [ ] **Step 1: Failing test** — dataclass construction + enum order + cfg mapping with overrides. **Step 2: FAIL. Step 3: Implement. Step 4: PASS. Step 5: Commit** `feat(ball): mode-search scaffolding (modes, dataclasses, config)`.

---

### Task 3: Fitter-reuse layer + sound segment-fit cache

**Files:** Modify `src/utils/ball_mode_search.py`; Test `tests/test_ball_mode_search_fit.py`.

`_SegmentSolver` wraps a real `_Solver` (built from the same kwargs) to reuse `_fit_arc`/`_rolling_span` and the obs-collection (`_interior_obs`) WITHOUT reimplementing the gap-fill/p_flight skip rules. Key function:

```python
def fit_segment(seg_solver, fa, fb, mode, player_id, anchor_world) -> SegmentFit
```

returns `(worlds: dict[int,np.ndarray], residual_px: float, underconstrained: bool, kind: str, boundary_vel: tuple[np.ndarray|None, np.ndarray|None])`. FLIGHT fits endpoint-free from interior pixels + in-range `world_fixes` (no endpoint knot unless `fa`/`fb` is a manual anchor → pass its world as a knot). ROLLING via `fit_rolling_segment`. Memoized by `(fa, fb, mode, player_id, fa_anchor, fb_anchor, wfixes_sig)`; a shared `_fit_calls` counter raises `BudgetExceeded` past `max_segment_fit_calls`. Boundary velocities computed for ALL modes (flight: parabola end-velocity; rolling: finite-diff of the roll fit; possessed: FK joint velocity; stationary: zero) — F4.

- [ ] **Step 1: Failing tests** — flight segment fit recovers a synthetic arc; rolling fit recovers a roll; cache returns identical object for repeated key + increments counter once; `BudgetExceeded` raised past the cap; boundary_vel non-None for every mode. **Step 2: FAIL. Step 3: Implement. Step 4: PASS** + `tests/test_ball_flight.py` green. **Step 5: Commit** `feat(ball): segment-fit reuse layer with sound frame-keyed cache`.

---

### Task 4: Possessed mode

**Files:** Modify `src/utils/ball_mode_search.py`; Test `tests/test_ball_mode_search_possessed.py`.

A possessed segment tethers the ball to a player's foot via `PlayerContext` FK: `worlds[f] = foot_world(player_id, f)`; cost = soft pixel residual between projected foot world and the ball pixel (`possessed_tether_px` scale) + per-frame penalty for frames where FK is unavailable (held/interpolated). The possessing player is searched over the **2 players nearest the ball's last confident pixel at segment start**, each a separate hypothesis branch (dedup by player_id). State emitted is always `grounded` (F13).

- [ ] **Step 1: Failing test** — synthetic: ball rides player P1's foot for a span; `fit_segment(mode=POSSESSED, player_id="P1")` yields worlds within tol of the foot track and low residual; a wrong player_id yields high residual; missing-FK frames are held, not crashed. **Step 2: FAIL. Step 3: Implement. Step 4: PASS. Step 5: Commit** `feat(ball): possessed mode tethered to player FK`.

---

### Task 5: Scoring

**Files:** Modify `src/utils/ball_mode_search.py`; Test `tests/test_ball_mode_search_scoring.py`.

`segment_cost(fit, mode, cfg)` = raw-px residual (F5) + `segment_cost_constant` (BIC parsimony) + `out_of_view_frame_penalty*len` for OUT_OF_VIEW + `underconstrained` flag penalty. `transition_cost(prev_seg, bp, next_mode, cfg)` = event-agreement bonus (transition coincident with a high-score breakpoint) − or `unexplained_break_penalty` (transition with no event) / `ignored_event_penalty` (high-score event with no transition) + `velocity_discontinuity_weight*‖Δv‖` where both boundary velocities are defined (restitution-aware only for flight↔ground with defined v_z — F4). `kind_weight=1.0` uniform for v1 (F6).

- [ ] **Step 1: Failing tests** — pin each term: a tighter-px flight beats a looser-px rolling on the same pixels (proves raw-px, not gate-normalized — F5); an unexplained break costs more than an event-aligned one; a velocity-continuous bounce costs less than a discontinuous one. **Step 2: FAIL. Step 3: Implement. Step 4: PASS. Step 5: Commit** `feat(ball): mode-search scoring (raw-px residual + transition priors)`.

---

### Task 6: Beam loop

**Files:** Modify `src/utils/ball_mode_search.py`; Test `tests/test_ball_mode_search_beam.py`.

Left-to-right beam over sorted breakpoints. State = `Hypothesis`; per column, extend each beam hypothesis by every `eligible_modes(span)` (and possessed player branches), score, keep top `beam_width` by quantized `(cost, frame, mode_enum)`. Manual-anchor frames are FORCED breakpoints and prune hypotheses that span them (F-pruning not penalty). Dominance collapse de-dupes near-identical partitions; `runner_up` = lowest-cost hypothesis whose **partition differs** from the winner (F11). Determinism: cost-quantized tie-break (F10).

- [ ] **Step 1: Failing tests** — (a) the canonical multi-impact case: a synthetic flight→bounce→flight→roll with events at the bounces resolves into ≥3 segments with the correct modes (the 454-488 analogue); (b) a manual anchor mid-flight is never spanned; (c) two identical runs → identical winning partition (determinism); (d) the 40-spurious-break case resolves to ≤4 segments (the 201-282 analogue). **Step 2: FAIL. Step 3: Implement. Step 4: PASS. Step 5: Commit** `feat(ball): mode-sequence beam search loop`.

---

### Task 7: Render to SolveResult + entry point

**Files:** Modify `src/utils/ball_mode_search.py`, `src/pipeline/quality_report.py`; Test `tests/test_ball_mode_search_render.py`.

`solve_modes(**kwargs) -> SolveResult`: build `_Solver`/`_SegmentSolver`, breakpoints (manual anchors hard + `detect_event_candidates` permissive + boundaries), run the beam, then `render(winner)` → call Task-0's `commit_span`/`apply_node_authority`/`apply_restitution_flags` to assemble the EXACT `SolveResult` shape (full-frame `state_by_frame` defaulting "missing"; flight segments only for FLIGHT kind; `world_fixes` already consumed in fits). Add `diagnostics["mode_search"] = {hypotheses_explored, beam_width, winning_cost, runner_up_cost, fit_calls, budget_hit}` and emit `diagnostics["out_of_view_spans"]` (+ `underconstrained_spans` for OUT_OF_VIEW > N frames — F14). quality_report surfaces both.

- [ ] **Step 1: Failing tests** — `solve_modes` on a synthetic scene returns a `SolveResult` whose `state_by_frame` covers all frames, uses only valid `State` literals (assert against `schemas/ball_track.State`), flight frames have non-null segment ids, possessed frames are `grounded`; mode_search + out_of_view diag present. **Step 2: FAIL. Step 3: Implement. Step 4: PASS** + the Task-0 golden test still green. **Step 5: Commit** `feat(ball): render mode-search winner to SolveResult + diagnostics`.

---

### Task 8: Stage seam + whole-shot piecewise fallback + config

**Files:** Modify `src/stages/ball.py`, `config/default.yaml`; Test `tests/test_ball_stage_global_solver.py`.

In `_solve_shot`, branch on `cfg.get("solver","piecewise")`: `"global"` → `solve_modes(**same_kwargs)`, wrapped so a `BudgetExceeded` (or any solve_modes exception) **falls back to `solve_piecewise(**same_kwargs)`** (F8) with a `logger.warning` + a diag flag `mode_search_fallback: true`. `"piecewise"` → unchanged. Config: `ball.solver: piecewise` (default) + `ball.mode_search.*` block (all `ModeSearchCfg` fields) with a comment that global is opt-in until validated.

- [ ] **Step 1: Failing tests** — (a) `solver: global` end-to-end on the two-camera synthetic produces a valid BallTrack; (b) a forced budget of 1 triggers the whole-shot piecewise fallback and the result equals the piecewise result (assert `mode_search_fallback` flag + track matches `solver: piecewise`); (c) `solver: piecewise` path byte-identical to today. **Step 2: FAIL. Step 3: Implement. Step 4: PASS** + all stage suites green. **Step 5: Commit** `feat(ball): ball.solver piecewise|global seam with whole-shot fallback`.

---

### Task 9: Full suite + real-clip validation

- [ ] **Step 1:** `.venv311/bin/python -m pytest tests/ -q -k "ball or anchor or tracker or quality"` green.
- [ ] **Step 2:** Re-run ball stage with `--config` overriding `ball.solver: global` on `output-origi` (restore manual anchors first) and `output-kroupi`. CPU.
- [ ] **Step 3:** Acceptance (spec Phase 2 + critique F16):
  - origi01 span 454-488 → ≥3 segments incl. net-impact + bounce, no underconstrained flag, ≤8px/segment.
  - origi01 201-282 → ≤4 segments ≤8px.
  - **origi02 203-262 + the 273-302 fix run** → the global solver builds flight segments there consuming the cross-replay fixes (the Phase-1.5 payoff); cross-view consistency (consistency tool) improves vs Phase-1.5.
  - **Frame-coverage ≥ piecewise on every clip** (F16) and residual ≤ piecewise where comparable; runtime ≤3× piecewise; worst-case `fit_calls` < `max_segment_fit_calls`.
  - kroupi: global solve valid, no crash; piecewise remains default.
- [ ] **Step 4:** Append `## Phase 2 validation results` to the design spec; commit `docs: phase 2 mode-search validation results`. If acceptance shows global ≥ piecewise across clips, note the recommendation to flip the default in a follow-up.

---

## Risk register (from the design review, with mitigations baked into tasks)

| Risk | Mitigation | Task |
|---|---|---|
| Output-contract drift (closure extraction) | golden byte-identical test before any render work | T0 |
| New states break the viewer | map onto existing `State` literals; assert in render test | T0/T7 |
| Fit-cache unsound across hypotheses | endpoint-free fits + frame+anchor-bool key | T3 |
| Search-cost blowup | beam_width cap + soft-NMS breakpoints + whole-shot piecewise fallback | T1/T6/T8 |
| Coverage regression vs piecewise | explicit frame-coverage parity acceptance check | T9 |
| Audit self-reference (`_audit_auto_nodes`→`_solve_span`) | global path skips the audit; anchors=hard breakpoints, auto frames=candidates | T6/T7 |
| Determinism across machines | cost-quantized tie-break; tolerance-band determinism test | T6 |
| Scoring overfit on 3 clips | uniform kind_weight v1; surface winner/runner-up gap in diag | T5/T7 |
