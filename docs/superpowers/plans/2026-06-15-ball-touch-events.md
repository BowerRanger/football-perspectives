# Ball Touch-Event Tracking — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make a sparse, editable set of 3-D ball *events* (player touches pinned to the contacting body joint, plus bounce/goal/rest physics waypoints) the default ball product; let UE interpolate between them; keep the dense solver behind `ball.solver=piecewise|global`.

**Architecture:** Approach C — refactor the ball stage's back half into a strategy. A new `EventResolver` resolves each merged anchor/event to a `BallKeyframe` (touches body-pinned), derives interpolation `BallSegment`s, and a pure `interpolate_events()` renders a derived dense `BallTrack` for glTF/web. `ball.solver=events` is the new default. The dense solver path is untouched.

**Tech Stack:** Python 3 (numpy/scipy, dataclasses), pytest (light venv — no torch needed for the new units), FastAPI + vanilla JS (web), Unreal-free Python for the UE evaluator.

**Spec:** [`docs/superpowers/specs/2026-06-15-ball-touch-events-design.md`](../specs/2026-06-15-ball-touch-events-design.md)

---

## File structure

| File | Responsibility | Action |
|------|----------------|--------|
| `src/schemas/ball_anchor.py` | add `confidence`, `end_frame` to `BallAnchor` (+ load) | Modify |
| `src/schemas/ball_keyframes.py` | add `end_frame` to `BallKeyframe`; add `BallSegment` + `segments` to `BallKeyframeSet` (+ load/save) | Modify |
| `src/utils/ball_segments.py` | derive `BallSegment`s from ordered keyframes (segment kind + hints) | Create |
| `src/utils/ball_interpolate.py` | `interpolate_events(keyframe_set, fps) -> BallTrack` (pure §10 evaluator) | Create |
| `src/utils/ball_event_resolver.py` | `resolve_events(...) -> EventResolveResult` (body-pin touches, ray∩geometry waypoints, build keyframes+segments) | Create |
| `src/stages/ball.py` | dispatch `ball.solver=events`; wire resolver + interpolator + emit | Modify |
| `src/utils/ball_auto_anchor.py` | propagate `BallEvent.score` → `BallAnchor.confidence` | Modify |
| `config/default.yaml` | document `ball.solver: events\|piecewise\|global`; default `events` | Modify |
| `src/pipeline/quality_report.py` | report ball `mode` + `derived` | Modify |
| `src/web/server.py` | `confidence`/`end_frame` in payload; `GET /joints-near`; events list serving | Modify |
| `src/web/static/ball_anchor_editor.html` | event/touch list panel (list, add/edit/delete, auto vs manual, click-suggest) | Modify |
| `Content/Python/football_perspectives/ball_segment_interp.py` (UE side) | Unreal-free ballistic+roll segment evaluator mirroring `ball_interpolate` | Create |
| `tests/test_ball_*` | unit + integration tests per task | Create/Modify |

---

## Phase 1 — Core (events mode)

### Task 1: Extend `BallAnchor` with `confidence` + `end_frame`

**Files:** Modify `src/schemas/ball_anchor.py`; Test `tests/test_ball_anchor_schema.py`

- [ ] **Step 1 — failing test:**
```python
from pathlib import Path
from src.schemas.ball_anchor import BallAnchor, BallAnchorSet

def test_anchor_confidence_and_end_frame_roundtrip(tmp_path: Path):
    s = BallAnchorSet("clip", (1920, 1080), (
        BallAnchor(frame=5, image_xy=(10.0, 20.0), state="player_touch",
                   player_id="P001", bone="r_foot", confidence=0.42),
    ))
    p = tmp_path / "a.json"; s.save(p)
    back = BallAnchorSet.load(p)
    assert back.anchors[0].confidence == 0.42
    assert back.anchors[0].end_frame is None

def test_anchor_confidence_defaults_to_one_for_legacy(tmp_path: Path):
    p = tmp_path / "legacy.json"
    p.write_text('{"clip_id":"c","image_size":[100,100],'
                 '"anchors":[{"frame":0,"image_xy":[1,2],"state":"grounded"}]}')
    back = BallAnchorSet.load(p)
    assert back.anchors[0].confidence == 1.0
```
- [ ] **Step 2 — run, expect FAIL** (`TypeError: unexpected keyword 'confidence'`): `pytest tests/test_ball_anchor_schema.py -q`
- [ ] **Step 3 — implement:** add `confidence: float = 1.0` and `end_frame: int | None = None` to the dataclass (after `spin`); in `load()` parse `confidence=float(a.get("confidence", 1.0))` and `end_frame=int(a["end_frame"]) if a.get("end_frame") is not None else None`; clamp confidence to `[0,1]`.
- [ ] **Step 4 — run, expect PASS.**
- [ ] **Step 5 — commit:** `feat(ball): BallAnchor gains confidence + end_frame (backward-compatible)`

### Task 2: Extend `BallKeyframe`/`BallKeyframeSet` with `end_frame` + `BallSegment`

**Files:** Modify `src/schemas/ball_keyframes.py`; Test `tests/test_ball_keyframes_segments.py`

- [ ] **Step 1 — failing test:** assert a `BallSegment(start_frame, end_frame, kind, hints)` round-trips inside `BallKeyframeSet.segments`, and a legacy keyframes file (no `segments`) loads with `segments == ()`. Assert `BallKeyframe(..., end_frame=12)` round-trips.
- [ ] **Step 2 — run, expect FAIL.**
- [ ] **Step 3 — implement:** add `SegmentKind = Literal["ballistic","roll","carry","rest","free_flight"]`; `@dataclass(frozen=True) BallSegment(start_frame:int, end_frame:int, kind:str, hints:dict)`; add `end_frame: int | None = None` to `BallKeyframe` (+ parse in `_load_keyframe`); add `segments: tuple[BallSegment, ...] = ()` to `BallKeyframeSet`; parse `segments` in `load()` (default `()`), include in `save()` (already via `asdict`).
- [ ] **Step 4 — run, expect PASS.**
- [ ] **Step 5 — commit:** `feat(ball): BallSegment + keyframe end_frame for sparse interpolation`

### Task 3: `interpolate_events()` — pure §10 reference interpolator

**Files:** Create `src/utils/ball_interpolate.py`; Test `tests/test_ball_interpolate.py`

Implements the §10 contract. `interpolate_events(keyframe_set, fps) -> BallTrack`. Per segment:
- `ballistic`: parabola through the two endpoint `world_xyz` under gravity `g=hints["gravity"]` (default −9.81 z). Solve `v0` from endpoints + Δt; sample each frame. Optional Magnus via `omega_rad_s` (reuse `ball_orientation`).
- `roll`/`rest`: linear/eased ground interpolation (constant for rest).
- `carry`: linear between endpoints in P1 (refined in Phase 3).
- `free_flight`: gravity arc between bracketing 3-D endpoints; gap frames `world_xyz=None`/`state="flight"`.
- Quaternions via `integrate_orientation`.

- [ ] **Step 1 — failing tests:** (a) two grounded keyframes 0→10 → 11 frames, midpoint XY is the average, all `z≈ball_radius`; (b) ballistic launch at f0 (z=0) and land at f10 (z=0) with apex implied → midpoint `z` is the parabola max and `> 0`; (c) a `free_flight` bracket yields `None` world for interior frames; (d) output length == clip frame count and `BallFrame.frame` is contiguous.
- [ ] **Step 2 — run, expect FAIL.**
- [ ] **Step 3 — implement** the evaluators (numpy only).
- [ ] **Step 4 — run, expect PASS.**
- [ ] **Step 5 — commit:** `feat(ball): pure interpolate_events reference interpolator (sparse->dense)`

### Task 4: `ball_segments.derive_segments()` — keyframes → segments

**Files:** Create `src/utils/ball_segments.py`; Test `tests/test_ball_segments.py`

`derive_segments(keyframes: Sequence[BallKeyframe], n_frames, fps) -> tuple[BallSegment, ...]`. Walk consecutive resolved keyframes; classify each gap:
- launch state (`kick`/`shot` touch/`volley`/`header`) → next contact = **ballistic**;
- both endpoints grounded/rest = **roll** (or **rest** if positions ~equal);
- a `carry` anchor span = **carry**;
- a bracket touching `off_screen_flight` = **free_flight**.
Handle clip-boundary open segments (spec §10): open-start hold/back-extrapolate; open-end bound to `n_frames-1`, omit zero-length, flag `hints["open_ended"]`/`["inferred"]`. Carry `gravity`, `omega_rad_s`, `player_id`, `restitution` into `hints`.

- [ ] **Step 1 — failing tests:** touch(shot)→bounce yields one `ballistic` segment with correct frame range; grounded→grounded yields `roll`; trailing launch with no successor yields a `ballistic` bounded to last frame with `hints["open_ended"]`; empty input yields `()`.
- [ ] **Step 2–4 — run/implement/run.**
- [ ] **Step 5 — commit:** `feat(ball): derive interpolation segments from keyframes`

### Task 5: `ball_event_resolver.resolve_events()` — body-pinned resolution

**Files:** Create `src/utils/ball_event_resolver.py`; Test `tests/test_ball_event_resolver.py`

`resolve_events(*, anchor_by_frame, player_ctx, per_frame_K/R/t, distortion, ball_radius, goal_geometry, ground_touch_frames, n_frames, fps, clip_id, image_size) -> EventResolveResult` where `EventResolveResult` carries `keyframe_set: BallKeyframeSet` (incl. segments), `world_by_frame: dict[int,(world,conf)]` (from `interpolate_events`), `state_by_frame`, and `diagnostics`. Touch resolution (spec §7):
- `player_touch` → `player_ctx.joint_world(fi, player_id, bone)`; if a confident `image_xy` exists, ray-refine via `project_point_onto_pixel_ray`; else use joint position directly (occlusion case). Offset by `ball_radius` along the camera→joint ray. `depth_source="player_bone"`.
- waypoints (`bounce`/`grounded`/`goal_impact`) → reuse existing `_resolve_anchor_world` rules.

Use a fake `player_ctx` (returns fixed joint worlds) and identity-ish camera in tests so no torch/video is needed.

- [ ] **Step 1 — failing tests:** (a) a single airborne `player_touch` with a known joint world and a ball pixel on the joint's ray resolves to ≈ joint world (within ball_radius); (b) the **occluded** case (`image_xy=None`) still resolves to the joint world (not `None`); (c) the resulting `keyframe_set.keyframes` has one entry per anchor with `depth_source="player_bone"`; (d) `world_by_frame` spans `n_frames` via the interpolator.
- [ ] **Step 2–4 — run/implement/run.**
- [ ] **Step 5 — commit:** `feat(ball): EventResolver — body-pinned touch resolution + keyframes/segments`

### Task 6: Propagate event score → anchor confidence

**Files:** Modify `src/utils/ball_auto_anchor.py`; Test `tests/test_ball_auto_anchor_confidence.py`

- [ ] **Step 1 — failing test:** `generate_auto_anchors(...)` over a synthetic touch event with `score=0.7` yields a `BallAnchor` with `confidence==0.7`.
- [ ] **Step 2–4:** in `_event_candidates`/`_grounded_candidates`, pass `confidence=ev.score`/`cand.score` into the `BallAnchor(...)` constructor.
- [ ] **Step 5 — commit:** `feat(ball): carry auto-event score into anchor confidence`

### Task 7: Dispatch `ball.solver=events` in the stage + diag mode/derived

**Files:** Modify `src/stages/ball.py` (~1476 dispatch, ~1647 diag); Modify `config/default.yaml`; Modify/Add `tests/test_ball_stage_global_solver.py`

- [ ] **Step 1 — failing test:** update `test_default_solver_matches_explicit_piecewise` → `..._events`: assert the default run's `_ball_diag.json` has `solver=="events"` and `derived==True`; assert `ball.solver=piecewise` reproduces the old `solver=="piecewise"`. (Use the existing synthetic fixture in that file.)
- [ ] **Step 2 — run, expect FAIL.**
- [ ] **Step 3 — implement:** branch `if solver_name == "events": result = resolve_events(...)` (build an `EventResolveResult` exposing `.world_by_frame/.state_by_frame/.flight_segments/.diagnostics` so the existing emit code at 1517+ runs unchanged); emit keyframes from `result.keyframe_set` (skip re-build when present); set `diag["mode"]=solver_name` and `diag["derived"]=(solver_name=="events")`; default `cfg.get("solver","events")`; raise `ValueError` on unknown solver. In `config/default.yaml` set `ball.solver: events` with a comment listing `events|piecewise|global`.
- [ ] **Step 4 — run, expect PASS** (and run the full `tests/test_ball_stage*.py` to catch regressions; fix solver-name/residual assertions to pin `ball.solver=piecewise`).
- [ ] **Step 5 — commit:** `feat(ball): events resolver is the default ball.solver (piecewise/global behind flag)`

### Task 8: Quality report mode/derived

**Files:** Modify `src/pipeline/quality_report.py`; Test `tests/test_quality_report_ball.py`

- [ ] **Step 1 — failing test:** a diag with `mode="events", derived=true` surfaces `mode`/`derived` in the ball section.
- [ ] **Step 2–4:** read `diag.get("mode", diag.get("solver"))` and `diag.get("derived", False)` in `_ball_shot_entry`; aggregate.
- [ ] **Step 5 — commit:** `feat(quality): report ball mode + derived-track flag`

---

## Phase 2 — Web viewer event/touch list

### Task 9: API — confidence/end_frame in payload + `GET /joints-near`
**Files:** Modify `src/web/server.py`; Test `tests/test_web_ball_anchors_api.py` (FastAPI `TestClient`).
- [ ] Add `confidence`/`end_frame` (optional) to `BallAnchorEntry` Pydantic model + POST round-trip test.
- [ ] Add `GET /joints-near?shot=&frame=&u=&v=&r=` → reads the shot's `PlayerContext` and returns candidate `{player_id, bone, uv, dist_px, confidence}` sorted by pixel distance (read-only). Test with a fixture output dir.
- [ ] Commit per endpoint.

### Task 10: Event list panel in `ball_anchor_editor.html`
**Files:** Modify `src/web/static/ball_anchor_editor.html`.
- [ ] Right-column **Event List**: chronological rows (`time · kind · player(dot) · bone · touch_type · confidence`); auto rows dashed with confirm/dismiss; click-row seeks.
- [ ] **Add touch:** player+bone+kind selectors (populated from tracks + `VALID_BONES`); click ball → `GET /joints-near` suggests player/bone; create anchor at current frame.
- [ ] **Edit/Delete/Dismiss** wired to existing `POST /ball-anchors/{shot}`.
- [ ] Manual smoke test via `python recon.py serve` (documented; no automated browser test in scope).
- [ ] Commit.

---

## Phase 3 — Carry / possession

### Task 11: `ball_possession.detect_carry_spans()`
**Files:** Create `src/utils/ball_possession.py`; Test `tests/test_ball_possession.py`.
- [ ] Detect spans of same-player touches with ball near feet + small relative displacement → `carry` anchor (`end_frame`). TDD with synthetic touch sequences.
- [ ] Add `carry` to `BallAnchorState`/`KeyframeState` + validation (`player_id`, foot `bone`, `end_frame>frame`).
- [ ] Wire into the events core; `carry` interpolation follows player ground path (extend `ball_interpolate` carry branch + `ball_segments`).
- [ ] Commit per sub-step.

---

## Stretch — UE-side interpolation tooling

### Task 12: Unreal-free segment evaluator (UE side)
**Files:** Create `Content/Python/football_perspectives/ball_segment_interp.py`; Test `tests/test_ue_ball_segment_interp.py` (the math is Unreal-free; test from the repo venv by importing the file by path).
- [ ] Port the `ballistic` + `roll` evaluators from `ball_interpolate` (gravity + optional Magnus curl; sign convention matching the existing `CurlStrengthDegPerSec`). Pure-python, numpy-optional.
- [ ] Unit test: a ballistic segment produces the same sampled positions (within tol) as the pipeline `interpolate_events` for the same endpoints/hints (parity test guards divergence — spec §17).
- [ ] Document the editor-utility wiring (reads `keyframes_json.segments`, keys the ball transform track via existing `build_sequence` hooks) as comments/README; live-editor keying is a manual follow-up (not run unattended).
- [ ] Commit.

---

## Validation gate (run before declaring done)
- [ ] `pytest tests/ -q` green (light venv). Note any torch-gated tests skipped.
- [ ] `ruff`/`black` clean on changed files.
- [ ] Quality-report ball section shows `mode`/`derived`.
- [ ] Final review pass (self code-review on the diff).
