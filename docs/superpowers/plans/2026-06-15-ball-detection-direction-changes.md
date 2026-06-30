# Ball Detection & Direction-Change Touch Detection — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Raise automatic player-touch detection from ~1/59 on gberch toward ≥70–80% recall by (A) detecting the ball at usable resolution, (B) adding a camera-compensated motion channel + robust direction-change segmentation, and (C) pose-anchored touch hypotheses/attribution.

**Architecture:** Three phases feeding the existing `BallEvent`→`generate_auto_anchors`→events-mode `EventResolver` stream (no downstream schema change). A wraps the detector with zoom/tile high-res; B fuses optical-flow candidates and replaces local velocity-breaks with global piecewise segmentation; C uses dense poses to propose/attribute touches.

**Tech Stack:** Python (numpy/scipy/opencv), pytest (light venv; torch only for live WASB runs), existing modules `ball_detector`/`ball_second_pass`/`ball_tracker`/`ball_auto_events`/`ball_player_context`/`camera_projection`.

**Spec:** [`docs/superpowers/specs/2026-06-15-ball-detection-direction-changes-design.md`](../specs/2026-06-15-ball-detection-direction-changes-design.md)

**Validation harness:** gberch's 59 manual anchors (backed up at `/tmp/gberch_preregen_backup/gberch_ball_anchors.json`) are pseudo-ground-truth. After each phase: `recon.py run --output ./output --stages ball` then compare auto touches to the manual set.

---

## File structure

| File | Responsibility | Action |
|------|----------------|--------|
| `src/utils/ball_highres_detect.py` | Phase A: zoom-refine + tiled-relocate over a base detector | Create |
| `src/utils/ball_motion_flow.py` | Phase B1: camera-comp homography warp + motion-blob ball candidates | Create |
| `src/utils/ball_traj_segment.py` | Phase B2: robust piecewise fit → direction-change `_Break`s | Create |
| `src/utils/ball_pose_touch.py` | Phase C: foot kinematics + touch hypotheses + attribution | Create |
| `src/utils/ball_touch_recall.py` | Validation: recall/precision of auto touches vs a manual anchor set | Create |
| `src/stages/ball.py` | wire high-res into `_detect_loop`; motion fusion; recall diag | Modify |
| `src/utils/ball_auto_events.py` | breakpoint source = `ball_traj_segment`; attributor = `ball_pose_touch` | Modify |
| `src/utils/ball_auto_anchor.py` | high-recall gate tuning | Modify |
| `src/pipeline/quality_report.py` | surface `touch_recall_vs_manual` | Modify |
| `config/default.yaml` | `ball.highres/motion/segment/pose_touch.*` | Modify |

Each module < 400 lines. Pure-logic modules (`ball_traj_segment`, `ball_motion_flow` math, `ball_pose_touch` kinematics, `ball_touch_recall`, `ball_highres_detect` crop-mapping) are torch-free and unit-tested in the light venv. Live WASB integration is validated by the real-clip gberch run.

---

## Phase 0 — Validation harness first (so every later phase is measurable)

### Task 0: `ball_touch_recall.match_touches`

**Files:** Create `src/utils/ball_touch_recall.py`; Test `tests/test_ball_touch_recall.py`

- [ ] **Step 1 — failing test:**
```python
from src.utils.ball_touch_recall import match_touches

def test_recall_matches_by_frame_and_bone():
    manual = [(100, "P1", "r_foot"), (200, "P2", "head"), (300, "P1", "l_foot")]
    auto =   [(101, "P1", "r_foot"), (260, "P2", "head")]  # 1 hit (±2fr), 1 miss-frame, 1 missed
    r = match_touches(manual, auto, frame_tol=2)
    assert r["n_manual"] == 3 and r["n_auto"] == 2
    assert r["true_positive"] == 1          # frame 100~101 + bone match
    assert r["recall"] == 1/3
    assert r["precision"] == 1/2
    assert r["false_positive"] == 1
```
- [ ] **Step 2 — run, expect FAIL** (ModuleNotFound): `.venv/bin/python -m pytest tests/test_ball_touch_recall.py -q`
- [ ] **Step 3 — implement** `match_touches(manual, auto, *, frame_tol=2, require_bone=True)`: greedy nearest-frame matching of auto→manual where `|Δframe|≤frame_tol` and (optionally) bone agrees; return `{n_manual, n_auto, true_positive, false_positive, recall, precision}`. Also add `touches_from_anchor_set(path)` that loads a `BallAnchorSet` and returns `[(frame, player_id, bone)]` for `state=="player_touch"`.
- [ ] **Step 4 — run, expect PASS.**
- [ ] **Step 5 — commit:** `feat(ball): touch recall/precision metric vs a manual anchor set`

### Task 0b: baseline measurement (no code) — record current gberch recall

- [ ] Run `.venv/bin/python -m pytest` baseline is green, then compute baseline:
```bash
.venv/bin/python -c "
from src.utils.ball_touch_recall import touches_from_anchor_set, match_touches
m=touches_from_anchor_set('/tmp/gberch_preregen_backup/gberch_ball_anchors.json')
a=touches_from_anchor_set('output/ball/gberch_ball_anchors_auto.json')
print(match_touches(m,a,frame_tol=2,require_bone=False))"
```
Record the baseline (expected ~1 TP of ~the manual player_touch count) in the commit message of Task 0.

---

## Phase A — high-resolution detection

### Task A1: `ball_highres_detect` crop/tile candidate mapping (pure)

**Files:** Create `src/utils/ball_highres_detect.py`; Test `tests/test_ball_highres_detect.py`

The pure geometry first (torch-free): mapping candidates from a crop/tile back to full-frame coords, and the tile grid generator.

- [ ] **Step 1 — failing test:**
```python
from src.utils.ball_highres_detect import map_crop_candidates, tile_windows

def test_map_crop_candidates_offsets_to_full_frame():
    cands = [(10.0, 20.0, 0.9)]
    assert map_crop_candidates(cands, x0=100, y0=50) == [(110.0, 70.0, 0.9)]

def test_tile_windows_cover_frame_with_overlap():
    wins = tile_windows(w=1000, h=600, tile=400, overlap=100)
    # windows are (x0,y0,tile) and union covers [0,1000]x[0,600]
    assert all(t == 400 for _, _, t in wins)
    assert min(x for x, _, _ in wins) == 0 and max(x for x, _, _ in wins) + 400 >= 1000
    assert max(y for _, y, _ in wins) + 400 >= 600
```
- [ ] **Step 2 — run, expect FAIL.**
- [ ] **Step 3 — implement** `map_crop_candidates(cands, x0, y0)` (add offset) and `tile_windows(w, h, tile, overlap)` (stride = tile-overlap, clamp last window to edge). (Mirror the existing `ball_second_pass.map_crop_candidates` so the two agree.)
- [ ] **Step 4–5 — run/commit:** `feat(ball): high-res crop/tile candidate geometry`

### Task A2: `HighResDetector` wrapper (zoom refine + tile relocate)

**Files:** Modify `src/utils/ball_highres_detect.py`; Test extend `tests/test_ball_highres_detect.py`

- [ ] **Step 1 — failing test** with a fake base detector that returns a hit only when the ball is "large enough" in the crop (simulating resolution dependence):
```python
class _FakeDetector:
    def detect_candidates(self, frame, min_score, top_k=5):
        # returns center of frame with score ~ frame area (proxy for ball apparent size)
        h, w = frame.shape[:2]
        return [(w/2, h/2, min(0.95, (w*h)/(1280*720)))]

def test_zoom_refine_raises_confidence(monkeypatch):
    from src.utils.ball_highres_detect import HighResDetector
    import numpy as np
    base=_FakeDetector(); hr=HighResDetector(base, zoom_crop_px=320)
    frame=np.zeros((720,1280,3),np.uint8)
    # coarse over full frame -> low score; zoom around (640,360) -> high score
    coarse = base.detect_candidates(frame, 0.05)[0]
    refined = hr.refine(frame, center=(640,360))
    assert refined is not None and refined[2] > coarse[2]
```
- [ ] **Step 2 — run, expect FAIL.**
- [ ] **Step 3 — implement** `HighResDetector(base, *, zoom_crop_px, tile, overlap, top_k, min_score)` with: `refine(frame, center)` → crop `zoom_crop_px` around center, `base.detect_candidates(crop)`, `map_crop_candidates`, return best; `relocate(frame)` → `tile_windows`, detect per tile, map+merge, return top-K. Reuse `ball_second_pass` zoom semantics. (No torch — base is injected.)
- [ ] **Step 4–5 — run/commit:** `feat(ball): HighResDetector zoom-refine + tile-relocate`

### Task A3: wire high-res into `_detect_loop` + config

**Files:** Modify `src/stages/ball.py` (`_detect_loop`, ~720–816, and `_build_detector`); Modify `config/default.yaml`; Test `tests/test_ball_stage_highres.py`

- [ ] **Step 1 — failing test:** integration test with `FakeBallDetector` whose hit rate depends on input size (small=miss); assert events-mode run with `ball.highres.enabled=true` yields **more** `detector`/`detector_hires` observations than with it disabled, on a synthetic small-ball scene. (Reuse the `_build_scene` harness from `tests/test_ball_stage_global_solver.py`.)
- [ ] **Step 2 — run, expect FAIL.**
- [ ] **Step 3 — implement:** in `_detect_loop`, after the coarse `detector.detect_candidates`, if `ball.highres.enabled` and (corridor/coarse hit present) and (`apparent_ball_px < trigger_min_ball_px` or coarse conf `< trigger_max_conf`), call `HighResDetector.refine(frame, center)`; on a `tile_on_gap_frames`-long miss run, call `relocate(frame)`. Take the better candidate; tag `sources[f]="detector_hires"`/`"tile"`. Add `ball.highres.*` to `config/default.yaml` (enabled true; `zoom_crop_px:320, trigger_min_ball_px:8, trigger_max_conf:0.5, always_zoom_when_located:true, tile_on_gap_frames:6, tile_grid:auto, tile_overlap_px:96, top_k:5`).
- [ ] **Step 4 — run, expect PASS** + full `tests/ -k ball` green.
- [ ] **Step 5 — commit:** `feat(ball): high-res zoom/tile detection in the detect loop (Phase A)`

### Task A4: measure Phase A on gberch

- [ ] Run `.venv/bin/python recon.py run --output ./output --stages ball`, then the Task-0b recall snippet. Record coverage + recall delta. **Gate:** coverage should rise materially (target trajectory toward >60%); commit a one-line note. If no improvement, debug before Phase B.

---

## Phase B — motion channel + direction-change segmentation

### Task B1: camera-compensation homography (pure)

**Files:** Create `src/utils/ball_motion_flow.py`; Test `tests/test_ball_motion_flow.py`

- [ ] **Step 1 — failing test:** a static world point projects to the same pixel after warping frame *f*→*f+1* by `H = K₁R₁R₀ᵀK₀⁻¹` (pan cancelled):
```python
import numpy as np
from src.utils.ball_motion_flow import frame_homography
def test_homography_cancels_pure_rotation():
    K=np.array([[1000,0,640],[0,1000,360],[0,0,1.]])
    R0=np.eye(3); th=np.deg2rad(2)
    R1=np.array([[np.cos(th),0,np.sin(th)],[0,1,0],[-np.sin(th),0,np.cos(th)]])
    H=frame_homography(K,R0,K,R1)
    # a pixel in frame1 maps back near where the same ray was in frame0
    p1=np.array([700,360,1.]); p0=H@p1; p0/=p0[2]
    assert abs(p0[0]-? ) # see impl: round-trips a shared world ray
```
  (Concrete assertion in impl: warping frame1 by `H` aligns a static background pixel to frame0 within <1px on synthetic data.)
- [ ] **Step 2–4:** implement `frame_homography(K0,R0,K1,R1) -> 3x3` (`K0 R0 R1^T K1^-1`, mapping frame1 pixels into frame0), and `warp_to_reference(img, H, size)`. Test with a synthetic textured image + known rotation; assert post-warp diff with frame0 background is ~0 while a moving dot remains.
- [ ] **Step 5 — commit:** `feat(ball): camera-compensation homography for motion detection`

### Task B2: motion-blob ball candidates

**Files:** Modify `src/utils/ball_motion_flow.py`; Test extend.
- [ ] TDD `motion_candidates(prev, cur, H, *, max_ball_px, min_speed_px, exclude_boxes) -> [(u,v,score,(vx,vy))]`: warp prev→cur frame by camera-comp `H`, abs-diff, threshold, connected-components, keep small/round/fast blobs not inside `exclude_boxes` (player bboxes from tracks). Tests: synthetic frame with a moving small dot + a large slow region → returns the dot, not the region; blob inside an exclude box is dropped. Commit `feat(ball): camera-compensated motion-blob ball candidates`.

### Task B3: robust piecewise direction-change segmentation

**Files:** Create `src/utils/ball_traj_segment.py`; Test `tests/test_ball_traj_segment.py`
- [ ] TDD `segment_track(uvs, *, fps, cfg) -> list[_Break]` (import `_Break` from `ball_auto_events`): RANSAC-fit piecewise linear (rolling) / quadratic (flight) segments over the sparse pixel track; DP/greedy breakpoint search minimizing `residual + breakpoint_penalty·(#segments)`; emit a `_Break` at each corner with `dir_change_deg`, `dspeed_px`, speed_before/after, vy_before/after, strength. **Deterministic** (fixed `numpy` seed per call from a `cfg.seed`). Tests: synthetic roll→kick→parabola→bounce recovers corners within ±2 frames; survives 30% randomly-dropped points; no spurious corners on a clean straight roll. Commit `feat(ball): robust piecewise direction-change segmentation`.

### Task B4: wire motion + segmentation into the pipeline

**Files:** Modify `src/utils/ball_auto_events.py` (breakpoint source), `src/stages/ball.py` (motion fusion in `_detect_loop`), `config/default.yaml`; Test `tests/test_ball_auto_events_segmentation.py`.
- [ ] Make `detect_events`/`detect_event_candidates` use `ball_traj_segment.segment_track` as the break source when `ball.segment.enabled` (fall back to `_raw_break_candidates` otherwise). Fuse `ball_motion_flow.motion_candidates` with the high-res candidates in `_detect_loop` (agreement→boost; motion-only→corridor-gate). Add `ball.motion.*` + `ball.segment.*` config. Test: an events-mode synthetic run with a turning ball recovers the turn as a touch/bounce that the local-break path missed. Commit `feat(ball): motion fusion + segmentation breakpoints in events mode (Phase B)`.

### Task B5: measure Phase B on gberch (same as A4). Gate on recall ↑.

---

## Phase C — pose-anchored touches

### Task C1: foot/body kinematics (pure)

**Files:** Create `src/utils/ball_pose_touch.py`; Test `tests/test_ball_pose_touch.py`
- [ ] TDD `joint_velocity(player_ctx, frame, player_id, bone, fps)` and `joint_accel(...)` via central finite-difference of `player_ctx.joint_world(f±1)` (return None at gaps). Test with a fake player_ctx on a known parabolic foot path → velocity/accel match analytic within tol. Commit `feat(ball): foot/body kinematics from pose context`.

### Task C2: touch hypotheses + high-recall attribution

**Files:** Modify `src/utils/ball_pose_touch.py`; Test extend.
- [ ] TDD `attribute_and_fuse(breaks, player_ctx, ball_uv_by_frame, *, cfg) -> list[BallEvent]`: for each break/corner, find the nearest active joint with `relaxed_radius_px` + a `kinematic_bonus_weight·alignment` term (foot velocity vs ball outbound direction); also emit pose-only hypotheses where a foot accelerates into the corridor with a bracketing direction change. De-dup one-per-(player,frame). Widen bones to knee/head/chest/hands. Tests: a touch just outside 25px but with aligned foot velocity is recovered; a pose hypothesis fires across a detection gap; non-touching nearby player is not attributed. Commit `feat(ball): pose-anchored touch hypotheses + high-recall attribution (Phase C)`.

### Task C3: wire attribution + relax gates

**Files:** Modify `src/utils/ball_auto_events.py` (use `attribute_and_fuse` for touch classification), `src/utils/ball_auto_anchor.py` (relaxed `min_event_score`, `contact_max_gap_m` under a high-recall flag), `config/default.yaml`. Test integration on a synthetic shot. Commit `feat(ball): pose attribution wired into events + high-recall gates`.

### Task C4: measure Phase C on gberch (A4 method). **Final gate:** recall ≥70–80% of manual player_touch anchors at ≤~1.5× false positives.

---

## Phase D — diagnostics + final validation

### Task D1: surface recall in diag + quality report
**Files:** Modify `src/stages/ball.py` (write `touch_recall_vs_manual` to `_ball_diag.json` when a manual anchor set exists), `src/pipeline/quality_report.py`. TDD the quality-report passthrough. Commit `feat(quality): touch_recall_vs_manual diagnostic`.

### Task D2: full validation
- [ ] `.venv/bin/python -m pytest tests/ -q -m "not fbx"` green.
- [ ] gberch single-shot recall meets the success criteria; spot-check origi01/kroupi.
- [ ] Restore gberch manual anchors from backup (optional) and note final numbers.

---

## Self-review notes
- **Spec coverage:** A→Tasks A1–A4; B→B1–B5; C→C1–C4; validation/metrics→Task 0 + D1–D2; config→A3/B4/C3. All spec sections mapped.
- **Determinism:** RANSAC seeded via `cfg.seed` (Task B3) — no unseeded randomness.
- **Type consistency:** `_Break` reused from `ball_auto_events` (B3); `BallEvent` reused (C2); `match_touches` signature stable across Task 0/A4/B5/C4.
- **Note:** B3/C2 RANSAC/optical-flow tasks carry test *intent* + signatures rather than full final code; the implementer writes the body under TDD (the failing test pins behaviour). This is a deliberate density choice for the algorithm-heavy modules.
