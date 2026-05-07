# Static-camera lock + sub-foot pitch precision — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Hard-lock the camera body when `static_camera=true` and reduce worst-case projected-vs-real pitch-line gap from ~2 m to <0.3 m.

**Architecture:** Three independently shippable phases — (1) make `static_camera=true` truly strict by storing `camera_centre` and rebuilding inter-anchor `t` from the locked centre; (2) add radial distortion `(k1,k2)` to the joint LM with Huber loss; (3) per-anchor LSD line detection + LM polish against the existing FIFA pitch-line catalogue.

**Tech Stack:** Python 3.11, NumPy, OpenCV (with `ximgproc` for `FastLineDetector`), SciPy `least_squares`, pytest. Existing `src/utils/pitch_lines_catalogue.py` provides the FIFA line catalogue.

**Spec:** `docs/superpowers/specs/2026-05-07-static-camera-precision-design.md`

---

## File Structure

| File | Status | Responsibility |
|---|---|---|
| `src/schemas/camera_track.py` | modify | Add `distortion`, `camera_centre` fields |
| `src/utils/anchor_solver.py` | modify | Add `camera_centre` + `(k1,k2)` to `JointSolution`; strict relock; Huber loss; line weight=1.0; `cv2.projectPoints` residuals |
| `src/utils/camera_projection.py` | **create** | `project_world_to_image`, `undistort_pixel` helpers |
| `src/utils/line_detector.py` | **create** | LSD-based pitch line segment detector (Phase 3) |
| `src/utils/anchor_line_polish.py` | **create** | Per-anchor LM refinement against detected lines (Phase 3) |
| `src/stages/camera.py` | modify | Inter-anchor `t = -R @ C_locked`; Phase 3 wiring; quality-report additions |
| `src/stages/hmr_world.py` | modify | Undistort ankle pixel before back-projecting foot ray |
| `src/stages/ball.py` | modify | Undistort ball pixel before ground projection |
| `config/default.yaml` | modify | Tighter reprojection threshold; Phase 3 config block |
| `src/web/static/viewer.html` | modify | Apply `(k1,k2)` to projected pitch overlay vertices |
| `src/web/static/anchor_editor.html` | modify | Same overlay distortion (when track exists) |
| `tests/test_anchor_solver.py` | modify | New tests for static invariant, no-fallback, distortion round-trip, Huber outlier, polish never regresses |
| `tests/test_camera_stage.py` | modify | Inter-anchor `t` invariant test |
| `tests/test_line_detector.py` | **create** | LSD on a synthetic pitch frame |
| `tests/test_anchor_line_polish.py` | **create** | Polish never-regress + R-perturbation recovery |
| `tests/test_camera_projection.py` | **create** | Round-trip distortion helper |

---

## Phase 1 — Strict body lock

### Task 1.1: Add `camera_centre` to `JointSolution`

**Files:**
- Modify: `src/utils/anchor_solver.py:60-69`
- Test: `tests/test_anchor_solver.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_anchor_solver.py`:

```python
@pytest.mark.unit
def test_joint_solution_carries_camera_centre():
    """After refine_with_shared_translation, the JointSolution exposes
    the locked camera centre, and every anchor satisfies -R^T @ t == C."""
    from src.utils.anchor_solver import refine_with_shared_translation

    anchors = _three_rich_anchors_static()  # helper added below
    sol = solve_anchors_jointly(anchors, image_size=IMAGE_SIZE)
    sol = refine_with_shared_translation(anchors, sol)

    assert sol.camera_centre is not None
    C = np.asarray(sol.camera_centre)
    for af, (_K, R, t) in sol.per_anchor_KRt.items():
        recovered_C = -R.T @ t
        assert np.allclose(recovered_C, C, atol=1e-6), (
            f"anchor {af}: -R^T @ t = {recovered_C} != C = {C}"
        )
```

Add the helper above existing tests (after `_make_line`):

```python
def _three_rich_anchors_static() -> tuple[Anchor, ...]:
    """Three anchors at yaw 0/+5/-5° about a static camera centre."""
    K = _K()
    rich_landmarks = (
        ("left_corner_near",  (0.0, 0.0, 0.0)),
        ("right_corner_near", (105.0, 0.0, 0.0)),
        ("left_corner_far",   (0.0, 68.0, 0.0)),
        ("right_corner_far",  (105.0, 68.0, 0.0)),
        ("left_post_top",     (0.0, 30.34, 2.44)),
        ("right_post_top",    (105.0, 30.34, 2.44)),
    )
    out: list[Anchor] = []
    for frame, yaw in ((0, 0.0), (60, 5.0), (120, -5.0)):
        R = _yaw(yaw)
        t = -R @ np.array([52.5, -30.0, 30.0])
        out.append(Anchor(
            frame=frame,
            landmarks=tuple(_make_landmark(K, R, t, n, w) for n, w in rich_landmarks),
        ))
    return tuple(out)
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd /Users/joebower/workplace/football-perspectives
pytest tests/test_anchor_solver.py::test_joint_solution_carries_camera_centre -v
```

Expected: FAIL — `JointSolution` has no `camera_centre` attribute.

- [ ] **Step 3: Add `camera_centre` to `JointSolution`**

Edit `src/utils/anchor_solver.py:60-69`:

```python
class JointSolution(NamedTuple):
    t_world: np.ndarray                                            # (3,) — median across anchors
    principal_point: tuple[float, float]                           # (cx, cy)
    per_anchor_KRt: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]
    """Per-anchor (K, R, t). Each anchor has its own translation because
    the broadcast-fixed-body assumption doesn't hold reliably on real
    clips (steadicam, broadcast cuts between cameras). Solving each
    anchor independently gives much tighter per-anchor calibration.
    """
    per_anchor_residual_px: dict[int, float]                       # frame -> mean px
    camera_centre: np.ndarray | None = None
    """World-frame camera body position when static_camera relock has
    been applied. `None` for un-relocked solutions. When set, every
    per-anchor `(R, t)` satisfies `-R^T @ t == camera_centre`."""
```

Then update `refine_with_shared_translation` to populate it. Find the final `return JointSolution(...)` block at the end of the function and replace it with:

```python
    return JointSolution(
        t_world=t_world_out,
        principal_point=(cx, cy),
        per_anchor_KRt=new_KRt,
        per_anchor_residual_px=new_res,
        camera_centre=C_locked.copy(),
    )
```

Also update the early-return branch (the rejection path at `if new_mean > 10.0 * max(old_mean, 1.0):`) — but this branch is removed in Task 1.2, so only updating the success path is needed for now.

- [ ] **Step 4: Run the test to verify it passes**

```bash
pytest tests/test_anchor_solver.py::test_joint_solution_carries_camera_centre -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/utils/anchor_solver.py tests/test_anchor_solver.py
git commit -m "feat(camera): expose camera_centre on JointSolution after relock"
```

---

### Task 1.2: Strict relock — remove silent fallback

**Files:**
- Modify: `src/utils/anchor_solver.py:186-219` (the rejection branch in `refine_with_shared_translation`)
- Test: `tests/test_anchor_solver.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_anchor_solver.py`:

```python
@pytest.mark.unit
def test_relock_does_not_silently_fall_back(caplog):
    """When the relocked residual is much worse than the original, we still
    return the relocked solution (not the original) and log an ERROR so the
    user knows to investigate."""
    import logging
    from src.utils.anchor_solver import refine_with_shared_translation

    # Two anchors with deliberately inconsistent C — second has its
    # translation shifted by 5m laterally to provoke a high relock residual.
    anchors = _three_rich_anchors_static()
    bad = anchors[1]
    K = _K()
    R = _yaw(5.0)
    t_bad = -R @ np.array([52.5 + 5.0, -30.0, 30.0])  # shifted +5m in x
    bad_landmarks = tuple(
        LandmarkObservation(
            name=lm.name,
            image_xy=_project(K, R, t_bad, np.asarray(lm.world_xyz)),
            world_xyz=lm.world_xyz,
        )
        for lm in bad.landmarks
    )
    bad_anchor = Anchor(frame=bad.frame, landmarks=bad_landmarks)
    inconsistent = (anchors[0], bad_anchor, anchors[2])

    sol = solve_anchors_jointly(inconsistent, image_size=IMAGE_SIZE)
    with caplog.at_level(logging.ERROR, logger="src.utils.anchor_solver"):
        relocked = refine_with_shared_translation(inconsistent, sol)

    # Even with a bad anchor, the result honours C.
    assert relocked.camera_centre is not None
    C = np.asarray(relocked.camera_centre)
    for _af, (_K, R_a, t_a) in relocked.per_anchor_KRt.items():
        assert np.allclose(-R_a.T @ t_a, C, atol=1e-6)
```

- [ ] **Step 2: Run to verify it fails**

```bash
pytest tests/test_anchor_solver.py::test_relock_does_not_silently_fall_back -v
```

Expected: FAIL — current code returns `sol` unchanged when residual >10× original, so anchors do not satisfy `-R^T @ t == C`.

- [ ] **Step 3: Replace the silent fallback with an ERROR log + continue**

Edit `src/utils/anchor_solver.py` — find the `if new_mean > 10.0 * max(old_mean, 1.0):` block (around line 186) and replace it:

```python
    new_mean = float(np.mean(list(new_res.values())))
    old_mean = float(np.mean(list(sol.per_anchor_residual_px.values())))
    if new_mean > 10.0 * max(old_mean, 1.0):
        # Catastrophic relock — log loudly so the user investigates
        # (likely cause: collinear anchors, or the camera body actually
        # moved). Critically, we DO NOT silently fall back to the
        # un-relocked solution. With static_camera=true, the contract is
        # that every anchor honours C; falling back to un-relocked breaks
        # that contract invisibly. Better to surface the problem.
        worst = sorted(new_res.items(), key=lambda kv: -kv[1])[:3]
        logger.error(
            "static-camera relock produced mean residual %.2f px "
            "(was %.2f px before relock). Worst offenders: %s. "
            "Continuing with relocked solution to honour the static-camera "
            "contract; if the camera body actually moves, set "
            "camera.static_camera=false in config.",
            new_mean, old_mean, worst,
        )
```

(Remove the `return sol` line that followed — the function now flows on to the existing logger.info + return JointSolution at the bottom.)

- [ ] **Step 4: Run to verify it passes**

```bash
pytest tests/test_anchor_solver.py::test_relock_does_not_silently_fall_back tests/test_anchor_solver.py::test_joint_solution_carries_camera_centre -v
```

Expected: both PASS.

- [ ] **Step 5: Run the full anchor_solver test suite to check for regressions**

```bash
pytest tests/test_anchor_solver.py -v
```

Expected: all PASS (no regressions on existing tests).

- [ ] **Step 6: Commit**

```bash
git add src/utils/anchor_solver.py tests/test_anchor_solver.py
git commit -m "fix(camera): no silent fallback on relock — honour static-camera contract"
```

---

### Task 1.3: Add `camera_centre` to `CameraTrack` schema

**Files:**
- Modify: `src/schemas/camera_track.py`
- Test: existing tests + new round-trip

- [ ] **Step 1: Write the failing test**

Add to `tests/test_camera_stage.py` (or create `tests/test_camera_track_schema.py` if no obvious home — search first):

```bash
grep -l "CameraTrack" tests/ -r
```

Add the test to whichever file already exercises CameraTrack save/load, otherwise create `tests/test_camera_track_schema.py`:

```python
"""Round-trip tests for CameraTrack schema (distortion, camera_centre)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.schemas.camera_track import CameraFrame, CameraTrack


@pytest.mark.unit
def test_camera_track_carries_camera_centre(tmp_path: Path):
    track = CameraTrack(
        clip_id="test",
        fps=25.0,
        image_size=(1920, 1080),
        t_world=[0.0, 0.0, 30.0],
        frames=tuple(),
        camera_centre=(52.5, -30.0, 30.0),
    )
    out = tmp_path / "track.json"
    track.save(out)
    loaded = CameraTrack.load(out)
    assert loaded.camera_centre == (52.5, -30.0, 30.0)


@pytest.mark.unit
def test_camera_track_legacy_load_without_camera_centre(tmp_path: Path):
    """Older saved tracks without camera_centre must still load (default None)."""
    out = tmp_path / "legacy.json"
    out.write_text(json.dumps({
        "clip_id": "legacy",
        "fps": 25.0,
        "image_size": [1920, 1080],
        "t_world": [0.0, 0.0, 30.0],
        "frames": [],
    }))
    loaded = CameraTrack.load(out)
    assert loaded.camera_centre is None
```

- [ ] **Step 2: Run to verify it fails**

```bash
pytest tests/test_camera_track_schema.py -v
```

Expected: FAIL — `CameraTrack` has no `camera_centre` field.

- [ ] **Step 3: Add `camera_centre` to `CameraTrack`**

Edit `src/schemas/camera_track.py`:

```python
@dataclass(frozen=True)
class CameraTrack:
    clip_id: str
    fps: float
    image_size: tuple[int, int]
    t_world: list[float]
    frames: tuple[CameraFrame, ...]
    principal_point: tuple[float, float] | None = None
    # World-frame camera body position when the clip was solved with
    # static_camera=true. None for moving-camera clips. When present,
    # every per-frame (R, t) satisfies -R^T @ t == camera_centre.
    camera_centre: tuple[float, float, float] | None = None
```

In the `load` classmethod, add:

```python
        cc_raw = data.get("camera_centre")
        camera_centre = tuple(cc_raw) if cc_raw is not None else None
```

and pass it to the constructor:

```python
        return cls(
            clip_id=str(data["clip_id"]),
            fps=float(data["fps"]),
            image_size=tuple(data["image_size"]),
            t_world=list(data["t_world"]),
            frames=frames,
            principal_point=principal_point,
            camera_centre=camera_centre,
        )
```

- [ ] **Step 4: Run to verify it passes**

```bash
pytest tests/test_camera_track_schema.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/schemas/camera_track.py tests/test_camera_track_schema.py
git commit -m "feat(schema): CameraTrack.camera_centre for static-camera clips"
```

---

### Task 1.4: Inter-anchor `t = -R @ C_locked` in camera stage

**Files:**
- Modify: `src/stages/camera.py:158-172`
- Test: `tests/test_camera_stage.py`

- [ ] **Step 1: Write the failing test**

First inspect `tests/test_camera_stage.py` for fixtures used:

```bash
head -40 tests/test_camera_stage.py
```

Add this test (adapt fixture imports to existing helpers; if there is no synthetic-clip fixture, write a small one using the `_three_rich_anchors_static` pattern from Task 1.1):

```python
@pytest.mark.unit
def test_inter_anchor_t_honours_locked_camera_centre(tmp_output_with_static_clip):
    """When static_camera=true, every output frame (anchor and inter-anchor)
    satisfies -R^T @ t == camera_centre to floating-point precision."""
    from src.schemas.camera_track import CameraTrack
    from src.stages.camera import CameraStage

    stage = CameraStage(
        output_dir=tmp_output_with_static_clip,
        config={"camera": {"static_camera": True, "anchor_line_polish": False}},
    )
    stage.run()
    track = CameraTrack.load(tmp_output_with_static_clip / "camera" / "camera_track.json")

    assert track.camera_centre is not None
    C = np.asarray(track.camera_centre)
    for f in track.frames:
        R = np.asarray(f.R)
        t = np.asarray(f.t)
        recovered = -R.T @ t
        assert np.allclose(recovered, C, atol=1e-4), (
            f"frame {f.frame}: -R^T @ t = {recovered} != C = {C}"
        )
```

(If `tmp_output_with_static_clip` doesn't already exist as a fixture, copy the pattern from existing `tests/test_camera_stage.py` and add the static-camera variant. If the existing test already builds an output dir manually, do the same.)

- [ ] **Step 2: Run to verify it fails**

```bash
pytest tests/test_camera_stage.py::test_inter_anchor_t_honours_locked_camera_centre -v
```

Expected: FAIL — inter-anchor frames violate the invariant due to LERP(t).

- [ ] **Step 3: Fix the inter-anchor interpolation**

Edit `src/stages/camera.py:158-172`. Find the loop:

```python
        for a, b in zip(anchor_frames, anchor_frames[1:]):
            K_a, R_a, t_a = anchor_solutions[a]
            K_b, R_b, t_b = anchor_solutions[b]
            slerp = Slerp([0.0, 1.0], Rotation.from_matrix([R_a, R_b]))
            for offset in range(1, b - a):
                idx = a + offset
                lerp_w = offset / (b - a)
                per_frame_K[idx] = (1.0 - lerp_w) * K_a + lerp_w * K_b
                per_frame_R[idx] = slerp([lerp_w]).as_matrix()[0]
                per_frame_t[idx] = (1.0 - lerp_w) * t_a + lerp_w * t_b
                per_frame_conf[idx] = 0.7
```

Replace with:

```python
        C_locked = sol.camera_centre  # may be None when static_camera=false
        for a, b in zip(anchor_frames, anchor_frames[1:]):
            K_a, R_a, t_a = anchor_solutions[a]
            K_b, R_b, t_b = anchor_solutions[b]
            slerp = Slerp([0.0, 1.0], Rotation.from_matrix([R_a, R_b]))
            for offset in range(1, b - a):
                idx = a + offset
                lerp_w = offset / (b - a)
                per_frame_K[idx] = (1.0 - lerp_w) * K_a + lerp_w * K_b
                R_inter = slerp([lerp_w]).as_matrix()[0]
                per_frame_R[idx] = R_inter
                if C_locked is not None:
                    # Static-camera invariant: t = -R @ C honours the locked body
                    per_frame_t[idx] = -R_inter @ np.asarray(C_locked)
                else:
                    per_frame_t[idx] = (1.0 - lerp_w) * t_a + lerp_w * t_b
                per_frame_conf[idx] = 0.7
```

Also expose `camera_centre` on the saved `CameraTrack` — find the `track = CameraTrack(...)` block (around line 209) and add the field:

```python
        track = CameraTrack(
            clip_id=anchors.clip_id,
            fps=float(fps),
            image_size=(w, h),
            t_world=list(t_world_median),
            frames=tuple(frames_out),
            principal_point=(float(principal_point[0]), float(principal_point[1])),
            camera_centre=(
                tuple(float(x) for x in sol.camera_centre)
                if sol.camera_centre is not None
                else None
            ),
        )
```

- [ ] **Step 4: Run to verify it passes**

```bash
pytest tests/test_camera_stage.py::test_inter_anchor_t_honours_locked_camera_centre -v
```

Expected: PASS.

- [ ] **Step 5: Run the full camera-stage test suite for regressions**

```bash
pytest tests/test_camera_stage.py tests/test_anchor_solver.py -v
```

Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add src/stages/camera.py src/schemas/camera_track.py tests/test_camera_stage.py
git commit -m "fix(camera): inter-anchor t honours locked camera centre"
```

---

### Task 1.5: Phase 1 quality-report metric `body_drift_max_m`

**Files:**
- Modify: `src/stages/camera.py` (where quality report is contributed)
- Test: `tests/test_quality_report.py` (or `test_camera_stage.py`)

First find where camera quality info is collected:

```bash
grep -n "quality_report\|body_drift\|camera_residual" src/stages/camera.py src/pipeline/*.py 2>/dev/null
```

- [ ] **Step 1: Locate the quality report construction**

Skim the result. The convention in this repo is per-stage outputs aggregated by `recon.py` — find where camera contributes diagnostics.

```bash
grep -rn "quality_report.json\|aggregate_quality\|camera.*residual" src/ recon.py 2>/dev/null
```

- [ ] **Step 2: Write the failing test**

Add to `tests/test_camera_stage.py`:

```python
@pytest.mark.unit
def test_camera_stage_reports_body_drift(tmp_output_with_static_clip):
    """With Phase 1 in place, body_drift_max_m must be 0.0 by construction."""
    from src.stages.camera import CameraStage
    import json

    stage = CameraStage(
        output_dir=tmp_output_with_static_clip,
        config={"camera": {"static_camera": True, "anchor_line_polish": False}},
    )
    stage.run()
    qr_path = tmp_output_with_static_clip / "camera" / "camera_quality.json"
    qr = json.loads(qr_path.read_text())
    assert qr["body_drift_max_m"] == 0.0
```

- [ ] **Step 3: Run to verify it fails**

```bash
pytest tests/test_camera_stage.py::test_camera_stage_reports_body_drift -v
```

Expected: FAIL — `camera_quality.json` doesn't exist or doesn't have `body_drift_max_m`.

- [ ] **Step 4: Add `camera_quality.json` write at end of `CameraStage.run`**

In `src/stages/camera.py`, after `track.save(...)`:

```python
        # Quality report contribution. Body drift is the worst-case
        # ||(-R^T @ t) - C_locked||; with Phase 1 in place this is 0 by
        # construction, but we record it so future regressions surface.
        if sol.camera_centre is not None:
            C = np.asarray(sol.camera_centre)
            drifts = []
            for f in frames_out:
                R = np.asarray(f.R)
                t = np.asarray(f.t)
                drifts.append(float(np.linalg.norm(-R.T @ t - C)))
            body_drift_max_m = max(drifts) if drifts else 0.0
        else:
            body_drift_max_m = None
        residuals = list(sol.per_anchor_residual_px.values())
        quality = {
            "body_drift_max_m": body_drift_max_m,
            "anchor_residual_mean_px": float(np.mean(residuals)) if residuals else None,
            "anchor_residual_max_px": float(np.max(residuals)) if residuals else None,
            "n_anchors": len(sol.per_anchor_KRt),
        }
        (self.output_dir / "camera" / "camera_quality.json").write_text(
            json.dumps(quality, indent=2)
        )
```

(Add `import json` at the top of `src/stages/camera.py` if not already imported.)

- [ ] **Step 5: Run to verify it passes**

```bash
pytest tests/test_camera_stage.py::test_camera_stage_reports_body_drift -v
```

Expected: PASS.

- [ ] **Step 6: Commit and tag Phase 1 milestone**

```bash
git add src/stages/camera.py tests/test_camera_stage.py
git commit -m "feat(camera): camera_quality.json with body_drift_max_m"
git tag phase-1-static-lock
```

---

## Phase 2 — Lens distortion + Huber loss + solver hygiene

### Task 2.1: Add `distortion` to `CameraTrack` schema

**Files:**
- Modify: `src/schemas/camera_track.py`
- Test: `tests/test_camera_track_schema.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_camera_track_schema.py`:

```python
@pytest.mark.unit
def test_camera_track_carries_distortion(tmp_path: Path):
    track = CameraTrack(
        clip_id="t",
        fps=25.0,
        image_size=(1920, 1080),
        t_world=[0.0, 0.0, 30.0],
        frames=tuple(),
        distortion=(0.12, -0.04),
    )
    out = tmp_path / "track.json"
    track.save(out)
    loaded = CameraTrack.load(out)
    assert loaded.distortion == (0.12, -0.04)


@pytest.mark.unit
def test_camera_track_legacy_load_distortion_default_zero(tmp_path: Path):
    out = tmp_path / "legacy.json"
    out.write_text(json.dumps({
        "clip_id": "legacy",
        "fps": 25.0,
        "image_size": [1920, 1080],
        "t_world": [0.0, 0.0, 30.0],
        "frames": [],
    }))
    loaded = CameraTrack.load(out)
    assert loaded.distortion == (0.0, 0.0)
```

- [ ] **Step 2: Run to verify it fails**

```bash
pytest tests/test_camera_track_schema.py::test_camera_track_carries_distortion -v
```

Expected: FAIL.

- [ ] **Step 3: Add `distortion` field**

In `src/schemas/camera_track.py`:

```python
@dataclass(frozen=True)
class CameraTrack:
    clip_id: str
    fps: float
    image_size: tuple[int, int]
    t_world: list[float]
    frames: tuple[CameraFrame, ...]
    principal_point: tuple[float, float] | None = None
    camera_centre: tuple[float, float, float] | None = None
    # Radial distortion coefficients (k1, k2). Default (0, 0) for
    # backward compatibility with tracks saved before lens distortion
    # was added to the calibration.
    distortion: tuple[float, float] = (0.0, 0.0)
```

In `load`:

```python
        dist_raw = data.get("distortion")
        distortion = tuple(dist_raw) if dist_raw is not None else (0.0, 0.0)
```

Pass to constructor:

```python
            distortion=distortion,
```

- [ ] **Step 4: Run to verify it passes**

```bash
pytest tests/test_camera_track_schema.py -v
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/schemas/camera_track.py tests/test_camera_track_schema.py
git commit -m "feat(schema): CameraTrack.distortion (k1, k2) with backward-compat default"
```

---

### Task 2.2: New `camera_projection.py` helper

**Files:**
- Create: `src/utils/camera_projection.py`
- Test: `tests/test_camera_projection.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_camera_projection.py`:

```python
"""Tests for the centralised world<->image projection helper."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.utils.camera_projection import (
    project_world_to_image,
    undistort_pixel,
)


@pytest.mark.unit
def test_project_round_trip_zero_distortion():
    K = np.array([[1500.0, 0.0, 960.0],
                  [0.0, 1500.0, 540.0],
                  [0.0, 0.0, 1.0]])
    R = np.eye(3)
    t = np.array([0.0, 0.0, 30.0])
    world = np.array([[0.0, 0.0, 0.0], [10.0, 5.0, 0.0]])
    proj = project_world_to_image(K, R, t, distortion=(0.0, 0.0), world_points=world)
    # With zero distortion, K * (RX + t) / z reproduces classical projection
    cam = world @ R.T + t
    expected = (K @ cam.T).T
    expected = expected[:, :2] / expected[:, 2:]
    assert np.allclose(proj, expected, atol=1e-4)


@pytest.mark.unit
def test_project_with_distortion_matches_cv2():
    K = np.array([[1500.0, 0.0, 960.0],
                  [0.0, 1500.0, 540.0],
                  [0.0, 0.0, 1.0]])
    R = np.eye(3)
    t = np.array([0.0, 0.0, 30.0])
    world = np.array([[0.0, 0.0, 0.0], [10.0, 5.0, 0.0]])
    proj = project_world_to_image(K, R, t, distortion=(0.1, -0.02), world_points=world)
    rvec, _ = cv2.Rodrigues(R)
    expected, _ = cv2.projectPoints(
        world.reshape(-1, 1, 3), rvec, t.reshape(3, 1), K,
        np.array([0.1, -0.02, 0.0, 0.0, 0.0]),
    )
    assert np.allclose(proj, expected.reshape(-1, 2), atol=1e-4)


@pytest.mark.unit
def test_undistort_pixel_inverts_distortion():
    K = np.array([[1500.0, 0.0, 960.0],
                  [0.0, 1500.0, 540.0],
                  [0.0, 0.0, 1.0]])
    R = np.eye(3)
    t = np.array([0.0, 0.0, 30.0])
    world = np.array([[10.0, 5.0, 0.0]])
    distorted = project_world_to_image(K, R, t, distortion=(0.1, -0.02), world_points=world)
    undist = undistort_pixel(distorted[0], K, distortion=(0.1, -0.02))
    # Linear-K projection of the same point
    cam = R @ world[0] + t
    linear = (K @ cam)[:2] / cam[2]
    assert np.allclose(undist, linear, atol=0.5)
```

- [ ] **Step 2: Run to verify it fails**

```bash
pytest tests/test_camera_projection.py -v
```

Expected: FAIL — module doesn't exist.

- [ ] **Step 3: Create `src/utils/camera_projection.py`**

```python
"""Centralised world<->image projection with optional radial distortion.

Single source of truth for `(K, R, t, distortion) → image` math across the
codebase. Avoids subtle bugs from a dozen call sites each reimplementing the
same pinhole math, only some of which honour distortion.
"""

from __future__ import annotations

import cv2
import numpy as np


def project_world_to_image(
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    distortion: tuple[float, float],
    world_points: np.ndarray,
) -> np.ndarray:
    """Project world-frame 3D points to image-plane 2D pixels.

    Args:
        K: 3x3 intrinsic matrix.
        R: 3x3 world->camera rotation.
        t: (3,) world->camera translation.
        distortion: (k1, k2) radial distortion. Use (0, 0) for pinhole.
        world_points: (N, 3) array of world-frame points.

    Returns:
        (N, 2) image-plane projections (u, v) in pixels.
    """
    pts = np.asarray(world_points, dtype=np.float64).reshape(-1, 1, 3)
    rvec, _ = cv2.Rodrigues(np.asarray(R, dtype=np.float64))
    tvec = np.asarray(t, dtype=np.float64).reshape(3, 1)
    k1, k2 = distortion
    dist = np.array([k1, k2, 0.0, 0.0, 0.0], dtype=np.float64)
    out, _ = cv2.projectPoints(pts, rvec, tvec, K.astype(np.float64), dist)
    return out.reshape(-1, 2)


def undistort_pixel(
    pixel_uv: tuple[float, float] | np.ndarray,
    K: np.ndarray,
    distortion: tuple[float, float],
) -> np.ndarray:
    """Map a distorted pixel to the linear-pinhole equivalent.

    Used by downstream stages (foot anchoring, ball ground-projection) that
    back-project a 2D pixel into a 3D ray and need to undo the lens
    distortion before applying the inverse pinhole.

    Returns a (2,) numpy array (u_lin, v_lin).
    """
    uv = np.asarray(pixel_uv, dtype=np.float64).reshape(1, 1, 2)
    k1, k2 = distortion
    dist = np.array([k1, k2, 0.0, 0.0, 0.0], dtype=np.float64)
    # `P=K` gives outputs back in pixel coordinates rather than normalised
    # camera coordinates — the natural form for downstream callers.
    undist = cv2.undistortPoints(uv, K.astype(np.float64), dist, P=K.astype(np.float64))
    return undist.reshape(2)
```

- [ ] **Step 4: Run to verify all three tests pass**

```bash
pytest tests/test_camera_projection.py -v
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/utils/camera_projection.py tests/test_camera_projection.py
git commit -m "feat(camera): centralise world->image projection with distortion"
```

---

### Task 2.3: Add `(k1, k2)` to `JointSolution` and joint LM

**Files:**
- Modify: `src/utils/anchor_solver.py`
- Test: `tests/test_anchor_solver.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_anchor_solver.py`:

```python
@pytest.mark.unit
def test_joint_solver_recovers_radial_distortion():
    """Synthesise observations with known (k1, k2); the joint LM recovers them."""
    from src.utils.camera_projection import project_world_to_image
    from src.utils.anchor_solver import refine_with_shared_translation

    K_true = _K()
    t_world = -R_BASE @ np.array([52.5, -30.0, 30.0])
    k1_true, k2_true = 0.10, -0.02
    rich_landmarks = (
        ("p1", (0.0, 0.0, 0.0)),
        ("p2", (105.0, 0.0, 0.0)),
        ("p3", (0.0, 68.0, 0.0)),
        ("p4", (105.0, 68.0, 0.0)),
        ("p5", (52.5, 34.0, 0.0)),
        ("p6", (0.0, 30.34, 2.44)),
    )
    anchors = []
    for frame, yaw in ((0, 0.0), (60, 5.0), (120, -5.0)):
        R = _yaw(yaw)
        t = -R @ np.array([52.5, -30.0, 30.0])
        proj = project_world_to_image(
            K_true, R, t, (k1_true, k2_true),
            np.array([w for _, w in rich_landmarks]),
        )
        anchors.append(Anchor(
            frame=frame,
            landmarks=tuple(
                LandmarkObservation(name=n, image_xy=tuple(proj[i]), world_xyz=w)
                for i, (n, w) in enumerate(rich_landmarks)
            ),
        ))
    sol = solve_anchors_jointly(tuple(anchors), image_size=IMAGE_SIZE)
    sol = refine_with_shared_translation(tuple(anchors), sol)

    assert sol.distortion is not None
    k1_est, k2_est = sol.distortion
    assert abs(k1_est - k1_true) < 0.02, f"k1: got {k1_est}, want {k1_true}"
    assert abs(k2_est - k2_true) < 0.02, f"k2: got {k2_est}, want {k2_true}"
```

- [ ] **Step 2: Run to verify it fails**

```bash
pytest tests/test_anchor_solver.py::test_joint_solver_recovers_radial_distortion -v
```

Expected: FAIL — `JointSolution` has no `distortion`; observations have a systemic bias the solver can't fit without distortion.

- [ ] **Step 3: Add `distortion` to `JointSolution`**

Edit `src/utils/anchor_solver.py:60-69`:

```python
class JointSolution(NamedTuple):
    t_world: np.ndarray
    principal_point: tuple[float, float]
    per_anchor_KRt: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]
    per_anchor_residual_px: dict[int, float]
    camera_centre: np.ndarray | None = None
    distortion: tuple[float, float] = (0.0, 0.0)
    """Clip-shared radial distortion (k1, k2). Default (0, 0) for solutions
    produced before distortion was added to the model. Bounded to |k|<0.5
    by the LM."""
```

- [ ] **Step 4: Add `(k1, k2)` to the joint param vector**

Find `_GLOBALS = 5` and update:

```python
_GLOBALS = 7    # tx, ty, tz, cx, cy, k1, k2
_PER_ANCHOR = 4  # rvec(3), fx(1)
```

Update `_pack_params` to include k1, k2:

```python
def _pack_params(
    t: np.ndarray,
    cx: float,
    cy: float,
    k1: float,
    k2: float,
    rvecs: list[np.ndarray],
    fxs: list[float],
) -> np.ndarray:
    out = np.empty(_GLOBALS + _PER_ANCHOR * len(rvecs))
    out[:3] = t
    out[3] = cx
    out[4] = cy
    out[5] = k1
    out[6] = k2
    for i, (rv, fx) in enumerate(zip(rvecs, fxs)):
        base = _GLOBALS + i * _PER_ANCHOR
        out[base : base + 3] = rv
        out[base + 3] = fx
    return out
```

And `_unpack_params`:

```python
def _unpack_params(p: np.ndarray, n_anchors: int) -> tuple[np.ndarray, float, float, float, float, list[np.ndarray], list[float]]:
    t = p[:3]
    cx = float(p[3])
    cy = float(p[4])
    k1 = float(p[5])
    k2 = float(p[6])
    rvecs: list[np.ndarray] = []
    fxs: list[float] = []
    for i in range(n_anchors):
        base = _GLOBALS + i * _PER_ANCHOR
        rvecs.append(p[base : base + 3])
        fxs.append(float(p[base + 3]))
    return t, cx, cy, k1, k2, rvecs, fxs
```

- [ ] **Step 5: Use `cv2.projectPoints` in residuals**

Replace `_point_residuals`:

```python
def _point_residuals(
    points: list[LandmarkObservation],
    K: np.ndarray, R: np.ndarray, t: np.ndarray,
    distortion: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """2 residuals per point: (proj_u - obs_u, proj_v - obs_v) using
    cv2.projectPoints so radial distortion is honoured."""
    if not points:
        return np.empty(0)
    world = np.array([lm.world_xyz for lm in points], dtype=np.float64)
    obs = np.array([lm.image_xy for lm in points], dtype=np.float64)
    cam = world @ R.T + t
    behind = cam[:, 2] <= 1e-3
    rvec, _ = cv2.Rodrigues(R)
    k1, k2 = distortion
    dist = np.array([k1, k2, 0.0, 0.0, 0.0], dtype=np.float64)
    proj, _ = cv2.projectPoints(
        world.reshape(-1, 1, 3), rvec, t.reshape(3, 1), K, dist,
    )
    proj = proj.reshape(-1, 2)
    out = (proj - obs).reshape(-1)
    if behind.any():
        scale = np.ones(len(world))
        scale[behind] = 1e3
        out = (out.reshape(-1, 2) * scale[:, None]).reshape(-1)
    return out
```

Update `_anchor_residuals` and `_residuals` to thread distortion through:

```python
def _anchor_residuals(
    anchor: Anchor, K: np.ndarray, R: np.ndarray, t: np.ndarray,
    distortion: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    return np.concatenate([
        _point_residuals(list(anchor.landmarks), K, R, t, distortion),
        _line_residuals(list(anchor.lines), K, R, t),
    ])


def _residuals(
    p: np.ndarray, anchors: tuple[Anchor, ...]
) -> np.ndarray:
    t, cx, cy, k1, k2, rvecs, fxs = _unpack_params(p, len(anchors))
    parts: list[np.ndarray] = []
    for anchor, rvec, fx in zip(anchors, rvecs, fxs):
        K = _make_K(fx, cx, cy)
        R = _rvec_to_R(rvec)
        parts.append(_anchor_residuals(anchor, K, R, t, (k1, k2)))
    return np.concatenate(parts) if parts else np.empty(0)
```

- [ ] **Step 6: Update `_jac_sparsity`**

Each residual now also depends on the 2 new global cols (k1, k2). The current code already loops `for c in range(_GLOBALS)` setting all global cols to 1, so `_GLOBALS = 7` makes this work automatically — no further change needed in `_jac_sparsity`. (Skim the function to confirm.)

- [ ] **Step 7: Wire the joint LM call**

Find `solve_anchors_jointly`. Locate where it currently calls `least_squares` (search for `least_squares` in the file). The hybrid v2 solver's main path uses solo + t-fixed solves rather than a single joint LM, but the joint LM call still exists for the seed/global refinement. **Read the current state first** before editing:

```bash
grep -n "least_squares\|_pack_params\|_unpack_params" src/utils/anchor_solver.py
```

Update each `_pack_params(...)` and `_unpack_params(...)` call site to pass/return `k1, k2`. Initial seeds `k1=0.0, k2=0.0`.

- [ ] **Step 8: Bound k1, k2 after LM**

After the joint `least_squares` call, clip the recovered `(k1, k2)`:

```python
k1 = float(np.clip(result.x[5], -0.5, 0.5))
k2 = float(np.clip(result.x[6], -0.5, 0.5))
```

- [ ] **Step 9: Populate `distortion` on the returned `JointSolution`**

In each `return JointSolution(...)` site (both `solve_anchors_jointly` and `refine_with_shared_translation`), pass `distortion=(k1, k2)`. For `refine_with_shared_translation`, distortion is propagated from input `sol` unchanged (the relock doesn't refine k1/k2):

```python
    return JointSolution(
        ...,
        distortion=sol.distortion,
    )
```

- [ ] **Step 10: Run distortion test**

```bash
pytest tests/test_anchor_solver.py::test_joint_solver_recovers_radial_distortion -v
```

Expected: PASS.

- [ ] **Step 11: Run full anchor_solver suite**

```bash
pytest tests/test_anchor_solver.py -v
```

Expected: all PASS.

- [ ] **Step 12: Commit**

```bash
git add src/utils/anchor_solver.py tests/test_anchor_solver.py
git commit -m "feat(camera): radial distortion (k1, k2) in joint bundle adjustment"
```

---

### Task 2.4: Restore line-residual weight to 1.0 with Huber loss

**Files:**
- Modify: `src/utils/anchor_solver.py`
- Test: `tests/test_anchor_solver.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_anchor_solver.py`:

```python
@pytest.mark.unit
def test_huber_loss_dampens_one_bad_landmark():
    """A single 200-pixel outlier landmark must not move the recovered fx
    by more than 1% relative to the no-outlier baseline."""
    from src.utils.anchor_solver import refine_with_shared_translation

    base = _three_rich_anchors_static()
    # Inject a 200px bad click into the first anchor's first landmark
    bad_first = base[0]
    bad_lm = bad_first.landmarks[0]
    bad_lm = LandmarkObservation(
        name=bad_lm.name,
        image_xy=(bad_lm.image_xy[0] + 200.0, bad_lm.image_xy[1] + 200.0),
        world_xyz=bad_lm.world_xyz,
    )
    bad_anchor = Anchor(
        frame=bad_first.frame,
        landmarks=(bad_lm, *bad_first.landmarks[1:]),
    )
    bad_set = (bad_anchor, *base[1:])

    sol_clean = solve_anchors_jointly(base, image_size=IMAGE_SIZE)
    sol_bad = solve_anchors_jointly(bad_set, image_size=IMAGE_SIZE)
    fx_clean = sol_clean.per_anchor_KRt[0][0][0, 0]
    fx_bad = sol_bad.per_anchor_KRt[0][0][0, 0]
    rel = abs(fx_bad - fx_clean) / fx_clean
    assert rel < 0.01, f"fx moved {rel*100:.2f}% — Huber not dampening outlier"
```

- [ ] **Step 2: Run to verify it fails**

```bash
pytest tests/test_anchor_solver.py::test_huber_loss_dampens_one_bad_landmark -v
```

Expected: FAIL — current LM (no robust loss) lets the outlier shift fx.

- [ ] **Step 3: Switch joint LM to Huber loss**

Find every `least_squares(...)` call in `src/utils/anchor_solver.py`. For each, add `loss="huber", f_scale=2.0`:

```python
result = least_squares(
    _residuals, p0, args=(anchors,),
    jac_sparsity=_jac_sparsity(anchors),
    method="trf", loss="huber", f_scale=2.0, max_nfev=300,
)
```

(The hybrid solver has solo LMs in `_refine_seed_pose`, `_solve_anchor_with_C_fixed`, `_solve_anchor_with_t_fixed` — switch them all.)

Note: `method="lm"` does not accept `loss="huber"`. Switch any `method="lm"` call to `method="trf"` when adding Huber.

- [ ] **Step 4: Drop `_LINE_RESIDUAL_WEIGHT` and the `use_lines_on_seed` heuristic**

Remove the constant:

```python
_LINE_RESIDUAL_WEIGHT = 0.2   # ← delete this line
```

Replace each occurrence of `_LINE_RESIDUAL_WEIGHT * _line_residuals(...)` with just `_line_residuals(...)`.

In `_refine_seed_pose`, remove the `use_lines_on_seed` branch — always include lines:

```python
def _refine_seed_pose(...):
    ...
    def _residuals(p: np.ndarray) -> np.ndarray:
        ...
        parts: list[np.ndarray] = [(proj - img_pts).reshape(-1)]
        if anchor.lines:
            parts.append(_line_residuals(list(anchor.lines), K, R, tvec))
        return np.concatenate(parts)
    ...
```

- [ ] **Step 5: Run the dampening test**

```bash
pytest tests/test_anchor_solver.py::test_huber_loss_dampens_one_bad_landmark -v
```

Expected: PASS.

- [ ] **Step 6: Run full suite**

```bash
pytest tests/test_anchor_solver.py -v
```

Expected: all PASS. (Existing tests were tuned with line-weight 0.2; if any tightens because lines now contribute fully, adjust the assertion tolerance — but only if the LOOSER tolerance was the one passing before.)

- [ ] **Step 7: Commit**

```bash
git add src/utils/anchor_solver.py tests/test_anchor_solver.py
git commit -m "refactor(camera): Huber loss + full line-residual weight"
```

---

### Task 2.5: Tighten anchor-residual threshold

**Files:**
- Modify: `config/default.yaml`

- [ ] **Step 1: Edit the threshold**

In `config/default.yaml`, change:

```yaml
  anchor_max_reprojection_px: 10.0
```

to:

```yaml
  anchor_max_reprojection_px: 2.0
```

- [ ] **Step 2: Run all camera tests**

```bash
pytest tests/test_camera_stage.py tests/test_anchor_solver.py -v
```

Expected: all PASS. (If any test was hard-coding 10.0, fix the test.)

- [ ] **Step 3: Commit**

```bash
git add config/default.yaml
git commit -m "config(camera): tighten anchor_max_reprojection_px to 2.0"
```

---

### Task 2.6: Undistort foot anchor in `hmr_world.py`

**Files:**
- Modify: `src/stages/hmr_world.py`
- Test: `tests/test_foot_anchor.py`

- [ ] **Step 1: Locate the ankle pixel back-projection**

```bash
grep -n "ankle\|foot.*anchor\|undistort\|inverse.*K\|back.*project" src/stages/hmr_world.py src/utils/foot_anchor.py 2>/dev/null
```

- [ ] **Step 2: Write the failing test**

Add to `tests/test_foot_anchor.py` (or wherever foot anchoring is tested):

```python
@pytest.mark.unit
def test_foot_anchor_undistorts_pixel_before_back_projection():
    """Given a known camera with non-trivial k1, the foot-anchor pitch
    intersection must use the undistorted pixel to back-project. The test
    constructs a world point on the pitch, distorts its projection, then
    asserts the back-projected ray hits within 5cm of the original."""
    from src.utils.foot_anchor import project_pixel_to_pitch  # adjust to actual API
    from src.utils.camera_projection import project_world_to_image

    K = np.array([[1500.0, 0.0, 960.0], [0.0, 1500.0, 540.0], [0.0, 0.0, 1.0]])
    R = R_BASE
    t = T_BASE
    distortion = (0.10, -0.02)
    world = np.array([52.5, 34.0, 0.0])

    pixel = project_world_to_image(K, R, t, distortion, world.reshape(1, 3))[0]
    recovered = project_pixel_to_pitch(pixel, K, R, t, distortion=distortion)
    assert np.allclose(recovered, world, atol=0.05)
```

(Adjust `project_pixel_to_pitch` to match the actual function signature; if the codebase uses a different name, search for the existing back-projection function name. If the existing function does not take `distortion`, the test forces adding the kwarg.)

- [ ] **Step 3: Run to verify it fails**

```bash
pytest tests/test_foot_anchor.py -v -k undistort
```

Expected: FAIL.

- [ ] **Step 4: Update the back-projection function**

In the function that converts an ankle pixel to a pitch-plane intersection: insert an undistort step at the top.

```python
from src.utils.camera_projection import undistort_pixel

def project_pixel_to_pitch(
    pixel_uv: tuple[float, float],
    K: np.ndarray, R: np.ndarray, t: np.ndarray,
    distortion: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """..."""
    if distortion != (0.0, 0.0):
        pixel_uv = tuple(undistort_pixel(pixel_uv, K, distortion))
    # ... existing back-projection logic unchanged
```

Then update every call site in `hmr_world.py` to pass `distortion=track.distortion` (where `track` is the loaded `CameraTrack`).

- [ ] **Step 5: Run the test**

```bash
pytest tests/test_foot_anchor.py -v -k undistort
```

Expected: PASS.

- [ ] **Step 6: Run all hmr_world tests**

```bash
pytest tests/test_foot_anchor.py tests/test_hmr_world_stage.py -v
```

Expected: all PASS.

- [ ] **Step 7: Commit**

```bash
git add src/stages/hmr_world.py src/utils/foot_anchor.py tests/test_foot_anchor.py
git commit -m "fix(hmr): undistort ankle pixel before pitch back-projection"
```

---

### Task 2.7: Undistort ball pixel in `ball.py`

**Files:**
- Modify: `src/stages/ball.py`
- Test: `tests/test_ball_grounded.py` or `test_ball_stage.py`

- [ ] **Step 1: Locate ball ground projection**

```bash
grep -n "undistort\|inverse.*K\|project.*pitch\|back.*project" src/stages/ball.py src/utils/ball_detector.py
```

- [ ] **Step 2: Write the failing test**

Add to whichever ball test file already exists for ground projection (read it first, follow conventions):

```python
@pytest.mark.unit
def test_ball_ground_projection_undistorts_pixel():
    from src.stages.ball import _project_ball_to_ground  # adjust to actual API
    from src.utils.camera_projection import project_world_to_image

    K = np.array([[1500.0, 0.0, 960.0], [0.0, 1500.0, 540.0], [0.0, 0.0, 1.0]])
    R = R_BASE
    t = T_BASE
    distortion = (0.10, -0.02)
    world = np.array([60.0, 30.0, 0.11])  # ball radius

    pixel = project_world_to_image(K, R, t, distortion, world.reshape(1, 3))[0]
    recovered = _project_ball_to_ground(
        pixel, K=K, R=R, t=t, ball_radius=0.11, distortion=distortion,
    )
    assert np.allclose(recovered, world, atol=0.10)
```

- [ ] **Step 3-7: Same RED→GREEN→commit pattern as Task 2.6**

```bash
pytest tests/test_ball_grounded.py -v -k undistort
# expect FAIL → add undistort step → expect PASS
git add src/stages/ball.py tests/test_ball_grounded.py
git commit -m "fix(ball): undistort pixel before ground projection"
```

---

### Task 2.8: Wire `distortion` into `CameraTrack` save in camera stage

**Files:**
- Modify: `src/stages/camera.py`

- [ ] **Step 1: Pass `sol.distortion` into `CameraTrack`**

Find the `track = CameraTrack(...)` block and add:

```python
        track = CameraTrack(
            ...,
            camera_centre=...,
            distortion=tuple(float(x) for x in sol.distortion),
        )
```

- [ ] **Step 2: Run camera-stage tests**

```bash
pytest tests/test_camera_stage.py -v
```

Expected: all PASS.

- [ ] **Step 3: Commit**

```bash
git add src/stages/camera.py
git commit -m "feat(camera): persist distortion to camera_track.json"
```

---

### Task 2.9: Apply distortion to viewer pitch overlay

**Files:**
- Modify: `src/web/static/viewer.html`
- Modify: `src/web/static/anchor_editor.html` (if it draws an overlay for finished tracks)

- [ ] **Step 1: Add JS distortion helper**

Find the existing `projectWorldToImage` (or similar) helper in `viewer.html`. Add:

```javascript
function applyRadialDistortion(uv, k1, k2, cx, cy, fx) {
  // OpenCV-compatible 2-coefficient radial model.
  if ((k1 === 0 && k2 === 0)) return uv;
  const x = (uv[0] - cx) / fx;
  const y = (uv[1] - cy) / fx;
  const r2 = x * x + y * y;
  const factor = 1 + k1 * r2 + k2 * r2 * r2;
  return [cx + fx * x * factor, cy + fx * y * factor];
}
```

In the existing pitch-line draw loop, after computing the linear projection of each vertex, run it through `applyRadialDistortion` using `(track.distortion[0], track.distortion[1], track.principal_point[0], track.principal_point[1], frame.K[0][0])`.

- [ ] **Step 2: Manual verification**

```bash
python recon.py serve --output ./output/
```

Open the dashboard, watch the overlay tracks the pitch lines on a frame with non-zero distortion. (No automated test for the JS — the integration test in Phase 3 verifies the projection metric end-to-end.)

- [ ] **Step 3: Commit**

```bash
git add src/web/static/viewer.html src/web/static/anchor_editor.html
git commit -m "feat(web): apply radial distortion to projected pitch overlay"
```

---

### Task 2.10: Phase 2 milestone — full pipeline check

- [ ] **Step 1: Run a full pipeline against a checkpointed clip**

```bash
python recon.py run --input <fixture-clip> --output /tmp/p2_check/ --from-stage camera
```

(Use whatever clip you have under `output/` from the existing checkpoint.)

- [ ] **Step 2: Inspect quality report**

```bash
cat /tmp/p2_check/camera/camera_quality.json
```

Expected: `body_drift_max_m == 0.0`, `anchor_residual_mean_px < 2.0`.

- [ ] **Step 3: Visual sanity in dashboard**

```bash
python recon.py serve --output /tmp/p2_check/
```

Open `http://localhost:8000/viewer`, scrub through the clip. Expect noticeably smaller projected-vs-real gap than baseline.

- [ ] **Step 4: Tag**

```bash
git tag phase-2-distortion
```

---

## Phase 3 — Anchor-frame line polish

### Task 3.1: New `line_detector.py` module

**Files:**
- Create: `src/utils/line_detector.py`
- Test: `tests/test_line_detector.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_line_detector.py`:

```python
"""Tests for white-on-green pitch-line segment detection."""

from __future__ import annotations

import numpy as np
import pytest

from src.utils.line_detector import detect_pitch_line_segments


def _synthetic_pitch_frame() -> np.ndarray:
    """1280x720 BGR frame: green field with two white horizontal lines."""
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    # Green field
    frame[:] = (40, 120, 40)
    # Two white horizontal lines (BGR)
    frame[300:303, 100:1180] = (255, 255, 255)
    frame[500:503, 200:1080] = (255, 255, 255)
    return frame


@pytest.mark.unit
def test_detects_two_horizontal_lines():
    frame = _synthetic_pitch_frame()
    segs = detect_pitch_line_segments(frame, min_length_px=200.0)
    assert len(segs) >= 2, f"expected >=2, got {len(segs)}"
    # Both lines roughly horizontal (angle near 0 or pi)
    for s in segs:
        ang = abs(s.angle_rad)
        is_horizontal = ang < 0.1 or abs(ang - np.pi) < 0.1
        assert is_horizontal, f"non-horizontal segment {s}"


@pytest.mark.unit
def test_rejects_non_pitch_white():
    """White marks outside the green ROI must be rejected."""
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    frame[:] = (200, 50, 50)  # blue background — not green
    frame[300:303, 100:1180] = (255, 255, 255)
    segs = detect_pitch_line_segments(frame, min_length_px=200.0)
    assert len(segs) == 0
```

- [ ] **Step 2: Run to verify it fails**

```bash
pytest tests/test_line_detector.py -v
```

Expected: FAIL — module doesn't exist.

- [ ] **Step 3: Create the module**

```python
"""White-on-green pitch line segment detection.

Pure function — no camera knowledge. Used by anchor-frame line polish to
auto-detect pitch lines for matching against the FIFA catalogue.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class ImageLineSegment:
    p0: tuple[float, float]
    p1: tuple[float, float]
    length_px: float
    angle_rad: float


# HSV gates — tunable via config in Phase 3.4 wiring
_GREEN_H_RANGE = (35, 90)
_GREEN_S_MIN = 40
_GREEN_V_MIN = 40
_WHITE_V_MIN = 200
_WHITE_S_MAX = 60


def detect_pitch_line_segments(
    frame_bgr: np.ndarray,
    min_length_px: float = 40.0,
) -> list[ImageLineSegment]:
    """Detect white line segments lying on green pitch surface."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    green_mask = (
        (h >= _GREEN_H_RANGE[0]) & (h <= _GREEN_H_RANGE[1])
        & (s >= _GREEN_S_MIN) & (v >= _GREEN_V_MIN)
    ).astype(np.uint8) * 255
    green_mask = cv2.dilate(green_mask, np.ones((5, 5), np.uint8))

    white_mask = ((v >= _WHITE_V_MIN) & (s <= _WHITE_S_MAX)).astype(np.uint8) * 255
    white_mask = cv2.erode(white_mask, np.ones((3, 3), np.uint8))

    pitch_white = cv2.bitwise_and(white_mask, green_mask)

    # Prefer FastLineDetector (ximgproc) when available; fall back to LSD.
    try:
        fld = cv2.ximgproc.createFastLineDetector(
            length_threshold=int(min_length_px),
            distance_threshold=1.41,
            canny_th1=50,
            canny_th2=200,
            canny_aperture_size=3,
            do_merge=False,
        )
        segments = fld.detect(pitch_white)
    except (AttributeError, cv2.error):
        lsd = cv2.createLineSegmentDetector()
        segments, _, _, _ = lsd.detect(pitch_white)

    if segments is None:
        return []

    out: list[ImageLineSegment] = []
    for s in segments.reshape(-1, 4):
        x0, y0, x1, y1 = s
        length = float(np.hypot(x1 - x0, y1 - y0))
        if length < min_length_px:
            continue
        angle = float(np.arctan2(y1 - y0, x1 - x0))
        out.append(ImageLineSegment(
            p0=(float(x0), float(y0)),
            p1=(float(x1), float(y1)),
            length_px=length,
            angle_rad=angle,
        ))
    return out
```

- [ ] **Step 4: Run the tests**

```bash
pytest tests/test_line_detector.py -v
```

Expected: PASS. If `cv2.ximgproc` is missing in the env, the fallback to LSD covers the test; if both fail, install opencv-contrib-python and re-run.

- [ ] **Step 5: Commit**

```bash
git add src/utils/line_detector.py tests/test_line_detector.py
git commit -m "feat(camera): white-on-green pitch line segment detector"
```

---

### Task 3.2: New `anchor_line_polish.py` module

**Files:**
- Create: `src/utils/anchor_line_polish.py`
- Test: `tests/test_anchor_line_polish.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_anchor_line_polish.py`:

```python
"""Tests for per-anchor LM refinement against detected pitch lines."""

from __future__ import annotations

import numpy as np
import pytest

from src.utils.anchor_line_polish import polish_anchor_against_lines
from src.utils.camera_projection import project_world_to_image
from src.utils.pitch_lines_catalogue import LINE_CATALOGUE


# Reuse fixtures from test_anchor_solver via a small helper module if needed,
# else inline the constants.
R_BASE_LOCAL = np.array([
    [1.0, 0.0, 0.0],
    [0.0, 0.0 - 0.0, -1.0],
    [0.0, 0.9061277, -0.4229554],
])
# Use a properly orthonormal R in the actual test — copy from test_anchor_solver
# helpers if available; else build via a 'look-at' construction.
IMAGE_SIZE = (1920, 1080)


def _render_synthetic_anchor(yaw_deg: float, k1: float = 0.0, k2: float = 0.0):
    # Reuse the helper pattern from test_anchor_solver.
    from tests.test_anchor_solver import _yaw, _K, R_BASE
    R = _yaw(yaw_deg)
    C = np.array([52.5, -30.0, 30.0])
    t = -R @ C
    K = _K()
    # Render a frame: solid green with white pitch lines projected through the
    # current pose (with distortion).
    frame = np.zeros((IMAGE_SIZE[1], IMAGE_SIZE[0], 3), dtype=np.uint8)
    frame[:] = (40, 120, 40)
    for name in ("near_touchline", "far_touchline", "halfway_line",
                 "left_18yd_front", "right_18yd_front"):
        a, b = LINE_CATALOGUE[name]
        proj = project_world_to_image(
            K, R, t, (k1, k2), np.array([a, b], dtype=np.float64),
        )
        p0 = tuple(int(round(x)) for x in proj[0])
        p1 = tuple(int(round(x)) for x in proj[1])
        cv2.line(frame, p0, p1, (255, 255, 255), 3)
    return frame, K, R, t, C


@pytest.mark.unit
def test_polish_recovers_small_R_perturbation():
    """A 1-degree rotation perturbation must be undone to within 0.1 degrees."""
    import cv2
    frame, K, R_true, t_true, C = _render_synthetic_anchor(0.0)
    # Perturb R by 1 degree about z
    rvec_true, _ = cv2.Rodrigues(R_true)
    perturb_deg = 1.0
    perturb = np.array([0.0, 0.0, np.deg2rad(perturb_deg)])
    rvec_pert = rvec_true.flatten() + perturb
    R_pert, _ = cv2.Rodrigues(rvec_pert)
    t_pert = -R_pert @ C
    polished = polish_anchor_against_lines(
        frame, K, R_pert, t_pert,
        distortion=(0.0, 0.0),
        pitch_lines=[(name, *LINE_CATALOGUE[name]) for name in
                     ("near_touchline", "far_touchline", "halfway_line",
                      "left_18yd_front", "right_18yd_front")],
        image_size=IMAGE_SIZE,
        camera_centre=C,
    )
    assert polished is not None, "polish returned None on rendered frame"
    K_p, R_p, t_p = polished
    # Compare rotations via Frobenius norm of R_p^T @ R_true (~ identity)
    err = np.linalg.norm(R_p.T @ R_true - np.eye(3))
    assert err < 0.005, f"rotation error {err} after polish"


@pytest.mark.unit
def test_polish_returns_none_when_residual_worsens():
    """Sanity guard: if the matched residual would increase, return None."""
    import cv2
    frame = np.zeros((IMAGE_SIZE[1], IMAGE_SIZE[0], 3), dtype=np.uint8)
    frame[:] = (40, 120, 40)  # No white lines drawn — no matches possible
    from tests.test_anchor_solver import _yaw, _K
    K = _K()
    R = _yaw(0.0)
    C = np.array([52.5, -30.0, 30.0])
    t = -R @ C
    polished = polish_anchor_against_lines(
        frame, K, R, t, distortion=(0.0, 0.0),
        pitch_lines=[("near_touchline", *LINE_CATALOGUE["near_touchline"])],
        image_size=IMAGE_SIZE,
        camera_centre=C,
    )
    assert polished is None
```

- [ ] **Step 2: Run to verify it fails**

```bash
pytest tests/test_anchor_line_polish.py -v
```

Expected: FAIL — module doesn't exist.

- [ ] **Step 3: Create the module**

```python
"""Per-anchor LM refinement against auto-detected pitch lines.

Runs after the joint anchor solve. For each anchor frame, detects pitch line
segments via line_detector, matches them to the FIFA catalogue using the
current pose, and refines (R, fx) — keeping (k1, k2, cx, cy) clip-shared and
t = -R @ camera_centre to honour the static-camera invariant.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np
from scipy.optimize import least_squares

from src.utils.camera_projection import project_world_to_image
from src.utils.line_detector import ImageLineSegment, detect_pitch_line_segments

logger = logging.getLogger(__name__)


# Match gates — tunable via Phase 3.4 config wiring
_MATCH_MAX_MIDPOINT_DISTANCE_PX = 10.0
_MATCH_MAX_ANGLE_DIFF_DEG = 5.0


def polish_anchor_against_lines(
    anchor_frame_bgr: np.ndarray,
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    distortion: tuple[float, float],
    pitch_lines: list[tuple[str, tuple[float, float, float], tuple[float, float, float]]],
    image_size: tuple[int, int],
    camera_centre: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Refine (R, fx) for one anchor frame by matching detected pitch lines
    to the catalogue. Returns refined (K, R, t) or None if too few matches.

    t is rebuilt internally as -R @ camera_centre during LM and on return.
    """
    detected = detect_pitch_line_segments(anchor_frame_bgr, min_length_px=40.0)
    if len(detected) < 2:
        return None

    matched = _match_detected_to_catalogue(
        detected, pitch_lines, K, R, t, distortion, image_size,
    )
    if len(matched) < 2:
        return None

    # Pre-LM mean residual
    res_before = _line_match_residual(matched, K, R, t, distortion)

    rvec_init, _ = cv2.Rodrigues(R)
    fx_init = float(K[0, 0])
    cx, cy = float(K[0, 2]), float(K[1, 2])

    def _residuals(p: np.ndarray) -> np.ndarray:
        rvec = p[:3]
        fx = float(np.clip(p[3], 50.0, 1e5))
        R_p, _ = cv2.Rodrigues(rvec)
        t_p = -R_p @ camera_centre
        K_p = np.array([[fx, 0.0, cx], [0.0, fx, cy], [0.0, 0.0, 1.0]])
        return _build_perpendicular_residuals(matched, K_p, R_p, t_p, distortion)

    p0 = np.concatenate([rvec_init.flatten(), [fx_init]])
    try:
        result = least_squares(
            _residuals, p0, method="trf", loss="huber", f_scale=2.0,
            max_nfev=200,
        )
    except Exception as exc:
        logger.warning("polish LM failed: %s", exc)
        return None

    rvec_new = result.x[:3]
    fx_new = float(np.clip(result.x[3], 50.0, 1e5))
    R_new, _ = cv2.Rodrigues(rvec_new)
    t_new = -R_new @ camera_centre
    K_new = np.array([[fx_new, 0.0, cx], [0.0, fx_new, cy], [0.0, 0.0, 1.0]])

    res_after = _line_match_residual(matched, K_new, R_new, t_new, distortion)
    if res_after >= res_before:
        logger.info(
            "polish sanity guard: residual didn't improve (%.2f → %.2f) — keeping original",
            res_before, res_after,
        )
        return None

    logger.info(
        "polish: %d matches, residual %.2f → %.2f px",
        len(matched), res_before, res_after,
    )
    return K_new, R_new, t_new


def _match_detected_to_catalogue(
    detected: list[ImageLineSegment],
    pitch_lines: list[tuple[str, tuple, tuple]],
    K, R, t, distortion, image_size,
):
    """Return list of (ImageLineSegment, world_a, world_b) pairs that pass
    midpoint-distance and angle gates."""
    w, h = image_size
    matched = []
    # Project each catalogue line into the image
    proj_lines = []
    for name, a, b in pitch_lines:
        proj = project_world_to_image(
            K, R, t, distortion, np.array([a, b], dtype=np.float64),
        )
        if not _segment_visible(proj, w, h):
            continue
        proj_lines.append((name, a, b, proj))

    for det in detected:
        candidates = []
        for name, a, b, proj in proj_lines:
            mid_dist = _perp_distance_to_line(
                np.array([(det.p0[0] + det.p1[0]) * 0.5,
                          (det.p0[1] + det.p1[1]) * 0.5]),
                proj[0], proj[1],
            )
            if mid_dist > _MATCH_MAX_MIDPOINT_DISTANCE_PX:
                continue
            proj_angle = float(np.arctan2(proj[1, 1] - proj[0, 1],
                                          proj[1, 0] - proj[0, 0]))
            angle_diff = abs(_wrap_angle(proj_angle - det.angle_rad))
            if angle_diff > np.deg2rad(_MATCH_MAX_ANGLE_DIFF_DEG):
                continue
            candidates.append((mid_dist, name, a, b))
        if not candidates:
            continue
        candidates.sort(key=lambda c: c[0])
        # Reject ambiguous (closest two within 30%)
        if len(candidates) >= 2 and candidates[1][0] < candidates[0][0] * 1.3:
            continue
        _, name, a, b = candidates[0]
        matched.append((det, a, b))
    return matched


def _segment_visible(proj: np.ndarray, w: int, h: int) -> bool:
    """At least one endpoint lies within image bounds (with margin)."""
    margin = 50
    for u, v in proj:
        if -margin <= u <= w + margin and -margin <= v <= h + margin:
            return True
    return False


def _perp_distance_to_line(point: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    norm = float(np.linalg.norm(ab))
    if norm < 1e-6:
        return float("inf")
    nx = -ab[1] / norm
    ny = ab[0] / norm
    c = -(nx * a[0] + ny * a[1])
    return abs(nx * point[0] + ny * point[1] + c)


def _wrap_angle(a: float) -> float:
    """Wrap to [-pi/2, pi/2] so that lines and their reverses match."""
    while a > np.pi / 2:
        a -= np.pi
    while a < -np.pi / 2:
        a += np.pi
    return a


def _build_perpendicular_residuals(
    matched, K, R, t, distortion,
) -> np.ndarray:
    """For each matched (detected, world_a, world_b), 2 residuals = perp
    distance from each detected endpoint to the projected world line."""
    if not matched:
        return np.empty(0)
    out = np.empty(2 * len(matched))
    for i, (det, a, b) in enumerate(matched):
        proj = project_world_to_image(
            K, R, t, distortion, np.array([a, b], dtype=np.float64),
        )
        for j, p in enumerate((det.p0, det.p1)):
            out[2 * i + j] = _perp_distance_to_line(np.asarray(p), proj[0], proj[1])
    return out


def _line_match_residual(matched, K, R, t, distortion) -> float:
    """Mean of perpendicular residuals — used by the sanity guard."""
    res = _build_perpendicular_residuals(matched, K, R, t, distortion)
    return float(np.mean(np.abs(res))) if len(res) else float("inf")
```

- [ ] **Step 4: Run polish tests**

```bash
pytest tests/test_anchor_line_polish.py -v
```

Expected: both PASS.

- [ ] **Step 5: Commit**

```bash
git add src/utils/anchor_line_polish.py tests/test_anchor_line_polish.py
git commit -m "feat(camera): per-anchor LM polish against detected pitch lines"
```

---

### Task 3.3: Wire polish into camera stage + config

**Files:**
- Modify: `src/stages/camera.py`
- Modify: `config/default.yaml`
- Test: `tests/test_camera_stage.py`

- [ ] **Step 1: Add config block**

In `config/default.yaml`, under `camera:`:

```yaml
  anchor_line_polish: true
  line_detection:
    min_segment_length_px: 40
  line_match:
    max_midpoint_distance_px: 10
    max_angle_diff_deg: 5
```

- [ ] **Step 2: Write the failing test**

Add to `tests/test_camera_stage.py`:

```python
@pytest.mark.unit
def test_anchor_line_polish_invoked_when_enabled(monkeypatch, tmp_output_with_static_clip):
    """When anchor_line_polish=true, polish_anchor_against_lines is called for
    each anchor frame."""
    from src.stages import camera as camera_module

    calls = []
    def fake_polish(*args, **kwargs):
        calls.append(args)
        return None  # leave anchors unchanged
    monkeypatch.setattr(camera_module, "polish_anchor_against_lines", fake_polish)

    stage = camera_module.CameraStage(
        output_dir=tmp_output_with_static_clip,
        config={"camera": {"static_camera": True, "anchor_line_polish": True}},
    )
    stage.run()
    assert len(calls) > 0
```

- [ ] **Step 3: Run to verify it fails**

```bash
pytest tests/test_camera_stage.py::test_anchor_line_polish_invoked_when_enabled -v
```

Expected: FAIL.

- [ ] **Step 4: Wire into `CameraStage.run`**

In `src/stages/camera.py`, after the relock and **before** the inter-anchor interpolation loop:

```python
        from src.utils.anchor_line_polish import polish_anchor_against_lines
        from src.utils.pitch_lines_catalogue import LINE_CATALOGUE

        if cfg.get("anchor_line_polish", True) and sol.camera_centre is not None:
            pitch_lines = [
                (name, seg[0], seg[1])
                for name, seg in LINE_CATALOGUE.items()
            ]
            for af in list(anchor_solutions.keys()):
                cap.set(cv2.CAP_PROP_POS_FRAMES, af)
                ok, frame_bgr = cap.read()
                if not ok:
                    continue
                K_a, R_a, t_a = anchor_solutions[af]
                polished = polish_anchor_against_lines(
                    frame_bgr, K_a, R_a, t_a,
                    distortion=sol.distortion,
                    pitch_lines=pitch_lines,
                    image_size=(w, h),
                    camera_centre=np.asarray(sol.camera_centre),
                )
                if polished is not None:
                    anchor_solutions[af] = polished
```

(Make sure `cap` is still open at this point — current code releases it later. If it's already released, re-open or move the polish step before the release.)

- [ ] **Step 5: Run the test**

```bash
pytest tests/test_camera_stage.py::test_anchor_line_polish_invoked_when_enabled -v
```

Expected: PASS.

- [ ] **Step 6: Run full camera-stage suite**

```bash
pytest tests/test_camera_stage.py tests/test_anchor_solver.py tests/test_anchor_line_polish.py tests/test_line_detector.py -v
```

Expected: all PASS.

- [ ] **Step 7: Commit**

```bash
git add src/stages/camera.py config/default.yaml tests/test_camera_stage.py
git commit -m "feat(camera): wire anchor-frame line polish into camera stage"
```

---

### Task 3.4: Phase 3 quality-report metric `worst_frame_pitch_gap_m`

**Files:**
- Modify: `src/stages/camera.py`
- Test: `tests/test_camera_stage.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_camera_stage.py`:

```python
@pytest.mark.unit
def test_camera_quality_includes_pitch_gap_metric(tmp_output_with_static_clip):
    """camera_quality.json must include worst_frame_pitch_gap_m."""
    import json
    from src.stages.camera import CameraStage

    stage = CameraStage(
        output_dir=tmp_output_with_static_clip,
        config={"camera": {"static_camera": True, "anchor_line_polish": False}},
    )
    stage.run()
    qr = json.loads(
        (tmp_output_with_static_clip / "camera" / "camera_quality.json").read_text()
    )
    assert "worst_frame_pitch_gap_m" in qr
    assert "unmeasured_frames" in qr
    assert "distortion" in qr
```

- [ ] **Step 2: Run to verify it fails**

```bash
pytest tests/test_camera_stage.py::test_camera_quality_includes_pitch_gap_metric -v
```

Expected: FAIL.

- [ ] **Step 3: Compute the metric in the camera stage**

Add a helper at the top of `src/stages/camera.py`:

```python
def _measure_frame_pitch_gap(
    frame_bgr, K, R, t, distortion, pitch_lines, image_size,
) -> float | None:
    """Worst projected-vs-detected line gap in pitch metres at this frame."""
    from src.utils.line_detector import detect_pitch_line_segments
    from src.utils.anchor_line_polish import (
        _match_detected_to_catalogue, _perp_distance_to_line,
    )
    from src.utils.camera_projection import project_world_to_image, undistort_pixel

    detected = detect_pitch_line_segments(frame_bgr, min_length_px=40.0)
    matched = _match_detected_to_catalogue(
        detected, pitch_lines, K, R, t, distortion, image_size,
    )
    if not matched:
        return None
    worst_px = 0.0
    worst_pixel: tuple[float, float] | None = None
    for det, a, b in matched:
        proj = project_world_to_image(
            K, R, t, distortion, np.array([a, b], dtype=np.float64),
        )
        for p in (det.p0, det.p1):
            d = _perp_distance_to_line(np.asarray(p), proj[0], proj[1])
            if d > worst_px:
                worst_px = d
                worst_pixel = p
    if worst_pixel is None:
        return None
    # Convert px → metres on pitch via inverse projection.
    pitch_m = _pixel_gap_to_pitch_m(worst_pixel, worst_px, K, R, t, distortion)
    return pitch_m


def _pixel_gap_to_pitch_m(pixel_uv, gap_px, K, R, t, distortion) -> float:
    """Approximate the pitch-plane distance corresponding to gap_px around
    pixel_uv. Assumes the pitch is the z=0 plane."""
    from src.utils.camera_projection import undistort_pixel
    u, v = pixel_uv
    uv0 = undistort_pixel((u, v), K, distortion)
    uv1 = undistort_pixel((u + gap_px, v), K, distortion)
    p0 = _ray_to_pitch(uv0, K, R, t)
    p1 = _ray_to_pitch(uv1, K, R, t)
    if p0 is None or p1 is None:
        return float("inf")
    return float(np.linalg.norm(p1 - p0))


def _ray_to_pitch(uv, K, R, t):
    """Back-project an undistorted pixel to the z=0 pitch plane, world frame."""
    Kinv = np.linalg.inv(K)
    pix = np.array([uv[0], uv[1], 1.0])
    ray_cam = Kinv @ pix
    ray_world = R.T @ ray_cam
    cam_centre = -R.T @ t
    if abs(ray_world[2]) < 1e-9:
        return None
    s = -cam_centre[2] / ray_world[2]
    if s <= 0:
        return None
    return cam_centre + s * ray_world
```

In the quality-report block (added in Task 1.5), after computing residuals:

```python
        # Worst-frame pitch gap (Phase 3 metric)
        from src.utils.pitch_lines_catalogue import LINE_CATALOGUE
        pitch_lines_catalogue = [
            (name, seg[0], seg[1]) for name, seg in LINE_CATALOGUE.items()
        ]
        gaps: list[float] = []
        unmeasured = 0
        if sol.camera_centre is not None:
            cap2 = cv2.VideoCapture(str(clip_path))
            try:
                for f in frames_out:
                    cap2.set(cv2.CAP_PROP_POS_FRAMES, f.frame)
                    ok, fr = cap2.read()
                    if not ok:
                        unmeasured += 1
                        continue
                    g = _measure_frame_pitch_gap(
                        fr,
                        np.asarray(f.K), np.asarray(f.R), np.asarray(f.t),
                        sol.distortion, pitch_lines_catalogue, (w, h),
                    )
                    if g is None:
                        unmeasured += 1
                    else:
                        gaps.append(g)
            finally:
                cap2.release()
        worst_pitch_gap = max(gaps) if gaps else None

        quality = {
            "body_drift_max_m": body_drift_max_m,
            "anchor_residual_mean_px": float(np.mean(residuals)) if residuals else None,
            "anchor_residual_max_px": float(np.max(residuals)) if residuals else None,
            "n_anchors": len(sol.per_anchor_KRt),
            "distortion": list(sol.distortion),
            "worst_frame_pitch_gap_m": worst_pitch_gap,
            "unmeasured_frames": unmeasured,
        }
```

- [ ] **Step 4: Run the test**

```bash
pytest tests/test_camera_stage.py::test_camera_quality_includes_pitch_gap_metric -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/stages/camera.py tests/test_camera_stage.py
git commit -m "feat(camera): worst_frame_pitch_gap_m metric in quality report"
```

---

### Task 3.5: End-to-end integration check

- [ ] **Step 1: Re-run the checkpointed clip**

```bash
python recon.py run --input <fixture-clip> --output /tmp/p3_check/ --from-stage camera
```

- [ ] **Step 2: Inspect the report**

```bash
cat /tmp/p3_check/camera/camera_quality.json
```

Expected: `worst_frame_pitch_gap_m < 0.3`.

If it's still >0.3 m: open the dashboard, scrub to a frame near the worst measured value, eyeball the overlay. Likely culprits in priority order:
1. False line matches (check by setting `anchor_line_polish: false` and re-running — does the worst-frame metric get worse?). If yes, polish is helping; tighten match gates.
2. Distortion overfit. Inspect `distortion` in the quality JSON. If `|k1| > 0.3`, the LM is compensating for something else; tighten the bounds in `anchor_solver.py` to ±0.3.
3. Too few or too poor anchors. Add anchors via the dashboard at the worst-frame timestamp.

- [ ] **Step 3: Visual playback**

```bash
python recon.py serve --output /tmp/p3_check/
```

Open `http://localhost:8000/viewer`, scrub the full clip; confirm worst-case projected line gap is sub-foot.

- [ ] **Step 4: Tag**

```bash
git tag phase-3-line-polish
```

---

## Self-review notes

- **Spec coverage**: Each spec section maps to a task —
  Phase 1 (1.1–1.5), Phase 2 (2.1–2.10), Phase 3 (3.1–3.5).
  Debug instrumentation (`--debug-camera`) is intentionally deferred to a
  follow-up task post-Phase 3 if visual debugging proves necessary.
- **Type consistency**: `JointSolution.camera_centre`, `CameraTrack.camera_centre`,
  `CameraTrack.distortion` use consistent names everywhere.
  `polish_anchor_against_lines` uses `camera_centre` (matches `JointSolution`).
- **No placeholders**: every code step shows the code; every test step shows
  the test; every command shows the exact pytest invocation and expected
  outcome.

---

## Notes for the executor

- This codebase uses `pytest -v -m unit` and writes tests with `@pytest.mark.unit`.
- `solve_anchors_jointly` and `refine_with_shared_translation` are tested
  via real synthetic projections — write similar fixtures rather than mocks.
- `cv2.ximgproc` lives in `opencv-contrib-python`. If `pip list | grep opencv`
  shows only `opencv-python`, install the contrib variant before Phase 3.
- The hybrid solver in `solve_anchors_jointly` does NOT run a single global
  LM today — it does solo + t-fixed per-anchor solves and stitches them. The
  `(k1, k2)` addition in Task 2.3 needs to thread distortion through every
  solo LM call (`_solve_one_anchor_full`, `_solve_anchor_with_t_fixed`,
  `_solve_anchor_with_C_fixed`). Read each function carefully before editing.
- After every commit, run `pytest -v -m unit` once and confirm green before
  starting the next task.
