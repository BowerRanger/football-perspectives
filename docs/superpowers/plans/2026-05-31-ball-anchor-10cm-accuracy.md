# Ball Anchor 10 cm Accuracy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the reconstructed ball land within 10 cm of its true pitch position in 3D — guaranteed at player-touch and ground-contact anchors, lateral everywhere — by making the user's clicked pixel authoritative and recovering airborne depth from gravity + bracketing hard knots.

**Architecture:** Five config-gated changes to the anchor-driven ball reconstruction in `src/stages/ball.py` (+ a soft residual in `bundle_adjust.py` and a quality-report diagnostic). C1 ray-constrains airborne `player_touch`; C2 lets gravity + ≥2 hard knots determine each flight arc (free `p0`); C4 snaps every airborne-anchored frame onto its clicked ray; C3a flags under-constrained spans; C3b is an optional coarse size prior. A real-clip harness (C5) is the 10 cm acceptance gate.

**Tech Stack:** Python 3.11, numpy, scipy.optimize.least_squares, OpenCV, pytest. Project venv at `.venv` (`source .venv/bin/activate`). All `pytest`/`python` commands assume the venv is active.

**Spec:** `docs/superpowers/specs/2026-05-31-ball-anchor-10cm-accuracy-design.md`

---

## File Map

| Path | Status | Responsibility |
|---|---|---|
| `tests/test_ball_anchor_accuracy.py` | NEW | C5: re-run anchor-driven reconstruction on real clips (no-op detector), assert per-state lateral ≤ 10 cm + reproj ≤ tol. Skips when output dirs absent. |
| `src/stages/ball.py` | MODIFY | C1 ray-constrained airborne `player_touch` in `_resolve_anchor_world`; C2 free-`p0`-when-≥2-knots in the Phase-2 fit; C4 ray-faithful snap pass; C3a per-span knot-count diagnostic. |
| `src/utils/bundle_adjust.py` | MODIFY | C3b optional `size_depth_frames` soft residual. |
| `src/pipeline/quality_report.py` | MODIFY | C3a: surface under-constrained flight spans in the ball section. |
| `config/default.yaml` | MODIFY | New `ball:` keys: `free_p0_min_hard_knots`, `ray_faithful_tolerance_px`, `min_hard_knots_warn`, `size_depth_prior`. |
| `tests/test_ball_ray_constrained_touch.py` | NEW | C1 unit tests. |
| `tests/test_bundle_adjust_free_p0_knots.py` | NEW | C2 unit tests (free `p0` + 2 knots recovers depth). |
| `tests/test_ball_ray_faithful.py` | NEW | C4 unit tests (snap onto ray helper). |
| `tests/test_ball_size_depth_prior.py` | NEW | C3b unit tests. |

**Reference (read before editing):**
- `src/stages/ball.py` — `_resolve_anchor_world` (~line 394), Phase-2 span fit (~line 1500-1836), final BallFrame assembly (~line 1907), `_apply_hard_knot_anchor_overrides` (~line 481).
- `src/utils/bundle_adjust.py` — `fit_parabola_to_image_observations` (line 25); the free-`p0` residual path already includes `knot_frames` + `z_range_frames`.
- `src/utils/foot_anchor.py` — `ankle_ray_to_pitch`; `src/utils/camera_projection.py` — `project_world_to_image`, `undistort_pixel`.
- Test fixtures to copy: `tests/test_ball_stage_anchors.py` (`_camera_pose`, `_write_blank_clip`, `_save_camera_track`, `_save_manifest`, `_minimal_cfg`).
- Harness source: `docs/superpowers/notes/ball-accuracy/rerun_and_measure.py`, `measure_anchor_error.py`.

---

## Task 1: C5 — promote the validation harness + acceptance gate

**Files:**
- Create: `tests/test_ball_anchor_accuracy.py`

- [ ] **Step 1: Write the harness test**

Create `tests/test_ball_anchor_accuracy.py`. It re-runs the anchor-driven reconstruction (no-op detector) on the real clips and measures per-state lateral error vs the clicked ray. It SKIPS when output dirs are absent (CI), and runs fully on the dev machine.

```python
"""Acceptance gate: ball within 10cm at anchors on real clips.

Re-runs the ball stage with a no-op detector (anchored frames reconstruct
from anchors + camera + refined_poses, independent of WASB) and measures the
perpendicular distance from each emitted world_xyz to the camera ray through
the user's clicked anchor pixel. Skips when the output dirs are not present.

Also runnable directly for a per-state table:
    python tests/test_ball_anchor_accuracy.py
"""
from __future__ import annotations

import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.schemas.ball_anchor import BallAnchorSet
from src.schemas.ball_track import BallTrack
from src.schemas.camera_track import CameraTrack
from src.stages.ball import BallStage
from src.utils.ball_detector import BallDetector
from src.utils.camera_projection import project_world_to_image, undistort_pixel

ROOT = Path(__file__).resolve().parents[1]
CLIPS = [
    ("gberch", "output", "gberch"),
    ("kroupi01", "output-kroupi", "kroupi01"),
    ("origi01", "output-origi", "origi01"),
]
# States the goal must guarantee within 10cm laterally (contact + ground).
CONTACT_GROUND = {
    "grounded", "kick", "bounce", "catch", "goal_impact", "player_touch",
}


class _NoopDetector(BallDetector):
    def detect(self, frame):  # noqa: ANN001
        return None


def _ray(uv, K, R, t, dist):
    uv = np.asarray(uv, float)
    if dist != (0.0, 0.0):
        uv = undistort_pixel(uv, K, dist)
    C = -R.T @ t
    d = R.T @ (np.linalg.inv(K) @ np.array([uv[0], uv[1], 1.0]))
    return C, d / np.linalg.norm(d)


def _lateral(P, C, d):
    v = P - C
    return float(np.linalg.norm(P - (C + float(np.dot(v, d)) * d)))


def _rerun_and_collect(clip_id, out_dir, shot_id):
    out = ROOT / out_dir
    cam_p = out / "camera" / f"{shot_id}_camera_track.json"
    clip = out / "shots" / f"{shot_id}.mp4"
    anc_p = out / "ball" / f"{shot_id}_ball_anchors.json"
    if not (cam_p.exists() and clip.exists() and anc_p.exists()):
        pytest.skip(f"{clip_id}: output dir not present")
    cfg = yaml.safe_load((ROOT / "config" / "default.yaml").read_text())
    tmp = Path(tempfile.mkdtemp(prefix="ball_acc_"))
    stage = BallStage(config=cfg, output_dir=out, ball_detector=_NoopDetector())
    stage.shot_filter = shot_id
    track_out = tmp / f"{shot_id}_ball_track.json"
    stage._run_shot(shot_id, clip, cam_p, track_out, cfg["ball"], _NoopDetector())

    anc = BallAnchorSet.load(anc_p)
    track = BallTrack.load(track_out)
    cam = CameraTrack.load(cam_p)
    dist = cam.distortion
    K = {f.frame: np.array(f.K) for f in cam.frames}
    R = {f.frame: np.array(f.R) for f in cam.frames}
    tfb = np.array(cam.t_world)
    T = {f.frame: (np.array(f.t) if f.t is not None else tfb) for f in cam.frames}
    f2w = {f.frame: f.world_xyz for f in track.frames}
    rows = []  # (state, lateral_m, reproj_px, frame)
    for a in anc.anchors:
        if a.image_xy is None or a.frame not in K:
            continue
        w = f2w.get(a.frame)
        if w is None:
            continue
        P = np.array(w, float)
        uvp = project_world_to_image(K[a.frame], R[a.frame], T[a.frame], dist,
                                     P.reshape(1, 3))[0]
        reproj = float(np.linalg.norm(uvp - np.array(a.image_xy)))
        C, d = _ray(a.image_xy, K[a.frame], R[a.frame], T[a.frame], dist)
        rows.append((a.state, _lateral(P, C, d), reproj, a.frame))
    return rows


@pytest.mark.integration
@pytest.mark.parametrize("clip_id,out_dir,shot_id", CLIPS)
def test_contact_ground_anchors_within_10cm(clip_id, out_dir, shot_id):
    rows = _rerun_and_collect(clip_id, out_dir, shot_id)
    bad = [(s, round(lat, 3), f) for (s, lat, _, f) in rows
           if s in CONTACT_GROUND and lat > 0.10]
    assert not bad, f"{clip_id}: contact/ground anchors >10cm lateral: {bad}"
```

- [ ] **Step 2: Run it — record the current baseline (expect FAILs)**

Run: `pytest tests/test_ball_anchor_accuracy.py -v`
Expected: `gberch` and `kroupi01` PASS (contact/ground already 0.00 m); `origi01` FAILS — its airborne `player_touch` anchors (frames 282/310/338/440) are up to 2.9 m off via the SMPL bone. This failure is the RED that Task 2 (C1) turns green.

- [ ] **Step 3: Add a `__main__` per-state table for ad-hoc measurement**

Append to `tests/test_ball_anchor_accuracy.py`:

```python
def _print_table():
    for clip_id, out_dir, shot_id in CLIPS:
        try:
            rows = _rerun_and_collect(clip_id, out_dir, shot_id)
        except Exception as exc:  # pytest.skip raises outside a test
            print(f"\n### {clip_id}: {exc}")
            continue
        by = defaultdict(list)
        for s, lat, rp, _ in rows:
            by[s].append((lat, rp))
        print(f"\n### {clip_id}")
        allv = []
        for s in sorted(by):
            lats = np.array([v[0] for v in by[s]])
            rps = np.array([v[1] for v in by[s]])
            allv.extend(lats.tolist())
            print(f"  {s:<14} n={len(lats):<3} lat med={np.median(lats):.3f} "
                  f"max={lats.max():.3f} reproj max={rps.max():6.1f} "
                  f">10cm:{int((lats>0.10).sum())}/{len(lats)}")
        allv = np.array(allv)
        if len(allv):
            print(f"  OVERALL med={np.median(allv):.3f} max={allv.max():.3f} "
                  f">10cm:{int((allv>0.10).sum())}/{len(allv)}")


if __name__ == "__main__":
    _print_table()
```

- [ ] **Step 4: Commit**

```bash
git add tests/test_ball_anchor_accuracy.py
git commit -m "test(ball): anchor-accuracy harness + 10cm acceptance gate (C5)"
```

---

## Task 2: C1 — ray-constrain airborne `player_touch`

**Files:**
- Modify: `src/stages/ball.py` (`_resolve_anchor_world`, and a new module-level helper)
- Create: `tests/test_ball_ray_constrained_touch.py`

- [ ] **Step 1: Write the failing unit test**

Create `tests/test_ball_ray_constrained_touch.py`:

```python
"""C1: airborne player_touch must lie on the clicked-pixel ray."""
from __future__ import annotations

import numpy as np

from src.stages.ball import _project_point_onto_pixel_ray
from src.utils.camera_projection import project_world_to_image


def _cam():
    K = np.array([[1500.0, 0, 640.0], [0, 1500.0, 360.0], [0, 0, 1.0]])
    R = np.eye(3)
    t = np.array([0.0, 0.0, 60.0])  # camera 60 m back along -z looking +z
    return K, R, t


def test_projects_off_ray_bone_onto_clicked_ray():
    K, R, t = _cam()
    # A bone 1.5 m off to the side of where the user clicked.
    clicked_uv = (640.0, 300.0)
    bone = np.array([1.5, 2.0, 5.0])  # arbitrary off-ray 3D point
    out = _project_point_onto_pixel_ray(bone, clicked_uv, K, R, t, (0.0, 0.0))
    # Result must reproject to the clicked pixel (lateral -> 0).
    uvp = project_world_to_image(K, R, t, (0.0, 0.0), out.reshape(1, 3))[0]
    assert np.linalg.norm(uvp - np.array(clicked_uv)) < 0.5


def test_preserves_along_ray_depth_of_point():
    K, R, t = _cam()
    clicked_uv = (700.0, 380.0)
    bone = np.array([0.7, 1.0, 8.0])
    out = _project_point_onto_pixel_ray(bone, clicked_uv, K, R, t, (0.0, 0.0))
    C = -R.T @ t
    d = R.T @ (np.linalg.inv(K) @ np.array([clicked_uv[0], clicked_uv[1], 1.0]))
    d = d / np.linalg.norm(d)
    # out is C + (bone-C).d * d : along-ray depth equals the bone's projection.
    expected_depth = float(np.dot(bone - C, d))
    assert np.dot(out - C, d) == np.float64(expected_depth) or \
        abs(float(np.dot(out - C, d)) - expected_depth) < 1e-6
```

- [ ] **Step 2: Run it — expect ImportError**

Run: `pytest tests/test_ball_ray_constrained_touch.py -v`
Expected: `ImportError: cannot import name '_project_point_onto_pixel_ray'`.

- [ ] **Step 3: Add the helper and use it in `_resolve_anchor_world`**

In `src/stages/ball.py`, add this module-level helper near `_demote_run_to_missing` (after line ~105):

```python
def _project_point_onto_pixel_ray(
    point: np.ndarray,
    uv: tuple[float, float],
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    distortion: tuple[float, float],
) -> np.ndarray:
    """Return the point on the camera ray through pixel ``uv`` that lies at
    ``point``'s along-ray depth. Keeps the user's clicked lateral position
    (the result reprojects to ``uv``) while taking depth from ``point``.
    """
    from src.utils.camera_projection import undistort_pixel

    uv_arr = np.asarray(uv, dtype=float)
    if distortion != (0.0, 0.0):
        uv_arr = undistort_pixel(uv_arr, K, distortion)
    C = -R.T @ t
    d_world = R.T @ (np.linalg.inv(K) @ np.array([uv_arr[0], uv_arr[1], 1.0]))
    d_hat = d_world / np.linalg.norm(d_world)
    depth = float(np.dot(np.asarray(point, dtype=float) - C, d_hat))
    return C + depth * d_hat
```

Then in `_resolve_anchor_world`, change the airborne `player_touch` branch (currently `return np.asarray(bone_world, dtype=float)`):

```python
    if anc.state == "player_touch" and fi not in ground_touch_frames:
        bone_world = bone_lookup.bone_world(anc)
        if bone_world is not None:
            # C1: keep the user's clicked lateral position; take only the
            # depth from the (HMR-drifting) bone. Projects the bone onto the
            # clicked-pixel ray so lateral error -> 0.
            return _project_point_onto_pixel_ray(
                np.asarray(bone_world, dtype=float), uv,
                K, R, t, distortion,
            )
        # Fall through to fallback ray-cast at z=1.0 below.
```

(`uv`, `K`, `R`, `t`, `distortion` are already in scope in `_resolve_anchor_world`.)

- [ ] **Step 4: Run unit + harness — expect pass**

Run: `pytest tests/test_ball_ray_constrained_touch.py tests/test_ball_anchor_accuracy.py -v`
Expected: unit tests PASS; `origi01` acceptance now PASSES for `player_touch` (airborne touches snapped onto the clicked ray → ≤ 10 cm lateral).

- [ ] **Step 5: Run the full ball suite — no regressions**

Run: `pytest tests/test_ball_stage.py tests/test_ball_stage_anchors.py tests/test_ball_anchor_endpoints.py -v`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add src/stages/ball.py tests/test_ball_ray_constrained_touch.py
git commit -m "feat(ball): ray-constrain airborne player_touch to clicked ray (C1)"
```

---

## Task 3: C2 (characterization) — free `p0` + 2 knots recovers depth

**Files:**
- Create: `tests/test_bundle_adjust_free_p0_knots.py`

The fitter already supports free `p0` with `knot_frames`. This task proves the physics property the C2 caller change relies on, before touching `ball.py`.

- [ ] **Step 1: Write the test**

Create `tests/test_bundle_adjust_free_p0_knots.py`:

```python
"""C2 physics: gravity + 2 hard knots determine the arc (depth included)."""
from __future__ import annotations

import numpy as np

from src.utils.bundle_adjust import fit_parabola_to_image_observations


def _cam():
    K = np.array([[1800.0, 0, 960.0], [0, 1800.0, 540.0], [0, 0, 1.0]])
    R = np.eye(3)
    t = np.array([0.0, 0.0, 0.0])
    return K, R, t


def _project(p, K, R, t):
    cam = R @ p + t
    return (float((K @ cam)[0] / cam[2]), float((K @ cam)[1] / cam[2]))


def test_free_p0_two_knots_recovers_depth():
    K, R, t = _cam()
    # True arc, camera 40 m back: shift world so it's in front of camera.
    t = np.array([0.0, 0.0, 40.0])
    p0 = np.array([-8.0, 2.0, 0.11])
    v0 = np.array([10.0, 1.0, 9.0])
    fps = 30.0
    g = np.array([0.0, 0.0, -9.81])
    n = 16
    obs, Ks, Rs, ts = [], [], [], []
    for i in range(n):
        dt = i / fps
        p = p0 + v0 * dt + 0.5 * g * dt ** 2
        obs.append((i, _project(p, K, R, t)))
        Ks.append(K); Rs.append(R); ts.append(t)
    p_end = p0 + v0 * ((n - 1) / fps) + 0.5 * g * ((n - 1) / fps) ** 2
    # Free p0 (p0_fixed=None) + start & end as hard knots.
    knots = {0: p0, n - 1: p_end}
    rp0, rv0, resid = fit_parabola_to_image_observations(
        obs, Ks=Ks, Rs=Rs, t_world=ts, fps=fps,
        p0_fixed=None, knot_frames=knots,
    )
    assert np.linalg.norm(rp0 - p0) < 0.10        # depth recovered
    assert np.linalg.norm(rv0 - v0) < 0.5
    assert resid < 2.0
```

- [ ] **Step 2: Run it — expect pass (characterization)**

Run: `pytest tests/test_bundle_adjust_free_p0_knots.py -v`
Expected: PASS. If it does not, the C2 caller change is unsafe — stop and revisit before Task 4.

- [ ] **Step 3: Commit**

```bash
git add tests/test_bundle_adjust_free_p0_knots.py
git commit -m "test(bundle_adjust): free p0 + 2 knots recovers depth (C2 characterization)"
```

---

## Task 4: C2 — Phase-2 fit frees `p0` when the span has ≥2 hard knots

**Files:**
- Modify: `config/default.yaml` (`ball.free_p0_min_hard_knots`)
- Modify: `src/stages/ball.py` (Phase-2 `p0_pin` selection, ~lines 1665-1707)

- [ ] **Step 1: Add config**

In `config/default.yaml`, under `ball:`, add:

```yaml
  free_p0_min_hard_knots: 2
```

Verify: `python -c "import yaml; print(yaml.safe_load(open('config/default.yaml'))['ball']['free_p0_min_hard_knots'])"` → `2`.

- [ ] **Step 2: Read the variable into the stage**

In `src/stages/ball.py` `_run_shot`, near where `plaus_cfg`/`pitch_dims` are built (~line 757), add:

```python
        free_p0_min_hard_knots = int(cfg.get("free_p0_min_hard_knots", 2))
```

- [ ] **Step 3: Change the Phase-2 `p0_pin` selection**

In the Phase-2 block, replace the existing p0-pin selection (the block starting `first_anc = span[0][1]` and ending before the `try: p2_p0, ... = fit_parabola_to_image_observations(`):

```python
                first_anc = span[0][1]
                if 0 in knots and first_anc.state in HARD_KNOT_STATES:
                    p0_pin = knots.pop(0)
                elif (
                    0 not in knots
                    and first_anc.state in AIRBORNE_STATES
                    and first_anc.image_xy is not None
                    and fa_span in per_frame_K
                ):
                    try:
                        p0_pin = ankle_ray_to_pitch(
                            first_anc.image_xy,
                            K=per_frame_K[fa_span], R=per_frame_R[fa_span], t=per_frame_t[fa_span],
                            plane_z=state_to_height(first_anc.state),
                            distortion=distortion,
                        )
                        p0_pin = np.asarray(p0_pin, dtype=float)
                        z_ranges.pop(0, None)
                    except (ValueError, Exception):
                        p0_pin = None
```

with:

```python
                first_anc = span[0][1]
                # C2: when >=2 hard knots bracket the span, gravity + the
                # knots fully determine the 6-DOF arc (depth included), so
                # leave p0 FREE and keep every knot. Pinning p0 here would
                # throw away the arc-curvature depth information. Below the
                # threshold, keep the historical safe pinning (p0 free-drifts
                # along the ray without >=2 knots).
                free_p0 = len(knots) >= free_p0_min_hard_knots
                if free_p0:
                    p0_pin = None  # keep all entries in `knots`
                elif 0 in knots and first_anc.state in HARD_KNOT_STATES:
                    p0_pin = knots.pop(0)
                elif (
                    0 not in knots
                    and first_anc.state in AIRBORNE_STATES
                    and first_anc.image_xy is not None
                    and fa_span in per_frame_K
                ):
                    try:
                        p0_pin = ankle_ray_to_pitch(
                            first_anc.image_xy,
                            K=per_frame_K[fa_span], R=per_frame_R[fa_span], t=per_frame_t[fa_span],
                            plane_z=state_to_height(first_anc.state),
                            distortion=distortion,
                        )
                        p0_pin = np.asarray(p0_pin, dtype=float)
                        z_ranges.pop(0, None)
                    except (ValueError, Exception):
                        p0_pin = None
```

Then in the `fit_parabola_to_image_observations(...)` call just below, demote the z-bucket weight when `p0` is free so the buckets can't fight the determined arc — change the call to add `z_range_weight`:

```python
                    p2_p0, p2_v0, p2_resid = fit_parabola_to_image_observations(
                        obs_p2, Ks=Ks_p2, Rs=Rs_p2, t_world=ts_p2,
                        fps=camera.fps, distortion=distortion,
                        p0_fixed=p0_pin, knot_frames=knots or None,
                        z_range_frames=z_ranges or None,
                        z_range_weight=(20.0 if free_p0 else 200.0),
                    )
```

- [ ] **Step 4: Run harness + ball suite**

Run: `pytest tests/test_ball_anchor_accuracy.py tests/test_ball_stage.py tests/test_ball_stage_anchors.py tests/test_ball_stage_layered.py -v`
Expected: contact/ground stay PASS; airborne lateral/reproj on the clips drop sharply for spans bracketed by ≥2 hard knots. Run `python tests/test_ball_anchor_accuracy.py` and confirm the airborne `>10cm` counts fall vs the Task-1 baseline.

- [ ] **Step 5: Commit**

```bash
git add config/default.yaml src/stages/ball.py
git commit -m "feat(ball): free p0 when >=2 hard knots so gravity recovers airborne depth (C2)"
```

---

## Task 5: C4 — ray-faithful guarantee for airborne-anchored frames

**Files:**
- Modify: `config/default.yaml` (`ball.ray_faithful_tolerance_px`)
- Modify: `src/stages/ball.py` (new pass after the final hard-knot override, ~line 1905)
- Create: `tests/test_ball_ray_faithful.py`

- [ ] **Step 1: Add config**

In `config/default.yaml` under `ball:` add:

```yaml
  ray_faithful_tolerance_px: 3.0
```

- [ ] **Step 2: Write the failing unit test**

Create `tests/test_ball_ray_faithful.py`:

```python
"""C4: snap an off-ray anchored frame onto the clicked ray, keep depth."""
from __future__ import annotations

import numpy as np

from src.stages.ball import _snap_world_onto_pixel_ray
from src.utils.camera_projection import project_world_to_image


def _cam():
    K = np.array([[1500.0, 0, 640.0], [0, 1500.0, 360.0], [0, 0, 1.0]])
    R = np.eye(3)
    t = np.array([0.0, 0.0, 50.0])
    return K, R, t


def test_off_ray_point_snaps_onto_ray():
    K, R, t = _cam()
    clicked = (700.0, 300.0)
    world = np.array([2.0, -1.0, 6.0])  # off the clicked ray
    snapped = _snap_world_onto_pixel_ray(world, clicked, K, R, t, (0.0, 0.0))
    uvp = project_world_to_image(K, R, t, (0.0, 0.0), snapped.reshape(1, 3))[0]
    assert np.linalg.norm(uvp - np.array(clicked)) < 0.5


def test_on_ray_point_unchanged():
    K, R, t = _cam()
    clicked = (640.0, 360.0)
    C = -R.T @ t
    d = R.T @ (np.linalg.inv(K) @ np.array([clicked[0], clicked[1], 1.0]))
    d = d / np.linalg.norm(d)
    world = C + 7.0 * d  # already on the ray
    snapped = _snap_world_onto_pixel_ray(world, clicked, K, R, t, (0.0, 0.0))
    assert np.linalg.norm(snapped - world) < 1e-6
```

- [ ] **Step 3: Run it — expect ImportError**

Run: `pytest tests/test_ball_ray_faithful.py -v`
Expected: `ImportError: cannot import name '_snap_world_onto_pixel_ray'`.

- [ ] **Step 4: Implement the helper + the pass**

In `src/stages/ball.py`, add the helper next to `_project_point_onto_pixel_ray` (it is the same projection — define `_snap_world_onto_pixel_ray` as an alias for clarity at call sites, or reuse directly). Add:

```python
def _snap_world_onto_pixel_ray(
    world: np.ndarray,
    uv: tuple[float, float],
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    distortion: tuple[float, float],
) -> np.ndarray:
    """Move ``world`` onto the camera ray through ``uv``, preserving its
    along-ray depth (lateral error -> 0). Same math as
    :func:`_project_point_onto_pixel_ray`; named separately for the C4 pass.
    """
    return _project_point_onto_pixel_ray(world, uv, K, R, t, distortion)
```

Then, in `_run_shot`, immediately after the FINAL `_apply_hard_knot_anchor_overrides(...)` call (~line 1905, just before `per_frame_out: list[BallFrame] = []`), insert the C4 pass:

```python
        # C4 — ray-faithfulness guarantee. The user's clicked pixel is hard
        # lateral ground truth. For every AIRBORNE-soft anchored frame
        # (airborne_low/mid/high) whose emitted world reprojects farther than
        # the tolerance from the click, snap it onto the clicked ray, keeping
        # the fitted along-ray depth. Hard-knot states are already pinned
        # on-ray, so they are skipped here.
        ray_faithful_tol_px = float(cfg.get("ray_faithful_tolerance_px", 3.0))
        _RAY_SNAP_STATES = frozenset({
            "airborne_low", "airborne_mid", "airborne_high",
        })
        for fi, anc in anchor_by_frame.items():
            if anc.state not in _RAY_SNAP_STATES or anc.image_xy is None:
                continue
            if fi not in per_frame_world or fi not in per_frame_K:
                continue
            world, conf = per_frame_world[fi]
            uvp = project_world_to_image(
                per_frame_K[fi], per_frame_R[fi], per_frame_t[fi],
                distortion, np.asarray(world, dtype=float).reshape(1, 3),
            )[0]
            if float(np.linalg.norm(uvp - np.array(anc.image_xy))) <= ray_faithful_tol_px:
                continue
            snapped = _snap_world_onto_pixel_ray(
                np.asarray(world, dtype=float),
                (float(anc.image_xy[0]), float(anc.image_xy[1])),
                per_frame_K[fi], per_frame_R[fi], per_frame_t[fi], distortion,
            )
            per_frame_world[fi] = (snapped, conf)
```

Add the import at the top of `src/stages/ball.py` (with the other `src.utils` imports):

```python
from src.utils.camera_projection import project_world_to_image
```

- [ ] **Step 5: Run unit + harness + ball suite**

Run: `pytest tests/test_ball_ray_faithful.py tests/test_ball_anchor_accuracy.py tests/test_ball_stage.py -v`
Expected: unit PASS; `python tests/test_ball_anchor_accuracy.py` shows **no airborne anchor > 10 cm lateral** on any clip (every anchored frame now sits on its clicked ray within tolerance). Contact/ground unchanged.

- [ ] **Step 6: Commit**

```bash
git add config/default.yaml src/stages/ball.py tests/test_ball_ray_faithful.py
git commit -m "feat(ball): ray-faithful snap guarantees anchored frames on clicked ray (C4)"
```

---

## Task 6: C3a — under-constrained-span diagnostic

**Files:**
- Modify: `config/default.yaml` (`ball.min_hard_knots_warn`)
- Modify: `src/stages/ball.py` (collect per-span hard-knot counts; log + stash for the report)
- Modify: `src/pipeline/quality_report.py` (surface the spans)
- Create: `tests/test_ball_underconstrained_diag.py`

- [ ] **Step 1: Add config**

In `config/default.yaml` under `ball:` add:

```yaml
  min_hard_knots_warn: 2
```

- [ ] **Step 2: Write the failing test**

Create `tests/test_ball_underconstrained_diag.py`:

```python
"""C3a: flight spans with <2 hard knots are flagged."""
from __future__ import annotations

from src.stages.ball import _underconstrained_spans


def test_flags_zero_and_one_knot_spans_only():
    # spans: list of (fa, fb, n_hard_knots)
    spans = [(10, 25, 2), (40, 55, 1), (70, 85, 0)]
    flagged = _underconstrained_spans(spans, min_hard_knots=2)
    assert (40, 55, 1) in flagged
    assert (70, 85, 0) in flagged
    assert (10, 25, 2) not in flagged
    assert len(flagged) == 2
```

- [ ] **Step 3: Run it — expect ImportError**

Run: `pytest tests/test_ball_underconstrained_diag.py -v`
Expected: `ImportError: cannot import name '_underconstrained_spans'`.

- [ ] **Step 4: Implement the helper + wire it into the stage**

In `src/stages/ball.py`, add the module-level helper near `_demote_run_to_missing`:

```python
def _underconstrained_spans(
    spans: list[tuple[int, int, int]],
    min_hard_knots: int,
) -> list[tuple[int, int, int]]:
    """Return flight spans (fa, fb, n_hard_knots) with fewer than
    ``min_hard_knots`` hard 3D knots — monocularly depth-under-determined.
    """
    return [s for s in spans if s[2] < min_hard_knots]
```

In `_run_shot`, accumulate per-span knot counts during the Phase-2 loop. Where `knots` is built for each span (just before the `p0_pin` selection added in Task 4), record the count. Add near the start of `_run_shot` (with the other locals):

```python
        span_knot_counts: list[tuple[int, int, int]] = []
        min_hard_knots_warn = int(cfg.get("min_hard_knots_warn", 2))
```

Then inside the Phase-2 `for span in spans:` loop, right after `knots`/`z_ranges` are fully built and before the fit, add:

```python
                span_knot_counts.append((fa_span, fb_span, len(knots)))
```

After the Phase-2 loop completes, log + persist:

```python
        underconstrained = _underconstrained_spans(
            span_knot_counts, min_hard_knots_warn,
        )
        for fa_uc, fb_uc, nk in underconstrained:
            logger.warning(
                "ball: flight span %d-%d has %d<%d hard knots — depth is "
                "monocularly under-determined; add a kick/bounce/goal_impact/"
                "grounded anchor to bracket it",
                fa_uc, fb_uc, nk, min_hard_knots_warn,
            )
        self._underconstrained_spans = getattr(
            self, "_underconstrained_spans", {}
        )
        self._underconstrained_spans[shot_id or "(legacy)"] = [
            {"start": fa_uc, "end": fb_uc, "hard_knots": nk}
            for fa_uc, fb_uc, nk in underconstrained
        ]
```

- [ ] **Step 5: Surface it in the quality report**

In `src/pipeline/quality_report.py`, inside the `if ball_path.exists():` block where `report["ball"]` is built (~line 154), add a per-clip diagnostic. Since the report reads the saved track, recompute the under-constrained spans from the saved `flight_segments` (each segment's `frame_range` and how many of its frames coincide with hard-knot-resolved anchors is not in the track) — instead, read the count the stage already stored. Add a sidecar write in the stage and read it here:

In `src/stages/ball.py`, after `track.save(ball_out_path)` in `_run_shot`, write the diagnostic sidecar:

```python
        diag_path = ball_out_path.with_name(
            ball_out_path.stem.replace("_ball_track", "_ball_diag") + ".json"
        )
        import json as _json
        diag_path.write_text(_json.dumps({
            "underconstrained_spans": self._underconstrained_spans.get(
                shot_id or "(legacy)", []
            ),
        }, indent=2))
```

In `src/pipeline/quality_report.py`, within the ball section, load and include it:

```python
        diag_path = ball_path.with_name(
            ball_path.stem.replace("_ball_track", "_ball_diag") + ".json"
        )
        if diag_path.exists():
            import json as _json
            try:
                report["ball"]["underconstrained_spans"] = _json.loads(
                    diag_path.read_text()
                ).get("underconstrained_spans", [])
            except Exception:
                report["ball"]["underconstrained_spans"] = []
```

- [ ] **Step 6: Run tests**

Run: `pytest tests/test_ball_underconstrained_diag.py tests/test_ball_stage.py tests/test_ball_stage_anchors.py -v`
Expected: all PASS.

- [ ] **Step 7: Commit**

```bash
git add config/default.yaml src/stages/ball.py src/pipeline/quality_report.py tests/test_ball_underconstrained_diag.py
git commit -m "feat(ball): flag under-constrained flight spans in quality report (C3a)"
```

---

## Task 7: Validate on real clips + regenerate stale tracks

**Files:**
- No code change. Validation + regeneration.

- [ ] **Step 1: Print the final per-state table**

Run: `python tests/test_ball_anchor_accuracy.py`
Expected: every clip shows contact/ground anchors at 0.00 m and **0** airborne anchors > 10 cm lateral. Record the table in the PR description.

- [ ] **Step 2: Run the acceptance gate**

Run: `pytest tests/test_ball_anchor_accuracy.py -v`
Expected: all three clips PASS `test_contact_ground_anchors_within_10cm`.

- [ ] **Step 3: Regenerate the stale origi01 track with the full pipeline**

Run (WASB detector; refreshes unanchored frames too):
`python recon.py run --input output-origi/shots/origi01.mp4 --output ./output-origi/ --from-stage ball`
Expected: `output-origi/ball/origi01_ball_track.json` regenerated; `quality_report.json` shows any remaining under-constrained spans (C3a).

- [ ] **Step 4: Commit regenerated artifacts (if tracked) and note remaining gaps**

```bash
git add output-origi/ball/origi01_ball_track.json output-origi/quality_report.json 2>/dev/null || true
git commit -m "chore(ball): regenerate origi01 track with ray-faithful reconstruction" || true
```

If under-constrained spans remain, the C3a diagnostic names them — adding one bracketing anchor per span and re-running closes the depth gap (C2 then resolves it exactly).

---

## Task 8 (CONDITIONAL): C3b — coarse size-depth prior

**Build only if Task 7 still shows <2-knot airborne spans whose depth matters and the user wants smoothing there.** Off by default; not a 10 cm mechanism (a ~5 px ball at 90 m gives ±10–20 m depth).

**Files:**
- Modify: `config/default.yaml` (`ball.size_depth_prior`)
- Modify: `src/utils/bundle_adjust.py` (`size_depth_frames` residual)
- Create: `tests/test_ball_size_depth_prior.py`

- [ ] **Step 1: Add config**

```yaml
  size_depth_prior:
    enabled: false
    weight: 20.0
    min_pixel_diameter: 3.0
    max_pixel_diameter: 40.0
```

- [ ] **Step 2: Write the failing test**

Create `tests/test_ball_size_depth_prior.py`:

```python
"""C3b: size-depth soft residual pulls along-axis depth toward D_est."""
from __future__ import annotations

import numpy as np

from src.utils.bundle_adjust import fit_parabola_to_image_observations


def _cam():
    K = np.array([[1800.0, 0, 960.0], [0, 1800.0, 540.0], [0, 0, 1.0]])
    R = np.eye(3)
    t = np.array([0.0, 0.0, 30.0])
    return K, R, t


def _project(p, K, R, t):
    cam = R @ p + t
    return (float((K @ cam)[0] / cam[2]), float((K @ cam)[1] / cam[2]))


def test_size_depth_prior_pulls_depth():
    K, R, t = _cam()
    p0 = np.array([-5.0, 1.0, 1.0]); v0 = np.array([8.0, 0.0, 6.0])
    fps = 30.0; g = np.array([0.0, 0.0, -9.81]); n = 10
    obs, Ks, Rs, ts = [], [], [], []
    for i in range(n):
        dt = i / fps
        p = p0 + v0 * dt + 0.5 * g * dt ** 2
        obs.append((i, _project(p, K, R, t)))
        Ks.append(K); Rs.append(R); ts.append(t)
    # Camera-frame depth of the true p0 (cam z).
    cam_z0 = float((R @ p0 + t)[2])
    size_depth = {0: (R, t, cam_z0)}
    rp0, rv0, resid = fit_parabola_to_image_observations(
        obs, Ks=Ks, Rs=Rs, t_world=ts, fps=fps,
        p0_fixed=None, size_depth_frames=size_depth, size_depth_weight=50.0,
    )
    assert abs(float((R @ rp0 + t)[2]) - cam_z0) < 0.5


def test_disabled_when_no_size_frames():
    K, R, t = _cam()
    p0 = np.array([-5.0, 1.0, 1.0]); v0 = np.array([8.0, 0.0, 6.0])
    fps = 30.0; g = np.array([0.0, 0.0, -9.81]); n = 10
    obs, Ks, Rs, ts = [], [], [], []
    for i in range(n):
        dt = i / fps
        p = p0 + v0 * dt + 0.5 * g * dt ** 2
        obs.append((i, _project(p, K, R, t)))
        Ks.append(K); Rs.append(R); ts.append(t)
    rp0, _, _ = fit_parabola_to_image_observations(
        obs, Ks=Ks, Rs=Rs, t_world=ts, fps=fps, p0_fixed=None,
        size_depth_frames=None,
    )
    assert np.linalg.norm(rp0 - p0) < 0.5
```

- [ ] **Step 3: Run it — expect TypeError (unknown kwarg)**

Run: `pytest tests/test_ball_size_depth_prior.py -v`
Expected: FAIL — `fit_parabola_to_image_observations` got an unexpected keyword `size_depth_frames`.

- [ ] **Step 4: Add the residual to the fitter**

In `src/utils/bundle_adjust.py`, add to the signature of `fit_parabola_to_image_observations` (after `z_range_weight`):

```python
    size_depth_frames: dict[int, tuple[np.ndarray, np.ndarray, float]] | None = None,
    size_depth_weight: float = 20.0,
```

In BOTH `_residuals` and `_residuals_v0only`, after the `z_range_frames` block, add (use the local `p0`/`v0` of each function — `p0_pin` in the v0-only variant):

```python
        if size_depth_frames:
            for rel_idx, (R_k, t_k, depth_est) in size_depth_frames.items():
                dt_k = rel_idx / fps
                pos_k = p0 + v0 * dt_k + 0.5 * (dt_k ** 2) * g_vec
                cam_z = float((np.asarray(R_k) @ pos_k + np.asarray(t_k))[2])
                residuals.append(np.array([size_depth_weight * (cam_z - float(depth_est))]))
```

(In `_residuals_v0only`, replace `p0` with `p0_pin` in the `pos_k` line.)

Document the two new params in the docstring (mirror the `z_range_frames` style).

- [ ] **Step 5: Run it — expect pass**

Run: `pytest tests/test_ball_size_depth_prior.py tests/test_ball_flight.py tests/test_ball_spin_fit.py -v`
Expected: all PASS (new tests pass; existing unchanged — defaults make the new param a no-op).

- [ ] **Step 6 (only if wiring into the stage is wanted): measure d_px at anchor pixels**

Add an anchor-frame ball-radius measurement (radial intensity profile around the clicked pixel) and feed `size_depth_frames` into the Phase-2 fit for `<2`-knot spans, gated by `ball.size_depth_prior.enabled`. Keep it behind the disabled-by-default flag; only enable after the harness shows it helps. (This step is deliberately scoped out of the default rollout — the diagnostic C3a + adding a bracketing anchor is the recommended path to 10 cm.)

- [ ] **Step 7: Commit**

```bash
git add config/default.yaml src/utils/bundle_adjust.py tests/test_ball_size_depth_prior.py
git commit -m "feat(bundle_adjust): optional size-depth soft residual (C3b, off by default)"
```

---

## Self-review checklist (run before execution)

- [ ] Every contact/ground state still pins to 0.00 m (Task 1 gate + Task 2/4 keep it).
- [ ] `_project_point_onto_pixel_ray` (Task 2) and `_snap_world_onto_pixel_ray` (Task 5) share one implementation; both reproject to the clicked pixel.
- [ ] C2 frees `p0` only when `len(knots) >= free_p0_min_hard_knots`; otherwise historical pinning is intact (no regression on 1/0-knot spans).
- [ ] C4 only touches `airborne_low/mid/high` anchors; hard-knot states are skipped.
- [ ] No placeholders: every code step shows real code; every run step shows the exact command + expected outcome.
