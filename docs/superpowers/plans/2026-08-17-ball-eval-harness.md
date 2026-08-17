# Ball Sub-20 cm Eval Harness (W1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the measurement harness that grades the ball stage's dense track
against every available ground-truth source (A1–A5 of the spec), plus the
hold-out runner, and record the committed baseline for all four clips.

**Architecture:** A pure numpy metrics module (`src/utils/ball_eval.py`) with
frozen-dataclass rows, plus a CLI (`scripts/eval_ball_accuracy.py`) that
re-runs the ball stage in an overlay temp dir (symlinked stage inputs, filtered
anchors for hold-out) and emits JSON/markdown reports. No changes to the stage.

**Tech Stack:** Python 3.11 (`.venv311`), numpy, pytest (`unit`/`integration`
markers), existing schemas (`BallAnchorSet`, `BallTrack`, `CameraTrack`,
`BallFixSet`), `BallStage._run_shot` compat shim, `NoopDetector` pattern from
`docs/superpowers/notes/ball-accuracy/rerun_and_measure.py`.

**Spec:** `docs/superpowers/specs/2026-08-17-ball-sub20cm-accuracy-design.md`

## Global Constraints

- Always run Python/pytest via `.venv311/bin/python` (torch-pinned venv).
- Tests marked `@pytest.mark.unit` run with no clip data; `@pytest.mark.integration`
  may read the real `output*/` dirs and must `pytest.skip` when absent.
- Never write into the real `output*/` dirs; harness writes only to temp
  overlays and `docs/superpowers/notes/ball-accuracy/`.
- Immutable data: frozen dataclasses, no mutation of loaded schema objects.
- Files ≤ 800 lines; functions ≤ 50 lines where practical.
- Known-failing tests on main (do not fix here, do not regress further):
  `test_ball_stage.py::test_aerial_arc_promotes_grounded_run_to_flight`,
  `test_blender_export_smpl_skeleton.py::test_player_fbx_has_24_bones_and_full_keyframes`.
- Commit only files created/edited by this campaign — the working tree has
  unrelated user edits (CLAUDE.md, README.md, .vscode/, prototypes/…); never
  `git add -A`.

---

### Task 1: Ray/GT primitives in `src/utils/ball_eval.py`

**Files:**
- Create: `src/utils/ball_eval.py`
- Test: `tests/test_ball_eval.py`

**Interfaces:**
- Consumes: `src.utils.camera_projection.undistort_pixel`,
  `project_world_to_image`; `src.utils.ball_anchor_heights.GROUND_LEVEL_STATES`,
  `AIRBORNE_BUCKETS`; `src.schemas.ball_anchor.BallAnchor`.
- Produces (later tasks rely on these exact names):
  - `pixel_ray(uv, K, R, t, distortion=(0.0, 0.0)) -> tuple[np.ndarray, np.ndarray]` — `(C, d_hat)`
  - `point_ray_distance(P, C, d_hat) -> tuple[float, float]` — `(perp_m, along_m)`
  - `ray_plane_z(C, d_hat, z) -> np.ndarray | None` — ray ∩ horizontal plane
  - `anchor_gt_world(anchor, K, R, t, distortion, *, ball_radius, joint_world=None) -> tuple[np.ndarray | None, str]`
    with kind ∈ `{"ground_exact", "joint_depth", "ray_only", "none"}`
    (`bounce` counts ground_exact; `goal_impact` returns ray_only in v1 —
    geometry wiring is a follow-up noted in the CLI task).

- [ ] **Step 1: Write the failing tests** — synthetic pinhole camera looking
  down the +y axis from (0, −20, 10); known world points project/round-trip.

```python
import numpy as np
import pytest

from src.schemas.ball_anchor import BallAnchor

pytestmark = pytest.mark.unit


def _cam():
    """Simple pinhole: camera at (0,-20,10), looking at pitch origin."""
    K = np.array([[1500.0, 0, 960], [0, 1500.0, 540], [0, 0, 1.0]])
    fwd = np.array([0.0, 20.0, -10.0]); fwd /= np.linalg.norm(fwd)
    up = np.array([0.0, 0.0, 1.0])
    right = np.cross(fwd, up); right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    R = np.stack([right, down, fwd])          # world→cam rows
    C = np.array([0.0, -20.0, 10.0])
    t = -R @ C
    return K, R, t


def test_pixel_ray_roundtrip():
    from src.utils.ball_eval import pixel_ray, point_ray_distance
    from src.utils.camera_projection import project_world_to_image
    K, R, t = _cam()
    P = np.array([2.0, 5.0, 0.11])
    uv = project_world_to_image(K, R, t, (0.0, 0.0), P.reshape(1, 3))[0]
    C, d = pixel_ray(uv, K, R, t)
    perp, along = point_ray_distance(P, C, d)
    assert perp < 1e-6
    assert along > 0


def test_ray_plane_z_recovers_ground_point():
    from src.utils.ball_eval import pixel_ray, ray_plane_z
    from src.utils.camera_projection import project_world_to_image
    K, R, t = _cam()
    P = np.array([-3.0, 12.0, 0.11])
    uv = project_world_to_image(K, R, t, (0.0, 0.0), P.reshape(1, 3))[0]
    C, d = pixel_ray(uv, K, R, t)
    X = ray_plane_z(C, d, 0.11)
    assert np.allclose(X, P, atol=1e-6)


def test_ray_plane_z_none_when_parallel():
    from src.utils.ball_eval import ray_plane_z
    X = ray_plane_z(np.array([0.0, 0.0, 5.0]), np.array([0.0, 1.0, 0.0]), 0.11)
    assert X is None


def test_anchor_gt_world_ground_exact():
    from src.utils.ball_eval import anchor_gt_world
    from src.utils.camera_projection import project_world_to_image
    K, R, t = _cam()
    P = np.array([4.0, 8.0, 0.11])
    uv = project_world_to_image(K, R, t, (0.0, 0.0), P.reshape(1, 3))[0]
    anc = BallAnchor(frame=5, state="grounded", image_xy=(float(uv[0]), float(uv[1])))
    gt, kind = anchor_gt_world(anc, K, R, t, (0.0, 0.0), ball_radius=0.11)
    assert kind == "ground_exact"
    assert np.allclose(gt, P, atol=1e-6)


def test_anchor_gt_world_joint_depth_projects_joint_onto_ray():
    from src.utils.ball_eval import anchor_gt_world, pixel_ray, point_ray_distance
    from src.utils.camera_projection import project_world_to_image
    K, R, t = _cam()
    true_ball = np.array([1.0, 6.0, 0.3])
    uv = project_world_to_image(K, R, t, (0.0, 0.0), true_ball.reshape(1, 3))[0]
    joint = true_ball + np.array([0.05, 0.4, 0.05])   # FK depth drift off-ray
    anc = BallAnchor(frame=5, state="player_touch", image_xy=(float(uv[0]), float(uv[1])),
                     player_id="P001", bone="r_foot")
    gt, kind = anchor_gt_world(anc, K, R, t, (0.0, 0.0), ball_radius=0.11,
                               joint_world=tuple(joint))
    assert kind == "joint_depth"
    C, d = pixel_ray(uv, K, R, t)
    perp, _ = point_ray_distance(np.asarray(gt), C, d)
    assert perp < 1e-9                      # GT lies on the clicked ray
    assert np.linalg.norm(gt - true_ball) < 0.45   # depth from joint ≈ ball depth


def test_anchor_gt_world_airborne_is_ray_only_and_no_pixel_is_none():
    from src.utils.ball_eval import anchor_gt_world
    K, R, t = _cam()
    anc = BallAnchor(frame=5, state="airborne_low", image_xy=(900.0, 400.0))
    gt, kind = anchor_gt_world(anc, K, R, t, (0.0, 0.0), ball_radius=0.11)
    assert (gt, kind) == (None, "ray_only")
    anc2 = BallAnchor(frame=6, state="off_screen_flight", image_xy=None)
    assert anchor_gt_world(anc2, K, R, t, (0.0, 0.0), ball_radius=0.11) == (None, "none")
```

- [ ] **Step 2: Run to verify failure** —
  `.venv311/bin/python -m pytest tests/test_ball_eval.py -q` →
  `ModuleNotFoundError`/`ImportError` for `src.utils.ball_eval`.
  (If `BallAnchor` kwargs differ — check `src/schemas/ball_anchor.py` first and
  adjust the test constructors to the real field names before writing the module.)

- [ ] **Step 3: Implement the primitives**

```python
"""Metric primitives for grading the ball track against ground truth.

Pure numpy — no torch, no video. See the sub-20cm accuracy spec §4.
"""

from __future__ import annotations

import numpy as np

from src.schemas.ball_anchor import BallAnchor
from src.utils.ball_anchor_heights import GROUND_LEVEL_STATES
from src.utils.camera_projection import undistort_pixel


def pixel_ray(uv, K, R, t, distortion=(0.0, 0.0)):
    """Camera centre + unit world-space ray direction through pixel ``uv``."""
    uv = np.asarray(uv, dtype=float)
    if tuple(distortion) != (0.0, 0.0):
        uv = undistort_pixel(uv, K, distortion)
    R = np.asarray(R, dtype=float)
    C = -R.T @ np.asarray(t, dtype=float)
    d = R.T @ (np.linalg.inv(K) @ np.array([uv[0], uv[1], 1.0]))
    return C, d / np.linalg.norm(d)


def point_ray_distance(P, C, d_hat):
    """(perpendicular distance, along-ray depth) of point ``P`` from a ray."""
    v = np.asarray(P, dtype=float) - C
    along = float(np.dot(v, d_hat))
    return float(np.linalg.norm(v - along * d_hat)), along


def ray_plane_z(C, d_hat, z):
    """Intersect the ray with the horizontal plane ``Z=z`` (forward only)."""
    dz = float(d_hat[2])
    if abs(dz) < 1e-9:
        return None
    s = (float(z) - float(C[2])) / dz
    if s <= 0:
        return None
    return np.asarray(C, dtype=float) + s * np.asarray(d_hat, dtype=float)


# player_touch GT: clicked ray fixes lateral; the contacting joint fixes depth.
_GROUND_EXACT_STATES = frozenset(GROUND_LEVEL_STATES) | {"bounce"}


def anchor_gt_world(anchor: BallAnchor, K, R, t, distortion, *,
                    ball_radius: float, joint_world=None):
    """Best-available GT world position for an anchor, with its kind."""
    if anchor.image_xy is None:
        return None, "none"
    C, d = pixel_ray(anchor.image_xy, K, R, t, distortion)
    if anchor.state in _GROUND_EXACT_STATES:
        X = ray_plane_z(C, d, ball_radius)
        return (X, "ground_exact") if X is not None else (None, "ray_only")
    if anchor.state == "player_touch" and joint_world is not None:
        _, along = point_ray_distance(np.asarray(joint_world, float), C, d)
        if along > 0:
            return C + along * d, "joint_depth"
    return None, "ray_only"
```

- [ ] **Step 4: Run to verify pass** —
  `.venv311/bin/python -m pytest tests/test_ball_eval.py -q` → all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/utils/ball_eval.py tests/test_ball_eval.py
git commit -m "feat: ball eval ray/GT primitives (sub-20cm campaign W1)"
```

---

### Task 2: Error rows — anchors, fixes, dense lateral

**Files:**
- Modify: `src/utils/ball_eval.py`
- Test: `tests/test_ball_eval.py` (append)

**Interfaces:**
- Produces:
  - `AnchorEvalRow(frame, state, kind, held_out, lateral_m, err_3d_m, reproj_px, depth_m)` frozen dataclass; `err_3d_m` is `None` for `ray_only`.
  - `FixEvalRow(frame, err_3d_m, ray_miss_m)` frozen.
  - `DenseEvalRow(frame, lateral_m, confidence, source)` frozen.
  - `eval_rows_at_anchors(world_by_frame, anchors, cams, *, ball_radius, distortion, joint_world_fn=None, held_out_frames=frozenset()) -> tuple[AnchorEvalRow, ...]`
    where `cams: dict[int, tuple[K, R, t]]`, `world_by_frame: dict[int, tuple[float,float,float]]`,
    `joint_world_fn: Callable[[int frame, str player_id, str bone], tuple | None] | None`.
  - `eval_rows_at_fixes(world_by_frame, fixes) -> tuple[FixEvalRow, ...]` where
    `fixes: Sequence[tuple[int, tuple[float,float,float], float]]` (frame, xyz, ray_miss_m).
  - `dense_lateral_rows(world_by_frame, observations, cams, *, distortion, min_confidence) -> tuple[DenseEvalRow, ...]`
    where `observations: Sequence[tuple[int, tuple[float,float], float, str]]` (frame, uv, conf, source).

- [ ] **Step 1: Write the failing tests** (append to `tests/test_ball_eval.py`)

```python
def test_eval_rows_at_anchors_grades_holdout_and_kinds():
    from src.utils.ball_eval import eval_rows_at_anchors
    from src.utils.camera_projection import project_world_to_image
    K, R, t = _cam()
    P_true = np.array([4.0, 8.0, 0.11])
    uv = project_world_to_image(K, R, t, (0.0, 0.0), P_true.reshape(1, 3))[0]
    anchors = [BallAnchor(frame=3, state="grounded",
                          image_xy=(float(uv[0]), float(uv[1])))]
    world = {3: tuple(P_true + np.array([0.05, 0.0, 0.0]))}   # 5cm off
    rows = eval_rows_at_anchors(world, anchors, {3: (K, R, t)},
                                ball_radius=0.11, distortion=(0.0, 0.0),
                                held_out_frames=frozenset({3}))
    (row,) = rows
    assert row.held_out and row.kind == "ground_exact"
    assert 0.03 < row.err_3d_m < 0.07
    assert row.lateral_m <= row.err_3d_m + 1e-9
    assert row.reproj_px > 0


def test_eval_rows_at_anchors_missing_track_frame_gives_none_errors():
    from src.utils.ball_eval import eval_rows_at_anchors
    K, R, t = _cam()
    anchors = [BallAnchor(frame=9, state="grounded", image_xy=(900.0, 700.0))]
    (row,) = eval_rows_at_anchors({}, anchors, {9: (K, R, t)},
                                  ball_radius=0.11, distortion=(0.0, 0.0))
    assert row.err_3d_m is None and row.lateral_m is None


def test_eval_rows_at_fixes():
    from src.utils.ball_eval import eval_rows_at_fixes
    world = {10: (1.0, 2.0, 3.0)}
    (row,) = eval_rows_at_fixes(world, [(10, (1.0, 2.0, 3.5), 0.2)])
    assert abs(row.err_3d_m - 0.5) < 1e-9 and row.ray_miss_m == 0.2


def test_dense_lateral_rows_filters_low_confidence():
    from src.utils.ball_eval import dense_lateral_rows
    from src.utils.camera_projection import project_world_to_image
    K, R, t = _cam()
    P = np.array([0.0, 10.0, 0.11])
    uv = project_world_to_image(K, R, t, (0.0, 0.0), P.reshape(1, 3))[0]
    obs = [(4, (float(uv[0]), float(uv[1])), 0.9, "detector"),
           (5, (float(uv[0]), float(uv[1])), 0.1, "detector")]
    world = {4: tuple(P + np.array([0.1, 0.0, 0.0])), 5: tuple(P)}
    rows = dense_lateral_rows(world, obs, {4: (K, R, t), 5: (K, R, t)},
                              distortion=(0.0, 0.0), min_confidence=0.5)
    assert len(rows) == 1 and rows[0].frame == 4
    assert 0.05 < rows[0].lateral_m < 0.15
```

- [ ] **Step 2: Run to verify failure** — `ImportError: cannot import name 'eval_rows_at_anchors'`.

- [ ] **Step 3: Implement** (append to `ball_eval.py`)

```python
from collections.abc import Callable, Sequence   # top of file
from dataclasses import dataclass
from src.utils.camera_projection import project_world_to_image


@dataclass(frozen=True)
class AnchorEvalRow:
    frame: int
    state: str
    kind: str
    held_out: bool
    lateral_m: float | None
    err_3d_m: float | None
    reproj_px: float | None
    depth_m: float | None


@dataclass(frozen=True)
class FixEvalRow:
    frame: int
    err_3d_m: float | None
    ray_miss_m: float


@dataclass(frozen=True)
class DenseEvalRow:
    frame: int
    lateral_m: float
    confidence: float
    source: str


def eval_rows_at_anchors(world_by_frame, anchors, cams, *, ball_radius,
                         distortion, joint_world_fn=None,
                         held_out_frames=frozenset()):
    rows = []
    for anc in anchors:
        if anc.image_xy is None or anc.frame not in cams:
            continue
        K, R, t = cams[anc.frame]
        joint = None
        if joint_world_fn is not None and anc.state == "player_touch" \
                and anc.player_id and anc.bone:
            joint = joint_world_fn(anc.frame, anc.player_id, anc.bone)
        gt, kind = anchor_gt_world(anc, K, R, t, distortion,
                                   ball_radius=ball_radius, joint_world=joint)
        w = world_by_frame.get(anc.frame)
        if w is None:
            rows.append(AnchorEvalRow(anc.frame, anc.state, kind,
                                      anc.frame in held_out_frames,
                                      None, None, None, None))
            continue
        P = np.asarray(w, dtype=float)
        C, d = pixel_ray(anc.image_xy, K, R, t, distortion)
        lateral, depth = point_ray_distance(P, C, d)
        uvp = project_world_to_image(K, R, t, distortion, P.reshape(1, 3))[0]
        reproj = float(np.linalg.norm(uvp - np.asarray(anc.image_xy, float)))
        err3d = float(np.linalg.norm(P - gt)) if gt is not None else None
        rows.append(AnchorEvalRow(anc.frame, anc.state, kind,
                                  anc.frame in held_out_frames,
                                  lateral, err3d, reproj, depth))
    return tuple(rows)


def eval_rows_at_fixes(world_by_frame, fixes):
    rows = []
    for frame, xyz, ray_miss in fixes:
        w = world_by_frame.get(int(frame))
        err = (float(np.linalg.norm(np.asarray(w, float) - np.asarray(xyz, float)))
               if w is not None else None)
        rows.append(FixEvalRow(int(frame), err, float(ray_miss)))
    return tuple(rows)


def dense_lateral_rows(world_by_frame, observations, cams, *, distortion,
                       min_confidence):
    rows = []
    for frame, uv, conf, source in observations:
        if conf < min_confidence or frame not in cams:
            continue
        w = world_by_frame.get(int(frame))
        if w is None:
            continue
        K, R, t = cams[frame]
        C, d = pixel_ray(uv, K, R, t, distortion)
        lateral, _ = point_ray_distance(np.asarray(w, float), C, d)
        rows.append(DenseEvalRow(int(frame), lateral, float(conf), str(source)))
    return tuple(rows)
```

- [ ] **Step 4: Run to verify pass** — full `tests/test_ball_eval.py` green.

- [ ] **Step 5: Commit** — `git add`, `git commit -m "feat: ball eval error rows (anchors/fixes/dense)"`.

---

### Task 3: Naturalness validator

**Files:**
- Modify: `src/utils/ball_eval.py`
- Test: `tests/test_ball_eval.py` (append)

**Interfaces:**
- Produces:
  - `NaturalnessCfg(max_heading_change_deg=12.0, min_speed_m_s=2.0, event_window_frames=2, flight_g_tol=0.25, roll_speedup_tol=0.15, min_roll_speed_m_s=1.0)` frozen dataclass.
  - `Violation(frame, kind, value, limit)` frozen; kind ∈ `{"heading_break", "flight_gravity", "roll_speedup"}`.
  - `naturalness_violations(frames, event_frames, fps, *, cfg=NaturalnessCfg()) -> tuple[Violation, ...]`
    where `frames` is the `BallTrack.frames` sequence (uses `.frame`,
    `.world_xyz`, `.state`) and `event_frames` is any int collection.

- [ ] **Step 1: Write the failing tests**

```python
def _mk_frames(worlds, states=None):
    from src.schemas.ball_track import BallFrame
    out = []
    for i, w in enumerate(worlds):
        out.append(BallFrame(frame=i, world_xyz=w,
                             state=(states[i] if states else "grounded"),
                             confidence=1.0))
    return out


def test_naturalness_flags_heading_break_away_from_events():
    from src.utils.ball_eval import naturalness_violations
    # Straight roll at 6 m/s that suddenly turns 90° at frame 5 — no event.
    fps = 30.0
    pts = [(0.2 * i, 0.0, 0.11) for i in range(6)]
    pts += [(1.0, 0.2 * (i - 5), 0.11) for i in range(6, 11)]
    v = naturalness_violations(_mk_frames(pts), event_frames=set(), fps=fps)
    assert any(x.kind == "heading_break" and abs(x.frame - 5) <= 1 for x in v)


def test_naturalness_allows_break_at_event():
    from src.utils.ball_eval import naturalness_violations
    fps = 30.0
    pts = [(0.2 * i, 0.0, 0.11) for i in range(6)]
    pts += [(1.0, 0.2 * (i - 5), 0.11) for i in range(6, 11)]
    v = naturalness_violations(_mk_frames(pts), event_frames={5}, fps=fps)
    assert not [x for x in v if x.kind == "heading_break"]


def test_naturalness_flags_linear_flight_as_gravity_violation():
    from src.utils.ball_eval import naturalness_violations
    # "Flight" frames moving in a straight 3D line (no gravity curvature).
    pts = [(0.3 * i, 0.0, 1.0 + 0.05 * i) for i in range(12)]
    v = naturalness_violations(_mk_frames(pts, states=["flight"] * 12),
                               event_frames=set(), fps=30.0)
    assert any(x.kind == "flight_gravity" for x in v)


def test_naturalness_accepts_true_parabola_and_steady_roll():
    from src.utils.ball_eval import naturalness_violations
    fps, g = 30.0, -9.81
    v0 = np.array([6.0, 0.0, 5.0])
    pts = [tuple(np.array([0, 0, 0.11]) + v0 * (i / fps)
                 + 0.5 * np.array([0, 0, g]) * (i / fps) ** 2) for i in range(15)]
    v = naturalness_violations(_mk_frames(pts, states=["flight"] * 15),
                               event_frames=set(), fps=fps)
    assert not [x for x in v if x.kind == "flight_gravity"]
    roll = [(0.2 * i, 0.05 * i, 0.11) for i in range(12)]   # constant velocity
    v2 = naturalness_violations(_mk_frames(roll), event_frames=set(), fps=fps)
    assert not v2
```

- [ ] **Step 2: Run to verify failure.**

- [ ] **Step 3: Implement** — central-difference velocities; heading change via
  `atan2` deltas on the XY velocity; flight z-acceleration via second
  differences compared to `-9.81 * (1 ± flight_g_tol)`; roll speed-up check on
  consecutive XY speeds. Skip any window containing a `None` world or a frame
  within `event_window_frames` of an event. Flight-gravity checks need ≥ 3
  consecutive flight frames; evaluate the *median* z-accel per flight run (not
  per frame) so float noise doesn't fire it.

```python
@dataclass(frozen=True)
class NaturalnessCfg:
    max_heading_change_deg: float = 12.0
    min_speed_m_s: float = 2.0
    event_window_frames: int = 2
    flight_g_tol: float = 0.25
    roll_speedup_tol: float = 0.15
    min_roll_speed_m_s: float = 1.0


@dataclass(frozen=True)
class Violation:
    frame: int
    kind: str
    value: float
    limit: float


def naturalness_violations(frames, event_frames, fps, *, cfg=NaturalnessCfg()):
    ev = set(int(e) for e in event_frames)
    near_event = lambda f: any(abs(f - e) <= cfg.event_window_frames for e in ev)
    by_frame = {f.frame: f for f in frames}
    idx = sorted(by_frame)
    out: list[Violation] = []
    # Per-frame velocities on consecutive triples.
    for f in idx:
        a, b, c = by_frame.get(f - 1), by_frame.get(f), by_frame.get(f + 1)
        if not (a and b and c) or None in (a.world_xyz, b.world_xyz, c.world_xyz):
            continue
        pa, pb, pc = (np.asarray(x.world_xyz, float) for x in (a, b, c))
        v_in, v_out = (pb - pa) * fps, (pc - pb) * fps
        sp_in, sp_out = np.linalg.norm(v_in[:2]), np.linalg.norm(v_out[:2])
        if min(sp_in, sp_out) > cfg.min_speed_m_s and not near_event(f):
            dh = np.degrees(abs(np.arctan2(v_out[1], v_out[0])
                                - np.arctan2(v_in[1], v_in[0])))
            dh = min(dh, 360.0 - dh)
            if dh > cfg.max_heading_change_deg:
                out.append(Violation(f, "heading_break", float(dh),
                                     cfg.max_heading_change_deg))
        if b.state != "flight" and not near_event(f) \
                and min(sp_in, sp_out) > cfg.min_roll_speed_m_s \
                and sp_out > sp_in * (1.0 + cfg.roll_speedup_tol):
            out.append(Violation(f, "roll_speedup", float(sp_out / sp_in),
                                 1.0 + cfg.roll_speedup_tol))
    # Flight runs: median vertical acceleration ≈ g.
    run: list[int] = []
    for f in idx + [None]:
        fr = by_frame.get(f) if f is not None else None
        if fr is not None and fr.state == "flight" and fr.world_xyz is not None:
            run.append(f); continue
        if len(run) >= 4:
            zs = np.array([by_frame[r].world_xyz[2] for r in run])
            az = np.diff(zs, 2) * fps * fps
            az_med = float(np.median(az))
            lo, hi = -9.81 * (1 + cfg.flight_g_tol), -9.81 * (1 - cfg.flight_g_tol)
            interior = [r for r in run[1:-1] if not near_event(r)]
            if interior and not (lo <= az_med <= hi):
                out.append(Violation(run[len(run) // 2], "flight_gravity",
                                     az_med, -9.81))
        run = []
    return tuple(out)
```

- [ ] **Step 4: Run to verify pass.**
- [ ] **Step 5: Commit** — `git commit -m "feat: ball naturalness validator"`.

---

### Task 4: Stratified hold-out split

**Files:**
- Modify: `src/utils/ball_eval.py`
- Test: `tests/test_ball_eval.py` (append)

**Interfaces:**
- Produces: `split_anchors(anchors, *, fold, n_folds=2) -> tuple[tuple[BallAnchor, ...], tuple[BallAnchor, ...]]`
  — `(kept, held_out)`; deterministic; within each state class (sorted by
  frame), member `i` is held out iff `i % n_folds == fold`. Running every fold
  holds each anchor out exactly once.

- [ ] **Step 1: Write the failing tests**

```python
def test_split_anchors_stratified_and_exhaustive():
    from src.utils.ball_eval import split_anchors
    anchors = [BallAnchor(frame=f, state=("grounded" if f % 2 else "airborne_low"),
                          image_xy=(10.0, 10.0)) for f in range(10)]
    seen = set()
    for fold in (0, 1):
        kept, held = split_anchors(anchors, fold=fold, n_folds=2)
        assert len(kept) + len(held) == 10 and not (set(a.frame for a in kept)
                                                    & set(a.frame for a in held))
        # Both state classes present in the kept half.
        assert {a.state for a in kept} == {"grounded", "airborne_low"}
        seen |= {a.frame for a in held}
    assert seen == set(range(10))          # every anchor held out exactly once


def test_split_anchors_deterministic():
    from src.utils.ball_eval import split_anchors
    anchors = [BallAnchor(frame=f, state="grounded", image_xy=(1.0, 1.0))
               for f in (5, 1, 9, 3)]
    a = split_anchors(anchors, fold=0)
    b = split_anchors(anchors, fold=0)
    assert [x.frame for x in a[0]] == [x.frame for x in b[0]]
```

- [ ] **Step 2: Run to verify failure.**
- [ ] **Step 3: Implement**

```python
def split_anchors(anchors, *, fold, n_folds=2):
    by_state: dict[str, list] = {}
    for a in sorted(anchors, key=lambda a: a.frame):
        by_state.setdefault(a.state, []).append(a)
    kept, held = [], []
    for state in sorted(by_state):
        for i, a in enumerate(by_state[state]):
            (held if i % n_folds == fold else kept).append(a)
    kept.sort(key=lambda a: a.frame)
    held.sort(key=lambda a: a.frame)
    return tuple(kept), tuple(held)
```

- [ ] **Step 4: Run to verify pass.**
- [ ] **Step 5: Commit** — `git commit -m "feat: stratified anchor hold-out split"`.

---

### Task 5: Report summary

**Files:**
- Modify: `src/utils/ball_eval.py`
- Test: `tests/test_ball_eval.py` (append)

**Interfaces:**
- Produces: `summarize(anchor_rows, fix_rows, dense_rows, violations, *, threshold_m=0.20) -> dict`
  JSON-safe dict with keys `anchors_held_out`, `anchors_kept`, `fixes`,
  `dense`, `naturalness`. Each anchors/fixes/dense entry:
  `{"n", "n_3d", "p50", "p95", "max", "n_over"}` computed over `err_3d_m`
  where present else `lateral_m` (fixes: over `err_3d_m`; dense: `lateral_m`);
  `naturalness`: `{"n_violations", "by_kind": {...}}`. `None` errors are
  counted in `"n_missing"` per section, excluded from percentiles.

- [ ] **Step 1: Write the failing test**

```python
def test_summarize_shapes_and_thresholds():
    from src.utils.ball_eval import (AnchorEvalRow, DenseEvalRow, FixEvalRow,
                                     Violation, summarize)
    a = [AnchorEvalRow(1, "grounded", "ground_exact", True, 0.05, 0.25, 3.0, 40.0),
         AnchorEvalRow(2, "airborne_low", "ray_only", True, 0.10, None, 2.0, 50.0),
         AnchorEvalRow(3, "grounded", "ground_exact", False, 0.01, 0.01, 0.5, 45.0)]
    s = summarize(a, [FixEvalRow(4, 0.30, 0.2)],
                  [DenseEvalRow(5, 0.40, 0.9, "detector")],
                  [Violation(6, "heading_break", 30.0, 12.0)])
    ho = s["anchors_held_out"]
    assert ho["n"] == 2 and ho["n_over"] == 1 and abs(ho["max"] - 0.25) < 1e-9
    assert s["anchors_kept"]["n"] == 1
    assert s["fixes"]["n_over"] == 1 and s["dense"]["n_over"] == 1
    assert s["naturalness"]["by_kind"]["heading_break"] == 1
    import json; json.dumps(s)     # JSON-safe
```

- [ ] **Step 2: Run to verify failure.**
- [ ] **Step 3: Implement** — helper `_stats(errs, threshold)` returning the
  dict; anchors partitioned by `held_out`; per-row error = `err_3d_m` if not
  `None` else `lateral_m`.

```python
def _stats(errs, threshold, n_missing=0):
    vals = np.array([e for e in errs if e is not None], dtype=float)
    if len(vals) == 0:
        return {"n": 0, "n_3d": 0, "p50": None, "p95": None, "max": None,
                "n_over": 0, "n_missing": int(n_missing)}
    return {
        "n": int(len(vals)),
        "p50": float(np.median(vals)),
        "p95": float(np.percentile(vals, 95)),
        "max": float(vals.max()),
        "n_over": int((vals > threshold).sum()),
        "n_missing": int(n_missing),
    }


def summarize(anchor_rows, fix_rows, dense_rows, violations, *,
              threshold_m=0.20):
    def anchor_err(r):
        return r.err_3d_m if r.err_3d_m is not None else r.lateral_m
    held = [r for r in anchor_rows if r.held_out]
    kept = [r for r in anchor_rows if not r.held_out]
    def _anchor_stats(rows):
        errs = [anchor_err(r) for r in rows]
        st = _stats([e for e in errs if e is not None], threshold_m,
                    n_missing=sum(1 for e in errs if e is None))
        st["n_3d"] = sum(1 for r in rows if r.err_3d_m is not None)
        return st
    by_kind: dict[str, int] = {}
    for v in violations:
        by_kind[v.kind] = by_kind.get(v.kind, 0) + 1
    return {
        "anchors_held_out": _anchor_stats(held),
        "anchors_kept": _anchor_stats(kept),
        "fixes": _stats([r.err_3d_m for r in fix_rows], threshold_m,
                        n_missing=sum(1 for r in fix_rows if r.err_3d_m is None)),
        "dense": _stats([r.lateral_m for r in dense_rows], threshold_m),
        "naturalness": {"n_violations": len(violations), "by_kind": by_kind},
        "threshold_m": threshold_m,
    }
```

  (Adjust the Step-1 test if `_stats` key set differs — the test is the
  contract; make them agree, keeping `n`, `p50`, `p95`, `max`, `n_over`.)

- [ ] **Step 4: Run to verify pass.**
- [ ] **Step 5: Commit** — `git commit -m "feat: ball eval summary"`.

---

### Task 6: Overlay runner + CLI `scripts/eval_ball_accuracy.py`

**Files:**
- Create: `scripts/eval_ball_accuracy.py`
- Test: `tests/test_ball_eval_cli.py`

**Interfaces:**
- Consumes: everything above; `BallStage`, `BallDetector`,
  `BallAnchorSet`, `BallTrack`, `CameraTrack`, `BallFixSet`
  (`src/schemas/ball_fixes.py` — check its `load` API before use);
  `PlayerContext.load(output_dir, shot_id, per_frame_K=…, per_frame_R=…, per_frame_t=…, distortion=…)`.
- Produces:
  - `build_overlay(src_output: Path, tmp_root: Path, shot_id: str, kept: BallAnchorSet | None) -> Path`
    — creates `tmp_root/overlay` with symlinks to every top-level entry of
    `src_output` EXCEPT `ball*` (skip `ball`, `ball_pre_*`, `ball_finetune_*`,
    `renders`); real `ball/` dir containing the kept-anchors file (or a copy of
    the original when `kept is None`).
  - `run_and_evaluate(src_output: Path, shot_id: str, *, detector: str, holdout: bool, n_folds: int, config: dict) -> dict`
    — returns `{"summary": …, "per_fold": […], "clip": shot_id, "detector": …}`.
  - CLI: `.venv311/bin/python scripts/eval_ball_accuracy.py --output output --shot gberch [--holdout] [--n-folds 2] [--detector noop|wasb] [--json PATH] [--threshold 0.20]`

- [ ] **Step 1: Write the failing test** — unit-test `build_overlay` with a fake
  output dir; integration-test the full runner on gberch (skip when absent).

```python
import json
from pathlib import Path

import pytest


@pytest.mark.unit
def test_build_overlay_symlinks_inputs_and_filters_anchors(tmp_path):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from eval_ball_accuracy import build_overlay
    from src.schemas.ball_anchor import BallAnchor, BallAnchorSet

    src = tmp_path / "out"
    for d in ("camera", "refined_poses", "shots", "tracks", "ball", "ball_pre_x"):
        (src / d).mkdir(parents=True)
    (src / "camera" / "s1_camera_track.json").write_text("{}")
    full = BallAnchorSet(clip_id="s1", anchors=(
        BallAnchor(frame=1, state="grounded", image_xy=(5.0, 5.0)),))
    full.save(src / "ball" / "s1_ball_anchors.json")

    kept = BallAnchorSet(clip_id="s1", anchors=())
    ov = build_overlay(src, tmp_path / "work", "s1", kept)
    assert (ov / "camera").is_symlink()
    assert not (ov / "ball").is_symlink() and (ov / "ball").is_dir()
    assert not (ov / "ball_pre_x").exists()
    saved = BallAnchorSet.load(ov / "ball" / "s1_ball_anchors.json")
    assert len(saved.anchors) == 0


@pytest.mark.integration
def test_run_and_evaluate_gberch_noop():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from eval_ball_accuracy import run_and_evaluate
    import yaml
    root = Path(__file__).resolve().parents[1]
    out = root / "output"
    if not (out / "ball" / "gberch_ball_anchors.json").exists():
        pytest.skip("gberch output not present")
    cfg = yaml.safe_load(open(root / "config" / "default.yaml"))
    rep = run_and_evaluate(out, "gberch", detector="noop", holdout=True,
                           n_folds=2, config=cfg)
    assert rep["summary"]["anchors_held_out"]["n"] > 0
    json.dumps(rep)
```

  (`BallAnchorSet` constructor/save/load kwargs: verify against
  `src/schemas/ball_anchor.py` and adapt the test to the real API — e.g. if
  `clip_id` is named differently or `save` takes a directory.)

- [ ] **Step 2: Run to verify failure** — module not found.

- [ ] **Step 3: Implement the script** — key parts:

```python
"""Grade the ball stage's dense track against ground truth (sub-20cm spec §4).

Re-runs the ball stage in a temp OVERLAY of the output dir (symlinked stage
inputs; ball/ replaced with kept anchors) so real outputs are never touched.
"""
from __future__ import annotations

import argparse, json, sys, tempfile
from pathlib import Path

import numpy as np, yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.schemas.ball_anchor import BallAnchorSet
from src.schemas.ball_track import BallTrack
from src.schemas.camera_track import CameraTrack
from src.stages.ball import BallStage
from src.utils.ball_detector import BallDetector
from src.utils import ball_eval as BE

_SKIP_TOP = ("ball", "renders")          # prefixes not linked into overlays


class NoopDetector(BallDetector):
    SUPPORTS_REDETECT = False
    def detect(self, frame):
        return None


def build_overlay(src_output, tmp_root, shot_id, kept):
    ov = Path(tmp_root) / "overlay"
    ov.mkdir(parents=True, exist_ok=True)
    for entry in sorted(Path(src_output).iterdir()):
        if any(entry.name == p or entry.name.startswith(p + "_")
               for p in _SKIP_TOP) or entry.name.startswith("ball"):
            continue
        (ov / entry.name).symlink_to(entry.resolve())
    (ov / "ball").mkdir(exist_ok=True)
    anchors_src = Path(src_output) / "ball" / f"{shot_id}_ball_anchors.json"
    dst = ov / "ball" / f"{shot_id}_ball_anchors.json"
    if kept is not None:
        kept.save(dst)
    elif anchors_src.exists():
        dst.write_text(anchors_src.read_text())
    return ov
```

  `run_and_evaluate` mechanics (follow `rerun_and_measure.py`, updated):
  1. Load full `BallAnchorSet`, `CameraTrack`; build `cams = {f: (K, R, t)}`
     with the per-frame `t` fallback to `cam.t_world` exactly as in
     `docs/superpowers/notes/ball-accuracy/measure_anchor_error.py`.
  2. Folds: `[None]` when `holdout=False`, else `range(n_folds)`; per fold call
     `BE.split_anchors`, `build_overlay`, then
     `stage = BallStage(config=config, output_dir=overlay, ball_detector=det)`;
     `stage._run_shot(shot_id, overlay/"shots"/f"{shot_id}.mp4", overlay/"camera"/f"{shot_id}_camera_track.json", overlay/"ball"/f"{shot_id}_ball_track.json", config["ball"], det)`.
     `det` = `NoopDetector()` or the stage's real WASB detector for
     `--detector wasb` (build it the same way `BallStage` does by default —
     read `src/stages/ball.py`'s constructor/detector factory and reuse it).
  3. Load the produced track; `world_by_frame = {f.frame: f.world_xyz for f in track.frames if f.world_xyz}`.
  4. `joint_world_fn`: build `PlayerContext.load(overlay, shot_id, per_frame_K=…, per_frame_R=…, per_frame_t=…, distortion=…)` once and pass its
     `.joint_world`. If loading raises, pass `None` (rows fall back to ray_only).
  5. Event frames for naturalness: keyframe frames from the produced
     `{shot}_ball_keyframes.json` (all keyframes are events/waypoints — direction
     may legitimately change at any of them) plus segment boundary frames.
  6. GT extras: fixes from `{src_output}/ball/{shot_id}_ball_fixes.json` if
     present (also check `output-origi-global` layout: the CLI takes the fixes
     path as `--fixes` override); observations from the OVERLAY run's
     `{shot}_ball_observations.json` (for wasb runs) →
     `BE.dense_lateral_rows(min_confidence=0.5)`.
  7. Evaluate vs the FULL anchor set with `held_out_frames` = that fold's
     held-out frames; aggregate rows across folds (held-out rows only from
     their own fold; kept rows from fold 0 only, to avoid double counting);
     `BE.summarize(...)`; return the dict.

- [ ] **Step 4: Run tests** — unit green;
  `.venv311/bin/python -m pytest tests/test_ball_eval_cli.py -q -m "unit or integration"` green
  (integration runs the real gberch no-op rerun — expect ~1 min).

- [ ] **Step 5: Commit** — `git commit -m "feat: eval_ball_accuracy CLI with overlay hold-out runner"`.

---

### Task 7: Baseline measurement, recorded and committed

**Files:**
- Create: `docs/superpowers/notes/ball-accuracy/2026-08-17-baseline.md`

**Interfaces:**
- Consumes: the CLI. Produces: the committed baseline every later change is
  compared against (spec §3 "gates ratchet from baseline").

- [ ] **Step 1:** Run hold-out + full eval on all four clips with the no-op
  detector:

```bash
for spec in "output gberch" "output-origi origi01" "output-kroupi kroupi01" "output-japan s013"; do
  set -- $spec
  .venv311/bin/python scripts/eval_ball_accuracy.py --output "$1" --shot "$2" \
    --holdout --detector noop --json "docs/superpowers/notes/ball-accuracy/baseline_$2_noop_holdout.json"
  .venv311/bin/python scripts/eval_ball_accuracy.py --output "$1" --shot "$2" \
    --detector noop --json "docs/superpowers/notes/ball-accuracy/baseline_$2_noop_full.json"
done
```

- [ ] **Step 2:** Real-detector runs where runtime allows (assess: gberch
  first; if < ~10 min/clip on MPS, run all):

```bash
.venv311/bin/python scripts/eval_ball_accuracy.py --output output --shot gberch \
  --holdout --detector wasb --json docs/superpowers/notes/ball-accuracy/baseline_gberch_wasb_holdout.json
.venv311/bin/python scripts/eval_ball_accuracy.py --output output-origi --shot origi01 \
  --detector wasb --fixes output-origi-global/ball/origi01_ball_fixes.json \
  --json docs/superpowers/notes/ball-accuracy/baseline_origi01_wasb_full.json
```

- [ ] **Step 3:** Write `2026-08-17-baseline.md`: one table per clip
  (held-out p50/p95/max/n_over by kind; fixes; dense; naturalness counts), a
  "dominant error sources" section naming the worst offenders with frames, and
  the exact commands used.

- [ ] **Step 4:** Commit baseline JSONs + markdown:

```bash
git add docs/superpowers/notes/ball-accuracy/
git commit -m "docs: sub-20cm campaign baseline measurements"
```

---

## Self-review notes

- Spec coverage: §4 fully (Tasks 1–6), §3 measurement of A1–A5 (A1 Tasks 2+4+6,
  A2 Task 2/6, A3 Task 2/6, A4 Task 3, A5 covered by anchor rows at touch
  keyframes + W2's later gates), §11 step 1 (Task 7). W2–W5 are separate plans
  by design (scope check).
- Real-API risk: `BallAnchor`/`BallAnchorSet`/`BallFixSet` constructor details
  and the stage's default WASB detector factory are checked in-task before the
  tests are finalized (called out inline in Tasks 1 and 6).
- Type consistency: `cams: dict[int, (K, R, t)]` + separate `distortion` kwarg
  used identically in Tasks 1, 2, 6.
