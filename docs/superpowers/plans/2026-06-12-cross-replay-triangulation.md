# Cross-Replay Triangulation Implementation Plan (Phase 1.5)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Phase 1.5 of [the ball v2 design](../specs/2026-06-12-ball-v2-design.md): triangulate ball detections across sync-grouped shots (live + replay) into sparse 3D fixes that soft-constrain the flight fits — killing monocular depth ambiguity on the events broadcasts replay.

**Architecture:** The ball stage becomes three passes: detect-all-shots (existing pass-1 + second pass, unchanged), triangulate-per-sync-group (new pure module: local sub-frame offset refinement by ray-miss minimisation, then gated midpoint triangulation; fixes persisted per shot), solve-all-shots (existing flow, with fixes passed to the piecewise solver and threaded into the LM flight fitters as weighted 3D residuals). Operator sync offsets are never overwritten; shots without a synced partner take the exact pre-1.5 path.

**Tech Stack:** numpy/scipy pure modules (light-venv testable), existing `pixel_ray`/fixtures, `SyncMap` schema (v1 files auto-migrate), LM fitters in `bundle_adjust.py`.

**Conventions:** Repo rules: TDD, frozen dataclasses, type annotations. Tests: `.venv311/bin/python -m pytest <files> -q` from repo root. Conventional commits, NO Co-Authored-By/attribution footer. Reference reading: `prototypes/replay_triangulation.py` (validated prototype this productionizes), `src/schemas/sync_map.py` (offset convention: shot frame f ↔ reference frame f − frame_offset).

---

## File structure

| File | Action | Responsibility |
|---|---|---|
| `src/utils/ball_cross_replay.py` | Create | Pure triangulation: cfg, ray pairing, offset refinement, gated triangulation |
| `src/schemas/ball_fixes.py` | Create | `BallFix`/`BallFixSet` sidecar schema (save/load) |
| `src/utils/bundle_adjust.py` | Modify | Optional `world_fixes` residuals in parabola + Magnus fitters |
| `src/utils/ball_piecewise_solver.py` | Modify | `world_fixes` param; flight fits consume in-range fixes; diag counts |
| `src/stages/ball.py` | Modify | Three-pass run(); group triangulation pass; fixes into solve |
| `config/default.yaml` | Modify | `ball.cross_replay.*` block |
| `src/pipeline/quality_report.py` | Modify | Surface `cross_replay` diag |
| `tests/test_ball_cross_replay.py` | Create | Unit tests (two synthetic cameras) |
| `tests/test_ball_fixes_schema.py` | Create | Schema round-trip |
| `tests/test_bundle_adjust_world_fixes.py` | Create | Fitter fix-residual tests |
| `tests/test_ball_piecewise_fixes.py` | Create | Solver fix-consumption tests |
| `tests/test_ball_stage_cross_replay.py` | Create | Stage integration (synthetic 2-camera group) |
| `prototypes/cross_view_consistency.py` | Create | Validation metric script (untracked-style helper, committed) |

---

### Task 1: Pure triangulation module

**Files:** Create `src/utils/ball_cross_replay.py`; Test `tests/test_ball_cross_replay.py`.

- [ ] **Step 1: Failing tests.** Create `tests/test_ball_cross_replay.py`:

```python
"""Cross-replay triangulation: pairing, offset refinement, gated fixes.

Two synthetic broadcast cameras 35 m apart observe the same analytic
trajectory; detections in the second view are offset by a known sync
delta. Pins: triangulation recovers 3D within tolerance, gates reject
poor parallax / inconsistent rays, and ray-miss refinement recovers a
deliberately-wrong saved offset.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.utils.ball_cross_replay import (
    CrossReplayCfg,
    interp_uv,
    refine_pair_offset,
    triangulate_pair,
    triangulate_rays,
)
from tests.fixtures.ball_synthetic import broadcast_camera, project_track

FPS = 25.0


def _two_cameras():
    KA, RA, tA = broadcast_camera(cam_centre=(52.5, -20.0, 15.0))
    KB, RB, tB = broadcast_camera(cam_centre=(20.0, -25.0, 18.0))
    return (KA, RA, tA), (KB, RB, tB)


def _arc_worlds(n: int) -> dict[int, np.ndarray]:
    # 2-second arc: kick at (40, 30), apex ~5 m.
    out = {}
    v0 = np.array([8.0, 4.0, 9.81])
    p0 = np.array([40.0, 30.0, 0.11])
    for f in range(n):
        t = f / FPS
        out[f] = p0 + v0 * t + 0.5 * np.array([0.0, 0.0, -9.81]) * t * t
    return out


def _obs(pixels: dict[int, tuple[float, float]], conf: float = 0.9):
    return {f: (uv, conf) for f, uv in pixels.items()}


@pytest.mark.unit
def test_triangulate_rays_recovers_point():
    (KA, RA, tA), (KB, RB, tB) = _two_cameras()
    world = np.array([45.0, 32.0, 3.0])
    pixA = project_track({0: world}, KA, RA, tA)[0]
    pixB = project_track({0: world}, KB, RB, tB)[0]
    point, miss, parallax = triangulate_rays(pixA, KA, RA, tA, pixB, KB, RB, tB)
    assert np.linalg.norm(point - world) < 0.05
    assert miss < 0.01
    assert parallax > 8.0


@pytest.mark.unit
def test_interp_uv_linear_between_neighbours():
    obs = _obs({10: (100.0, 200.0), 12: (110.0, 220.0)})
    uv = interp_uv(obs, 11.0, min_conf=0.3, max_span=3)
    assert uv == pytest.approx((105.0, 210.0))
    uv_frac = interp_uv(obs, 10.5, min_conf=0.3, max_span=3)
    assert uv_frac == pytest.approx((102.5, 205.0))
    assert interp_uv(obs, 20.0, min_conf=0.3, max_span=3) is None


@pytest.mark.unit
def test_triangulate_pair_inliers_and_gates():
    (KA, RA, tA), (KB, RB, tB) = _two_cameras()
    n = 40
    worlds = _arc_worlds(n)
    pixA = project_track(worlds, KA, RA, tA)
    pixB = project_track(worlds, KB, RB, tB)
    # B is offset by +5: B frame f shows the event A saw at frame f-5.
    obsA = _obs(pixA)
    obsB = _obs({f + 5: uv for f, uv in pixB.items()})
    camsA = {f: (KA, RA, tA) for f in range(n)}
    camsB = {f: (KB, RB, tB) for f in range(n + 5)}
    cfg = CrossReplayCfg()
    fixes = triangulate_pair(
        obs_a=obsA, cams_a=camsA, obs_b=obsB, cams_b=camsB,
        offset_b_minus_a=5.0, cfg=cfg,
    )
    assert len(fixes) >= 30
    for fx in fixes:
        assert np.linalg.norm(np.asarray(fx.xyz) - worlds[fx.frame_a]) < 0.10
        assert fx.ray_miss_m <= cfg.max_ray_miss_m
        assert fx.parallax_deg >= cfg.min_parallax_deg
    # A decoy detection in B far off the epipolar geometry is rejected.
    obsB_bad = dict(obsB)
    obsB_bad[10 + 5] = ((pixB[10][0] + 300.0, pixB[10][1]), 0.9)
    fixes_bad = triangulate_pair(
        obs_a=obsA, cams_a=camsA, obs_b=obsB_bad, cams_b=camsB,
        offset_b_minus_a=5.0, cfg=cfg,
    )
    assert all(fx.frame_a != 10 for fx in fixes_bad)


@pytest.mark.unit
def test_refine_pair_offset_recovers_true_delta():
    (KA, RA, tA), (KB, RB, tB) = _two_cameras()
    n = 40
    worlds = _arc_worlds(n)
    obsA = _obs(project_track(worlds, KA, RA, tA))
    obsB = _obs({f + 5: uv for f, uv in
                 project_track(worlds, KB, RB, tB).items()})
    camsA = {f: (KA, RA, tA) for f in range(n)}
    camsB = {f: (KB, RB, tB) for f in range(n + 5)}
    cfg = CrossReplayCfg()
    # Saved offset is wrong by 2 frames; refinement must find ~5.0.
    refined, median_miss, n_pairs = refine_pair_offset(
        obs_a=obsA, cams_a=camsA, obs_b=obsB, cams_b=camsB,
        saved_offset=7.0, cfg=cfg,
    )
    assert refined == pytest.approx(5.0, abs=cfg.offset_search_step)
    assert n_pairs >= cfg.min_pairs_for_refine
    assert median_miss < 0.05


@pytest.mark.unit
def test_refine_pair_offset_keeps_saved_when_too_few_pairs():
    (KA, RA, tA), (KB, RB, tB) = _two_cameras()
    worlds = _arc_worlds(4)
    obsA = _obs(project_track(worlds, KA, RA, tA))
    obsB = _obs({f + 5: uv for f, uv in
                 project_track(worlds, KB, RB, tB).items()})
    camsA = {f: (KA, RA, tA) for f in range(4)}
    camsB = {f: (KB, RB, tB) for f in range(9)}
    refined, _, n_pairs = refine_pair_offset(
        obs_a=obsA, cams_a=camsA, obs_b=obsB, cams_b=camsB,
        saved_offset=7.0, cfg=CrossReplayCfg(),
    )
    assert n_pairs < CrossReplayCfg().min_pairs_for_refine
    assert refined == 7.0
```

- [ ] **Step 2: Run, verify FAIL** (ModuleNotFoundError).

- [ ] **Step 3: Implement** `src/utils/ball_cross_replay.py`:

```python
"""Cross-replay triangulation (ball v2 Phase 1.5).

Shots in one sync group film the same real moment from different
cameras. Pairing their per-frame ball detections through the group's
sync offset turns them into an ad-hoc stereo rig: the midpoint of the
two rays' common perpendicular is a 3D fix, and the perpendicular's
length (ray miss) is a built-in consistency gate.

The saved sync offset is refined LOCALLY by minimising median ray miss
over a sub-frame grid — sync_map.json is never written (operator
offsets win); the caller surfaces disagreements as a review cue.

Pure module: no file/video access; the stage owns I/O.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.utils.camera_projection import pixel_ray

# observation maps: frame -> ((u, v), confidence)
Obs = dict[int, tuple[tuple[float, float], float]]
# camera maps: frame -> (K, R, t)
Cams = dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]


@dataclass(frozen=True)
class CrossReplayCfg:
    enabled: bool = True
    min_conf: float = 0.3
    max_ray_miss_m: float = 1.0
    min_parallax_deg: float = 8.0
    offset_search_radius_frames: float = 4.0
    offset_search_step: float = 0.25
    min_pairs_for_refine: int = 8
    fix_weight_px_per_m: float = 30.0


@dataclass(frozen=True)
class PairFix:
    """One triangulated point, keyed by both shots' frames."""

    frame_a: int
    frame_b: int
    xyz: tuple[float, float, float]
    ray_miss_m: float
    parallax_deg: float


def triangulate_rays(
    uv_a, K_a, R_a, t_a,
    uv_b, K_b, R_b, t_b,
    distortion_a: tuple[float, float] = (0.0, 0.0),
    distortion_b: tuple[float, float] = (0.0, 0.0),
) -> tuple[np.ndarray, float, float]:
    """Midpoint of the common perpendicular between two pixel rays.

    Returns ``(point, miss_m, parallax_deg)``; ``point`` is NaN and the
    gates are unsatisfiable when either ray depth is non-positive.
    """
    c1, d1 = pixel_ray(uv_a, K_a, R_a, t_a, distortion_a)
    c2, d2 = pixel_ray(uv_b, K_b, R_b, t_b, distortion_b)
    cos = float(np.clip(abs(np.dot(d1, d2)), 0.0, 1.0))
    parallax = float(np.degrees(np.arccos(cos)))
    A = np.stack([d1, -d2], axis=1)
    b = c2 - c1
    (s, u), *_ = np.linalg.lstsq(A, b, rcond=None)
    if s <= 0 or u <= 0:  # intersection behind a camera: invalid pair
        return np.full(3, np.nan), float("inf"), 0.0
    p1 = c1 + s * d1
    p2 = c2 + u * d2
    return (p1 + p2) / 2.0, float(np.linalg.norm(p1 - p2)), parallax


def interp_uv(
    obs: Obs,
    f: float,
    min_conf: float,
    max_span: int = 3,
) -> tuple[float, float] | None:
    """Observation at a (possibly fractional) frame, linearly
    interpolated between the nearest confident detections no more than
    ``max_span`` frames apart."""
    lo, hi = int(np.floor(f)), int(np.ceil(f))
    if lo == hi:
        rec = obs.get(lo)
        return rec[0] if rec is not None and rec[1] >= min_conf else None
    a = next(
        ((g, obs[g]) for g in range(lo, lo - max_span, -1)
         if g in obs and obs[g][1] >= min_conf),
        None,
    )
    b = next(
        ((g, obs[g]) for g in range(hi, hi + max_span)
         if g in obs and obs[g][1] >= min_conf),
        None,
    )
    if a is None or b is None or b[0] - a[0] > max_span:
        return None
    w = (f - a[0]) / (b[0] - a[0])
    ua, va = a[1][0]
    ub, vb = b[1][0]
    return ((1 - w) * ua + w * ub, (1 - w) * va + w * vb)


def _pairs_at_offset(
    obs_a: Obs, cams_a: Cams, obs_b: Obs, cams_b: Cams,
    offset_b_minus_a: float, cfg: CrossReplayCfg,
):
    """Yield (frame_a, frame_b, uv_a, uv_b, cams...) for every B
    detection whose synced A-frame has (interpolated) evidence.

    Convention (sync_map): B frame f_b shows the instant A saw at
    ``f_a = f_b - offset_b_minus_a``.
    """
    for f_b, (uv_b, conf_b) in sorted(obs_b.items()):
        if conf_b < cfg.min_conf or f_b not in cams_b:
            continue
        f_a = f_b - offset_b_minus_a
        f_a_int = int(round(f_a))
        if f_a_int not in cams_a:
            continue
        uv_a = interp_uv(obs_a, f_a, cfg.min_conf)
        if uv_a is None:
            continue
        yield f_a_int, f_b, uv_a, uv_b


def refine_pair_offset(
    *,
    obs_a: Obs, cams_a: Cams, obs_b: Obs, cams_b: Cams,
    saved_offset: float, cfg: CrossReplayCfg,
    distortion_a: tuple[float, float] = (0.0, 0.0),
    distortion_b: tuple[float, float] = (0.0, 0.0),
) -> tuple[float, float, int]:
    """Scan offsets around the saved value, minimising median ray miss.

    Returns ``(refined_offset, median_miss_at_refined, n_pairs)``. The
    saved offset is returned unchanged when fewer than
    ``min_pairs_for_refine`` pairs exist at it.
    """

    def _median_miss(offset: float) -> tuple[float, int]:
        misses = []
        for f_a, f_b, uv_a, uv_b in _pairs_at_offset(
            obs_a, cams_a, obs_b, cams_b, offset, cfg,
        ):
            K_a, R_a, t_a = cams_a[f_a]
            K_b, R_b, t_b = cams_b[f_b]
            _, miss, _ = triangulate_rays(
                uv_a, K_a, R_a, t_a, uv_b, K_b, R_b, t_b,
                distortion_a, distortion_b,
            )
            if np.isfinite(miss):
                misses.append(miss)
        if not misses:
            return float("inf"), 0
        return float(np.median(misses)), len(misses)

    base_miss, base_pairs = _median_miss(saved_offset)
    if base_pairs < cfg.min_pairs_for_refine:
        return float(saved_offset), base_miss, base_pairs

    best = (float(saved_offset), base_miss, base_pairs)
    r = cfg.offset_search_radius_frames
    step = cfg.offset_search_step
    for off in np.arange(saved_offset - r, saved_offset + r + step / 2, step):
        miss, n = _median_miss(float(off))
        if n >= cfg.min_pairs_for_refine and miss < best[1]:
            best = (float(off), miss, n)
    return best


def triangulate_pair(
    *,
    obs_a: Obs, cams_a: Cams, obs_b: Obs, cams_b: Cams,
    offset_b_minus_a: float, cfg: CrossReplayCfg,
    distortion_a: tuple[float, float] = (0.0, 0.0),
    distortion_b: tuple[float, float] = (0.0, 0.0),
) -> list[PairFix]:
    """Gated triangulation of every synced detection pair."""
    fixes: list[PairFix] = []
    for f_a, f_b, uv_a, uv_b in _pairs_at_offset(
        obs_a, cams_a, obs_b, cams_b, offset_b_minus_a, cfg,
    ):
        K_a, R_a, t_a = cams_a[f_a]
        K_b, R_b, t_b = cams_b[f_b]
        point, miss, parallax = triangulate_rays(
            uv_a, K_a, R_a, t_a, uv_b, K_b, R_b, t_b,
            distortion_a, distortion_b,
        )
        if not np.all(np.isfinite(point)):
            continue
        if miss > cfg.max_ray_miss_m or parallax < cfg.min_parallax_deg:
            continue
        fixes.append(PairFix(
            frame_a=f_a, frame_b=f_b,
            xyz=(float(point[0]), float(point[1]), float(point[2])),
            ray_miss_m=miss, parallax_deg=parallax,
        ))
    return fixes
```

- [ ] **Step 4: Run** `tests/test_ball_cross_replay.py` — PASS.
- [ ] **Step 5: Commit** `feat(ball): pure cross-replay triangulation module`.

---

### Task 2: Fixes sidecar schema

**Files:** Create `src/schemas/ball_fixes.py`; Test `tests/test_ball_fixes_schema.py`.

- [ ] **Step 1: Failing test** — `tests/test_ball_fixes_schema.py`:

```python
"""BallFixSet sidecar round-trip."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.schemas.ball_fixes import BallFix, BallFixSet


@pytest.mark.unit
def test_round_trip(tmp_path: Path):
    fs = BallFixSet(
        clip_id="origi01",
        group_id="",
        cross_replay={
            "partner_shots": ["origi02"],
            "saved_offset": -142.0,
            "refined_offset": -144.0,
            "offset_disagreement_frames": 2.0,
            "n_pairs": 21,
            "n_inlier_fixes": 11,
            "median_ray_miss_m": 0.31,
            "median_parallax_deg": 26.4,
        },
        fixes=(
            BallFix(frame=328, xyz=(12.7, 50.1, 8.28), ray_miss_m=0.15,
                    parallax_deg=23.1, partner_shot="origi02",
                    partner_frame=184),
        ),
    )
    p = tmp_path / "origi01_ball_fixes.json"
    fs.save(p)
    loaded = BallFixSet.load(p)
    assert loaded == fs
    assert loaded.fixes[0].xyz == pytest.approx((12.7, 50.1, 8.28))
```

- [ ] **Step 2: FAIL.**  **Step 3: Implement** `src/schemas/ball_fixes.py` (follow the dataclass+save/load style of `src/schemas/ball_anchor.py` — frozen dataclasses, `save()` writes indented json atomically if the neighbouring schemas do, `load()` reconstructs; `fixes` as a tuple). Fields exactly as the test uses.
- [ ] **Step 4: PASS.**  **Step 5: Commit** `feat(ball): ball fixes sidecar schema`.

---

### Task 3: `world_fixes` in the LM flight fitters

**Files:** Modify `src/utils/bundle_adjust.py`; Test `tests/test_bundle_adjust_world_fixes.py`.

Both `fit_parabola_to_image_observations` and `fit_magnus_trajectory` gain:

```python
    world_fixes: list[tuple[int, np.ndarray, float]] | None = None,
```

— `(frame_index, xyz, weight_px_per_m)` triples, `frame_index` in the same
absolute-clip-frame space as `observations`. Implementation: inside the
residual function, after computing positions `pts` at observation times,
ALSO integrate/evaluate positions at each fix's time (`(fix_frame -
frame_idx[0]) / fps`; for the parabola this is closed-form `p0 + v0*dt +
0.5*g*dt²`, for Magnus add the fix times into the RK4 `sample_times` grid)
and append `weight * (pos - xyz)` (3 residuals per fix). Fixes whose frame
falls outside `[first_obs_frame, last_obs_frame]` are still valid (the
model extrapolates); keep them. `world_fixes=None` or `[]` must leave the
residual vector byte-identical to today (regression guard).

- [ ] **Step 1: Failing tests** — `tests/test_bundle_adjust_world_fixes.py`:

```python
"""world_fixes: 3D soft constraints in the LM flight fitters."""

from __future__ import annotations

import numpy as np
import pytest

from src.utils.bundle_adjust import (
    fit_magnus_trajectory,
    fit_parabola_to_image_observations,
)
from tests.fixtures.ball_synthetic import broadcast_camera, project_track

FPS = 25.0
G = np.array([0.0, 0.0, -9.81])


def _arc(n, p0, v0):
    return {f: p0 + v0 * (f / FPS) + 0.5 * G * (f / FPS) ** 2
            for f in range(n)}


def _fit_inputs(worlds, K, R, t, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    pix = project_track(worlds, K, R, t)
    obs = []
    for f, (u, v) in sorted(pix.items()):
        du, dv = rng.normal(0.0, noise, 2) if noise else (0.0, 0.0)
        obs.append((f, (u + du, v + dv)))
    n = len(obs)
    return obs, [K] * n, [R] * n, [t] * n


@pytest.mark.unit
def test_parabola_unchanged_without_fixes():
    K, R, t = broadcast_camera()
    worlds = _arc(30, np.array([40.0, 30.0, 0.11]), np.array([8.0, 4.0, 9.0]))
    obs, Ks, Rs, ts = _fit_inputs(worlds, K, R, t)
    a = fit_parabola_to_image_observations(obs, Ks=Ks, Rs=Rs, t_world=ts, fps=FPS)
    b = fit_parabola_to_image_observations(
        obs, Ks=Ks, Rs=Rs, t_world=ts, fps=FPS, world_fixes=[])
    assert np.allclose(a[0], b[0]) and np.allclose(a[1], b[1])
    assert a[2] == pytest.approx(b[2])


@pytest.mark.unit
def test_parabola_fixes_pull_depth_to_truth():
    """Sparse noisy monocular obs leave depth soft; two exact 3D fixes
    must pull the recovered trajectory onto the truth."""
    K, R, t = broadcast_camera()
    p0, v0 = np.array([40.0, 30.0, 0.11]), np.array([8.0, 4.0, 9.0])
    worlds = _arc(30, p0, v0)
    sparse = {f: worlds[f] for f in range(0, 30, 6)}  # 5 obs only
    obs, Ks, Rs, ts = _fit_inputs(sparse, K, R, t, noise=2.0)
    fixes = [(8, np.asarray(worlds[8]), 30.0), (20, np.asarray(worlds[20]), 30.0)]
    p0n, v0n, _ = fit_parabola_to_image_observations(
        obs, Ks=Ks, Rs=Rs, t_world=ts, fps=FPS)
    p0f, v0f, _ = fit_parabola_to_image_observations(
        obs, Ks=Ks, Rs=Rs, t_world=ts, fps=FPS, world_fixes=fixes)
    err_n = np.linalg.norm(p0n - p0) + np.linalg.norm(v0n - v0)
    err_f = np.linalg.norm(p0f - p0) + np.linalg.norm(v0f - v0)
    assert err_f <= err_n + 1e-9
    assert np.linalg.norm(p0f - p0) < 0.5


@pytest.mark.unit
def test_magnus_accepts_fixes_and_respects_weight():
    K, R, t = broadcast_camera()
    p0, v0 = np.array([40.0, 30.0, 0.11]), np.array([8.0, 4.0, 9.0])
    worlds = _arc(30, p0, v0)
    obs, Ks, Rs, ts = _fit_inputs(worlds, K, R, t, noise=1.0)
    wrong_fix = [(15, np.asarray(worlds[15]) + np.array([0.0, 5.0, 0.0]), 1e-6)]
    p0w, v0w, _, _ = fit_magnus_trajectory(
        obs, Ks=Ks, Rs=Rs, t_world=ts, fps=FPS,
        omega_mag_bound=10.0, world_fixes=wrong_fix)
    # Near-zero weight: the wrong fix must not drag the solution.
    assert np.linalg.norm(p0w - p0) < 0.5
```

- [ ] **Step 2: FAIL.**  **Step 3: Implement** as described above (read both fitters fully first; the Magnus fitter's sample-times grid must include fix times — extend `_integrate_magnus_positions` sampling or evaluate by inserting fix dts into the integration timeline and indexing them out; do NOT change the existing returned residual semantics).
- [ ] **Step 4: PASS** + run `tests/test_ball_flight.py tests/test_ball_spin_fit.py tests/test_bundle_adjust_free_p0_knots.py tests/test_bundle_adjust_knot_frames.py tests/test_bundle_adjust_p0_fixed.py` (no regressions).
- [ ] **Step 5: Commit** `feat(ball): world-fix soft constraints in LM flight fitters`.

---

### Task 4: Solver consumes fixes

**Files:** Modify `src/utils/ball_piecewise_solver.py`; Test `tests/test_ball_piecewise_fixes.py`.

`solve_piecewise(...)` gains `world_fixes: Mapping[int, tuple[np.ndarray, float]] | None = None` (frame → (xyz, weight)); `_Solver` stores it. Every LM flight-fit call site (node-bracketed ballistic spans, open-span arcs, Magnus refinement) passes the fixes whose frames fall inside the segment's frame range, converted to the fitters' `world_fixes` list form. Diag: each entry in `diagnostics["segments"]` gains `"fixes_used": <int>`; the per-segment value is 0 when none.

- [ ] **Step 1: Failing test** — `tests/test_ball_piecewise_fixes.py`: synthetic single-camera arc (reuse `tests/fixtures/ball_synthetic.py` helpers and the solver-call pattern from `tests/test_ball_piecewise_solver.py`'s open-span tests): build a 40-frame flight with sparse noisy detections between two grounded runs, call `solve_piecewise` twice — without fixes and with 3 exact fixes (truth points, weight 30.0) — and assert: (a) with-fixes solution's mean 3D error vs truth over the flight frames is ≤ the without-fixes error; (b) `diagnostics["segments"]` contains a segment with `fixes_used >= 1` in the with-fixes run and all `fixes_used == 0` without; (c) results are deterministic across two identical calls. Write the full test by adapting the existing open-span test's scaffolding (same fixtures, same SolverCfg()).
- [ ] **Step 2: FAIL.**  **Step 3: Implement.** Read the solver's flight-fit call sites first (`fit_parabola_to_image_observations` / `fit_magnus_trajectory` calls); add a small helper `self._fixes_in(fa, fb) -> list[tuple[int, np.ndarray, float]]`.
- [ ] **Step 4: PASS** + `tests/test_ball_piecewise_solver.py tests/test_ball_grounded.py tests/test_ball_flight.py` green.
- [ ] **Step 5: Commit** `feat(ball): piecewise solver consumes cross-replay world fixes`.

---

### Task 5: Stage three-pass restructure + group triangulation

**Files:** Modify `src/stages/ball.py`; Test `tests/test_ball_stage_cross_replay.py`.

**Contract (read `run()` and `_run_shot` first):**

1. Split `_run_shot` into:
   - `_detect_shot(shot_id, clip_path, camera_path, ball_out_path, cfg, detector) -> _DetectArtifacts | None` — everything from camera load through the observations-sidecar write (manual-anchor load + detect loop + second pass + coverage). `_DetectArtifacts` (frozen dataclass, module-level) carries: steps, raw_confidences, sources, detection_coverage, camera fields (per_frame_K/R/t, distortion, camera object), manual_by_frame, n_clip, paths.
   - `_solve_shot(artifacts, cfg, fixes: dict[int, tuple[np.ndarray, float]] | None)` — the rest (player context, events, anchors, nodes, solve with `world_fixes=fixes`, outputs, diag). The diag gains `"cross_replay": <dict | None>` (the per-shot summary computed in step 2 below, None when no partner).
2. `run()` becomes: loop A `_detect_shot` for every active shot (collect artifacts); then `_triangulate_groups(artifacts_by_shot, cfg)` (below); then loop B `_solve_shot` per shot. The legacy no-manifest path detects then solves its single shot with `fixes=None` (no group → unchanged).
3. `_triangulate_groups`: load `shots/sync_map.json` if present (`SyncMap.load` — v1 migrates automatically). For each group with ≥ 2 member shots that are in `artifacts_by_shot`: take the member with `frame_offset` closest to 0 as side A (the reference), every other member as side B; for each (A, B) pair: build `Obs` maps from the artifacts (`{frame: (uv, conf)}` from steps with `uv is not None`, `frame in sources`, not outlier — i.e. accepted evidence incl. second-pass) and `Cams` maps from per-frame K/R/t; `delta_saved = offset_B - offset_A`; `refine_pair_offset`; `triangulate_pair` at the refined delta; convert each `PairFix` into per-shot fixes `{frame: (xyz, weight)}` for BOTH shots (weight = `cfg.fix_weight_px_per_m`); write each shot's `ball/<shot>_ball_fixes.json` via `BallFixSet` (fixes for that shot keyed by its own frame, partner metadata + the cross_replay summary dict from the spec); return `fixes_by_shot` + summaries.
4. Config helper `_cross_replay_cfg(cfg)` mapping `ball.cross_replay.*` → `CrossReplayCfg` (same pattern as `_second_pass_cfg`).
5. `config/default.yaml`: add the `cross_replay:` block from the spec (after the `second_pass:` block, with a comment noting operator sync offsets are never overwritten and disagreements are flagged).

- [ ] **Step 1: Failing integration test** — `tests/test_ball_stage_cross_replay.py`: copy the helper functions from `tests/test_ball_stage_second_pass.py` (`_camera_pose`, `_save_camera_track`, `_write_blank_clip`, `_project`) and add a second camera pose helper (`_camera_pose_b`, centre shifted ~30 m along x — derive it the same way `_camera_pose` builds R/t, e.g. centre (20.0, -30.0, 25.0) looking at the pitch). Build a two-shot manifest (`shotA`, `shotB`) + v1 `sync_map.json` (`{"reference_shot": "shotA", "alignments": [{"shot_id": "shotA", "frame_offset": 0, ...}, {"shot_id": "shotB", "frame_offset": 5, ...}]}`); both clips blank; scripted detections: a rolling+flight trajectory projected through each camera (shotB shifted +5 frames). Use one `FakeBallDetector` per... NOTE: `BallStage` takes ONE injected detector used for all shots — script it with shotA's detections followed by shotB's (detect() cycles in call order across the run; shotA has n frames then shotB n+5 — compute the concatenated list carefully, and disable second_pass + appearance_bridge in config to keep call order deterministic). Assertions: (a) `ball/shotA_ball_fixes.json` and `ball/shotB_ball_fixes.json` exist with ≥ 10 fixes each and `cross_replay.refined_offset` ≈ 5; (b) each fix xyz within 0.15 m of the synthetic truth at that frame; (c) both shots' diag contains `cross_replay` with `n_inlier_fixes ≥ 10`; (d) both ball tracks solve without error and shotA's flight frames' world positions are within 0.5 m of truth; (e) a single-shot run (kroupi-style: one shot, no sync_map) produces NO fixes sidecar and a diag `cross_replay: null`.
- [ ] **Step 2: FAIL.**  **Step 3: Implement** per the contract.
- [ ] **Step 4: PASS** + full stage suites: `tests/test_ball_stage.py tests/test_ball_stage_second_pass.py tests/test_ball_stage_anchors.py tests/test_ball_stage_keyframes.py tests/test_ball_stage_layered.py` green (single-shot tests must be unaffected — no sync_map in their fixtures).
- [ ] **Step 5: Commit** `feat(ball): three-pass ball stage with cross-replay triangulated fixes`.

---

### Task 6: Quality report + validation tooling

- [ ] **Step 1:** `src/pipeline/quality_report.py` `_ball_shot_entry`: add `"cross_replay": diag.get("cross_replay")` passthrough (same pattern as `detection_coverage`); extend the existing quality-report diag test fixture with a `cross_replay` dict and assert passthrough (same style as the `detection_coverage` assertion added previously).
- [ ] **Step 2:** Create `prototypes/cross_view_consistency.py`: loads two shots' solved `_ball_track.json` + the fixes sidecar's refined offset, maps frames through the offset, and prints median/p90 3D distance between the two tracks on frames where both have `world_xyz` (split by state: flight vs grounded). Self-contained, argv: `output_dir shotA shotB` (default `output-origi origi01 origi02`).
- [ ] **Step 3:** Run targeted tests; commit `feat(quality): cross-replay diagnostics in quality report + consistency tool`.

---

### Task 7: Full suite + real-clip validation

- [ ] **Step 1:** `.venv311/bin/python -m pytest tests/ -q -k "ball or anchor or tracker"` — all green.
- [ ] **Step 2:** Wipe + re-run ball stage on `output-origi` (restore `origi01_ball_anchors.json` from `output-origi/ball_pre_phase1/` first — manual anchors are stage INPUTS living in the ball dir) and on `output-kroupi` (restore its anchors file too). CPU runs, ~4 min each.
- [ ] **Step 3:** Check acceptance (spec Phase 1.5): ≥ 8 inlier fixes on the origi pair; refined offset ≈ −144 with disagreement flagged; fix-consuming segments within 3D agreement; kroupi diag `cross_replay: null` and metrics byte-comparable to Phase 1 final. Run `prototypes/cross_view_consistency.py` before/after (the "before" = Phase-1 tracks snapshotted in `output-origi/ball_phase1_default/`).
- [ ] **Step 4:** Append a `## Phase 1.5 validation results` section to the design spec with the table; commit `docs: cross-replay triangulation validation results`.
