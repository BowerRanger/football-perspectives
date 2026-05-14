# Static-Camera Solve From Detected Painted Lines — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a camera track with a single fixed camera centre (zero body motion) solved from the sub-pixel painted-line detector, wired into the camera stage and validated on the gberch clip.

**Architecture:** Three new pure, independently-testable units — a static-C bundle adjustment (`static_line_solver.py`), a C-profile diagnostic (`static_c_profile.py`), and a detect-all-frames helper (`detect_lines_for_frames`) — orchestrated by a new camera-stage method `_refine_with_static_line_solve`. The C-profile seeds the bundle adjustment with the correct camera centre and answers whether sub-pixel static-C is reachable; an iterative re-detection loop removes per-frame bootstrap bias.

**Tech Stack:** Python 3.14, numpy, scipy (`least_squares`, `lil_matrix`), OpenCV (`cv2`), pytest. Tests run with `.venv/bin/python3 -m pytest`.

---

## Background for the implementer

You are joining a project that reconstructs 3D football scenes from a single broadcast camera. The camera stage (`src/stages/camera.py`) solves a per-frame camera `(K, R, t)` where `K` is intrinsics, `R` is world→camera rotation, `t` is world→camera translation. The camera *centre* in world coordinates is `C = -R.T @ t`. The design intent: the broadcast camera body is fixed, so `C` should be one value for the whole clip — only pan/tilt (`R`) and zoom (`fx`, the `K[0,0]`/`K[1,1]` focal length) vary per frame.

A painted-line detector (`src/utils/line_detector.py`) finds white pitch lines in each frame at sub-pixel accuracy and returns `LineObservation` objects (a detected `image_segment` paired with a known `world_segment`). Per-frame independent solves from these lines hit 0.99 px line-fitting RMS but let `C` wander 16 m across the clip — an under-determined per-frame problem. This plan builds a solve that locks one `C` across all frames.

Full design rationale: `docs/superpowers/specs/2026-05-14-static-camera-line-solve-design.md`.

### Key existing code you will reuse

- `src/schemas/anchor.py` — `LineObservation(name, image_segment, world_segment, world_direction)`, `LandmarkObservation(name, image_xy, world_xyz)`, `Anchor(frame, landmarks, lines)`, `AnchorSet(clip_id, image_size, anchors, stadium)`. All frozen dataclasses. `AnchorSet.load(path)` parses the JSON.
- `src/schemas/camera_track.py` — `CameraTrack(clip_id, fps, image_size, t_world, frames, principal_point, camera_centre, distortion)` and `CameraFrame(frame, K, R, confidence, is_anchor, t)`. `camera_centre` is `tuple[float,float,float] | None`.
- `src/utils/anchor_solver.py`:
  - `_make_K(fx, cx, cy) -> np.ndarray` — builds the 3×3 intrinsic matrix `[[fx,0,cx],[0,fx,cy],[0,0,1]]`.
  - `_is_rich(anchor, min_points=6) -> bool` — whether an anchor can solo-solve.
  - `_point_residuals_distorted(landmarks, K, rvec, tvec, distortion_k1k2) -> np.ndarray` — 2 residuals per landmark, distortion is a 2-tuple `(k1, k2)`.
  - `_line_residuals(lines, K, R, t) -> np.ndarray` — 2 residuals per line, **not distortion-aware** (uses raw `K @ cam`). Do NOT extend this; the new module gets its own distortion-aware variant.
- `src/utils/camera_projection.py` — `project_world_to_image(K, R, t, distortion, world_points)` where `distortion` is `(k1, k2)`.
- `src/utils/line_detector.py` — `DetectorConfig(search_strip_px, sample_step_px, min_gradient, ...)`, `DetectedLine(name, image_segment, world_segment, confidence, n_samples)` (a `NamedTuple`), `detect_painted_lines_in_frame(frame_bgr, K, R, t, distortion, world_lines, cfg) -> list[DetectedLine]`.
- `src/utils/line_camera_refine.py` — `PITCH_LINE_CATALOGUE` (dict of name → world_segment, painted z=0 pitch lines only), `refine_camera_from_lines(...)` (the existing independent per-frame path — leave it alone).
- `src/utils/pitch_lines_catalogue.py` — `LINE_CATALOGUE` (dict of name → world_segment or world_direction).

### Test conventions

- Tests live in `tests/`, named `test_*.py`. Run a single test: `.venv/bin/python3 -m pytest tests/test_foo.py::test_name -v`.
- `tests/test_anchor_solver.py` has reusable geometry constants and helpers you should mirror:
  - `R_BASE`, `T_BASE` — a physically valid broadcast pose (camera at world `(52.5, -30, 30)`).
  - `_yaw(angle_deg) -> np.ndarray` — world-z yaw applied as `R_BASE @ R_yaw.T`.
  - `_project(K, R, t, world) -> (u, v)` — pinhole projection.
  - `_make_line(K, R, t, name, alpha=0.2, beta=0.8) -> LineObservation` — projects two points along `LINE_CATALOGUE[name]` to build a clean synthetic line observation.
- `tests/fixtures/synthetic_clip.py` — `render_synthetic_clip(...)` renders a dot-content clip; **note it renders point landmarks as dots, not painted lines**, so it cannot exercise the line detector directly. Task 3 includes a line-drawing helper for that.
- Mark fast tests `@pytest.mark.unit`, file-IO/stage tests `@pytest.mark.integration`.

---

## File structure

**New files:**
- `src/utils/static_line_solver.py` — `StaticCameraSolution` dataclass, `_project_points_distorted`, `_line_residuals_distorted`, `solve_static_camera_from_lines`. One responsibility: given per-frame line observations + a seed, return one locked camera centre + per-frame `(R, fx)`.
- `src/utils/static_c_profile.py` — `CProfileResult` dataclass, `make_c_grid`, `profile_camera_centre`. One responsibility: given per-frame line observations, sweep candidate camera centres and report line-RMS-vs-C.
- `scripts/profile_static_c.py` — thin CLI over `profile_camera_centre` for experiment runs.
- `tests/test_static_line_solver.py`, `tests/test_static_c_profile.py`, `tests/test_detect_lines_for_frames.py`, `tests/test_camera_stage_static_line.py`.

**Modified files:**
- `src/utils/line_camera_refine.py` — add `detect_lines_for_frames`.
- `src/stages/camera.py` — add `_refine_with_static_line_solve`, branch the `line_extraction` path on `static_camera`, write the line-derived `camera_centre`, include `K/R/t` in the `detected_lines.json` side-output.
- `config/default.yaml` — add `camera.line_extraction_lens_model`.
- `scripts/global_solve_from_lines.py`, `scripts/iterative_global_solve.py` — rewire to call the new modules.
- `docs/superpowers/notes/2026-05-14-camera-1px-experiment.md` — add a Phase 4 section.

### Design note on the lens model (consciously simplifies the spec's Step-1 gate)

The spec frames Brown-Conrady as a separate build step gated by the C-profile result. In practice the distortion-aware line residual uses `cv2.projectPoints`, which already takes a 5-element distortion vector `[k1, k2, p1, p2, k3]`. Supporting both `pinhole_k1k2` and `brown_conrady` is therefore a 5-line difference in parameter packing, not a separate subsystem. So **Task 1 builds both lens models** (it is the same code). The C-profile result still gates *engagement*: Task 6 (checkpoint) decides which model the config defaults to and whether Task 7's Brown-Conrady validation pass is worth running. The genuinely deferred rung remains zoom-dependent distortion.

---

## Task 1: `static_line_solver.py` — the static-C bundle adjustment

**Files:**
- Create: `src/utils/static_line_solver.py`
- Test: `tests/test_static_line_solver.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_static_line_solver.py`:

```python
"""Unit tests for the static-camera bundle adjustment from detected lines."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.schemas.anchor import LineObservation
from src.utils.pitch_lines_catalogue import LINE_CATALOGUE
from src.utils.static_line_solver import (
    StaticCameraSolution,
    solve_static_camera_from_lines,
)

# Physically valid broadcast pose (mirrors tests/test_anchor_solver.py).
_LOOK = np.array([0.0, 64.0, -30.0])
_LOOK = _LOOK / np.linalg.norm(_LOOK)
_RIGHT = np.array([1.0, 0.0, 0.0])
_DOWN = np.cross(_LOOK, _RIGHT)
R_BASE = np.array([_RIGHT, _DOWN, _LOOK], dtype=float)
C_TRUE = np.array([52.5, -30.0, 30.0])
IMAGE_SIZE = (1920, 1080)
CX, CY = IMAGE_SIZE[0] / 2.0, IMAGE_SIZE[1] / 2.0
FX_TRUE = 1500.0

# Lines visible from the broadcast pose — left penalty box plus halfway,
# touchlines and goal line so the geometry is well spread.
_LINE_NAMES = [
    "left_18yd_front", "left_18yd_near_edge", "left_18yd_far_edge",
    "left_6yd_front", "near_touchline", "left_goal_line", "halfway_line",
]


def _yaw(angle_deg: float) -> np.ndarray:
    a = np.deg2rad(angle_deg)
    Ry = np.array([[np.cos(a), -np.sin(a), 0.0],
                   [np.sin(a), np.cos(a), 0.0],
                   [0.0, 0.0, 1.0]])
    return R_BASE @ Ry.T


def _project(K, R, t, world):
    cam = R @ np.asarray(world, float) + t
    pix = K @ cam
    return float(pix[0] / pix[2]), float(pix[1] / pix[2])


def _make_line(K, R, t, name, alpha=0.2, beta=0.8) -> LineObservation:
    seg = LINE_CATALOGUE[name]
    pa, pb = np.asarray(seg[0]), np.asarray(seg[1])
    A = pa + alpha * (pb - pa)
    B = pa + beta * (pb - pa)
    return LineObservation(
        name=name,
        image_segment=(_project(K, R, t, A), _project(K, R, t, B)),
        world_segment=seg,
    )


def _synthetic_clip(yaws, fxs):
    """Build per-frame clean line observations + per-frame seeds for a
    known static C."""
    per_frame_lines: dict[int, list[LineObservation]] = {}
    per_frame_seeds: dict[int, tuple[np.ndarray, float]] = {}
    for i, (yaw, fx) in enumerate(zip(yaws, fxs)):
        R = _yaw(yaw)
        t = -R @ C_TRUE
        K = np.array([[fx, 0, CX], [0, fx, CY], [0, 0, 1.0]])
        lines = []
        for name in _LINE_NAMES:
            seg = LINE_CATALOGUE[name]
            # only keep lines whose endpoints project in front of the camera
            cam = np.asarray(seg, float) @ R.T + t
            if (cam[:, 2] > 0.1).all():
                lines.append(_make_line(K, R, t, name))
        per_frame_lines[i] = lines
        rvec, _ = cv2.Rodrigues(R)
        per_frame_seeds[i] = (rvec.reshape(3), fx)
    return per_frame_lines, per_frame_seeds


@pytest.mark.unit
def test_solver_recovers_known_static_centre_from_clean_lines():
    yaws = [-6.0, -3.0, 0.0, 3.0, 6.0]
    fxs = [1500.0, 1510.0, 1520.0, 1530.0, 1540.0]
    per_frame_lines, per_frame_seeds = _synthetic_clip(yaws, fxs)

    # Seed C 1.5 m off truth, seeds rvec/fx slightly perturbed.
    c_seed = C_TRUE + np.array([1.5, -1.0, 0.8])
    perturbed = {
        f: (rv + np.deg2rad(1.5), fx * 1.02)
        for f, (rv, fx) in per_frame_seeds.items()
    }

    sol = solve_static_camera_from_lines(
        per_frame_lines, IMAGE_SIZE,
        c_seed=c_seed, lens_seed=(CX, CY, 0.0, 0.0),
        per_frame_seeds=perturbed, lens_model="pinhole_k1k2",
    )

    assert isinstance(sol, StaticCameraSolution)
    # Exactly one camera centre, recovered tight.
    assert np.linalg.norm(sol.camera_centre - C_TRUE) < 0.05
    # Every frame's t satisfies -R.T @ t == the single C.
    for fid, (K, R, t) in sol.per_frame_KRt.items():
        c_frame = -R.T @ t
        assert np.linalg.norm(c_frame - sol.camera_centre) < 1e-6
    # Clean synthetic lines → sub-pixel RMS.
    assert sol.line_rms_mean < 0.05


@pytest.mark.unit
def test_solver_returns_one_centre_for_every_frame():
    yaws = [-5.0, 0.0, 5.0]
    fxs = [1500.0, 1500.0, 1500.0]
    per_frame_lines, per_frame_seeds = _synthetic_clip(yaws, fxs)
    sol = solve_static_camera_from_lines(
        per_frame_lines, IMAGE_SIZE,
        c_seed=C_TRUE + 0.5, lens_seed=(CX, CY, 0.0, 0.0),
        per_frame_seeds=per_frame_seeds, lens_model="pinhole_k1k2",
    )
    assert set(sol.per_frame_KRt.keys()) == set(per_frame_lines.keys())
    assert sol.camera_centre.shape == (3,)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python3 -m pytest tests/test_static_line_solver.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.utils.static_line_solver'`

- [ ] **Step 3: Write minimal implementation**

Create `src/utils/static_line_solver.py`:

```python
"""Static-camera bundle adjustment from detected painted lines.

Solves one fixed camera centre ``C`` across every frame while pan/tilt
(``rvec``) and zoom (``fx``) vary per frame. The sub-pixel painted-line
observations from ``line_detector.py`` provide the constraints; optional
low-weight point-landmark hints catch gross basin errors without setting
the gauge (the lines' ``world_segment``s already fix it).

This is the reusable core behind ``scripts/global_solve_from_lines.py``
and the camera stage's static-C line path. Unlike the legacy global
solve it has NO per-frame motion budget — ``C`` is strictly one
3-vector.

Lens model:
  - ``pinhole_k1k2``  — shared ``(cx, cy, k1, k2)``; ``p1=p2=k3=0`` fixed.
  - ``brown_conrady`` — shared ``(cx, cy, k1, k2, p1, p2, k3)``.
Both feed a 5-element OpenCV distortion vector into the line residual.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from src.schemas.anchor import LandmarkObservation, LineObservation
from src.utils.anchor_solver import _make_K, _point_residuals_distorted

logger = logging.getLogger(__name__)

LensModel = Literal["pinhole_k1k2", "brown_conrady"]

# How many distortion coefficients are *free parameters* per lens model.
_N_FREE_DIST: dict[str, int] = {"pinhole_k1k2": 2, "brown_conrady": 5}


@dataclass(frozen=True)
class StaticCameraSolution:
    """Result of a static-camera line solve.

    ``camera_centre`` is the single locked C; every entry in
    ``per_frame_KRt`` satisfies ``-R.T @ t == camera_centre``.
    """

    camera_centre: np.ndarray                  # (3,)
    principal_point: tuple[float, float]       # (cx, cy)
    distortion: tuple[float, ...]              # (k1, k2) or (k1, k2, p1, p2, k3)
    per_frame_KRt: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]
    per_frame_line_rms: dict[int, float]
    lens_model: LensModel

    @property
    def line_rms_mean(self) -> float:
        vals = [v for v in self.per_frame_line_rms.values() if np.isfinite(v)]
        return float(np.mean(vals)) if vals else float("nan")


def _dist5(distortion: tuple[float, ...]) -> np.ndarray:
    """Pad a (k1,k2) or (k1,k2,p1,p2,k3) tuple to OpenCV's 5-vector."""
    out = np.zeros(5, dtype=np.float64)
    out[: len(distortion)] = distortion
    return out


def _project_points_distorted(
    K: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    dist5: np.ndarray,
    world_points: np.ndarray,
) -> np.ndarray:
    """Project (N,3) world points to (N,2) pixels with a 5-element
    OpenCV distortion vector."""
    pts = np.asarray(world_points, dtype=np.float64).reshape(-1, 1, 3)
    out, _ = cv2.projectPoints(
        pts,
        np.asarray(rvec, dtype=np.float64).reshape(3, 1),
        np.asarray(tvec, dtype=np.float64).reshape(3, 1),
        K.astype(np.float64),
        dist5,
    )
    return out.reshape(-1, 2)


def _line_residuals_distorted(
    lines: list[LineObservation],
    K: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    dist5: np.ndarray,
) -> np.ndarray:
    """2 residuals per position-known line: perpendicular pixel distance
    from each detected image endpoint to the distortion-projected world
    line. Direction-only lines are not produced by the painted-line
    detector, so they get a large finite sentinel."""
    out = np.empty(2 * len(lines))
    for i, ln in enumerate(lines):
        if ln.world_segment is None:
            out[2 * i] = out[2 * i + 1] = 1e6
            continue
        world = np.array(
            [ln.world_segment[0], ln.world_segment[1]], dtype=np.float64
        )
        # Behind-camera guard.
        cam = world @ cv2.Rodrigues(np.asarray(rvec, np.float64))[0].T + tvec
        if (cam[:, 2] <= 1e-3).all():
            out[2 * i] = out[2 * i + 1] = 1e6
            continue
        proj = _project_points_distorted(K, rvec, tvec, dist5, world)
        pa, pb = proj[0], proj[1]
        d = pb - pa
        norm = float(np.linalg.norm(d))
        if norm < 1e-6:
            out[2 * i] = out[2 * i + 1] = 1e6
            continue
        nx, ny = -d[1] / norm, d[0] / norm
        c = -(nx * pa[0] + ny * pa[1])
        for j, (u, v) in enumerate(ln.image_segment):
            out[2 * i + j] = nx * u + ny * v + c
    return out


def solve_static_camera_from_lines(
    per_frame_lines: dict[int, list[LineObservation]],
    image_size: tuple[int, int],
    *,
    c_seed: np.ndarray,
    lens_seed: tuple[float, float, float, float],
    per_frame_seeds: dict[int, tuple[np.ndarray, float]],
    point_hints: dict[int, list[LandmarkObservation]] | None = None,
    lens_model: LensModel = "pinhole_k1k2",
    point_hint_weight: float = 0.05,
    max_nfev: int = 4000,
) -> StaticCameraSolution:
    """Solve one fixed camera centre across all frames in
    ``per_frame_lines``.

    Parameters
    ----------
    per_frame_lines
        frame id -> list of position-known ``LineObservation``.
    image_size
        ``(width, height)`` — used only for ``(cx, cy)`` bounds.
    c_seed
        Initial guess for the shared camera centre (3,). The C-profile
        diagnostic supplies the good seed; a poor seed risks a local
        minimum.
    lens_seed
        ``(cx, cy, k1, k2)`` initial lens guess. ``p1, p2, k3`` seed at 0.
    per_frame_seeds
        frame id -> ``(rvec, fx)`` initial pose/zoom guess. Required for
        every frame in ``per_frame_lines``.
    point_hints
        Optional frame id -> landmark list, added at ``point_hint_weight``
        as a basin regulariser. Does NOT set the gauge.
    lens_model
        ``pinhole_k1k2`` (default) or ``brown_conrady``.
    """
    W, H = image_size
    fids = sorted(per_frame_lines.keys())
    if not fids:
        raise ValueError("solve_static_camera_from_lines: no frames given")

    n_dist = _N_FREE_DIST[lens_model]
    SHARED = 2 + n_dist + 3      # cx, cy, dist..., Cx, Cy, Cz
    PER = 4                      # rvec(3), fx
    n = len(fids)

    cx_s, cy_s, k1_s, k2_s = lens_seed
    p0 = np.empty(SHARED + PER * n)
    lower = np.empty_like(p0)
    upper = np.empty_like(p0)

    p0[0], p0[1] = cx_s, cy_s
    lower[0], upper[0] = W / 2 - 150, W / 2 + 150
    lower[1], upper[1] = H / 2 - 150, H / 2 + 150
    # distortion seeds + bounds
    dist_seed = [k1_s, k2_s, 0.0, 0.0, 0.0][:n_dist]
    dist_lo = [-0.5, -0.5, -0.1, -0.1, -0.5][:n_dist]
    dist_hi = [0.5, 0.5, 0.1, 0.1, 0.5][:n_dist]
    for j in range(n_dist):
        p0[2 + j] = dist_seed[j]
        lower[2 + j] = dist_lo[j]
        upper[2 + j] = dist_hi[j]
    c_base = 2 + n_dist
    p0[c_base : c_base + 3] = c_seed
    lower[c_base : c_base + 3] = np.asarray(c_seed) - 5.0
    upper[c_base : c_base + 3] = np.asarray(c_seed) + 5.0

    for i, fid in enumerate(fids):
        if fid not in per_frame_seeds:
            raise ValueError(f"per_frame_seeds missing frame {fid}")
        rvec0, fx0 = per_frame_seeds[fid]
        base = SHARED + i * PER
        p0[base : base + 3] = np.asarray(rvec0, dtype=np.float64).reshape(3)
        p0[base + 3] = fx0
        lower[base : base + 3] = -np.pi
        upper[base : base + 3] = np.pi
        lower[base + 3] = fx0 * 0.5
        upper[base + 3] = fx0 * 2.0

    hints = point_hints or {}

    def _unpack_shared(p: np.ndarray):
        cx, cy = float(p[0]), float(p[1])
        dist = tuple(float(p[2 + j]) for j in range(n_dist))
        C = p[c_base : c_base + 3]
        return cx, cy, dist, C

    def residuals(p: np.ndarray) -> np.ndarray:
        cx, cy, dist, C = _unpack_shared(p)
        d5 = _dist5(dist)
        parts: list[np.ndarray] = []
        for i, fid in enumerate(fids):
            base = SHARED + i * PER
            rvec = p[base : base + 3]
            fx = float(np.clip(p[base + 3], 50.0, 1e5))
            R_i, _ = cv2.Rodrigues(rvec)
            t_i = -R_i @ C
            K_i = _make_K(fx, cx, cy)
            parts.append(
                _line_residuals_distorted(per_frame_lines[fid], K_i, rvec, t_i, d5)
            )
            hint = hints.get(fid)
            if hint:
                parts.append(
                    point_hint_weight
                    * _point_residuals_distorted(
                        hint, K_i, rvec, t_i, (dist[0], dist[1])
                    )
                )
        return np.concatenate(parts) if parts else np.empty(0)

    # Sparse Jacobian: each frame's residuals touch SHARED cols + its PER cols.
    n_res_per_frame = []
    for fid in fids:
        n_res = 2 * len(per_frame_lines[fid])
        if fid in hints:
            n_res += 2 * len(hints[fid])
        n_res_per_frame.append(n_res)
    total_res = sum(n_res_per_frame)
    total_par = SHARED + PER * n
    spar = lil_matrix((total_res, total_par), dtype=np.uint8)
    row = 0
    for i, n_res in enumerate(n_res_per_frame):
        base = SHARED + i * PER
        for jr in range(n_res):
            spar[row + jr, 0:SHARED] = 1
            spar[row + jr, base : base + PER] = 1
        row += n_res

    result = least_squares(
        residuals, p0, bounds=(lower, upper),
        method="trf", loss="huber", f_scale=2.0,
        max_nfev=max_nfev, jac_sparsity=spar.tocsr(),
        xtol=1e-12, ftol=1e-12, gtol=1e-12,
    )

    cx, cy, dist, C = _unpack_shared(result.x)
    d5 = _dist5(dist)
    per_frame_KRt: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    per_frame_line_rms: dict[int, float] = {}
    for i, fid in enumerate(fids):
        base = SHARED + i * PER
        rvec = result.x[base : base + 3]
        fx = float(result.x[base + 3])
        R_i, _ = cv2.Rodrigues(rvec)
        t_i = -R_i @ np.asarray(C)
        K_i = _make_K(fx, cx, cy)
        per_frame_KRt[fid] = (K_i, R_i, t_i)
        r = _line_residuals_distorted(per_frame_lines[fid], K_i, rvec, t_i, d5)
        per_frame_line_rms[fid] = (
            float(np.sqrt((r ** 2).mean())) if r.size else float("nan")
        )

    return StaticCameraSolution(
        camera_centre=np.asarray(C, dtype=np.float64).copy(),
        principal_point=(cx, cy),
        distortion=tuple(dist),
        per_frame_KRt=per_frame_KRt,
        per_frame_line_rms=per_frame_line_rms,
        lens_model=lens_model,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python3 -m pytest tests/test_static_line_solver.py -v`
Expected: PASS (both tests)

- [ ] **Step 5: Commit**

```bash
git add src/utils/static_line_solver.py tests/test_static_line_solver.py
git commit -m "feat(camera): static-camera bundle adjustment from detected lines"
```

---

## Task 2: `static_c_profile.py` — the C-profile diagnostic

**Files:**
- Create: `src/utils/static_c_profile.py`
- Test: `tests/test_static_c_profile.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_static_c_profile.py`:

```python
"""Unit tests for the camera-centre profile diagnostic."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.schemas.anchor import LineObservation
from src.utils.pitch_lines_catalogue import LINE_CATALOGUE
from src.utils.static_c_profile import (
    CProfileResult,
    make_c_grid,
    profile_camera_centre,
)

_LOOK = np.array([0.0, 64.0, -30.0])
_LOOK = _LOOK / np.linalg.norm(_LOOK)
_RIGHT = np.array([1.0, 0.0, 0.0])
_DOWN = np.cross(_LOOK, _RIGHT)
R_BASE = np.array([_RIGHT, _DOWN, _LOOK], dtype=float)
C_TRUE = np.array([52.5, -30.0, 30.0])
IMAGE_SIZE = (1920, 1080)
CX, CY = IMAGE_SIZE[0] / 2.0, IMAGE_SIZE[1] / 2.0
_LINE_NAMES = [
    "left_18yd_front", "left_18yd_near_edge", "left_18yd_far_edge",
    "left_6yd_front", "near_touchline", "left_goal_line", "halfway_line",
]


def _yaw(angle_deg):
    a = np.deg2rad(angle_deg)
    Ry = np.array([[np.cos(a), -np.sin(a), 0.0],
                   [np.sin(a), np.cos(a), 0.0],
                   [0.0, 0.0, 1.0]])
    return R_BASE @ Ry.T


def _project(K, R, t, world):
    cam = R @ np.asarray(world, float) + t
    pix = K @ cam
    return float(pix[0] / pix[2]), float(pix[1] / pix[2])


def _synthetic_clip(yaws, fx=1500.0):
    per_frame_lines = {}
    per_frame_bootstrap = {}
    for i, yaw in enumerate(yaws):
        R = _yaw(yaw)
        t = -R @ C_TRUE
        K = np.array([[fx, 0, CX], [0, fx, CY], [0, 0, 1.0]])
        lines = []
        for name in _LINE_NAMES:
            seg = LINE_CATALOGUE[name]
            cam = np.asarray(seg, float) @ R.T + t
            if (cam[:, 2] > 0.1).all():
                pa = np.asarray(seg[0]) + 0.2 * (np.asarray(seg[1]) - np.asarray(seg[0]))
                pb = np.asarray(seg[0]) + 0.8 * (np.asarray(seg[1]) - np.asarray(seg[0]))
                lines.append(LineObservation(
                    name=name,
                    image_segment=(_project(K, R, t, pa), _project(K, R, t, pb)),
                    world_segment=seg,
                ))
        per_frame_lines[i] = lines
        rvec, _ = cv2.Rodrigues(R)
        per_frame_bootstrap[i] = (rvec.reshape(3), fx)
    return per_frame_lines, per_frame_bootstrap


@pytest.mark.unit
def test_make_c_grid_spans_the_requested_box():
    grid = make_c_grid(np.array([10.0, 20.0, 30.0]), extent_m=6.0, n_steps=5)
    assert grid.shape == (125, 3)
    assert np.isclose(grid[:, 0].min(), 4.0)
    assert np.isclose(grid[:, 0].max(), 16.0)
    # The centre is on the grid.
    assert np.any(np.all(np.isclose(grid, [10.0, 20.0, 30.0]), axis=1))


@pytest.mark.unit
def test_profile_argmin_lands_on_true_centre():
    per_frame_lines, bootstrap = _synthetic_clip([-6.0, -2.0, 2.0, 6.0])
    grid = make_c_grid(C_TRUE, extent_m=4.0, n_steps=5)
    result = profile_camera_centre(
        per_frame_lines, IMAGE_SIZE,
        c_grid=grid, lens_seed=(CX, CY, 0.0, 0.0),
        per_frame_bootstrap=bootstrap,
    )
    assert isinstance(result, CProfileResult)
    # Clean synthetic lines → the true C must be (near) the argmin.
    assert np.linalg.norm(result.argmin_c - C_TRUE) < 1.1
    # Mean RMS at the argmin is sub-pixel.
    best_idx = int(np.argmin(result.mean_rms))
    assert result.mean_rms[best_idx] < 0.1
    # Per-frame seeds at the argmin are returned for every frame.
    assert set(result.per_frame_seeds.keys()) == set(per_frame_lines.keys())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python3 -m pytest tests/test_static_c_profile.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.utils.static_c_profile'`

- [ ] **Step 3: Write minimal implementation**

Create `src/utils/static_c_profile.py`:

```python
"""Camera-centre profile diagnostic.

Sweeps candidate static camera centres on a 3-D grid. At each candidate
``C``, every frame's ``(rvec, fx)`` is solved independently with ``C``
pinned; the per-frame line RMS is aggregated to mean / P95 / max as a
function of ``C``.

Two jobs in one:
  * **seed-finder** — the argmin ``C`` (and the per-frame ``(rvec, fx)``
    at it) seed the static-camera bundle adjustment.
  * **honesty check** — if no grid ``C`` gets the mean RMS sub-pixel,
    a single static camera genuinely cannot fit the detected lines under
    the current lens model.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np
from scipy.optimize import least_squares

from src.schemas.anchor import LineObservation
from src.utils.anchor_solver import _make_K
from src.utils.static_line_solver import _dist5, _line_residuals_distorted

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CProfileResult:
    """Output of :func:`profile_camera_centre`."""

    grid_points: np.ndarray                                   # (M, 3)
    mean_rms: np.ndarray                                      # (M,)
    p95_rms: np.ndarray                                       # (M,)
    max_rms: np.ndarray                                       # (M,)
    argmin_c: np.ndarray                                      # (3,)
    per_frame_seeds: dict[int, tuple[np.ndarray, float]]      # at argmin_c


def make_c_grid(
    c_center: np.ndarray, *, extent_m: float, n_steps: int
) -> np.ndarray:
    """Build an (n_steps**3, 3) grid of candidate centres spanning
    ``c_center +/- extent_m`` on each axis. ``n_steps`` should be odd so
    ``c_center`` itself is on the grid."""
    c_center = np.asarray(c_center, dtype=np.float64)
    axis = np.linspace(-extent_m, extent_m, n_steps)
    gx, gy, gz = np.meshgrid(axis, axis, axis, indexing="ij")
    offsets = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
    return c_center[None, :] + offsets


def _solve_frame_at_fixed_c(
    lines: list[LineObservation],
    cx: float,
    cy: float,
    dist5: np.ndarray,
    C: np.ndarray,
    rvec_seed: np.ndarray,
    fx_seed: float,
) -> tuple[np.ndarray, float, float]:
    """LM-solve one frame's (rvec, fx) with C pinned. Returns
    ``(rvec, fx, line_rms)``."""

    def res(p: np.ndarray) -> np.ndarray:
        rvec = p[0:3]
        fx = float(np.clip(p[3], 50.0, 1e5))
        R, _ = cv2.Rodrigues(rvec)
        t = -R @ C
        K = _make_K(fx, cx, cy)
        return _line_residuals_distorted(lines, K, rvec, t, dist5)

    p0 = np.array([*np.asarray(rvec_seed, float).reshape(3), float(fx_seed)])
    lower = np.array([-np.pi, -np.pi, -np.pi, fx_seed * 0.5])
    upper = np.array([np.pi, np.pi, np.pi, fx_seed * 2.0])
    result = least_squares(
        res, p0, bounds=(lower, upper),
        method="trf", loss="huber", f_scale=2.0, max_nfev=300,
    )
    rvec = result.x[0:3]
    fx = float(result.x[3])
    R, _ = cv2.Rodrigues(rvec)
    t = -R @ C
    K = _make_K(fx, cx, cy)
    r = _line_residuals_distorted(lines, K, rvec, t, dist5)
    rms = float(np.sqrt((r ** 2).mean())) if r.size else float("nan")
    return rvec, fx, rms


def profile_camera_centre(
    per_frame_lines: dict[int, list[LineObservation]],
    image_size: tuple[int, int],
    *,
    c_grid: np.ndarray,
    lens_seed: tuple[float, float, float, float],
    per_frame_bootstrap: dict[int, tuple[np.ndarray, float]],
) -> CProfileResult:
    """Profile line-fitting RMS as a function of the static camera
    centre over ``c_grid``.

    ``lens_seed`` is ``(cx, cy, k1, k2)`` held fixed throughout the
    profile (the profile answers "where is C", not "what is the lens").
    ``per_frame_bootstrap`` provides the per-frame ``(rvec, fx)`` seeds
    for the inner solves.
    """
    cx, cy, k1, k2 = lens_seed
    dist5 = _dist5((k1, k2))
    fids = sorted(per_frame_lines.keys())

    m = len(c_grid)
    mean_rms = np.full(m, np.inf)
    p95_rms = np.full(m, np.inf)
    max_rms = np.full(m, np.inf)
    seeds_per_grid: list[dict[int, tuple[np.ndarray, float]]] = []

    # Warm-start each grid cell from the previous cell's solution so the
    # inner LMs converge fast; first cell uses the bootstrap.
    warm = {f: per_frame_bootstrap[f] for f in fids}
    for gi, C in enumerate(c_grid):
        rms_vals = []
        cell_seeds: dict[int, tuple[np.ndarray, float]] = {}
        for fid in fids:
            rvec_seed, fx_seed = warm[fid]
            rvec, fx, rms = _solve_frame_at_fixed_c(
                per_frame_lines[fid], cx, cy, dist5, np.asarray(C, float),
                rvec_seed, fx_seed,
            )
            cell_seeds[fid] = (rvec, fx)
            rms_vals.append(rms)
        arr = np.array(rms_vals, dtype=np.float64)
        finite = arr[np.isfinite(arr)]
        if finite.size:
            mean_rms[gi] = float(finite.mean())
            p95_rms[gi] = float(np.percentile(finite, 95))
            max_rms[gi] = float(finite.max())
        seeds_per_grid.append(cell_seeds)
        warm = cell_seeds  # warm-start the next cell

    best = int(np.argmin(mean_rms))
    return CProfileResult(
        grid_points=np.asarray(c_grid, dtype=np.float64),
        mean_rms=mean_rms,
        p95_rms=p95_rms,
        max_rms=max_rms,
        argmin_c=np.asarray(c_grid[best], dtype=np.float64).copy(),
        per_frame_seeds=seeds_per_grid[best],
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python3 -m pytest tests/test_static_c_profile.py -v`
Expected: PASS (all three tests)

- [ ] **Step 5: Commit**

```bash
git add src/utils/static_c_profile.py tests/test_static_c_profile.py
git commit -m "feat(camera): C-profile diagnostic for static-camera line solve"
```

---

## Task 3: `detect_lines_for_frames` — detect-all-frames helper

**Files:**
- Modify: `src/utils/line_camera_refine.py` (append a new function)
- Test: `tests/test_detect_lines_for_frames.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_detect_lines_for_frames.py`:

```python
"""Unit tests for the detect-all-frames orchestration helper."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.schemas.anchor import LineObservation
from src.utils.camera_projection import project_world_to_image
from src.utils.line_camera_refine import PITCH_LINE_CATALOGUE, detect_lines_for_frames

_LOOK = np.array([0.0, 64.0, -30.0])
_LOOK = _LOOK / np.linalg.norm(_LOOK)
_RIGHT = np.array([1.0, 0.0, 0.0])
_DOWN = np.cross(_LOOK, _RIGHT)
R_BASE = np.array([_RIGHT, _DOWN, _LOOK], dtype=float)
C_TRUE = np.array([52.5, -30.0, 30.0])
IMAGE_SIZE = (1280, 720)


def _camera(fx=900.0):
    w, h = IMAGE_SIZE
    K = np.array([[fx, 0, w / 2], [0, fx, h / 2], [0, 0, 1.0]])
    R = R_BASE
    t = -R @ C_TRUE
    return K, R, t


def _draw_pitch_lines_frame(K, R, t, line_names):
    """Render a green frame with the named catalogue lines painted as
    ~5 px white stripes — enough for the ridge detector to lock on."""
    w, h = IMAGE_SIZE
    img = np.full((h, w, 3), (60, 110, 60), dtype=np.uint8)
    for name in line_names:
        seg = np.array(PITCH_LINE_CATALOGUE[name], dtype=float)
        cam = seg @ R.T + t
        if (cam[:, 2] <= 0.1).any():
            continue
        proj = project_world_to_image(K, R, t, (0.0, 0.0), seg)
        a = tuple(int(round(v)) for v in proj[0])
        b = tuple(int(round(v)) for v in proj[1])
        cv2.line(img, a, b, (255, 255, 255), thickness=5)
    return img


@pytest.mark.unit
def test_detect_lines_for_frames_returns_only_well_covered_frames():
    K, R, t = _camera()
    line_names = [
        "left_18yd_front", "left_18yd_near_edge", "left_18yd_far_edge",
        "left_6yd_front", "near_touchline",
    ]
    good = _draw_pitch_lines_frame(K, R, t, line_names)
    blank = np.full((IMAGE_SIZE[1], IMAGE_SIZE[0], 3), (60, 110, 60), dtype=np.uint8)

    frames_bgr = {0: good, 1: blank}
    cameras = {
        0: {"K": K, "R": R, "t": t},
        1: {"K": K, "R": R, "t": t},
        # frame 2 has a camera but no frame image — must be skipped silently
        2: {"K": K, "R": R, "t": t},
    }
    out = detect_lines_for_frames(frames_bgr, cameras, (0.0, 0.0))

    # The blank frame yields no lines and is excluded; frame 2 has no image.
    assert 1 not in out
    assert 2 not in out
    # The well-drawn frame yields >= 2 LineObservations.
    assert 0 in out
    assert len(out[0]) >= 2
    assert all(isinstance(ln, LineObservation) for ln in out[0])
    assert all(ln.world_segment is not None for ln in out[0])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python3 -m pytest tests/test_detect_lines_for_frames.py -v`
Expected: FAIL — `ImportError: cannot import name 'detect_lines_for_frames'`

- [ ] **Step 3: Write minimal implementation**

Append to `src/utils/line_camera_refine.py` (after `refine_camera_from_lines`):

```python
def detect_lines_for_frames(
    frames_bgr: dict[int, np.ndarray],
    cameras: dict[int, dict[str, np.ndarray]],
    distortion: tuple[float, float],
    detector_cfg: DetectorConfig | None = None,
    *,
    min_confidence: float = 0.5,
    min_n_samples: int = 40,
    min_lines: int = 2,
) -> dict[int, list[LineObservation]]:
    """Detect painted pitch lines across many frames using per-frame
    bootstrap cameras.

    ``frames_bgr`` and ``cameras`` are both keyed by frame id. A frame
    is included in the output only if it has a bootstrap camera, a
    decoded image, and at least ``min_lines`` detections passing the
    confidence / sample-count gates. Frames that fail any check are
    silently dropped — callers keep their propagated camera for those.
    """
    if detector_cfg is None:
        detector_cfg = DetectorConfig()
    out: dict[int, list[LineObservation]] = {}
    for fid, frame in frames_bgr.items():
        cam = cameras.get(fid)
        if cam is None:
            continue
        dets = detect_painted_lines_in_frame(
            frame, cam["K"], cam["R"], cam["t"], distortion,
            PITCH_LINE_CATALOGUE, detector_cfg,
        )
        usable = [
            d for d in dets
            if d.confidence >= min_confidence and d.n_samples >= min_n_samples
        ]
        if len(usable) >= min_lines:
            out[fid] = [
                LineObservation(
                    name=d.name,
                    image_segment=d.image_segment,
                    world_segment=d.world_segment,
                )
                for d in usable
            ]
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python3 -m pytest tests/test_detect_lines_for_frames.py -v`
Expected: PASS

Note: if the test fails because the drawn lines are too thin/thick for the ridge detector, adjust `thickness=5` in `_draw_pitch_lines_frame` to `4` and re-run — the detector expects a 2–6 px painted line. Do not change the detector.

- [ ] **Step 5: Commit**

```bash
git add src/utils/line_camera_refine.py tests/test_detect_lines_for_frames.py
git commit -m "feat(camera): detect-all-frames helper for static-camera line solve"
```

---

## Task 4: Camera stage — `_refine_with_static_line_solve`

**Files:**
- Modify: `src/stages/camera.py`
- Test: `tests/test_camera_stage_static_line.py`

This task wires the three new units into the camera stage. When
`camera.line_extraction` is true AND `camera.static_camera` is true, the
new method runs detect → C-profile → bundle adjustment → iterative
re-detection, then writes one shared camera centre into the track.

- [ ] **Step 1: Write the failing test**

Create `tests/test_camera_stage_static_line.py`:

```python
"""Integration test: the camera stage's static-camera line-solve path
produces a track with exactly one camera centre."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from src.schemas.anchor import Anchor, AnchorSet, LandmarkObservation
from src.schemas.camera_track import CameraTrack
from src.stages.camera import CameraStage
from src.utils.camera_projection import project_world_to_image
from src.utils.line_camera_refine import PITCH_LINE_CATALOGUE

_LOOK = np.array([0.0, 64.0, -30.0])
_LOOK = _LOOK / np.linalg.norm(_LOOK)
_RIGHT = np.array([1.0, 0.0, 0.0])
_DOWN = np.cross(_LOOK, _RIGHT)
R_BASE = np.array([_RIGHT, _DOWN, _LOOK], dtype=float)
C_TRUE = np.array([52.5, -30.0, 30.0])
IMAGE_SIZE = (1280, 720)
FPS = 30.0
_LINE_NAMES = [
    "left_18yd_front", "left_18yd_near_edge", "left_18yd_far_edge",
    "left_6yd_front", "near_touchline",
]
# Six landmarks (two non-coplanar) so the anchor solve is identifiable.
_LANDMARKS = [
    ("near_left_corner", np.array([0, 0, 0.0])),
    ("near_right_corner", np.array([105, 0, 0.0])),
    ("far_left_corner", np.array([0, 68, 0.0])),
    ("far_right_corner", np.array([105, 68, 0.0])),
    ("near_left_corner_flag_top", np.array([0, 0, 1.5])),
    ("left_goal_crossbar_left", np.array([0, 30.34, 2.44])),
]


def _yaw(angle_deg):
    a = np.deg2rad(angle_deg)
    Ry = np.array([[np.cos(a), -np.sin(a), 0.0],
                   [np.sin(a), np.cos(a), 0.0],
                   [0.0, 0.0, 1.0]])
    return R_BASE @ Ry.T


def _project(K, R, t, world):
    cam = R @ np.asarray(world, float) + t
    pix = K @ cam
    return float(pix[0] / pix[2]), float(pix[1] / pix[2])


def _frame(K, R, t):
    w, h = IMAGE_SIZE
    img = np.full((h, w, 3), (60, 110, 60), dtype=np.uint8)
    for name in _LINE_NAMES:
        seg = np.array(PITCH_LINE_CATALOGUE[name], dtype=float)
        cam = seg @ R.T + t
        if (cam[:, 2] <= 0.1).any():
            continue
        proj = project_world_to_image(K, R, t, (0.0, 0.0), seg)
        a = tuple(int(round(v)) for v in proj[0])
        b = tuple(int(round(v)) for v in proj[1])
        cv2.line(img, a, b, (255, 255, 255), thickness=5)
    return img


def _write_manifest(output_dir, shot_id, n_frames):
    from src.schemas.shots import Shot, ShotsManifest
    end = max(0, n_frames - 1)
    ShotsManifest(
        source_file="test", fps=FPS, total_frames=n_frames,
        shots=[Shot(id=shot_id, start_frame=0, end_frame=end,
                    start_time=0.0, end_time=(end + 1) / FPS,
                    clip_file=f"shots/{shot_id}.mp4")],
    ).save(output_dir / "shots" / "shots_manifest.json")


@pytest.mark.integration
def test_static_line_solve_track_has_single_camera_centre(tmp_path: Path) -> None:
    n_frames = 12
    fx = 900.0
    w, h = IMAGE_SIZE
    K = np.array([[fx, 0, w / 2], [0, fx, h / 2], [0, 0, 1.0]])
    yaws = np.linspace(-6.0, 6.0, n_frames)

    shots = tmp_path / "shots"
    shots.mkdir(parents=True)
    vw = cv2.VideoWriter(
        str(shots / "play.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), FPS, IMAGE_SIZE
    )
    Rs, ts = [], []
    for yaw in yaws:
        R = _yaw(float(yaw))
        t = -R @ C_TRUE
        Rs.append(R)
        ts.append(t)
        vw.write(_frame(K, R, t))
    vw.release()
    _write_manifest(tmp_path, "play", n_frames)

    # Anchors on the first, middle, last frame — point landmarks only.
    anchor_frames = [0, n_frames // 2, n_frames - 1]
    anchors = []
    for af in anchor_frames:
        lms = tuple(
            LandmarkObservation(
                name=name,
                image_xy=_project(K, Rs[af], ts[af], world),
                world_xyz=tuple(world),
            )
            for name, world in _LANDMARKS
        )
        anchors.append(Anchor(frame=af, landmarks=lms))
    AnchorSet(clip_id="play", image_size=IMAGE_SIZE,
              anchors=tuple(anchors)).save(tmp_path / "camera" / "play_anchors.json")

    stage = CameraStage(
        config={"camera": {
            "static_camera": True,
            "line_extraction": True,
            "lens_from_anchor": False,
        }},
        output_dir=tmp_path,
    )
    stage.run()

    track = CameraTrack.load(tmp_path / "camera" / "play_camera_track.json")
    # The static-C line solve must report exactly one camera centre.
    assert track.camera_centre is not None
    C = np.array(track.camera_centre)
    # Every per-frame (R, t) must satisfy -R.T @ t == that single C.
    for f in track.frames:
        if f.t is None:
            continue
        R = np.array(f.R)
        t = np.array(f.t)
        c_frame = -R.T @ t
        assert np.linalg.norm(c_frame - C) < 1e-3, (
            f"frame {f.frame}: camera centre {c_frame} != track C {C}"
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python3 -m pytest tests/test_camera_stage_static_line.py -v`
Expected: FAIL — the current `line_extraction` path runs `_refine_with_line_extraction` (independent per-frame solves), so `track.camera_centre` is the anchor-solve C and per-frame centres drift; the `-R.T @ t == C` assertion fails (or `camera_centre` is the wrong value).

- [ ] **Step 3: Add `_refine_with_static_line_solve` to `src/stages/camera.py`**

Add this method to the `CameraStage` class (place it directly after `_refine_with_line_extraction`):

```python
    def _refine_with_static_line_solve(
        self,
        cap: cv2.VideoCapture,
        shot_id: str,
        anchors: AnchorSet,
        cfg: dict,
        per_frame_K: list,
        per_frame_R: list,
        per_frame_t: list,
        per_frame_conf: list,
        is_anchor: list,
        distortion: tuple[float, float],
        detected_lines_by_frame: dict[int, list],
    ) -> np.ndarray | None:
        """Static-camera line solve: detect painted lines on every
        propagated frame, profile the camera centre, bundle-adjust one
        shared centre, then iteratively re-detect under the coherent
        cameras. Writes per-frame ``(K, R, t)`` back in place and returns
        the single locked camera centre (or ``None`` if it bailed and
        left the propagated cameras untouched).
        """
        from src.utils.anchor_solver import _is_rich
        from src.utils.line_detector import DetectorConfig
        from src.utils.line_camera_refine import detect_lines_for_frames
        from src.utils.static_c_profile import make_c_grid, profile_camera_centre
        from src.utils.static_line_solver import solve_static_camera_from_lines

        det_cfg = DetectorConfig(
            search_strip_px=int(cfg.get("line_extraction_strip_px", 25)),
            min_gradient=float(cfg.get("line_extraction_min_gradient", 10.0)),
        )
        lens_model = str(cfg.get("line_extraction_lens_model", "pinhole_k1k2"))
        n_rounds = int(cfg.get("line_extraction_static_rounds", 3))
        point_hint_weight = float(
            cfg.get("line_extraction_point_hint_weight", 0.05)
        )
        dist2 = (float(distortion[0]), float(distortion[1]))

        covered = [
            i for i in range(len(per_frame_K)) if per_frame_K[i] is not None
        ]
        frames_bgr: dict[int, np.ndarray] = {}
        for i in covered:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, frame = cap.read()
            if ok:
                frames_bgr[i] = frame

        def _cameras_from_arrays() -> dict[int, dict]:
            return {
                i: {"K": per_frame_K[i], "R": per_frame_R[i], "t": per_frame_t[i]}
                for i in frames_bgr
            }

        # Step 0 — detect under the propagated bootstrap cameras.
        per_frame_lines = detect_lines_for_frames(
            frames_bgr, _cameras_from_arrays(), dist2, det_cfg,
        )
        if len(per_frame_lines) < 2:
            logger.warning(
                "static line solve: only %d frame(s) yielded detected lines; "
                "keeping the propagated cameras unchanged",
                len(per_frame_lines),
            )
            return None

        # Per-frame (rvec, fx) bootstrap seeds from the propagated cameras.
        bootstrap: dict[int, tuple[np.ndarray, float]] = {}
        for fid in per_frame_lines:
            rv, _ = cv2.Rodrigues(per_frame_R[fid])
            bootstrap[fid] = (rv.reshape(3), float(per_frame_K[fid][0, 0]))

        # Seed C from the propagated centres (rich-anchor frames preferred).
        rich = {a.frame for a in anchors.anchors if _is_rich(a)}
        seed_cs = [
            -per_frame_R[f].T @ per_frame_t[f]
            for f in per_frame_lines if f in rich
        ] or [
            -per_frame_R[f].T @ per_frame_t[f] for f in per_frame_lines
        ]
        c_center = np.median(np.stack(seed_cs), axis=0)
        cx0 = float(per_frame_K[covered[0]][0, 2])
        cy0 = float(per_frame_K[covered[0]][1, 2])
        lens_seed = (cx0, cy0, dist2[0], dist2[1])

        # Step 1 — C-profile: coarse grid then a fine grid around its argmin.
        coarse = profile_camera_centre(
            per_frame_lines, anchors.image_size,
            c_grid=make_c_grid(c_center, extent_m=7.5, n_steps=7),
            lens_seed=lens_seed, per_frame_bootstrap=bootstrap,
        )
        fine = profile_camera_centre(
            per_frame_lines, anchors.image_size,
            c_grid=make_c_grid(coarse.argmin_c, extent_m=2.0, n_steps=5),
            lens_seed=lens_seed, per_frame_bootstrap=coarse.per_frame_seeds,
        )
        logger.info(
            "static line solve: C-profile argmin=%s mean line RMS=%.3f px",
            np.round(fine.argmin_c, 3).tolist(),
            float(np.min(fine.mean_rms)),
        )

        # Steps 2 + 3 — bundle adjustment + iterative re-detection.
        anchor_landmarks = {
            a.frame: list(a.landmarks) for a in anchors.anchors if a.landmarks
        }
        c_seed = fine.argmin_c
        seeds = fine.per_frame_seeds
        sol = None
        for round_idx in range(max(1, n_rounds)):
            sol = solve_static_camera_from_lines(
                per_frame_lines, anchors.image_size,
                c_seed=c_seed, lens_seed=lens_seed,
                per_frame_seeds=seeds, point_hints=anchor_landmarks,
                lens_model=lens_model, point_hint_weight=point_hint_weight,
            )
            if round_idx < n_rounds - 1:
                cams = {
                    fid: {"K": K, "R": R, "t": t}
                    for fid, (K, R, t) in sol.per_frame_KRt.items()
                }
                redet = detect_lines_for_frames(
                    frames_bgr, cams, tuple(sol.distortion[:2]), det_cfg,
                )
                if len(redet) >= 2:
                    per_frame_lines = redet
                c_seed = sol.camera_centre
                seeds = {
                    fid: (cv2.Rodrigues(R)[0].reshape(3), float(K[0, 0]))
                    for fid, (K, R, _t) in sol.per_frame_KRt.items()
                }

        assert sol is not None
        C = sol.camera_centre

        # Write the solved cameras back in place.
        for fid, (K, R, t) in sol.per_frame_KRt.items():
            per_frame_K[fid] = K
            per_frame_R[fid] = R
            per_frame_t[fid] = t
            rms = sol.per_frame_line_rms.get(fid, float("nan"))
            if np.isfinite(rms):
                per_frame_conf[fid] = max(0.3, min(1.0, 1.0 - rms / 6.0))
            detected_lines_by_frame[fid] = [
                {
                    "name": ln.name,
                    "image_segment": [list(ln.image_segment[0]),
                                      list(ln.image_segment[1])],
                    "world_segment": [list(ln.world_segment[0]),
                                      list(ln.world_segment[1])],
                }
                for ln in per_frame_lines.get(fid, [])
            ]

        # One-C consistency: frames the solve skipped still share C.
        for i in covered:
            if i not in sol.per_frame_KRt and per_frame_R[i] is not None:
                per_frame_t[i] = -per_frame_R[i] @ C

        rms_arr = np.array(
            [v for v in sol.per_frame_line_rms.values() if np.isfinite(v)]
        )
        if rms_arr.size:
            logger.info(
                "static line solve: locked C=%s across %d frames — line RMS "
                "mean=%.3f median=%.3f max=%.3f frac<1px=%.2f",
                np.round(C, 3).tolist(), len(sol.per_frame_KRt),
                float(rms_arr.mean()), float(np.median(rms_arr)),
                float(rms_arr.max()), float((rms_arr < 1.0).mean()),
            )
        return C
```

- [ ] **Step 4: Branch the `line_extraction` path and write the line-derived `camera_centre`**

In `src/stages/camera.py`, find the `line_extraction` block in `_run_shot` (currently around lines 260–267):

```python
        detected_lines_by_frame: dict[int, list] = {}
        if bool(cfg.get("line_extraction", False)):
            self._refine_with_line_extraction(
                cap, shot_id, anchors, cfg,
                per_frame_K, per_frame_R, per_frame_t, per_frame_conf,
                is_anchor, tuple(sol.distortion),
                detected_lines_by_frame,
            )
```

Replace it with:

```python
        detected_lines_by_frame: dict[int, list] = {}
        static_line_centre: np.ndarray | None = None
        if bool(cfg.get("line_extraction", False)):
            if static_camera:
                static_line_centre = self._refine_with_static_line_solve(
                    cap, shot_id, anchors, cfg,
                    per_frame_K, per_frame_R, per_frame_t, per_frame_conf,
                    is_anchor, tuple(sol.distortion),
                    detected_lines_by_frame,
                )
            else:
                self._refine_with_line_extraction(
                    cap, shot_id, anchors, cfg,
                    per_frame_K, per_frame_R, per_frame_t, per_frame_conf,
                    is_anchor, tuple(sol.distortion),
                    detected_lines_by_frame,
                )
```

Then find the `CameraTrack(...)` construction in `_run_shot` (currently around lines 304–317) and change the `camera_centre` argument. The current code is:

```python
            camera_centre=(
                tuple(float(x) for x in sol.camera_centre)
                if sol.camera_centre is not None
                else None
            ),
```

Replace with:

```python
            camera_centre=(
                tuple(float(x) for x in static_line_centre)
                if static_line_centre is not None
                else (
                    tuple(float(x) for x in sol.camera_centre)
                    if sol.camera_centre is not None
                    else None
                )
            ),
```

- [ ] **Step 5: Include `K/R/t` in the `detected_lines.json` side-output**

In `_run_shot`, find the detected-lines debug JSON write (currently around lines 324–337). The current `"frames"` value is:

```python
                "frames": {
                    str(k): {"lines": v}
                    for k, v in sorted(detected_lines_by_frame.items())
                },
```

Replace with a version that also records the solved camera per frame, so the side-output is self-contained for the dashboard and for `scripts/global_solve_from_lines.py` / `scripts/profile_static_c.py`:

```python
                "frames": {
                    str(k): {
                        "lines": v,
                        "K": per_frame_K[k].tolist(),
                        "R": per_frame_R[k].tolist(),
                        "t": list(per_frame_t[k]),
                    }
                    for k, v in sorted(detected_lines_by_frame.items())
                    if per_frame_K[k] is not None
                },
```

- [ ] **Step 6: Run the test to verify it passes**

Run: `.venv/bin/python3 -m pytest tests/test_camera_stage_static_line.py -v`
Expected: PASS

If the inner LMs are slow on the 12-frame fixture, that is acceptable for an `@pytest.mark.integration` test; it should still finish in well under a minute.

- [ ] **Step 7: Run the existing camera-stage tests to confirm nothing broke**

Run: `.venv/bin/python3 -m pytest tests/test_camera_stage.py tests/test_anchor_solver.py -v`
Expected: PASS (same set as before — `test_camera_stage_recovers_anchor_frames_exactly` passes, `test_camera_stage_recovers_trajectory` stays skipped). The existing camera-stage tests use `static_camera=False` and no `line_extraction`, so they exercise the untouched path.

- [ ] **Step 8: Commit**

```bash
git add src/stages/camera.py tests/test_camera_stage_static_line.py
git commit -m "feat(camera): wire static-camera line solve into the camera stage"
```

---

## Task 5: Config key + `profile_static_c.py` script + rewire experiment scripts

**Files:**
- Modify: `config/default.yaml`
- Create: `scripts/profile_static_c.py`
- Modify: `scripts/global_solve_from_lines.py`
- Modify: `scripts/iterative_global_solve.py`

- [ ] **Step 1: Add the config key**

In `config/default.yaml`, under the `camera:` section, after the `line_extraction_max_iters` line, add:

```yaml
  # Lens model for the static-camera line solve (only used when
  # static_camera + line_extraction are both true). pinhole_k1k2 shares
  # (cx, cy, k1, k2); brown_conrady additionally frees tangential
  # (p1, p2) and k3. Start with pinhole_k1k2 — the C-profile diagnostic
  # (scripts/profile_static_c.py) tells you whether the richer model is
  # needed. See docs/superpowers/notes/2026-05-14-camera-1px-experiment.md.
  line_extraction_lens_model: pinhole_k1k2
```

- [ ] **Step 2: Create the `profile_static_c.py` CLI**

Create `scripts/profile_static_c.py`:

```python
"""Camera-centre profile diagnostic CLI.

Reads detected painted lines + a bootstrap camera track, sweeps a 3-D
grid of candidate static camera centres, and prints the line-fitting RMS
as a function of C. The decisive readout is the mean RMS at the argmin
centre: sub-pixel means a single static camera CAN fit the detected
lines under the current lens model; ~4 px means it cannot.

Usage:
  .venv/bin/python3 scripts/profile_static_c.py \\
      output/camera/<shot>_detected_lines.json \\
      output/camera/<shot>_camera_track.json \\
      [output/camera/<shot>_anchors.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from src.schemas.anchor import AnchorSet, LineObservation
from src.schemas.camera_track import CameraTrack
from src.utils.anchor_solver import _is_rich
from src.utils.static_c_profile import make_c_grid, profile_camera_centre


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("detected_lines", type=Path)
    parser.add_argument("bootstrap_camera", type=Path,
                        help="CameraTrack JSON providing per-frame (rvec, fx) seeds.")
    parser.add_argument("anchors", type=Path, nargs="?", default=None)
    parser.add_argument("--coarse-extent-m", type=float, default=7.5)
    parser.add_argument("--coarse-steps", type=int, default=7)
    parser.add_argument("--fine-extent-m", type=float, default=2.0)
    parser.add_argument("--fine-steps", type=int, default=5)
    args = parser.parse_args()

    with open(args.detected_lines) as f:
        data = json.load(f)
    frames = data["frames"]
    image_size = tuple(data["image_size"])

    per_frame_lines: dict[int, list[LineObservation]] = {}
    for fid_str, body in frames.items():
        lines = [
            LineObservation(
                name=ln["name"],
                image_segment=tuple(map(tuple, ln["image_segment"])),
                world_segment=tuple(map(tuple, ln["world_segment"])),
            )
            for ln in body["lines"]
        ]
        if len(lines) >= 2:
            per_frame_lines[int(fid_str)] = lines

    track = CameraTrack.load(args.bootstrap_camera)
    boot: dict[int, tuple[np.ndarray, float]] = {}
    seed_cs: list[np.ndarray] = []
    rich: set[int] = set()
    if args.anchors is not None:
        aset = AnchorSet.load(args.anchors)
        rich = {a.frame for a in aset.anchors if _is_rich(a)}
    for cf in track.frames:
        if cf.frame not in per_frame_lines or cf.t is None:
            continue
        R = np.array(cf.R)
        t = np.array(cf.t)
        rv, _ = cv2.Rodrigues(R)
        boot[cf.frame] = (rv.reshape(3), float(np.array(cf.K)[0, 0]))
        seed_cs.append(-R.T @ t)

    # Keep only frames that have both lines and a bootstrap camera.
    per_frame_lines = {f: per_frame_lines[f] for f in boot}
    if len(per_frame_lines) < 2:
        print("not enough frames with both detected lines and a bootstrap camera")
        return 1

    rich_cs = [
        -np.array(cf.R).T @ np.array(cf.t)
        for cf in track.frames
        if cf.frame in rich and cf.frame in boot and cf.t is not None
    ]
    c_center = np.median(np.stack(rich_cs if rich_cs else seed_cs), axis=0)
    cx, cy = (track.principal_point
              if track.principal_point is not None
              else (image_size[0] / 2.0, image_size[1] / 2.0))
    k1, k2 = track.distortion
    lens_seed = (float(cx), float(cy), float(k1), float(k2))

    print(f"loaded {len(per_frame_lines)} frames; C seed = {np.round(c_center, 3)}")
    print(f"lens seed (cx, cy, k1, k2) = {np.round(lens_seed, 4)}")

    coarse = profile_camera_centre(
        per_frame_lines, image_size,
        c_grid=make_c_grid(c_center, extent_m=args.coarse_extent_m,
                           n_steps=args.coarse_steps),
        lens_seed=lens_seed, per_frame_bootstrap=boot,
    )
    print(f"coarse argmin C = {np.round(coarse.argmin_c, 3)}  "
          f"mean RMS = {coarse.mean_rms.min():.3f} px")

    fine = profile_camera_centre(
        per_frame_lines, image_size,
        c_grid=make_c_grid(coarse.argmin_c, extent_m=args.fine_extent_m,
                           n_steps=args.fine_steps),
        lens_seed=lens_seed, per_frame_bootstrap=coarse.per_frame_seeds,
    )
    best = int(np.argmin(fine.mean_rms))
    print(f"\nfine argmin C = {np.round(fine.argmin_c, 3)}")
    print(f"  mean line RMS = {fine.mean_rms[best]:.3f} px")
    print(f"  P95  line RMS = {fine.p95_rms[best]:.3f} px")
    print(f"  max  line RMS = {fine.max_rms[best]:.3f} px")
    print()
    if fine.mean_rms[best] < 1.0:
        print("VERDICT: sub-pixel static-C IS reachable under pinhole_k1k2 — "
              "keep line_extraction_lens_model: pinhole_k1k2.")
    else:
        print("VERDICT: pinhole_k1k2 floor is above 1 px — try "
              "line_extraction_lens_model: brown_conrady (Task 7).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 3: Verify the script imports cleanly**

Run: `.venv/bin/python3 -c "import scripts.profile_static_c"`
Expected: no output, exit 0 (no import errors).

- [ ] **Step 4: Rewire `scripts/global_solve_from_lines.py`**

Replace the body of `scripts/global_solve_from_lines.py` so it delegates to `solve_static_camera_from_lines` instead of its own inline LM. The script no longer supports `--max-motion-m` (the new solver has no per-frame motion budget — `C` is strictly one vector). New file content:

```python
"""Global static-camera solve from per-frame detected lines.

Reads ``output/camera/<shot>_detected_lines.json`` (per-frame painted-
line observations with per-frame bootstrap cameras) and fits one camera
body across every frame via :func:`solve_static_camera_from_lines`:

  Shared:     cx, cy, distortion, Cx, Cy, Cz   (one fixed camera centre)
  Per-frame:  rvec(3), fx

Reports per-frame line RMS and per-anchor-frame point-landmark deviance.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from src.schemas.anchor import AnchorSet, LineObservation
from src.schemas.camera_track import CameraFrame, CameraTrack
from src.utils.anchor_solver import _is_rich
from src.utils.camera_projection import project_world_to_image
from src.utils.static_line_solver import solve_static_camera_from_lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("detected_lines", type=Path)
    parser.add_argument("anchors", type=Path)
    parser.add_argument("--lens-model", default="pinhole_k1k2",
                        choices=["pinhole_k1k2", "brown_conrady"])
    parser.add_argument("--point-hint-weight", type=float, default=0.05)
    parser.add_argument("--output-camera-track", type=Path, default=None)
    args = parser.parse_args()

    with open(args.detected_lines) as f:
        data = json.load(f)
    W, H = data["image_size"]
    fps = data.get("fps", 30.0)
    shot_id = data["shot_id"]
    frames = data["frames"]

    aset = AnchorSet.load(args.anchors)
    anchor_landmarks = {
        a.frame: list(a.landmarks) for a in aset.anchors if a.landmarks
    }
    rich_frames = {a.frame for a in aset.anchors if _is_rich(a)}

    per_frame_lines: dict[int, list[LineObservation]] = {}
    per_frame_seeds: dict[int, tuple[np.ndarray, float]] = {}
    seed_cs: list[np.ndarray] = []
    for fid_str, body in frames.items():
        fid = int(fid_str)
        lines = [
            LineObservation(
                name=ln["name"],
                image_segment=tuple(map(tuple, ln["image_segment"])),
                world_segment=tuple(map(tuple, ln["world_segment"])),
            )
            for ln in body["lines"]
        ]
        if len(lines) < 2 or "K" not in body:
            continue
        K = np.array(body["K"])
        R = np.array(body["R"])
        t = np.array(body["t"])
        per_frame_lines[fid] = lines
        rv, _ = cv2.Rodrigues(R)
        per_frame_seeds[fid] = (rv.reshape(3), float(K[0, 0]))
        if fid in rich_frames:
            seed_cs.append(-R.T @ t)
    if not seed_cs:
        seed_cs = [
            -np.array(frames[str(f)]["R"]).T @ np.array(frames[str(f)]["t"])
            for f in per_frame_lines
        ]
    c_seed = np.median(np.stack(seed_cs), axis=0)
    print(f"loaded {len(per_frame_lines)} frames; C seed = {np.round(c_seed, 3)}")

    sol = solve_static_camera_from_lines(
        per_frame_lines, (int(W), int(H)),
        c_seed=c_seed, lens_seed=(W / 2.0, H / 2.0, 0.0, 0.0),
        per_frame_seeds=per_frame_seeds, point_hints=anchor_landmarks,
        lens_model=args.lens_model, point_hint_weight=args.point_hint_weight,
    )

    rms = np.array([v for v in sol.per_frame_line_rms.values() if np.isfinite(v)])
    print(f"\nrecovered C = {np.round(sol.camera_centre, 3)}")
    print(f"recovered lens: pp={np.round(sol.principal_point, 1)} "
          f"distortion={np.round(sol.distortion, 4)}")
    print(f"line RMS: mean={rms.mean():.3f} median={np.median(rms):.3f} "
          f"max={rms.max():.3f}")
    print(f"  frac <1px={(rms < 1.0).mean():.3f}  <2px={(rms < 2.0).mean():.3f}")

    print("\nPoint-landmark devs on rich anchor frames:")
    for fid in sorted(anchor_landmarks):
        if fid not in sol.per_frame_KRt or fid not in rich_frames:
            continue
        K, R, t = sol.per_frame_KRt[fid]
        pts = np.array([lm.world_xyz for lm in anchor_landmarks[fid]])
        obs = np.array([lm.image_xy for lm in anchor_landmarks[fid]])
        proj = project_world_to_image(
            K, R, t, (sol.distortion[0], sol.distortion[1]), pts
        )
        devs = np.linalg.norm(proj - obs, axis=1)
        line_rms = sol.per_frame_line_rms.get(fid, float("nan"))
        print(f"  f{fid:>4}: line RMS={line_rms:5.3f}  "
              f"pt mean={devs.mean():5.2f}  pt max={devs.max():5.2f}")

    if args.output_camera_track:
        frames_out = [
            CameraFrame(
                frame=fid, K=K.tolist(), R=R.tolist(),
                confidence=1.0, is_anchor=fid in rich_frames, t=list(t),
            )
            for fid, (K, R, t) in sorted(sol.per_frame_KRt.items())
        ]
        first = frames_out[0]
        CameraTrack(
            clip_id=shot_id, fps=fps, image_size=(int(W), int(H)),
            t_world=list(first.t), frames=tuple(frames_out),
            principal_point=sol.principal_point,
            camera_centre=tuple(float(x) for x in sol.camera_centre),
            distortion=(sol.distortion[0], sol.distortion[1]),
        ).save(args.output_camera_track)
        print(f"\nwrote camera track to {args.output_camera_track}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 5: Rewire `scripts/iterative_global_solve.py`**

Replace the inline `detect_all` and `joint_solve` helpers in `scripts/iterative_global_solve.py` with calls to the shared modules. Replace the whole file with:

```python
"""Iterative detect -> static-C solve -> re-detect loop.

Each outer iteration: detect painted lines under the current cameras,
solve one shared static camera centre, then re-detect under the
coherent cameras. Tests the hypothesis that per-frame detection bias
(from biased bootstrap cameras) closes once the cameras agree.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from src.schemas.anchor import AnchorSet
from src.utils.anchor_solver import _is_rich
from src.utils.camera_projection import project_world_to_image
from src.utils.line_camera_refine import detect_lines_for_frames
from src.utils.line_detector import DetectorConfig
from src.utils.static_line_solver import solve_static_camera_from_lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("clip", type=Path)
    parser.add_argument("bootstrap_camera", type=Path)
    parser.add_argument("anchors", type=Path)
    parser.add_argument("--strip-px", type=int, default=20)
    parser.add_argument("--point-hint-weight", type=float, default=0.05)
    parser.add_argument("--lens-model", default="pinhole_k1k2",
                        choices=["pinhole_k1k2", "brown_conrady"])
    parser.add_argument("--n-outer-iters", type=int, default=3)
    args = parser.parse_args()

    with open(args.bootstrap_camera) as f:
        track = json.load(f)
    distortion = tuple(track.get("distortion", [0.0, 0.0]))
    W, H = track["image_size"]
    cams_init = {fr["frame"]: fr for fr in track["frames"]}

    aset = AnchorSet.load(args.anchors)
    anchor_landmarks = {
        a.frame: list(a.landmarks) for a in aset.anchors if a.landmarks
    }
    rich_frames = {a.frame for a in aset.anchors if _is_rich(a)}

    cap = cv2.VideoCapture(str(args.clip))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_bgr: dict[int, np.ndarray] = {}
    for fi in range(n_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        if ok and fi in cams_init:
            frames_bgr[fi] = frame
    cap.release()
    print(f"loaded {len(frames_bgr)} frames")

    cameras = {
        fi: {
            "K": np.array(cams_init[fi]["K"]),
            "R": np.array(cams_init[fi]["R"]),
            "t": np.array(cams_init[fi]["t"]),
        }
        for fi in frames_bgr
    }
    det_cfg = DetectorConfig(search_strip_px=args.strip_px)
    dist2 = (float(distortion[0]), float(distortion[1]))

    for outer in range(args.n_outer_iters):
        print(f"\n=== outer iter {outer + 1} ===")
        per_frame_lines = detect_lines_for_frames(
            frames_bgr, cameras, dist2, det_cfg,
        )
        print(f"  detected lines on {len(per_frame_lines)} frames")
        if len(per_frame_lines) < 2:
            print("  too few frames with lines; stopping")
            break

        seeds = {}
        seed_cs = []
        for fid in per_frame_lines:
            rv, _ = cv2.Rodrigues(cameras[fid]["R"])
            seeds[fid] = (rv.reshape(3), float(cameras[fid]["K"][0, 0]))
            if fid in rich_frames:
                seed_cs.append(
                    -cameras[fid]["R"].T @ cameras[fid]["t"]
                )
        if not seed_cs:
            seed_cs = [
                -cameras[f]["R"].T @ cameras[f]["t"] for f in per_frame_lines
            ]
        c_seed = np.median(np.stack(seed_cs), axis=0)

        sol = solve_static_camera_from_lines(
            per_frame_lines, (int(W), int(H)),
            c_seed=c_seed, lens_seed=(W / 2.0, H / 2.0, dist2[0], dist2[1]),
            per_frame_seeds=seeds, point_hints=anchor_landmarks,
            lens_model=args.lens_model, point_hint_weight=args.point_hint_weight,
        )
        cameras = {
            fid: {"K": K, "R": R, "t": t}
            for fid, (K, R, t) in sol.per_frame_KRt.items()
        }
        dist2 = (sol.distortion[0], sol.distortion[1])

        rms = np.array(
            [v for v in sol.per_frame_line_rms.values() if np.isfinite(v)]
        )
        print(f"  C = {np.round(sol.camera_centre, 3)}")
        print(f"  line RMS: mean={rms.mean():.3f} median={np.median(rms):.3f} "
              f"max={rms.max():.3f}  frac<1px={(rms < 1.0).mean():.3f}")
        for fid in sorted(anchor_landmarks):
            if fid not in cameras or fid not in rich_frames:
                continue
            K, R, t = cameras[fid]["K"], cameras[fid]["R"], cameras[fid]["t"]
            pts = np.array([lm.world_xyz for lm in anchor_landmarks[fid]])
            obs = np.array([lm.image_xy for lm in anchor_landmarks[fid]])
            proj = project_world_to_image(K, R, t, dist2, pts)
            devs = np.linalg.norm(proj - obs, axis=1)
            print(f"    f{fid:>4}: pt mean={devs.mean():5.2f} "
                  f"max={devs.max():5.2f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 6: Verify both scripts import cleanly**

Run: `.venv/bin/python3 -c "import scripts.global_solve_from_lines, scripts.iterative_global_solve, scripts.profile_static_c"`
Expected: exit 0, no import errors.

- [ ] **Step 7: Run the full new test suite + the camera/anchor tests**

Run: `.venv/bin/python3 -m pytest tests/test_static_line_solver.py tests/test_static_c_profile.py tests/test_detect_lines_for_frames.py tests/test_camera_stage_static_line.py tests/test_camera_stage.py tests/test_anchor_solver.py -v`
Expected: PASS (all, with the one pre-existing `test_camera_stage_recovers_trajectory` still skipped).

- [ ] **Step 8: Commit**

```bash
git add config/default.yaml scripts/profile_static_c.py scripts/global_solve_from_lines.py scripts/iterative_global_solve.py
git commit -m "feat(camera): config key + rewire experiment scripts onto the static-C solver"
```

---

## Task 6: CHECKPOINT — run the C-profile diagnostic on gberch

This is a decision gate, not a code change. It determines whether
Task 7 (Brown-Conrady) is needed.

- [ ] **Step 1: Run the profile diagnostic on the gberch clip**

Run:
```bash
.venv/bin/python3 scripts/profile_static_c.py \
    output/camera/gberch_detected_lines.json \
    output/camera/gberch_camera_track.json \
    output/camera/gberch_anchors.json
```

The `gberch_detected_lines.json` and `gberch_camera_track.json` on disk
were produced by earlier experiment runs; the detected lines there are
fine as profile *input* even though they predate this work (the script
only reads the `lines` arrays and uses the camera track for bootstrap
`(rvec, fx)` seeds).

Note: if `gberch_detected_lines.json` frames lack a `K`/`R`/`t` block,
that is expected for the *pre-existing* file — `profile_static_c.py`
reads the bootstrap cameras from `gberch_camera_track.json`, not from
the detected-lines file, so it still works.

- [ ] **Step 2: Read the verdict and decide**

The script prints a `VERDICT:` line.

- **If mean line RMS at the fine argmin is `< 1.0 px`** → `pinhole_k1k2`
  is sufficient. The config default stays `pinhole_k1k2`. **Skip Task 7.**
  Proceed to Task 8.
- **If mean line RMS is `>= 1.0 px`** → do **Task 7** (add the
  Brown-Conrady regression test and flip the config default), then
  Task 8.

Record the printed numbers (argmin C, mean / P95 / max RMS, verdict) —
they go into the Phase 4 note in Task 8.

- [ ] **Step 3: No commit** (diagnostic only — nothing changed on disk).

---

## Task 7: (CONDITIONAL) Brown-Conrady regression test + config default

**Do this task only if Task 6's verdict was `>= 1.0 px`.** The solver
already supports `lens_model="brown_conrady"` (built in Task 1); this
task adds a regression test proving it recovers tangential distortion
and flips the config default.

**Files:**
- Modify: `tests/test_static_line_solver.py`
- Modify: `config/default.yaml`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_static_line_solver.py`:

```python
@pytest.mark.unit
def test_brown_conrady_recovers_tangential_distortion():
    """With p1/p2 baked into the synthetic projection, brown_conrady
    must recover a tighter fit than pinhole_k1k2."""
    yaws = [-6.0, -3.0, 0.0, 3.0, 6.0]
    fxs = [1500.0] * 5
    dist_true = np.array([0.08, -0.04, 0.012, -0.009, 0.0])  # k1,k2,p1,p2,k3

    per_frame_lines: dict[int, list[LineObservation]] = {}
    per_frame_seeds: dict[int, tuple[np.ndarray, float]] = {}
    for i, (yaw, fx) in enumerate(zip(yaws, fxs)):
        R = _yaw(yaw)
        t = -R @ C_TRUE
        K = np.array([[fx, 0, CX], [0, fx, CY], [0, 0, 1.0]])
        rvec, _ = cv2.Rodrigues(R)
        lines = []
        for name in _LINE_NAMES:
            seg = np.array(LINE_CATALOGUE[name], dtype=float)
            cam = seg @ R.T + t
            if not (cam[:, 2] > 0.1).all():
                continue
            pa = seg[0] + 0.2 * (seg[1] - seg[0])
            pb = seg[0] + 0.8 * (seg[1] - seg[0])
            world = np.array([pa, pb])
            proj, _ = cv2.projectPoints(
                world.reshape(-1, 1, 3), rvec, t.reshape(3, 1), K, dist_true
            )
            proj = proj.reshape(-1, 2)
            lines.append(LineObservation(
                name=name,
                image_segment=(tuple(proj[0]), tuple(proj[1])),
                world_segment=tuple(map(tuple, LINE_CATALOGUE[name])),
            ))
        per_frame_lines[i] = lines
        per_frame_seeds[i] = (rvec.reshape(3), fx)

    common = dict(
        c_seed=C_TRUE + 0.5, lens_seed=(CX, CY, 0.0, 0.0),
        per_frame_seeds=per_frame_seeds,
    )
    sol_pinhole = solve_static_camera_from_lines(
        per_frame_lines, IMAGE_SIZE, lens_model="pinhole_k1k2", **common
    )
    sol_brown = solve_static_camera_from_lines(
        per_frame_lines, IMAGE_SIZE, lens_model="brown_conrady", **common
    )
    # brown_conrady can model the tangential terms; pinhole cannot.
    assert sol_brown.line_rms_mean < sol_pinhole.line_rms_mean
    assert sol_brown.line_rms_mean < 0.1
    assert len(sol_brown.distortion) == 5
```

- [ ] **Step 2: Run test to verify it passes**

Run: `.venv/bin/python3 -m pytest tests/test_static_line_solver.py::test_brown_conrady_recovers_tangential_distortion -v`
Expected: PASS (the solver code already supports `brown_conrady` — this is a regression test, not new behaviour).

If it fails because pinhole already fits well enough that `brown_conrady` is not strictly better, increase the tangential magnitudes in `dist_true` (e.g. `p1=0.02, p2=-0.015`) and re-run.

- [ ] **Step 3: Flip the config default**

In `config/default.yaml`, change:
```yaml
  line_extraction_lens_model: pinhole_k1k2
```
to:
```yaml
  line_extraction_lens_model: brown_conrady
```

- [ ] **Step 4: Commit**

```bash
git add tests/test_static_line_solver.py config/default.yaml
git commit -m "feat(camera): brown_conrady lens model for static-C line solve"
```

---

## Task 8: Validation on gberch + Phase 4 note

**Files:**
- Modify: `docs/superpowers/notes/2026-05-14-camera-1px-experiment.md`

- [ ] **Step 1: Run the rewired global solve on gberch**

Run:
```bash
.venv/bin/python3 scripts/global_solve_from_lines.py \
    output/camera/gberch_detected_lines.json \
    output/camera/gberch_anchors.json \
    --lens-model <pinhole_k1k2 OR brown_conrady, per Task 6 verdict> \
    --output-camera-track output/camera/gberch_static_line_track.json
```

If this fails with `KeyError: 'K'`, the pre-existing
`gberch_detected_lines.json` has no per-frame camera blocks. In that
case, regenerate it by running the camera stage's static-line path
first (Step 2), which writes a self-contained `detected_lines.json`,
then re-run this command. Record the printed line-RMS distribution,
recovered C, and per-anchor point deviances.

- [ ] **Step 2: Run the full camera stage static-line path on gberch**

Confirm the output directory has the gberch shot + anchors, then:
```bash
.venv/bin/python3 recon.py run --input <gberch clip> --output ./output/ --from-stage camera --stages camera
```
(Use the actual gberch clip path and output dir for this project. If the
exact invocation differs, the goal is simply: run the `camera` stage
with `line_extraction: true` and `static_camera: true` on the gberch
shot.)

Then verify the produced track:
```bash
.venv/bin/python3 -c "
from src.schemas.camera_track import CameraTrack
import numpy as np
tr = CameraTrack.load('output/camera/gberch_camera_track.json')
C = np.array(tr.camera_centre)
print('camera_centre:', np.round(C, 4))
dev = []
for f in tr.frames:
    if f.t is None: continue
    c = -np.array(f.R).T @ np.array(f.t)
    dev.append(float(np.linalg.norm(c - C)))
print('max per-frame |C_frame - C|:', max(dev) if dev else 'n/a')
print('frames:', len(tr.frames))
"
```
Expected: `max per-frame |C_frame - C|` is ~0 (well under 1e-3 m) —
confirming body motion is zero by construction.

- [ ] **Step 3: Write the Phase 4 section in the experiment note**

Append a `# Phase 4 — Static-camera solve from detected lines` section
to `docs/superpowers/notes/2026-05-14-camera-1px-experiment.md`. Include,
using the real numbers gathered in Task 6 and Steps 1–2:

- The C-profile result: coarse + fine argmin C, and the mean / P95 / max
  line RMS at the argmin. State the verdict (was sub-pixel static-C
  reachable under `pinhole_k1k2`, or was `brown_conrady` needed?).
- The final `global_solve_from_lines.py` result on gberch: recovered C,
  line RMS distribution (mean / median / max / frac<1px), and the
  per-anchor point-landmark deviances.
- Confirmation that body motion is **0 by construction** (the
  `max per-frame |C_frame - C|` number from Step 2).
- What landed: the three new modules (`static_line_solver`,
  `static_c_profile`, `detect_lines_for_frames`), the camera-stage
  `_refine_with_static_line_solve` path, the `line_extraction_lens_model`
  config key, and the rewired experiment scripts.
- Honest floor statement: whatever line RMS the static-C solve reached
  is the reported floor — static-C was the hard gate and was met; line
  RMS is documented, not gated on.
- If `brown_conrady` was still above 1 px: note that zoom-dependent
  distortion is the documented next rung, deferred to a follow-up.

- [ ] **Step 4: Run the whole test suite once more**

Run: `.venv/bin/python3 -m pytest tests/ -q`
Expected: the same pass/skip/xfail counts as before this work started,
plus the new tests passing. Investigate any *new* failure; a
pre-existing unrelated failure (e.g. the Blender FBX snapshot test
mentioned in the experiment note) is acceptable to leave as-is.

- [ ] **Step 5: Commit**

```bash
git add docs/superpowers/notes/2026-05-14-camera-1px-experiment.md
git commit -m "docs: Phase 4 — static-camera solve from detected lines"
```

---

## Self-Review

**1. Spec coverage:**

| Spec section | Task |
|---|---|
| `static_line_solver.py` — `solve_static_camera_from_lines`, `StaticCameraSolution`, no `dC` | Task 1 |
| Distortion-aware line residual in the new module (not `_line_residuals`) | Task 1 (`_line_residuals_distorted`) |
| Both lens models (`pinhole_k1k2`, `brown_conrady`) | Task 1 (built), Task 7 (regression test + default) |
| `static_c_profile.py` — `profile_camera_centre`, `CProfileResult`, coarse→fine grid | Task 2 + Task 4 (coarse→fine orchestration) |
| `scripts/profile_static_c.py` | Task 5 |
| `detect_lines_for_frames` in `line_camera_refine.py` | Task 3 |
| Camera-stage `_refine_with_static_line_solve`, branch on `static_camera` | Task 4 |
| One-C consistency for skipped frames (`t = -R @ C`) | Task 4 (Step 3) |
| Single `camera_centre` written into the track | Task 4 (Step 4) |
| `detected_lines.json` includes `K/R/t` | Task 4 (Step 5) |
| Iterative re-detection (Steps 0–3) | Task 4 (the `n_rounds` loop) |
| Step-1-as-decision-gate | Task 6 (checkpoint) |
| Lens-model escalation gated by the profile | Task 6 → Task 7 |
| One new config key `line_extraction_lens_model` | Task 5 (Step 1) |
| Refactor `global_solve_from_lines.py` / `iterative_global_solve.py` | Task 5 |
| Error handling: <2 lines → keep propagated; detection empty / LM raises → fall back | Task 4 (the `len(per_frame_lines) < 2` guard; `least_squares` with bounds does not raise on non-convergence) |
| Existing `test_anchor_solver.py` / `test_camera_stage.py` stay green | Task 4 (Step 7), Task 5 (Step 7), Task 8 (Step 4) |
| Validation on gberch + Phase 4 note | Task 8 |
| Deferred: zoom-dependent distortion | Noted in Task 8 (Step 3) |

No gaps.

**2. Placeholder scan:** Task 8 Step 1 leaves the `--lens-model` value and the gberch clip path to be filled from the Task 6 verdict / the actual project layout — these are genuinely runtime-determined (the checkpoint decides the model; the clip path is environment-specific), not plan placeholders. Every code step contains complete code.

**3. Type consistency:**
- `solve_static_camera_from_lines(per_frame_lines, image_size, *, c_seed, lens_seed, per_frame_seeds, point_hints=None, lens_model="pinhole_k1k2", point_hint_weight=0.05, max_nfev=4000)` — same signature used in Task 1 test, Task 4, Task 5 (both scripts).
- `StaticCameraSolution` fields `camera_centre`, `principal_point`, `distortion`, `per_frame_KRt`, `per_frame_line_rms`, `lens_model` + `line_rms_mean` property — consumed consistently in Tasks 4, 5, 7.
- `profile_camera_centre(per_frame_lines, image_size, *, c_grid, lens_seed, per_frame_bootstrap)` → `CProfileResult(grid_points, mean_rms, p95_rms, max_rms, argmin_c, per_frame_seeds)` — same in Task 2 test, Task 4, Task 5 script.
- `make_c_grid(c_center, *, extent_m, n_steps)` — same in Task 2 and Task 4 and Task 5.
- `detect_lines_for_frames(frames_bgr, cameras, distortion, detector_cfg=None, *, min_confidence=0.5, min_n_samples=40, min_lines=2)` — same in Task 3 test, Task 4, Task 5 (`iterative_global_solve.py`).
- `lens_seed` is consistently `(cx, cy, k1, k2)` (4 floats) everywhere.
- `_dist5` and `_line_residuals_distorted` are defined in `static_line_solver.py` (Task 1) and imported by `static_c_profile.py` (Task 2) — consistent.

No inconsistencies found.
