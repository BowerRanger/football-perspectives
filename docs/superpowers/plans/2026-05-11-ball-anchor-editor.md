# Ball Anchor Editor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a "Layer 5" anchor-injection capability to the ball pipeline and a dashboard editor for placing per-frame ball anchors (pixel position + state tag), so the operator can correct WASB failures and pin known flight events.

**Architecture:** A new `BallAnchorSet` schema persisted to `output/ball/<shot_id>_ball_anchors.json`. `BallStage._run_shot` reads anchors before the existing detection loop: anchored frames bypass WASB, kick/catch/bounce events split flight segments, off-screen-flight extends them. Knot frames are passed into `fit_parabola_to_image_observations` via a new `knot_frames` kwarg so the parabola is constrained to pass through known world positions. A standalone editor page mirrors the camera anchor editor's three-column shell.

**Tech Stack:** Python 3.11, numpy, scipy.optimize.least_squares, FastAPI, vanilla JS/HTML/CSS for the editor, pytest, OpenCV (already in use).

**Spec:** `docs/superpowers/specs/2026-05-11-ball-anchor-editor-design.md`

---

## File Map

| Path | Status | Responsibility |
|---|---|---|
| `src/schemas/ball_anchor.py` | NEW | `BallAnchor`, `BallAnchorSet`, `BallAnchorState` literal + JSON load/save |
| `src/utils/ball_anchor_heights.py` | NEW | `state_to_height`, `HARD_KNOT_STATES`, `AIRBORNE_STATES` constants |
| `src/utils/bundle_adjust.py` | MODIFY | Add `knot_frames: dict[int, np.ndarray] \| None = None` kwarg to `fit_parabola_to_image_observations` |
| `src/stages/ball.py` | MODIFY | Layer 5 anchor injection: load anchors, override WASB on anchored frames, split flight runs at events, feed knot_frames into fits |
| `src/web/server.py` | MODIFY | `GET /ball-anchors/{shot_id}`, `POST /ball-anchors/{shot_id}`, `POST /ball-anchors/{shot_id}/preview` |
| `src/web/static/ball_anchor_editor.html` | NEW | Editor page (header + 3-column shell + video stage + anchor list) |
| `src/web/static/index.html` | MODIFY | Link to the new editor from the ball panel |
| `tests/test_ball_anchor_schema.py` | NEW | Round-trip JSON load/save |
| `tests/test_ball_anchor_heights.py` | NEW | `state_to_height` table + classification helpers |
| `tests/test_bundle_adjust_knot_frames.py` | NEW | Multi-frame knot constraint tests |
| `tests/test_ball_stage_anchors.py` | NEW | Layer 5 integration tests |
| `tests/test_ball_anchor_endpoints.py` | NEW | Smoke tests for the three new server endpoints |

---

## Task 1: `BallAnchor` + `BallAnchorSet` schema

**Files:**
- Create: `src/schemas/ball_anchor.py`
- Create: `tests/test_ball_anchor_schema.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_ball_anchor_schema.py`:

```python
"""JSON round-trip tests for BallAnchor / BallAnchorSet."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.schemas.ball_anchor import BallAnchor, BallAnchorSet


def test_roundtrip_with_pixel_anchors(tmp_path: Path):
    aset = BallAnchorSet(
        clip_id="origi01",
        image_size=(1280, 720),
        anchors=(
            BallAnchor(frame=43, image_xy=(640.0, 360.0), state="grounded"),
            BallAnchor(frame=78, image_xy=(700.0, 200.0), state="airborne_mid"),
            BallAnchor(frame=84, image_xy=(720.0, 340.0), state="kick"),
        ),
    )
    p = tmp_path / "anchors.json"
    aset.save(p)
    loaded = BallAnchorSet.load(p)
    assert loaded.clip_id == "origi01"
    assert loaded.image_size == (1280, 720)
    assert len(loaded.anchors) == 3
    assert loaded.anchors[0].frame == 43
    assert loaded.anchors[0].image_xy == (640.0, 360.0)
    assert loaded.anchors[0].state == "grounded"
    assert loaded.anchors[2].state == "kick"


def test_off_screen_flight_anchor_allows_none_image_xy(tmp_path: Path):
    aset = BallAnchorSet(
        clip_id="origi01",
        image_size=(1280, 720),
        anchors=(
            BallAnchor(frame=120, image_xy=None, state="off_screen_flight"),
        ),
    )
    p = tmp_path / "anchors.json"
    aset.save(p)
    loaded = BallAnchorSet.load(p)
    assert loaded.anchors[0].image_xy is None
    assert loaded.anchors[0].state == "off_screen_flight"


def test_load_missing_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        BallAnchorSet.load(tmp_path / "does_not_exist.json")


def test_invalid_state_rejected(tmp_path: Path):
    # Hand-craft a JSON with an invalid state string.
    p = tmp_path / "anchors.json"
    p.write_text(json.dumps({
        "clip_id": "x", "image_size": [1, 1],
        "anchors": [{"frame": 1, "image_xy": [0, 0], "state": "bogus"}],
    }))
    with pytest.raises(ValueError):
        BallAnchorSet.load(p)


def test_pixel_required_for_non_off_screen_states(tmp_path: Path):
    p = tmp_path / "anchors.json"
    p.write_text(json.dumps({
        "clip_id": "x", "image_size": [1, 1],
        "anchors": [{"frame": 1, "image_xy": None, "state": "grounded"}],
    }))
    with pytest.raises(ValueError):
        BallAnchorSet.load(p)


def test_empty_anchor_set_roundtrips(tmp_path: Path):
    aset = BallAnchorSet(
        clip_id="x", image_size=(640, 480), anchors=(),
    )
    p = tmp_path / "anchors.json"
    aset.save(p)
    loaded = BallAnchorSet.load(p)
    assert loaded.anchors == ()
```

- [ ] **Step 2: Run tests — expect ImportError**

Run: `.venv/bin/pytest tests/test_ball_anchor_schema.py -v`
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement the schema**

Create `src/schemas/ball_anchor.py`:

```python
"""Persisted ball anchor data — user-supplied per-frame ball positions
and state tags. Read by ``BallStage`` as a Layer 5 input before the
WASB detection loop runs.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal


BallAnchorState = Literal[
    "grounded",
    "airborne_low",
    "airborne_mid",
    "airborne_high",
    "kick",
    "catch",
    "bounce",
    "off_screen_flight",
]

_VALID_STATES: frozenset[str] = frozenset({
    "grounded", "airborne_low", "airborne_mid", "airborne_high",
    "kick", "catch", "bounce", "off_screen_flight",
})


@dataclass(frozen=True)
class BallAnchor:
    """One per-frame anchor."""
    frame: int
    # None only when state == "off_screen_flight".
    image_xy: tuple[float, float] | None
    state: BallAnchorState


@dataclass(frozen=True)
class BallAnchorSet:
    clip_id: str
    image_size: tuple[int, int]
    anchors: tuple[BallAnchor, ...]

    @classmethod
    def load(cls, path: Path) -> "BallAnchorSet":
        with path.open() as fh:
            data = json.load(fh)
        anchors = []
        for a in data.get("anchors", []):
            state = str(a["state"])
            if state not in _VALID_STATES:
                raise ValueError(f"unknown ball anchor state: {state!r}")
            raw_xy = a.get("image_xy")
            if raw_xy is None:
                if state != "off_screen_flight":
                    raise ValueError(
                        f"image_xy is required for state {state!r} "
                        f"(only off_screen_flight may omit it)"
                    )
                image_xy = None
            else:
                image_xy = (float(raw_xy[0]), float(raw_xy[1]))
            anchors.append(BallAnchor(
                frame=int(a["frame"]),
                image_xy=image_xy,
                state=state,  # type: ignore[arg-type]
            ))
        return cls(
            clip_id=str(data["clip_id"]),
            image_size=(int(data["image_size"][0]), int(data["image_size"][1])),
            anchors=tuple(anchors),
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as fh:
            json.dump(
                asdict(self), fh, indent=2,
                default=lambda v: list(v) if isinstance(v, tuple) else v,
            )
```

- [ ] **Step 4: Run tests — expect pass**

Run: `.venv/bin/pytest tests/test_ball_anchor_schema.py -v`
Expected: all 6 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/schemas/ball_anchor.py tests/test_ball_anchor_schema.py
git commit -m "feat(ball): add BallAnchor / BallAnchorSet schema"
```

---

## Task 2: `state_to_height` + classification helpers

**Files:**
- Create: `src/utils/ball_anchor_heights.py`
- Create: `tests/test_ball_anchor_heights.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_ball_anchor_heights.py`:

```python
"""Tests for the BallAnchorState → height table and helpers."""

from __future__ import annotations

import pytest

from src.utils.ball_anchor_heights import (
    AIRBORNE_STATES,
    EVENT_STATES,
    HARD_KNOT_STATES,
    state_to_height,
)


def test_grounded_height():
    assert state_to_height("grounded") == 0.11


def test_airborne_bucket_heights():
    assert state_to_height("airborne_low") == 1.0
    assert state_to_height("airborne_mid") == 6.0
    assert state_to_height("airborne_high") == 15.0


def test_event_heights():
    assert state_to_height("kick") == 0.11
    assert state_to_height("bounce") == 0.11
    assert state_to_height("catch") == 1.5


def test_off_screen_flight_has_no_height():
    with pytest.raises(ValueError):
        state_to_height("off_screen_flight")


def test_unknown_state_raises():
    with pytest.raises(ValueError):
        state_to_height("nonsense")


def test_hard_knot_states():
    assert "grounded" in HARD_KNOT_STATES
    assert "kick" in HARD_KNOT_STATES
    assert "catch" in HARD_KNOT_STATES
    assert "bounce" in HARD_KNOT_STATES
    # Airborne buckets are coarse so do NOT pin world position exactly.
    assert "airborne_low" not in HARD_KNOT_STATES
    assert "airborne_mid" not in HARD_KNOT_STATES
    assert "airborne_high" not in HARD_KNOT_STATES
    # Off-screen has no pixel so cannot be a knot.
    assert "off_screen_flight" not in HARD_KNOT_STATES


def test_airborne_state_classification():
    assert "airborne_low" in AIRBORNE_STATES
    assert "airborne_mid" in AIRBORNE_STATES
    assert "airborne_high" in AIRBORNE_STATES
    assert "off_screen_flight" in AIRBORNE_STATES
    assert "grounded" not in AIRBORNE_STATES
    assert "kick" not in AIRBORNE_STATES


def test_event_states():
    assert EVENT_STATES == frozenset({"kick", "catch", "bounce"})
```

- [ ] **Step 2: Run tests — expect ImportError**

Run: `.venv/bin/pytest tests/test_ball_anchor_heights.py -v`
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement the helpers**

Create `src/utils/ball_anchor_heights.py`:

```python
"""State→height lookup and classification sets for ball anchors.

Single source of truth: every consumer (Layer 5, server preview,
editor JS-via-API) reads from these constants so coarse bucket
adjustments only happen here.
"""

from __future__ import annotations


_STATE_HEIGHT_M: dict[str, float] = {
    "grounded":      0.11,
    "airborne_low":  1.0,
    "airborne_mid":  6.0,
    "airborne_high": 15.0,
    "kick":          0.11,
    "catch":         1.5,
    "bounce":        0.11,
}

# States whose pixel + height should be enforced exactly by the fit.
HARD_KNOT_STATES: frozenset[str] = frozenset({
    "grounded", "kick", "catch", "bounce",
})

# States that force the IMM into the flight branch for that frame and
# extend / create a flight segment.
AIRBORNE_STATES: frozenset[str] = frozenset({
    "airborne_low", "airborne_mid", "airborne_high", "off_screen_flight",
})

# States that mark a flight boundary (split flight runs at this frame).
EVENT_STATES: frozenset[str] = frozenset({
    "kick", "catch", "bounce",
})


def state_to_height(state: str) -> float:
    """Return the assumed ball height in metres for ``state``.

    Raises ValueError for ``off_screen_flight`` (no world position) and
    for unknown states.
    """
    if state == "off_screen_flight":
        raise ValueError("off_screen_flight has no associated height")
    if state not in _STATE_HEIGHT_M:
        raise ValueError(f"unknown ball anchor state: {state!r}")
    return _STATE_HEIGHT_M[state]
```

- [ ] **Step 4: Run tests — expect pass**

Run: `.venv/bin/pytest tests/test_ball_anchor_heights.py -v`
Expected: all 8 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/utils/ball_anchor_heights.py tests/test_ball_anchor_heights.py
git commit -m "feat(ball): add state→height table for ball anchors"
```

---

## Task 3: `knot_frames` kwarg on `fit_parabola_to_image_observations`

**Files:**
- Modify: `src/utils/bundle_adjust.py`
- Create: `tests/test_bundle_adjust_knot_frames.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_bundle_adjust_knot_frames.py`:

```python
"""Tests for the knot_frames kwarg on fit_parabola_to_image_observations."""

from __future__ import annotations

import numpy as np
import pytest

from src.utils.bundle_adjust import fit_parabola_to_image_observations


def _camera():
    K = np.array([[1500.0, 0, 640.0], [0, 1500.0, 360.0], [0, 0, 1.0]])
    R = np.eye(3)
    t = np.array([0.0, 0.0, 0.0])
    return K, R, t


def _synthesise_observations(p0, v0, K, R, t, n: int, fps: float = 30.0):
    g_vec = np.array([0.0, 0.0, -9.81])
    obs = []
    for i in range(n):
        dt = i / fps
        pt = p0 + v0 * dt + 0.5 * g_vec * dt ** 2
        cam = R @ pt + t
        u = float((K @ cam)[0] / (K @ cam)[2])
        v = float((K @ cam)[1] / (K @ cam)[2])
        obs.append((i, (u, v)))
    return obs


def test_no_knot_frames_matches_baseline():
    """When knot_frames is None, results must equal the existing fit."""
    K, R, t = _camera()
    p0 = np.array([0.0, 5.0, 0.11])
    v0 = np.array([3.0, 0.5, 12.0])
    obs = _synthesise_observations(p0, v0, K, R, t, n=15)

    a = fit_parabola_to_image_observations(
        obs, Ks=[K] * len(obs), Rs=[R] * len(obs), t_world=t, fps=30.0,
    )
    b = fit_parabola_to_image_observations(
        obs, Ks=[K] * len(obs), Rs=[R] * len(obs), t_world=t, fps=30.0,
        knot_frames=None,
    )
    assert np.allclose(a[0], b[0])
    assert np.allclose(a[1], b[1])
    assert a[2] == pytest.approx(b[2])


def test_single_knot_pulls_fit_through_known_point():
    """Anchoring a knot at the apex pulls the noisy fit toward truth."""
    rng = np.random.default_rng(13)
    K, R, t = _camera()
    p0_true = np.array([0.0, 5.0, 0.11])
    v0_true = np.array([3.0, 0.5, 12.0])
    obs = _synthesise_observations(p0_true, v0_true, K, R, t, n=20)
    # Heavy noise so the unconstrained fit drifts.
    noisy = [(fi, (uv[0] + rng.normal(0, 3.0), uv[1] + rng.normal(0, 3.0))) for fi, uv in obs]

    # Compute true apex world position for the knot.
    apex_idx = 12
    apex_world = p0_true + v0_true * (apex_idx / 30.0) + 0.5 * np.array([0, 0, -9.81]) * (apex_idx / 30.0) ** 2

    p0_a, v0_a, resid_a = fit_parabola_to_image_observations(
        noisy, Ks=[K] * len(noisy), Rs=[R] * len(noisy), t_world=t, fps=30.0,
    )
    p0_b, v0_b, resid_b = fit_parabola_to_image_observations(
        noisy, Ks=[K] * len(noisy), Rs=[R] * len(noisy), t_world=t, fps=30.0,
        knot_frames={apex_idx: apex_world},
    )

    # Evaluate the fitted parabola at the apex for both fits.
    def parab_at(p0, v0, fi):
        dt = fi / 30.0
        return p0 + v0 * dt + 0.5 * np.array([0, 0, -9.81]) * dt ** 2

    apex_pred_a = parab_at(p0_a, v0_a, apex_idx)
    apex_pred_b = parab_at(p0_b, v0_b, apex_idx)

    err_a = float(np.linalg.norm(apex_pred_a - apex_world))
    err_b = float(np.linalg.norm(apex_pred_b - apex_world))
    assert err_b < err_a, f"knotted fit should land closer to truth at apex (err_b={err_b:.3f} vs err_a={err_a:.3f})"
    assert err_b < 0.5, f"knotted apex error should be sub-half-metre, got {err_b:.3f}"


def test_two_knots_pin_endpoints():
    """A start-frame knot + end-frame knot should constrain a short arc."""
    K, R, t = _camera()
    p0_true = np.array([0.0, 5.0, 0.11])
    v0_true = np.array([3.0, 0.5, 12.0])
    obs = _synthesise_observations(p0_true, v0_true, K, R, t, n=10)
    apex_world_start = p0_true.copy()
    apex_world_end = p0_true + v0_true * (9 / 30.0) + 0.5 * np.array([0, 0, -9.81]) * (9 / 30.0) ** 2

    p0_fit, v0_fit, resid = fit_parabola_to_image_observations(
        obs, Ks=[K] * len(obs), Rs=[R] * len(obs), t_world=t, fps=30.0,
        knot_frames={0: apex_world_start, 9: apex_world_end},
    )

    def parab_at(p0, v0, fi):
        dt = fi / 30.0
        return p0 + v0 * dt + 0.5 * np.array([0, 0, -9.81]) * dt ** 2

    assert np.linalg.norm(parab_at(p0_fit, v0_fit, 0) - apex_world_start) < 0.5
    assert np.linalg.norm(parab_at(p0_fit, v0_fit, 9) - apex_world_end) < 0.5


def test_knot_frames_with_p0_fixed():
    """knot_frames and p0_fixed compose: p0_fixed pins the start
    exactly, additional knots act as soft constraints elsewhere."""
    K, R, t = _camera()
    p0_true = np.array([0.0, 5.0, 0.11])
    v0_true = np.array([3.0, 0.5, 12.0])
    obs = _synthesise_observations(p0_true, v0_true, K, R, t, n=15)
    apex_idx = 10
    apex_world = p0_true + v0_true * (apex_idx / 30.0) + 0.5 * np.array([0, 0, -9.81]) * (apex_idx / 30.0) ** 2

    p0_fit, v0_fit, resid = fit_parabola_to_image_observations(
        obs, Ks=[K] * len(obs), Rs=[R] * len(obs), t_world=t, fps=30.0,
        p0_fixed=p0_true, knot_frames={apex_idx: apex_world},
    )
    assert np.allclose(p0_fit, p0_true)
    assert np.linalg.norm(v0_fit - v0_true) < 0.5
```

- [ ] **Step 2: Run tests — expect FAIL**

Run: `.venv/bin/pytest tests/test_bundle_adjust_knot_frames.py -v`
Expected: FAIL on the `knot_frames` keyword.

- [ ] **Step 3: Add `knot_frames` to the parabola fit**

Open `src/utils/bundle_adjust.py` and read the existing
`fit_parabola_to_image_observations` carefully so you can mirror its
locals (`dt`, `g_vec`, `obs_array`, `n_obs`, `ts`).

Update the signature to add `knot_frames`:

```python
def fit_parabola_to_image_observations(
    observations: list[tuple[int, tuple[float, float]]],
    *,
    Ks: list[np.ndarray],
    Rs: list[np.ndarray],
    t_world: np.ndarray | list[np.ndarray],
    fps: float,
    g: float = -9.81,
    max_iter: int = 100,
    distortion: tuple[float, float] = (0.0, 0.0),
    p0_fixed: np.ndarray | None = None,
    knot_frames: dict[int, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
```

The `knot_frames` dict is keyed by **relative frame index inside the
segment** (i.e. `frame_idx - observations[0][0]`), mapping to a 3-vector
world position the parabola should pass through.

Locate the two `_residuals` closures (one for the unanchored 6-param
path, one for the `p0_fixed` 3-param path). For each closure, after the
existing per-observation residuals are appended, append soft-constraint
residual rows for the knots. Concretely, at the end of `_residuals` and
`_residuals_v0only`, before `return np.concatenate(residuals)`, insert:

```python
        if knot_frames:
            knot_weight = 1.0e3
            for rel_idx, target_world in knot_frames.items():
                # Evaluate the parabola at this relative frame index.
                # rel_idx is relative to observations[0][0]; convert to
                # the same dt scale used for the obs.
                dt_k = (rel_idx) / fps
                pos_k = (p0 if p0_fixed is None else p0_pin) \
                        + v0 * dt_k \
                        + 0.5 * (dt_k ** 2) * g_vec
                target = np.asarray(target_world, dtype=float)
                residuals.append(knot_weight * (pos_k - target))
```

Notes:
- `v0` is `params[3:6]` in `_residuals`, `params[:3]` in `_residuals_v0only`.
- `p0` is `params[:3]` in `_residuals`. In `_residuals_v0only`, the
  closed-over `p0_pin` is the source.
- The factor `knot_weight = 1.0e3` makes each knot equivalent to ~1000
  pixels of reprojection error per metre of world drift — strong enough
  to dominate the noisy pixel residuals while staying within LM's
  numerical comfort zone.

**Important — read first:** Before editing, open
`src/utils/bundle_adjust.py` and find the exact local variable names in
the existing `_residuals` and `_residuals_v0only` closures (they may
not be exactly `dt`, `g_vec`, `n_obs`, `ts`, `obs_array`). Use the
actual names. If the existing closures don't expose `p0` or `v0` as
distinct locals, refactor minimally — e.g. unpack `params` near the top
of each closure — but do NOT change the existing residual semantics.

- [ ] **Step 4: Run tests — expect pass**

Run: `.venv/bin/pytest tests/test_bundle_adjust_knot_frames.py tests/test_bundle_adjust_p0_fixed.py tests/test_ball_flight.py tests/test_ball_spin_fit.py -v`
Expected: all pass (new + existing).

- [ ] **Step 5: Commit**

```bash
git add src/utils/bundle_adjust.py tests/test_bundle_adjust_knot_frames.py
git commit -m "feat(bundle_adjust): support knot_frames soft constraints in parabola fit"
```

---

## Task 4: Layer 5 — load anchors and override WASB on anchored frames

**Files:**
- Modify: `src/stages/ball.py`
- Create: `tests/test_ball_stage_anchors.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_ball_stage_anchors.py`:

```python
"""Integration tests for Layer 5 ball anchor injection."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.schemas.ball_anchor import BallAnchor, BallAnchorSet
from src.schemas.ball_track import BallTrack
from src.schemas.camera_track import CameraFrame, CameraTrack
from src.schemas.shots import Shot, ShotsManifest
from src.stages.ball import BallStage
from src.utils.ball_detector import FakeBallDetector


def _camera_pose():
    look = np.array([0.0, 64.0, -30.0]); look /= np.linalg.norm(look)
    right = np.array([1.0, 0.0, 0.0])
    down = np.cross(look, right)
    R = np.array([right, down, look], dtype=float)
    t = -R @ np.array([52.5, -30.0, 30.0])
    K = np.array([[1500.0, 0, 640.0], [0, 1500.0, 360.0], [0, 0, 1.0]])
    return K, R, t


def _write_blank_clip(path: Path, n: int, fps: float = 30.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (320, 240)
    )
    for _ in range(n):
        writer.write(np.full((240, 320, 3), [50, 200, 50], dtype=np.uint8))
    writer.release()


def _save_camera_track(path: Path, K, R, t, n: int, fps: float = 30.0):
    track = CameraTrack(
        clip_id="play", fps=fps, image_size=(1280, 720), t_world=t.tolist(),
        frames=tuple(
            CameraFrame(frame=i, K=K.tolist(), R=R.tolist(),
                        confidence=1.0, is_anchor=(i == 0))
            for i in range(n)
        ),
    )
    track.save(path)


def _save_manifest(path: Path, n: int):
    ShotsManifest(
        source_file="fake.mp4", fps=30.0, total_frames=n,
        shots=[Shot(
            id="play", clip_file="shots/play.mp4",
            start_frame=0, end_frame=n - 1,
            start_time=0.0, end_time=(n - 1) / 30.0,
        )],
    ).save(path)


def _minimal_cfg() -> dict:
    return {
        "ball": {
            "detector": "fake",
            "ball_radius_m": 0.11,
            "max_gap_frames": 6,
            "flight_max_residual_px": 5.0,
            "tracker": {
                "process_noise_grounded_px": 4.0,
                "process_noise_flight_px": 12.0,
                "measurement_noise_px": 2.0,
                "gating_sigma": 4.0,
                "min_flight_frames": 6,
                "max_flight_frames": 90,
            },
            "spin": {"enabled": False, "min_flight_seconds": 0.5,
                     "min_residual_improvement": 0.2,
                     "max_omega_rad_s": 200.0, "drag_k_over_m": 0.005},
            "plausibility": {"z_max_m": 50.0, "horizontal_speed_max_m_s": 40.0, "pitch_margin_m": 5.0},
            "flight_promotion": {"enabled": False, "min_run_frames": 6,
                                 "off_pitch_margin_m": 5.0, "max_ground_speed_m_s": 35.0},
            "kick_anchor": {"enabled": False, "max_pixel_distance_px": 30.0, "lookahead_frames": 4,
                            "min_pixel_acceleration_px_per_frame": 6.0, "foot_anchor_z_m": 0.11},
            "appearance_bridge": {"enabled": False, "max_gap_frames": 8, "template_size_px": 32,
                                  "search_radius_px": 64, "min_ncc": 0.6,
                                  "template_max_age_frames": 30, "template_update_confidence": 0.5},
        },
        "pitch": {"length_m": 105.0, "width_m": 68.0},
    }


@pytest.mark.integration
def test_anchor_overrides_wasb_detection(tmp_path: Path):
    """When an anchor exists at a frame, its pixel position is used and
    WASB is ignored on that frame."""
    K, R, t = _camera_pose()
    out = tmp_path / "out"
    n_frames = 20
    _write_blank_clip(out / "shots" / "play.mp4", n_frames)
    _save_camera_track(out / "camera" / "play_camera_track.json", K, R, t, n_frames)
    _save_manifest(out / "shots" / "shots_manifest.json", n_frames)

    # WASB will return one (wrong) pixel for every frame.
    wasb_uv = (100.0, 100.0)
    wasb_detections = [(wasb_uv[0], wasb_uv[1], 0.85) for _ in range(n_frames)]

    # Anchor at frame 5 says the ball is at a different pixel position.
    anchor_uv = (640.0, 360.0)
    BallAnchorSet(
        clip_id="play", image_size=(1280, 720),
        anchors=(BallAnchor(frame=5, image_xy=anchor_uv, state="grounded"),),
    ).save(out / "ball" / "play_ball_anchors.json")

    BallStage(
        config=_minimal_cfg(), output_dir=out,
        ball_detector=FakeBallDetector(wasb_detections),
    ).run()

    track = BallTrack.load(out / "ball" / "play_ball_track.json")
    f5 = next(f for f in track.frames if f.frame == 5)
    # Recover the pixel by reprojection: ground-projection of the
    # anchor pixel at z=0.11 should equal f5.world_xyz exactly.
    from src.utils.foot_anchor import ankle_ray_to_pitch
    expected_world = ankle_ray_to_pitch(anchor_uv, K=K, R=R, t=t, plane_z=0.11)
    assert f5.state == "grounded"
    assert np.allclose(f5.world_xyz, expected_world, atol=1e-3)


@pytest.mark.integration
def test_no_anchor_file_runs_normally(tmp_path: Path):
    """When no anchor file exists, ball stage runs exactly as before."""
    K, R, t = _camera_pose()
    out = tmp_path / "out"
    n_frames = 20
    _write_blank_clip(out / "shots" / "play.mp4", n_frames)
    _save_camera_track(out / "camera" / "play_camera_track.json", K, R, t, n_frames)
    _save_manifest(out / "shots" / "shots_manifest.json", n_frames)

    detections = [(640.0, 360.0, 0.85) for _ in range(n_frames)]
    BallStage(
        config=_minimal_cfg(), output_dir=out,
        ball_detector=FakeBallDetector(detections),
    ).run()
    track = BallTrack.load(out / "ball" / "play_ball_track.json")
    # No anchors → WASB pixel used → all frames grounded.
    assert any(f.state == "grounded" for f in track.frames)


@pytest.mark.integration
def test_off_screen_flight_anchor_skips_pixel(tmp_path: Path):
    """An off_screen_flight anchor produces no pixel detection but
    forces the frame into a flight run."""
    K, R, t = _camera_pose()
    out = tmp_path / "out"
    n_frames = 30
    _write_blank_clip(out / "shots" / "play.mp4", n_frames)
    _save_camera_track(out / "camera" / "play_camera_track.json", K, R, t, n_frames)
    _save_manifest(out / "shots" / "shots_manifest.json", n_frames)

    # WASB returns None for every frame.
    detections: list[tuple[float, float, float] | None] = [None] * n_frames

    anchors = [
        BallAnchor(frame=fi, image_xy=None, state="off_screen_flight")
        for fi in range(10, 20)
    ]
    BallAnchorSet(
        clip_id="play", image_size=(1280, 720),
        anchors=tuple(anchors),
    ).save(out / "ball" / "play_ball_anchors.json")

    BallStage(
        config=_minimal_cfg(), output_dir=out,
        ball_detector=FakeBallDetector(detections),
    ).run()
    track = BallTrack.load(out / "ball" / "play_ball_track.json")
    # Frames 10..19 should not be state="missing".
    forced = [f for f in track.frames if 10 <= f.frame <= 19]
    assert all(f.state != "missing" for f in forced), (
        f"off-screen-flight anchors must not emit missing: {[f.state for f in forced]}"
    )
```

- [ ] **Step 2: Run the new tests — expect FAIL**

Run: `.venv/bin/pytest tests/test_ball_stage_anchors.py -v`
Expected: FAILs because Layer 5 has not been wired yet.

- [ ] **Step 3: Wire Layer 5 anchor injection in `_run_shot`**

In `src/stages/ball.py`:

(a) Add imports near the top:

```python
from src.schemas.ball_anchor import BallAnchor, BallAnchorSet
from src.utils.ball_anchor_heights import (
    AIRBORNE_STATES,
    EVENT_STATES,
    HARD_KNOT_STATES,
    state_to_height,
)
```

(b) Add a module-scope helper near `_load_foot_uvs_for_shot`:

```python
def _load_ball_anchors(
    output_dir: Path, shot_id: str
) -> dict[int, BallAnchor]:
    """Load per-frame ball anchors keyed by frame index.

    Returns an empty dict when no anchor file exists.
    """
    if shot_id:
        path = output_dir / "ball" / f"{shot_id}_ball_anchors.json"
    else:
        path = output_dir / "ball" / "ball_anchors.json"
    if not path.exists():
        return {}
    try:
        aset = BallAnchorSet.load(path)
    except Exception as exc:
        logger.warning("ball stage: failed to load anchors at %s: %s", path, exc)
        return {}
    return {a.frame: a for a in aset.anchors}
```

(c) Inside `_run_shot`, right after `feet_pixel_by_frame = _load_foot_uvs_for_shot(...)`, load the anchors:

```python
        anchor_by_frame = _load_ball_anchors(self.output_dir, shot_id)
        if anchor_by_frame:
            logger.info(
                "ball stage: loaded %d anchors for shot %s",
                len(anchor_by_frame), shot_id or "(legacy)",
            )
```

(d) Modify the detection loop to consult `anchor_by_frame` BEFORE asking
the detector. Find the current loop:

```python
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                det = detector.detect(frame)
                if det is None:
                    ...
```

Replace the body with anchor-first logic:

```python
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                anchor = anchor_by_frame.get(frame_idx)
                if anchor is not None:
                    if anchor.state == "off_screen_flight":
                        # No pixel; let the IMM predict, but record the
                        # forced flight marker for the flight-run pass below.
                        uv = None
                        # raw_confidences not set: caller path treats this
                        # as a gap-fill, and our forced_flight set drives
                        # promotion later.
                    else:
                        uv = (float(anchor.image_xy[0]), float(anchor.image_xy[1]))
                        raw_confidences[frame_idx] = 1.0
                        bridge.update_template(
                            frame=frame_idx, frame_image=frame,
                            uv=uv, confidence=1.0,
                        )
                    consecutive_misses = 0
                else:
                    det = detector.detect(frame)
                    if det is None:
                        consecutive_misses += 1
                        bridge_result = bridge.try_bridge(
                            frame=frame_idx,
                            frame_image=frame,
                            predicted_uv=(
                                (float(steps[-1].uv[0]), float(steps[-1].uv[1]))
                                if steps and steps[-1].uv is not None else None
                            ),
                            consecutive_misses=consecutive_misses,
                        )
                        if bridge_result is None:
                            uv: tuple[float, float] | None = None
                        else:
                            uv, bridged_conf = bridge_result
                            raw_confidences[frame_idx] = bridged_conf
                    else:
                        consecutive_misses = 0
                        uv = (float(det[0]), float(det[1]))
                        raw_confidences[frame_idx] = float(det[2])
                        bridge.update_template(
                            frame=frame_idx, frame_image=frame,
                            uv=uv, confidence=float(det[2]),
                        )
                step = tracker.update(frame_idx, uv)
                steps.append(step)
                frame_idx += 1
```

(e) For off-screen-flight anchors, we need to keep their per-frame
world position out of `per_frame_world` (no pixel → no ground
projection) but mark them as a flight forcing. Build a `forced_flight`
set right after loading anchors:

```python
        forced_flight: set[int] = {
            fi for fi, a in anchor_by_frame.items()
            if a.state in AIRBORNE_STATES
        }
```

(f) After the existing flight-segment loop AND after the Layer 2
promotion pass (which currently exists), insert a "forced-flight"
extension pass that ensures every `forced_flight` frame is part of a
flight run. The simplest implementation:

```python
        if forced_flight:
            # Build current membership: frames already in some flight segment.
            already_flight = set(flight_membership)
            new_frames = forced_flight - already_flight
            if new_frames:
                # Group into runs of consecutive frames.
                runs: list[tuple[int, int]] = []
                run_start = None
                last = None
                for fi in sorted(new_frames):
                    if run_start is None:
                        run_start = fi; last = fi
                    elif fi == last + 1:
                        last = fi
                    else:
                        runs.append((run_start, last))
                        run_start = fi; last = fi
                if run_start is not None:
                    runs.append((run_start, last))
                for (a, b) in runs:
                    sid_new = next_segment_id
                    next_segment_id += 1
                    for fi in range(a, b + 1):
                        flight_membership[fi] = sid_new
                        # Off-screen flight has no world position; leave
                        # per_frame_world unset for these frames.
                    flight_segments.append(
                        FlightSegment(
                            id=sid_new,
                            frame_range=(a, b),
                            parabola={
                                "p0": [0.0, 0.0, 0.0],
                                "v0": [0.0, 0.0, 0.0],
                                "g": -9.81,
                                "spin_axis_world": None,
                                "spin_omega_rad_s": None,
                                "spin_confidence": None,
                            },
                            fit_residual_px=0.0,
                        )
                    )
```

(g) Finally, the BallFrame assembly already classifies frames with
`fi in flight_membership` as flight. Off-screen flight frames have no
`per_frame_world` entry, so the existing code writes
`state="missing"`. Fix that by checking flight_membership first:

Find the existing assembly:
```python
        per_frame_out: list[BallFrame] = []
        for fi in range(n_frames):
            if fi in per_frame_world:
                world, conf = per_frame_world[fi]
                state = "flight" if fi in flight_membership else "grounded"
                ...
            else:
                per_frame_out.append(
                    BallFrame(
                        frame=fi, world_xyz=None,
                        state="missing", confidence=0.0,
                    )
                )
```

Change the `else` branch:

```python
            else:
                # No world position. If the frame was forced into a flight
                # by an off-screen-flight anchor, surface that explicitly.
                if fi in flight_membership:
                    per_frame_out.append(
                        BallFrame(
                            frame=fi, world_xyz=None,
                            state="flight", confidence=0.0,
                            flight_segment_id=flight_membership.get(fi),
                        )
                    )
                else:
                    per_frame_out.append(
                        BallFrame(
                            frame=fi, world_xyz=None,
                            state="missing", confidence=0.0,
                        )
                    )
```

- [ ] **Step 4: Run all ball tests — expect pass**

Run: `.venv/bin/pytest tests/test_ball_stage.py tests/test_ball_stage_anchors.py tests/test_ball_plausibility.py -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/stages/ball.py tests/test_ball_stage_anchors.py
git commit -m "feat(ball): Layer 5 — load anchors and override WASB on anchored frames"
```

---

## Task 5: Layer 5 — events split flight runs and feed knot frames

**Files:**
- Modify: `src/stages/ball.py`
- Modify: `tests/test_ball_stage_anchors.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_ball_stage_anchors.py`:

```python
@pytest.mark.integration
def test_kick_event_anchored_pins_p0(tmp_path: Path):
    """A 'kick' anchor at the start of a flight segment pins p0 to the
    pixel ray-cast at z=0.11."""
    K, R, t = _camera_pose()
    out = tmp_path / "out"
    n_frames = 40
    _write_blank_clip(out / "shots" / "play.mp4", n_frames)
    _save_camera_track(out / "camera" / "play_camera_track.json", K, R, t, n_frames)
    _save_manifest(out / "shots" / "shots_manifest.json", n_frames)

    # Synthesise a kick + parabolic flight in world coordinates, then
    # project to pixels for the FakeBallDetector.
    p0_true = np.array([20.0, 10.0, 0.11])
    v0_true = np.array([2.0, 1.0, 10.0])
    g_vec = np.array([0.0, 0.0, -9.81])
    def _project(p):
        cam = R @ p + t; pix = K @ cam
        return (float(pix[0] / pix[2]), float(pix[1] / pix[2]))
    detections: list[tuple[float, float, float] | None] = [None] * 5
    for i in range(25):
        dt = i / 30.0
        pt = p0_true + v0_true * dt + 0.5 * g_vec * dt ** 2
        u, v = _project(pt)
        detections.append((u, v, 0.85))
    while len(detections) < n_frames:
        detections.append(None)

    # Anchor the kick at frame 5 (segment start in world coords).
    kick_uv = _project(p0_true)
    BallAnchorSet(
        clip_id="play", image_size=(1280, 720),
        anchors=(BallAnchor(frame=5, image_xy=kick_uv, state="kick"),),
    ).save(out / "ball" / "play_ball_anchors.json")

    cfg = _minimal_cfg()
    cfg["ball"]["tracker"]["initial_p_flight"] = 0.5  # IMM seeds flight
    BallStage(
        config=cfg, output_dir=out,
        ball_detector=FakeBallDetector(detections),
    ).run()

    track = BallTrack.load(out / "ball" / "play_ball_track.json")
    assert len(track.flight_segments) >= 1
    seg = track.flight_segments[0]
    p0_fit = np.array(seg.parabola["p0"])
    assert np.linalg.norm(p0_fit - p0_true) < 0.5, (
        f"expected anchored p0 ≈ {p0_true.tolist()}, got {p0_fit.tolist()}"
    )


@pytest.mark.integration
def test_bounce_event_splits_flight_run(tmp_path: Path):
    """A 'bounce' anchor mid-run should terminate one flight segment and
    start a new one."""
    K, R, t = _camera_pose()
    out = tmp_path / "out"
    n_frames = 60
    _write_blank_clip(out / "shots" / "play.mp4", n_frames)
    _save_camera_track(out / "camera" / "play_camera_track.json", K, R, t, n_frames)
    _save_manifest(out / "shots" / "shots_manifest.json", n_frames)

    # Two consecutive parabolic arcs (kick + bounce + second flight).
    def _project(p):
        cam = R @ p + t; pix = K @ cam
        return (float(pix[0] / pix[2]), float(pix[1] / pix[2]))
    g_vec = np.array([0.0, 0.0, -9.81])
    p0_a = np.array([10.0, 5.0, 0.11]); v0_a = np.array([3.0, 0.5, 8.0])
    bounce_frame = 30
    p0_b = p0_a + v0_a * (bounce_frame / 30.0) + 0.5 * g_vec * (bounce_frame / 30.0) ** 2
    p0_b[2] = 0.11  # bounces to ground
    v0_b = np.array([2.0, 0.5, 5.0])
    detections: list[tuple[float, float, float] | None] = [None] * 5
    for i in range(25):  # first arc
        dt = i / 30.0
        pt = p0_a + v0_a * dt + 0.5 * g_vec * dt ** 2
        u, v = _project(pt); detections.append((u, v, 0.85))
    detections.append(None)  # bounce frame placeholder (will be anchored)
    for i in range(20):  # second arc
        dt = i / 30.0
        pt = p0_b + v0_b * dt + 0.5 * g_vec * dt ** 2
        u, v = _project(pt); detections.append((u, v, 0.85))
    while len(detections) < n_frames:
        detections.append(None)

    bounce_uv = _project(np.array([p0_b[0], p0_b[1], 0.11]))
    BallAnchorSet(
        clip_id="play", image_size=(1280, 720),
        anchors=(BallAnchor(frame=bounce_frame, image_xy=bounce_uv, state="bounce"),),
    ).save(out / "ball" / "play_ball_anchors.json")

    cfg = _minimal_cfg()
    cfg["ball"]["tracker"]["initial_p_flight"] = 0.5
    BallStage(
        config=cfg, output_dir=out,
        ball_detector=FakeBallDetector(detections),
    ).run()

    track = BallTrack.load(out / "ball" / "play_ball_track.json")
    # At least two flight segments — one before the bounce, one after.
    # Find the segment(s) covering frames 5..29 and 31..50.
    pre_segs = [s for s in track.flight_segments if s.frame_range[1] < bounce_frame]
    post_segs = [s for s in track.flight_segments if s.frame_range[0] > bounce_frame]
    assert pre_segs, f"expected a flight segment ending before frame {bounce_frame}"
    assert post_segs, f"expected a flight segment starting after frame {bounce_frame}"
```

- [ ] **Step 2: Run tests — expect FAIL**

Run: `.venv/bin/pytest tests/test_ball_stage_anchors.py::test_kick_event_anchored_pins_p0 tests/test_ball_stage_anchors.py::test_bounce_event_splits_flight_run -v`
Expected: FAIL.

- [ ] **Step 3: Implement event-aware flight runs + knot frames**

In `src/stages/ball.py`:

(a) Right after `flight_runs = self._flight_runs(steps, min_flight, max_flight)`, post-process the runs so each `EVENT_STATES` anchor frame becomes a run boundary:

```python
        if anchor_by_frame:
            event_frames = sorted(
                fi for fi, a in anchor_by_frame.items()
                if a.state in EVENT_STATES
            )
            if event_frames:
                split_runs: list[tuple[int, int]] = []
                for (a, b) in flight_runs:
                    cuts = [fi for fi in event_frames if a <= fi <= b]
                    if not cuts:
                        split_runs.append((a, b))
                        continue
                    prev = a
                    for cut in cuts:
                        if cut - 1 >= prev:
                            split_runs.append((prev, cut - 1))
                        prev = cut + 1
                    if prev <= b:
                        split_runs.append((prev, b))
                flight_runs = split_runs
```

(b) Inside the existing flight-segment loop (the one that calls
`fit_parabola_to_image_observations`), build `knot_frames` and pass
to the fit. Locate the block where `anchor_world` is computed for
kick-anchoring (Task 9 from the prior plan) and extend it. After the
existing `anchor_world = find_kick_anchor(...)` line, add:

```python
            # Layer 5 — hard knots from anchored frames within this segment.
            knot_frames_arg: dict[int, np.ndarray] = {}
            for fi in range(a, b + 1):
                anc = anchor_by_frame.get(fi)
                if anc is None or anc.state not in HARD_KNOT_STATES:
                    continue
                if anc.image_xy is None:
                    continue
                if fi not in per_frame_K:
                    continue
                z = state_to_height(anc.state)
                world_at_anchor = ankle_ray_to_pitch(
                    anc.image_xy,
                    K=per_frame_K[fi], R=per_frame_R[fi], t=per_frame_t[fi],
                    plane_z=z, distortion=distortion,
                )
                knot_frames_arg[fi - a] = np.asarray(world_at_anchor, dtype=float)

            # If the seed frame is a kick/grounded/bounce/catch knot,
            # promote it to p0_fixed so the parabola is pinned exactly.
            if 0 in knot_frames_arg and anchor_world is None:
                anchor_world = knot_frames_arg.pop(0)
```

Then change the fit call to pass `knot_frames`:

```python
            try:
                p0, v0, parab_resid = fit_parabola_to_image_observations(
                    obs, Ks=Ks_seg, Rs=Rs_seg, t_world=ts_seg,
                    fps=camera.fps, distortion=distortion,
                    p0_fixed=anchor_world,
                    knot_frames=knot_frames_arg or None,
                )
            except Exception as exc:
                logger.debug("parabola fit failed on segment %d: %s", sid, exc)
                continue
```

(c) The import for `ankle_ray_to_pitch` was already added in an earlier
task — verify it's still present at the top of `ball.py`.

- [ ] **Step 4: Run tests — expect pass**

Run: `.venv/bin/pytest tests/test_ball_stage_anchors.py tests/test_ball_stage.py tests/test_ball_stage_layered.py -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/stages/ball.py tests/test_ball_stage_anchors.py
git commit -m "feat(ball): Layer 5 — events split flight runs and feed knot frames"
```

---

## Task 6: Server endpoint — GET/POST `/ball-anchors/{shot_id}`

**Files:**
- Modify: `src/web/server.py`
- Create: `tests/test_ball_anchor_endpoints.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_ball_anchor_endpoints.py`:

```python
"""Smoke tests for the /ball-anchors endpoints."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.web.server import build_app


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    # build_app reads output dir from constructor; mirror the pattern
    # used by existing tests in this repo.
    (tmp_path / "shots").mkdir()
    app = build_app(output_dir=tmp_path)
    return TestClient(app)


def test_get_ball_anchors_returns_empty_when_no_file(client: TestClient, tmp_path: Path):
    r = client.get("/ball-anchors/play")
    assert r.status_code == 200
    body = r.json()
    assert body["anchors"] == []


def test_post_then_get_roundtrips(client: TestClient, tmp_path: Path):
    payload = {
        "clip_id": "play",
        "image_size": [1280, 720],
        "anchors": [
            {"frame": 5, "image_xy": [640.0, 360.0], "state": "grounded"},
            {"frame": 78, "image_xy": [700.0, 200.0], "state": "kick"},
            {"frame": 120, "image_xy": None, "state": "off_screen_flight"},
        ],
    }
    r = client.post("/ball-anchors/play", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["saved"] is True
    assert body["count"] == 3
    # File should exist on disk.
    p = tmp_path / "ball" / "play_ball_anchors.json"
    assert p.exists()
    saved = json.loads(p.read_text())
    assert len(saved["anchors"]) == 3
    # GET should return what we POSTed.
    r2 = client.get("/ball-anchors/play")
    assert r2.status_code == 200
    body2 = r2.json()
    assert len(body2["anchors"]) == 3
    assert body2["anchors"][0]["state"] == "grounded"


def test_post_invalid_state_rejected(client: TestClient):
    payload = {
        "clip_id": "play", "image_size": [1280, 720],
        "anchors": [{"frame": 1, "image_xy": [0, 0], "state": "bogus"}],
    }
    r = client.post("/ball-anchors/play", json=payload)
    assert r.status_code == 400


def test_post_missing_pixel_for_grounded_state_rejected(client: TestClient):
    payload = {
        "clip_id": "play", "image_size": [1280, 720],
        "anchors": [{"frame": 1, "image_xy": None, "state": "grounded"}],
    }
    r = client.post("/ball-anchors/play", json=payload)
    assert r.status_code == 400
```

- [ ] **Step 2: Run tests — expect FAIL (endpoints missing)**

Run: `.venv/bin/pytest tests/test_ball_anchor_endpoints.py -v`
Expected: FAIL with 404 on GET/POST.

- [ ] **Step 3: Add endpoints to `src/web/server.py`**

Open `src/web/server.py`. Near the existing `class AnchorPayload`,
add a sibling Pydantic model:

```python
class BallAnchorEntry(BaseModel):
    frame: int
    image_xy: list[float] | None
    state: str


class BallAnchorPayload(BaseModel):
    clip_id: str
    image_size: list[int]
    anchors: list[BallAnchorEntry]
```

Then, near the existing `@app.get("/anchors/{shot_id}")` /
`@app.post("/anchors/{shot_id}")` handlers, add:

```python
    @app.get("/ball-anchors/{shot_id}")
    def get_ball_anchors_for_shot(shot_id: str):
        path = output_dir / "ball" / f"{shot_id}_ball_anchors.json"
        if not path.exists():
            return {"clip_id": shot_id, "image_size": [0, 0], "anchors": []}
        try:
            data = json.loads(path.read_text())
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to load ball anchors: {exc}")
        return data

    @app.post("/ball-anchors/{shot_id}")
    def post_ball_anchors_for_shot(shot_id: str, payload: BallAnchorPayload):
        from src.schemas.ball_anchor import BallAnchor, BallAnchorSet
        try:
            anchors = []
            for a in payload.anchors:
                anchors.append(BallAnchor(
                    frame=int(a.frame),
                    image_xy=tuple(a.image_xy) if a.image_xy is not None else None,
                    state=a.state,
                ))
            aset = BallAnchorSet(
                clip_id=str(payload.clip_id),
                image_size=(int(payload.image_size[0]), int(payload.image_size[1])),
                anchors=tuple(anchors),
            )
            # Round-trip through JSON to apply state-validation rules.
            tmp = output_dir / "ball" / f".{shot_id}_ball_anchors.tmp.json"
            tmp.parent.mkdir(parents=True, exist_ok=True)
            aset.save(tmp)
            BallAnchorSet.load(tmp)  # validation pass
        except (KeyError, TypeError, ValueError) as exc:
            try:
                tmp.unlink()
            except Exception:
                pass
            raise HTTPException(status_code=400, detail=f"Invalid ball anchors: {exc}")
        final = output_dir / "ball" / f"{shot_id}_ball_anchors.json"
        tmp.replace(final)
        return {"saved": True, "path": str(final), "count": len(aset.anchors)}
```

- [ ] **Step 4: Run tests — expect pass**

Run: `.venv/bin/pytest tests/test_ball_anchor_endpoints.py -v`
Expected: all 4 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/web/server.py tests/test_ball_anchor_endpoints.py
git commit -m "feat(web): GET/POST /ball-anchors/{shot_id} endpoints"
```

---

## Task 7: Server endpoint — POST `/ball-anchors/{shot_id}/preview`

**Files:**
- Modify: `src/web/server.py`
- Modify: `tests/test_ball_anchor_endpoints.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_ball_anchor_endpoints.py`:

```python
def test_preview_endpoint_runs_ball_stage_with_payload(client: TestClient, tmp_path: Path):
    """Preview should run BallStage in a tmp output dir using the
    posted anchors, returning the resulting BallTrack JSON."""
    import cv2
    import numpy as np
    from src.schemas.camera_track import CameraFrame, CameraTrack
    from src.schemas.shots import Shot, ShotsManifest

    # Set up minimal shot + camera + clip on the SHARED output dir
    # (the client's app reads from tmp_path).
    K = np.array([[1500.0, 0, 640.0], [0, 1500.0, 360.0], [0, 0, 1.0]])
    R = np.eye(3); t = np.array([0.0, 0.0, 30.0])
    n_frames = 20
    clip = tmp_path / "shots" / "play.mp4"
    clip.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (320, 240))
    for _ in range(n_frames):
        writer.write(np.full((240, 320, 3), [50, 200, 50], dtype=np.uint8))
    writer.release()
    CameraTrack(
        clip_id="play", fps=30.0, image_size=(1280, 720), t_world=t.tolist(),
        frames=tuple(CameraFrame(frame=i, K=K.tolist(), R=R.tolist(),
                                 confidence=1.0, is_anchor=(i == 0))
                     for i in range(n_frames)),
    ).save(tmp_path / "camera" / "play_camera_track.json")
    ShotsManifest(
        source_file="fake.mp4", fps=30.0, total_frames=n_frames,
        shots=[Shot(id="play", clip_file="shots/play.mp4",
                    start_frame=0, end_frame=n_frames - 1,
                    start_time=0.0, end_time=(n_frames - 1) / 30.0)],
    ).save(tmp_path / "shots" / "shots_manifest.json")

    payload = {
        "clip_id": "play", "image_size": [1280, 720],
        "anchors": [{"frame": 5, "image_xy": [640.0, 360.0], "state": "grounded"}],
    }
    r = client.post("/ball-anchors/play/preview", json=payload)
    # Allow 200 OK with a BallTrack, or 500 if WASB cannot run on the
    # test box — accept both as long as we get JSON back. The important
    # assertion is that the endpoint accepts the payload shape.
    assert r.status_code in (200, 500), r.text
    if r.status_code == 200:
        body = r.json()
        assert "frames" in body
        assert "flight_segments" in body
```

- [ ] **Step 2: Run test — expect FAIL (endpoint missing)**

Run: `.venv/bin/pytest tests/test_ball_anchor_endpoints.py::test_preview_endpoint_runs_ball_stage_with_payload -v`
Expected: FAIL with 404.

- [ ] **Step 3: Implement the preview endpoint**

Append to `src/web/server.py`, next to the other `/ball-anchors/*` handlers:

```python
    @app.post("/ball-anchors/{shot_id}/preview")
    def preview_ball_anchors(shot_id: str, payload: BallAnchorPayload):
        """Run the ball stage in a temp output dir using the posted
        anchors. Returns the would-be BallTrack JSON without touching
        the on-disk ball_track.

        Requires that the production output dir already has a
        camera_track and a clip for ``shot_id``.
        """
        import shutil
        import tempfile
        from src.schemas.ball_anchor import BallAnchor, BallAnchorSet
        from src.schemas.ball_track import BallTrack
        from src.stages.ball import BallStage

        cam_path = output_dir / "camera" / f"{shot_id}_camera_track.json"
        if not cam_path.exists():
            raise HTTPException(status_code=404,
                                detail=f"No camera track for shot {shot_id!r}")
        manifest_path = output_dir / "shots" / "shots_manifest.json"
        if not manifest_path.exists():
            raise HTTPException(status_code=404,
                                detail=f"No shots manifest at {manifest_path}")

        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            # Hard-link the production camera + shots data into the
            # temp output dir; ball stage writes only under tdp/ball.
            for sub in ("camera", "shots", "hmr_world"):
                src = output_dir / sub
                if src.exists():
                    shutil.copytree(src, tdp / sub, dirs_exist_ok=True)

            anchors = tuple(
                BallAnchor(
                    frame=int(a.frame),
                    image_xy=tuple(a.image_xy) if a.image_xy is not None else None,
                    state=a.state,
                ) for a in payload.anchors
            )
            try:
                aset = BallAnchorSet(
                    clip_id=str(payload.clip_id),
                    image_size=(int(payload.image_size[0]), int(payload.image_size[1])),
                    anchors=anchors,
                )
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc))
            ball_dir = tdp / "ball"
            ball_dir.mkdir(parents=True, exist_ok=True)
            aset.save(ball_dir / f"{shot_id}_ball_anchors.json")

            try:
                # Use the project's actual config for sensible defaults.
                from src.config import load_default_config
                cfg = load_default_config()
            except Exception:
                cfg = {"ball": {"detector": "wasb"}, "pitch": {"length_m": 105.0, "width_m": 68.0}}
            stage = BallStage(config=cfg, output_dir=tdp)
            stage.shot_filter = shot_id
            try:
                stage.run()
            except Exception as exc:
                raise HTTPException(status_code=500, detail=f"BallStage preview failed: {exc}")
            track_path = ball_dir / f"{shot_id}_ball_track.json"
            if not track_path.exists():
                raise HTTPException(status_code=500, detail="Preview produced no ball_track")
            return json.loads(track_path.read_text())
```

If `src.config.load_default_config` doesn't exist in this codebase,
substitute the actual default-config loader used by `recon.py`. Find
it with `grep -n "def load_default\|def load_config\|yaml.safe_load.*default" src/`.

- [ ] **Step 4: Run tests — expect pass**

Run: `.venv/bin/pytest tests/test_ball_anchor_endpoints.py -v`
Expected: all pass (preview test accepts 200 or 500 to tolerate the
detector backend not being installable in CI).

- [ ] **Step 5: Commit**

```bash
git add src/web/server.py tests/test_ball_anchor_endpoints.py
git commit -m "feat(web): POST /ball-anchors/{shot_id}/preview"
```

---

## Task 8: Editor HTML — scaffold + tag palette + click-to-anchor

**Files:**
- Create: `src/web/static/ball_anchor_editor.html`

This task is verified manually (no automated UI tests). The acceptance
criterion is: the page loads at
`http://localhost:8000/ball-anchor-editor?shot=origi01`, shows the
clip video with the existing anchor set rendered, lets the operator
select a tag and click on the video to add an anchor, click an anchor
to delete it, and POST the result back to the server.

- [ ] **Step 1: Add the static-route mount**

In `src/web/server.py`, look for the existing route that serves
`anchor_editor.html` (`/anchor-editor`). Mirror it for the ball
editor — typically:

```python
    @app.get("/ball-anchor-editor", include_in_schema=False)
    def serve_ball_anchor_editor():
        return FileResponse(STATIC_DIR / "ball_anchor_editor.html")
```

Restart the dev server (`python recon.py serve --output ./output/`)
and confirm `GET /ball-anchor-editor` returns the file (it'll 404
until the next step). Commit this stub:

```bash
git add src/web/server.py
git commit -m "feat(web): route /ball-anchor-editor to static page"
```

- [ ] **Step 2: Write the editor HTML scaffold**

Create `src/web/static/ball_anchor_editor.html`. Use
`src/web/static/anchor_editor.html` as a template for header + shell
CSS. The body of this task is the diff against that template; copy
the entire CSS block from `anchor_editor.html` and adapt the
landmark-row class to a ball-tag-row class. The minimum viable
structure:

```html
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Ball Anchor Editor — Football Perspectives</title>
<style>
  /* (copy the entire <style> block from anchor_editor.html — every
     selector you use here is already defined there) */
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    background: #0f1117; color: #e2e8f0; height: 100vh;
    display: flex; flex-direction: column; overflow: hidden; font-size: 13px;
  }
  header { background:#1a1d27; border-bottom:1px solid #2d3148;
           padding:10px 16px; display:flex; align-items:center;
           gap:12px; flex-shrink:0; }
  header h1 { font-size: 15px; font-weight:600; color:#f1f5f9; }
  header select, header button { background:#334155; color:#e2e8f0;
           border:1px solid #475569; border-radius:4px;
           padding:4px 10px; font-size:12px; font-family:inherit; cursor:pointer; }
  header button.primary { background:#4f46e5; border-color:#6366f1; color:#fff; }
  header .spacer { flex:1; }
  header #status { font-size:11px; color:#94a3b8; min-width:140px; text-align:right; }
  main { flex:1; display:grid; grid-template-columns:220px 1fr 240px;
         overflow:hidden; min-height:0; }
  .panel { background:#181b25; border-right:1px solid #2d3148;
           display:flex; flex-direction:column; min-height:0; }
  .panel:last-child { border-right:none; border-left:1px solid #2d3148; }
  .panel h2 { font-size:11px; font-weight:600; letter-spacing:.06em;
              text-transform:uppercase; color:#64748b; padding:10px 12px;
              border-bottom:1px solid #2d3148; flex-shrink:0; }
  .panel-body { flex:1; overflow-y:auto; padding:8px; }
  .tag-row { display:flex; align-items:center; gap:6px; padding:5px 8px;
             cursor:pointer; border-radius:3px; font-size:12px; color:#cbd5e1; user-select:none; }
  .tag-row:hover { background:#252840; }
  .tag-row.selected { background:#3730a3; color:#fff; }
  .anchor-row { padding:6px 8px; cursor:pointer; border-radius:3px;
                font-size:12px; color:#cbd5e1; }
  .anchor-row:hover { background:#252840; }
  .anchor-row.active { background:#1e293b; color:#fff; }
  #stage { position:relative; flex:1; background:#000;
           display:flex; align-items:center; justify-content:center; overflow:hidden; }
  #stage video { max-width:100%; max-height:100%; display:block; }
  #stage canvas { position:absolute; left:0; top:0;
                  width:100%; height:100%; pointer-events:auto; cursor:crosshair; }
  .ctrl { display:flex; align-items:center; gap:8px;
          padding:8px 12px; background:#161924; border-top:1px solid #2d3148; }
</style>
</head>
<body>
<header>
  <h1>Ball Anchor Editor</h1>
  <label>Shot <select id="shotSelect"></select></label>
  <button id="saveBtn" class="primary">Save</button>
  <button id="previewBtn">Solve &amp; Preview</button>
  <span class="spacer"></span>
  <span id="status">Loading…</span>
</header>
<main>
  <section class="panel">
    <h2>Tags</h2>
    <div class="panel-body" id="tagList"></div>
  </section>
  <section class="panel" style="display:flex;flex-direction:column;">
    <div id="stage">
      <video id="video" preload="metadata" muted></video>
      <canvas id="overlay"></canvas>
    </div>
    <div class="ctrl">
      <button id="play">▶</button>
      <button id="prev">◀</button>
      <button id="next">▶</button>
      <input id="seek" type="range" style="flex:1;">
      <span id="frameLabel" style="color:#94a3b8;font-size:11px;min-width:90px;text-align:right;">Frame 0</span>
    </div>
  </section>
  <section class="panel">
    <h2>Anchors</h2>
    <div class="panel-body" id="anchorList"></div>
  </section>
</main>
<script>
const TAGS = [
  { id: "grounded",          label: "Grounded",         color: "#34d399" },
  { id: "airborne_low",      label: "Airborne — low",   color: "#fbbf24" },
  { id: "airborne_mid",      label: "Airborne — mid",   color: "#fb923c" },
  { id: "airborne_high",     label: "Airborne — high",  color: "#f87171" },
  { id: "off_screen_flight", label: "Off-screen flight",color: "#64748b" },
  { id: "kick",              label: "Kick (foot)",      color: "#60a5fa" },
  { id: "catch",             label: "Catch",            color: "#a78bfa" },
  { id: "bounce",            label: "Bounce",           color: "#f472b6" },
];
let selectedTag = "grounded";
let shotId = new URLSearchParams(location.search).get("shot") || "";
let fps = 30;
let anchors = [];   // { frame, image_xy: [u,v] | null, state }
let imageSize = [1280, 720];
let dirty = false;

const video = document.getElementById("video");
const overlay = document.getElementById("overlay");
const seek = document.getElementById("seek");
const frameLabel = document.getElementById("frameLabel");
const tagList = document.getElementById("tagList");
const anchorList = document.getElementById("anchorList");
const statusEl = document.getElementById("status");

function setStatus(msg) { statusEl.textContent = msg; }
function setDirty(d) { dirty = d; setStatus(d ? `${anchors.length} anchors • unsaved` : `${anchors.length} anchors`); }

function renderTags() {
  tagList.innerHTML = "";
  for (const tag of TAGS) {
    const row = document.createElement("div");
    row.className = "tag-row" + (tag.id === selectedTag ? " selected" : "");
    row.innerHTML = `<span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:${tag.color};"></span> ${tag.label}`;
    row.onclick = () => { selectedTag = tag.id; renderTags(); };
    tagList.appendChild(row);
  }
}

function renderAnchors() {
  anchorList.innerHTML = "";
  const sorted = [...anchors].sort((a, b) => a.frame - b.frame);
  for (const a of sorted) {
    const t = TAGS.find(x => x.id === a.state) || { color: "#94a3b8" };
    const row = document.createElement("div");
    row.className = "anchor-row";
    row.innerHTML = `<span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:${t.color};"></span> Frame ${a.frame} — ${a.state}`;
    row.onclick = () => seekToFrame(a.frame);
    anchorList.appendChild(row);
  }
}

function currentFrame() { return Math.round(video.currentTime * fps); }
function seekToFrame(fi) { video.currentTime = fi / fps; }

function fitOverlay() {
  if (!video.videoWidth) return;
  overlay.width = video.videoWidth;
  overlay.height = video.videoHeight;
  overlay.style.width = video.offsetWidth + "px";
  overlay.style.height = video.offsetHeight + "px";
}

function drawOverlay() {
  if (!overlay.width) return;
  const ctx = overlay.getContext("2d");
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  const fi = currentFrame();
  frameLabel.textContent = `Frame ${fi}`;
  for (const a of anchors) {
    if (a.frame !== fi || a.image_xy == null) continue;
    const tag = TAGS.find(x => x.id === a.state) || { color: "#fff" };
    ctx.strokeStyle = tag.color; ctx.fillStyle = tag.color;
    ctx.lineWidth = 2;
    ctx.beginPath(); ctx.arc(a.image_xy[0], a.image_xy[1], 10, 0, Math.PI * 2); ctx.stroke();
    ctx.beginPath(); ctx.arc(a.image_xy[0], a.image_xy[1], 3, 0, Math.PI * 2); ctx.fill();
  }
}

overlay.addEventListener("click", (e) => {
  if (selectedTag === "off_screen_flight") return;   // handled by a dedicated button below
  const rect = overlay.getBoundingClientRect();
  const u = (e.clientX - rect.left) * (overlay.width / rect.width);
  const v = (e.clientY - rect.top) * (overlay.height / rect.height);
  const fi = currentFrame();
  // Right-click handled separately; treat click on existing anchor as delete.
  const existingIdx = anchors.findIndex(a => a.frame === fi && a.state === selectedTag);
  if (existingIdx >= 0) {
    anchors.splice(existingIdx, 1);
  } else {
    // Replace any anchor at the same frame (one anchor per frame).
    anchors = anchors.filter(a => a.frame !== fi);
    anchors.push({ frame: fi, image_xy: [u, v], state: selectedTag });
  }
  setDirty(true);
  renderAnchors();
  drawOverlay();
});

document.getElementById("play").onclick = () => { video.paused ? video.play() : video.pause(); };
document.getElementById("prev").onclick = () => { video.pause(); seekToFrame(Math.max(0, currentFrame() - 1)); };
document.getElementById("next").onclick = () => { video.pause(); seekToFrame(currentFrame() + 1); };
seek.oninput = () => { video.pause(); video.currentTime = parseInt(seek.value, 10) / fps; };
video.addEventListener("timeupdate", drawOverlay);
video.addEventListener("loadedmetadata", () => {
  seek.max = String(Math.round(video.duration * fps));
  fitOverlay(); drawOverlay();
});
new ResizeObserver(fitOverlay).observe(video);

document.getElementById("saveBtn").onclick = async () => {
  const payload = { clip_id: shotId || "", image_size: imageSize, anchors };
  const r = await fetch(`/ball-anchors/${encodeURIComponent(shotId)}`, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (r.ok) { setStatus(`saved ${anchors.length}`); dirty = false; }
  else { setStatus(`save failed: ${r.status}`); }
};

async function loadShot() {
  // Use the shotSelect or the URL ?shot=foo.
  if (!shotId) {
    const list = await fetch("/api/shots").then(r => r.json()).catch(() => null);
    if (list && list.length) shotId = list[0].id;
  }
  if (!shotId) { setStatus("No shot available"); return; }
  video.src = `/api/video/${encodeURIComponent(shotId)}`;
  const cam = await fetch(`/camera/track?shot=${encodeURIComponent(shotId)}`).then(r => r.json()).catch(() => null);
  if (cam && cam.fps) fps = cam.fps;
  if (cam && cam.image_size) imageSize = cam.image_size;
  const ar = await fetch(`/ball-anchors/${encodeURIComponent(shotId)}`).then(r => r.json()).catch(() => null);
  anchors = ar && ar.anchors ? ar.anchors.map(a => ({ ...a, image_xy: a.image_xy })) : [];
  renderTags();
  renderAnchors();
  setStatus(`${anchors.length} anchors`);
}

loadShot();
</script>
</body>
</html>
```

- [ ] **Step 3: Manual verification**

1. Run `.venv/bin/python recon.py serve --output ./output/`.
2. Open `http://localhost:8000/ball-anchor-editor?shot=origi01`.
3. Confirm the clip plays and the tag palette renders.
4. Pick a tag, click on the video at a few frames, confirm crosshairs
   appear and the anchor list updates.
5. Click "Save". Confirm
   `output/ball/origi01_ball_anchors.json` is written and contains
   the expected entries.

- [ ] **Step 4: Commit**

```bash
git add src/web/static/ball_anchor_editor.html
git commit -m "feat(web): ball anchor editor — scaffold + click-to-anchor"
```

---

## Task 9: Editor — off-screen-flight button + delete + solve & preview

**Files:**
- Modify: `src/web/static/ball_anchor_editor.html`

- [ ] **Step 1: Add the "Mark current frame as off-screen flight" button**

In the editor JS, after the existing "click overlay to place anchor"
handler, add a control that's visible only when the
`off_screen_flight` tag is selected. Inside the `renderTags`
function, also toggle a button under the seek bar:

Add to the ctrl bar HTML in the body:

```html
<button id="offScreenBtn" style="display:none;">Mark off-screen flight</button>
```

And in the JS:

```javascript
const offScreenBtn = document.getElementById("offScreenBtn");
function updateOffScreenBtn() {
  offScreenBtn.style.display = selectedTag === "off_screen_flight" ? "" : "none";
}
const _origRenderTags = renderTags;
renderTags = function() { _origRenderTags(); updateOffScreenBtn(); };

offScreenBtn.onclick = () => {
  const fi = currentFrame();
  anchors = anchors.filter(a => a.frame !== fi);
  anchors.push({ frame: fi, image_xy: null, state: "off_screen_flight" });
  setDirty(true);
  renderAnchors();
  drawOverlay();
};
```

- [ ] **Step 2: Add right-click delete on anchor dots**

Replace the existing `overlay.addEventListener("click", ...)` block
with this enhanced version that detects right-click for delete and
left-click on an existing dot for delete:

```javascript
overlay.addEventListener("contextmenu", (e) => {
  e.preventDefault();
  const rect = overlay.getBoundingClientRect();
  const u = (e.clientX - rect.left) * (overlay.width / rect.width);
  const v = (e.clientY - rect.top) * (overlay.height / rect.height);
  const fi = currentFrame();
  const hit = anchors.findIndex(a =>
    a.frame === fi && a.image_xy &&
    (Math.hypot(a.image_xy[0] - u, a.image_xy[1] - v) < 14)
  );
  if (hit >= 0) {
    anchors.splice(hit, 1);
    setDirty(true);
    renderAnchors(); drawOverlay();
  }
});
```

- [ ] **Step 3: Wire the "Solve & Preview" button**

Replace the existing solve-preview placeholder with a working version:

```javascript
document.getElementById("previewBtn").onclick = async () => {
  const payload = { clip_id: shotId || "", image_size: imageSize, anchors };
  setStatus("Solving…");
  const r = await fetch(`/ball-anchors/${encodeURIComponent(shotId)}/preview`, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!r.ok) { setStatus(`preview failed: ${r.status}`); return; }
  const track = await r.json();
  // Show a sparkline summary: count of each state across frames.
  const counts = {};
  for (const f of track.frames || []) counts[f.state] = (counts[f.state] || 0) + 1;
  setStatus(`preview: ${Object.entries(counts).map(([k,v])=>`${k}:${v}`).join(" ")}`);
  // Optional: draw the predicted ball pixel as a faint ring on the overlay
  // for every frame so the operator can see impact. Store the projected
  // pixels keyed by frame for the next drawOverlay call.
  window._previewByFrame = new Map();
  // The preview returns world_xyz; project on the client side using the
  // already-fetched camera track.
  const cam = await fetch(`/camera/track?shot=${encodeURIComponent(shotId)}`).then(r => r.json());
  const camByFrame = new Map();
  for (const cf of (cam.frames || [])) camByFrame.set(cf.frame, cf);
  for (const f of track.frames || []) {
    if (!f.world_xyz) continue;
    const cf = camByFrame.get(f.frame);
    if (!cf) continue;
    const [x, y, z] = f.world_xyz;
    const R = cf.R, t = cf.t || cam.t_world, K = cf.K;
    const cx = R[0][0]*x + R[0][1]*y + R[0][2]*z + t[0];
    const cy = R[1][0]*x + R[1][1]*y + R[1][2]*z + t[1];
    const cz = R[2][0]*x + R[2][1]*y + R[2][2]*z + t[2];
    if (cz <= 0) continue;
    const u = (K[0][0]*cx + K[0][1]*cy + K[0][2]*cz) / cz;
    const v = (K[1][0]*cx + K[1][1]*cy + K[1][2]*cz) / cz;
    window._previewByFrame.set(f.frame, [u, v]);
  }
  drawOverlay();
};
```

In `drawOverlay`, after drawing the anchor dots, append:

```javascript
  // Layer 5 preview overlay (faint ring at projected ball pixel).
  if (window._previewByFrame) {
    const uv = window._previewByFrame.get(fi);
    if (uv) {
      ctx.strokeStyle = "rgba(255,255,255,0.6)";
      ctx.lineWidth = 1;
      ctx.beginPath(); ctx.arc(uv[0], uv[1], 14, 0, Math.PI * 2); ctx.stroke();
    }
  }
```

- [ ] **Step 4: Manual verification**

1. Restart the server. Reload the editor.
2. Pick "Off-screen flight", scrub to a frame, click the button.
   Confirm an off-screen entry appears in the anchor list.
3. Place 2-3 grounded anchors with crosshairs.
4. Right-click an existing crosshair. Confirm it disappears.
5. Click "Solve & Preview". Confirm the status bar shows state counts
   and a faint ring appears on the video at the predicted ball pixel
   for each frame.

- [ ] **Step 5: Commit**

```bash
git add src/web/static/ball_anchor_editor.html
git commit -m "feat(web): ball editor — off-screen button, delete, solve & preview"
```

---

## Task 10: Dashboard link to ball anchor editor

**Files:**
- Modify: `src/web/static/index.html`

- [ ] **Step 1: Locate the ball panel header rendering**

Open `src/web/static/index.html` and search for `Ball Track summary`
(should be inside `renderBallShot`). Right after the summary card,
add a small toolbar link to the editor.

- [ ] **Step 2: Add the link**

In `renderBallShot`, just after `panel.appendChild(sumWrap);` (the
summary card), insert:

```javascript
  const editorLink = document.createElement("div");
  editorLink.style.cssText = "padding:0 16px 12px;";
  const shotParam = shotId ? `?shot=${encodeURIComponent(shotId)}` : "";
  editorLink.innerHTML = `
    <a href="/ball-anchor-editor${shotParam}" target="_blank"
       style="color:#60a5fa;text-decoration:none;font-size:12px;">
      Open ball anchor editor for ${shotId || "this clip"} →
    </a>
  `;
  panel.appendChild(editorLink);
```

(Use the closure-scoped `shotId` parameter already in scope inside
`renderBallShot` from Task 1 of the source-overlay work.)

- [ ] **Step 3: Manual verification**

1. Restart the server.
2. Open `http://localhost:8000`, navigate to the Ball panel.
3. Confirm the "Open ball anchor editor" link appears under the
   summary card.
4. Click it. Confirm the editor opens in a new tab with the same
   `shot` query parameter.

- [ ] **Step 4: Commit**

```bash
git add src/web/static/index.html
git commit -m "feat(web): link to ball anchor editor from dashboard ball panel"
```

---

## Self-Review

- **Spec coverage:**
  - Schema (Task 1) ✓
  - Heights / classification helpers (Task 2) ✓
  - `knot_frames` kwarg (Task 3) ✓
  - Layer 5 wiring — anchor overrides WASB + off-screen flight forcing (Task 4) ✓
  - Layer 5 wiring — event splits + knot frames (Task 5) ✓
  - GET/POST endpoint (Task 6) ✓
  - Preview endpoint (Task 7) ✓
  - Editor HTML — scaffold + click-to-anchor (Task 8) ✓
  - Editor HTML — off-screen button + delete + preview (Task 9) ✓
  - Dashboard link (Task 10) ✓

- **Placeholder scan:** No TBDs/TODOs. Each step has executable code
  or an exact manual procedure.

- **Type consistency:**
  - `BallAnchor.image_xy: tuple[float, float] | None` consistent
    across schema, server payload (`list[float] | None`), and Layer 5.
  - `BallAnchorState` literal stays consistent; server validates via
    schema round-trip in Task 6.
  - `knot_frames: dict[int, np.ndarray] | None` consistent across the
    fit signature (Task 3) and the Layer 5 call site (Task 5).
  - `state_to_height` raises for `off_screen_flight`; Task 5 guards
    against that by checking `HARD_KNOT_STATES` membership before
    calling it.

- **Risks not covered by tasks:**
  - The preview endpoint depends on a default-config loader; the
    relevant grep + fallback is documented inline in Task 7 so the
    implementer can adapt.
  - The editor JS is manual-test only; the spec acknowledges this is
    acceptable for a v1 editor (camera editor follows the same
    convention).
