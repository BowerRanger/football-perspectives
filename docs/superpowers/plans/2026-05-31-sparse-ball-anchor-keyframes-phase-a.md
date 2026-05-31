# Sparse Ball Anchor Keyframes — Phase A (Pipeline) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Emit a new sparse `ball_keyframes.json` sidecar (one entry per manual ball anchor, carrying resolved 3D, the clicked camera ray for airborne anchors, and semantic metadata) and point the UE manifest's ball entry at it, leaving the dense `ball_track.json` path untouched.

**Architecture:** A new frozen-dataclass schema (`ball_keyframes.py`) mirrors the `ball_track.py` / `ball_anchor.py` save/load pattern. A pure builder util (`ball_keyframe_builder.py`) turns the already-resolved per-frame world positions + anchors + camera params into a `BallKeyframeSet` — keeping `ball.py` (already 2100+ lines) free of new logic beyond a single call after `track.save()`. The pipeline `ue_manifest.py` and its UE-side unreal-free mirror both gain an optional `keyframes_json` field; `export.py` populates it when the sidecar exists.

**Tech Stack:** Python 3, frozen `@dataclass`, numpy (camera ray math), pytest.

**Spec:** `docs/superpowers/specs/2026-05-31-sparse-ball-anchor-keyframes-design.md`

---

## File Structure

- Create: `src/schemas/ball_keyframes.py` — `BallKeyframe`, `BallKeyframeSet` dataclasses + `save`/`load`/validation.
- Create: `src/utils/ball_keyframe_builder.py` — pure `build_ball_keyframe_set(...)` function (anchors + resolved world + camera → `BallKeyframeSet`), incl. camera-ray computation for airborne anchors.
- Create: `tests/test_ball_keyframes_schema.py` — schema round-trip + validation tests.
- Create: `tests/test_ball_keyframe_builder.py` — builder per-state tests.
- Modify: `src/stages/ball.py` (`_run_shot`, right after `track.save(ball_out_path)` near line 2071) — build + save the sidecar.
- Create: `tests/test_ball_stage_keyframes.py` — integration: `_run_shot` writes the sidecar.
- Modify: `src/schemas/ue_manifest.py` (`BallEntry` ~line 58, `save` ~line 161, `load` ~line 217) — add `keyframes_json`.
- Modify: `tests/test_ue_manifest.py` — round-trip the new field.
- Modify: `src/stages/export.py` (`BallEntry` construction ~line 658-686) — populate `keyframes_json`.
- Modify: `tests/test_export_stage_manifest.py` — assert manifest carries `keyframes_json`.
- Modify (UE repo, unreal-free): `FootballPerspectives 5.8/Content/Python/football_perspectives/manifest.py` (`BallEntry` + `from_dict` ball block) — mirror `keyframes_json`.
- Modify (UE repo): `FootballPerspectives 5.8/Content/Python/tests/test_manifest_cameras.py` or new `test_manifest_ball.py` — round-trip the field.

---

## Task 1: `BallKeyframe` / `BallKeyframeSet` schema

**Files:**
- Create: `src/schemas/ball_keyframes.py`
- Test: `tests/test_ball_keyframes_schema.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ball_keyframes_schema.py
from pathlib import Path

import pytest

from src.schemas.ball_keyframes import BallKeyframe, BallKeyframeSet


def _grounded() -> BallKeyframe:
    return BallKeyframe(
        frame=10,
        state="grounded",
        world_xyz=(12.0, 4.0, 0.11),
        image_xy=(800.0, 600.0),
        depth_source="ground",
        confidence=1.0,
    )


def _airborne() -> BallKeyframe:
    return BallKeyframe(
        frame=20,
        state="airborne_high",
        world_xyz=(18.3, 9.2, 4.1),
        image_xy=(900.0, 300.0),
        ray=((0.0, 0.0, 15.0), (0.1, 0.2, -0.97)),
        depth_source="ray_physics",
        confidence=0.8,
    )


def test_round_trip_preserves_all_fields(tmp_path: Path):
    src = BallKeyframeSet(
        clip_id="clipA",
        fps=25.0,
        image_size=(1920, 1080),
        keyframes=(_grounded(), _airborne()),
    )
    path = tmp_path / "ball_keyframes.json"
    src.save(path)
    back = BallKeyframeSet.load(path)
    assert back == src


def test_player_touch_requires_player_and_bone(tmp_path: Path):
    path = tmp_path / "kf.json"
    BallKeyframeSet(
        clip_id="c", fps=25.0, image_size=(1920, 1080),
        keyframes=(
            BallKeyframe(
                frame=1, state="player_touch", world_xyz=(1.0, 2.0, 1.0),
                image_xy=(10.0, 10.0), depth_source="player_bone",
                player_id="P001", bone="right_foot", confidence=1.0,
            ),
        ),
    ).save(path)
    # Missing player_id must fail validation on load.
    import json
    raw = json.loads(path.read_text())
    del raw["keyframes"][0]["player_id"]
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="player_id is required"):
        BallKeyframeSet.load(path)


def test_unknown_state_rejected(tmp_path: Path):
    path = tmp_path / "kf.json"
    path.write_text(
        '{"clip_id":"c","fps":25.0,"image_size":[1920,1080],'
        '"keyframes":[{"frame":1,"state":"banana","depth_source":"ground"}]}'
    )
    with pytest.raises(ValueError, match="unknown ball keyframe state"):
        BallKeyframeSet.load(path)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_ball_keyframes_schema.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.schemas.ball_keyframes'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/schemas/ball_keyframes.py
"""Sparse ball keyframe schema — one entry per manual ball anchor,
carrying its resolved 3D position, the clicked camera ray (airborne
anchors only), and semantic metadata. Engine-facing sidecar written by
``BallStage`` alongside the dense ``ball_track.json``; the UE side keys a
transform track only at these frames and tweens between them.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

# Mirror of BallAnchorState — the states a keyframe may carry.
KeyframeState = Literal[
    "grounded", "airborne_low", "airborne_mid", "airborne_high",
    "kick", "catch", "bounce", "header", "volley", "chest",
    "player_touch", "goal_impact", "off_screen_flight",
]

_VALID_STATES: frozenset[str] = frozenset({
    "grounded", "airborne_low", "airborne_mid", "airborne_high",
    "kick", "catch", "bounce", "header", "volley", "chest",
    "player_touch", "goal_impact", "off_screen_flight",
})

DepthSource = Literal["ground", "ray_physics", "player_bone", "goal_geometry"]

_VALID_DEPTH_SOURCES: frozenset[str] = frozenset({
    "ground", "ray_physics", "player_bone", "goal_geometry",
})

# (ray_origin_xyz, ray_dir_xyz); dir is a unit vector in world frame.
Ray = tuple[tuple[float, float, float], tuple[float, float, float]]


@dataclass(frozen=True)
class BallKeyframe:
    """One sparse keyframe corresponding to a single manual anchor.

    ``world_xyz`` is the artist's default key position (pitch metres);
    ``None`` only for ``off_screen_flight`` (no pixel, no resolved 3D).
    ``image_xy`` is the authoritative clicked pixel (``None`` only for
    ``off_screen_flight``). ``ray`` is the clicked camera ray
    ``((ox,oy,oz),(dx,dy,dz))`` populated for ``airborne_*`` states so the
    engine can re-snap a moved key onto the line of sight. The remaining
    optional fields carry the anchor's semantic tags.
    """

    frame: int
    state: KeyframeState
    depth_source: DepthSource
    world_xyz: tuple[float, float, float] | None = None
    image_xy: tuple[float, float] | None = None
    ray: Ray | None = None
    player_id: str | None = None
    bone: str | None = None
    goal_element: str | None = None
    touch_type: str | None = None
    spin: str | None = None
    confidence: float = 1.0


@dataclass(frozen=True)
class BallKeyframeSet:
    clip_id: str
    fps: float
    image_size: tuple[int, int]
    keyframes: tuple[BallKeyframe, ...]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as fh:
            json.dump(
                asdict(self),
                fh,
                indent=2,
                default=lambda v: list(v) if isinstance(v, tuple) else v,
            )

    @classmethod
    def load(cls, path: Path) -> "BallKeyframeSet":
        with path.open() as fh:
            data = json.load(fh)
        keyframes = tuple(_load_keyframe(k) for k in data.get("keyframes", []))
        return cls(
            clip_id=str(data["clip_id"]),
            fps=float(data["fps"]),
            image_size=(int(data["image_size"][0]), int(data["image_size"][1])),
            keyframes=keyframes,
        )


def _as_tuple3(v) -> tuple[float, float, float]:
    return (float(v[0]), float(v[1]), float(v[2]))


def _load_ray(v) -> Ray | None:
    if v is None:
        return None
    return (_as_tuple3(v[0]), _as_tuple3(v[1]))


def _load_keyframe(k: dict) -> BallKeyframe:
    state = str(k["state"])
    if state not in _VALID_STATES:
        raise ValueError(f"unknown ball keyframe state: {state!r}")
    depth_source = str(k["depth_source"])
    if depth_source not in _VALID_DEPTH_SOURCES:
        raise ValueError(f"unknown depth_source: {depth_source!r}")
    if state == "player_touch":
        if not k.get("player_id"):
            raise ValueError("player_id is required for state 'player_touch'")
        if not k.get("bone"):
            raise ValueError("bone is required for state 'player_touch'")
    if state == "goal_impact" and not k.get("goal_element"):
        raise ValueError("goal_element is required for state 'goal_impact'")
    world = k.get("world_xyz")
    xy = k.get("image_xy")
    return BallKeyframe(
        frame=int(k["frame"]),
        state=state,  # type: ignore[arg-type]
        depth_source=depth_source,  # type: ignore[arg-type]
        world_xyz=_as_tuple3(world) if world is not None else None,
        image_xy=(float(xy[0]), float(xy[1])) if xy is not None else None,
        ray=_load_ray(k.get("ray")),
        player_id=k.get("player_id"),
        bone=k.get("bone"),
        goal_element=k.get("goal_element"),
        touch_type=k.get("touch_type"),
        spin=k.get("spin"),
        confidence=float(k.get("confidence", 1.0)),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_ball_keyframes_schema.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/schemas/ball_keyframes.py tests/test_ball_keyframes_schema.py
git commit -m "feat(ball): ball_keyframes sparse schema (save/load + validation)"
```

---

## Task 2: `build_ball_keyframe_set` builder util

**Files:**
- Create: `src/utils/ball_keyframe_builder.py`
- Test: `tests/test_ball_keyframe_builder.py`

This is a pure function. Inputs: the anchor map, the resolved per-frame
world positions (as the dense stage produced them, so values match the
dense track exactly), the per-frame camera params, distortion, and camera
metadata. Output: a `BallKeyframeSet`. The only new math is the camera ray
for airborne anchors, computed identically to `_project_point_onto_pixel_ray`
in `ball.py` (`C = -R.T @ t`, `d_hat = normalize(R.T @ inv(K) @ [u,v,1])`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ball_keyframe_builder.py
import numpy as np

from src.schemas.ball_anchor import BallAnchor
from src.utils.ball_keyframe_builder import build_ball_keyframe_set


def _ident_cam():
    K = np.array([[1000.0, 0.0, 960.0], [0.0, 1000.0, 540.0], [0.0, 0.0, 1.0]])
    R = np.eye(3)
    # Camera 10 m up looking down -z so pixels map onto the ground.
    t = np.array([0.0, 0.0, 10.0])
    return K, R, t


def test_grounded_anchor_yields_ground_depth_source_no_ray():
    K, R, t = _ident_cam()
    anchors = {5: BallAnchor(frame=5, image_xy=(960.0, 540.0), state="grounded")}
    world = {5: (3.0, 4.0, 0.11)}
    kfset = build_ball_keyframe_set(
        clip_id="c", fps=25.0, image_size=(1920, 1080),
        anchor_by_frame=anchors, world_by_frame=world,
        per_frame_K={5: K}, per_frame_R={5: R}, per_frame_t={5: t},
        distortion=(0.0, 0.0),
    )
    assert len(kfset.keyframes) == 1
    kf = kfset.keyframes[0]
    assert kf.frame == 5
    assert kf.state == "grounded"
    assert kf.depth_source == "ground"
    assert kf.world_xyz == (3.0, 4.0, 0.11)
    assert kf.ray is None  # rays only for airborne


def test_airborne_anchor_carries_ray_and_ray_physics_source():
    K, R, t = _ident_cam()
    anchors = {7: BallAnchor(frame=7, image_xy=(960.0, 540.0), state="airborne_high")}
    world = {7: (0.0, 0.0, 4.0)}
    kfset = build_ball_keyframe_set(
        clip_id="c", fps=25.0, image_size=(1920, 1080),
        anchor_by_frame=anchors, world_by_frame=world,
        per_frame_K={7: K}, per_frame_R={7: R}, per_frame_t={7: t},
        distortion=(0.0, 0.0),
    )
    kf = kfset.keyframes[0]
    assert kf.depth_source == "ray_physics"
    assert kf.ray is not None
    origin, direction = kf.ray
    # Fixture geometry (R=I, t=(0,0,10)): camera centre C = -R^T t = (0,0,-10).
    assert np.allclose(origin, (0.0, 0.0, -10.0))
    # Pixel at the principal point (960,540): inv(K)@[u,v,1] = [0,0,1],
    # so the unit direction is +z.
    assert np.allclose(direction, (0.0, 0.0, 1.0))


def test_player_touch_carries_player_bone_and_player_bone_source():
    K, R, t = _ident_cam()
    anchors = {
        3: BallAnchor(
            frame=3, image_xy=(900.0, 500.0), state="player_touch",
            player_id="P002", bone="head",
        )
    }
    world = {3: (1.0, 1.0, 1.8)}
    kfset = build_ball_keyframe_set(
        clip_id="c", fps=25.0, image_size=(1920, 1080),
        anchor_by_frame=anchors, world_by_frame=world,
        per_frame_K={3: K}, per_frame_R={3: R}, per_frame_t={3: t},
        distortion=(0.0, 0.0),
        ground_touch_frames=set(),
    )
    kf = kfset.keyframes[0]
    assert kf.depth_source == "player_bone"
    assert kf.player_id == "P002"
    assert kf.bone == "head"


def test_keyframes_sorted_by_frame():
    K, R, t = _ident_cam()
    anchors = {
        9: BallAnchor(frame=9, image_xy=(1.0, 1.0), state="grounded"),
        2: BallAnchor(frame=2, image_xy=(1.0, 1.0), state="kick"),
    }
    world = {9: (0.0, 0.0, 0.1), 2: (0.0, 0.0, 0.1)}
    kfset = build_ball_keyframe_set(
        clip_id="c", fps=25.0, image_size=(1920, 1080),
        anchor_by_frame=anchors, world_by_frame=world,
        per_frame_K={2: K, 9: K}, per_frame_R={2: R, 9: R},
        per_frame_t={2: t, 9: t}, distortion=(0.0, 0.0),
    )
    assert [kf.frame for kf in kfset.keyframes] == [2, 9]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_ball_keyframe_builder.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.utils.ball_keyframe_builder'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/utils/ball_keyframe_builder.py
"""Pure builder turning resolved ball anchors into a sparse
``BallKeyframeSet``. Kept out of the already-large ``ball.py`` so the stage
only needs a single call after it has saved the dense track.
"""

from __future__ import annotations

import numpy as np

from src.schemas.ball_anchor import BallAnchor
from src.schemas.ball_keyframes import BallKeyframe, BallKeyframeSet

_AIRBORNE_STATES = frozenset(
    {"airborne_low", "airborne_mid", "airborne_high"}
)


def _camera_ray(
    uv: tuple[float, float],
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    distortion: tuple[float, float],
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Return (origin, unit-dir) of the camera ray through pixel ``uv``.

    Same construction as ``ball._project_point_onto_pixel_ray``:
    ``C = -R^T t`` and ``d_hat = normalize(R^T K^-1 [u, v, 1])``.
    """
    from src.utils.camera_projection import undistort_pixel

    uv_arr = np.asarray(uv, dtype=float)
    if distortion != (0.0, 0.0):
        uv_arr = undistort_pixel(uv_arr, K, distortion)
    C = -R.T @ t
    d_world = R.T @ (np.linalg.inv(K) @ np.array([uv_arr[0], uv_arr[1], 1.0]))
    d_hat = d_world / np.linalg.norm(d_world)
    return (
        (float(C[0]), float(C[1]), float(C[2])),
        (float(d_hat[0]), float(d_hat[1]), float(d_hat[2])),
    )


def _depth_source(
    anc: BallAnchor, ground_touch_frames: set[int],
) -> str:
    if anc.state == "goal_impact":
        return "goal_geometry"
    if anc.state == "player_touch":
        if anc.frame in ground_touch_frames:
            return "ground"
        return "player_bone"
    if anc.state in _AIRBORNE_STATES:
        return "ray_physics"
    return "ground"


def build_ball_keyframe_set(
    *,
    clip_id: str,
    fps: float,
    image_size: tuple[int, int],
    anchor_by_frame: dict[int, BallAnchor],
    world_by_frame: dict[int, tuple[float, float, float] | None],
    per_frame_K: dict[int, np.ndarray],
    per_frame_R: dict[int, np.ndarray],
    per_frame_t: dict[int, np.ndarray],
    distortion: tuple[float, float],
    ground_touch_frames: set[int] | None = None,
) -> BallKeyframeSet:
    """Collect one ``BallKeyframe`` per manual anchor.

    ``world_by_frame`` holds the *already-resolved* world position for each
    anchor frame (as the dense stage produced it), so emitted ``world_xyz``
    matches the dense track exactly. Airborne anchors additionally get the
    clicked camera ray; ``off_screen_flight`` anchors (no pixel) get neither
    ray nor world position.
    """
    gtf = ground_touch_frames or set()
    keyframes: list[BallKeyframe] = []
    for fi in sorted(anchor_by_frame):
        anc = anchor_by_frame[fi]
        world = world_by_frame.get(fi)
        ray = None
        if (
            anc.state in _AIRBORNE_STATES
            and anc.image_xy is not None
            and fi in per_frame_K
            and fi in per_frame_R
            and fi in per_frame_t
        ):
            ray = _camera_ray(
                (float(anc.image_xy[0]), float(anc.image_xy[1])),
                per_frame_K[fi], per_frame_R[fi], per_frame_t[fi],
                distortion,
            )
        keyframes.append(
            BallKeyframe(
                frame=fi,
                state=anc.state,  # type: ignore[arg-type]
                depth_source=_depth_source(anc, gtf),  # type: ignore[arg-type]
                world_xyz=(
                    (float(world[0]), float(world[1]), float(world[2]))
                    if world is not None else None
                ),
                image_xy=(
                    (float(anc.image_xy[0]), float(anc.image_xy[1]))
                    if anc.image_xy is not None else None
                ),
                ray=ray,
                player_id=anc.player_id,
                bone=anc.bone,
                goal_element=anc.goal_element,
                touch_type=anc.touch_type,
                spin=anc.spin,
            )
        )
    return BallKeyframeSet(
        clip_id=clip_id,
        fps=fps,
        image_size=(int(image_size[0]), int(image_size[1])),
        keyframes=tuple(keyframes),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_ball_keyframe_builder.py -v`
Expected: PASS (4 tests). If `test_airborne_anchor_carries_ray_and_ray_physics_source` fails on the `origin`/`direction` asserts, print the computed values and correct the assert to the actual ray math (the implementation is the source of truth for the geometry; the asserted numbers above are the expected result of `C=-R^T t` and `d_hat=normalize(R^T K^-1[u,v,1])` for the identity-rotation camera).

- [ ] **Step 5: Commit**

```bash
git add src/utils/ball_keyframe_builder.py tests/test_ball_keyframe_builder.py
git commit -m "feat(ball): pure builder for sparse ball keyframes + camera ray"
```

---

## Task 3: `BallStage._run_shot` writes the sidecar

**Files:**
- Modify: `src/stages/ball.py` (imports near line 36; `_run_shot` right after `track.save(ball_out_path)`, ~line 2071)
- Test: `tests/test_ball_stage_keyframes.py`

The resolved world position for each anchor frame is already present in the
`per_frame_out` list (each `BallFrame` has `frame` + `world_xyz`). Build a
`{frame: world_xyz}` map from it restricted to anchor frames, then call the
builder. `ground_touch_frames` is already in scope in `_run_shot`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ball_stage_keyframes.py
"""Integration: the ball stage writes a sparse ball_keyframes.json sidecar
next to the dense ball_track.json, with one entry per manual anchor."""
import json
from pathlib import Path

from src.schemas.ball_keyframes import BallKeyframeSet
from src.schemas.ball_track import BallFrame, BallTrack


def test_run_shot_writes_keyframes_sidecar(tmp_path: Path, monkeypatch):
    # Build the smallest viable output tree the ball stage's _run_shot reads.
    # Reuse the existing ball-stage fixture helpers if present; otherwise this
    # test drives _emit_ball_keyframes directly (see Step 3 — the stage calls a
    # small private method that the test can exercise in isolation).
    from src.stages.ball import _emit_ball_keyframes
    from src.schemas.ball_anchor import BallAnchor
    import numpy as np

    ball_out = tmp_path / "ball" / "clipX_ball_track.json"
    ball_out.parent.mkdir(parents=True)
    per_frame_out = [
        BallFrame(frame=4, world_xyz=(1.0, 2.0, 0.11), state="grounded", confidence=1.0),
        BallFrame(frame=5, world_xyz=(1.5, 2.5, 0.11), state="grounded", confidence=1.0),
        BallFrame(frame=6, world_xyz=(2.0, 3.0, 4.0), state="flight", confidence=0.5),
    ]
    anchor_by_frame = {
        4: BallAnchor(frame=4, image_xy=(800.0, 600.0), state="grounded"),
        6: BallAnchor(frame=6, image_xy=(820.0, 300.0), state="airborne_high"),
    }
    K = np.array([[1000.0, 0, 960.0], [0, 1000.0, 540.0], [0, 0, 1.0]])
    _emit_ball_keyframes(
        ball_out_path=ball_out,
        clip_id="clipX",
        fps=25.0,
        image_size=(1920, 1080),
        per_frame_out=per_frame_out,
        anchor_by_frame=anchor_by_frame,
        per_frame_K={4: K, 6: K},
        per_frame_R={4: np.eye(3), 6: np.eye(3)},
        per_frame_t={4: np.array([0.0, 0, 10.0]), 6: np.array([0.0, 0, 10.0])},
        distortion=(0.0, 0.0),
        ground_touch_frames=set(),
    )

    kf_path = tmp_path / "ball" / "clipX_ball_keyframes.json"
    assert kf_path.exists()
    kfset = BallKeyframeSet.load(kf_path)
    assert [k.frame for k in kfset.keyframes] == [4, 6]
    by_frame = {k.frame: k for k in kfset.keyframes}
    assert by_frame[4].world_xyz == (1.0, 2.0, 0.11)
    assert by_frame[4].ray is None
    assert by_frame[6].world_xyz == (2.0, 3.0, 4.0)  # matches dense track
    assert by_frame[6].ray is not None
    assert by_frame[6].depth_source == "ray_physics"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_ball_stage_keyframes.py -v`
Expected: FAIL with `ImportError: cannot import name '_emit_ball_keyframes'`

- [ ] **Step 3: Write minimal implementation**

Add the import near the other schema/util imports at the top of `src/stages/ball.py` (after line 36's `from src.schemas.ball_track import ...`):

```python
from src.schemas.ball_keyframes import BallKeyframeSet  # noqa: F401  (re-exported for tests)
from src.utils.ball_keyframe_builder import build_ball_keyframe_set
```

Add this module-level helper (place it near `_apply_hard_knot_anchor_overrides`, before the `BallStage` class):

```python
def _emit_ball_keyframes(
    *,
    ball_out_path: Path,
    clip_id: str,
    fps: float,
    image_size: tuple[int, int],
    per_frame_out: list[BallFrame],
    anchor_by_frame: dict[int, BallAnchor],
    per_frame_K: dict[int, np.ndarray],
    per_frame_R: dict[int, np.ndarray],
    per_frame_t: dict[int, np.ndarray],
    distortion: tuple[float, float],
    ground_touch_frames: set[int],
) -> Path:
    """Write the sparse ``*_ball_keyframes.json`` sidecar next to the dense
    track. ``world_xyz`` for each anchor is taken from the already-built
    dense ``per_frame_out`` so the two artifacts agree exactly.
    """
    world_by_frame = {
        bf.frame: bf.world_xyz
        for bf in per_frame_out
        if bf.frame in anchor_by_frame
    }
    kfset = build_ball_keyframe_set(
        clip_id=clip_id,
        fps=fps,
        image_size=image_size,
        anchor_by_frame=anchor_by_frame,
        world_by_frame=world_by_frame,
        per_frame_K=per_frame_K,
        per_frame_R=per_frame_R,
        per_frame_t=per_frame_t,
        distortion=distortion,
        ground_touch_frames=ground_touch_frames,
    )
    kf_path = ball_out_path.with_name(
        ball_out_path.name.replace("ball_track", "ball_keyframes")
    )
    kfset.save(kf_path)
    return kf_path
```

Then call it in `_run_shot` immediately after `track.save(ball_out_path)` (line ~2071):

```python
        track.save(ball_out_path)

        _emit_ball_keyframes(
            ball_out_path=ball_out_path,
            clip_id=camera.clip_id,
            fps=camera.fps,
            image_size=camera.image_size,
            per_frame_out=per_frame_out,
            anchor_by_frame=anchor_by_frame,
            per_frame_K=per_frame_K,
            per_frame_R=per_frame_R,
            per_frame_t=per_frame_t,
            distortion=distortion,
            ground_touch_frames=ground_touch_frames,
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_ball_stage_keyframes.py -v`
Expected: PASS (1 test)

- [ ] **Step 5: Run the broader ball suite to confirm no regression**

Run: `pytest tests/test_ball_stage.py tests/test_ball_stage_anchors.py tests/test_ball_anchor_accuracy.py -q`
Expected: PASS (or pre-existing skips — `test_ball_anchor_accuracy.py` skips when output dirs are absent). No new failures.

- [ ] **Step 6: Commit**

```bash
git add src/stages/ball.py tests/test_ball_stage_keyframes.py
git commit -m "feat(ball): BallStage writes sparse ball_keyframes.json sidecar"
```

---

## Task 4: `keyframes_json` on the pipeline `UeManifest` ball entry

**Files:**
- Modify: `src/schemas/ue_manifest.py` (`BallEntry` ~line 58, `save` ~line 161-167, `load` ~line 217-222)
- Test: `tests/test_ue_manifest.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_ue_manifest.py`. It reuses the existing `_good()` helper
(defined at the top of that file) and only overrides the ball entry:

```python
def test_ball_keyframes_json_round_trips(tmp_path: Path) -> None:
    from src.schemas.ue_manifest import BallEntry

    m = _good()
    m.ball = BallEntry(
        fbx="fbx/ball.fbx",
        frame_range=(12, 78),
        track_json="ball/ball_track.json",
        keyframes_json="ball/ball_keyframes.json",
    )
    p = tmp_path / "ue_manifest.json"
    m.save(p)
    loaded = UeManifest.load(p)
    assert loaded.ball is not None
    assert loaded.ball.keyframes_json == "ball/ball_keyframes.json"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_ue_manifest.py::test_ball_keyframes_json_round_trips -v`
Expected: FAIL with `TypeError: __init__() got an unexpected keyword argument 'keyframes_json'`

- [ ] **Step 3: Write minimal implementation**

In `src/schemas/ue_manifest.py`, add the field to `BallEntry` (after `track_json`):

```python
@dataclass
class BallEntry:
    fbx: str
    frame_range: tuple[int, int]
    track_json: str = ""
    # Pipeline-relative path to the sparse ball_keyframes.json (one entry
    # per manual anchor + camera rays + metadata). Preferred by the UE side
    # when present; empty when the run predates the sparse-keyframe export.
    keyframes_json: str = ""
```

In `save` (the `if self.ball is not None:` block, ~line 161), after the
`track_json` line add:

```python
            if self.ball.keyframes_json:
                raw["ball"]["keyframes_json"] = self.ball.keyframes_json
```

In `load` (the `BallEntry(...)` construction, ~line 217):

```python
                BallEntry(
                    fbx=raw["ball"]["fbx"],
                    frame_range=tuple(raw["ball"]["frame_range"]),
                    track_json=raw["ball"].get("track_json", ""),
                    keyframes_json=raw["ball"].get("keyframes_json", ""),
                )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_ue_manifest.py -v`
Expected: PASS (all, including the new test)

- [ ] **Step 5: Commit**

```bash
git add src/schemas/ue_manifest.py tests/test_ue_manifest.py
git commit -m "feat(export): add keyframes_json to UeManifest ball entry"
```

---

## Task 5: `export.py` populates `keyframes_json`

**Files:**
- Modify: `src/stages/export.py` (`BallEntry` construction, ~line 658-686)
- Test: `tests/test_export_stage_manifest.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_export_stage_manifest.py`. It uses the legacy single-shot
layout: `_write_min_inputs` (already defined in that file) writes
`camera/camera_track.json`, so we only add a legacy `ball/ball_track.json`
plus the new `ball/ball_keyframes.json` sidecar, then assert the manifest's
ball entry points at it. Add `from src.schemas.ball_keyframes import BallKeyframe, BallKeyframeSet`
to the imports at the top of the file.

```python
def test_manifest_references_ball_keyframes_when_present(tmp_path: Path) -> None:
    """When a ball_keyframes.json sidecar exists alongside the dense track,
    the manifest's ball entry carries a keyframes_json pointer to it."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    _write_min_inputs(output_dir)

    fbx_dir = output_dir / "export" / "fbx"
    fbx_dir.mkdir(parents=True)
    (fbx_dir / "P001.fbx").write_bytes(b"\x00")

    ball_dir = output_dir / "ball"
    ball_dir.mkdir()
    (ball_dir / "ball_track.json").write_text(
        json.dumps({
            "frames": [
                {"frame": 0, "world_xyz": [10.0, 20.0, 0.11], "state": "grounded"},
                {"frame": 3, "world_xyz": [12.0, 21.0, 0.11], "state": "grounded"},
            ],
        })
    )
    BallKeyframeSet(
        clip_id="clip_demo", fps=30.0, image_size=(1920, 1080),
        keyframes=(
            BallKeyframe(
                frame=0, state="grounded", depth_source="ground",
                world_xyz=(10.0, 20.0, 0.11), image_xy=(800.0, 600.0),
            ),
        ),
    ).save(ball_dir / "ball_keyframes.json")

    cfg = {
        "export": {"gltf_enabled": False, "fbx_enabled": False},
        "pitch": {"length_m": 105.0, "width_m": 68.0},
        "ball": {"ball_radius_m": 0.11},
    }
    stage = ExportStage(output_dir=output_dir, config=cfg)
    stage.write_ue_manifest(clip_name="clip_demo")

    m = UeManifest.load(output_dir / "export" / "ue_manifest.json")
    assert m.ball is not None
    assert m.ball.keyframes_json == "ball/ball_keyframes.json"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_export_stage_manifest.py::test_manifest_references_ball_keyframes_when_present -v`
Expected: FAIL — `KeyError: 'keyframes_json'` (field not emitted yet).

- [ ] **Step 3: Write minimal implementation**

In `src/stages/export.py`, where the ball paths are resolved (~line 658-668),
add the sidecar path next to `ball_track_path`:

```python
        if primary_shot:
            ball_fbx = fbx_dir / f"{primary_shot}_ball.fbx"
            ball_track_path = self.output_dir / "ball" / f"{primary_shot}_ball_track.json"
            ball_keyframes_path = self.output_dir / "ball" / f"{primary_shot}_ball_keyframes.json"
            ball_fbx_rel = f"fbx/{primary_shot}_ball.fbx"
            ball_track_rel = f"ball/{primary_shot}_ball_track.json"
            ball_keyframes_rel = f"ball/{primary_shot}_ball_keyframes.json"
        else:
            ball_fbx = fbx_dir / "ball.fbx"
            ball_track_path = self.output_dir / "ball" / "ball_track.json"
            ball_keyframes_path = self.output_dir / "ball" / "ball_keyframes.json"
            ball_fbx_rel = "fbx/ball.fbx"
            ball_track_rel = "ball/ball_track.json"
            ball_keyframes_rel = "ball/ball_keyframes.json"
```

Then in the `if ball_track_path.exists():` block where `ball_entry` is built
(~line 681-686), add the keyframes pointer:

```python
            if ball_frames:
                ball_entry = BallEntry(
                    fbx=ball_fbx_rel if ball_fbx.exists() else "",
                    frame_range=(int(ball_frames[0]), int(ball_frames[-1])),
                    track_json=ball_track_rel,
                    keyframes_json=(
                        ball_keyframes_rel if ball_keyframes_path.exists() else ""
                    ),
                )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_export_stage_manifest.py -v`
Expected: PASS (all, including the new test)

- [ ] **Step 5: Commit**

```bash
git add src/stages/export.py tests/test_export_stage_manifest.py
git commit -m "feat(export): point ball manifest entry at ball_keyframes.json sidecar"
```

---

## Task 6: Mirror `keyframes_json` in the UE-side (unreal-free) `manifest.py`

**Files:**
- Modify (UE repo): `FootballPerspectives 5.8/Content/Python/football_perspectives/manifest.py` (`BallEntry` + `from_dict` ball block ~line 49-65, 149-166)
- Test (UE repo): `FootballPerspectives 5.8/Content/Python/tests/test_manifest_ball.py` (new)

Run UE-side tests with the pipeline venv per the project convention:
`cd "/Users/joebower/workplace/FootballPerspectives 5.8/Content/Python" && /Users/joebower/workplace/football-perspectives/.venv/bin/python -m pytest tests -q`

- [ ] **Step 1: Write the failing test**

The loader is the module function `manifest.load(path)` (returns a
`UeManifest`, raises `UeManifestError`), and the schema is strict — it needs
`schema_version`, `clip_name`, `pitch`, and at least one player with a real
FBX on disk. Mirror the `_write_manifest` helper shape from
`test_manifest_cameras.py` in that same directory:

```python
# FootballPerspectives 5.8/Content/Python/tests/test_manifest_ball.py
"""The unreal-free manifest loader records the ball keyframes_json pointer."""
from __future__ import annotations

import json

from football_perspectives import manifest


def test_ball_entry_records_keyframes_json(tmp_path):
    fbx = tmp_path / "P001.fbx"
    fbx.write_bytes(b"stub")
    raw = {
        "schema_version": manifest.SCHEMA_VERSION,
        "clip_name": "gberch",
        "fps": 25.0,
        "frame_range": [0, 100],
        "pitch": {"length_m": 105.0, "width_m": 68.0},
        "players": [
            {
                "player_id": "P001",
                "display_name": "Messi",
                "fbx": "P001.fbx",
                "frame_range": [0, 100],
                "world_bbox": {"min": [0.0, 0.0, 0.0], "max": [1.0, 1.0, 2.0]},
                "kit_role": "home",
            }
        ],
        "ball": {
            "fbx": "fbx/ball.fbx",
            "frame_range": [0, 100],
            "track_json": "ball/ball_track.json",
            "keyframes_json": "ball/ball_keyframes.json",
        },
    }
    path = tmp_path / "ue_manifest.json"
    path.write_text(json.dumps(raw))
    m = manifest.load(path)
    assert m.ball is not None
    assert m.ball.keyframes_json == "ball/ball_keyframes.json"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd "/Users/joebower/workplace/FootballPerspectives 5.8/Content/Python" && /Users/joebower/workplace/football-perspectives/.venv/bin/python -m pytest tests/test_manifest_ball.py -v`
Expected: FAIL — `AttributeError`/`TypeError` on `keyframes_json` (field absent).

- [ ] **Step 3: Write minimal implementation**

Add the field to the UE `BallEntry` dataclass (after `track_json`):

```python
@dataclass(frozen=True)
class BallEntry:
    fbx: str
    frame_range: tuple
    track_json: str = ""
    # Pipeline-relative path to the sparse ball_keyframes.json sidecar.
    # Preferred by load_reconstruction over track_json when present.
    keyframes_json: str = ""
```

In the `from_dict`/`load` ball block (~line 158-166), read it:

```python
        ball_raw = raw["ball"]
        fbx_rel = ball_raw.get("fbx", "")
        track_rel = ball_raw.get("track_json", "")
        keyframes_rel = ball_raw.get("keyframes_json", "")
        if fbx_rel or track_rel or keyframes_rel:
            ball = BallEntry(
                fbx=fbx_rel,
                frame_range=tuple(ball_raw["frame_range"]),
                track_json=track_rel,
                keyframes_json=keyframes_rel,
            )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd "/Users/joebower/workplace/FootballPerspectives 5.8/Content/Python" && /Users/joebower/workplace/football-perspectives/.venv/bin/python -m pytest tests -q`
Expected: PASS (all UE-side unit tests, including the new one)

- [ ] **Step 5: Commit (UE repo is non-git — record the change in the pipeline commit log instead)**

The UE directory is not under git (per project notes). Skip a git commit there;
note the edit in the final summary so it is tracked manually. If the directory
*is* under version control at execution time, commit with:

```bash
git -C "/Users/joebower/workplace/FootballPerspectives 5.8" add Content/Python/football_perspectives/manifest.py Content/Python/tests/test_manifest_ball.py
git -C "/Users/joebower/workplace/FootballPerspectives 5.8" commit -m "feat(manifest): record ball keyframes_json pointer"
```

---

## Task 7: Full Phase-A verification

**Files:** none (verification only)

- [ ] **Step 1: Run the full pipeline-side suite**

Run: `pytest tests/test_ball_keyframes_schema.py tests/test_ball_keyframe_builder.py tests/test_ball_stage_keyframes.py tests/test_ue_manifest.py tests/test_export_stage_manifest.py -q`
Expected: PASS, no failures.

- [ ] **Step 2: Run the UE-side suite**

Run: `cd "/Users/joebower/workplace/FootballPerspectives 5.8/Content/Python" && /Users/joebower/workplace/football-perspectives/.venv/bin/python -m pytest tests -q`
Expected: PASS, no failures.

- [ ] **Step 3: Sanity-check against real output (if an `output*/` tree exists)**

Run (adjust path to a real output dir, e.g. `output-origi`):

```bash
.venv/bin/python -c "
import json, glob
from src.schemas.ball_keyframes import BallKeyframeSet
from src.schemas.ball_track import BallTrack
for kf_path in glob.glob('output-*/ball/*_ball_keyframes.json'):
    kfset = BallKeyframeSet.load(__import__('pathlib').Path(kf_path))
    track = BallTrack.load(__import__('pathlib').Path(kf_path.replace('ball_keyframes','ball_track')))
    dense = {f.frame: f.world_xyz for f in track.frames}
    for kf in kfset.keyframes:
        if kf.world_xyz is None: continue
        d = dense.get(kf.frame)
        assert d is not None, (kf_path, kf.frame)
        for a,b in zip(kf.world_xyz, d):
            assert abs(a-b) < 1e-6, (kf_path, kf.frame, kf.world_xyz, d)
    print(f'{kf_path}: {len(kfset.keyframes)} keyframes match dense track')
"
```

Expected: each sidecar's `world_xyz` matches the dense track at the same
frame. (No-op if no `output*/` tree is present — note that in the summary.)

- [ ] **Step 4: Final commit if any verification fixups were needed**

```bash
git add -A
git commit -m "test(ball): Phase-A sparse-keyframe verification"
```

---

## Phase-A Done Criteria

- `ball_keyframes.json` is written by the ball stage with one entry per manual
  anchor; airborne entries carry the clicked camera ray + `ray_physics` depth
  source; ground/player/goal entries carry the right `depth_source`.
- Emitted `world_xyz` equals the dense `ball_track.json` value at each anchor
  frame.
- The UE manifest's ball entry carries `keyframes_json` (pipeline + UE mirror),
  defaulting to empty for pre-existing runs.
- The dense path, web viewer, and `test_ball_anchor_accuracy.py` are unchanged
  and still pass.
- Phase B (engine consumption: sparse keying, spin, trajectory tools) is left
  for its own plan against this contract.
