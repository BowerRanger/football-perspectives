# Ball Phase 2 — Shot Chains + Landmark-Coincidence Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement Phase 2 of `docs/superpowers/specs/2026-07-02-ball-stage-improvement-design.md`: first-class goal-mouth **shot chains** (strike → optional deflections → terminal impact, each node a hard 3-D knot, with time-of-flight/ordering validation and auto-proposal) and **landmark-coincidence ball fixes** (a grounded anchor snapped to a known pitch landmark or line — an exact hard knot from one click).

**Architecture:** No new solve path — `player_touch(touch_type="shot") → goal_impact` keyframe pairs already classify as `ballistic` in `src/utils/ball_segments.py` (`_implies_flight`), so chains compile down to ordinary anchors. What's new: (1) schema carries `BallAnchor.landmark` and `BallAnchorSet.shot_chains`; (2) a pure landmark resolver snaps grounded anchors to catalogue landmarks/lines in **both** anchor-resolution paths (piecewise `_resolve_anchor_world` in `ball.py` and events-mode `_resolve_waypoint_world` in `ball_event_resolver.py` — events is the default solver); (3) a pure `ball_shot_chain.py` validates chains against resolved keyframes and proposes strike→impact pairs from detected events; (4) the ball stage writes proposals into the auto sidecar and per-chain warnings into the diag; (5) two read-only suggest endpoints (goal element by ray residual, pitch fix by ground-point proximity) power the editor's new goal-impact, pitch-fix, and shot-chain authoring — the palette has **no goal_impact tag today**, so this adds goal-impact authoring for the first time.

**Tech Stack:** Python 3.11, pytest, FastAPI + TestClient, numpy (tests + geometry), vanilla JS in a single HTML file.

## Scope notes — conscious v1 simplifications (do not "fix" these)

- **Keeper saves** author as ordinary `player_touch` anchors via the existing `/joints-near` suggest (hands are in `TOUCH_BONES`); the spec's "restricted to the keeper's track" refinement is deferred — the operator clicks the keeper directly.
- **Impulse consistency at deflections** (spec §6.3) rides the solver's existing restitution diagnostics (`flagged_bounces` in the diag); no chain-level restitution check in v1. Chain validation covers membership, resolution, and time-of-flight speed.
- **Auto-proposal pairing** is temporal (last touch within `pair_window_frames`); the spec's "ball direction points goalward" filter is deferred — implausible pairs are surfaced by the chain's `launch_speed` warning instead of suppressed.

## Global Constraints

- Type annotations on all new function signatures; frozen dataclasses; never mutate an input — return new objects.
- New utility modules torch-free and import-light (`pitch_landmarks` / `pitch_lines_catalogue` / `goal_geometry` are all light; keep it that way).
- Sidecar/endpoint behaviour is enrichment: suggest endpoints never 500 (empty result on failure, mirroring `/joints-near`); stage-side chain validation must never kill the stage.
- Schema changes must be backward compatible: old JSON without `landmark`/`shot_chains` loads unchanged; old readers ignore the new keys.
- Manual anchors always win; auto proposals are suggestions (dashed in UI, confirm/dismiss later — Phase 3 event list).
- Warn band for implied launch speed: **8–45 m/s** (`ball.shot_chain.launch_speed_warn_min_m_s: 8.0` / `launch_speed_warn_max_m_s: 45.0`), pairing window **75 frames** (`pair_window_frames: 75`) — config keys verbatim.
- Landmark field grammar: a `LANDMARK_CATALOGUE` name (point), or `line:<LINE_CATALOGUE name>` (line). Valid only on `grounded` anchors.
- Commit format `<type>: <description>` (feat/fix/test/docs/chore), no attribution trailers.
- Run tests with the repo venv from the repo root: `.venv/bin/python -m pytest`.
- All paths relative to `/Users/joebower/workplace/football-perspectives`.

---

### Task 1: Schema — `BallAnchor.landmark` + `BallAnchorSet.shot_chains`

**Files:**
- Modify: `src/schemas/ball_anchor.py` (dataclasses at :37-92, `load` at :94-204)
- Test: `tests/test_ball_anchor_schema_phase2.py`

**Interfaces:**
- Consumes: `LANDMARK_CATALOGUE: dict[str, PitchLandmark]` from `src/utils/pitch_landmarks.py`; `LINE_CATALOGUE: dict[str, tuple[tuple[float,float,float], tuple[float,float,float]]]` from `src/utils/pitch_lines_catalogue.py`.
- Produces: `BallAnchor.landmark: str | None = None` (new optional field, valid only on `grounded` state, name validated against the catalogues with the `line:` prefix grammar); `BallAnchorSet.shot_chains: tuple[tuple[int, ...], ...] = ()` (each chain ≥ 2 strictly-ascending frame numbers). Every later task relies on exactly these names.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_ball_anchor_schema_phase2.py`:

```python
"""Phase-2 schema additions: BallAnchor.landmark + BallAnchorSet.shot_chains."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.schemas.ball_anchor import BallAnchor, BallAnchorSet


def _write(tmp_path: Path, payload: dict) -> Path:
    p = tmp_path / "anchors.json"
    p.write_text(json.dumps(payload))
    return p


def _base(**extra) -> dict:
    payload = {
        "clip_id": "play", "image_size": [1280, 720],
        "anchors": [{"frame": 5, "image_xy": [10.0, 20.0], "state": "grounded"}],
    }
    payload.update(extra)
    return payload


def test_landmark_roundtrip_on_grounded(tmp_path: Path):
    payload = _base()
    payload["anchors"][0]["landmark"] = "left_goal_left_post_base"
    aset = BallAnchorSet.load(_write(tmp_path, payload))
    assert aset.anchors[0].landmark == "left_goal_left_post_base"
    out = tmp_path / "roundtrip.json"
    aset.save(out)
    assert json.loads(out.read_text())["anchors"][0]["landmark"] == \
        "left_goal_left_post_base"


def test_landmark_line_prefix_accepted(tmp_path: Path):
    from src.utils.pitch_lines_catalogue import LINE_CATALOGUE
    line_name = sorted(LINE_CATALOGUE)[0]
    payload = _base()
    payload["anchors"][0]["landmark"] = f"line:{line_name}"
    aset = BallAnchorSet.load(_write(tmp_path, payload))
    assert aset.anchors[0].landmark == f"line:{line_name}"


def test_landmark_rejected_on_non_grounded(tmp_path: Path):
    payload = _base()
    payload["anchors"][0]["state"] = "airborne_low"
    payload["anchors"][0]["landmark"] = "left_goal_left_post_base"
    with pytest.raises(ValueError, match="landmark"):
        BallAnchorSet.load(_write(tmp_path, payload))


def test_unknown_landmark_rejected(tmp_path: Path):
    payload = _base()
    payload["anchors"][0]["landmark"] = "no_such_feature"
    with pytest.raises(ValueError, match="landmark"):
        BallAnchorSet.load(_write(tmp_path, payload))


def test_shot_chains_roundtrip(tmp_path: Path):
    payload = _base(shot_chains=[[10, 34], [50, 61, 70]])
    aset = BallAnchorSet.load(_write(tmp_path, payload))
    assert aset.shot_chains == ((10, 34), (50, 61, 70))
    out = tmp_path / "roundtrip.json"
    aset.save(out)
    assert json.loads(out.read_text())["shot_chains"] == [[10, 34], [50, 61, 70]]


def test_shot_chain_must_be_ascending_and_len2(tmp_path: Path):
    with pytest.raises(ValueError, match="shot_chain"):
        BallAnchorSet.load(_write(tmp_path, _base(shot_chains=[[34, 10]])))
    with pytest.raises(ValueError, match="shot_chain"):
        BallAnchorSet.load(_write(tmp_path, _base(shot_chains=[[10]])))


def test_legacy_payload_without_new_fields_loads(tmp_path: Path):
    aset = BallAnchorSet.load(_write(tmp_path, _base()))
    assert aset.anchors[0].landmark is None
    assert aset.shot_chains == ()


def test_default_construction_unchanged():
    a = BallAnchor(frame=1, image_xy=(1.0, 2.0), state="grounded")
    assert a.landmark is None
    s = BallAnchorSet(clip_id="c", image_size=(1280, 720), anchors=(a,))
    assert s.shot_chains == ()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_ball_anchor_schema_phase2.py -v`
Expected: FAIL — `TypeError: BallAnchor.__init__() got an unexpected keyword argument`-style failures / `AttributeError: landmark`.

- [ ] **Step 3: Implement the schema additions**

In `src/schemas/ball_anchor.py`:

1. Add the field to `BallAnchor` (after `end_frame: int | None = None`):

```python
    # Pitch-feature coincidence for a grounded anchor: a LANDMARK_CATALOGUE
    # name (point) or "line:<LINE_CATALOGUE name>". The ball stage snaps the
    # anchor's world x,y to the feature (exact hard knot); the clicked pixel
    # remains authoring provenance. None for ordinary anchors.
    landmark: str | None = None
```

2. Add the field to `BallAnchorSet` (after `anchors`):

```python
    # Operator-authored shot chains: each entry is >= 2 strictly-ascending
    # member anchor frames (strike -> [deflections...] -> terminal impact).
    # Grouping only — members are ordinary anchors; the ball stage validates
    # each chain and reports warnings in the diag sidecar.
    shot_chains: tuple[tuple[int, ...], ...] = ()
```

3. In `load`, inside the per-anchor loop (after the `spin` validation block, before `confidence`), add:

```python
            landmark = a.get("landmark")
            if landmark:
                if state != "grounded":
                    raise ValueError(
                        f"landmark is only valid on state 'grounded'; "
                        f"got state {state!r}"
                    )
                from src.utils.pitch_landmarks import LANDMARK_CATALOGUE
                from src.utils.pitch_lines_catalogue import LINE_CATALOGUE
                if landmark.startswith("line:"):
                    if landmark[5:] not in LINE_CATALOGUE:
                        raise ValueError(
                            f"unknown landmark line {landmark!r}"
                        )
                elif landmark not in LANDMARK_CATALOGUE:
                    raise ValueError(f"unknown landmark {landmark!r}")
```

and pass `landmark=str(landmark) if landmark else None,` in the `BallAnchor(...)` construction.

4. In `load`, after the anchor loop and before `return cls(...)`, parse chains:

```python
        shot_chains: list[tuple[int, ...]] = []
        for chain in data.get("shot_chains", []):
            frames = tuple(int(f) for f in chain)
            if len(frames) < 2:
                raise ValueError(
                    f"shot_chain needs >= 2 member frames; got {frames}"
                )
            if any(b <= a_ for a_, b in zip(frames, frames[1:])):
                raise ValueError(
                    f"shot_chain frames must be strictly ascending; got {frames}"
                )
            shot_chains.append(frames)
```

and pass `shot_chains=tuple(shot_chains),` in `return cls(...)`.

(`save` needs no change — `asdict` + `json.dump` serialize the new tuple fields as arrays natively.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_ball_anchor_schema_phase2.py tests/test_ball_anchor_schema.py -v`
Expected: all PASS (second file guards no regression to existing validation).

- [ ] **Step 5: Commit**

```bash
git add src/schemas/ball_anchor.py tests/test_ball_anchor_schema_phase2.py
git commit -m "feat: BallAnchor.landmark + BallAnchorSet.shot_chains schema"
```

---

### Task 2: Pure landmark resolver + pitch-fix suggester (`src/utils/ball_landmark_fix.py`)

**Files:**
- Create: `src/utils/ball_landmark_fix.py`
- Test: `tests/test_ball_landmark_fix.py`

**Interfaces:**
- Consumes: `LANDMARK_CATALOGUE` (`PitchLandmark(name, world_xyz)`), `LINE_CATALOGUE` (name → `((x,y,z),(x,y,z))` endpoints), `ankle_ray_to_pitch(uv, *, K, R, t, plane_z, distortion) -> np.ndarray` from `src/utils/foot_anchor.py`.
- Produces (later tasks rely on exactly these):
  - `resolve_landmark_world(image_xy: tuple[float, float] | None, landmark: str, *, K: np.ndarray | None, R: np.ndarray | None, t: np.ndarray | None, distortion: tuple[float, float], ball_radius: float) -> np.ndarray | None` — point landmark → `(lm.x, lm.y, ball_radius)` (camera not needed); `line:<name>` → clicked-pixel ground ray ∩ `z=ball_radius` plane, projected onto the line segment in 2-D (clamped to endpoints), z = `ball_radius`; `None` when unresolvable (bad name, missing camera/pixel for a line).
  - `suggest_pitch_fixes(ground_xy: tuple[float, float], *, max_distance_m: float = 2.0, limit: int = 5) -> list[dict]` — ranked `{"name": str, "kind": "landmark"|"line", "distance_m": float, "world_xy": [x, y]}`; `name` for lines already carries the `line:` prefix; only ground-level point landmarks (world z ≤ 0.2) are considered.
  - `project_onto_segment_2d(p: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float]` — closest point on segment (clamped).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_ball_landmark_fix.py`:

```python
"""Landmark-coincidence resolution: point snap, line snap, suggestions."""

from __future__ import annotations

import numpy as np
import pytest

from src.utils.ball_landmark_fix import (
    project_onto_segment_2d,
    resolve_landmark_world,
    suggest_pitch_fixes,
)
from src.utils.pitch_landmarks import LANDMARK_CATALOGUE
from src.utils.pitch_lines_catalogue import LINE_CATALOGUE

BALL_R = 0.11


def _camera_pose():
    look = np.array([0.0, 64.0, -30.0])
    look /= np.linalg.norm(look)
    right = np.array([1.0, 0.0, 0.0])
    down = np.cross(look, right)
    R = np.array([right, down, look], dtype=float)
    t = -R @ np.array([52.5, -30.0, 30.0])
    K = np.array([[1500.0, 0, 640.0], [0, 1500.0, 360.0], [0, 0, 1.0]])
    return K, R, t


def _project(p, K, R, t):
    cam = R @ np.asarray(p, dtype=float) + t
    pix = K @ cam
    return float(pix[0] / pix[2]), float(pix[1] / pix[2])


def test_point_landmark_snaps_exactly_ignoring_camera():
    name = "left_goal_left_post_base"
    lm = LANDMARK_CATALOGUE[name]
    world = resolve_landmark_world(
        (999.0, 999.0), name, K=None, R=None, t=None,
        distortion=(0.0, 0.0), ball_radius=BALL_R,
    )
    assert world is not None
    assert world[0] == pytest.approx(lm.world_xyz[0])
    assert world[1] == pytest.approx(lm.world_xyz[1])
    assert world[2] == pytest.approx(BALL_R)


def test_line_landmark_snaps_ground_point_onto_line():
    K, R, t = _camera_pose()
    line_name = sorted(LINE_CATALOGUE)[0]
    (ax, ay, _az), (bx, by, _bz) = LINE_CATALOGUE[line_name]
    # A true point slightly OFF the line at ball height; its click pixel
    # must snap back onto the line.
    mid = np.array([(ax + bx) / 2.0, (ay + by) / 2.0, BALL_R])
    off = mid + np.array([0.3, 0.3, 0.0])
    uv = _project(off, K, R, t)
    world = resolve_landmark_world(
        uv, f"line:{line_name}", K=K, R=R, t=t,
        distortion=(0.0, 0.0), ball_radius=BALL_R,
    )
    assert world is not None
    snapped = project_onto_segment_2d(
        (world[0], world[1]), (ax, ay), (bx, by))
    assert world[0] == pytest.approx(snapped[0], abs=1e-6)
    assert world[1] == pytest.approx(snapped[1], abs=1e-6)
    assert world[2] == pytest.approx(BALL_R)
    # And it landed near the true off-line point (within the 0.3m offset + eps).
    assert np.hypot(world[0] - off[0], world[1] - off[1]) < 0.5


def test_line_landmark_without_camera_returns_none():
    line_name = sorted(LINE_CATALOGUE)[0]
    assert resolve_landmark_world(
        (10.0, 10.0), f"line:{line_name}", K=None, R=None, t=None,
        distortion=(0.0, 0.0), ball_radius=BALL_R,
    ) is None


def test_unknown_name_returns_none():
    assert resolve_landmark_world(
        (10.0, 10.0), "no_such_feature", K=None, R=None, t=None,
        distortion=(0.0, 0.0), ball_radius=BALL_R,
    ) is None


def test_project_onto_segment_clamps_to_endpoints():
    assert project_onto_segment_2d((-5.0, 0.0), (0.0, 0.0), (10.0, 0.0)) == (0.0, 0.0)
    assert project_onto_segment_2d((15.0, 3.0), (0.0, 0.0), (10.0, 0.0)) == (10.0, 0.0)
    assert project_onto_segment_2d((4.0, 3.0), (0.0, 0.0), (10.0, 0.0)) == (4.0, 0.0)


def test_suggest_ranks_nearest_first_and_prefixes_lines():
    name = "left_goal_left_post_base"
    lm = LANDMARK_CATALOGUE[name]
    near = (lm.world_xyz[0] + 0.2, lm.world_xyz[1] + 0.1)
    out = suggest_pitch_fixes(near, max_distance_m=2.0, limit=5)
    assert out, "expected at least the nearby post base"
    assert out[0]["name"] == name
    assert out[0]["kind"] == "landmark"
    assert out[0]["distance_m"] == pytest.approx(np.hypot(0.2, 0.1), abs=1e-6)
    for item in out:
        assert (item["kind"] == "line") == item["name"].startswith("line:")
        assert item["distance_m"] <= 2.0
    assert [i["distance_m"] for i in out] == sorted(i["distance_m"] for i in out)


def test_suggest_ignores_elevated_landmarks():
    # Crossbar endpoints live at z=2.44 — never a grounded-ball fix.
    out = suggest_pitch_fixes((0.0, 30.34), max_distance_m=1.0, limit=10)
    assert all("crossbar" not in i["name"] for i in out if i["kind"] == "landmark")


def test_suggest_empty_when_nothing_in_range():
    assert suggest_pitch_fixes((52.5, 20.0), max_distance_m=0.05, limit=5) == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_ball_landmark_fix.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.utils.ball_landmark_fix'`

- [ ] **Step 3: Implement the module**

Create `src/utils/ball_landmark_fix.py`:

```python
"""Landmark-coincidence ball fixes (spec §5.3).

A grounded ball anchor may name a pitch feature it visibly coincides with
(penalty spot, line crossing, corner arc). The feature's exact FIFA world
coordinates make the anchor an exact hard knot: for a point landmark the
world x,y IS the landmark; for a line the clicked-pixel ground ray is
snapped onto the line (1-D). Pure and torch-free — consumed by both anchor
resolution paths and the web suggest endpoint.
"""

from __future__ import annotations

import numpy as np

from src.utils.foot_anchor import ankle_ray_to_pitch
from src.utils.pitch_landmarks import LANDMARK_CATALOGUE
from src.utils.pitch_lines_catalogue import LINE_CATALOGUE

LINE_PREFIX = "line:"
# Point landmarks above this height (crossbar ends, flag tops) are never
# grounded-ball fixes.
_MAX_GROUND_LANDMARK_Z = 0.2


def project_onto_segment_2d(
    p: tuple[float, float],
    a: tuple[float, float],
    b: tuple[float, float],
) -> tuple[float, float]:
    """Closest point to ``p`` on segment ``a``-``b`` (2-D, clamped)."""
    ax, ay = float(a[0]), float(a[1])
    bx, by = float(b[0]), float(b[1])
    px, py = float(p[0]), float(p[1])
    dx, dy = bx - ax, by - ay
    denom = dx * dx + dy * dy
    if denom <= 0.0:
        return (ax, ay)
    s = ((px - ax) * dx + (py - ay) * dy) / denom
    s = min(1.0, max(0.0, s))
    return (ax + s * dx, ay + s * dy)


def resolve_landmark_world(
    image_xy: tuple[float, float] | None,
    landmark: str,
    *,
    K: np.ndarray | None,
    R: np.ndarray | None,
    t: np.ndarray | None,
    distortion: tuple[float, float],
    ball_radius: float,
) -> np.ndarray | None:
    """World position of a landmark-coincident grounded ball anchor.

    Point landmark: the landmark's x,y at ball height (camera-independent).
    ``line:<name>``: clicked-pixel ground ray at z=ball_radius, projected
    onto the line segment. None when unresolvable.
    """
    if landmark.startswith(LINE_PREFIX):
        seg = LINE_CATALOGUE.get(landmark[len(LINE_PREFIX):])
        if seg is None or image_xy is None \
                or K is None or R is None or t is None:
            return None
        try:
            ground = ankle_ray_to_pitch(
                (float(image_xy[0]), float(image_xy[1])),
                K=K, R=R, t=t, plane_z=ball_radius, distortion=distortion,
            )
        except ValueError:
            return None
        (ax, ay, _), (bx, by, _) = seg
        sx, sy = project_onto_segment_2d(
            (float(ground[0]), float(ground[1])), (ax, ay), (bx, by))
        return np.array([sx, sy, ball_radius], dtype=float)
    lm = LANDMARK_CATALOGUE.get(landmark)
    if lm is None:
        return None
    return np.array(
        [lm.world_xyz[0], lm.world_xyz[1], ball_radius], dtype=float)


def suggest_pitch_fixes(
    ground_xy: tuple[float, float],
    *,
    max_distance_m: float = 2.0,
    limit: int = 5,
) -> list[dict]:
    """Pitch features near a ground point, nearest first.

    Lines are named with the ``line:`` prefix so a suggestion's ``name``
    can be stored directly in ``BallAnchor.landmark``.
    """
    gx, gy = float(ground_xy[0]), float(ground_xy[1])
    items: list[dict] = []
    for lm in LANDMARK_CATALOGUE.values():
        if lm.world_xyz[2] > _MAX_GROUND_LANDMARK_Z:
            continue
        d = float(np.hypot(lm.world_xyz[0] - gx, lm.world_xyz[1] - gy))
        if d <= max_distance_m:
            items.append({
                "name": lm.name, "kind": "landmark", "distance_m": d,
                "world_xy": [lm.world_xyz[0], lm.world_xyz[1]],
            })
    for name, ((ax, ay, _), (bx, by, _)) in LINE_CATALOGUE.items():
        sx, sy = project_onto_segment_2d((gx, gy), (ax, ay), (bx, by))
        d = float(np.hypot(sx - gx, sy - gy))
        if d <= max_distance_m:
            items.append({
                "name": f"{LINE_PREFIX}{name}", "kind": "line",
                "distance_m": d, "world_xy": [sx, sy],
            })
    items.sort(key=lambda i: (i["distance_m"], i["name"]))
    return items[:limit]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_ball_landmark_fix.py -v`
Expected: 9 PASSED.

- [ ] **Step 5: Commit**

```bash
git add src/utils/ball_landmark_fix.py tests/test_ball_landmark_fix.py
git commit -m "feat: landmark-coincidence resolver + pitch-fix suggester"
```

---

### Task 3: Landmark-aware anchor resolution in both solver paths

**Files:**
- Modify: `src/stages/ball.py` (`_resolve_anchor_world`, :268-350)
- Modify: `src/utils/ball_event_resolver.py` (`_resolve_waypoint_world`, :101-135)
- Test: `tests/test_ball_landmark_resolution.py`

**Interfaces:**
- Consumes: `resolve_landmark_world(...)` from Task 2 (exact signature there).
- Produces: a grounded anchor with `landmark` set resolves to the snapped world position in the **piecewise** path (`_resolve_anchor_world`) and the **events** path (`_resolve_waypoint_world`, the default solver). Fallback: if the landmark fails to resolve, behaviour is unchanged (existing ray-cast).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_ball_landmark_resolution.py`:

```python
"""Grounded anchors with a landmark snap to the feature in BOTH anchor
resolution paths (piecewise _resolve_anchor_world, events
_resolve_waypoint_world)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from src.schemas.ball_anchor import BallAnchor
from src.utils.pitch_landmarks import LANDMARK_CATALOGUE

BALL_R = 0.11
NAME = "left_goal_left_post_base"


def _camera_pose():
    look = np.array([0.0, 64.0, -30.0])
    look /= np.linalg.norm(look)
    right = np.array([1.0, 0.0, 0.0])
    down = np.cross(look, right)
    R = np.array([right, down, look], dtype=float)
    t = -R @ np.array([52.5, -30.0, 30.0])
    K = np.array([[1500.0, 0, 640.0], [0, 1500.0, 360.0], [0, 0, 1.0]])
    return K, R, t


def _anchor() -> BallAnchor:
    return BallAnchor(frame=5, image_xy=(400.0, 400.0), state="grounded",
                      landmark=NAME)


def test_piecewise_path_snaps_grounded_landmark():
    from src.stages.ball import _resolve_anchor_world
    from src.utils.goal_geometry import GoalGeometry

    K, R, t = _camera_pose()
    lm = LANDMARK_CATALOGUE[NAME]
    world = _resolve_anchor_world(
        anc=_anchor(), fi=5, ground_touch_frames=set(),
        # grounded+landmark path must not touch the player context
        player_ctx=SimpleNamespace(),
        per_frame_K={5: K}, per_frame_R={5: R}, per_frame_t={5: t},
        distortion=(0.0, 0.0), ball_radius=BALL_R,
        goal_geometry=GoalGeometry.from_pitch_config(
            {"length_m": 105.0, "width_m": 68.0, "goal_height_m": 2.44,
             "goal_width_m": 7.32, "goal_depth_m": 1.5}),
    )
    assert world is not None
    assert world[0] == pytest.approx(lm.world_xyz[0])
    assert world[1] == pytest.approx(lm.world_xyz[1])
    assert world[2] == pytest.approx(BALL_R)


def test_events_path_snaps_grounded_landmark():
    from src.utils.ball_event_resolver import _resolve_waypoint_world

    K, R, t = _camera_pose()
    lm = LANDMARK_CATALOGUE[NAME]
    world = _resolve_waypoint_world(
        _anchor(), 5, K, R, t, (0.0, 0.0), BALL_R, None)
    assert world is not None
    assert world[0] == pytest.approx(lm.world_xyz[0])
    assert world[1] == pytest.approx(lm.world_xyz[1])
    assert world[2] == pytest.approx(BALL_R)


def test_grounded_without_landmark_unchanged_ray_cast():
    from src.utils.ball_event_resolver import _resolve_waypoint_world
    from src.utils.foot_anchor import ankle_ray_to_pitch
    from src.utils.ball_anchor_heights import state_to_height

    K, R, t = _camera_pose()
    anc = BallAnchor(frame=5, image_xy=(400.0, 400.0), state="grounded")
    world = _resolve_waypoint_world(anc, 5, K, R, t, (0.0, 0.0), BALL_R, None)
    expected = ankle_ray_to_pitch(
        (400.0, 400.0), K=K, R=R, t=t,
        plane_z=state_to_height("grounded"), distortion=(0.0, 0.0))
    assert world == pytest.approx(expected)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_ball_landmark_resolution.py -v`
Expected: the two landmark tests FAIL (world resolves via ray-cast, not the landmark snap); the no-landmark test PASSES.

- [ ] **Step 3: Add the branch to both resolvers**

In `src/stages/ball.py`, inside `_resolve_anchor_world`, insert as the FIRST state branch (before the `goal_impact` branch at ~:306), using the function's existing local `uv`/`K`/`R`/`t` variables (read the function first — it extracts these near the top):

```python
    if anc.state == "grounded" and anc.landmark:
        from src.utils.ball_landmark_fix import resolve_landmark_world
        world = resolve_landmark_world(
            anc.image_xy, anc.landmark, K=K, R=R, t=t,
            distortion=distortion, ball_radius=ball_radius,
        )
        if world is not None:
            return world
        # else fall through to the ordinary grounded ray-cast
```

In `src/utils/ball_event_resolver.py`, inside `_resolve_waypoint_world`, insert immediately after the `uv = (...)` line (:114) and before the `goal_impact` branch:

```python
    if anc.state == "grounded" and anc.landmark:
        from src.utils.ball_landmark_fix import resolve_landmark_world
        world = resolve_landmark_world(
            anc.image_xy, anc.landmark, K=K, R=R, t=t,
            distortion=distortion, ball_radius=ball_radius,
        )
        if world is not None:
            return world
```

Note: `_resolve_anchor_world`'s guard clauses may return early when camera matrices are missing — the landmark branch must sit after `uv`/`K`/`R`/`t` are bound but before any state-specific resolution. Match the surrounding code's local variable names exactly (read the function; do not assume the names in this snippet if they differ — `distortion`/`ball_radius` are parameters, so those are stable).

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_ball_landmark_resolution.py tests/test_ball_event_resolver.py tests/test_ball_stage_anchors.py -v`
Expected: all PASS (neighbouring suites guard no regression).

- [ ] **Step 5: Commit**

```bash
git add src/stages/ball.py src/utils/ball_event_resolver.py tests/test_ball_landmark_resolution.py
git commit -m "feat: landmark-aware grounded anchor resolution in both solver paths"
```

---

### Task 4: Shot-chain module (`src/utils/ball_shot_chain.py`) + config block

**Files:**
- Create: `src/utils/ball_shot_chain.py`
- Modify: `config/default.yaml` (add `ball.shot_chain` block after the `kinematic_touch` block at ~:537)
- Test: `tests/test_ball_shot_chain.py`

**Interfaces:**
- Consumes: `BallEvent` (`src/utils/ball_auto_events.py:98-107`: frame/kind/score/player_id/bone/goal_element/end_frame); `BallKeyframeSet` (`src/schemas/ball_keyframes.py`: `.keyframes` tuple of `BallKeyframe(frame, state, ..., world_xyz)`).
- Produces (later tasks rely on exactly these):
  - `ShotChainCfg(enabled: bool = True, pair_window_frames: int = 75, launch_speed_warn_min_m_s: float = 8.0, launch_speed_warn_max_m_s: float = 45.0)` — frozen dataclass.
  - `propose_shot_chains(events: Sequence[BallEvent], cfg: ShotChainCfg) -> tuple[tuple[int, int], ...]` — for each `goal_impact` event, pair it with the LAST `touch` event strictly before it within `pair_window_frames`; one chain per impact; frame-sorted.
  - `chain_warnings(chain: Sequence[int], keyframes: "BallKeyframeSet | None", fps: float, cfg: ShotChainCfg) -> list[dict]` — warning dicts `{"kind": "missing_keyframe"|"unresolved_world"|"launch_speed", "frames": [..], "detail": str}`; `launch_speed` carries `"speed_m_s": float` too. Empty list = chain OK.
  - Config keys under `ball.shot_chain` exactly as in Global Constraints.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_ball_shot_chain.py`:

```python
"""Shot-chain proposal pairing + validation warnings."""

from __future__ import annotations

import pytest

from src.schemas.ball_keyframes import BallKeyframe, BallKeyframeSet
from src.utils.ball_auto_events import BallEvent
from src.utils.ball_shot_chain import (
    ShotChainCfg,
    chain_warnings,
    propose_shot_chains,
)

CFG = ShotChainCfg()


def _kfset(kfs: list[BallKeyframe]) -> BallKeyframeSet:
    return BallKeyframeSet(
        clip_id="play", fps=30.0, image_size=(1280, 720),
        keyframes=tuple(kfs), segments=(),
    )


def _kf(frame: int, world) -> BallKeyframe:
    return BallKeyframe(
        frame=frame, state="grounded", depth_source="ground",
        world_xyz=world,
    )


def test_propose_pairs_last_touch_before_impact():
    events = [
        BallEvent(frame=10, kind="touch", score=0.8, player_id="P1", bone="r_foot"),
        BallEvent(frame=30, kind="touch", score=0.6, player_id="P2", bone="l_foot"),
        BallEvent(frame=60, kind="goal_impact", score=0.9, goal_element="back_net"),
    ]
    assert propose_shot_chains(events, CFG) == ((30, 60),)


def test_propose_respects_window_and_disabled():
    events = [
        BallEvent(frame=1, kind="touch", score=0.8, player_id="P1", bone="r_foot"),
        BallEvent(frame=200, kind="goal_impact", score=0.9, goal_element="post"),
    ]
    assert propose_shot_chains(events, CFG) == ()  # 199 > 75 frame window
    assert propose_shot_chains(
        [BallEvent(frame=10, kind="touch", score=0.8, player_id="P1", bone="r_foot"),
         BallEvent(frame=40, kind="goal_impact", score=0.9, goal_element="post")],
        ShotChainCfg(enabled=False),
    ) == ()


def test_propose_one_chain_per_impact_multiple_impacts():
    events = [
        BallEvent(frame=10, kind="touch", score=0.8, player_id="P1", bone="r_foot"),
        BallEvent(frame=40, kind="goal_impact", score=0.9, goal_element="crossbar"),
        BallEvent(frame=55, kind="goal_impact", score=0.7, goal_element="back_net"),
    ]
    assert propose_shot_chains(events, CFG) == ((10, 40), (10, 55))


def test_warnings_ok_chain_is_empty():
    # 30 frames at 30 fps between two knots 20 m apart -> 20 m/s: in band.
    kfs = _kfset([_kf(10, (10.0, 34.0, 0.11)), _kf(40, (30.0, 34.0, 0.11))])
    assert chain_warnings([10, 40], kfs, 30.0, CFG) == []


def test_warnings_flag_launch_speed_out_of_band():
    # 2 m in 1 s -> 2 m/s: below the 8 m/s floor (a mis-clicked frame).
    kfs = _kfset([_kf(10, (10.0, 34.0, 0.11)), _kf(40, (12.0, 34.0, 0.11))])
    warns = chain_warnings([10, 40], kfs, 30.0, CFG)
    assert len(warns) == 1
    assert warns[0]["kind"] == "launch_speed"
    assert warns[0]["frames"] == [10, 40]
    assert warns[0]["speed_m_s"] == pytest.approx(2.0)


def test_warnings_flag_missing_and_unresolved():
    kfs = _kfset([_kf(10, (10.0, 34.0, 0.11)), _kf(40, None)])
    warns = chain_warnings([10, 40, 99], kfs, 30.0, CFG)
    kinds = {w["kind"] for w in warns}
    assert "unresolved_world" in kinds   # frame 40 has no world
    assert "missing_keyframe" in kinds   # frame 99 has no keyframe


def test_warnings_none_keyframes_degrades():
    warns = chain_warnings([10, 40], None, 30.0, CFG)
    assert [w["kind"] for w in warns] == ["missing_keyframe"]


def test_cfg_from_default_yaml_keys():
    import yaml
    from pathlib import Path
    cfg = yaml.safe_load(
        Path("config/default.yaml").read_text())["ball"]["shot_chain"]
    assert cfg["enabled"] is True
    assert cfg["pair_window_frames"] == 75
    assert cfg["launch_speed_warn_min_m_s"] == 8.0
    assert cfg["launch_speed_warn_max_m_s"] == 45.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_ball_shot_chain.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.utils.ball_shot_chain'`

- [ ] **Step 3: Implement the module + config**

Create `src/utils/ball_shot_chain.py`:

```python
"""Shot chains (spec §6): strike -> [deflections...] -> terminal impact.

A chain is a grouping of ordinary anchors — no new solve path. Segments
between flight-implying members are already ballistic via
ball_segments._implies_flight; this module adds (a) auto-proposal pairing
(each detected goal_impact <- the last preceding touch within a window)
and (b) per-chain validation warnings against the resolved keyframes
(missing members, unresolved worlds, implied launch speed outside the
shot envelope — which catches a mis-clicked frame immediately).
Pure and torch-free.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:  # pragma: no cover — typing only
    from src.schemas.ball_keyframes import BallKeyframeSet
    from src.utils.ball_auto_events import BallEvent


@dataclass(frozen=True)
class ShotChainCfg:
    enabled: bool = True
    pair_window_frames: int = 75
    launch_speed_warn_min_m_s: float = 8.0
    launch_speed_warn_max_m_s: float = 45.0


def propose_shot_chains(
    events: "Sequence[BallEvent]", cfg: ShotChainCfg,
) -> tuple[tuple[int, int], ...]:
    """Strike->impact pairs: each goal_impact event claims the LAST touch
    event strictly before it within ``pair_window_frames``."""
    if not cfg.enabled:
        return ()
    touches = sorted(e.frame for e in events if e.kind == "touch")
    chains: list[tuple[int, int]] = []
    for impact in sorted(e.frame for e in events if e.kind == "goal_impact"):
        candidates = [
            f for f in touches
            if f < impact and impact - f <= cfg.pair_window_frames
        ]
        if candidates:
            chains.append((candidates[-1], impact))
    return tuple(chains)


def chain_warnings(
    chain: Sequence[int],
    keyframes: "BallKeyframeSet | None",
    fps: float,
    cfg: ShotChainCfg,
) -> list[dict]:
    """Validation warnings for one chain; empty list means the chain is
    consistent with the resolved keyframes."""
    frames = [int(f) for f in chain]
    by_frame = {
        kf.frame: kf for kf in (keyframes.keyframes if keyframes else ())
    }
    warnings: list[dict] = []
    missing = [f for f in frames if f not in by_frame]
    if missing:
        warnings.append({
            "kind": "missing_keyframe", "frames": missing,
            "detail": "chain member frame(s) have no resolved keyframe "
                      "— place an anchor there",
        })
    unresolved = [
        f for f in frames
        if f in by_frame and by_frame[f].world_xyz is None
    ]
    if unresolved:
        warnings.append({
            "kind": "unresolved_world", "frames": unresolved,
            "detail": "chain member keyframe(s) have no 3-D position",
        })
    usable = [f for f in frames if by_frame.get(f) is not None
              and by_frame[f].world_xyz is not None]
    for a, b in zip(usable, usable[1:]):
        wa, wb = by_frame[a].world_xyz, by_frame[b].world_xyz
        dt = (b - a) / fps
        if dt <= 0:
            continue
        speed = math.dist(wa, wb) / dt
        if not (cfg.launch_speed_warn_min_m_s
                <= speed <= cfg.launch_speed_warn_max_m_s):
            warnings.append({
                "kind": "launch_speed", "frames": [a, b],
                "speed_m_s": speed,
                "detail": f"implied speed {speed:.1f} m/s outside "
                          f"[{cfg.launch_speed_warn_min_m_s:.0f}, "
                          f"{cfg.launch_speed_warn_max_m_s:.0f}] m/s "
                          "— check the clicked frames",
            })
    return warnings
```

In `config/default.yaml`, immediately after the `kinematic_touch:` block (inside the `ball:` mapping, matching its 2-space indentation), add:

```yaml
  shot_chain:
    enabled: true              # auto-propose strike->impact chains from events
    pair_window_frames: 75     # max frames between strike touch and goal impact
    launch_speed_warn_min_m_s: 8.0   # warn band for implied speed between
    launch_speed_warn_max_m_s: 45.0  # consecutive chain knots (mis-click catch)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_ball_shot_chain.py -v`
Expected: 8 PASSED.

- [ ] **Step 5: Commit**

```bash
git add src/utils/ball_shot_chain.py config/default.yaml tests/test_ball_shot_chain.py
git commit -m "feat: shot-chain proposal pairing + validation warnings"
```

---

### Task 5: Ball stage integration — proposals into the auto sidecar, chain warnings into the diag

**Files:**
- Modify: `src/stages/ball.py`:
  - config builder next to `_kinematic_touch_cfg` (~:557)
  - auto-sidecar write (`BallAnchorSet(...)` construction at ~:1516-1520)
  - manual-chain loading (helper next to `_load_ball_anchors` at :205-220)
  - diag write (`diag: dict = {...}` at ~:1878-1914), keyframe path already computed at ~:1829
- Test: `tests/test_ball_stage_shot_chains.py`

**Interfaces:**
- Consumes: `ShotChainCfg`, `propose_shot_chains`, `chain_warnings` from Task 4 (exact signatures there); `BallAnchorSet.shot_chains` from Task 1; existing locals in `_solve_shot`: `events` (tuple[BallEvent]), `artifacts.camera_fps`, `event_keyframes` (BallKeyframeSet | None, bound at ~:1824), `kf_path` naming (`ball_track` → `ball_keyframes`).
- Produces: auto sidecar `{shot}_ball_anchors_auto.json` carries `shot_chains` (the proposals); diag `{shot}_ball_diag.json` gains `"shot_chains": [{"frames": [...], "source": "manual"|"auto", "warnings": [...]}]`. The preview endpoint (Task 6) reads this diag key.

- [ ] **Step 1: Write the failing stage-level test**

Create `tests/test_ball_stage_shot_chains.py`:

```python
"""BallStage shot-chain integration: auto proposals reach the auto sidecar;
manual + auto chains are validated into the diag sidecar."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.schemas.ball_anchor import BallAnchor, BallAnchorSet
from src.schemas.camera_track import CameraFrame, CameraTrack
from src.schemas.shots import Shot, ShotsManifest
from src.stages.ball import BallStage
from src.utils.ball_auto_events import BallEvent
from src.utils.ball_detector import FakeBallDetector

N_FRAMES = 60
FPS = 30.0


def _camera_pose():
    look = np.array([0.0, 64.0, -30.0])
    look /= np.linalg.norm(look)
    right = np.array([1.0, 0.0, 0.0])
    down = np.cross(look, right)
    R = np.array([right, down, look], dtype=float)
    t = -R @ np.array([52.5, -30.0, 30.0])
    K = np.array([[1500.0, 0, 640.0], [0, 1500.0, 360.0], [0, 0, 1.0]])
    return K, R, t


def _project(p, K, R, t):
    cam = R @ np.asarray(p, dtype=float) + t
    pix = K @ cam
    return float(pix[0] / pix[2]), float(pix[1] / pix[2])


def _build_scene(tmp_path: Path):
    out = tmp_path / "out"
    K, R, t = _camera_pose()
    clip = out / "shots" / "play.mp4"
    clip.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(clip), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (1280, 720))
    for _ in range(N_FRAMES):
        writer.write(np.full((720, 1280, 3), [50, 200, 50], dtype=np.uint8))
    writer.release()
    CameraTrack(
        clip_id="play", fps=FPS, image_size=(1280, 720),
        t_world=t.tolist(),
        frames=tuple(
            CameraFrame(frame=i, K=K.tolist(), R=R.tolist(),
                        confidence=1.0, is_anchor=(i == 0))
            for i in range(N_FRAMES)),
    ).save(out / "camera" / "play_camera_track.json")
    ShotsManifest(
        source_file="fake.mp4", fps=FPS, total_frames=N_FRAMES,
        shots=[Shot(id="play", clip_file="shots/play.mp4",
                    start_frame=0, end_frame=N_FRAMES - 1,
                    start_time=0.0, end_time=(N_FRAMES - 1) / FPS)],
    ).save(out / "shots" / "shots_manifest.json")
    detections = []
    for i in range(N_FRAMES):
        p = np.array([30.0 + 0.2 * i, 34.0, 0.11])
        u, v = _project(p, K, R, t)
        detections.append((u, v, 0.9))
    return out, detections


def _cfg() -> dict:
    return {
        "ball": {"detector": "fake",
                 "appearance_bridge": {"enabled": False}},
        "pitch": {"length_m": 105.0, "width_m": 68.0},
    }


@pytest.mark.integration
def test_auto_proposal_reaches_auto_sidecar(tmp_path: Path, monkeypatch):
    out, detections = _build_scene(tmp_path)

    synthetic = (
        BallEvent(frame=15, kind="touch", score=0.8,
                  player_id="P001", bone="r_foot"),
        BallEvent(frame=40, kind="goal_impact", score=0.9,
                  goal_element="back_net"),
    )
    monkeypatch.setattr(
        "src.stages.ball.detect_events", lambda **kwargs: synthetic)
    BallStage(config=_cfg(), output_dir=out,
              ball_detector=FakeBallDetector(detections)).run()

    auto = json.loads(
        (out / "ball" / "play_ball_anchors_auto.json").read_text())
    assert [15, 40] in [list(c) for c in auto.get("shot_chains", [])]

    diag = json.loads((out / "ball" / "play_ball_diag.json").read_text())
    auto_chains = [c for c in diag["shot_chains"] if c["source"] == "auto"]
    assert any(c["frames"] == [15, 40] for c in auto_chains)


@pytest.mark.integration
def test_manual_chain_validated_into_diag(tmp_path: Path):
    out, detections = _build_scene(tmp_path)
    # Manual anchors 20 frames apart, ~4 m apart on the roll -> ~6 m/s,
    # below the 8 m/s shot floor -> launch_speed warning expected.
    K, R, t = _camera_pose()
    uv_a = _project(np.array([34.0, 34.0, 0.11]), K, R, t)
    uv_b = _project(np.array([38.0, 34.0, 0.11]), K, R, t)
    BallAnchorSet(
        clip_id="play", image_size=(1280, 720),
        anchors=(
            BallAnchor(frame=20, image_xy=uv_a, state="grounded"),
            BallAnchor(frame=40, image_xy=uv_b, state="grounded"),
        ),
        shot_chains=((20, 40),),
    ).save(out / "ball" / "play_ball_anchors.json")

    BallStage(config=_cfg(), output_dir=out,
              ball_detector=FakeBallDetector(detections)).run()

    diag = json.loads((out / "ball" / "play_ball_diag.json").read_text())
    manual = [c for c in diag["shot_chains"] if c["source"] == "manual"]
    assert len(manual) == 1
    assert manual[0]["frames"] == [20, 40]
    kinds = {w["kind"] for w in manual[0]["warnings"]}
    assert "launch_speed" in kinds
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_ball_stage_shot_chains.py -v`
Expected: FAIL — `KeyError: 'shot_chains'` (diag has no such key) / proposal missing from the auto sidecar.

- [ ] **Step 3: Wire the stage**

In `src/stages/ball.py`:

1. Import (top of file, with the other `ball_*` utility imports):

```python
from src.utils.ball_shot_chain import (
    ShotChainCfg,
    chain_warnings,
    propose_shot_chains,
)
```

2. Config builder next to `_kinematic_touch_cfg` (~:557):

```python
def _shot_chain_cfg(cfg_dict: dict) -> ShotChainCfg:
    """Build a ShotChainCfg from the ``ball.shot_chain`` sub-tree."""
    base = ShotChainCfg()
    return ShotChainCfg(
        enabled=bool(cfg_dict.get("enabled", base.enabled)),
        pair_window_frames=int(cfg_dict.get(
            "pair_window_frames", base.pair_window_frames)),
        launch_speed_warn_min_m_s=float(cfg_dict.get(
            "launch_speed_warn_min_m_s", base.launch_speed_warn_min_m_s)),
        launch_speed_warn_max_m_s=float(cfg_dict.get(
            "launch_speed_warn_max_m_s", base.launch_speed_warn_max_m_s)),
    )
```

3. Manual-chain loader next to `_load_ball_anchors` (:205-220), same file-path logic:

```python
def _load_manual_shot_chains(
    output_dir: Path, shot_id: str
) -> tuple[tuple[int, ...], ...]:
    """Shot chains from the manual anchor sidecar; () when absent/invalid."""
    if shot_id:
        path = output_dir / "ball" / f"{shot_id}_ball_anchors.json"
    else:
        path = output_dir / "ball" / "ball_anchors.json"
    if not path.exists():
        return ()
    try:
        return BallAnchorSet.load(path).shot_chains
    except Exception as exc:  # noqa: BLE001 — chains are enrichment
        logger.warning(
            "ball stage: failed to load shot chains at %s: %s", path, exc)
        return ()
```

4. In `_solve_shot`, right after the kinematic-touch merge block (after `events = merge_touch_events(...)` / its except, ~:1503) and before the auto-anchor block, compute the proposals:

```python
        chain_cfg = _shot_chain_cfg(cfg.get("shot_chain", {}))
        proposed_chains = propose_shot_chains(events, chain_cfg)
```

5. In the auto-sidecar write (~:1516-1520), pass the proposals through:

```python
                BallAnchorSet(
                    clip_id=artifacts.camera_clip_id,
                    image_size=artifacts.camera_image_size,
                    anchors=auto_anchors,
                    shot_chains=proposed_chains,
                ).save(auto_anchor_path(ball_out_path.parent, shot_id))
```

6. Chain validation into the diag. The keyframe sidecar path is computed inside the events-mode branch (~:1829); hoist it so both branches share it — before the `try:` at ~:1823 add:

```python
        kf_path = ball_out_path.with_name(
            ball_out_path.name.replace("ball_track", "ball_keyframes")
        )
```

(and delete the now-duplicate `kf_path = ...` line inside the events branch). Then, just before the `diag: dict = {...}` construction (~:1878), add:

```python
        shot_chain_diag: list[dict] = []
        try:
            kf_set = (
                BallKeyframeSet.load(kf_path) if kf_path.exists() else None
            )
            manual_chains = _load_manual_shot_chains(
                self.output_dir, shot_id)
            for source, chains in (
                ("manual", manual_chains), ("auto", proposed_chains),
            ):
                for chain in chains:
                    shot_chain_diag.append({
                        "frames": [int(f) for f in chain],
                        "source": source,
                        "warnings": chain_warnings(
                            chain, kf_set, artifacts.camera_fps, chain_cfg),
                    })
        except Exception as exc:  # noqa: BLE001 — validation never kills the stage
            logger.warning("ball: shot-chain validation failed: %s", exc)
```

Add `"shot_chains": shot_chain_diag,` to the `diag` dict (after `"contact_gaps": contact_gaps,`). Import `BallKeyframeSet` from `src.schemas.ball_keyframes` at the top of the file if not already imported (check — `_emit_ball_keyframes` may already import it).

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_ball_stage_shot_chains.py tests/test_ball_stage_kinematic_wiring.py tests/test_ball_stage.py tests/test_ball_stage_keyframes.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/stages/ball.py tests/test_ball_stage_shot_chains.py
git commit -m "feat: shot-chain proposals in auto sidecar + chain warnings in diag"
```

---

### Task 6: Server — suggest endpoints + payload/preview extensions

**Files:**
- Modify: `src/web/server.py`:
  - `BallAnchorEntry` / `BallAnchorPayload` pydantic models (~:1902-1905)
  - the POST `/ball-anchors/{shot_id}` handler's `BallAnchor(...)` construction and `BallAnchorSet(...)` construction (~:2040-2090)
  - two new GET routes after `/joints-near` (ends ~:2037)
  - the POST `/ball-anchors/{shot_id}/preview` handler (~:2092-2173)
- Test: `tests/test_web_ball_phase2_api.py`

**Interfaces:**
- Consumes: `goal_element_candidates(image_xy, *, K, R, t, distortion, geometry) -> list[tuple[str, float, float, np.ndarray]]` (`src/utils/goal_geometry.py:113-142`, `(element, residual_m, ray_s, world)` sorted); `GoalGeometry.from_pitch_config`; `suggest_pitch_fixes` + `ankle_ray_to_pitch` (Task 2 / `foot_anchor.py`); schema fields from Task 1; diag key `shot_chains` from Task 5. Camera loading pattern: copy `/joints-near`'s (server.py:1996-2020) — per-shot `{shot}_camera_track.json` with unprefixed fallback.
- Produces:
  - `GET /goal-element-suggest?shot&frame&u&v` → `{"candidates": [{"element": str, "residual_m": float, "world_xyz": [x,y,z]}]}` (sorted best-first; `{"candidates": []}` on any failure — never 500s).
  - `GET /pitch-fix-suggest?shot&frame&u&v` → `{"ground_xy": [x, y] | null, "suggestions": [{"name", "kind", "distance_m", "world_xy"}]}` (empty on failure).
  - `BallAnchorEntry.landmark: str | None = None`; `BallAnchorPayload.shot_chains: list[list[int]] = []`; POST save persists both; GET `/ball-anchors/{shot_id}` returns them (it returns the raw JSON, so persistence is sufficient).
  - Preview response gains `"shot_chain_warnings"` (the diag's `shot_chains` list from the temp run) alongside the existing BallTrack keys.

- [ ] **Step 1: Write the failing endpoint tests**

Create `tests/test_web_ball_phase2_api.py`:

```python
"""Phase-2 web API: suggest endpoints + landmark/shot_chains persistence."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from src.schemas.camera_track import CameraFrame, CameraTrack
from src.web.server import create_app


def _client(tmp_path: Path) -> TestClient:
    return TestClient(create_app(output_dir=tmp_path, config_path=None))


def _camera_pose():
    look = np.array([0.0, 64.0, -30.0])
    look /= np.linalg.norm(look)
    right = np.array([1.0, 0.0, 0.0])
    down = np.cross(look, right)
    R = np.array([right, down, look], dtype=float)
    t = -R @ np.array([52.5, -30.0, 30.0])
    K = np.array([[1500.0, 0, 640.0], [0, 1500.0, 360.0], [0, 0, 1.0]])
    return K, R, t


def _save_camera(tmp_path: Path, shot: str = "play", n: int = 10) -> tuple:
    K, R, t = _camera_pose()
    CameraTrack(
        clip_id=shot, fps=30.0, image_size=(1280, 720),
        t_world=t.tolist(),
        frames=tuple(
            CameraFrame(frame=i, K=K.tolist(), R=R.tolist(),
                        confidence=1.0, is_anchor=(i == 0))
            for i in range(n)),
    ).save(tmp_path / "camera" / f"{shot}_camera_track.json")
    return K, R, t


def _project(p, K, R, t):
    cam = R @ np.asarray(p, dtype=float) + t
    pix = K @ cam
    return float(pix[0] / pix[2]), float(pix[1] / pix[2])


def test_goal_element_suggest_ranks_by_residual(tmp_path: Path):
    K, R, t = _save_camera(tmp_path)
    # Pixel of the near-goal left post at mid height (x=0, y=30.34, z=1.2).
    u, v = _project(np.array([0.0, 30.34, 1.2]), K, R, t)
    r = _client(tmp_path).get(
        "/goal-element-suggest",
        params={"shot": "play", "frame": 0, "u": u, "v": v})
    assert r.status_code == 200
    cands = r.json()["candidates"]
    assert cands, "expected goal element candidates for a post pixel"
    assert cands[0]["element"] == "post"
    assert cands[0]["residual_m"] < 0.2
    assert len(cands[0]["world_xyz"]) == 3


def test_goal_element_suggest_graceful_without_camera(tmp_path: Path):
    r = _client(tmp_path).get(
        "/goal-element-suggest",
        params={"shot": "play", "frame": 0, "u": 10, "v": 10})
    assert r.status_code == 200
    assert r.json() == {"candidates": []}


def test_pitch_fix_suggest_finds_post_base(tmp_path: Path):
    K, R, t = _save_camera(tmp_path)
    u, v = _project(np.array([0.2, 30.3, 0.11]), K, R, t)
    r = _client(tmp_path).get(
        "/pitch-fix-suggest",
        params={"shot": "play", "frame": 0, "u": u, "v": v})
    assert r.status_code == 200
    body = r.json()
    assert body["ground_xy"] is not None
    names = [s["name"] for s in body["suggestions"]]
    assert "left_goal_left_post_base" in names


def test_pitch_fix_suggest_graceful_without_camera(tmp_path: Path):
    r = _client(tmp_path).get(
        "/pitch-fix-suggest",
        params={"shot": "play", "frame": 0, "u": 10, "v": 10})
    assert r.status_code == 200
    assert r.json() == {"ground_xy": None, "suggestions": []}


def test_post_persists_landmark_and_shot_chains(tmp_path: Path):
    client = _client(tmp_path)
    payload = {
        "clip_id": "play", "image_size": [1280, 720],
        "anchors": [
            {"frame": 5, "image_xy": [100.0, 200.0], "state": "grounded",
             "landmark": "left_goal_left_post_base"},
            {"frame": 30, "image_xy": [300.0, 400.0], "state": "player_touch",
             "player_id": "P1", "bone": "r_foot", "touch_type": "shot"},
            {"frame": 55, "image_xy": [500.0, 300.0], "state": "goal_impact",
             "goal_element": "back_net"},
        ],
        "shot_chains": [[30, 55]],
    }
    r = client.post("/ball-anchors/play", json=payload)
    assert r.status_code == 200, r.text
    got = client.get("/ball-anchors/play").json()
    assert got["anchors"][0]["landmark"] == "left_goal_left_post_base"
    assert got["shot_chains"] == [[30, 55]]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_web_ball_phase2_api.py -v`
Expected: the suggest tests FAIL with 404 (routes absent); the persistence test FAILS (`shot_chains` missing / landmark dropped).

- [ ] **Step 3: Implement the server changes**

In `src/web/server.py`:

1. Extend the pydantic models (~:1902-1905):

```python
class BallAnchorEntry(BaseModel):
    # ... existing fields unchanged ...
    landmark: str | None = None


class BallAnchorPayload(BaseModel):
    clip_id: str
    image_size: list[int]
    anchors: list[BallAnchorEntry]
    shot_chains: list[list[int]] = []
```

(Show only the additions — keep every existing field exactly as it is.)

2. In the POST `/ball-anchors/{shot_id}` handler: add `landmark=entry.landmark,` to the `BallAnchor(...)` construction, and `shot_chains=tuple(tuple(int(f) for f in c) for c in payload.shot_chains),` to the `BallAnchorSet(...)` construction. (Read the handler; the construction sites follow the `anchors_in: list[BallAnchor] = []` loop at ~:2043.)

3. New routes, inserted after the `/joints-near` function ends (~:2037) — both copy its camera-loading pattern and never-500 contract:

```python
    def _load_shot_camera(shot: str):
        """Camera matrices for a shot (best-effort, joints-near pattern)."""
        import numpy as np
        from src.schemas.camera_track import CameraTrack

        cam_path = output_dir / "camera" / f"{shot}_camera_track.json"
        if not cam_path.exists():
            cam_path = output_dir / "camera" / "camera_track.json"
        if not cam_path.exists():
            return None
        camera = CameraTrack.load(cam_path)
        per_frame_K = {f.frame: np.array(f.K) for f in camera.frames}
        per_frame_R = {f.frame: np.array(f.R) for f in camera.frames}
        t_world = np.array(camera.t_world)
        per_frame_t = {
            f.frame: (np.array(f.t) if f.t is not None else t_world)
            for f in camera.frames
        }
        return camera, per_frame_K, per_frame_R, per_frame_t

    @app.get("/goal-element-suggest")
    def goal_element_suggest(shot: str, frame: int, u: float, v: float):
        """Ranked goal elements for a clicked pixel (ray residual to the
        3-D goal geometry) — the editor's auto-suggest for goal_impact
        authoring. Best-effort: empty candidates when camera/config are
        unavailable."""
        from src.utils.goal_geometry import (
            GoalGeometry,
            goal_element_candidates,
        )

        try:
            loaded = _load_shot_camera(shot)
            if loaded is None:
                return {"candidates": []}
            camera, per_frame_K, per_frame_R, per_frame_t = loaded
            fi = int(frame)
            geometry = GoalGeometry.from_pitch_config(
                (config or {}).get("pitch", {}))
            cands = goal_element_candidates(
                (float(u), float(v)),
                K=per_frame_K[fi], R=per_frame_R[fi], t=per_frame_t[fi],
                distortion=camera.distortion, geometry=geometry,
            )
            return {"candidates": [
                {"element": el, "residual_m": float(res),
                 "world_xyz": [float(w[0]), float(w[1]), float(w[2])]}
                for el, res, _s, w in cands
            ]}
        except Exception as exc:  # noqa: BLE001 — suggestion helper, never 500s
            logger.debug("goal-element-suggest failed for %s: %s", shot, exc)
            return {"candidates": []}

    @app.get("/pitch-fix-suggest")
    def pitch_fix_suggest(shot: str, frame: int, u: float, v: float):
        """Pitch features near the clicked pixel's ground point — the
        editor's suggest for landmark-coincidence fixes. Best-effort."""
        from src.utils.ball_landmark_fix import suggest_pitch_fixes
        from src.utils.foot_anchor import ankle_ray_to_pitch

        try:
            loaded = _load_shot_camera(shot)
            if loaded is None:
                return {"ground_xy": None, "suggestions": []}
            camera, per_frame_K, per_frame_R, per_frame_t = loaded
            fi = int(frame)
            ball_radius = float(
                ((config or {}).get("ball", {})).get("ball_radius_m", 0.11))
            ground = ankle_ray_to_pitch(
                (float(u), float(v)),
                K=per_frame_K[fi], R=per_frame_R[fi], t=per_frame_t[fi],
                plane_z=ball_radius, distortion=camera.distortion,
            )
            gxy = (float(ground[0]), float(ground[1]))
            return {
                "ground_xy": [gxy[0], gxy[1]],
                "suggestions": suggest_pitch_fixes(gxy),
            }
        except Exception as exc:  # noqa: BLE001 — suggestion helper, never 500s
            logger.debug("pitch-fix-suggest failed for %s: %s", shot, exc)
            return {"ground_xy": None, "suggestions": []}
```

Note on `config`: `create_app` loads the pipeline config — find the existing local (search for how other routes read pitch/ball config inside `create_app`; if none exists, load once near the top of `create_app`: `config = load_config(config_path) if config_path else load_config()` using `src.pipeline.config.load_config`, which falls back to defaults). Use whatever name the file already binds if present.

4. Preview: in the POST `/ball-anchors/{shot_id}/preview` handler, (a) persist the posted chains into the temp manual sidecar — the handler already builds a `BallAnchorSet` from the payload; add the same `shot_chains=...` kwarg as in the save handler; (b) after the temp stage run, before returning the track JSON, attach the diag's chain report:

```python
            diag_path = tmp_ball_dir / f"{shot_id}_ball_diag.json"
            if diag_path.exists():
                try:
                    track_json["shot_chain_warnings"] = json.loads(
                        diag_path.read_text()).get("shot_chains", [])
                except Exception:  # noqa: BLE001 — enrichment only
                    pass
```

(Adapt the two local names — the temp ball dir and the response dict — to the handler's actual variables; read the handler at ~:2092-2173 first.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_web_ball_phase2_api.py tests/test_web_ball_anchors_api.py tests/test_web_ball_quality_api.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/web/server.py tests/test_web_ball_phase2_api.py
git commit -m "feat: goal-element + pitch-fix suggest endpoints; persist landmark/shot_chains"
```

---

### Task 7: Editor UI — goal-impact, pitch-fix, and shot-chain authoring

**Files:**
- Modify: `src/web/static/ball_anchor_editor.html`
- Test: `tests/test_web_ball_editor_phase2.py`

**Interfaces:**
- Consumes: `GET /goal-element-suggest` and `GET /pitch-fix-suggest` (Task 6 shapes); save/load of `landmark` + `shot_chains` (Task 6); preview `shot_chain_warnings`. Existing file internals (verified): `TAGS` array (:142-193, currently 10 tags, NO goal_impact), `selectedTag`, `anchors`, canvas click handler (`overlay.addEventListener("click", ...)`, :436-487), `updateOffScreenBtnVisibility()` (:529-536), off-screen button pattern (`offScreenBtn`, markup :127, handler :538-545), save handler (`saveBtn.onclick`, :547-560), `loadShot()` (:610-662), `renderAnchors()`, `seekToFrame(fi)`, `currentFrame()`.
- Produces: DOM ids `goalAuthor`/`goalElement`, `pitchFixAuthor`/`pitchFixName`, `chainBtn`/`chainList`; TAGS entries `goal_impact` and `pitch_fix` (the latter is UI-only — it saves a `grounded` anchor with `landmark`); JS global `shotChains: number[][]` serialized in the save payload and loaded in `loadShot()`.

- [ ] **Step 1: Write the failing markup test**

Create `tests/test_web_ball_editor_phase2.py`:

```python
"""The ball anchor editor ships goal-impact, pitch-fix, and shot-chain
authoring (palette entries, sub-forms, suggest-endpoint wiring,
shot_chains persistence)."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from src.web.server import create_app


def _client(tmp_path: Path) -> TestClient:
    return TestClient(create_app(output_dir=tmp_path, config_path=None))


def test_editor_served_with_phase2_authoring(tmp_path: Path):
    html = _client(tmp_path).get("/ball-anchor-editor").text
    # Palette gained goal_impact and the pitch-fix mode.
    assert 'id: "goal_impact"' in html
    assert 'id: "pitch_fix"' in html
    # Authoring sub-forms.
    assert 'id="goalAuthor"' in html
    assert 'id="goalElement"' in html
    assert 'id="pitchFixAuthor"' in html
    assert 'id="pitchFixName"' in html
    # Suggest endpoints wired.
    assert "/goal-element-suggest?shot=" in html
    assert "/pitch-fix-suggest?shot=" in html
    # Shot-chain authoring + persistence.
    assert 'id="chainBtn"' in html
    assert 'id="chainList"' in html
    assert "shot_chains" in html
    # Preview warnings surfaced.
    assert "shot_chain_warnings" in html
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_web_ball_editor_phase2.py -v`
Expected: FAIL on the first assertion.

- [ ] **Step 3: Add palette entries + sub-form markup**

In `src/web/static/ball_anchor_editor.html`:

1. Append to the `TAGS` array (after the `player_touch` entry, matching the existing entry shape):

```javascript
  {
    id: "goal_impact", label: "Goal impact", color: "#fbbf24",
    range: "ray ∩ goal, event",
    description: "Ball strikes the goal frame or net (post, crossbar, back net, side net). Hard knot at the ray-geometry intersection. The element is auto-suggested from your click; override it in the panel below.",
  },
  {
    id: "pitch_fix", label: "Pitch fix", color: "#a3e635",
    range: "z = 0.11 m, exact",
    description: "Ball visibly coincides with a known pitch feature (penalty spot, line, corner arc). Saves a grounded anchor snapped to the feature's exact FIFA coordinates — an exact hard knot from one click.",
  },
```

2. Sub-form markup, inserted immediately after the `touchAuthor` div's closing `</div>` (:103-115):

```html
    <div id="goalAuthor" style="display:none;border-top:1px solid #2d3148;padding:8px 10px;flex-shrink:0;">
      <div style="font-size:10px;color:#64748b;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:6px;">Goal impact</div>
      <label style="display:block;font-size:11px;color:#94a3b8;">Element
        <select id="goalElement" style="width:100%;background:#0f1117;color:#e2e8f0;border:1px solid #334155;border-radius:3px;padding:3px 6px;font-size:12px;">
          <option value="auto" selected>auto (suggest from click)</option>
          <option value="post">post</option>
          <option value="crossbar">crossbar</option>
          <option value="back_net">back net</option>
          <option value="side_net">side net</option>
        </select>
      </label>
      <div style="font-size:10px;color:#64748b;margin-top:6px;line-height:1.4;">Click where the ball strikes the goal — the nearest element is picked by ray residual.</div>
    </div>
    <div id="pitchFixAuthor" style="display:none;border-top:1px solid #2d3148;padding:8px 10px;flex-shrink:0;">
      <div style="font-size:10px;color:#64748b;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:6px;">Pitch fix</div>
      <label style="display:block;font-size:11px;color:#94a3b8;">Feature
        <select id="pitchFixName" style="width:100%;background:#0f1117;color:#e2e8f0;border:1px solid #334155;border-radius:3px;padding:3px 6px;font-size:12px;"></select>
      </label>
      <div style="font-size:10px;color:#64748b;margin-top:6px;line-height:1.4;">Click the ball — nearby pitch features are suggested (nearest first).</div>
    </div>
```

3. Shot-chain controls: next to the existing `offScreenBtn` (markup :127) add:

```html
      <button id="chainBtn">Start shot chain</button>
```

and, inside the left panel below the anchor list container (`anchorList`), add:

```html
    <div id="chainList" style="border-top:1px solid #2d3148;padding:6px 10px;font-size:11px;color:#94a3b8;"></div>
```

- [ ] **Step 4: Add the JS**

In the `<script>` block:

1. New state + sub-form visibility. Extend `updateOffScreenBtnVisibility()` (:529-536):

```javascript
let shotChains = [];        // number[][] — persisted with the anchor set
let activeChain = null;     // number[] while recording, else null

function updateOffScreenBtnVisibility() {
  offScreenBtn.style.display = selectedTag === "off_screen_flight" ? "" : "none";
  touchAuthor.style.display = selectedTag === "player_touch" ? "" : "none";
  document.getElementById("goalAuthor").style.display =
    selectedTag === "goal_impact" ? "" : "none";
  document.getElementById("pitchFixAuthor").style.display =
    selectedTag === "pitch_fix" ? "" : "none";
}
```

(`shotChains`/`activeChain` go with the other `let` state near :194-200.)

2. Canvas click handler: inside `overlay.addEventListener("click", ...)` (:436-487), insert two new branches after the `player_touch` branch and before the generic placement, plus chain recording at both placement exits. The full new branches:

```javascript
  if (selectedTag === "goal_impact") {
    let element = document.getElementById("goalElement").value;
    if (element === "auto") {
      try {
        const res = await fetch(
          `/goal-element-suggest?shot=${encodeURIComponent(shotId)}` +
          `&frame=${fi}&u=${u}&v=${v}`
        ).then(r => r.json());
        if (res.candidates && res.candidates.length) {
          element = res.candidates[0].element;
          document.getElementById("goalElement").value = element;
        }
      } catch (_err) { /* suggestion is best-effort */ }
    }
    if (element === "auto") {
      setStatus("goal impact needs an element (no goal under cursor — pick one manually)");
      return;
    }
    anchors = anchors.filter(a => a.frame !== fi);
    anchors.push({ frame: fi, image_xy: [u, v], state: "goal_impact",
                   goal_element: element, confidence: 1.0 });
    recordChainFrame(fi);
    setDirty(true);
    renderAnchors();
    drawOverlay();
    return;
  }

  if (selectedTag === "pitch_fix") {
    const sel = document.getElementById("pitchFixName");
    try {
      const res = await fetch(
        `/pitch-fix-suggest?shot=${encodeURIComponent(shotId)}` +
        `&frame=${fi}&u=${u}&v=${v}`
      ).then(r => r.json());
      sel.innerHTML = "";
      for (const s of (res.suggestions || [])) {
        const opt = document.createElement("option");
        opt.value = s.name;
        opt.textContent = `${s.name} (${s.distance_m.toFixed(2)} m)`;
        sel.appendChild(opt);
      }
    } catch (_err) { /* suggestion is best-effort */ }
    if (!sel.options.length) {
      setStatus("no pitch feature within range of that click");
      return;
    }
    anchors = anchors.filter(a => a.frame !== fi);
    anchors.push({ frame: fi, image_xy: [u, v], state: "grounded",
                   landmark: sel.value, confidence: 1.0 });
    recordChainFrame(fi);
    setDirty(true);
    renderAnchors();
    drawOverlay();
    return;
  }
```

In the `player_touch` branch and the generic placement branch, add `recordChainFrame(fi);` immediately after their `anchors.push(...)` lines.

3. Chain recording + rendering (place after the `offScreenBtn.onclick` handler, :538-545):

```javascript
function recordChainFrame(fi) {
  if (activeChain && !activeChain.includes(fi)) {
    activeChain.push(fi);
    renderChainBtn();
  }
}

function renderChainBtn() {
  const btn = document.getElementById("chainBtn");
  btn.textContent = activeChain
    ? `End shot chain (${activeChain.length})` : "Start shot chain";
}

function renderChains() {
  const el = document.getElementById("chainList");
  el.innerHTML = "";
  if (!shotChains.length) return;
  const head = document.createElement("div");
  head.style.cssText = "color:#64748b;font-size:10px;text-transform:uppercase;margin-bottom:4px;";
  head.textContent = `Shot chains (${shotChains.length})`;
  el.appendChild(head);
  shotChains.forEach((chain, idx) => {
    const row = document.createElement("div");
    row.style.cssText = "display:flex;align-items:center;gap:6px;padding:2px 0;cursor:pointer;";
    const label = document.createElement("span");
    label.style.cssText = "flex:1;min-width:0;";
    label.textContent = `⚽ ${chain.join(" → ")}`;
    label.onclick = () => seekToFrame(chain[0]);
    const del = document.createElement("button");
    del.textContent = "✕";
    del.title = "Delete this chain (anchors stay)";
    del.style.cssText = "background:#334155;color:#e2e8f0;border:1px solid #475569;border-radius:3px;font-size:10px;padding:0 5px;cursor:pointer;";
    del.onclick = (ev) => {
      ev.stopPropagation();
      shotChains.splice(idx, 1);
      setDirty(true);
      renderChains();
    };
    row.appendChild(label);
    row.appendChild(del);
    el.appendChild(row);
  });
}

document.getElementById("chainBtn").onclick = () => {
  if (activeChain) {
    if (activeChain.length >= 2) {
      shotChains.push([...activeChain].sort((a, b) => a - b));
      setDirty(true);
      renderChains();
      setStatus(`shot chain saved (${activeChain.length} anchors)`);
    } else {
      setStatus("shot chain needs ≥ 2 anchors — discarded");
    }
    activeChain = null;
  } else {
    activeChain = [];
    setStatus("shot chain recording — place strike, deflections, impact, then End");
  }
  renderChainBtn();
};
```

4. Persistence round-trip:
   - In `saveBtn.onclick` (:547-560), change the payload line to `const payload = { clip_id: shotId || "", image_size: imageSize, anchors, shot_chains: shotChains };`.
   - In `loadShot()` (:610-662), add `landmark: a.landmark ?? null,` to the manual-anchor mapping object, and after the anchors mapping add `shotChains = (ar && ar.shot_chains) ? ar.shot_chains.map(c => c.map(Number)) : []; renderChains();`.
   - In `renderAnchors()`, extend the manual-row `detail` expression to also show landmarks: change the goal-element fallback to `` a.goal_element ? ` (${a.goal_element})` : a.landmark ? ` (⚑ ${a.landmark})` : "" ``.
5. Preview warnings: find the preview response handler (the `previewBtn` flow) and, after the response JSON is parsed, add:

```javascript
      const chainWarns = (data.shot_chain_warnings || [])
        .flatMap(c => c.warnings.map(w => `[${c.frames.join("→")}] ${w.detail}`));
      if (chainWarns.length) setStatus(`preview: ${chainWarns[0]} (+${chainWarns.length - 1} more)`);
```

(Adapt the parsed-response variable name to the handler's actual local; read it first.)

- [ ] **Step 5: Run the tests**

Run: `.venv/bin/python -m pytest tests/test_web_ball_editor_phase2.py tests/test_web_ball_editor_touch_panel.py tests/test_web_ball_quality_timeline.py -v`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add src/web/static/ball_anchor_editor.html tests/test_web_ball_editor_phase2.py
git commit -m "feat: goal-impact, pitch-fix and shot-chain authoring in ball editor"
```

---

### Task 8: Full-suite verification

- [ ] **Step 1: Run the full test suite**

Run: `.venv/bin/python -m pytest -q`
Expected: everything passes except the known env-dependent Blender test (`test_blender_export_smpl_skeleton.py`) when Blender is absent.

- [ ] **Step 2: Lint the touched Python**

Run: `.venv/bin/python -m ruff check src/schemas/ball_anchor.py src/utils/ball_landmark_fix.py src/utils/ball_shot_chain.py src/stages/ball.py src/utils/ball_event_resolver.py src/web/server.py tests/test_ball_anchor_schema_phase2.py tests/test_ball_landmark_fix.py tests/test_ball_landmark_resolution.py tests/test_ball_shot_chain.py tests/test_ball_stage_shot_chains.py tests/test_web_ball_phase2_api.py tests/test_web_ball_editor_phase2.py`
Expected: clean (fix anything flagged before committing).

- [ ] **Step 3: Commit any fixups**

```bash
git add -A && git commit -m "chore: phase-2 lint/test fixups"   # only if needed
```
