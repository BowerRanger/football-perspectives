# Engine-Shaped Ball from Sparse Keyframes — Phase B Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the UE "Load Reconstruction" path consume the sparse `ball_keyframes.json` (Phase A) — keying the ball transform track only at anchor frames with per-state interpolation, placing null-depth airborne keys on their ray, and giving `BP_BallActor` motion-derived rolling spin plus per-flight-span preset curl.

**Architecture:** Pure, `unreal`-free logic (`ball_keyframes.py` parse, `ball_motion.py` interpolation/ray-fallback/curl) is unit-tested with the pipeline venv. Three early in-editor spikes nail the uncertain MovieScene/Blueprint Python APIs. The thin `import unreal` glue (`load_reconstruction`, `build_sequence`) and the `BP_BallActor` spin graph are then built against the spike findings and verified via probe scripts + MCP screenshots.

**Tech Stack:** Python 3 (frozen dataclasses, pytest), Unreal Engine 5.8 Python (`unreal` MovieScene scripting), UE MCP toolsets (BlueprintTools, ObjectTools, ActorTools, EditorAppToolset, LogsToolset, ProgrammaticToolset).

**Spec:** `docs/superpowers/specs/2026-06-05-ball-keyframes-engine-phase-b-design.md`

**Repos / working dirs:**
- UE Python (NON-git — do not `git commit` there): `/Users/joebower/workplace/FootballPerspectives 5.8/Content/Python/`
- Pipeline venv for the UE-side unit tests: `/Users/joebower/workplace/football-perspectives/.venv/bin/python`
- UE unit-test command: `cd "/Users/joebower/workplace/FootballPerspectives 5.8/Content/Python" && /Users/joebower/workplace/football-perspectives/.venv/bin/python -m pytest tests -q`

---

## File Structure

UE Python dir `FootballPerspectives 5.8/Content/Python/`:
- Create: `football_perspectives/ball_keyframes.py` — unreal-free parse of `ball_keyframes.json` → `BallKeyframe`/`BallKeyframeSet`.
- Create: `football_perspectives/ball_motion.py` — unreal-free pure helpers: `resolved_position_m`, `key_interp_modes`, `flight_curls`.
- Create: `tests/test_ball_keyframes.py`, `tests/test_ball_motion.py` — unit tests (pipeline venv).
- Create: `_smoke/notes_phase_b.md` — recorded spike findings (the exact UE API to use).
- Create: `_smoke/probe_ball_spin_tick.py`, `_smoke/probe_ball_key_interp.py`, `_smoke/probe_ball_var_keying.py` — spike probes.
- Modify: `football_perspectives/load_reconstruction.py` (`_load_ball_keys` + `build` call) — keyframes-aware load with dense fallback.
- Modify: `football_perspectives/build_sequence.py` (`_add_ball_spawnable` + `build` signature) — sparse keying, per-key interp, per-flight curl keying.
- Create: `_smoke/probe_ball_keyframes.py` — end-to-end in-editor verification.
- Asset: `/Game/Players/BP_BallActor` — add spin variables + Tick graph (via MCP).

> The UE directory is not under git. After each task, **report the changed files** so they can be tracked manually. The `git add`/`commit` steps below apply ONLY to files inside the pipeline repo (none in this plan except notes you may choose to mirror); for UE files, substitute "save the file(s) and report them."

---

## PART 1 — Unreal-free core (unit-tested with the pipeline venv)

### Task 1: `ball_keyframes.py` — parse the sidecar

**Files:**
- Create: `FootballPerspectives 5.8/Content/Python/football_perspectives/ball_keyframes.py`
- Test: `FootballPerspectives 5.8/Content/Python/tests/test_ball_keyframes.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ball_keyframes.py
"""Unreal-free parse of the Phase A ball_keyframes.json contract."""
from __future__ import annotations

import json

from football_perspectives import ball_keyframes


def _write(tmp_path, payload):
    p = tmp_path / "ball_keyframes.json"
    p.write_text(json.dumps(payload))
    return p


def test_parses_world_and_ray_and_spin(tmp_path):
    path = _write(tmp_path, {
        "clip_id": "c", "fps": 25.0, "image_size": [1920, 1080],
        "keyframes": [
            {"frame": 4, "state": "grounded", "depth_source": "ground",
             "world_xyz": [1.0, 2.0, 0.11], "image_xy": [800.0, 600.0]},
            {"frame": 9, "state": "airborne_high", "depth_source": "ray_physics",
             "world_xyz": None, "image_xy": [820.0, 300.0],
             "ray": [[0.0, 0.0, 15.0], [0.1, 0.2, -0.97]]},
            {"frame": 12, "state": "player_touch", "depth_source": "player_bone",
             "world_xyz": [3.0, 4.0, 1.8], "image_xy": [900.0, 400.0],
             "player_id": "P001", "bone": "head",
             "touch_type": "shot", "spin": "topspin"},
        ],
    })
    kfset = ball_keyframes.load(path)
    assert kfset.fps == 25.0
    assert [k.frame for k in kfset.keyframes] == [4, 9, 12]
    g, air, touch = kfset.keyframes
    assert g.world_xyz == (1.0, 2.0, 0.11)
    assert g.ray is None
    assert air.world_xyz is None
    assert air.ray == ((0.0, 0.0, 15.0), (0.1, 0.2, -0.97))
    assert touch.spin == "topspin"
    assert touch.touch_type == "shot"


def test_missing_optional_fields_default_to_none(tmp_path):
    path = _write(tmp_path, {
        "clip_id": "c", "fps": 30.0, "image_size": [1920, 1080],
        "keyframes": [
            {"frame": 0, "state": "grounded", "depth_source": "ground",
             "world_xyz": [0.0, 0.0, 0.11], "image_xy": [1.0, 1.0]},
        ],
    })
    kf = ball_keyframes.load(path).keyframes[0]
    assert kf.ray is None
    assert kf.spin is None
    assert kf.touch_type is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd "/Users/joebower/workplace/FootballPerspectives 5.8/Content/Python" && /Users/joebower/workplace/football-perspectives/.venv/bin/python -m pytest tests/test_ball_keyframes.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'football_perspectives.ball_keyframes'`

- [ ] **Step 3: Write minimal implementation**

```python
# football_perspectives/ball_keyframes.py
"""Unreal-free parse of the Phase A ``ball_keyframes.json`` sidecar.

Mirrors ``manifest.py`` in spirit: importable with a plain pytest (no
``import unreal``), so the parse logic is unit-tested with the pipeline
venv. Only the fields the UE consumer needs are surfaced.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

Vec3 = Tuple[float, float, float]
Ray = Tuple[Vec3, Vec3]


@dataclass(frozen=True)
class BallKeyframe:
    frame: int
    state: str
    world_xyz: Optional[Vec3] = None
    ray: Optional[Ray] = None
    spin: Optional[str] = None
    touch_type: Optional[str] = None


@dataclass(frozen=True)
class BallKeyframeSet:
    fps: float
    keyframes: Tuple[BallKeyframe, ...]


def _vec3(v) -> Vec3:
    return (float(v[0]), float(v[1]), float(v[2]))


def _ray(v) -> Optional[Ray]:
    if v is None:
        return None
    return (_vec3(v[0]), _vec3(v[1]))


def load(path) -> BallKeyframeSet:
    """Parse ``ball_keyframes.json`` at ``path`` into a ``BallKeyframeSet``."""
    data = json.loads(Path(path).read_text())
    kfs: List[BallKeyframe] = []
    for k in data.get("keyframes", []):
        world = k.get("world_xyz")
        kfs.append(BallKeyframe(
            frame=int(k["frame"]),
            state=str(k["state"]),
            world_xyz=_vec3(world) if world is not None else None,
            ray=_ray(k.get("ray")),
            spin=k.get("spin"),
            touch_type=k.get("touch_type"),
        ))
    return BallKeyframeSet(fps=float(data["fps"]), keyframes=tuple(kfs))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd "/Users/joebower/workplace/FootballPerspectives 5.8/Content/Python" && /Users/joebower/workplace/football-perspectives/.venv/bin/python -m pytest tests/test_ball_keyframes.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Save & report** (UE dir is non-git)

Report: created `football_perspectives/ball_keyframes.py` and `tests/test_ball_keyframes.py`.

---

### Task 2: `ball_motion.resolved_position_m` — world or ray-fallback

**Files:**
- Create: `FootballPerspectives 5.8/Content/Python/football_perspectives/ball_motion.py`
- Test: `FootballPerspectives 5.8/Content/Python/tests/test_ball_motion.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ball_motion.py
"""Unreal-free ball-motion helpers (interp, ray-fallback, curl)."""
from __future__ import annotations

from football_perspectives import ball_motion
from football_perspectives.ball_keyframes import BallKeyframe


def test_resolved_position_uses_world_xyz_when_present():
    kf = BallKeyframe(frame=1, state="grounded", world_xyz=(3.0, 4.0, 0.11))
    assert ball_motion.resolved_position_m(kf) == (3.0, 4.0, 0.11)


def test_resolved_position_ray_fallback_hits_canonical_height():
    # Ray from (0,0,20) straight down (-z); airborne_high canonical z = 15.0.
    kf = BallKeyframe(
        frame=2, state="airborne_high", world_xyz=None,
        ray=((0.0, 0.0, 20.0), (0.0, 0.0, -1.0)),
    )
    pos = ball_motion.resolved_position_m(kf)
    assert pos is not None
    assert abs(pos[2] - 15.0) < 1e-9
    assert abs(pos[0]) < 1e-9 and abs(pos[1]) < 1e-9


def test_resolved_position_none_when_no_world_and_no_ray():
    kf = BallKeyframe(frame=3, state="airborne_mid", world_xyz=None, ray=None)
    assert ball_motion.resolved_position_m(kf) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd "/Users/joebower/workplace/FootballPerspectives 5.8/Content/Python" && /Users/joebower/workplace/football-perspectives/.venv/bin/python -m pytest tests/test_ball_motion.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'football_perspectives.ball_motion'`

- [ ] **Step 3: Write minimal implementation**

```python
# football_perspectives/ball_motion.py
"""Unreal-free ball-motion helpers consumed by build_sequence.

All math is in PITCH metres (x along touchline, y across, z up), matching
the ball_keyframes contract. build_sequence applies the pitch→UE axis-swap
+ offset; this module stays pitch-native so it is unit-testable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

from football_perspectives.ball_keyframes import BallKeyframe

Vec3 = Tuple[float, float, float]

_AIRBORNE_STATES = frozenset({"airborne_low", "airborne_mid", "airborne_high"})
# Mirror src/utils/ball_anchor_heights._STATE_HEIGHT_M (pipeline side).
_CANONICAL_HEIGHT_M = {"airborne_low": 1.0, "airborne_mid": 6.0, "airborne_high": 15.0}


def resolved_position_m(kf: BallKeyframe) -> Optional[Vec3]:
    """Pitch-metre position for a keyframe.

    Uses ``world_xyz`` when present (already ray-faithful at physics depth).
    For an ``airborne_*`` keyframe with no world (under-determined depth),
    intersect the exported ``ray`` with the state's canonical-height plane.
    Returns ``None`` when neither is available.
    """
    if kf.world_xyz is not None:
        return kf.world_xyz
    if kf.ray is not None and kf.state in _CANONICAL_HEIGHT_M:
        (ox, oy, oz), (dx, dy, dz) = kf.ray
        if abs(dz) < 1e-9:
            return None
        z = _CANONICAL_HEIGHT_M[kf.state]
        t = (z - oz) / dz
        return (ox + t * dx, oy + t * dy, oz + t * dz)
    return None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd "/Users/joebower/workplace/FootballPerspectives 5.8/Content/Python" && /Users/joebower/workplace/football-perspectives/.venv/bin/python -m pytest tests/test_ball_motion.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Save & report** created `football_perspectives/ball_motion.py`, `tests/test_ball_motion.py`.

---

### Task 3: `ball_motion.key_interp_modes` — per-key cubic/linear

**Files:**
- Modify: `football_perspectives/ball_motion.py` (append)
- Test: `tests/test_ball_motion.py` (append)

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_ball_motion.py

def _kf(frame, state):
    return BallKeyframe(frame=frame, state=state, world_xyz=(0.0, 0.0, 0.0))


def test_interp_cubic_for_airborne_context_linear_for_ground():
    seq = [
        _kf(0, "grounded"),      # ground, next is kick(ground) -> linear
        _kf(1, "kick"),          # ground, next airborne -> cubic (neighbour airborne)
        _kf(2, "airborne_high"), # airborne -> cubic
        _kf(3, "bounce"),        # ground, prev airborne -> cubic
        _kf(4, "grounded"),      # ground, both neighbours ground -> linear
        _kf(5, "grounded"),      # ground, prev ground, last -> linear
    ]
    assert ball_motion.key_interp_modes(seq) == [
        "linear", "cubic", "cubic", "cubic", "linear", "linear",
    ]


def test_interp_single_keyframe_is_linear():
    assert ball_motion.key_interp_modes([_kf(0, "grounded")]) == ["linear"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd "/Users/joebower/workplace/FootballPerspectives 5.8/Content/Python" && /Users/joebower/workplace/football-perspectives/.venv/bin/python -m pytest tests/test_ball_motion.py -k interp -v`
Expected: FAIL — `AttributeError: module 'football_perspectives.ball_motion' has no attribute 'key_interp_modes'`

- [ ] **Step 3: Write minimal implementation** (append to `ball_motion.py`)

```python
def key_interp_modes(keyframes: List[BallKeyframe]) -> List[str]:
    """Per-key interpolation: ``"cubic"`` if the key's own state is
    ``airborne_*`` or either neighbour is ``airborne_*`` (smooth flight
    arcs); otherwise ``"linear"`` (ground rolls stay flat).
    """
    n = len(keyframes)
    modes: List[str] = []
    for i, kf in enumerate(keyframes):
        own = kf.state in _AIRBORNE_STATES
        prev_air = i > 0 and keyframes[i - 1].state in _AIRBORNE_STATES
        next_air = i < n - 1 and keyframes[i + 1].state in _AIRBORNE_STATES
        modes.append("cubic" if (own or prev_air or next_air) else "linear")
    return modes
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd "/Users/joebower/workplace/FootballPerspectives 5.8/Content/Python" && /Users/joebower/workplace/football-perspectives/.venv/bin/python -m pytest tests/test_ball_motion.py -k interp -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Save & report.**

---

### Task 4: `ball_motion.flight_curls` — per-flight-span curl

**Files:**
- Modify: `football_perspectives/ball_motion.py` (append)
- Test: `tests/test_ball_motion.py` (append)

Axis semantics mirror `src/utils/ball_spin_presets.omega_seed_from_preset` (pitch frame): side curls → vertical ±z; `topspin`/`backspin` → horizontal axis ⊥ travel (rotate horizontal v0 +90° about z, flip for backspin); `none`/`knuckle`/no-preset → zero strength. Strength is the preset's rad/s magnitude converted to deg/s. build_sequence applies the pitch→UE swap to the axis (Task 11).

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_ball_motion.py
import math


def _touch(frame, spin, x=0.0, y=0.0, z=0.0):
    return BallKeyframe(frame=frame, state="player_touch", world_xyz=(x, y, z),
                        touch_type="shot", spin=spin)


def _air(frame, x, y, z):
    return BallKeyframe(frame=frame, state="airborne_high", world_xyz=(x, y, z))


def _ground(frame, x, y):
    return BallKeyframe(frame=frame, state="grounded", world_xyz=(x, y, 0.0))


def test_single_flight_instep_curl_right_vertical_axis():
    seq = [_touch(10, "instep_curl_right", 0, 0, 0),
           _air(15, 5, 0, 10),
           _ground(20, 10, 0)]
    curls = ball_motion.flight_curls(seq)
    assert len(curls) == 1
    c = curls[0]
    assert c.launch_frame == 10
    assert c.curl_axis == (0.0, 0.0, 1.0)
    assert abs(c.curl_strength_deg_s - 15.0 * 180.0 / math.pi) < 1e-6


def test_two_flights_each_get_their_own_curl():
    seq = [_touch(10, "instep_curl_right", 0, 0, 0),
           _air(15, 5, 0, 10),
           _ground(20, 10, 0),
           _touch(30, "instep_curl_left", 10, 0, 0),
           _air(35, 15, 0, 10),
           _ground(40, 20, 0)]
    curls = ball_motion.flight_curls(seq)
    assert [c.launch_frame for c in curls] == [10, 30]
    assert curls[0].curl_axis == (0.0, 0.0, 1.0)
    assert curls[1].curl_axis == (0.0, 0.0, -1.0)


def test_flight_with_no_spin_launch_has_zero_strength():
    seq = [_ground(10, 0, 0),
           _air(15, 5, 0, 10),   # launch is the grounded key (no spin)
           _ground(20, 10, 0)]
    curls = ball_motion.flight_curls(seq)
    assert len(curls) == 1
    assert curls[0].curl_strength_deg_s == 0.0


def test_topspin_axis_perpendicular_to_travel():
    # Travel +x (launch at origin, first air at +x). topspin axis = (0,+1,0).
    seq = [_touch(10, "topspin", 0, 0, 0),
           _air(15, 5, 0, 10),
           _ground(20, 10, 0)]
    c = ball_motion.flight_curls(seq)[0]
    assert abs(c.curl_axis[0]) < 1e-9
    assert abs(c.curl_axis[1] - 1.0) < 1e-9
    assert abs(c.curl_axis[2]) < 1e-9
    assert abs(c.curl_strength_deg_s - 50.0 * 180.0 / math.pi) < 1e-6
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd "/Users/joebower/workplace/FootballPerspectives 5.8/Content/Python" && /Users/joebower/workplace/football-perspectives/.venv/bin/python -m pytest tests/test_ball_motion.py -k curl -v`
Expected: FAIL — `AttributeError: ... has no attribute 'flight_curls'`

- [ ] **Step 3: Write minimal implementation** (append to `ball_motion.py`)

```python
# Spin magnitudes mirror src/utils/ball_spin_presets.py (rad/s).
_CURL_RAD_S = 15.0
_SPIN_RAD_S = 50.0
_RAD_TO_DEG = 180.0 / math.pi


@dataclass(frozen=True)
class FlightCurl:
    launch_frame: int
    curl_axis: Vec3              # unit axis in PITCH frame
    curl_strength_deg_s: float


def _curl_for_preset(preset: Optional[str],
                     v0_h: Optional[Tuple[float, float]]) -> Tuple[Vec3, float]:
    if preset is None or preset in ("none", "knuckle"):
        return (0.0, 0.0, 1.0), 0.0
    if preset in ("instep_curl_right", "outside_curl_left"):
        return (0.0, 0.0, 1.0), _CURL_RAD_S * _RAD_TO_DEG
    if preset in ("instep_curl_left", "outside_curl_right"):
        return (0.0, 0.0, -1.0), _CURL_RAD_S * _RAD_TO_DEG
    if preset in ("topspin", "backspin"):
        if v0_h is None:
            return (0.0, 0.0, 1.0), 0.0
        vx, vy = v0_h
        n = math.hypot(vx, vy)
        if n < 1e-6:
            return (0.0, 0.0, 1.0), 0.0
        ax, ay = -vy / n, vx / n            # rotate +90° about z
        if preset == "backspin":
            ax, ay = -ax, -ay
        return (ax, ay, 0.0), _SPIN_RAD_S * _RAD_TO_DEG
    # Unknown preset → no curl (defensive; pipeline validates upstream).
    return (0.0, 0.0, 1.0), 0.0


def flight_curls(keyframes: List[BallKeyframe]) -> List["FlightCurl"]:
    """One ``FlightCurl`` per flight span (maximal run of ``airborne_*``
    keys). The launch is the key preceding the run (or the run's first key
    if it starts the clip); its spin preset (only honoured on a
    ``player_touch`` shot/volley) and the flight's initial travel direction
    determine the curl. No-spin launches yield zero strength.
    """
    out: List[FlightCurl] = []
    n = len(keyframes)
    i = 0
    while i < n:
        if keyframes[i].state not in _AIRBORNE_STATES:
            i += 1
            continue
        j = i
        while j + 1 < n and keyframes[j + 1].state in _AIRBORNE_STATES:
            j += 1
        launch = keyframes[i - 1] if i > 0 else keyframes[i]
        p_launch = resolved_position_m(launch)
        p_air = resolved_position_m(keyframes[i])
        v0_h: Optional[Tuple[float, float]] = None
        if p_launch is not None and p_air is not None:
            v0_h = (p_air[0] - p_launch[0], p_air[1] - p_launch[1])
        preset = launch.spin if launch.touch_type in ("shot", "volley") else None
        axis, strength = _curl_for_preset(preset, v0_h)
        out.append(FlightCurl(launch_frame=launch.frame,
                              curl_axis=axis, curl_strength_deg_s=strength))
        i = j + 1
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd "/Users/joebower/workplace/FootballPerspectives 5.8/Content/Python" && /Users/joebower/workplace/football-perspectives/.venv/bin/python -m pytest tests/test_ball_motion.py -q`
Expected: PASS (all ball_motion tests)

- [ ] **Step 5: Save & report.**

---

## PART 2 — In-editor spikes (resolve the uncertain UE APIs)

Each spike is a probe script run in the editor (via the MCP `ProgrammaticToolset.execute_tool_script` or the editor Python console). **Record the working API in `_smoke/notes_phase_b.md`** — Part 3/4 code references those findings. A spike "passes" when the finding is recorded (positive or negative); a negative finding selects the documented fallback.

### Task 5: Spike — does a spawnable's Tick run during Sequencer evaluation?

**Files:** Create `_smoke/probe_ball_spin_tick.py`; append finding to `_smoke/notes_phase_b.md`.

- [ ] **Step 1:** Write `_smoke/probe_ball_spin_tick.py` that: opens `BP_BallActor`, adds a temporary `EventTick` that calls `AddActorWorldRotation` on the actor by a fixed yaw/sec, compiles, spawns it into a throwaway Level Sequence with two location keys, evaluates/plays the sequence, and logs the actor's rotation at the start vs end frame. (Use existing `_smoke/probe_sequence.py` as the reference for creating a sequence + binding.)
- [ ] **Step 2:** Run it in-editor (MCP `execute_tool_script` or editor console). Read `GetLogEntries` for the logged rotations.
- [ ] **Step 3:** Record in `_smoke/notes_phase_b.md` under "Spike 1": whether the rotation advanced (Tick ran during eval) — YES → spin can live in the BP Tick (the plan's default). NO → record the fallback decision: curl spin will be keyed on the transform's rotation channels in `build_sequence` instead, and rolling spin dropped to a fixed flight-only spin. Remove the temporary EventTick from `BP_BallActor` afterwards.
- [ ] **Step 4: Report** the finding.

### Task 6: Spike — per-key interpolation/tangent mode on a transform channel from Python

**Files:** Create `_smoke/probe_ball_key_interp.py`; append finding to `_smoke/notes_phase_b.md`.

- [ ] **Step 1:** Write a probe that creates a throwaway sequence + a `MovieScene3DTransformTrack`, adds keys via `channel.add_key(...)`, then attempts to set per-key interpolation to cubic vs linear. Try, in order, and record which works in UE 5.8:
  (a) `add_key(frame, value, interpolation=unreal.MovieSceneKeyInterpolation.CUBIC)` if the kwarg exists;
  (b) the returned key handle's `set_interp_mode` / setting `RichCurveKey` tangent mode;
  (c) `unreal.SequencerScriptingMixinLibrary` / `MovieSceneScriptingChannel` key APIs (`set_interpolation`).
- [ ] **Step 2:** Run in-editor; inspect the resulting keys (read back interpolation) and/or screenshot the curve editor via `CaptureAssetImage`.
- [ ] **Step 3:** Record the EXACT working call in `_smoke/notes_phase_b.md` "Spike 2" (this becomes the `_set_key_interp(channel, frame, mode)` helper signature used in Task 11). If none works from Python, record the fallback: key all-cubic (accept occasional ground bow) and note the limitation.
- [ ] **Step 4: Report.**

### Task 7: Spike — keying an exposed Blueprint variable on a spawnable binding

**Files:** Create `_smoke/probe_ball_var_keying.py`; append finding to `_smoke/notes_phase_b.md`.

- [ ] **Step 1:** Add two temporary instance-editable, cinematics-exposed variables to `BP_BallActor` via MCP BlueprintTools (`add_variable` `CurlStrengthDegPerSec: float`, `add_struct_variable` `CurlAxis: Vector`), compile. Write a probe that spawns the ball into a throwaway sequence and tries to add a property track for `CurlStrengthDegPerSec` on the binding and key it (step interpolation) at two frames. Try: `binding.add_track(unreal.MovieSceneFloatTrack)` then `track.set_property_name_and_path("CurlStrengthDegPerSec", "CurlStrengthDegPerSec")`, add a section, key the channel.
- [ ] **Step 2:** Run in-editor; read back the keyed values / screenshot the binding's track.
- [ ] **Step 3:** Record in `_smoke/notes_phase_b.md` "Spike 3" the exact working track type + property-path call (used by Task 11's `_key_ball_curl`). If keying a BP var on a spawnable is not possible from Python, record the fallback: set a single template default via `binding.get_object_template().set_editor_property(...)` (one curl per clip) and log the lost multi-flight fidelity.
- [ ] **Step 4: Report.**

---

## PART 3 — `BP_BallActor` spin graph (authored via MCP, gated on Spike 1)

### Task 8: Add spin member variables to `BP_BallActor`

**Files:** Asset `/Game/Players/BP_BallActor` (MCP BlueprintTools / ObjectTools).

- [ ] **Step 1:** Via MCP, add member variables to `BP_BallActor` (all instance-editable; `CurlAxis`/`CurlStrengthDegPerSec` also cinematics-exposed per Spike 3):
  - `PrevLocation` : Vector (transient)
  - `BallRadiusCm` : Float, default `11.0`
  - `SpinEnabled` : Bool, default `true`
  - `RollMultiplier` : Float, default `1.0`
  - `CurlAxis` : Vector, default `(0,0,1)`
  - `CurlStrengthDegPerSec` : Float, default `0.0`
  (Reuse the temporary `CurlAxis`/`CurlStrengthDegPerSec` from Spike 3 if already present — verify with `list_variables`.)
- [ ] **Step 2:** `compile_blueprint`. Confirm via `list_variables` that all six exist.
- [ ] **Step 3: Report** the variable list.

### Task 9: Build the Tick spin graph in `BP_BallActor`

**Files:** Asset `/Game/Players/BP_BallActor` EventGraph (MCP BlueprintTools).

Implements the spec's Tick logic: compute velocity from `PrevLocation`; if airborne (`GetActorLocation().Z > BallRadiusCm * 1.5`) rotate `MeshComp` by `CurlAxis * CurlStrengthDegPerSec * DeltaSeconds`; else roll `MeshComp` about `normalize(cross((0,0,1), vel))` by `degrees(speed / BallRadiusCm) * RollMultiplier * DeltaSeconds`; then store `PrevLocation`. Guard on `SpinEnabled` and `DeltaSeconds > 0`.

- [ ] **Step 1:** With MCP BlueprintTools, build the EventTick graph node-by-node (`create_node`, `connect_pins`, `set_pin_value`): branch on `SpinEnabled`; `GetActorLocation`; `vel = (loc - PrevLocation) / DeltaSeconds`; `speed = VectorLength(vel)`; `Branch (loc.Z > BallRadiusCm*1.5)`; airborne path → `MeshComp.AddWorldRotation` (build a Rotator from `CurlAxis * CurlStrengthDegPerSec * DeltaSeconds` via `RotatorFromAxisAndAngle`); ground path → compute roll axis + angle, `MeshComp.AddWorldRotation`; both → `Set PrevLocation = loc`. Use `find_node_types`/`find_node_categories` to locate exact node names; lay out with `set_node_position` and `layout`-style spacing.
- [ ] **Step 2:** `compile_blueprint`; read `GetLogEntries` for compile errors and fix any dangling pins (`get_graph_info` to inspect).
- [ ] **Step 3:** Visual smoke: spawn `BP_BallActor` in the level at a height (e.g. Z=300), nudge it with `set_actor_transform` over a couple of frames or set `CurlStrengthDegPerSec` and use the probe from Spike 1, and `CaptureEditorImage` to confirm the mesh visibly rotates. (Rolling needs motion; the airborne-curl branch can be checked by setting Z high + `CurlStrengthDegPerSec` and observing rotation.)
- [ ] **Step 4: Report** with the captured image reference + compile status.

---

## PART 4 — UE glue (load + build_sequence), gated on Spikes 2 & 3

### Task 10: `load_reconstruction` — keyframes-aware ball load with dense fallback

**Files:** Modify `football_perspectives/load_reconstruction.py` (`_load_ball_keys` ~line 325, and the `build(... ball_keys=...)` call ~line 91). Verify via `_smoke/probe_ball_keyframes.py` (created in Task 12).

Introduce a richer return type so `build_sequence` gets positions + interp + curls. Keep the dense fallback returning the same shape with all-linear interp and no curls.

- [ ] **Step 1:** Add a `BallMotionData` dataclass near the top of `load_reconstruction.py`:

```python
from dataclasses import dataclass, field

@dataclass
class BallMotionData:
    # (frame, x_m, y_m, z_m) per anchor key, pitch metres.
    keys: list  # list[tuple[int, float, float, float]]
    interp_modes: list = field(default_factory=list)   # "cubic"|"linear" per key
    flight_curls: list = field(default_factory=list)   # ball_motion.FlightCurl
    sparse: bool = True   # False when produced by the dense fallback
```

- [ ] **Step 2:** Replace `_load_ball_keys` with a keyframes-first loader (keep the dense path as `_load_ball_keys_dense`):

```python
from football_perspectives import ball_keyframes, ball_motion

def _load_ball_motion(base, m) -> BallMotionData:
    """Prefer the sparse keyframes sidecar; fall back to the dense track."""
    if m.ball is not None and m.ball.keyframes_json:
        path = base / m.ball.keyframes_json
        if path.exists():
            kfset = ball_keyframes.load(path)
            kfs = list(kfset.keyframes)
            keys = []
            kept = []
            for kf in kfs:
                pos = ball_motion.resolved_position_m(kf)
                if pos is None:
                    continue
                keys.append((kf.frame, pos[0], pos[1], pos[2]))
                kept.append(kf)
            if keys:
                return BallMotionData(
                    keys=keys,
                    interp_modes=ball_motion.key_interp_modes(kept),
                    flight_curls=ball_motion.flight_curls(kept),
                    sparse=True,
                )
        unreal.log_warning(f"ball keyframes sidecar missing/empty at {path}; "
                            f"falling back to dense track")
    dense = _load_ball_keys_dense(base, m)
    return BallMotionData(keys=dense, interp_modes=["linear"] * len(dense),
                          flight_curls=[], sparse=False)
```

Rename the existing `_load_ball_keys` body to `_load_ball_keys_dense` (unchanged logic).

- [ ] **Step 3:** Update the `build(...)` call site (~line 85-98) to pass the motion data. Change `ball_keys = _load_ball_keys(base, m)` to `ball_motion_data = _load_ball_motion(base, m)` and pass `ball_motion=ball_motion_data` to `build_sequence.build(...)` (Task 11 adds that parameter; keep `ball_keys=ball_motion_data.keys` too if you stage the change, but the final form passes the struct).
- [ ] **Step 4:** Unit-cover the unreal-free slice: the parsing/positioning is already covered by Tasks 1–4. For `_load_ball_motion`, add a note in `notes_phase_b.md` that it is verified end-to-end by the Task 12 probe (it imports `unreal` for logging, so it is not unit-tested).
- [ ] **Step 5: Report** changed files.

### Task 11: `build_sequence._add_ball_spawnable` — sparse keys, per-key interp, per-flight curl

**Files:** Modify `football_perspectives/build_sequence.py` (`build` signature ~line 38; `_add_ball_spawnable` ~line 269; call site inside `build`). Uses Spike 2's `_set_key_interp` and Spike 3's `_key_ball_curl` findings.

- [ ] **Step 1:** Change `build(...)` to accept `ball_motion=None` (a `BallMotionData`) in addition to/in place of `ball_keys`, and pass it to `_add_ball_spawnable`. Keep accepting the legacy `ball_keys` list for back-compat (wrap into a `sparse=False` motion if only `ball_keys` is given).
- [ ] **Step 2:** Generalise `_add_ball_spawnable` to:
  - key the position channels at the motion's anchor frames (same axis-swap + offset as today);
  - apply per-key interpolation via the Spike-2 helper `_set_key_interp(channel, frame, mode)` using `motion.interp_modes`;
  - for each `FlightCurl`, swap its axis pitch→UE (`(ax, ay, az) → (ay, ax, az)`, matching the position swap) and key `CurlAxis` + `CurlStrengthDegPerSec` on the binding at `launch_frame` with STEP interpolation via the Spike-3 helper `_key_ball_curl(binding, frame, axis_ue, strength)`.
  - when `motion.sparse is False`, behave exactly as today (linear keys, no curl) — back-compat.

Intended shape (adapt the interp/curl calls to the Spike findings recorded in `notes_phase_b.md`):

```python
def _add_ball_spawnable(seq, bp_ball, motion, offset_x_cm, offset_y_cm, yaw_deg):
    if not motion.keys:
        return
    binding = seq.add_spawnable_from_class(bp_ball.generated_class())
    binding.set_display_name("ball")
    start_frame = motion.keys[0][0]
    end_frame = motion.keys[-1][0] + 1
    transform_track = binding.add_track(unreal.MovieScene3DTransformTrack)
    section = transform_track.add_section()
    section.set_range(start_frame, end_frame)
    channels = section.get_all_channels()
    for idx, val in ((3, 0.0), (4, 0.0), (5, 0.0), (6, 1.0), (7, 1.0), (8, 1.0)):
        channels[idx].set_default(val)
    loc_x, loc_y, loc_z = channels[0], channels[1], channels[2]
    modes = motion.interp_modes or ["linear"] * len(motion.keys)
    for (frame, x_m, y_m, z_m), mode in zip(motion.keys, modes):
        fn = unreal.FrameNumber(int(frame))
        loc_x.add_key(fn, float(y_m * 100.0 + offset_x_cm))
        loc_y.add_key(fn, float(x_m * 100.0 + offset_y_cm))
        loc_z.add_key(fn, float(z_m * 100.0))
        if motion.sparse:
            _set_key_interp(loc_x, fn, mode)   # Spike 2 helper
            _set_key_interp(loc_y, fn, mode)
            _set_key_interp(loc_z, fn, mode)
    for fc in motion.flight_curls:             # Spike 3 helper
        ax, ay, az = fc.curl_axis
        axis_ue = (ay, ax, az)                 # pitch→UE swap, matches position
        _key_ball_curl(binding, int(fc.launch_frame), axis_ue, fc.curl_strength_deg_s)
    _constrain_spawn_lifetime(binding, start_frame, end_frame)
```

Add `_set_key_interp` and `_key_ball_curl` as small helpers whose bodies are exactly the working calls recorded in Spikes 2 & 3 (fallbacks: `_set_key_interp` no-ops if Python can't set interp; `_key_ball_curl` sets the template default once if it can't key).

- [ ] **Step 3:** `compile`-free Python sanity: `cd "<UE>/Content/Python" && <venv>/python -c "import ast; ast.parse(open('football_perspectives/build_sequence.py').read())"` to confirm the file parses (it imports `unreal`, so cannot be imported outside the editor).
- [ ] **Step 4:** Verified end-to-end by Task 12. Report changed files.

---

## PART 5 — Integration verification

### Task 12: End-to-end in-editor probe + visual check

**Files:** Create `_smoke/probe_ball_keyframes.py`.

- [ ] **Step 1:** Write a probe that runs the real load path on a clip whose manifest carries `keyframes_json` (use an existing reconstruction under `Content/Reconstructions/` that has a sidecar, or stage one). It should call `load_reconstruction` (or the EUW entry) and then introspect the resulting `LS_<clip>`:
  - find the `ball` binding;
  - assert its transform track has a SPARSE set of keys (count == number of resolved anchors, not the full frame span);
  - assert per-key interpolation matches `ball_motion.key_interp_modes` for the clip's keyframes;
  - assert a `CurlStrengthDegPerSec`/`CurlAxis` track (or template default, per Spike 3) is present with the expected per-flight values.
- [ ] **Step 2:** Run in-editor; capture the Sequencer/viewport with `CaptureEditorImage` and read `GetLogEntries`. Confirm the ball tweens a smooth arc through flight and rolls on the ground; for a clip with a spin-tagged shot, confirm visible curl.
- [ ] **Step 3:** Back-compat check: run the same probe on a clip WITHOUT `keyframes_json` and confirm the ball still loads via the dense fallback (every-frame linear keys, no curl track), unchanged from current behaviour.
- [ ] **Step 4: Report** with captured images + a short pass/fail per acceptance bullet.

---

## Self-Review notes (for the executor)

- Tasks 1–4 are fully unit-tested with the pipeline venv and are the logic core; do them first and keep them green.
- Tasks 5–7 (spikes) GATE the implementation specifics of Tasks 9 & 11. Do not write the final `_set_key_interp` / `_key_ball_curl` / BP-Tick bodies before recording the spike findings — substitute the documented fallback if a spike is negative.
- The UE directory is NOT git-tracked: report changed files each task instead of committing.
- Coordinate frames: `ball_motion` is pitch-native; `build_sequence` owns the pitch→UE axis-swap for BOTH positions and curl axes. Keep the swap in one place.

## Phase-B Done Criteria

- Unreal-free `ball_keyframes` + `ball_motion` modules pass their unit tests.
- Loading a clip with `keyframes_json` yields a ball with sparse, per-state-interpolated transform keys, ray-fallback placement for null-depth airborne anchors, and per-flight-span curl; the ball rolls with motion-derived spin and curls a spin-tagged shot (visual capture).
- A clip without `keyframes_json` loads unchanged via the dense fallback.
- Spike findings (and any fallbacks taken) are recorded in `_smoke/notes_phase_b.md`.
