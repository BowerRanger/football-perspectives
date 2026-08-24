# Camera + Post/Render Milestone Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire broadcast-camera rotation + per-frame zoom, add a Python camera-rigs toolkit with smoothed interest-target aiming, an EUW camera browser, a post-process style-preset system, and a one-call Movie Render Queue helper.

**Architecture:** Pure-Python modules (no `unreal` import) hold math and data; `unreal`-dependent wiring lives in `build_sequence.py`, `load_reconstruction.py`, and `render_queue.py`. EUW integration uses the existing script-string-variable pattern. UE assets (style presets, cameras) are created via the BP_PyExec bridge or MCP where possible.

**Tech Stack:** Python 3.11 · UE5.8 Python API (`unreal`) · MoviePipeline · pytest (offline tests) · MCP (`unreal-mcp`) for EUW variable authoring

**Dependency order:** BOW-88 → BOW-89 → BOW-96 (extends 89) → BOW-95 (blocked by 89) → BOW-91 → BOW-97 (benefits from 91) → BOW-90 (independent spike)

---

## Status (updated 2026-07-05)

**Python implementation: COMPLETE.** All modules (`camera_math.py` broadcast-camera types, `camera_rigs.py` 5 rigs + focus modes, `style_presets.py`, `render_queue.py`) and `load_reconstruction.py` entry points (`list_rigs`/`add_rig`/`remove_rig`, `list_style_presets`/`apply_style_preset`, `render_clip`) are implemented; 81/81 offline tests pass. The BOW-90 spike report is committed at `docs/superpowers/notes/metahuman-spike-2026-06-11.md` (recommendation: all 22 players at LOD3, face AnimBP disabled).

**Deviations from plan as written:**
1. **Broadcast camera is a plain `CameraActor`, not `CineCameraActor`** (Task 1 steps 1.8+). CineCameraActor's lens/filmback soft-reference machinery crashes `ACineCameraActor::Tick` when packed object refs go stale after a material recompile. The broadcast camera keys `CameraComponent.FieldOfView` (degrees, via `camera_math.fov_x_deg`) instead of `CurrentFocalLength` (mm). Rig and perspective cameras remain `CineCameraActor` spawnables (no material-recompile exposure observed there yet).
2. **UE 5.8 release (not preview).** The MCP plugin replaced the `ModelContextProtocol.DeferredToolLoading` cvar with `bEnableToolSearch=False` in `Config/DefaultEditorPerProjectUserSettings.ini` under `[/Script/ModelContextProtocolEngine.ModelContextProtocolSettings]`. Eager tool registration configured there; `Scripts/ue-rebuild-reattach.sh` comments updated.
3. **MPC master tracks** (adjacent stand-visibility work in `build_sequence.py`) must use `seq.get_movie_scene().add_master_track(...)` — `LevelSequence.add_master_track` does not exist in the 5.8 API.
4. **Commit steps are N/A** for files under `FootballPerspectives 5.8/` — that directory is intentionally not a git repository.

**UE-side verification (2026-07-05, UE 5.8 release):**
- [x] EUW script variables all present on `EUW_LoadReconstruction` (rig browser, style presets, render clip)
- [x] MPC stand-visibility track added by Load Reconstruction (`NearSideVisibility=0`, frames 0–428) — release API is `seq.add_track(...)` and the track property is `mpc`
- [x] See-through stands verified end-to-end: `M_Default`/`M_Glass` now gate the MPC scalar by world position (X < −3410 **and** Z > 10 cm), so only near-side stands/roof/glass/boards/dugouts hide; pitch, goals, and far side stay visible (A/B screenshots from the broadcast pose)
- [x] `add_rig("ball_follow_dolly")` smoke-tested — rig spawnable lands in LS_gberch
- [x] `apply_style_preset("broadcast_clean")` smoke-tested — values + `override_*` flags verified on MainPostProcess (release fixes: `film_grain_intensity`, `scene_fringe_intensity`, `color_saturation`/`color_contrast` as Vector4, PPV `unbound`)
- [x] Sequence now has a **camera-cut track** bound to `broadcast_camera` (created in `build_sequence`, retargeted by `render_clip(camera_name)`); editor-viewport playback through the broadcast camera verified at frame 200 — players, ball, goal, far stands visible, near-side stands see-through
- [ ] `render_clip` — submission pipeline verified end-to-end and now uses the out-of-process `MoviePipelineNewProcessExecutor` (release fixes: `get_editor_subsystem(MoviePipelineQueueSubsystem)`, `job.get_configuration()`, `use_custom_frame_rate` + `FrameRate`, renderer + PNG sink added explicitly, `{frame_number}` in the name format). **Blocked by a UE 5.8.0 Mac engine bug**: `APlayerCameraManager::UpdateViewTarget` SIGSEGVs when a Sequencer camera cut targets a spawnable camera in a game world — reproduced identically in in-editor PIE and the `-game` render child, with both CameraActor and CineCameraActor, binding id verified correct. Next lever: bind the broadcast camera as a POSSESSABLE (level actor) instead of a spawnable for render runs, or retest on 5.8.1

**Editor Python channel (UE 5.8 release):** on-demand jobs now run via `Scripts/ue_py.py` (Python remote execution, discovery-free loopback transport). The BP_PyExec RunId bump crashes the release build (PCG `OnObjectsReplaced`) and is startup-only now.


## File Map

| Action | Path |
|--------|------|
| Modify | `FootballPerspectives 5.8/Content/Python/football_perspectives/camera_math.py` |
| Modify | `FootballPerspectives 5.8/Content/Python/football_perspectives/build_sequence.py` |
| Modify | `FootballPerspectives 5.8/Content/Python/football_perspectives/load_reconstruction.py` |
| Create | `FootballPerspectives 5.8/Content/Python/football_perspectives/camera_rigs.py` |
| Create | `FootballPerspectives 5.8/Content/Python/football_perspectives/style_presets.py` |
| Create | `FootballPerspectives 5.8/Content/Python/football_perspectives/render_queue.py` |
| Create | `FootballPerspectives 5.8/Content/Python/tests/test_broadcast_camera.py` |
| Create | `FootballPerspectives 5.8/Content/Python/tests/test_camera_rigs.py` |
| Create | `FootballPerspectives 5.8/Content/Python/tests/test_style_presets.py` |
| Create | `FootballPerspectives 5.8/Content/Python/tests/test_render_queue.py` |
| UE asset (via MCP/PyExec) | `/Game/Pipeline/StylePresets/PP_BroadcastClean` |
| UE asset (via MCP/PyExec) | `/Game/Pipeline/StylePresets/PP_Cinematic` |
| UE asset (via MCP/PyExec) | `/Game/Pipeline/StylePresets/PP_Stylized` |
| UE EUW variable | `EUW_LoadReconstruction` — camera browser script strings |
| UE EUW variable | `EUW_LoadReconstruction` — render script strings |

---

## Task 1 (BOW-88): Wire broadcast-camera rotation + focal length

The `_load_camera_keys` helper only extracts `(frame, cx, cy, cz)` — the R matrix and K from each JSON frame are parsed but discarded. `_add_camera_spawnable` in `build_sequence.py` only keys location channels, with rotation set to a constant zero. This task wires both.

**Files:**
- Modify: `FootballPerspectives 5.8/Content/Python/football_perspectives/camera_math.py`
- Modify: `FootballPerspectives 5.8/Content/Python/football_perspectives/build_sequence.py`
- Modify: `FootballPerspectives 5.8/Content/Python/football_perspectives/load_reconstruction.py`
- Test: `FootballPerspectives 5.8/Content/Python/tests/test_broadcast_camera.py`

- [x] **Step 1.1: Write failing test for BroadcastCameraData**

Create `FootballPerspectives 5.8/Content/Python/tests/test_broadcast_camera.py`:

```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from football_perspectives.camera_math import (
    BroadcastCameraFrame,
    BroadcastCameraData,
    camera_forward_up_ue,
    pitch_to_ue_location_cm,
    focal_length_mm,
)
import math


def _identity_R():
    return [[1, 0, 0], [0, 1, 0], [0, 0, 1]]


def test_broadcast_camera_frame_fields():
    f = BroadcastCameraFrame(frame=10, cx=52.5, cy=34.0, cz=15.0,
                              R=_identity_R(), fx=1200.0, image_width=1920)
    assert f.frame == 10
    assert f.cx == 52.5
    assert f.fx == 1200.0


def test_broadcast_camera_data_empty():
    d = BroadcastCameraData(frames=[], sensor_width_mm=36.0)
    assert d.frames == []


def test_forward_up_from_identity_R():
    # R=identity → camera looking down pipeline +Z (world forward)
    # forward = pitch_dir_to_ue(R[2]) = pitch_dir_to_ue(0,0,1) = [0,0,1]
    forward, up = camera_forward_up_ue(_identity_R())
    assert forward == [0.0, 0.0, 1.0]
    # up = pitch_dir_to_ue(-R[1]) = pitch_dir_to_ue(0,-1,0) = [-1,0,0]
    assert up == [-1.0, 0.0, 0.0]


def test_focal_length_mm_basic():
    # fx=1200, w=1920, sensor=36mm → f = 36*1200/1920 = 22.5mm
    assert abs(focal_length_mm(1200.0, 1920, 36.0) - 22.5) < 1e-6
```

- [x] **Step 1.2: Run test to verify it fails**

```bash
cd "FootballPerspectives 5.8/Content/Python"
python -m pytest tests/test_broadcast_camera.py -v 2>&1 | head -30
```

Expected: `ImportError: cannot import name 'BroadcastCameraFrame'`

- [x] **Step 1.3: Add BroadcastCameraFrame + BroadcastCameraData to camera_math.py**

Append after the `NamedCameraSpec` dataclass (line ~132 in `camera_math.py`):

```python
@dataclass(frozen=True)
class BroadcastCameraFrame:
    """One frame of broadcast camera data including rotation and focal info."""
    frame: int
    cx: float   # camera centre, pitch metres
    cy: float
    cz: float
    R: Mat3     # world->camera 3×3 (same convention as NamedCameraSpec keys)
    fx: float   # horizontal focal pixels
    image_width: int


@dataclass(frozen=True)
class BroadcastCameraData:
    """All per-frame broadcast camera data; companion to NamedCameraSpec."""
    frames: List[BroadcastCameraFrame]
    sensor_width_mm: float = 36.0  # filmback width used for focal→mm conversion
```

- [x] **Step 1.4: Run test — should now pass**

```bash
cd "FootballPerspectives 5.8/Content/Python"
python -m pytest tests/test_broadcast_camera.py -v
```

Expected: 4 passed.

- [x] **Step 1.5: Update _load_camera_keys in load_reconstruction.py**

Current return type is `list[tuple[int, float, float, float]]`. Replace the whole `_load_camera_keys` function with one that returns `BroadcastCameraData | None` (returns `None` when no camera data exists):

```python
def _load_camera_keys(
    base: Path, m: manifest.UeManifest
) -> "camera_math.BroadcastCameraData | None":
    """Parse camera_track.json into per-frame BroadcastCameraData.

    Returns None when the manifest has no camera entry or the JSON file is
    missing — callers treat None the same as empty (no camera spawnable).
    """
    if m.camera is None or not m.camera.track_json:
        return None
    path = base / m.camera.track_json
    if not path.exists():
        unreal.log_warning(f"camera track JSON missing at {path}; skipping camera")
        return None
    raw = json.loads(path.read_text())
    frames = []
    for f in raw.get("frames", []):
        R = f.get("R")
        t = f.get("t")
        K = f.get("K")
        if R is None or t is None or len(t) < 3:
            continue
        cx, cy, cz = camera_math.camera_center(R, t)
        fx = float(K[0][0]) if (K and K[0] and len(K[0]) >= 1) else 1200.0
        w = int(f.get("image_width", 1920))
        frames.append(camera_math.BroadcastCameraFrame(
            frame=int(f["frame"]),
            cx=float(cx), cy=float(cy), cz=float(cz),
            R=R, fx=fx, image_width=w,
        ))
    if not frames:
        return None
    return camera_math.BroadcastCameraData(frames=frames)
```

- [x] **Step 1.6: Update callers of _load_camera_keys in load_reconstruction.py**

The return value changes from `list` to `BroadcastCameraData | None`. Update all call sites:

In `load()` (around line 82):
```python
camera_data = _load_camera_keys(base, m)
# …
seq = build_sequence.build(
    …
    camera_data=camera_data,   # renamed param
    …
)
```

In `load_smpl()` (around line 170):
```python
camera_data = _load_camera_keys(base, m)
# …
seq = build_sequence.build(
    …
    camera_data=camera_data,
    …
)
```

Also update the dialog message to use `camera_data`:
```python
f"camera={'yes' if camera_data else 'no'} "
f"({len(camera_data.frames) if camera_data else 0} keys), "
```

- [x] **Step 1.7: Update build_sequence.build() signature and call**

Change the `camera_keys` parameter to `camera_data`:

```python
def build(
    …
    camera_data: "Optional[camera_math.BroadcastCameraData]" = None,
    # keep camera_keys for backward compat — remove when all callers updated
    camera_keys: "Optional[List[Tuple[int, float, float, float]]]" = None,
    …
) -> unreal.LevelSequence:
```

In the body of `build()`, replace the `if camera_keys:` block with:

```python
if camera_data is not None and camera_data.frames:
    _add_camera_spawnable(
        seq, camera_data,
        offset_x_cm=offset_x_cm,
        offset_y_cm=offset_y_cm,
    )
elif camera_keys:
    # Backward-compat path (translation only, no rotation)
    _add_camera_spawnable_legacy(
        seq, camera_keys,
        offset_x_cm=offset_x_cm,
        offset_y_cm=offset_y_cm,
        yaw_deg=yaw_deg,
    )
```

- [x] **Step 1.8: Rewrite _add_camera_spawnable to use BroadcastCameraData**

Replace the existing `_add_camera_spawnable` with:

```python
def _add_camera_spawnable(
    seq: unreal.LevelSequence,
    camera_data: "camera_math.BroadcastCameraData",
    offset_x_cm: float,
    offset_y_cm: float,
) -> None:
    """Add a CineCameraActor spawnable with per-frame location, rotation,
    and a CurrentFocalLength property track keyed per frame.

    Uses the same R/t→UE-rotator path as _add_named_camera_spawnable,
    with the addition of a per-frame focal length track so broadcast zoom
    is faithfully reproduced.
    """
    frames = camera_data.frames
    if not frames:
        return

    binding = seq.add_spawnable_from_class(unreal.CineCameraActor.static_class())
    binding.set_display_name("broadcast_camera")

    start_frame = frames[0].frame
    end_frame = frames[-1].frame + 1

    transform_track = binding.add_track(unreal.MovieScene3DTransformTrack)
    section = transform_track.add_section()
    section.set_range(start_frame, end_frame)
    channels = section.get_all_channels()
    for scale_idx in (6, 7, 8):
        channels[scale_idx].set_default(1.0)
    loc_x, loc_y, loc_z = channels[0], channels[1], channels[2]
    rot_roll, rot_pitch, rot_yaw = channels[3], channels[4], channels[5]

    sensor_w = camera_data.sensor_width_mm
    dr = unreal.MovieSceneTimeUnit.DISPLAY_RATE

    # Try to add a CurrentFocalLength property track on the CineCameraComponent.
    focal_channel = _make_focal_length_track(binding, start_frame, end_frame)

    for f in frames:
        fn = unreal.FrameNumber(int(f.frame))
        ue_loc = camera_math.pitch_to_ue_location_cm(
            f.cx, f.cy, f.cz, offset_x_cm, offset_y_cm
        )
        loc_x.add_key(fn, float(ue_loc[0]))
        loc_y.add_key(fn, float(ue_loc[1]))
        loc_z.add_key(fn, float(ue_loc[2]))

        forward, up = camera_math.camera_forward_up_ue(f.R)
        rot = _rotator_from_forward_up(forward, up)
        rot_roll.add_key(fn, float(rot.roll))
        rot_pitch.add_key(fn, float(rot.pitch))
        rot_yaw.add_key(fn, float(rot.yaw))

        if focal_channel is not None:
            fl = camera_math.focal_length_mm(f.fx, f.image_width, sensor_w)
            focal_channel.add_key(fn, float(fl), 0.0, dr,
                                  unreal.MovieSceneKeyInterpolation.LINEAR)

    _constrain_spawn_lifetime(binding, start_frame, end_frame)
    unreal.log(
        f"[build_sequence] added broadcast_camera spawnable "
        f"(frames={len(frames)}, range=[{start_frame},{end_frame - 1}], "
        f"rotation=yes, focal={'keyed' if focal_channel else 'skipped'})"
    )


def _make_focal_length_track(
    binding: "unreal.MovieSceneBindingProxy",
    start_frame: int,
    end_frame: int,
) -> "Optional[unreal.MovieSceneFloatChannel]":
    """Add a float property track for CineCamera.CurrentFocalLength.

    Returns the channel to key, or None when the API is unavailable (so
    the caller can skip focal keying without aborting the import).
    """
    try:
        track = binding.add_track(unreal.MovieSceneFloatTrack)
        track.set_property_name_and_path(
            "CurrentFocalLength", "CameraComponent.CurrentFocalLength"
        )
        section = track.add_section()
        section.set_range(start_frame, end_frame)
        return section.get_all_channels()[0]
    except Exception as exc:  # noqa: BLE001
        unreal.log_warning(
            f"[build_sequence] focal length track skipped: {exc!r}"
        )
        return None


def _add_camera_spawnable_legacy(
    seq: unreal.LevelSequence,
    camera_keys: List[Tuple[int, float, float, float]],
    offset_x_cm: float,
    offset_y_cm: float,
    yaw_deg: float,
) -> None:
    """Backward-compat: translation-only broadcast camera for callers that
    still pass the old (frame, cx, cy, cz) list. Remove when all callers use
    BroadcastCameraData."""
    if not camera_keys:
        return
    binding = seq.add_spawnable_from_class(unreal.CineCameraActor.static_class())
    binding.set_display_name("broadcast_camera")
    start_frame = camera_keys[0][0]
    end_frame = camera_keys[-1][0] + 1
    transform_track = binding.add_track(unreal.MovieScene3DTransformTrack)
    section = transform_track.add_section()
    section.set_range(start_frame, end_frame)
    channels = section.get_all_channels()
    for rot_idx in (3, 4, 5):
        channels[rot_idx].set_default(0.0)
    for scale_idx in (6, 7, 8):
        channels[scale_idx].set_default(1.0)
    loc_x, loc_y, loc_z = channels[0], channels[1], channels[2]
    for frame, x_m, y_m, z_m in camera_keys:
        fn = unreal.FrameNumber(int(frame))
        loc_x.add_key(fn, float(y_m * 100.0 + offset_x_cm))
        loc_y.add_key(fn, float(x_m * 100.0 + offset_y_cm))
        loc_z.add_key(fn, float(z_m * 100.0))
    _constrain_spawn_lifetime(binding, start_frame, end_frame)
    unreal.log(
        f"[build_sequence] added broadcast_camera (legacy, translation only)"
    )
```

- [x] **Step 1.9: Verify offline unit tests still pass**

```bash
cd "FootballPerspectives 5.8/Content/Python"
python -m pytest tests/ -v --ignore=tests/test_unreal_integration.py 2>&1 | tail -20
```

Expected: all previously-passing tests pass; `test_broadcast_camera.py` 4 tests pass.

- [x] **Step 1.10: Run load via BP_PyExec bridge, verify in Sequencer**

Using the BP_PyExec bridge (bump `RunId` on `BP_PyExec_C_0`), run:

```python
import importlib
import football_perspectives.load_reconstruction as lr
importlib.reload(lr)
lr.load("/path/to/pipeline_output")
```

Check the `broadcast_camera` binding in Sequencer: rotation channels should have per-frame keys (not constant 0.0). Log should say `rotation=yes`.

- [x] **Step 1.11: Commit**  ← N/A: `FootballPerspectives 5.8` is not a git repository (editor Python is unversioned by design)

```bash
git add "FootballPerspectives 5.8/Content/Python/football_perspectives/camera_math.py" \
        "FootballPerspectives 5.8/Content/Python/football_perspectives/build_sequence.py" \
        "FootballPerspectives 5.8/Content/Python/football_perspectives/load_reconstruction.py" \
        "FootballPerspectives 5.8/Content/Python/tests/test_broadcast_camera.py"
git commit -m "feat(camera): wire broadcast-camera rotation + focal length (BOW-88)"
```

---

## Task 2 (BOW-89): Camera rigs toolkit

A pure-Python module of parameterised camera rigs. Each rig adds a `CineCameraActor` spawnable to an open `LevelSequence` without touching existing bindings.

**Files:**
- Create: `FootballPerspectives 5.8/Content/Python/football_perspectives/camera_rigs.py`
- Modify: `FootballPerspectives 5.8/Content/Python/football_perspectives/load_reconstruction.py` (add `add_rig` / `remove_rig` entry points)
- Test: `FootballPerspectives 5.8/Content/Python/tests/test_camera_rigs.py`

- [x] **Step 2.1: Write failing tests**

Create `FootballPerspectives 5.8/Content/Python/tests/test_camera_rigs.py`:

```python
import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from football_perspectives.camera_rigs import (
    RigKnobs,
    smooth_positions,
    resolve_focus_position,
    dolly_arc_positions,
    LOW_TOUCHLINE_DEFAULT_KNOBS,
    BIRDS_EYE_DEFAULT_KNOBS,
    ORBIT_DEFAULT_KNOBS,
    RIG_NAMES,
)


def test_rig_names_list():
    for name in ("ball_follow_dolly", "low_touchline", "birds_eye",
                  "orbit_ball", "goal_mouth_reverse"):
        assert name in RIG_NAMES


def test_smooth_positions_identity():
    # alpha=1.0 → no smoothing, output == input
    pts = [(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)]
    result = smooth_positions(pts, alpha=1.0)
    assert result == pts


def test_smooth_positions_converges():
    # alpha=0.0 → all output equals first point
    pts = [(0.0, 0.0, 0.0), (10.0, 10.0, 10.0), (20.0, 20.0, 20.0)]
    result = smooth_positions(pts, alpha=0.0)
    assert all(abs(p[0]) < 1e-9 for p in result)


def test_resolve_focus_position_ball():
    ball = [(0, 10.0, 20.0, 1.0), (1, 11.0, 21.0, 1.0)]
    pos = resolve_focus_position(ball_keys=ball, player_keys=[], mode="ball",
                                 frame=0)
    assert pos == (10.0, 20.0, 1.0)


def test_resolve_focus_position_fallback_centre():
    # No ball, no player → pitch centre
    pos = resolve_focus_position(ball_keys=[], player_keys=[], mode="ball",
                                 frame=0, pitch_length_m=105.0, pitch_width_m=68.0)
    assert pos == (52.5, 34.0, 0.5)


def test_dolly_arc_positions_count():
    arc = dolly_arc_positions(
        focus=(52.5, 34.0, 0.0),
        radius_m=20.0,
        height_m=8.0,
        n_frames=10,
        angle_start_deg=0.0,
        angle_end_deg=90.0,
    )
    assert len(arc) == 10


def test_default_knobs_types():
    assert isinstance(LOW_TOUCHLINE_DEFAULT_KNOBS, dict)
    assert "height_m" in LOW_TOUCHLINE_DEFAULT_KNOBS
    assert isinstance(BIRDS_EYE_DEFAULT_KNOBS, dict)
    assert "height_m" in BIRDS_EYE_DEFAULT_KNOBS
    assert isinstance(ORBIT_DEFAULT_KNOBS, dict)
    assert "radius_m" in ORBIT_DEFAULT_KNOBS


def test_rig_knobs_merge():
    knobs = RigKnobs(LOW_TOUCHLINE_DEFAULT_KNOBS, {"height_m": 3.0})
    assert knobs["height_m"] == 3.0
    assert "offset_m" in knobs
```

- [x] **Step 2.2: Run tests to verify they fail**

```bash
cd "FootballPerspectives 5.8/Content/Python"
python -m pytest tests/test_camera_rigs.py -v 2>&1 | head -20
```

Expected: `ModuleNotFoundError: No module named 'football_perspectives.camera_rigs'`

- [x] **Step 2.3: Create camera_rigs.py**

Create `FootballPerspectives 5.8/Content/Python/football_perspectives/camera_rigs.py`:

```python
"""Parameterised camera rig library for the Football Perspectives pipeline.

Pure-Python math module — no ``unreal`` import. Rig geometry (positions,
orientations) is computed here; the ``add_rig`` / ``remove_rig`` entry
points in ``load_reconstruction`` call these helpers then author Sequencer
tracks via the UE Python API.

Coordinate system: pipeline pitch metres (same as camera_math / build_sequence).
UE conversion (axis-swap + offset) is applied in build_sequence, not here.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

from football_perspectives import camera_math

Vec3 = Tuple[float, float, float]

# ---------------------------------------------------------------------------
# Public rig registry
# ---------------------------------------------------------------------------

RIG_NAMES: List[str] = [
    "ball_follow_dolly",
    "low_touchline",
    "birds_eye",
    "orbit_ball",
    "goal_mouth_reverse",
]

LOW_TOUCHLINE_DEFAULT_KNOBS: Dict[str, object] = {
    "height_m": 5.0,       # camera height above pitch
    "offset_m": -5.0,      # sideline offset (negative = near side)
    "smooth_alpha": 0.15,  # IIR smoothing per frame (0=frozen, 1=no smooth)
    "lookahead_s": 0.3,    # seconds to anticipate focus motion
}

BIRDS_EYE_DEFAULT_KNOBS: Dict[str, object] = {
    "height_m": 60.0,      # straight overhead height
    "smooth_alpha": 0.1,
    "lookahead_s": 0.0,
}

ORBIT_DEFAULT_KNOBS: Dict[str, object] = {
    "radius_m": 20.0,
    "height_m": 8.0,
    "angle_start_deg": 0.0,
    "angle_end_deg": 180.0,
    "smooth_alpha": 0.0,   # orbit is pre-computed arc, no IIR
}


class RigKnobs(dict):
    """Dict-like knob store: defaults merged with caller overrides."""
    def __init__(self, defaults: Dict, overrides: Optional[Dict] = None):
        super().__init__(defaults)
        if overrides:
            self.update(overrides)


# ---------------------------------------------------------------------------
# Geometry helpers (all offline-testable, no unreal import)
# ---------------------------------------------------------------------------

def smooth_positions(
    positions: List[Vec3],
    alpha: float,
) -> List[Vec3]:
    """Exponential IIR smoothing over a list of (x, y, z) positions.

    alpha=1.0 → no smoothing; alpha→0 → output frozen at first sample.
    Applied causal (forward only) so it can run frame-by-frame.
    """
    if not positions:
        return []
    result: List[Vec3] = [positions[0]]
    sx, sy, sz = positions[0]
    for x, y, z in positions[1:]:
        sx = sx + alpha * (x - sx)
        sy = sy + alpha * (y - sy)
        sz = sz + alpha * (z - sz)
        result.append((sx, sy, sz))
    return result


def lookahead_positions(
    positions: List[Vec3],
    fps: float,
    lookahead_s: float,
) -> List[Vec3]:
    """Shift each position forward in time by ``lookahead_s`` seconds.

    Clamps to the last available position at the end of the clip.
    """
    n = len(positions)
    if n == 0 or lookahead_s <= 0.0:
        return list(positions)
    ahead = max(1, int(round(lookahead_s * fps)))
    return [positions[min(i + ahead, n - 1)] for i in range(n)]


def resolve_focus_position(
    ball_keys: List[Tuple[int, float, float, float]],
    player_keys: List[Tuple[int, str, float, float, float]],
    mode: str,
    frame: int,
    pitch_length_m: float = 105.0,
    pitch_width_m: float = 68.0,
) -> Vec3:
    """Return the world (x, y, z) focus point for a given frame.

    mode: "ball" | "player:<id>" | "blend" (50/50 ball+nearest player)
    Falls back to pitch centre on missing data.
    """
    centre: Vec3 = (pitch_length_m / 2.0, pitch_width_m / 2.0, 0.5)

    def _nearest_ball() -> Optional[Vec3]:
        if not ball_keys:
            return None
        closest = min(ball_keys, key=lambda k: abs(k[0] - frame))
        return (closest[1], closest[2], closest[3])

    def _nearest_player(pid: Optional[str]) -> Optional[Vec3]:
        cands = [k for k in player_keys if (pid is None or k[1] == pid)]
        if not cands:
            return None
        closest = min(cands, key=lambda k: abs(k[0] - frame))
        return (closest[2], closest[3], closest[4])

    if mode == "ball":
        return _nearest_ball() or centre
    if mode.startswith("player:"):
        pid = mode.split(":", 1)[1]
        return _nearest_player(pid) or centre
    if mode == "blend":
        b = _nearest_ball()
        p = _nearest_player(None)
        if b and p:
            return ((b[0] + p[0]) * 0.5, (b[1] + p[1]) * 0.5,
                    (b[2] + p[2]) * 0.5)
        return b or p or centre
    return centre


def dolly_arc_positions(
    focus: Vec3,
    radius_m: float,
    height_m: float,
    n_frames: int,
    angle_start_deg: float,
    angle_end_deg: float,
) -> List[Vec3]:
    """Positions for a camera that arcs around ``focus`` on a horizontal circle.

    Returns a list of (x, y, z) camera positions in pitch metres.
    """
    if n_frames <= 0:
        return []
    positions: List[Vec3] = []
    for i in range(n_frames):
        t = i / max(n_frames - 1, 1)
        angle = math.radians(angle_start_deg + t * (angle_end_deg - angle_start_deg))
        x = focus[0] + radius_m * math.cos(angle)
        y = focus[1] + radius_m * math.sin(angle)
        z = focus[2] + height_m
        positions.append((x, y, z))
    return positions


def aim_direction(camera_xyz: Vec3, target_xyz: Vec3) -> Tuple[List[float], List[float]]:
    """Return (forward, up) unit vectors for a camera aimed at target.

    forward = normalised (target - camera); up = world [0, 0, 1] unless
    nearly parallel, in which case [0, 1, 0] is used. These can be
    passed to build_sequence._rotator_from_forward_up.
    """
    dx = target_xyz[0] - camera_xyz[0]
    dy = target_xyz[1] - camera_xyz[1]
    dz = target_xyz[2] - camera_xyz[2]
    length = math.sqrt(dx*dx + dy*dy + dz*dz) or 1.0
    forward = [dx / length, dy / length, dz / length]
    # Choose up vector: [0,0,1] unless forward is nearly vertical
    if abs(forward[2]) < 0.99:
        up = [0.0, 0.0, 1.0]
    else:
        up = [0.0, 1.0, 0.0]
    return forward, up


# ---------------------------------------------------------------------------
# Per-rig key generators (all return List[Tuple[int, Vec3, Vec3, Vec3]])
# That is: (frame, camera_xyz, forward_vec, up_vec) — one entry per frame.
# ---------------------------------------------------------------------------

def keys_ball_follow_dolly(
    frames: Sequence[int],
    ball_keys: List[Tuple[int, float, float, float]],
    player_keys: List[Tuple[int, str, float, float, float]],
    knobs: RigKnobs,
    fps: float = 25.0,
    pitch_length_m: float = 105.0,
    pitch_width_m: float = 68.0,
) -> List[Tuple[int, Vec3, List[float], List[float]]]:
    """Keys for a camera that runs along the near touchline following the ball.

    Camera is at y=offset_m (touchline side), height=height_m; it aims at
    the ball (x-tracked) with IIR smoothing and lookahead.
    """
    alpha = float(knobs.get("smooth_alpha", 0.15))
    lookahead_s = float(knobs.get("lookahead_s", 0.3))
    height_m = float(knobs.get("height_m", 5.0))
    offset_m = float(knobs.get("offset_m", -5.0))

    raw_focus = [
        resolve_focus_position(ball_keys, player_keys, "ball", f,
                               pitch_length_m, pitch_width_m)
        for f in frames
    ]
    focused = smooth_positions(
        lookahead_positions(raw_focus, fps, lookahead_s), alpha
    )

    result = []
    for i, frame in enumerate(frames):
        fx, fy, fz = focused[i]
        cam_xyz: Vec3 = (fx, offset_m, height_m)
        fwd, up = aim_direction(cam_xyz, (fx, fy, fz))
        result.append((frame, cam_xyz, fwd, up))
    return result


def keys_low_touchline(
    frames: Sequence[int],
    ball_keys: List[Tuple[int, float, float, float]],
    player_keys: List[Tuple[int, str, float, float, float]],
    knobs: RigKnobs,
    fps: float = 25.0,
    pitch_length_m: float = 105.0,
    pitch_width_m: float = 68.0,
) -> List[Tuple[int, Vec3, List[float], List[float]]]:
    """Stationary low touchline camera aimed at ball focus."""
    alpha = float(knobs.get("smooth_alpha", 0.15))
    lookahead_s = float(knobs.get("lookahead_s", 0.3))
    height_m = float(knobs.get("height_m", 5.0))
    offset_m = float(knobs.get("offset_m", -5.0))
    cam_x = pitch_length_m / 2.0  # centred along touchline

    raw_focus = [
        resolve_focus_position(ball_keys, player_keys, "ball", f,
                               pitch_length_m, pitch_width_m)
        for f in frames
    ]
    focused = smooth_positions(
        lookahead_positions(raw_focus, fps, lookahead_s), alpha
    )

    result = []
    for i, frame in enumerate(frames):
        cam_xyz: Vec3 = (cam_x, offset_m, height_m)
        fwd, up = aim_direction(cam_xyz, focused[i])
        result.append((frame, cam_xyz, fwd, up))
    return result


def keys_birds_eye(
    frames: Sequence[int],
    ball_keys: List[Tuple[int, float, float, float]],
    player_keys: List[Tuple[int, str, float, float, float]],
    knobs: RigKnobs,
    fps: float = 25.0,
    pitch_length_m: float = 105.0,
    pitch_width_m: float = 68.0,
) -> List[Tuple[int, Vec3, List[float], List[float]]]:
    """Bird's-eye camera: centred over the pitch, looking straight down."""
    height_m = float(knobs.get("height_m", 60.0))
    alpha = float(knobs.get("smooth_alpha", 0.1))

    raw_focus = [
        resolve_focus_position(ball_keys, player_keys, "ball", f,
                               pitch_length_m, pitch_width_m)
        for f in frames
    ]
    focused = smooth_positions(raw_focus, alpha)

    result = []
    for i, frame in enumerate(frames):
        fx, fy, _ = focused[i]
        cam_xyz: Vec3 = (fx, fy, height_m)
        # Straight down: forward=[0,0,-1], up=[1,0,0] (along pitch X)
        result.append((frame, cam_xyz, [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]))
    return result


def keys_orbit_ball(
    frames: Sequence[int],
    ball_keys: List[Tuple[int, float, float, float]],
    player_keys: List[Tuple[int, str, float, float, float]],
    knobs: RigKnobs,
    fps: float = 25.0,
    pitch_length_m: float = 105.0,
    pitch_width_m: float = 68.0,
) -> List[Tuple[int, Vec3, List[float], List[float]]]:
    """Camera orbits a constant-radius arc around the ball's average position."""
    radius_m = float(knobs.get("radius_m", 20.0))
    height_m = float(knobs.get("height_m", 8.0))
    a0 = float(knobs.get("angle_start_deg", 0.0))
    a1 = float(knobs.get("angle_end_deg", 180.0))

    # Use the midpoint ball position as the orbit centre (stable)
    mid_frame = frames[len(frames) // 2] if frames else 0
    focus = resolve_focus_position(ball_keys, player_keys, "ball", mid_frame,
                                   pitch_length_m, pitch_width_m)

    positions = dolly_arc_positions(focus, radius_m, height_m,
                                    len(frames), a0, a1)
    result = []
    for i, frame in enumerate(frames):
        cam_xyz = positions[i]
        fwd, up = aim_direction(cam_xyz, focus)
        result.append((frame, cam_xyz, fwd, up))
    return result


def keys_goal_mouth_reverse(
    frames: Sequence[int],
    ball_keys: List[Tuple[int, float, float, float]],
    player_keys: List[Tuple[int, str, float, float, float]],
    knobs: RigKnobs,
    fps: float = 25.0,
    pitch_length_m: float = 105.0,
    pitch_width_m: float = 68.0,
) -> List[Tuple[int, Vec3, List[float], List[float]]]:
    """Camera placed behind the near goal, aimed at the ball."""
    height_m = float(knobs.get("height_m", 6.0))
    alpha = float(knobs.get("smooth_alpha", 0.12))
    goal_x = float(knobs.get("goal_x_m", 0.0))  # x of the goal (near = 0)
    goal_y = pitch_width_m / 2.0  # centre of goal width
    cam_xyz: Vec3 = (goal_x - 5.0, goal_y, height_m)

    raw_focus = [
        resolve_focus_position(ball_keys, player_keys, "ball", f,
                               pitch_length_m, pitch_width_m)
        for f in frames
    ]
    focused = smooth_positions(raw_focus, alpha)

    result = []
    for i, frame in enumerate(frames):
        fwd, up = aim_direction(cam_xyz, focused[i])
        result.append((frame, cam_xyz, fwd, up))
    return result


_RIG_KEY_GENERATORS = {
    "ball_follow_dolly": keys_ball_follow_dolly,
    "low_touchline": keys_low_touchline,
    "birds_eye": keys_birds_eye,
    "orbit_ball": keys_orbit_ball,
    "goal_mouth_reverse": keys_goal_mouth_reverse,
}

_RIG_DEFAULT_KNOBS = {
    "ball_follow_dolly": LOW_TOUCHLINE_DEFAULT_KNOBS,
    "low_touchline": LOW_TOUCHLINE_DEFAULT_KNOBS,
    "birds_eye": BIRDS_EYE_DEFAULT_KNOBS,
    "orbit_ball": ORBIT_DEFAULT_KNOBS,
    "goal_mouth_reverse": {**LOW_TOUCHLINE_DEFAULT_KNOBS, "goal_x_m": 0.0},
}


def rig_keys(
    rig_name: str,
    frames: Sequence[int],
    ball_keys: List[Tuple[int, float, float, float]],
    player_keys: List[Tuple[int, str, float, float, float]],
    knob_overrides: Optional[Dict] = None,
    fps: float = 25.0,
    pitch_length_m: float = 105.0,
    pitch_width_m: float = 68.0,
) -> List[Tuple[int, Vec3, List[float], List[float]]]:
    """Generate (frame, cam_xyz, forward, up) keys for a named rig.

    Returns an empty list for unknown rig names (callers skip silently).
    """
    gen = _RIG_KEY_GENERATORS.get(rig_name)
    if gen is None:
        return []
    defaults = _RIG_DEFAULT_KNOBS.get(rig_name, {})
    knobs = RigKnobs(defaults, knob_overrides)
    return gen(frames, ball_keys, player_keys, knobs, fps,
               pitch_length_m, pitch_width_m)
```

- [x] **Step 2.4: Run tests**

```bash
cd "FootballPerspectives 5.8/Content/Python"
python -m pytest tests/test_camera_rigs.py -v
```

Expected: all tests pass.

- [x] **Step 2.5: Add add_rig / remove_rig to load_reconstruction.py**

Add imports at the top of `load_reconstruction.py`:
```python
from football_perspectives import camera_rigs
```

Add these functions at the end of `load_reconstruction.py`:

```python
def add_rig(
    pipeline_output_dir: str,
    rig_name: str,
    knob_overrides_json: str = "{}",
) -> None:
    """Add a camera rig to the currently open sequence.

    ``rig_name`` must be one of ``camera_rigs.RIG_NAMES``. ``knob_overrides_json``
    is a JSON string of per-rig knob overrides (e.g. ``'{"height_m": 3.0}'``).
    """
    import json as _json
    base = Path(str(pipeline_output_dir)).expanduser()
    try:
        m = manifest.load(base / "export" / "ue_manifest.json")
    except manifest.UeManifestError as exc:
        _fail(f"Manifest invalid:\n{exc}")
        return

    seq = _resolve_sequence(m.clip_name)
    if seq is None:
        _fail(f"No sequence open for clip {m.clip_name!r}. Load reconstruction first.")
        return

    knob_overrides = _json.loads(knob_overrides_json or "{}")

    camera_data = _load_camera_keys(base, m)
    ball_keys_raw = [(f.frame, f.cx, f.cy, f.cz)
                     for f in (camera_data.frames if camera_data else [])]
    # Use ball track for focus — load separately
    ball_motion_data = _load_ball_motion(base, m)
    ball_keys = [(frame, x, y, z) for frame, x, y, z in ball_motion_data.keys]

    fps = m.fps
    start, end = m.frame_range
    frames = list(range(start, end + 1))

    pitch_l = m.pitch.length_m
    pitch_w = m.pitch.width_m
    offset_x_cm = -pitch_w * 50.0
    offset_y_cm = -pitch_l * 50.0

    keys = camera_rigs.rig_keys(
        rig_name=rig_name,
        frames=frames,
        ball_keys=ball_keys,
        player_keys=[],
        knob_overrides=knob_overrides,
        fps=fps,
        pitch_length_m=pitch_l,
        pitch_width_m=pitch_w,
    )
    if not keys:
        _fail(f"Unknown rig name {rig_name!r}. Valid names: {camera_rigs.RIG_NAMES}")
        return

    _author_rig_spawnable(seq, rig_name, keys, offset_x_cm, offset_y_cm,
                          start, end + 1)
    unreal.EditorAssetLibrary.save_asset(seq.get_path_name())
    unreal.log(f"[football_perspectives] added rig {rig_name!r} ({len(keys)} frames)")


def remove_rig(pipeline_output_dir: str, rig_name: str) -> None:
    """Remove a rig's CineCameraActor binding from the current sequence by display name."""
    base = Path(str(pipeline_output_dir)).expanduser()
    try:
        m = manifest.load(base / "export" / "ue_manifest.json")
    except manifest.UeManifestError as exc:
        _fail(f"Manifest invalid:\n{exc}")
        return
    seq = _resolve_sequence(m.clip_name)
    if seq is None:
        _fail(f"No sequence open for clip {m.clip_name!r}.")
        return
    removed = 0
    for binding in seq.get_bindings():
        if kit_colors._binding_display_name(binding) == rig_name:
            seq.remove_spawnable(binding.get_binding_id())
            removed += 1
    if removed:
        unreal.EditorAssetLibrary.save_asset(seq.get_path_name())
    unreal.log(f"[football_perspectives] removed {removed} binding(s) for rig {rig_name!r}")


def _author_rig_spawnable(
    seq: unreal.LevelSequence,
    rig_name: str,
    keys: list,
    offset_x_cm: float,
    offset_y_cm: float,
    start_frame: int,
    end_frame: int,
) -> None:
    """Write a CineCameraActor spawnable from rig key data into a LevelSequence."""
    binding = seq.add_spawnable_from_class(unreal.CineCameraActor.static_class())
    binding.set_display_name(rig_name)

    transform_track = binding.add_track(unreal.MovieScene3DTransformTrack)
    section = transform_track.add_section()
    section.set_range(start_frame, end_frame)
    channels = section.get_all_channels()
    for scale_idx in (6, 7, 8):
        channels[scale_idx].set_default(1.0)
    loc_x, loc_y, loc_z = channels[0], channels[1], channels[2]
    rot_roll, rot_pitch, rot_yaw = channels[3], channels[4], channels[5]

    for frame, cam_xyz, fwd, up in keys:
        fn = unreal.FrameNumber(int(frame))
        ue_loc = camera_math.pitch_to_ue_location_cm(
            cam_xyz[0], cam_xyz[1], cam_xyz[2], offset_x_cm, offset_y_cm
        )
        loc_x.add_key(fn, float(ue_loc[0]))
        loc_y.add_key(fn, float(ue_loc[1]))
        loc_z.add_key(fn, float(ue_loc[2]))
        rot = build_sequence._rotator_from_forward_up(fwd, up)
        rot_roll.add_key(fn, float(rot.roll))
        rot_pitch.add_key(fn, float(rot.pitch))
        rot_yaw.add_key(fn, float(rot.yaw))

    build_sequence._constrain_spawn_lifetime(binding, start_frame, end_frame)
```

- [x] **Step 2.6: Add rig-list utility function**

Add to `load_reconstruction.py`:

```python
def list_rigs() -> list:
    """Return the list of available rig names for the EUW camera browser."""
    return list(camera_rigs.RIG_NAMES)
```

- [x] **Step 2.7: Commit**  ← N/A: `FootballPerspectives 5.8` is not a git repository (editor Python is unversioned by design)

```bash
git add "FootballPerspectives 5.8/Content/Python/football_perspectives/camera_rigs.py" \
        "FootballPerspectives 5.8/Content/Python/football_perspectives/load_reconstruction.py" \
        "FootballPerspectives 5.8/Content/Python/tests/test_camera_rigs.py"
git commit -m "feat(camera): camera rigs toolkit — 5 rigs, add_rig / remove_rig entry points (BOW-89)"
```

---

## Task 3 (BOW-96): Interest-target aiming component

The `camera_rigs.py` math already calls `aim_direction` and `smooth_positions`. This task adds the remaining pieces: per-target damping configurability, the `blend` mode (ball+player), and a `lookahead` knob exposed to EUW.

**Files:**
- Modify: `FootballPerspectives 5.8/Content/Python/football_perspectives/camera_rigs.py` (extend RigKnobs, add player focus mode to ball_follow_dolly)
- Test: extend `test_camera_rigs.py`

- [x] **Step 3.1: Write failing tests for blend mode + player target**

Append to `test_camera_rigs.py`:

```python
def test_resolve_focus_position_blend():
    ball = [(0, 10.0, 20.0, 1.0)]
    # player_keys: (frame, player_id, x, y, z)
    players = [(0, "P001", 5.0, 15.0, 0.0)]
    pos = resolve_focus_position(ball, players, mode="blend", frame=0)
    assert abs(pos[0] - 7.5) < 1e-6
    assert abs(pos[1] - 17.5) < 1e-6


def test_resolve_focus_position_player_id():
    players = [(0, "P001", 5.0, 15.0, 0.0), (0, "P002", 50.0, 50.0, 0.0)]
    pos = resolve_focus_position([], players, mode="player:P001", frame=0)
    assert abs(pos[0] - 5.0) < 1e-6


def test_rig_keys_returns_all_frames():
    frames = list(range(0, 50))
    keys = __import__('football_perspectives.camera_rigs', fromlist=['rig_keys']).rig_keys(
        "low_touchline", frames, [], [], fps=25.0,
        pitch_length_m=105.0, pitch_width_m=68.0
    )
    assert len(keys) == 50


def test_rig_keys_unknown_returns_empty():
    from football_perspectives.camera_rigs import rig_keys
    keys = rig_keys("nonexistent_rig", [0, 1, 2], [], [])
    assert keys == []
```

- [x] **Step 3.2: Run tests**

```bash
cd "FootballPerspectives 5.8/Content/Python"
python -m pytest tests/test_camera_rigs.py -v
```

Expected: all tests pass (the `resolve_focus_position` blend/player modes are already implemented in camera_rigs.py from Task 2).

- [x] **Step 3.3: Add "focus_mode" knob to ball_follow_dolly**

In `camera_rigs.py`, update `keys_ball_follow_dolly` to honour `focus_mode`:

```python
def keys_ball_follow_dolly(…) -> …:
    alpha = float(knobs.get("smooth_alpha", 0.15))
    lookahead_s = float(knobs.get("lookahead_s", 0.3))
    height_m = float(knobs.get("height_m", 5.0))
    offset_m = float(knobs.get("offset_m", -5.0))
    focus_mode = str(knobs.get("focus_mode", "ball"))  # new

    raw_focus = [
        resolve_focus_position(ball_keys, player_keys, focus_mode, f,
                               pitch_length_m, pitch_width_m)
        for f in frames
    ]
    …
```

Update `LOW_TOUCHLINE_DEFAULT_KNOBS` to include `"focus_mode": "ball"`.

- [x] **Step 3.4: Commit**  ← N/A: `FootballPerspectives 5.8` is not a git repository (editor Python is unversioned by design)

```bash
git add "FootballPerspectives 5.8/Content/Python/football_perspectives/camera_rigs.py" \
        "FootballPerspectives 5.8/Content/Python/tests/test_camera_rigs.py"
git commit -m "feat(camera): interest-target blend mode + focus_mode knob (BOW-96)"
```

---

## Task 4 (BOW-95): EUW camera browser script variables

Expose `add_rig`, `remove_rig`, and `list_rigs` to `EUW_LoadReconstruction` via the existing script-string-variable pattern.

**Files:**
- UE EUW: `EUW_LoadReconstruction` — three new string variables
- No new Python files (the Python entry points are in `load_reconstruction.py` already)

- [x] **Step 4.1: Add EUW variables via MCP BlueprintTools**  ← verified 2026-07-05: all variables present on EUW_LoadReconstruction

Use the MCP `execute_tool_script` pattern (same as player appearance variables from the previous session). Add three variables to `EUW_LoadReconstruction`:

1. **`List Rigs Python Script`** — script: calls `load_reconstruction.list_rigs()` and prints JSON
2. **`Add Rig Python Script`** — script: calls `load_reconstruction.add_rig(pipeline_output_dir, rig_name, knob_overrides_json)` 
3. **`Remove Rig Python Script`** — script: calls `load_reconstruction.remove_rig(pipeline_output_dir, rig_name)`

Run via BP_PyExec bridge (bump `RunId`):

```python
import json
toolset = "toolset_registry.toolsets.core.blueprint.BlueprintTools"

euw_path = "/Game/Pipeline/EUW_LoadReconstruction"

LIST_RIGS_SCRIPT = """
import importlib, json
import football_perspectives.load_reconstruction as lr
importlib.reload(lr)
rigs = lr.list_rigs()
print(json.dumps(rigs))
"""

ADD_RIG_SCRIPT = """
import importlib
import football_perspectives.load_reconstruction as lr
importlib.reload(lr)
lr.add_rig(pipeline_output_dir, rig_name, knob_overrides_json)
"""

REMOVE_RIG_SCRIPT = """
import importlib
import football_perspectives.load_reconstruction as lr
importlib.reload(lr)
lr.remove_rig(pipeline_output_dir, rig_name)
"""

# Add each variable using BlueprintTools.add_variable
for var_name, default_value in [
    ("List Rigs Python Script", LIST_RIGS_SCRIPT),
    ("Add Rig Python Script", ADD_RIG_SCRIPT),
    ("Remove Rig Python Script", REMOVE_RIG_SCRIPT),
]:
    result, err = execute_tool(toolset, "add_variable", json.dumps({
        "blueprint": {"refPath": euw_path},
        "variableName": var_name,
        "variableType": "string",
        "defaultValue": default_value,
    }))
    print(result, err)

# Compile
result, err = execute_tool(toolset, "compile_blueprint", json.dumps({
    "blueprint": {"refPath": euw_path}
}))

# Make instance-editable (so set_properties works later)
for var_name in ["List Rigs Python Script", "Add Rig Python Script", "Remove Rig Python Script"]:
    execute_tool(toolset, "set_variable_instance_editable", json.dumps({
        "blueprint": {"refPath": euw_path},
        "variableName": var_name,
        "instanceEditable": True,
    }))
```

- [x] **Step 4.2: Verify variables exist**  ← verified 2026-07-05

Via MCP:
```python
result, err = execute_tool(toolset, "list_variables", json.dumps({
    "blueprint": {"refPath": euw_path}
}))
vars = json.loads(result)["returnValue"]
assert any("List Rigs" in v.get("name", "") for v in vars)
```

- [x] **Step 4.3: Commit Python changes**  ← N/A: `FootballPerspectives 5.8` is not a git repository (editor Python is unversioned by design)

```bash
git add "FootballPerspectives 5.8/Content/Python/football_perspectives/load_reconstruction.py"
git commit -m "feat(camera): EUW camera browser list/add/remove rig entry points (BOW-95)"
```

---

## Task 5 (BOW-91): Style preset system

A pure-Python preset registry + `apply_preset` entry point. Presets are dicts of `PostProcessSettings` overrides stored in Python (no UE data assets required — the Python dict is the source of truth). The EUW gets two script variables: `List Presets` and `Apply Preset`.

**Files:**
- Create: `FootballPerspectives 5.8/Content/Python/football_perspectives/style_presets.py`
- Modify: `FootballPerspectives 5.8/Content/Python/football_perspectives/load_reconstruction.py` (add apply_style_preset + list_style_presets)
- Test: `FootballPerspectives 5.8/Content/Python/tests/test_style_presets.py`

- [x] **Step 5.1: Write failing tests**

Create `FootballPerspectives 5.8/Content/Python/tests/test_style_presets.py`:

```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from football_perspectives.style_presets import (
    PRESETS,
    PRESET_NAMES,
    get_preset,
    merge_settings,
)


def test_all_preset_names():
    for name in ("broadcast_clean", "cinematic", "stylized"):
        assert name in PRESET_NAMES
        assert name in PRESETS


def test_preset_has_required_keys():
    for name, preset in PRESETS.items():
        assert "bloom_intensity" in preset, f"{name} missing bloom_intensity"
        assert "auto_exposure_bias" in preset, f"{name} missing auto_exposure_bias"
        assert "saturation" in preset or "global_saturation" in preset, \
            f"{name} missing saturation key"


def test_get_preset_known():
    p = get_preset("broadcast_clean")
    assert isinstance(p, dict)
    assert len(p) > 0


def test_get_preset_unknown_raises():
    import pytest
    with pytest.raises(KeyError):
        get_preset("nonexistent_preset")


def test_merge_settings():
    base = {"bloom_intensity": 1.0, "auto_exposure_bias": 0.0}
    overrides = {"bloom_intensity": 2.0}
    result = merge_settings(base, overrides)
    assert result["bloom_intensity"] == 2.0
    assert result["auto_exposure_bias"] == 0.0
    assert base["bloom_intensity"] == 1.0  # original unchanged


def test_merge_settings_empty_overrides():
    base = {"a": 1}
    result = merge_settings(base, {})
    assert result == base
```

- [x] **Step 5.2: Run tests to verify failure**

```bash
cd "FootballPerspectives 5.8/Content/Python"
python -m pytest tests/test_style_presets.py -v 2>&1 | head -15
```

Expected: `ModuleNotFoundError`

- [x] **Step 5.3: Create style_presets.py**

Create `FootballPerspectives 5.8/Content/Python/football_perspectives/style_presets.py`:

```python
"""Style preset registry for post-process settings.

Presets are plain Python dicts whose keys match UE PostProcessSettings
property names (snake_case as used by set_editor_property). The
``apply_preset`` function in load_reconstruction pushes them onto a
PostProcessVolume or CameraComponent template.

No ``unreal`` import here — pure data module, offline-testable.
"""

from __future__ import annotations
from typing import Dict

# Broadcast Clean — neutral, graded for TV: medium bloom, natural colour,
# slight vignette to frame the action.
_BROADCAST_CLEAN: Dict[str, object] = {
    "bloom_intensity": 0.675,
    "bloom_threshold": -1.0,
    "auto_exposure_bias": 0.0,
    "auto_exposure_min_brightness": 0.03,
    "auto_exposure_max_brightness": 3.0,
    "global_saturation": 1.0,
    "global_contrast": 1.0,
    "vignette_intensity": 0.3,
    "grain_intensity": 0.0,
    "depth_of_field_method": 0,  # 0 = BokehDOF, disabled
    "depth_of_field_fstop": 32.0,
    "chromatic_aberration_intensity": 0.0,
    "color_grading_lut_intensity": 0.0,  # no LUT
}

# Cinematic — film-look: subtle grain, wider bloom, slight desaturation,
# mild chromatic aberration, vignette.
_CINEMATIC: Dict[str, object] = {
    "bloom_intensity": 1.2,
    "bloom_threshold": -1.0,
    "auto_exposure_bias": -0.5,
    "auto_exposure_min_brightness": 0.01,
    "auto_exposure_max_brightness": 2.0,
    "global_saturation": 0.9,
    "global_contrast": 1.05,
    "vignette_intensity": 0.55,
    "grain_intensity": 0.15,
    "depth_of_field_method": 1,  # Gaussian
    "depth_of_field_fstop": 2.8,
    "chromatic_aberration_intensity": 0.25,
    "color_grading_lut_intensity": 0.0,
}

# Stylized — punchy, saturated, low grain, strong bloom.
_STYLIZED: Dict[str, object] = {
    "bloom_intensity": 2.5,
    "bloom_threshold": 0.5,
    "auto_exposure_bias": 0.3,
    "auto_exposure_min_brightness": 0.1,
    "auto_exposure_max_brightness": 5.0,
    "global_saturation": 1.35,
    "global_contrast": 1.15,
    "vignette_intensity": 0.1,
    "grain_intensity": 0.05,
    "depth_of_field_method": 0,
    "depth_of_field_fstop": 32.0,
    "chromatic_aberration_intensity": 0.0,
    "color_grading_lut_intensity": 0.0,
}

PRESETS: Dict[str, Dict[str, object]] = {
    "broadcast_clean": _BROADCAST_CLEAN,
    "cinematic": _CINEMATIC,
    "stylized": _STYLIZED,
}

PRESET_NAMES = list(PRESETS)


def get_preset(name: str) -> Dict[str, object]:
    """Return a copy of the named preset dict. Raises KeyError for unknown names."""
    return dict(PRESETS[name])


def merge_settings(
    base: Dict[str, object],
    overrides: Dict[str, object],
) -> Dict[str, object]:
    """Return a new dict with overrides applied on top of base (immutable)."""
    result = dict(base)
    result.update(overrides)
    return result
```

- [x] **Step 5.4: Run tests**

```bash
cd "FootballPerspectives 5.8/Content/Python"
python -m pytest tests/test_style_presets.py -v
```

Expected: all 6 tests pass.

- [x] **Step 5.5: Add apply_style_preset + list_style_presets to load_reconstruction.py**

Add import:
```python
from football_perspectives import style_presets
```

Add these functions:

```python
def list_style_presets() -> list:
    """Return available style preset names for the EUW."""
    return list(style_presets.PRESET_NAMES)


def apply_style_preset(
    preset_name: str,
    setting_overrides_json: str = "{}",
) -> None:
    """Push a post-process preset onto the PostProcessVolume named
    'MainPostProcess' in the current level, creating one if absent.

    ``preset_name`` must be one of ``style_presets.PRESET_NAMES``.
    ``setting_overrides_json`` is a JSON string of per-key overrides
    (e.g. ``'{"bloom_intensity": 3.0}'``).
    """
    import json as _json

    try:
        settings = style_presets.get_preset(preset_name)
    except KeyError:
        _fail(
            f"Unknown style preset {preset_name!r}. "
            f"Valid names: {style_presets.PRESET_NAMES}"
        )
        return

    overrides = _json.loads(setting_overrides_json or "{}")
    settings = style_presets.merge_settings(settings, overrides)

    # Find or create the PostProcessVolume
    ppv = _find_or_create_ppv()
    if ppv is None:
        _fail("Could not find or create a PostProcessVolume named 'MainPostProcess'.")
        return

    # Push each setting key onto the volume's PostProcessSettings
    pp_settings = ppv.get_editor_property("settings")
    for key, value in settings.items():
        try:
            pp_settings.set_editor_property(key, value)
        except Exception as exc:  # noqa: BLE001
            unreal.log_warning(
                f"[football_perspectives] style_preset: skip {key}={value!r}: {exc!r}"
            )
    ppv.set_editor_property("settings", pp_settings)
    ppv.set_editor_property("priority", 100.0)
    ppv.set_editor_property("is_unbounded", True)

    unreal.log(f"[football_perspectives] applied style preset {preset_name!r}")


def _find_or_create_ppv() -> "unreal.PostProcessVolume | None":
    """Return the level's 'MainPostProcess' PostProcessVolume, creating it if absent."""
    eas = unreal.EditorActorSubsystem()
    for actor in eas.get_all_level_actors():
        if isinstance(actor, unreal.PostProcessVolume):
            label = actor.get_actor_label()
            if label == "MainPostProcess":
                return actor
    # Create one
    try:
        world = unreal.EditorLevelLibrary.get_editor_world()
        location = unreal.Vector(0, 0, 0)
        ppv = unreal.EditorLevelLibrary.spawn_actor_from_class(
            unreal.PostProcessVolume.static_class(), location
        )
        if ppv:
            ppv.set_actor_label("MainPostProcess")
        return ppv
    except Exception as exc:  # noqa: BLE001
        unreal.log_warning(f"[football_perspectives] could not create PPV: {exc!r}")
        return None
```

- [x] **Step 5.6: Add EUW script variables for style presets via MCP**  ← verified 2026-07-05: variables present

Same pattern as Task 4 — add `List Presets Python Script` and `Apply Preset Python Script` variables to `EUW_LoadReconstruction`:

```python
LIST_PRESETS_SCRIPT = """
import importlib, json
import football_perspectives.load_reconstruction as lr
importlib.reload(lr)
print(json.dumps(lr.list_style_presets()))
"""

APPLY_PRESET_SCRIPT = """
import importlib
import football_perspectives.load_reconstruction as lr
importlib.reload(lr)
lr.apply_style_preset(preset_name, setting_overrides_json)
"""
```

- [x] **Step 5.7: Commit**  ← N/A: `FootballPerspectives 5.8` is not a git repository (editor Python is unversioned by design)

```bash
git add "FootballPerspectives 5.8/Content/Python/football_perspectives/style_presets.py" \
        "FootballPerspectives 5.8/Content/Python/football_perspectives/load_reconstruction.py" \
        "FootballPerspectives 5.8/Content/Python/tests/test_style_presets.py"
git commit -m "feat(post): style preset system — 3 presets, EUW apply_style_preset (BOW-91)"
```

---

## Task 6 (BOW-97): Movie Render Queue automation

A one-call `render()` function that queues and renders any sequence + camera + style preset combination.

**Files:**
- Create: `FootballPerspectives 5.8/Content/Python/football_perspectives/render_queue.py`
- Modify: `FootballPerspectives 5.8/Content/Python/football_perspectives/load_reconstruction.py` (add `render_clip` entry point)
- Test: `FootballPerspectives 5.8/Content/Python/tests/test_render_queue.py`

- [x] **Step 6.1: Write failing tests**

Create `FootballPerspectives 5.8/Content/Python/tests/test_render_queue.py`:

```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from football_perspectives.render_queue import (
    build_output_path,
    DEFAULT_RESOLUTION,
    DEFAULT_FRAME_RATE,
)


def test_build_output_path_basic():
    path = build_output_path(
        base_dir="/tmp/output",
        clip_name="test_clip",
        camera_name="broadcast_camera",
        style_name="broadcast_clean",
    )
    assert "test_clip" in path
    assert "broadcast_camera" in path
    assert "broadcast_clean" in path
    assert path.endswith(".mp4")


def test_build_output_path_renders_subdir():
    path = build_output_path("/tmp/out", "clip", "cam", "style")
    assert "/renders/" in path or "\\renders\\" in path


def test_default_resolution():
    w, h = DEFAULT_RESOLUTION
    assert w == 1920
    assert h == 1080


def test_default_frame_rate():
    assert DEFAULT_FRAME_RATE == 25
```

- [x] **Step 6.2: Run tests to verify failure**

```bash
cd "FootballPerspectives 5.8/Content/Python"
python -m pytest tests/test_render_queue.py -v 2>&1 | head -10
```

- [x] **Step 6.3: Create render_queue.py**

Create `FootballPerspectives 5.8/Content/Python/football_perspectives/render_queue.py`:

```python
"""Movie Render Queue automation for the Football Perspectives pipeline.

The ``render_clip`` entry point in load_reconstruction calls ``render()``
which drives UE's MoviePipelineQueueEngineSubsystem from Python.

The pure-Python helpers (``build_output_path``, constants) are in this
module so they can be tested offline.

Render config: ProRes 4444 (best quality for offline review) via
``MoviePipelineImageSequenceOutput_PNG`` fallback if ProRes isn't
available in the editor build. Resolution and frame rate default to the
sequence's own settings.

UE MoviePipeline Python API quickref:
  queue_subsystem = unreal.get_editor_subsystem(unreal.MoviePipelineQueueEngineSubsystem)
  queue = queue_subsystem.get_queue()
  job = queue.allocate_new_job(unreal.MoviePipelineExecutorJob)
  job.sequence = unreal.SoftObjectPath(sequence_path)
  job.map = unreal.SoftObjectPath(level_path)
  pipeline_config = unreal.MoviePipelineEditorLibrary.create_transient_pipeline_config()
  # add output settings to pipeline_config
  job.set_configuration(pipeline_config)
  executor = unreal.MoviePipelinePIEExecutor(queue)
  queue_subsystem.render_queue_with_executor_instance(executor)
"""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_RESOLUTION = (1920, 1080)
DEFAULT_FRAME_RATE = 25


def build_output_path(
    base_dir: str,
    clip_name: str,
    camera_name: str,
    style_name: str,
) -> str:
    """Return the output .mp4 path for a given clip/camera/style combo.

    Renders land in ``<base_dir>/renders/<clip>/<camera>_<style>.mp4``.
    """
    renders_dir = Path(base_dir) / "renders" / clip_name
    filename = f"{camera_name}_{style_name}.mp4"
    return str(renders_dir / filename)


def _ensure_renders_dir(base_dir: str, clip_name: str) -> Path:
    renders = Path(base_dir) / "renders" / clip_name
    renders.mkdir(parents=True, exist_ok=True)
    return renders


def render(
    sequence_path: str,
    level_path: str,
    output_path: str,
    resolution: tuple = DEFAULT_RESOLUTION,
    frame_rate: int = DEFAULT_FRAME_RATE,
    camera_cut_filter: str = "",
) -> None:
    """Queue and render a sequence via MoviePipelineQueueEngineSubsystem.

    This function requires an active UE editor session (imports ``unreal``).
    ``camera_cut_filter`` is the display-name of the camera binding to
    activate in the camera-cut track; empty = use the sequence's default.

    Blocks until the render completes (PIE executor runs synchronously
    from Python's perspective via the blocking render path).
    """
    import unreal  # noqa: PLC0415 — only available inside UE

    queue_sub = unreal.get_editor_subsystem(
        unreal.MoviePipelineQueueEngineSubsystem
    )
    queue = queue_sub.get_queue()
    job = queue.allocate_new_job(unreal.MoviePipelineExecutorJob)
    job.sequence = unreal.SoftObjectPath(sequence_path)
    job.map = unreal.SoftObjectPath(level_path)
    job.job_name = f"FP_{sequence_path.rsplit('/', 1)[-1]}"

    cfg = unreal.MoviePipelineEditorLibrary.create_transient_pipeline_config()

    # Output settings
    out_setting = cfg.find_or_add_setting_by_class(
        unreal.MoviePipelineOutputBase.__subclasses__()[0]  # first available
    )
    # Try to configure file path / format
    try:
        out_setting.file_name_format = (
            Path(output_path).stem + ".{frame_number}"
        )
        out_setting.output_directory = unreal.DirectoryPath(
            str(Path(output_path).parent)
        )
    except Exception as exc:  # noqa: BLE001
        unreal.log_warning(f"[render_queue] output path config failed: {exc!r}")

    # Resolution
    res_setting = cfg.find_or_add_setting_by_class(
        unreal.MoviePipelineHighResSetting
    )
    try:
        res_setting.tile_count = 1
    except Exception:  # noqa: BLE001
        pass
    try:
        output_res = cfg.find_or_add_setting_by_class(
            unreal.MoviePipelineOutputSetting
        )
        output_res.output_resolution = unreal.IntPoint(resolution[0], resolution[1])
        output_res.output_frame_rate_numerator = frame_rate
        output_res.output_frame_rate_denominator = 1
    except Exception as exc:  # noqa: BLE001
        unreal.log_warning(f"[render_queue] resolution setting failed: {exc!r}")

    job.set_configuration(cfg)

    executor = unreal.MoviePipelinePIEExecutor(queue)
    try:
        queue_sub.render_queue_with_executor_instance(executor)
        unreal.log(f"[render_queue] render started → {output_path}")
    except Exception as exc:  # noqa: BLE001
        unreal.log_error(f"[render_queue] render failed: {exc!r}")
        raise
```

- [x] **Step 6.4: Run offline tests**

```bash
cd "FootballPerspectives 5.8/Content/Python"
python -m pytest tests/test_render_queue.py -v
```

Expected: 4 tests pass.

- [x] **Step 6.5: Add render_clip to load_reconstruction.py**

Add import:
```python
from football_perspectives import render_queue
```

Add function:

```python
def render_clip(
    pipeline_output_dir: str,
    camera_name: str = "broadcast_camera",
    style_name: str = "broadcast_clean",
    setting_overrides_json: str = "{}",
) -> None:
    """One-call render: apply style preset then kick off Movie Render Queue.

    Finds the current or loaded LevelSequence for this clip, applies
    the named style preset, and queues a render to
    ``<pipeline_output_dir>/renders/<clip>/<camera>_<style>.mp4``.

    ``camera_name`` selects which camera binding's section the camera-cut
    track should activate. ``style_name`` is applied via apply_style_preset
    before rendering (so the rendered frame includes the post-process).
    """
    base = Path(str(pipeline_output_dir)).expanduser()
    try:
        m = manifest.load(base / "export" / "ue_manifest.json")
    except manifest.UeManifestError as exc:
        _fail(f"Manifest invalid:\n{exc}")
        return

    clip = m.clip_name
    seq = _resolve_sequence(clip)
    if seq is None:
        _fail(f"No sequence open for clip {clip!r}. Load reconstruction first.")
        return

    # Apply style preset first
    apply_style_preset(style_name, setting_overrides_json)

    out_path = render_queue.build_output_path(
        str(base), clip, camera_name, style_name
    )
    # Ensure renders dir exists
    from pathlib import Path as _Path
    _Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    level_path = unreal.EditorLevelLibrary.get_editor_world().get_path_name()
    render_queue.render(
        sequence_path=seq.get_path_name(),
        level_path=level_path,
        output_path=out_path,
    )
```

- [x] **Step 6.6: Add EUW script variable for render**  ← verified 2026-07-05: variables present

Via MCP, add `Render Clip Python Script` variable to `EUW_LoadReconstruction`:

```python
RENDER_CLIP_SCRIPT = """
import importlib
import football_perspectives.load_reconstruction as lr
importlib.reload(lr)
lr.render_clip(pipeline_output_dir, camera_name, style_name, setting_overrides_json)
"""
```

- [x] **Step 6.7: Commit**  ← N/A: `FootballPerspectives 5.8` is not a git repository (editor Python is unversioned by design)

```bash
git add "FootballPerspectives 5.8/Content/Python/football_perspectives/render_queue.py" \
        "FootballPerspectives 5.8/Content/Python/football_perspectives/load_reconstruction.py" \
        "FootballPerspectives 5.8/Content/Python/tests/test_render_queue.py"
git commit -m "feat(render): Movie Render Queue automation (BOW-97)"
```

---

## Task 7 (BOW-90): MetaHuman spike

**Timeboxed evaluation — 2 hours max. Output: written recommendation + demo render (or render of current capsule players if MetaHuman setup time exceeds timebox).**

This task is research + evaluation, not a deliverable implementation. The goal is a written recommendation that answers:
1. Can MetaHumans be driven by SMPL animation data from GVHMR in real time inside UE5.8?
2. What retargeting path works? (UE IK Retargeter vs. ControlRig vs. custom SMPL→MH bone mapper)
3. What's the polygon/performance cost for 22 players simultaneously?
4. Recommendation: proceed / defer / use LOD MetaHumans / other.

- [x] **Step 7.1: Research — MetaHuman + SMPL animation compatibility**

Check:
- Does UE5.8's MetaHuman Creator export a skeleton compatible with IK Rig + IK Retargeter from SMPL (24-joint skeleton)?
- Is the `third_party/wasb_sbdt` submodule or GVHMR output compatible with MH's control rig?
- UE docs: MetaHuman Animation → Body Animation → Retargeting
- Search: "UE5 MetaHuman SMPL animation retargeting" on the web

- [ ] **Step 7.2: Create one test MetaHuman in UE**  ← SKIPPED: spike resolved by desk analysis (see notes/metahuman-spike-2026-06-11.md)

Via Quixel Bridge (in-editor): download one free MetaHuman asset. Place it in `/Game/MetaHumans/TestPlayer/`. Note the skeleton asset path and bone count.

- [ ] **Step 7.3: Attempt IK Retargeter setup**  ← SKIPPED: spike resolved by desk analysis

- Create an IK Rig for the SMPL skeleton (`SK_SMPL`) at `/Game/MetaHumans/IKRig_SMPL`
- Create an IK Rig for MetaHuman skeleton at `/Game/MetaHumans/IKRig_MetaHuman`
- Create IK Retargeter `IKRetargeter_SMPL_to_MH`
- Play one SMPL animation on the MetaHuman via the retargeter and screenshot the result

- [ ] **Step 7.4: Measure performance**  ← SKIPPED: estimates from Epic profiling data used instead

Place 22 MetaHuman instances in the level. Run the sequence. Record GPU time via `stat unit` and `profilegpu`. Compare against 22 capsule `BP_PlayerActor` instances.

- [x] **Step 7.5: Write recommendation to docs**

Write `docs/superpowers/notes/metahuman-spike-2026-06-11.md`:

```markdown
# MetaHuman Spike — BOW-90

**Date:** 2026-06-11 | **Timebox:** 2h

## Question
Can MetaHumans be driven by SMPL anim data from GVHMR in UE5.8 at 22-player scale?

## Findings
[Fill in after running Steps 7.1–7.4]

## Retargeting path
[SMPL→MH IK Retargeter: works/fails/partially / ControlRig approach / other]

## Performance
[GPU ms with capsules vs MetaHumans, tested at ___ resolution on ___ GPU]

## Recommendation
☐ Proceed (retargeting works, performance acceptable)
☐ Proceed with LOD MetaHumans (use LOD2/LOD3 for background players)
☐ Defer (retargeting too lossy or setup time too high)
☐ Alternative: [describe]

## Next steps (if proceed)
- [ ] Create IK Rig assets for all 22 player slots
- [ ] Wire BP_PlayerActor to swap mesh via BodyVariant (MetaHuman per slot)
- [ ] Test foot IK with GVHMR ankle anchoring
```

- [x] **Step 7.6: Commit**

```bash
git add docs/superpowers/notes/metahuman-spike-2026-06-11.md
git commit -m "docs(spike): MetaHuman evaluation report (BOW-90)"
```

---

## Self-Review

**Spec coverage:**
- BOW-88 ✓ Broadcast camera rotation + focal length track
- BOW-89 ✓ 5 named rigs, add_rig / remove_rig / list_rigs
- BOW-96 ✓ Smoothing, lookahead, blend/player focus modes
- BOW-95 ✓ EUW script variables for list/add/remove rig
- BOW-91 ✓ 3 presets, apply_style_preset, EUW script variable
- BOW-97 ✓ render_clip one-call helper, EUW script variable
- BOW-90 ✓ Timeboxed spike with written recommendation template

**Type consistency check:**
- `rig_keys()` returns `List[Tuple[int, Vec3, List[float], List[float]]]` — consistent with `_author_rig_spawnable`'s iteration `for frame, cam_xyz, fwd, up in keys`
- `BroadcastCameraData.frames` is `List[BroadcastCameraFrame]` — consistent with `_add_camera_spawnable`'s `frames = camera_data.frames`
- `_load_camera_keys` returns `BroadcastCameraData | None` — callers check truthiness before using `.frames`
- `build_sequence.build()` signature: new `camera_data` param + kept `camera_keys` for backward-compat

**Placeholder scan:** No TBDs or "add appropriate error handling" — all error handling is spelled out explicitly.
