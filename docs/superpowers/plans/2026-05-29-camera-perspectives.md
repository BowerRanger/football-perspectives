# Camera Perspectives Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate per-player first-person (POV) and over-the-shoulder (OTS) virtual cameras from the existing reconstruction, selectable per-shot in the web Export panel, and bake them into the FBX/glTF output alongside the broadcast camera.

**Architecture:** A pure-math module (`src/utils/virtual_cameras.py`) turns a player's `SmplWorldTrack` (+ optional `BallTrack`) into a synthesised `CameraTrack`. The `export` stage reads a per-shot selection file, calls the generator, writes one `CameraTrack` JSON per rig, and records each in the UE manifest's new `cameras` list. Blender bakes one FBX per camera-track file; the glTF builder emits animated camera nodes. The web Export panel edits the selection file via new endpoints.

**Tech Stack:** Python 3, numpy, dataclasses, FastAPI (web), Blender headless (FBX), pytest.

**Spec:** `docs/superpowers/specs/2026-05-29-camera-perspectives-design.md`

---

## File Structure

| File | Responsibility | New/Modify |
|------|----------------|------------|
| `src/utils/smpl_skeleton.py` | Add `compute_joint_world_pose` (joint position **and** world rotation) | Modify |
| `src/utils/virtual_cameras.py` | Pure camera math + POV/OTS rig builders | Create |
| `src/schemas/camera_selection.py` | Per-shot selection schema + JSON IO | Create |
| `src/schemas/ue_manifest.py` | `NamedCameraEntry` + `cameras` list field | Modify |
| `src/stages/export.py` | `_generate_virtual_cameras`, wire into `run`/manifest | Modify |
| `src/utils/gltf_builder.py` | Emit extra animated camera nodes | Modify |
| `scripts/blender_export_fbx.py` | Loop over all per-shot camera-track files | Modify |
| `config/default.yaml` | `export.virtual_cameras` defaults | Modify |
| `src/web/server.py` | `available-players` + `camera-selection` GET/PUT | Modify |
| `src/web/static/index.html` | Perspective-cameras picker in Export panel | Modify |
| `tests/test_virtual_cameras.py` | Unit tests for math + rig builders | Create |
| `tests/test_camera_selection_schema.py` | Schema round-trip/validation | Create |
| `tests/test_smpl_skeleton.py` | Pose-returning FK test | Modify |
| `tests/test_export_virtual_cameras.py` | Export-stage integration | Create |
| `tests/test_ue_manifest_cameras.py` | Manifest `cameras` round-trip | Create |
| `tests/test_web_api_camera_selection.py` | Endpoint integration | Create |

**Conventions to honour (verified in the codebase):**
- `CameraFrame.R` is world→camera rotation, OpenCV-style: camera **+Z = optical ray into scene, +X = right, +Y = down** (see `src/utils/gltf_builder.py:415-428`). `CameraFrame.t` is per-frame world→camera translation, optional; camera centre `C = -R.T @ t`.
- Pitch world is **z-up**, metres. World-up = `(0, 0, 1)`.
- SMPL FK: `compute_joint_world(thetas, root_R, root_t, joint_idx)` already exists; head joint index is **15** (`SMPL_JOINT_NAMES[15] == "head"`).
- Run tests with: `python -m pytest <path> -v`.

---

## Task 1: Pose-returning SMPL forward kinematics

Add a sibling to `compute_joint_world` that also returns the joint's world rotation (needed for POV facing direction and OTS offset). Refactor the existing function to delegate (DRY).

**Files:**
- Modify: `src/utils/smpl_skeleton.py:112-158`
- Test: `tests/test_smpl_skeleton.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_smpl_skeleton.py`:

```python
def test_compute_joint_world_pose_returns_position_and_rotation() -> None:
    from src.utils.smpl_skeleton import (
        compute_joint_world,
        compute_joint_world_pose,
    )

    thetas = np.zeros((24, 3))
    root_R = np.eye(3)
    root_t = np.array([1.0, 2.0, 0.0])
    head_idx = 15

    pos, R_world = compute_joint_world_pose(thetas, root_R, root_t, head_idx)

    # Position matches the existing position-only helper.
    np.testing.assert_allclose(pos, compute_joint_world(thetas, root_R, root_t, head_idx))
    # Rest pose with identity root → identity world rotation for the joint.
    np.testing.assert_allclose(R_world, np.eye(3), atol=1e-9)


def test_compute_joint_world_pose_applies_root_rotation_to_orientation() -> None:
    from src.utils.smpl_skeleton import axis_angle_to_matrix, compute_joint_world_pose

    thetas = np.zeros((24, 3))
    # 90° about world z applied as the canonical→pitch root rotation.
    root_R = axis_angle_to_matrix(np.array([0.0, 0.0, np.pi / 2]))
    root_t = np.zeros(3)

    _, R_world = compute_joint_world_pose(thetas, root_R, root_t, 15)

    np.testing.assert_allclose(R_world, root_R, atol=1e-9)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_smpl_skeleton.py::test_compute_joint_world_pose_returns_position_and_rotation -v`
Expected: FAIL with `ImportError: cannot import name 'compute_joint_world_pose'`.

- [ ] **Step 3: Write minimal implementation**

In `src/utils/smpl_skeleton.py`, replace the body of `compute_joint_world` (lines 112-158) so the FK walk lives in a new pose-returning function and the old function delegates:

```python
def compute_joint_world_pose(
    thetas: np.ndarray,
    root_R: np.ndarray,
    root_t: np.ndarray,
    joint_idx: int,
    rest_joints: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Forward-kinematics: world position **and** world rotation of a joint.

    Returns ``(pos, R_world)`` where ``pos`` is the joint centre in pitch
    metres and ``R_world`` is the joint's global rotation expressed in the
    pitch frame (``root_R`` composed onto the canonical joint rotation).
    See :func:`compute_joint_world` for input conventions.
    """
    rest = (
        np.asarray(rest_joints, dtype=np.float64)
        if rest_joints is not None else SMPL_REST_JOINTS_YUP
    )
    thetas = np.asarray(thetas, dtype=np.float64).reshape(24, 3)
    local_rot = np.empty((24, 3, 3))
    for j in range(24):
        local_rot[j] = axis_angle_to_matrix(thetas[j])
    global_rot = np.empty((24, 3, 3))
    global_pos = np.empty((24, 3))
    global_rot[0] = local_rot[0]
    global_pos[0] = rest[0]
    for j in range(1, 24):
        p = SMPL_PARENTS[j]
        global_rot[j] = global_rot[p] @ local_rot[j]
        global_pos[j] = global_pos[p] + global_rot[p] @ (rest[j] - rest[p])
    root_R = np.asarray(root_R, dtype=np.float64)
    root_t = np.asarray(root_t, dtype=np.float64)
    j = int(joint_idx)
    pos = root_R @ global_pos[j] + root_t
    R_world = root_R @ global_rot[j]
    return pos, R_world


def compute_joint_world(
    thetas: np.ndarray,
    root_R: np.ndarray,
    root_t: np.ndarray,
    joint_idx: int,
    rest_joints: np.ndarray | None = None,
) -> np.ndarray:
    """Forward-kinematics: world position of ``joint_idx`` (single frame).

    Thin wrapper over :func:`compute_joint_world_pose` kept for existing
    callers (ball-anchor in ``src/stages/ball.py``). See that function for
    the full input/convention docs.
    """
    pos, _ = compute_joint_world_pose(thetas, root_R, root_t, joint_idx, rest_joints)
    return pos
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_smpl_skeleton.py -v`
Expected: PASS (new tests + all pre-existing `compute_joint_world` tests still green).

- [ ] **Step 5: Commit**

```bash
git add src/utils/smpl_skeleton.py tests/test_smpl_skeleton.py
git commit -m "feat: add pose-returning SMPL forward kinematics helper"
```

---

## Task 2: Camera math primitives

Pure helpers: intrinsics from FOV, and an OpenCV-convention look-at view matrix.

**Files:**
- Create: `src/utils/virtual_cameras.py`
- Test: `tests/test_virtual_cameras.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_virtual_cameras.py`:

```python
from __future__ import annotations

import numpy as np

from src.utils.virtual_cameras import intrinsics_from_fov, look_at_view


def test_intrinsics_from_fov_centres_principal_point() -> None:
    K = np.array(intrinsics_from_fov(90.0, (1920, 1080)))
    # 90° horizontal FOV over 1920 px → fx = 960.
    assert K[0, 0] == np.float64(960.0).round(6)
    np.testing.assert_allclose([K[0, 2], K[1, 2]], [960.0, 540.0])
    assert K[2, 2] == 1.0


def test_look_at_view_is_proper_rotation_and_centres_camera() -> None:
    center = np.array([0.0, -5.0, 1.7])
    target = np.array([0.0, 0.0, 0.0])
    R, t = look_at_view(center, target)

    # Proper rotation.
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-9)
    assert np.isclose(np.linalg.det(R), 1.0)
    # Recovered camera centre matches the requested centre: C = -R^T t.
    np.testing.assert_allclose(-R.T @ t, center, atol=1e-9)


def test_look_at_view_optical_axis_points_at_target() -> None:
    center = np.array([0.0, -5.0, 0.0])
    target = np.array([0.0, 0.0, 0.0])
    R, _ = look_at_view(center, target)
    # Camera +Z (row 2 of R) is the optical ray; should point center→target (+y).
    np.testing.assert_allclose(R[2], [0.0, 1.0, 0.0], atol=1e-9)


def test_look_at_view_handles_target_along_world_up() -> None:
    # Looking straight up must not blow up (cross with world-up degenerates).
    center = np.array([0.0, 0.0, 0.0])
    target = np.array([0.0, 0.0, 5.0])
    R, _ = look_at_view(center, target)
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-9)
    assert np.isclose(np.linalg.det(R), 1.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_virtual_cameras.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.utils.virtual_cameras'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/utils/virtual_cameras.py`:

```python
"""Synthesised player POV / over-the-shoulder cameras.

Pure math + rig builders. No file I/O — the export stage handles reading
selections and writing CameraTrack JSON. Conventions match the broadcast
camera: ``R`` is world->camera (OpenCV: +Z optical ray into scene, +X
right, +Y down); per-frame ``t`` satisfies camera-centre ``C = -R.T @ t``.
"""

from __future__ import annotations

import math

import numpy as np

WORLD_UP = np.array([0.0, 0.0, 1.0])


def _normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(v))
    return v / n if n > eps else v


def intrinsics_from_fov(fov_deg: float, image_size: tuple[int, int]) -> list[list[float]]:
    """3x3 K from a horizontal field of view. Principal point centred."""
    w, h = int(image_size[0]), int(image_size[1])
    f = (w / 2.0) / math.tan(math.radians(fov_deg) / 2.0)
    return [[f, 0.0, w / 2.0], [0.0, f, h / 2.0], [0.0, 0.0, 1.0]]


def look_at_view(
    center: np.ndarray,
    target: np.ndarray,
    up: np.ndarray = WORLD_UP,
) -> tuple[np.ndarray, np.ndarray]:
    """World->camera (R, t) for a camera at ``center`` looking at ``target``.

    Rows of ``R`` are the camera axes in world coords: right (+X), down
    (+Y), forward (+Z). ``t = -R @ center``.
    """
    center = np.asarray(center, dtype=np.float64).reshape(3)
    z = _normalize(np.asarray(target, dtype=np.float64).reshape(3) - center)
    up = np.asarray(up, dtype=np.float64).reshape(3)
    x = np.cross(z, up)
    if float(np.linalg.norm(x)) < 1e-9:
        # Optical axis parallel to up — pick an arbitrary stable basis.
        x = np.cross(z, np.array([0.0, 1.0, 0.0]))
        if float(np.linalg.norm(x)) < 1e-9:
            x = np.cross(z, np.array([1.0, 0.0, 0.0]))
    x = _normalize(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z], axis=0)
    t = -R @ center
    return R, t
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_virtual_cameras.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/utils/virtual_cameras.py tests/test_virtual_cameras.py
git commit -m "feat: add virtual-camera math primitives (intrinsics, look-at)"
```

---

## Task 3: POV and OTS rig builders

Build `CameraTrack`s from a `SmplWorldTrack` (+ optional `BallTrack`).

**Files:**
- Modify: `src/utils/virtual_cameras.py`
- Test: `tests/test_virtual_cameras.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_virtual_cameras.py`:

```python
from src.schemas.ball_track import BallFrame, BallTrack
from src.schemas.smpl_world import SmplWorldTrack
from src.utils.virtual_cameras import RigConfig, build_ots_track, build_pov_track


def _straight_standing_track(n: int = 3) -> SmplWorldTrack:
    frames = np.arange(n, dtype=np.int64)
    return SmplWorldTrack(
        player_id="P001",
        frames=frames,
        betas=np.zeros(10),
        thetas=np.zeros((n, 24, 3)),
        root_R=np.broadcast_to(np.eye(3), (n, 3, 3)).copy(),
        root_t=np.tile(np.array([10.0, 20.0, 0.0]), (n, 1)),
        confidence=np.ones(n),
        shot_id="shot_01",
    )


def test_build_pov_track_centres_camera_near_head_height() -> None:
    track = _straight_standing_track()
    cam = build_pov_track(track, RigConfig(), image_size=(1920, 1080), fps=30.0, clip_id="P001_pov")

    assert cam.clip_id == "P001_pov"
    assert len(cam.frames) == 3
    f0 = cam.frames[0]
    assert f0.t is not None
    R = np.array(f0.R)
    center = -R.T @ np.array(f0.t)
    # Head sits ~1.6-1.8 m above the pelvis ground position (x=10, y=20).
    assert np.isclose(center[0], 10.0, atol=0.2)
    assert np.isclose(center[1], 20.0, atol=0.2)
    assert 1.4 < center[2] < 2.0


def test_build_ots_track_aims_at_ball_when_present() -> None:
    track = _straight_standing_track()
    ball = BallTrack(
        clip_id="shot_01",
        fps=30.0,
        frames=(
            BallFrame(frame=0, world_xyz=[12.0, 22.0, 0.0]),
            BallFrame(frame=1, world_xyz=[12.0, 22.0, 0.0]),
            BallFrame(frame=2, world_xyz=[12.0, 22.0, 0.0]),
        ),
    )
    cam = build_ots_track(track, ball, RigConfig(), image_size=(1920, 1080), fps=30.0, clip_id="P001_ots")

    f0 = cam.frames[0]
    R = np.array(f0.R)
    center = -R.T @ np.array(f0.t)
    # Optical axis (row 2) points from the camera toward the ball.
    expect_dir = np.array([12.0, 22.0, 0.0]) - center
    expect_dir = expect_dir / np.linalg.norm(expect_dir)
    np.testing.assert_allclose(R[2], expect_dir, atol=1e-6)


def test_build_ots_track_without_ball_uses_forward_fallback() -> None:
    track = _straight_standing_track()
    cam = build_ots_track(track, None, RigConfig(), image_size=(1920, 1080), fps=30.0, clip_id="P001_ots")
    assert len(cam.frames) == 3
    # Still a valid rotation even with no ball target.
    R = np.array(cam.frames[0].R)
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-6)
```

> NOTE: Confirm `BallTrack`/`BallFrame` field names by reading `src/schemas/ball_track.py` before running — if the world position field is not `world_xyz`, adjust the test and `_ball_xy_by_frame` accordingly.

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_virtual_cameras.py -k "pov_track or ots_track" -v`
Expected: FAIL with `ImportError: cannot import name 'RigConfig'`.

- [ ] **Step 3: Write minimal implementation**

Append to `src/utils/virtual_cameras.py`:

```python
from dataclasses import dataclass

from src.schemas.camera_track import CameraFrame, CameraTrack
from src.utils.smpl_skeleton import compute_joint_world_pose

HEAD_JOINT_IDX = 15
# SMPL canonical (y-up) facing axis. +Z is "forward" out of the torso for the
# rest pose; sign may need flipping after the first real export — see spec
# "Open Questions". Kept as a module constant so tuning is a one-line change.
FACE_AXIS_CANONICAL = np.array([0.0, 0.0, 1.0])


@dataclass(frozen=True)
class RigConfig:
    pov_fov_deg: float = 75.0
    ots_fov_deg: float = 60.0
    ots_back_m: float = 0.4      # behind the head along ground-projected facing
    ots_up_m: float = 0.3        # above the head
    ots_right_m: float = 0.0     # lateral offset (right of facing)
    ball_target_max_occlusion_frames: int = 10


def _head_pose_world(track: SmplWorldTrack, i: int) -> tuple[np.ndarray, np.ndarray]:
    """Head (pos, R_world) for frame index ``i``, with root-only fallback."""
    try:
        return compute_joint_world_pose(
            track.thetas[i], track.root_R[i], track.root_t[i], HEAD_JOINT_IDX
        )
    except Exception:  # pragma: no cover - defensive; thetas malformed
        # Fallback: pelvis + fixed head offset, root rotation only.
        pos = np.asarray(track.root_t[i], dtype=np.float64) + np.array([0.0, 0.0, 1.6])
        return pos, np.asarray(track.root_R[i], dtype=np.float64)


def _ball_xy_by_frame(ball_track) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    if ball_track is None:
        return out
    for f in getattr(ball_track, "frames", ()):
        xyz = getattr(f, "world_xyz", None)
        if xyz is not None:
            out[int(f.frame)] = np.asarray(xyz, dtype=np.float64).reshape(3)
    return out


def _make_track(
    clip_id: str,
    image_size: tuple[int, int],
    fps: float,
    K: list[list[float]],
    per_frame: list[tuple[int, np.ndarray, np.ndarray, float]],
) -> CameraTrack:
    frames = tuple(
        CameraFrame(
            frame=int(fr),
            K=[list(map(float, row)) for row in K],
            R=[list(map(float, row)) for row in R],
            confidence=float(conf),
            is_anchor=False,
            t=[float(x) for x in t],
        )
        for (fr, R, t, conf) in per_frame
    )
    centres = np.array([-(np.array(R)).T @ np.array(t) for (_, R, t, _) in per_frame]) \
        if per_frame else np.zeros((0, 3))
    t_world = centres.mean(axis=0).tolist() if len(centres) else [0.0, 0.0, 0.0]
    return CameraTrack(
        clip_id=clip_id,
        fps=float(fps),
        image_size=(int(image_size[0]), int(image_size[1])),
        t_world=t_world,
        frames=frames,
    )


def build_pov_track(
    track: SmplWorldTrack,
    cfg: RigConfig,
    image_size: tuple[int, int],
    fps: float,
    clip_id: str,
) -> CameraTrack:
    K = intrinsics_from_fov(cfg.pov_fov_deg, image_size)
    per_frame: list[tuple[int, np.ndarray, np.ndarray, float]] = []
    for i, fr in enumerate(np.asarray(track.frames).tolist()):
        head_pos, head_R = _head_pose_world(track, i)
        facing = _normalize(head_R @ FACE_AXIS_CANONICAL)
        R, t = look_at_view(head_pos, head_pos + facing)
        per_frame.append((int(fr), R, t, float(track.confidence[i])))
    return _make_track(clip_id, image_size, fps, K, per_frame)


def build_ots_track(
    track: SmplWorldTrack,
    ball_track,
    cfg: RigConfig,
    image_size: tuple[int, int],
    fps: float,
    clip_id: str,
) -> CameraTrack:
    K = intrinsics_from_fov(cfg.ots_fov_deg, image_size)
    ball_xy = _ball_xy_by_frame(ball_track)
    per_frame: list[tuple[int, np.ndarray, np.ndarray, float]] = []
    last_target: np.ndarray | None = None
    frames_since_ball = 0
    for i, fr in enumerate(np.asarray(track.frames).tolist()):
        head_pos, head_R = _head_pose_world(track, i)
        facing = _normalize(head_R @ FACE_AXIS_CANONICAL)
        facing_ground = _normalize(np.array([facing[0], facing[1], 0.0]))
        right_ground = _normalize(np.cross(WORLD_UP, facing_ground))
        center = (
            head_pos
            - cfg.ots_back_m * facing_ground
            + cfg.ots_up_m * WORLD_UP
            + cfg.ots_right_m * right_ground
        )
        target = ball_xy.get(int(fr))
        if target is not None:
            last_target = target
            frames_since_ball = 0
        elif last_target is not None and frames_since_ball < cfg.ball_target_max_occlusion_frames:
            target = last_target
            frames_since_ball += 1
        else:
            target = head_pos + facing * 10.0
        R, t = look_at_view(center, target)
        per_frame.append((int(fr), R, t, float(track.confidence[i])))
    return _make_track(clip_id, image_size, fps, K, per_frame)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_virtual_cameras.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/utils/virtual_cameras.py tests/test_virtual_cameras.py
git commit -m "feat: add POV and OTS virtual-camera rig builders"
```

---

## Task 4: Camera selection schema

Per-shot selection file with validation.

**Files:**
- Create: `src/schemas/camera_selection.py`
- Test: `tests/test_camera_selection_schema.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_camera_selection_schema.py`:

```python
from __future__ import annotations

import pytest

from src.schemas.camera_selection import (
    CameraSelection,
    CameraSelectionError,
    RigSelection,
)


def test_round_trip(tmp_path) -> None:
    sel = CameraSelection(
        shot_id="shot_01",
        selections=(
            RigSelection(player_id="P003", rigs=("pov", "ots")),
            RigSelection(player_id="P012", rigs=("pov",)),
        ),
    )
    path = tmp_path / "shot_01_camera_selection.json"
    sel.save(path)
    loaded = CameraSelection.load(path)
    assert loaded == sel


def test_from_dict_rejects_unknown_rig() -> None:
    with pytest.raises(CameraSelectionError):
        CameraSelection.from_dict(
            {"shot_id": "shot_01", "selections": [{"player_id": "P003", "rigs": ["dolly"]}]}
        )


def test_from_dict_dedupes_and_orders_rigs() -> None:
    sel = CameraSelection.from_dict(
        {"shot_id": "s", "selections": [{"player_id": "P1", "rigs": ["ots", "pov", "ots"]}]}
    )
    assert sel.selections[0].rigs == ("pov", "ots")


def test_load_missing_returns_empty() -> None:
    sel = CameraSelection.empty("shot_01")
    assert sel.shot_id == "shot_01"
    assert sel.selections == ()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_camera_selection_schema.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.schemas.camera_selection'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/schemas/camera_selection.py`:

```python
"""Per-shot virtual-camera selection, edited from the web Export panel.

Persisted at ``output/export/{shot_id}_camera_selection.json``. The export
stage reads it to decide which players get POV/OTS cameras.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

VALID_RIGS: tuple[str, ...] = ("pov", "ots")


class CameraSelectionError(ValueError):
    """Raised when a selection payload fails validation."""


@dataclass(frozen=True)
class RigSelection:
    player_id: str
    rigs: tuple[str, ...]


@dataclass(frozen=True)
class CameraSelection:
    shot_id: str
    selections: tuple[RigSelection, ...] = ()

    @classmethod
    def empty(cls, shot_id: str) -> "CameraSelection":
        return cls(shot_id=shot_id, selections=())

    @classmethod
    def from_dict(cls, data: dict) -> "CameraSelection":
        shot_id = str(data.get("shot_id", ""))
        if not shot_id:
            raise CameraSelectionError("shot_id must be non-empty")
        out: list[RigSelection] = []
        for entry in data.get("selections", []) or []:
            pid = str(entry.get("player_id", ""))
            if not pid:
                raise CameraSelectionError("each selection needs a player_id")
            raw_rigs = entry.get("rigs", []) or []
            for r in raw_rigs:
                if r not in VALID_RIGS:
                    raise CameraSelectionError(
                        f"unknown rig {r!r}; valid: {VALID_RIGS}"
                    )
            # Dedupe and force canonical VALID_RIGS order for stable output.
            ordered = tuple(r for r in VALID_RIGS if r in set(raw_rigs))
            if ordered:
                out.append(RigSelection(player_id=pid, rigs=ordered))
        return cls(shot_id=shot_id, selections=tuple(out))

    def to_dict(self) -> dict:
        return {
            "shot_id": self.shot_id,
            "selections": [
                {"player_id": s.player_id, "rigs": list(s.rigs)}
                for s in self.selections
            ],
        }

    @classmethod
    def load(cls, path: Path) -> "CameraSelection":
        return cls.from_dict(json.loads(Path(path).read_text()))

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=2))
        tmp.replace(path)  # atomic on POSIX
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_camera_selection_schema.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/schemas/camera_selection.py tests/test_camera_selection_schema.py
git commit -m "feat: add per-shot camera selection schema"
```

---

## Task 5: UE manifest `cameras` list

Add `NamedCameraEntry` and a `cameras` list; keep the scalar `camera` (broadcast) for backwards compatibility.

**Files:**
- Modify: `src/schemas/ue_manifest.py`
- Test: `tests/test_ue_manifest_cameras.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_ue_manifest_cameras.py`:

```python
from __future__ import annotations

from src.schemas.ue_manifest import (
    CameraEntry,
    NamedCameraEntry,
    PitchInfo,
    PlayerEntry,
    UeManifest,
    WorldBBox,
    SCHEMA_VERSION,
)


def _manifest_with_cameras() -> UeManifest:
    return UeManifest(
        schema_version=SCHEMA_VERSION,
        clip_name="clip",
        fps=30.0,
        frame_range=(0, 10),
        pitch=PitchInfo(length_m=105.0, width_m=68.0),
        players=[
            PlayerEntry(
                player_id="P001",
                fbx="fbx/P001.fbx",
                frame_range=(0, 10),
                world_bbox=WorldBBox(min=(0, 0, 0), max=(1, 1, 1)),
            )
        ],
        camera=CameraEntry(
            fbx="fbx/camera.fbx", image_size=(1920, 1080), frame_range=(0, 10),
            track_json="camera/camera_track.json",
        ),
        cameras=[
            NamedCameraEntry(
                name="broadcast", fbx="fbx/camera.fbx", image_size=(1920, 1080),
                frame_range=(0, 10), track_json="camera/camera_track.json",
            ),
            NamedCameraEntry(
                name="P001_pov", fbx="", image_size=(1920, 1080),
                frame_range=(0, 10), track_json="camera/shot_01_P001_pov_camera_track.json",
            ),
        ],
    )


def test_cameras_round_trip(tmp_path) -> None:
    m = _manifest_with_cameras()
    path = tmp_path / "ue_manifest.json"
    m.save(path)
    loaded = UeManifest.load(path)
    assert [c.name for c in loaded.cameras] == ["broadcast", "P001_pov"]
    assert loaded.cameras[1].track_json.endswith("P001_pov_camera_track.json")


def test_cameras_optional_for_backwards_compat(tmp_path) -> None:
    m = _manifest_with_cameras()
    m.cameras = []  # legacy manifest without the new field
    path = tmp_path / "ue_manifest.json"
    m.save(path)
    loaded = UeManifest.load(path)
    assert loaded.cameras == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_ue_manifest_cameras.py -v`
Expected: FAIL with `ImportError: cannot import name 'NamedCameraEntry'`.

- [ ] **Step 3: Write minimal implementation**

In `src/schemas/ue_manifest.py`, add the dataclass after `CameraEntry` (after line 75):

```python
@dataclass
class NamedCameraEntry:
    name: str
    fbx: str
    image_size: tuple[int, int]
    frame_range: tuple[int, int]
    track_json: str = ""
```

Add the field to `UeManifest` (after the `camera` field, line 87):

```python
    cameras: list[NamedCameraEntry] = field(default_factory=list)
```

In `UeManifest.save`, after the `if self.camera is not None:` block (after line 146), serialise the list:

```python
        if self.cameras:
            raw["cameras"] = [
                {
                    "name": c.name,
                    "fbx": c.fbx,
                    "image_size": list(c.image_size),
                    "frame_range": list(c.frame_range),
                    **({"track_json": c.track_json} if c.track_json else {}),
                }
                for c in self.cameras
            ]
```

In `UeManifest.load`, add to the `cls(...)` call (after the `camera=(...)` block, before the closing `)` at line 195):

```python
            cameras=[
                NamedCameraEntry(
                    name=c["name"],
                    fbx=c["fbx"],
                    image_size=tuple(c["image_size"]),
                    frame_range=tuple(c["frame_range"]),
                    track_json=c.get("track_json", ""),
                )
                for c in raw.get("cameras", [])
            ],
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_ue_manifest_cameras.py tests/test_gltf_match_metadata.py -v`
Expected: PASS (and existing manifest-consuming tests unaffected — `cameras` defaults to `[]`).

- [ ] **Step 5: Commit**

```bash
git add src/schemas/ue_manifest.py tests/test_ue_manifest_cameras.py
git commit -m "feat: add named cameras list to UE manifest"
```

---

## Task 6: Config defaults

**Files:**
- Modify: `config/default.yaml` (under the existing `export:` block)

- [ ] **Step 1: Add the config block**

Under the `export:` section in `config/default.yaml`, add:

```yaml
  virtual_cameras:
    pov_fov_deg: 75.0
    ots_fov_deg: 60.0
    ots_back_m: 0.4
    ots_up_m: 0.3
    ots_right_m: 0.0
    ball_target_max_occlusion_frames: 10
```

- [ ] **Step 2: Verify it parses**

Run: `python -c "import yaml; print(yaml.safe_load(open('config/default.yaml'))['export']['virtual_cameras'])"`
Expected: prints the dict with the six keys above.

- [ ] **Step 3: Commit**

```bash
git add config/default.yaml
git commit -m "feat: add virtual_cameras config defaults"
```

---

## Task 7: Wire virtual cameras into the export stage

Generate per-rig `CameraTrack` JSON from the selection file and record them in the manifest.

**Files:**
- Modify: `src/stages/export.py`
- Test: `tests/test_export_virtual_cameras.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_export_virtual_cameras.py`. This drives the new helper directly with a fixture output dir:

```python
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.schemas.camera_selection import CameraSelection, RigSelection
from src.schemas.camera_track import CameraFrame, CameraTrack
from src.schemas.smpl_world import SmplWorldTrack
from src.stages.export import ExportStage


def _write_broadcast_camera(path: Path) -> None:
    frames = tuple(
        CameraFrame(frame=i, K=[[1000, 0, 960], [0, 1000, 540], [0, 0, 1]],
                    R=[[1, 0, 0], [0, 1, 0], [0, 0, 1]], confidence=1.0,
                    is_anchor=False, t=[0.0, 0.0, 0.0])
        for i in range(3)
    )
    CameraTrack(clip_id="shot_01", fps=30.0, image_size=(1920, 1080),
                t_world=[0, 0, 30], frames=frames).save(path)


def _write_player(path: Path) -> None:
    n = 3
    SmplWorldTrack(
        player_id="P003", frames=np.arange(n), betas=np.zeros(10),
        thetas=np.zeros((n, 24, 3)), root_R=np.broadcast_to(np.eye(3), (n, 3, 3)).copy(),
        root_t=np.tile([10.0, 20.0, 0.0], (n, 1)), confidence=np.ones(n), shot_id="shot_01",
    ).save(path)


def test_generate_virtual_cameras_writes_rig_tracks(tmp_path: Path) -> None:
    out = tmp_path
    (out / "camera").mkdir(parents=True)
    (out / "hmr_world").mkdir(parents=True)
    _write_broadcast_camera(out / "camera" / "shot_01_camera_track.json")
    _write_player(out / "hmr_world" / "P003_smpl_world.npz")
    CameraSelection(shot_id="shot_01",
                    selections=(RigSelection("P003", ("pov", "ots")),)).save(
        out / "export" / "shot_01_camera_selection.json")

    stage = ExportStage(output_dir=out, config={"export": {"virtual_cameras": {}}})
    named = stage._generate_virtual_cameras(shot_id="shot_01")

    pov = out / "camera" / "shot_01_P003_pov_camera_track.json"
    ots = out / "camera" / "shot_01_P003_ots_camera_track.json"
    assert pov.exists() and ots.exists()
    assert {c.name for c in named} == {"P003_pov", "P003_ots"}
    # POV camera centre is near head height.
    cam = CameraTrack.load(pov)
    R = np.array(cam.frames[0].R); t = np.array(cam.frames[0].t)
    assert 1.4 < (-R.T @ t)[2] < 2.0


def test_generate_virtual_cameras_no_selection_returns_empty(tmp_path: Path) -> None:
    out = tmp_path
    (out / "camera").mkdir(parents=True)
    _write_broadcast_camera(out / "camera" / "shot_01_camera_track.json")
    stage = ExportStage(output_dir=out, config={"export": {"virtual_cameras": {}}})
    assert stage._generate_virtual_cameras(shot_id="shot_01") == []
```

> NOTE: Confirm `ExportStage.__init__` signature by reading `src/pipeline/base.py` (the test assumes `ExportStage(output_dir=..., config=...)`, matching how the stage reads `self.output_dir`/`self.config`). Adjust construction if the base class differs.

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_export_virtual_cameras.py -v`
Expected: FAIL with `AttributeError: 'ExportStage' object has no attribute '_generate_virtual_cameras'`.

- [ ] **Step 3: Write minimal implementation**

In `src/stages/export.py`, add imports near the top:

```python
from src.schemas.camera_selection import CameraSelection
from src.utils import virtual_cameras as vcam
from src.schemas.ue_manifest import NamedCameraEntry
```

Add this method to `ExportStage` (e.g. after `_export_gltf_for_shot`):

```python
    def _virtual_camera_cfg(self) -> vcam.RigConfig:
        raw = ((self.config.get("export", {}) or {}).get("virtual_cameras", {})) or {}
        return vcam.RigConfig(
            pov_fov_deg=float(raw.get("pov_fov_deg", 75.0)),
            ots_fov_deg=float(raw.get("ots_fov_deg", 60.0)),
            ots_back_m=float(raw.get("ots_back_m", 0.4)),
            ots_up_m=float(raw.get("ots_up_m", 0.3)),
            ots_right_m=float(raw.get("ots_right_m", 0.0)),
            ball_target_max_occlusion_frames=int(
                raw.get("ball_target_max_occlusion_frames", 10)
            ),
        )

    def _generate_virtual_cameras(self, shot_id: str | None) -> list[NamedCameraEntry]:
        """Read the per-shot selection, write one CameraTrack JSON per rig,
        and return the NamedCameraEntry rows for the manifest. No-op when
        no selection file exists."""
        prefix = "" if shot_id is None else f"{shot_id}_"
        sel_path = self.output_dir / "export" / f"{prefix}camera_selection.json"
        if not sel_path.exists():
            return []
        selection = CameraSelection.load(sel_path)

        bcast_path = self.output_dir / "camera" / f"{prefix}camera_track.json"
        if not bcast_path.exists():
            logger.warning("[export] no broadcast camera for %s; skipping virtual cameras", shot_id)
            return []
        bcast = CameraTrack.load(bcast_path)
        image_size = tuple(bcast.image_size)
        fps = float(bcast.fps)
        cfg = self._virtual_camera_cfg()

        players = {t.player_id: t for t in _per_shot_smpl_tracks(self.output_dir, shot_id=shot_id)}
        ball_path = self.output_dir / "ball" / f"{prefix}ball_track.json"
        ball_track = BallTrack.load(ball_path) if ball_path.exists() else None

        entries: list[NamedCameraEntry] = []
        for sel in selection.selections:
            track = players.get(sel.player_id)
            if track is None:
                logger.warning("[export] selection player %s not in shot %s; skipping",
                               sel.player_id, shot_id)
                continue
            for rig in sel.rigs:
                clip_id = f"{prefix}{sel.player_id}_{rig}"
                if rig == "pov":
                    cam = vcam.build_pov_track(track, cfg, image_size, fps, clip_id)
                else:
                    cam = vcam.build_ots_track(track, ball_track, cfg, image_size, fps, clip_id)
                cam_path = self.output_dir / "camera" / f"{clip_id}_camera_track.json"
                cam.save(cam_path)
                entries.append(NamedCameraEntry(
                    name=f"{sel.player_id}_{rig}",
                    fbx="",
                    image_size=(int(image_size[0]), int(image_size[1])),
                    frame_range=(int(cam.frames[0].frame), int(cam.frames[-1].frame)),
                    track_json=f"camera/{clip_id}_camera_track.json",
                ))
        logger.info("[export] generated %d virtual camera(s) for shot %s", len(entries), shot_id)
        return entries
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_export_virtual_cameras.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/stages/export.py tests/test_export_virtual_cameras.py
git commit -m "feat: generate per-rig virtual camera tracks in export stage"
```

---

## Task 8: Call the generator from the export run and manifest

Hook `_generate_virtual_cameras` into the glTF emission loop and add the entries (plus broadcast) to the manifest's `cameras` list.

**Files:**
- Modify: `src/stages/export.py` (`_export_gltf_for_shot`, `write_ue_manifest`)
- Test: `tests/test_export_virtual_cameras.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_export_virtual_cameras.py`:

```python
def test_write_ue_manifest_includes_virtual_cameras(tmp_path: Path) -> None:
    out = tmp_path
    (out / "camera").mkdir(parents=True)
    (out / "hmr_world").mkdir(parents=True)
    (out / "export" / "fbx").mkdir(parents=True)
    _write_broadcast_camera(out / "camera" / "shot_01_camera_track.json")
    _write_player(out / "hmr_world" / "P003_smpl_world.npz")
    (out / "export" / "fbx" / "shot_01__P003.fbx").write_bytes(b"x")  # player fbx so manifest writes
    CameraSelection(shot_id="shot_01",
                    selections=(RigSelection("P003", ("pov",)),)).save(
        out / "export" / "shot_01_camera_selection.json")

    # Pre-generate the rig tracks (run() does this; here we call directly).
    stage = ExportStage(output_dir=out, config={"export": {"virtual_cameras": {}},
                                                 "pitch": {"length_m": 105.0, "width_m": 68.0}})
    stage._generate_virtual_cameras(shot_id="shot_01")
    # Minimal shots manifest so primary-shot resolution finds shot_01.
    (out / "shots").mkdir()
    (out / "shots" / "shots_manifest.json").write_text(json.dumps(
        {"shots": [{"id": "shot_01", "clip_file": "shot_01.mp4"}], "match": None}))
    stage.write_ue_manifest("shot_01")

    from src.schemas.ue_manifest import UeManifest
    m = UeManifest.load(out / "export" / "ue_manifest.json")
    names = {c.name for c in m.cameras}
    assert "broadcast" in names
    assert "P003_pov" in names
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_export_virtual_cameras.py::test_write_ue_manifest_includes_virtual_cameras -v`
Expected: FAIL — `m.cameras` is empty because `write_ue_manifest` does not populate it yet.

- [ ] **Step 3: Write minimal implementation**

In `src/stages/export.py`, call the generator inside `_export_gltf_for_shot` right after `camera_track = CameraTrack.load(camera_path)` so rig tracks exist before the FBX/manifest steps run:

```python
        self._generate_virtual_cameras(shot_id=shot_id)
```

In `write_ue_manifest`, where `camera_entry` is built (around lines 488-513), after that block add the `cameras` list assembly. Replace the `manifest = UeManifest(...)` construction so it includes `cameras`:

```python
        # Assemble the named-cameras list: broadcast first, then any
        # per-rig virtual cameras generated for the primary shot.
        named_cameras: list[NamedCameraEntry] = []
        if camera_entry is not None:
            named_cameras.append(NamedCameraEntry(
                name="broadcast",
                fbx=camera_entry.fbx,
                image_size=camera_entry.image_size,
                frame_range=camera_entry.frame_range,
                track_json=camera_entry.track_json,
            ))
        prefix = f"{primary_shot}_" if primary_shot else ""
        for cam_path in sorted(
            (self.output_dir / "camera").glob(f"{prefix}*_camera_track.json")
        ):
            stem = cam_path.stem[: -len("_camera_track")]
            rig_name = stem[len(prefix):] if prefix else stem
            if rig_name in ("camera", ""):  # the broadcast track itself
                continue
            cam_meta_v = json.loads(cam_path.read_text())
            v_frames = cam_meta_v.get("frames", [])
            if not v_frames:
                continue
            named_cameras.append(NamedCameraEntry(
                name=rig_name,
                fbx="",
                image_size=tuple(cam_meta_v.get("image_size", [1920, 1080])),
                frame_range=(int(v_frames[0]["frame"]), int(v_frames[-1]["frame"])),
                track_json=f"camera/{cam_path.name}",
            ))
```

Then add `cameras=named_cameras,` to the `UeManifest(...)` constructor call.

> NOTE: the broadcast track file is `{prefix}camera_track.json`; the glob `{prefix}*_camera_track.json` also matches it (stem `camera`), which the `rig_name in ("camera", "")` guard skips. Verify by reading the resulting `ue_manifest.json` in Step 4.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_export_virtual_cameras.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/stages/export.py tests/test_export_virtual_cameras.py
git commit -m "feat: surface virtual cameras in export run and UE manifest"
```

---

## Task 9: glTF camera nodes for the extra rigs

Emit each virtual camera as an animated glTF camera node (translation **and** rotation animated, since the centre moves per frame) and add them to `scene_metadata.json` under `cameras`.

**Files:**
- Modify: `src/utils/gltf_builder.py` (`SceneBundle`, `build_glb`)
- Modify: `src/stages/export.py` (`_export_gltf_for_shot` passes extra cameras)
- Test: `tests/test_gltf_match_metadata.py` (or a new `tests/test_gltf_extra_cameras.py`)

- [ ] **Step 1: Write the failing test**

Create `tests/test_gltf_extra_cameras.py`:

```python
from __future__ import annotations

import json

import numpy as np

from src.schemas.camera_track import CameraFrame, CameraTrack
from src.utils.gltf_builder import SceneBundle, build_glb


def _track(clip_id: str, with_per_frame_t: bool) -> CameraTrack:
    frames = tuple(
        CameraFrame(
            frame=i, K=[[1000, 0, 960], [0, 1000, 540], [0, 0, 1]],
            R=[[1, 0, 0], [0, 1, 0], [0, 0, 1]], confidence=1.0, is_anchor=False,
            t=[0.0, 0.0, float(-i)] if with_per_frame_t else None,
        )
        for i in range(3)
    )
    return CameraTrack(clip_id=clip_id, fps=30.0, image_size=(1920, 1080),
                       t_world=[0, 0, 30], frames=frames)


def test_extra_cameras_appear_in_metadata() -> None:
    bundle = SceneBundle(
        camera_track=_track("broadcast", False),
        players=(),
        ball_track=None,
        pitch_length_m=105.0,
        pitch_width_m=68.0,
        ball_radius_m=0.11,
        extra_cameras=(("P003_pov", _track("P003_pov", True)),),
    )
    glb, meta = build_glb(bundle)
    assert glb[:4] == b"glTF"
    names = {c["name"] for c in meta.get("cameras", [])}
    assert "P003_pov" in names
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_gltf_extra_cameras.py -v`
Expected: FAIL with `TypeError: __init__() got an unexpected keyword argument 'extra_cameras'`.

- [ ] **Step 3: Write minimal implementation**

In `src/utils/gltf_builder.py`, add a field to `SceneBundle` (after `camera_track`, near line 47):

```python
    extra_cameras: tuple = ()   # tuple[(name: str, CameraTrack), ...]
```

In `build_glb`, after the broadcast-camera block (after line 576, before building `metadata`), add:

```python
    extra_camera_meta: list[dict] = []
    for cam_name, vtrack in getattr(bundle, "extra_cameras", ()) or ():
        if not getattr(vtrack, "frames", None):
            continue
        vfirst = vtrack.frames[0]
        vK = np.asarray(vfirst.K, dtype=np.float64)
        vw, vh = vtrack.image_size
        vfx = float(vK[0, 0])
        vyfov = 2.0 * np.arctan2(float(vh) / 2.0, vfx)
        vaspect = float(vw) / float(vh) if vh else 1.0
        vcam_idx = g.add_camera(
            {"yfov": float(vyfov), "aspectRatio": float(vaspect),
             "znear": 0.05, "zfar": 1000.0}, cam_name)

        def _centre(fr):
            R = np.asarray(fr.R, dtype=np.float64)
            t = np.asarray(fr.t if fr.t is not None else vtrack.t_world, dtype=np.float64)
            return -R.T @ t

        c0 = _centre(vfirst)
        vnode_idx = g.add_node({
            "name": cam_name,
            "camera": vcam_idx,
            "translation": [float(c0[0]), float(c0[1]), float(c0[2])],
            "rotation": [float(q) for q in _camera_orientation_quat(np.asarray(vfirst.R))],
        })
        vfps = float(vtrack.fps) if vtrack.fps else fps
        vtimes = np.array([f.frame for f in vtrack.frames], dtype=np.float32) / max(vfps, 1e-6)
        vquats = np.array([_camera_orientation_quat(np.asarray(f.R, dtype=np.float64))
                           for f in vtrack.frames], dtype=np.float32)
        vtrans = np.array([_centre(f) for f in vtrack.frames], dtype=np.float32)
        vt_acc = g.add_accessor_scalar_f32(vtimes)
        vrot_acc = g.add_accessor_vec4_f32(vquats)
        vpos_acc = g.add_accessor_vec3_f32(vtrans)
        g.add_animation(
            name=f"{cam_name}_anim",
            samplers=[
                {"input": vt_acc, "output": vrot_acc, "interpolation": "LINEAR"},
                {"input": vt_acc, "output": vpos_acc, "interpolation": "LINEAR"},
            ],
            channels=[
                {"sampler": 0, "target": {"node": vnode_idx, "path": "rotation"}},
                {"sampler": 1, "target": {"node": vnode_idx, "path": "translation"}},
            ],
        )
        extra_camera_meta.append({
            "name": cam_name,
            "image_size": [int(vw), int(vh)],
            "frame_range": [int(vtrack.frames[0].frame), int(vtrack.frames[-1].frame)],
            "fps": float(vfps),
        })
```

Add `"cameras": extra_camera_meta,` to the `metadata` dict (next to the existing `"camera": camera_meta,` at line 586).

> NOTE: confirm `add_accessor_vec3_f32` exists on the builder (grep it). The vec4/scalar accessors are used by the broadcast path; if no vec3 accessor exists, add one mirroring `add_accessor_vec4_f32`.

In `src/stages/export.py` `_export_gltf_for_shot`, collect the rig tracks just written and pass them to the bundle. After `players = _per_shot_smpl_tracks(...)` and before building `SceneBundle`:

```python
        prefix = "" if shot_id is None else f"{shot_id}_"
        extra_cameras: list[tuple[str, CameraTrack]] = []
        for p in sorted((self.output_dir / "camera").glob(f"{prefix}*_camera_track.json")):
            stem = p.stem[: -len("_camera_track")]
            rig_name = stem[len(prefix):] if prefix else stem
            if rig_name in ("camera", ""):
                continue
            extra_cameras.append((rig_name, CameraTrack.load(p)))
```

and add `extra_cameras=tuple(extra_cameras),` to the `SceneBundle(...)` constructor.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_gltf_extra_cameras.py tests/test_export_virtual_cameras.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/utils/gltf_builder.py src/stages/export.py tests/test_gltf_extra_cameras.py
git commit -m "feat: emit animated glTF camera nodes for virtual cameras"
```

---

## Task 10: Blender bakes one FBX per camera-track file

Generalise the single broadcast-camera branch into a loop over **all** `*_camera_track.json` files for the shot.

**Files:**
- Modify: `scripts/blender_export_fbx.py:631-667`

- [ ] **Step 1: Generalise the camera loop**

Replace the camera FBX block (lines 631-667) so it iterates every camera-track file and names the FBX after the file stem. The existing block already loops `cam_track_paths`; extend the per-shot glob so rig tracks are included and the FBX name is derived from the track filename:

```python
    # --- Camera FBX (broadcast + per-player virtual cameras) ----------
    cam_dir = output_dir / "camera"
    if cam_dir.exists():
        for cam_path in sorted(cam_dir.glob("*_camera_track.json")):
            cam = json.loads(cam_path.read_text())
            frames = cam.get("frames", [])
            image_w, _ = cam.get("image_size", [1920, 1080])
            if not frames:
                continue
            _reset_scene()
            _set_unit_scale_metres()
            scene = bpy.context.scene
            scene.frame_start = int(frames[0]["frame"])
            scene.frame_end = int(frames[-1]["frame"])
            scene.render.fps = int(round(fps))
            cam_name = cam_path.stem[: -len("_camera_track")]  # e.g. shot_01_P003_pov
            first_t = frames[0].get("t") or cam.get("t_world", [0, 0, 0])
            bpy.ops.object.camera_add(location=tuple(first_t))
            cam_obj = bpy.context.active_object
            cam_obj.name = cam_name
            cam_data = cam_obj.data
            cam_data.sensor_width = float(image_w) / 100.0
            for f in frames:
                fr = int(f["frame"])
                fx = float(f["K"][0][0])
                cam_data.lens = fx * (cam_data.sensor_width / float(image_w))
                cam_data.keyframe_insert(data_path="lens", frame=fr)
                # Animate the camera centre when per-frame t is present.
                if f.get("t") is not None:
                    R = np.array(f["R"], dtype=float)
                    t = np.array(f["t"], dtype=float)
                    centre = -R.T @ t
                    cam_obj.location = tuple(float(x) for x in centre)
                    cam_obj.keyframe_insert(data_path="location", frame=fr)
            bpy.ops.object.select_all(action="DESELECT")
            cam_obj.select_set(True)
            _export_fbx(fbx_dir / f"{cam_name}.fbx")
```

> NOTE: the broadcast file is `{shot}_camera_track.json` (or legacy `camera_track.json`), so its `cam_name` becomes `{shot}_camera` / `camera` — preserving the existing `camera.fbx` / `{shot}_camera.fbx` output names the manifest already looks for. Confirm `import numpy as np` is present at the top of the script (it manipulates arrays elsewhere); add it if missing.

- [ ] **Step 2: Smoke-check the script parses**

Run: `python -c "import ast; ast.parse(open('scripts/blender_export_fbx.py').read()); print('ok')"`
Expected: prints `ok` (full Blender run requires Blender; covered by manual verification in Task 12).

- [ ] **Step 3: Commit**

```bash
git add scripts/blender_export_fbx.py
git commit -m "feat: bake one FBX per camera track (broadcast + virtual cameras)"
```

---

## Task 11: Web endpoints for selection + available players

**Files:**
- Modify: `src/web/server.py`
- Test: `tests/test_web_api_camera_selection.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_web_api_camera_selection.py`:

```python
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.schemas.smpl_world import SmplWorldTrack
from src.web.server import create_app


@pytest.fixture
def client(tmp_path: Path):
    app = create_app(output_dir=tmp_path, config_path=None)
    return TestClient(app), tmp_path


def _write_player(out: Path, pid: str, shot_id: str) -> None:
    (out / "hmr_world").mkdir(parents=True, exist_ok=True)
    n = 2
    SmplWorldTrack(
        player_id=pid, frames=np.arange(n), betas=np.zeros(10),
        thetas=np.zeros((n, 24, 3)), root_R=np.broadcast_to(np.eye(3), (n, 3, 3)).copy(),
        root_t=np.zeros((n, 3)), confidence=np.ones(n), shot_id=shot_id,
    ).save(out / "hmr_world" / f"{pid}_smpl_world.npz")


def test_available_players_lists_shot_players(client) -> None:
    c, out = client
    _write_player(out, "P003", "shot_01")
    _write_player(out, "P012", "shot_01")
    r = c.get("/api/export/available-players", params={"shot": "shot_01"})
    assert r.status_code == 200
    assert {p["player_id"] for p in r.json()["players"]} == {"P003", "P012"}


def test_selection_get_default_empty(client) -> None:
    c, _ = client
    r = c.get("/api/export/camera-selection", params={"shot": "shot_01"})
    assert r.status_code == 200
    assert r.json() == {"shot_id": "shot_01", "selections": []}


def test_selection_put_round_trip(client) -> None:
    c, out = client
    _write_player(out, "P003", "shot_01")
    body = {"shot_id": "shot_01", "selections": [{"player_id": "P003", "rigs": ["pov", "ots"]}]}
    r = c.put("/api/export/camera-selection", params={"shot": "shot_01"}, json=body)
    assert r.status_code == 200
    saved = (out / "export" / "shot_01_camera_selection.json")
    assert saved.exists()
    again = c.get("/api/export/camera-selection", params={"shot": "shot_01"})
    assert again.json()["selections"][0]["player_id"] == "P003"


def test_selection_put_rejects_unknown_player(client) -> None:
    c, out = client
    _write_player(out, "P003", "shot_01")
    body = {"shot_id": "shot_01", "selections": [{"player_id": "P999", "rigs": ["pov"]}]}
    r = c.put("/api/export/camera-selection", params={"shot": "shot_01"}, json=body)
    assert r.status_code == 400


def test_selection_put_rejects_unknown_rig(client) -> None:
    c, out = client
    _write_player(out, "P003", "shot_01")
    body = {"shot_id": "shot_01", "selections": [{"player_id": "P003", "rigs": ["dolly"]}]}
    r = c.put("/api/export/camera-selection", params={"shot": "shot_01"}, json=body)
    assert r.status_code == 400
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_web_api_camera_selection.py -v`
Expected: FAIL with 404s (routes not registered).

- [ ] **Step 3: Write minimal implementation**

In `src/web/server.py`, add these routes near the other `/api/export/*` handlers (after `list_export_shots`, ~line 819). Reuse the existing `re` import and `output_dir` closure variable:

```python
    @app.get("/api/export/available-players")
    def available_players(shot: str | None = None):
        """Player ids with SMPL data for the shot, for the camera picker."""
        from src.utils.player_names import display_name_for, load_player_names
        hmr_dir = output_dir / "hmr_world"
        if not hmr_dir.exists():
            return {"players": []}
        names = load_player_names(output_dir)
        seen: dict[str, str] = {}
        for npz in sorted(hmr_dir.glob("*_smpl_world.npz")):
            from src.schemas.smpl_world import SmplWorldTrack
            track = SmplWorldTrack.load(npz)
            if shot and (getattr(track, "shot_id", "") or "") not in (shot, ""):
                continue
            seen[track.player_id] = display_name_for(track.player_id, names)
        return {"players": [{"player_id": k, "display_name": v} for k, v in seen.items()]}

    def _selection_path(shot: str | None):
        prefix = "" if not shot else f"{shot}_"
        return output_dir / "export" / f"{prefix}camera_selection.json"

    @app.get("/api/export/camera-selection")
    def get_camera_selection(shot: str | None = None):
        from src.schemas.camera_selection import CameraSelection
        if shot and not re.fullmatch(r"[A-Za-z0-9_-]+", shot):
            raise HTTPException(status_code=400, detail="Invalid shot id")
        path = _selection_path(shot)
        if not path.exists():
            return {"shot_id": shot or "", "selections": []}
        return CameraSelection.load(path).to_dict()

    @app.put("/api/export/camera-selection")
    def put_camera_selection(payload: dict, shot: str | None = None):
        from src.schemas.camera_selection import (
            CameraSelection,
            CameraSelectionError,
        )
        from src.schemas.smpl_world import SmplWorldTrack
        if shot and not re.fullmatch(r"[A-Za-z0-9_-]+", shot):
            raise HTTPException(status_code=400, detail="Invalid shot id")
        try:
            selection = CameraSelection.from_dict(payload)
        except CameraSelectionError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        # Validate player ids exist in this shot.
        hmr_dir = output_dir / "hmr_world"
        valid_ids = set()
        if hmr_dir.exists():
            for npz in hmr_dir.glob("*_smpl_world.npz"):
                t = SmplWorldTrack.load(npz)
                if not shot or (getattr(t, "shot_id", "") or "") in (shot, ""):
                    valid_ids.add(t.player_id)
        unknown = [s.player_id for s in selection.selections if s.player_id not in valid_ids]
        if unknown:
            raise HTTPException(status_code=400, detail=f"unknown players: {unknown}")
        selection.save(_selection_path(shot))
        return selection.to_dict()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_web_api_camera_selection.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/web/server.py tests/test_web_api_camera_selection.py
git commit -m "feat: add camera-selection and available-players endpoints"
```

---

## Task 12: Export panel picker UI + manual verification

Add the perspective-cameras picker to the Export panel and verify end-to-end in the browser.

**Files:**
- Modify: `src/web/static/index.html` (`renderExport`, ~line 5601)

- [ ] **Step 1: Add the picker UI**

Inside `renderExport`, after the existing export-status table is appended and the active shot is known (`defaultShot`), add a panel that loads available players + current selection and lets the user toggle POV/OTS per player, with a Save button. Use the existing `makePanel`, `makeTable`, and `fetchJsonOrNull` helpers already used in this function:

```javascript
  // --- Perspective cameras picker -----------------------------------
  const { wrap: camWrap, body: camBody } = makePanel("Perspective cameras");
  panel.appendChild(camWrap);

  async function renderCameraPicker(shotId) {
    camBody.innerHTML = "";
    const avail = await fetchJsonOrNull(
      `/api/export/available-players?shot=${encodeURIComponent(shotId)}`);
    const sel = await fetchJsonOrNull(
      `/api/export/camera-selection?shot=${encodeURIComponent(shotId)}`);
    const players = (avail && avail.players) || [];
    const chosen = new Map();
    ((sel && sel.selections) || []).forEach(s => chosen.set(s.player_id, new Set(s.rigs)));

    if (!players.length) {
      camBody.innerHTML = "<p class='cell-dim'>No players with SMPL data for this shot yet.</p>";
      return;
    }
    const tbl = document.createElement("table");
    tbl.innerHTML = "<thead><tr><th>Player</th><th>POV</th><th>OTS</th></tr></thead>";
    const tb = document.createElement("tbody");
    players.forEach(p => {
      const rigs = chosen.get(p.player_id) || new Set();
      const tr = document.createElement("tr");
      const mk = (rig) => {
        const cb = document.createElement("input");
        cb.type = "checkbox"; cb.checked = rigs.has(rig);
        cb.dataset.player = p.player_id; cb.dataset.rig = rig;
        const td = document.createElement("td"); td.appendChild(cb); return td;
      };
      const nameTd = document.createElement("td");
      nameTd.textContent = p.display_name || p.player_id;
      tr.appendChild(nameTd); tr.appendChild(mk("pov")); tr.appendChild(mk("ots"));
      tb.appendChild(tr);
    });
    tbl.appendChild(tb); camBody.appendChild(tbl);

    const banner = document.createElement("p");
    banner.style.display = "none"; banner.className = "cell-green";
    const saveBtn = document.createElement("button");
    saveBtn.textContent = "Save selection";
    saveBtn.onclick = async () => {
      const byPlayer = new Map();
      camBody.querySelectorAll("input[type=checkbox]").forEach(cb => {
        if (!cb.checked) return;
        if (!byPlayer.has(cb.dataset.player)) byPlayer.set(cb.dataset.player, []);
        byPlayer.get(cb.dataset.player).push(cb.dataset.rig);
      });
      const body = {
        shot_id: shotId,
        selections: [...byPlayer.entries()].map(([player_id, r]) => ({ player_id, rigs: r })),
      };
      const resp = await fetch(
        `/api/export/camera-selection?shot=${encodeURIComponent(shotId)}`,
        { method: "PUT", headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body) });
      if (resp.ok) {
        banner.textContent =
          "Selection saved. Re-run the Export stage to generate the FBX cameras.";
      } else {
        banner.className = "cell-red";
        banner.textContent = "Save failed: " + (await resp.text());
      }
      banner.style.display = "";
    };
    camBody.appendChild(saveBtn); camBody.appendChild(banner);
  }

  await renderCameraPicker(defaultShot);
```

> NOTE: `defaultShot` is set later in the original function (line ~5656). Place this block **after** that assignment, or pass the resolved shot id in. Confirm by reading `renderExport` in full before editing.

- [ ] **Step 2: Manual verification (golden path)**

Run the dashboard against an output dir that already has at least one shot with HMR players and a broadcast camera:

```bash
python recon.py serve --output ./output/
```

Then in the browser:
1. Open `/` → Export panel. Confirm the "Perspective cameras" table lists the shot's players.
2. Tick POV + OTS for one player, click **Save selection**. Confirm the green banner appears.
3. Confirm `output/export/{shot}_camera_selection.json` was written with the right rigs.
4. Re-run export: `python recon.py run --input <clip> --output ./output/ --stages export`.
5. Confirm `output/camera/{shot}_{player}_pov_camera_track.json` and `..._ots_camera_track.json` exist.
6. Confirm `output/export/ue_manifest.json` `cameras` list contains `broadcast`, `{player}_pov`, `{player}_ots`.
7. Open `/viewer` and confirm the scene still loads (extra camera nodes don't break glB parsing).
8. If Blender is installed, confirm `output/export/fbx/{shot}_{player}_pov.fbx` exists.

- [ ] **Step 3: Verify the full test suite is green**

Run: `python -m pytest tests/ -q`
Expected: PASS (no regressions).

- [ ] **Step 4: Commit**

```bash
git add src/web/static/index.html
git commit -m "feat: add perspective-cameras picker to Export panel"
```

---

## Self-Review Notes

- **Spec coverage:** D1 (POV+OTS pair) → Tasks 3,7; D2 (per-shot selection) → Tasks 4,11; D3 (save + manual re-run) → Tasks 11,12; D4 (FBX + track_json) → Tasks 7,10; D5 (reuse CameraTrack) → Task 3; D6 (`cameras` list, scalar `camera` kept) → Tasks 5,8. Camera math → Tasks 1-3. Error handling (missing selection, unknown player, head-FK fallback, ball-missing fallback, Blender absent) → Tasks 3,7,10,11. Testing matrix → unit/integration/API tasks throughout.
- **Naming consistency:** `compute_joint_world_pose`, `RigConfig`, `build_pov_track`/`build_ots_track`, `CameraSelection`/`RigSelection`, `NamedCameraEntry`, `_generate_virtual_cameras`, `extra_cameras` used consistently across tasks.
- **Pre-flight reads flagged inline:** `src/schemas/ball_track.py` (field names), `src/pipeline/base.py` (`ExportStage` ctor), `add_accessor_vec3_f32` presence, and the full `renderExport` body — confirm these before running the dependent task.
- **Out of scope (per spec):** fixed angles, live viewer preview, auto-trigger, ball-carrier heuristic.
