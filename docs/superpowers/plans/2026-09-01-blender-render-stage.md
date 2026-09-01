# Blender Render Stage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A new pipeline stage #8 `render` that turns each shot's reconstruction into finished toon/cel-shaded MP4s (16:9 + optional 9:16) from broadcast, drone, POV, and OTS cameras, fully headless via Blender.

**Architecture:** `src/stages/render.py` is a thin orchestrator (mirrors `ExportStage._export_fbx`) that shells to `blender --background --python scripts/blender_render_scene.py`. The Blender script assembles a fully procedural scene (pitch/stadium from `pitch.py` geometry, SMPL players on armatures, ball from `prepare_ball_keys`) and renders EEVEE→MP4. Bpy-free logic (artifact readers, kit zones, camera math) lives in `src/utils/` and is pytest-covered; Blender-required tests use the existing `fbx` marker.

**Tech Stack:** Python 3.11 (`.venv311`), Blender ≥ 3.6 headless (`bpy`, EEVEE), numpy, existing `src/schemas` (`SmplWorldTrack`, `CameraTrack`) and `src/utils` (`smpl_skeleton`, `virtual_cameras`, `pitch`).

**Spec:** `docs/superpowers/specs/2026-09-01-blender-render-stage-design.md`

## Global Constraints

- Always run tests/scripts with `.venv311/bin/python` (repo convention; other venvs are scratch).
- Blender ≥ 3.6 on PATH or via `render.blender_path` (falls back to `export.blender_path`); stage warns + skips when absent — never hard-fails the pipeline.
- Never edit `third_party/` vendored code.
- SMPL model asset stays gitignored at `data/models/smpl_neutral.npz` (existing convention from the FBX exporter). Never commit model data.
- Locked decisions: toon/cel-shaded look; SMPL mesh bodies with capsule-limb fallback; 16:9 primary + per-camera 9:16 variant.
- SMPL FK convention: `thetas[0]` is IGNORED; `root_R` carries root world orientation (CLAUDE.md).
- Pitch frame: z-up, x along nearside touchline, 105×68 m (`src/utils/pitch.py`).
- The FBX exporter's nested bpy helpers in `main()` are NOT refactored (working, snapshot-tested code); only top-level bpy-free readers move (spec §3). The render script gets its own bpy builders.
- Commit format `<type>: <description>`, no attribution trailer.
- Every task's test step also runs the neighbours: `.venv311/bin/python -m pytest tests/test_blender_export_iter.py tests/test_export_stage.py -q` must stay green after any task touching shared code.

## File Structure

| File | Responsibility |
|---|---|
| Create `src/utils/blender_scene_io.py` | Bpy-free artifact readers shared by both Blender scripts (moved from `scripts/blender_export_fbx.py` + new loaders) |
| Create `src/utils/render_look.py` | Pure look math: kit zones, team colors, OpenCV→Blender camera matrices, lens conversion |
| Create `src/stages/render.py` | `RenderStage` orchestrator |
| Create `scripts/blender_render_scene.py` | Bpy-side scene assembly + EEVEE render CLI |
| Modify `scripts/blender_export_fbx.py` | Import readers from `blender_scene_io` (keep re-export names) |
| Modify `src/utils/virtual_cameras.py` | Add `build_drone_track` + `RigConfig` drone fields |
| Modify `src/pipeline/runner.py:14-52` | Register stage `render` |
| Modify `src/pipeline/quality_report.py` | `_render_section` |
| Modify `config/default.yaml` | `render:` block; drone params under `export.virtual_cameras` |
| Modify `CLAUDE.md` | Stage table row, commands, config keys |
| Tests | `tests/test_blender_scene_io.py`, `tests/test_render_look.py`, `tests/test_virtual_cameras_drone.py`, `tests/test_render_stage.py`, `tests/test_blender_render_scene.py` (fbx-marked) |

---

### Task 1: Extract bpy-free readers into `src/utils/blender_scene_io.py`

**Files:**
- Create: `src/utils/blender_scene_io.py`
- Modify: `scripts/blender_export_fbx.py` (delete moved bodies, import instead)
- Test: `tests/test_blender_scene_io.py`

**Interfaces:**
- Consumes: nothing new — code moves verbatim from `scripts/blender_export_fbx.py`.
- Produces (all bpy-free; `np_mod` param pattern kept):
  - `iter_player_fbx_entries(output_dir: Path, np_mod) -> Iterator[dict]` — keys `shot_id, player_id, frames, thetas, root_R, root_t` (moved verbatim from `blender_export_fbx.py:36`)
  - `prepare_ball_keys(ball_frames: list[dict]) -> list[dict]` (moved verbatim from `:125`)
  - `load_shot_ids(output_dir: Path) -> set[str]` (moved from `_load_shot_ids` at `:160`, renamed public)
  - `load_camera_track(path: Path) -> dict` — `json.loads(path.read_text())` with a `ValueError` naming the path on parse failure
  - `load_smpl_body_data(repo_root: Path, np_mod) -> tuple[dict | None, "np.ndarray"]` — the smpl-npz load + foot-midpoint re-anchor block moved verbatim from `blender_export_fbx.py:470-500` (returns `(smpl_data_or_None, pelvis_canon_shifted)`)

- [ ] **Step 1: Write failing import/behaviour tests**

```python
# tests/test_blender_scene_io.py
import json
import numpy as np
import pytest

from src.utils import blender_scene_io as bio


@pytest.mark.unit
def test_load_shot_ids_empty_when_no_manifest(tmp_path):
    assert bio.load_shot_ids(tmp_path) == set()


@pytest.mark.unit
def test_load_shot_ids_reads_manifest(tmp_path):
    shots = {"shots": [{"id": "shot01"}, {"id": "shot02"}, {"noid": True}]}
    (tmp_path / "shots").mkdir()
    (tmp_path / "shots" / "shots_manifest.json").write_text(json.dumps(shots))
    assert bio.load_shot_ids(tmp_path) == {"shot01", "shot02"}


@pytest.mark.unit
def test_load_camera_track_error_names_path(tmp_path):
    bad = tmp_path / "camera_track.json"
    bad.write_text("{not json")
    with pytest.raises(ValueError, match="camera_track.json"):
        bio.load_camera_track(bad)


@pytest.mark.unit
def test_load_smpl_body_data_missing_returns_none(tmp_path):
    data, pelvis = bio.load_smpl_body_data(tmp_path, np)
    assert data is None
    assert pelvis.shape == (3,)


@pytest.mark.unit
def test_fbx_script_reexports_readers():
    # Existing tests/tools import these names from the script module —
    # the move must keep them resolvable there.
    from scripts import blender_export_fbx as bef
    assert bef.iter_player_fbx_entries is bio.iter_player_fbx_entries
    assert bef.prepare_ball_keys is bio.prepare_ball_keys
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv311/bin/python -m pytest tests/test_blender_scene_io.py -q`
Expected: FAIL — `ModuleNotFoundError: src.utils.blender_scene_io`

- [ ] **Step 3: Create the module by MOVING code (not rewriting)**

Cut `iter_player_fbx_entries` (lines 36–123), `prepare_ball_keys` (125–158), `_load_shot_ids` (160–167) and the smpl-npz block (the `smpl_npz_path = repo_root / "data" / "models" / "smpl_neutral.npz"` … `pelvis_canon_shifted = …` region inside `main`, ~lines 470–500) into `src/utils/blender_scene_io.py`. Wrap the smpl block as:

```python
def load_smpl_body_data(repo_root, np_mod):
    """Load data/models/smpl_neutral.npz re-anchored on the foot midpoint.

    Returns (smpl_data | None, pelvis_canon_shifted). Mirrors the FBX
    exporter's historical behaviour exactly — see that script's git
    history for the original block.
    """
    smpl_npz_path = repo_root / "data" / "models" / "smpl_neutral.npz"
    pelvis_canon_shifted = np_mod.zeros(3, dtype=np_mod.float64)
    if not smpl_npz_path.exists():
        return None, pelvis_canon_shifted
    from src.utils.smpl_skeleton import SMPL_JOINT_NAMES
    smpl_data = dict(np_mod.load(smpl_npz_path))
    if "joint_positions" in smpl_data and "v_template" in smpl_data:
        jp = np_mod.asarray(smpl_data["joint_positions"], dtype=np_mod.float64)
        l_foot = SMPL_JOINT_NAMES.index("l_foot")
        r_foot = SMPL_JOINT_NAMES.index("r_foot")
        shift = -((jp[l_foot] + jp[r_foot]) / 2.0)
        smpl_data["joint_positions"] = (jp + shift).astype(np_mod.float32)
        smpl_data["v_template"] = (
            np_mod.asarray(smpl_data["v_template"], dtype=np_mod.float64) + shift
        ).astype(np_mod.float32)
        pelvis_canon_shifted = np_mod.asarray(
            smpl_data["joint_positions"][0], dtype=np_mod.float64
        )
    return smpl_data, pelvis_canon_shifted
```

Add `load_camera_track`:

```python
def load_camera_track(path):
    try:
        return json.loads(Path(path).read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid camera track JSON at {path}: {exc}") from exc
```

In `scripts/blender_export_fbx.py`, replace the moved bodies with:

```python
from src.utils.blender_scene_io import (  # noqa: F401 — re-exported names
    iter_player_fbx_entries,
    load_shot_ids as _load_shot_ids,
    prepare_ball_keys,
)
```

(top of file, after the existing stdlib imports; the script already inserts repo root on `sys.path` inside `main` — move that `sys.path` insert to module top so the import works when Blender loads the script), and inside `main` replace the smpl block with:

```python
from src.utils.blender_scene_io import load_smpl_body_data
smpl_data, pelvis_canon_shifted = load_smpl_body_data(repo_root, np)
smpl_joint_positions = (
    smpl_data.get("joint_positions") if smpl_data is not None else None
)
```

- [ ] **Step 4: Run new + existing tests**

Run: `.venv311/bin/python -m pytest tests/test_blender_scene_io.py tests/test_blender_export_iter.py tests/test_gltf_ball_rotation.py tests/test_export_stage.py -q`
Expected: PASS (note: `tests/test_blender_export_smpl_skeleton.py::test_player_fbx_has_24_bones_and_full_keyframes` is a pre-existing known failure on this Mac — unchanged status is acceptable; every other test green).

- [ ] **Step 5: Commit**

```bash
git add src/utils/blender_scene_io.py scripts/blender_export_fbx.py tests/test_blender_scene_io.py
git commit -m "refactor: extract bpy-free Blender readers into blender_scene_io"
```

---

### Task 2: `render:` config, `RenderStage` skeleton, runner registration

**Files:**
- Create: `src/stages/render.py`
- Modify: `src/pipeline/runner.py:14-52`, `config/default.yaml` (after the `export:` block)
- Test: `tests/test_render_stage.py`

**Interfaces:**
- Consumes: `BaseStage` (`src/pipeline/base.py` — `__init__(config, output_dir, **kwargs)`, abstract `run()`, `is_complete()`); `manifest.active_shots()` pattern (copy the iteration idiom from `ExportStage.run`, `src/stages/export.py:329`).
- Produces: `RenderStage(BaseStage)` with `run()`, `is_complete()`, and `_blender_args(shot_id: str) -> list[str]` (pure, unit-testable: the exact argv list passed to subprocess). Stage name string: `"render"`.

- [ ] **Step 1: Add config block to `config/default.yaml`** (directly after the `export:` block; drone rig params go under the existing `export.virtual_cameras`):

```yaml
render:
  enabled: true
  blender_path: null          # null → fall back to export.blender_path
  cameras: [broadcast, drone] # also accepts pov:<PID>, ots:<PID>
  resolution: [1920, 1080]
  vertical_variant: false     # additionally render 1080x1920 for non-broadcast cams
  samples: 16                 # EEVEE taa_render_samples
  style:
    palette:
      grass_light: "#4d9e46"
      grass_dark: "#3f8a3a"
      lines: "#f5f5f0"
      sky_top: "#9ecfe8"
      sky_bottom: "#e8f4d8"
      outline: "#1a1a1a"
    ramp_steps: 3
    outline_width_m: 0.02
    grass_stripes: 10
  teams:
    defaults:
      home: {shirt: "#c0392b", shorts: "#ffffff", socks: "#c0392b"}
      away: {shirt: "#2980b9", shorts: "#2c3e50", socks: "#2980b9"}
      referee: {shirt: "#222222", shorts: "#222222", socks: "#222222"}
    by_player: {}             # e.g. {P003: away} — overrides tracking team class
  aov_passes: false
  save_blend: false
```

And extend `export.virtual_cameras` with:

```yaml
    drone_fov_deg: 55.0
    drone_height_m: 40.0
    drone_back_m: 25.0
    drone_smooth_frames: 25
```

- [ ] **Step 2: Write failing tests**

```python
# tests/test_render_stage.py
import json
import os
import stat
from pathlib import Path

import pytest

from src.pipeline.runner import _stage_class, resolve_stages
from src.stages.render import RenderStage


def _cfg(**over):
    cfg = {
        "render": {
            "enabled": True,
            "blender_path": None,
            "cameras": ["broadcast", "drone"],
            "resolution": [640, 360],
            "vertical_variant": False,
            "samples": 4,
            "style": {"ramp_steps": 3, "outline_width_m": 0.02,
                      "grass_stripes": 10, "palette": {}},
            "teams": {"defaults": {}, "by_player": {}},
            "aov_passes": False,
            "save_blend": False,
        },
        "export": {"blender_path": "blender", "virtual_cameras": {}},
    }
    cfg["render"].update(over)
    return cfg


@pytest.mark.unit
def test_render_stage_registered():
    assert "render" in resolve_stages("all", None)
    assert resolve_stages("all", None)[-1] == "render"
    assert _stage_class("render") is RenderStage


@pytest.mark.unit
def test_blender_args_shape(tmp_path):
    stage = RenderStage(_cfg(), tmp_path)
    args = stage._blender_args("shot01")
    # [blender, --background, --python, <script>, --, --output-dir, ...]
    assert args[1] == "--background"
    assert args[4] == "--"
    assert "--shot" in args and args[args.index("--shot") + 1] == "shot01"
    assert "--cameras" in args
    assert args[args.index("--cameras") + 1] == "broadcast,drone"
    assert "--width" in args and args[args.index("--width") + 1] == "640"


@pytest.mark.unit
def test_run_warns_and_skips_without_blender(tmp_path, caplog):
    cfg = _cfg()
    cfg["export"]["blender_path"] = "/nonexistent/blender-bin"
    stage = RenderStage(cfg, tmp_path)
    stage.run()  # must not raise
    assert not (tmp_path / "render").exists() or not any(
        (tmp_path / "render").rglob("*.mp4"))
    assert any("blender" in r.message.lower() for r in caplog.records)


@pytest.mark.integration
def test_run_invokes_blender_stub_per_active_shot(tmp_path):
    # Fake blender: records argv, writes the expected mp4.
    stub = tmp_path / "fake_blender"
    log = tmp_path / "calls.jsonl"
    stub.write_text(
        "#!/bin/sh\n"
        f"echo \"$@\" >> {log}\n"
        # emulate the script writing its outputs
        "exit 0\n"
    )
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC)
    (tmp_path / "shots").mkdir()
    (tmp_path / "shots" / "shots_manifest.json").write_text(json.dumps(
        {"shots": [
            {"id": "shot01", "status": "active"},
            {"id": "shot02", "status": "excluded"},
        ]}))
    cfg = _cfg(blender_path=str(stub))
    stage = RenderStage(cfg, tmp_path)
    stage.run()
    calls = log.read_text().strip().splitlines()
    assert len(calls) == 1               # excluded shot skipped
    assert "--shot shot01" in calls[0]
```

(Adapt the manifest fixture fields to whatever `tests/test_export_stage_manifest.py` uses for active/excluded shots — copy its fixture helper rather than inventing a new shape.)

- [ ] **Step 3: Run to verify failure**

Run: `.venv311/bin/python -m pytest tests/test_render_stage.py -q`
Expected: FAIL — `ModuleNotFoundError: src.stages.render`

- [ ] **Step 4: Implement `src/stages/render.py`**

```python
"""Render stage — headless Blender toon renders of each shot.

Thin orchestrator: resolves the Blender binary (render.blender_path,
falling back to export.blender_path), then shells to
scripts/blender_render_scene.py once per active shot. Degrades to a
warning when Blender is missing — same posture as FBX export.
"""
from __future__ import annotations

import json
import logging
import shutil
import subprocess
from pathlib import Path

from src.pipeline.base import BaseStage

logger = logging.getLogger(__name__)

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "blender_render_scene.py"


class RenderStage(BaseStage):
    def is_complete(self) -> bool:
        render_dir = self.output_dir / "render"
        return render_dir.exists() and any(render_dir.rglob("*.mp4"))

    def _render_cfg(self) -> dict:
        return self.config.get("render", {})

    def _resolve_blender(self) -> str | None:
        cfg = self._render_cfg()
        path = cfg.get("blender_path") or self.config.get("export", {}).get(
            "blender_path", "blender")
        if Path(path).is_absolute():
            return path if Path(path).exists() else None
        return shutil.which(path)

    def _blender_args(self, shot_id: str) -> list[str]:
        cfg = self._render_cfg()
        w, h = cfg.get("resolution", [1920, 1080])
        blender = self._resolve_blender() or "blender"
        args = [
            blender, "--background", "--python", str(_SCRIPT), "--",
            "--output-dir", str(self.output_dir),
            "--shot", shot_id,
            "--cameras", ",".join(cfg.get("cameras", ["broadcast"])),
            "--width", str(int(w)), "--height", str(int(h)),
            "--samples", str(int(cfg.get("samples", 16))),
            "--style-json", json.dumps(cfg.get("style", {})),
        ]
        if cfg.get("vertical_variant"):
            args.append("--vertical")
        if cfg.get("aov_passes"):
            args.append("--aov")
        if cfg.get("save_blend"):
            args.append("--save-blend")
        return args

    def _active_shot_ids(self) -> list[str]:
        # Same manifest walk as ExportStage._export_shot_ids
        # (src/stages/export.py:357): active shots, or [""] legacy.
        manifest_path = self.output_dir / "shots" / "shots_manifest.json"
        if not manifest_path.exists():
            return [""]
        raw = json.loads(manifest_path.read_text())
        ids = [s["id"] for s in raw.get("shots", [])
               if s.get("id") and s.get("status", "active") != "excluded"]
        return ids or [""]

    def run(self) -> None:
        if not self._render_cfg().get("enabled", True):
            logger.info("render: disabled via config; skipping")
            return
        if self._resolve_blender() is None:
            logger.warning(
                "render: Blender not found (render.blender_path / "
                "export.blender_path); skipping renders. Install Blender "
                ">= 3.6 or set the config path.")
            return
        timings: dict[str, float] = {}
        for shot_id in self._active_shot_ids():
            args = self._blender_args(shot_id)
            logger.info("render: shot=%s -> %s", shot_id or "<legacy>", args)
            import time
            t0 = time.time()
            result = subprocess.run(args, capture_output=True, text=True)
            timings[shot_id or "clip"] = round(time.time() - t0, 1)
            if result.returncode != 0:
                logger.error("render: Blender failed for shot %s:\n%s",
                             shot_id, result.stderr[-4000:])
        out = self.output_dir / "render"
        out.mkdir(parents=True, exist_ok=True)
        (out / "render_timings.json").write_text(json.dumps(timings, indent=2))
```

**Exact manifest-walk parity:** before finishing, read `src/stages/export.py:357` (`_export_shot_ids`) and mirror its active/excluded logic exactly (including the legacy `[""]` fallback and any status-field naming) — the snippet above is the shape, the export stage is the authority.

Register in `src/pipeline/runner.py`: append `"render"` to `_STAGE_NAMES` (after `"export"`) and add to `_stage_class`:

```python
    if name == "render":
        from src.stages.render import RenderStage
        return RenderStage
```

- [ ] **Step 5: Run tests**

Run: `.venv311/bin/python -m pytest tests/test_render_stage.py tests/test_export_stage.py -q`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/stages/render.py src/pipeline/runner.py config/default.yaml tests/test_render_stage.py
git commit -m "feat: register render stage skeleton (headless Blender orchestrator)"
```

---

### Task 3: Drone camera preset in `virtual_cameras.py`

**Files:**
- Modify: `src/utils/virtual_cameras.py` (add fields to `RigConfig`; add `build_drone_track` after `build_ots_track`)
- Test: `tests/test_virtual_cameras_drone.py`

**Interfaces:**
- Consumes: `look_at_view(center, target) -> (R, t)`, `intrinsics_from_fov(fov_deg, image_size)`, `_make_track(clip_id, image_size, fps, K, per_frame)`, `_ball_xyz_by_frame(ball_track)` — all existing in the module; `SmplWorldTrack.root_t/frames`.
- Produces: `build_drone_track(tracks: Sequence[SmplWorldTrack], ball_track: object, cfg: RigConfig, image_size: tuple[int, int], fps: float, clip_id: str) -> CameraTrack`; `RigConfig` gains `drone_fov_deg: float = 55.0`, `drone_height_m: float = 40.0`, `drone_back_m: float = 25.0`, `drone_smooth_frames: int = 25`.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_virtual_cameras_drone.py
import numpy as np
import pytest

from src.schemas.smpl_world import SmplWorldTrack
from src.utils import virtual_cameras as vcam


def _static_track(pid, x, y, n=10):
    return SmplWorldTrack(
        player_id=pid,
        frames=np.arange(n),
        betas=np.zeros(10),
        thetas=np.zeros((n, 24, 3)),
        root_R=np.tile(np.eye(3), (n, 1, 1)),
        root_t=np.tile(np.array([x, y, 0.9]), (n, 1)),
        confidence=np.ones(n),
    )


@pytest.mark.unit
def test_drone_hovers_behind_and_above_centroid():
    tracks = [_static_track("P001", 40.0, 30.0), _static_track("P002", 60.0, 30.0)]
    cfg = vcam.RigConfig()
    track = vcam.build_drone_track(tracks, None, cfg, (1920, 1080), 25.0, "clip")
    assert len(track.frames) == 10
    fr = track.frames[0]
    R, t = np.asarray(fr.R), np.asarray(fr.t)
    C = -R.T @ t                       # camera centre in world
    assert C[2] == pytest.approx(cfg.drone_height_m, abs=1e-6)
    assert C[0] == pytest.approx(50.0, abs=1e-6)          # centroid x
    assert C[1] == pytest.approx(30.0 - cfg.drone_back_m, abs=1e-6)
    # looks at the centroid: forward (row 2 of R) points from C to centroid
    fwd = R[2]
    to_target = np.array([50.0, 30.0, 0.9]) - C
    assert np.dot(fwd, to_target / np.linalg.norm(to_target)) > 0.999


@pytest.mark.unit
def test_drone_smooths_jittery_centroid():
    n = 50
    zig = _static_track("P001", 50.0, 30.0, n)
    zig.root_t[::2, 0] += 5.0          # 5 m x-jitter every other frame
    cfg = vcam.RigConfig(drone_smooth_frames=25)
    track = vcam.build_drone_track([zig], None, cfg, (1920, 1080), 25.0, "clip")
    centres = np.array([-(np.asarray(f.R)).T @ np.asarray(f.t)
                        for f in track.frames])
    dx = np.abs(np.diff(centres[:, 0]))
    assert dx.max() < 1.0              # jitter absorbed by the moving average


@pytest.mark.unit
def test_drone_includes_ball_in_centroid():
    tracks = [_static_track("P001", 40.0, 30.0)]
    ball = type("BT", (), {"frames": [
        {"frame": i, "world_xyz": [80.0, 30.0, 0.11]} for i in range(10)]})()
    cfg = vcam.RigConfig()
    with_ball = vcam.build_drone_track(tracks, ball, cfg, (1920, 1080), 25.0, "c")
    without = vcam.build_drone_track(tracks, None, cfg, (1920, 1080), 25.0, "c")
    cx_with = -(np.asarray(with_ball.frames[0].R)).T @ np.asarray(with_ball.frames[0].t)
    cx_without = -(np.asarray(without.frames[0].R)).T @ np.asarray(without.frames[0].t)
    assert cx_with[0] > cx_without[0]  # ball at x=80 pulls the view right
```

(Adjust the fake ball object to match `_ball_xyz_by_frame`'s expected shape — read `src/utils/virtual_cameras.py:109` first and mimic exactly what it consumes.)

- [ ] **Step 2: Run to verify failure**

Run: `.venv311/bin/python -m pytest tests/test_virtual_cameras_drone.py -q`
Expected: FAIL — `AttributeError: ... has no attribute 'build_drone_track'` (and/or `TypeError` for the new `RigConfig` field).

- [ ] **Step 3: Implement**

Add the four fields to `RigConfig` with the defaults above, then:

```python
def build_drone_track(
    tracks: "Sequence[SmplWorldTrack]",
    ball_track: object,
    cfg: RigConfig,
    image_size: tuple[int, int],
    fps: float,
    clip_id: str,
) -> CameraTrack:
    """Elevated tactical camera tracking the smoothed action centroid.

    Per frame the action centroid is the mean of all player root
    positions present on that frame plus the ball (when tracked); the
    camera sits ``drone_back_m`` toward the near touchline (-y) and
    ``drone_height_m`` up, looking at the centroid. The centroid is
    smoothed with a centered moving average over ``drone_smooth_frames``
    frames so single-frame jitter never reaches the camera.
    """
    K = intrinsics_from_fov(cfg.drone_fov_deg, image_size)
    ball_xyz = _ball_xyz_by_frame(ball_track) if ball_track is not None else {}

    # Union of frame indices across tracks.
    all_frames = sorted({int(f) for tr in tracks
                         for f in np.asarray(tr.frames).tolist()})
    if not all_frames:
        return _make_track(clip_id, image_size, fps, K, [])

    # Raw per-frame centroid.
    by_frame_pos: dict[int, list[np.ndarray]] = {f: [] for f in all_frames}
    for tr in tracks:
        idx = {int(f): i for i, f in enumerate(np.asarray(tr.frames).tolist())}
        for f, i in idx.items():
            by_frame_pos[f].append(np.asarray(tr.root_t[i], dtype=np.float64))
    raw = []
    for f in all_frames:
        pts = list(by_frame_pos[f])
        if f in ball_xyz:
            pts.append(np.asarray(ball_xyz[f], dtype=np.float64))
        raw.append(np.mean(pts, axis=0))
    raw_arr = np.asarray(raw)

    # Centered moving average (edge-padded).
    win = max(1, int(cfg.drone_smooth_frames))
    pad = win // 2
    padded = np.pad(raw_arr, ((pad, pad), (0, 0)), mode="edge")
    kernel = np.ones(win) / win
    smooth = np.stack(
        [np.convolve(padded[:, k], kernel, mode="valid") for k in range(3)],
        axis=1,
    )[: len(all_frames)]

    per_frame: list[_FrameTuple] = []
    for f, target in zip(all_frames, smooth):
        centre = np.array([target[0],
                           target[1] - cfg.drone_back_m,
                           cfg.drone_height_m])
        R, t = look_at_view(centre, target)
        per_frame.append((int(f), R, t, 1.0))
    return _make_track(clip_id, image_size, fps, K, per_frame)
```

- [ ] **Step 4: Run tests**

Run: `.venv311/bin/python -m pytest tests/test_virtual_cameras_drone.py tests/test_virtual_cameras.py tests/test_export_virtual_cameras.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/utils/virtual_cameras.py tests/test_virtual_cameras_drone.py
git commit -m "feat: drone virtual-camera preset (smoothed action-centroid look-at)"
```

---

### Task 4: Pure look helpers — `src/utils/render_look.py`

**Files:**
- Create: `src/utils/render_look.py`
- Test: `tests/test_render_look.py`

**Interfaces:**
- Consumes: nothing project-specific (pure math + stdlib).
- Produces:
  - `kit_zone_for_height_fraction(f: float) -> str` — one of `"socks" | "skin" | "shorts" | "shirt"`
  - `hex_to_linear_rgba(hex_str: str) -> tuple[float, float, float, float]`
  - `resolve_player_colors(teams_cfg: dict, team_class: dict[str, tuple[str, str]]) -> dict[str, dict[str, tuple]]` — per player: `{"shirt": rgba, "shorts": rgba, "socks": rgba}`
  - `blender_camera_world_matrix(R: list[list[float]], t: list[float]) -> list[list[float]]` — 4×4 row-major world matrix for a Blender camera
  - `lens_mm_from_K(K: list[list[float]], width_px: int, sensor_mm: float = 36.0) -> float`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_render_look.py
import numpy as np
import pytest

from src.utils import render_look as rl
from src.utils.virtual_cameras import intrinsics_from_fov, look_at_view


@pytest.mark.unit
@pytest.mark.parametrize("frac,zone", [
    (0.05, "socks"), (0.14, "socks"),
    (0.20, "skin"), (0.47, "skin"),
    (0.50, "shorts"), (0.57, "shorts"),
    (0.60, "shirt"), (0.85, "shirt"),
    (0.90, "skin"), (1.00, "skin"),      # head
])
def test_kit_zones(frac, zone):
    assert rl.kit_zone_for_height_fraction(frac) == zone


@pytest.mark.unit
def test_hex_to_linear_rgba():
    r, g, b, a = rl.hex_to_linear_rgba("#ffffff")
    assert (r, g, b, a) == (1.0, 1.0, 1.0, 1.0)
    r, g, b, a = rl.hex_to_linear_rgba("#000000")
    assert (r, g, b) == (0.0, 0.0, 0.0)
    # mid-grey: linearised value must be < srgb value (gamma expansion)
    r, _, _, _ = rl.hex_to_linear_rgba("#808080")
    assert 0.15 < r < 0.25


@pytest.mark.unit
def test_resolve_player_colors_by_class_and_override():
    teams = {
        "defaults": {
            "home": {"shirt": "#ff0000", "shorts": "#ffffff", "socks": "#ff0000"},
            "away": {"shirt": "#0000ff", "shorts": "#000000", "socks": "#0000ff"},
        },
        "by_player": {"P009": "away"},
    }
    team_class = {"P001": ("home", "player"), "P009": ("home", "player")}
    colors = rl.resolve_player_colors(teams, team_class)
    assert colors["P001"]["shirt"] == rl.hex_to_linear_rgba("#ff0000")
    assert colors["P009"]["shirt"] == rl.hex_to_linear_rgba("#0000ff")  # override wins


@pytest.mark.unit
def test_blender_camera_matrix_position_and_forward():
    centre = np.array([10.0, 5.0, 20.0])
    target = np.array([50.0, 34.0, 0.0])
    R, t = look_at_view(centre, target)
    M = np.asarray(rl.blender_camera_world_matrix(
        [list(r) for r in R], list(t)))
    assert M[:3, 3] == pytest.approx(centre, abs=1e-9)   # translation = C
    # Blender cameras look down local -Z: -M[:3,2] must point at target.
    fwd = -M[:3, 2]
    expect = (target - centre) / np.linalg.norm(target - centre)
    assert fwd == pytest.approx(expect, abs=1e-9)


@pytest.mark.unit
def test_lens_mm_from_K():
    K = intrinsics_from_fov(46.8, (1920, 1080))  # ≈ 36mm-equiv horizontal fov
    lens = rl.lens_mm_from_K(K, 1920)
    assert lens == pytest.approx(41.6, abs=1.0)
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv311/bin/python -m pytest tests/test_render_look.py -q`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement**

```python
"""Pure look/color/camera math for the render stage (no bpy)."""
from __future__ import annotations

import numpy as np

# Rest-pose height fractions (0=sole, 1=crown). Arms inherit the shirt
# color in v1 (long-sleeve reading; acceptable under the toon look).
_ZONES = (
    (0.15, "socks"),
    (0.48, "skin"),      # legs
    (0.58, "shorts"),
    (0.86, "shirt"),     # torso + arms
    (1.01, "skin"),      # head/neck
)


def kit_zone_for_height_fraction(f: float) -> str:
    f = float(min(max(f, 0.0), 1.0))
    for upper, zone in _ZONES:
        if f < upper:
            return zone
    return "skin"


def _srgb_to_linear(c: float) -> float:
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def hex_to_linear_rgba(hex_str: str) -> tuple[float, float, float, float]:
    s = hex_str.lstrip("#")
    if len(s) != 6:
        raise ValueError(f"Expected #RRGGBB, got {hex_str!r}")
    srgb = [int(s[i:i + 2], 16) / 255.0 for i in (0, 2, 4)]
    lin = [0.0 if v == 0.0 else 1.0 if v == 1.0 else _srgb_to_linear(v)
           for v in srgb]
    return (lin[0], lin[1], lin[2], 1.0)


def resolve_player_colors(
    teams_cfg: dict,
    team_class: dict[str, tuple[str, str]],
) -> dict[str, dict[str, tuple]]:
    defaults = teams_cfg.get("defaults", {})
    overrides = teams_cfg.get("by_player", {})
    fallback = {"shirt": "#888888", "shorts": "#666666", "socks": "#888888"}
    out: dict[str, dict[str, tuple]] = {}
    for pid, (team, _cls) in team_class.items():
        key = overrides.get(pid, team)
        kit = defaults.get(key, fallback)
        out[pid] = {part: hex_to_linear_rgba(kit.get(part, fallback[part]))
                    for part in ("shirt", "shorts", "socks")}
    return out


def blender_camera_world_matrix(
    R: list[list[float]], t: list[float],
) -> list[list[float]]:
    """OpenCV world->camera (R, t) to a Blender camera world matrix.

    OpenCV camera axes: +X right, +Y down, +Z forward. Blender cameras
    look down -Z with +Y up, so the rotation columns flip on Y and Z.
    """
    R = np.asarray(R, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64).reshape(3)
    C = -R.T @ t
    R_bl = R.T @ np.diag([1.0, -1.0, -1.0])
    M = np.eye(4)
    M[:3, :3] = R_bl
    M[:3, 3] = C
    return [[float(v) for v in row] for row in M]


def lens_mm_from_K(
    K: list[list[float]], width_px: int, sensor_mm: float = 36.0,
) -> float:
    fx = float(K[0][0])
    return fx * sensor_mm / float(width_px)
```

- [ ] **Step 4: Run tests**

Run: `.venv311/bin/python -m pytest tests/test_render_look.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/utils/render_look.py tests/test_render_look.py
git commit -m "feat: pure render-look helpers (kit zones, colors, blender camera math)"
```

---

### Task 5: Blender script skeleton — environment, broadcast camera, ball, MP4 out

**Files:**
- Create: `scripts/blender_render_scene.py`
- Test: `tests/test_blender_render_scene.py` (pure-arg tests unmarked + smoke test with `@pytest.mark.fbx`)

**Interfaces:**
- Consumes: `blender_scene_io.load_camera_track / prepare_ball_keys / load_shot_ids`; `render_look.blender_camera_world_matrix / lens_mm_from_K / hex_to_linear_rgba`; `src/utils/pitch.py` constants (`PITCH_LENGTH`, `PITCH_WIDTH`, `_CIRCLE_R` — import the public pair, redeclare circle radius locally as `CENTRE_CIRCLE_R = 9.15` rather than importing a private name).
- Produces: CLI `blender --background --python scripts/blender_render_scene.py -- --output-dir D --shot S --cameras a,b --width W --height H --samples N --style-json J [--vertical] [--aov] [--save-blend] [--frame-start N] [--frame-end N]`; writes `output/render/<shot>/<camera>.mp4`. Internal functions later tasks extend: `_build_environment(style: dict) -> None`, `_build_ball(ball_keys: list[dict]) -> object`, `_add_camera_from_track(cam_id: str, track: dict, width: int, height: int) -> object`, `_render(camera_obj, out_path: Path, fps: float, frame_range: tuple[int, int], width: int, height: int, samples: int) -> None`.

- [ ] **Step 1: Write the arg-parsing test (pure, runs without Blender)**

```python
# tests/test_blender_render_scene.py
import pytest

from scripts.blender_render_scene import _parse_args


@pytest.mark.unit
def test_parse_args_after_double_dash():
    ns = _parse_args([
        "blender", "--background", "--",
        "--output-dir", "/tmp/o", "--shot", "shot01",
        "--cameras", "broadcast,drone", "--width", "640", "--height", "360",
        "--samples", "4", "--style-json", "{}",
    ])
    assert ns.shot == "shot01"
    assert ns.cameras == ["broadcast", "drone"]
    assert (ns.width, ns.height) == (640, 360)
    assert ns.vertical is False and ns.aov is False
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv311/bin/python -m pytest tests/test_blender_render_scene.py -q -m unit`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement the script**

Skeleton (structure identical to `blender_export_fbx.py`: module-level pure helpers + `main` that lazily imports bpy):

```python
"""Headless Blender toon renderer for the broadcast-mono pipeline.

Invoked by RenderStage via:

    blender --background --python scripts/blender_render_scene.py -- \
        --output-dir OUT --shot SHOT --cameras broadcast,drone ...

Assembles a fully procedural scene (no binary assets): pitch + lines
from src/utils/pitch.py geometry, procedural stadium bowl, players from
refined_poses NPZs, ball from the dense ball track. Renders EEVEE to
output/render/<shot>/<camera>.mp4.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _parse_args(argv: list[str]) -> argparse.Namespace:
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    p = argparse.ArgumentParser(description="Toon render of one shot")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--shot", default="")
    p.add_argument("--cameras", type=lambda s: s.split(","),
                   default=["broadcast"])
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--height", type=int, default=1080)
    p.add_argument("--samples", type=int, default=16)
    p.add_argument("--style-json", default="{}")
    p.add_argument("--vertical", action="store_true")
    p.add_argument("--aov", action="store_true")
    p.add_argument("--save-blend", action="store_true")
    p.add_argument("--frame-start", type=int, default=None)
    p.add_argument("--frame-end", type=int, default=None)
    return p.parse_args(argv)
```

`main(argv)` (called under `if __name__ == "__main__": sys.exit(main(sys.argv))`):

1. Parse args, `style = json.loads(args.style_json)`, resolve palette with the Task 2 defaults for missing keys.
2. `import bpy` (exit 2 with a stderr message when unavailable — copy the FBX script's pattern verbatim).
3. Version assert: `if bpy.app.version < (3, 6, 0): sys.stderr.write(...); return 2`.
4. Fresh scene: `bpy.ops.wm.read_factory_settings(use_empty=True)`.
5. `_build_environment(style)`:
   - Pitch: `bpy.ops.mesh.primitive_plane_add(size=1)`, scale to `(PITCH_LENGTH + 10, PITCH_WIDTH + 10, 1)` centred at `(PITCH_LENGTH/2, PITCH_WIDTH/2, 0)`; material `M_Grass` = node tree: `Texture Coordinate → Separate XYZ (x) → Math(multiply, grass_stripes/PITCH_LENGTH) → Math(floor) → Math(modulo 2) → MixRGB(grass_light, grass_dark) → Diffuse BSDF`.
   - Lines (all at z = 0.02, emission `lines` color, curve bevel depth 0.06): outer rectangle `(0,0)-(105,68)`, halfway line, centre circle (`bpy.ops.curve.primitive_bezier_circle_add(radius=9.15, location=(52.5, 34, 0.02))`), two penalty boxes (16.5 m deep × 40.32 m wide) and six-yard boxes (5.5 × 18.32) from `pitch.py`-mirroring constants declared at module top with a comment pointing at `src/utils/pitch.py`.
   - Goals: 4 cylinders (posts, r=0.06, h=2.44) + 2 horizontal cylinders (crossbars, length 7.32) at x=0 and x=105, y centred on 34.
   - Stadium bowl: `bpy.ops.mesh.primitive_circle_add(vertices=48, radius=95, fill_mode='NOTHING', location=(52.5, 34, 0))`, extrude/scale twice in edit-mesh via `bmesh` to make a raked ring (inner r=75 h=2 → outer r=95 h=18), flat dark two-tone material.
   - World: gradient via world nodes (`sky_top`/`sky_bottom` mix on Texture Coordinate Generated Z); one Sun light, rotation `(radians(50), 0, radians(-30))`, energy 3.
6. Ball: read `output/ball/<shot>_ball_track.json` (fall back to `ball/ball_track.json` when shot is `""` — mirror how the FBX exporter locates it), `keys = prepare_ball_keys(raw["frames"])`, UV sphere r=0.11, keyframe `location` and `rotation_quaternion` per key (`rotation_mode = "QUATERNION"`).
7. Cameras: for each requested id, locate the track: `broadcast` → `output/camera/<shot>_camera_track.json` (or `camera/camera_track.json` legacy); anything else → `output/render/<shot>/cameras/<id>_camera_track.json` (Task 8 writes these; error out with a clear message when missing). Build: `bpy.data.cameras.new`, per-frame `obj.matrix_world = Matrix(render_look.blender_camera_world_matrix(fr["R"], fr["t"]))` + `cam.lens = render_look.lens_mm_from_K(fr["K"], width)`, keyframe `location`, `rotation_euler`, and `data.lens` each frame; `cam.sensor_width = 36.0`, `cam.sensor_fit = 'HORIZONTAL'`.
8. `_render(...)`: scene fps from the broadcast track's `fps` field (default 25); `scene.render.engine = "BLENDER_EEVEE_NEXT"` with fallback to `"BLENDER_EEVEE"` when the enum is missing (pre-4.2 Blender); `scene.eevee.taa_render_samples = samples`; output `FFMPEG`/`MPEG4`/`H264`, `filepath = str(out_path)`; `scene.frame_start/end` from the track (or `--frame-start/--frame-end` when given); `bpy.ops.render.render(animation=True)`. Print `RENDER_TIMING <camera> <seconds> <n_frames>` to stdout (the stage's log captures it; the quality report parses `render_timings.json` instead — this is for eyeballing).
9. `--save-blend`: `bpy.ops.wm.save_as_mainfile(filepath=str(out_dir / "scene.blend"))` before rendering.

- [ ] **Step 4: Write the fixture-based smoke test (fbx-marked)**

```python
import json
import shutil
import subprocess

import numpy as np
import pytest

_BLENDER = shutil.which("blender")


def _write_min_fixture(root):
    """Minimal single-shot output dir: camera track + ball track, no players."""
    n = 3
    (root / "camera").mkdir(parents=True)
    K = [[1000.0, 0, 320.0], [0, 1000.0, 180.0], [0, 0, 1.0]]
    frames = []
    for i in range(n):
        # camera 20m up on the near touchline looking at pitch centre
        from src.utils.virtual_cameras import look_at_view
        R, t = look_at_view(np.array([52.5, -20.0, 20.0]),
                            np.array([52.5, 34.0, 0.0]))
        frames.append({"frame": i, "K": K,
                       "R": [list(r) for r in R], "t": list(t),
                       "confidence": 1.0, "is_anchor": False})
    (root / "camera" / "camera_track.json").write_text(json.dumps(
        {"clip_id": "clip", "fps": 25.0, "image_size": [640, 360],
         "frames": frames}))
    (root / "ball").mkdir()
    (root / "ball" / "ball_track.json").write_text(json.dumps(
        {"frames": [{"frame": i, "world_xyz": [52.5, 34.0, 0.11],
                     "state": "rolling"} for i in range(n)]}))


@pytest.mark.fbx
@pytest.mark.skipif(_BLENDER is None, reason="blender not on PATH")
def test_smoke_render_broadcast_mp4(tmp_path):
    _write_min_fixture(tmp_path)
    script = "scripts/blender_render_scene.py"
    res = subprocess.run(
        [_BLENDER, "--background", "--python", script, "--",
         "--output-dir", str(tmp_path), "--shot", "",
         "--cameras", "broadcast", "--width", "160", "--height", "90",
         "--samples", "1", "--style-json", "{}",
         "--frame-start", "0", "--frame-end", "2"],
        capture_output=True, text=True, timeout=600)
    assert res.returncode == 0, res.stderr[-3000:]
    out = tmp_path / "render" / "clip" / "broadcast.mp4"
    assert out.exists() and out.stat().st_size > 0
```

Match the ball-track JSON field names to what `prepare_ball_keys` consumes — read its docstring (`src/utils/blender_scene_io.py` after Task 1) and the real `output/ball/*_ball_track.json` on disk first; adjust the fixture keys (`world_xyz` vs `xyz` etc.) to the real schema. Same for the camera-track fixture vs `src/schemas/camera_track.py`. The `"clip"` output-directory name for the legacy empty shot id must match what `main` uses (`shot or "clip"` — keep consistent in both).

- [ ] **Step 5: Run both test groups**

Run: `.venv311/bin/python -m pytest tests/test_blender_render_scene.py -q`
Expected: unit test PASS; smoke test PASS locally (Blender is installed on this Mac) — note the wall-clock printed by `RENDER_TIMING`; if a 160×90 3-frame render exceeds ~60 s total, STOP and investigate the EEVEE backend before continuing (perf gate from spec §7).

- [ ] **Step 6: Commit**

```bash
git add scripts/blender_render_scene.py tests/test_blender_render_scene.py
git commit -m "feat: blender render script — procedural environment, broadcast cam, ball"
```

---

### Task 6: Players — armatures, capsule-limb fallback body, kit materials

**Files:**
- Modify: `scripts/blender_render_scene.py`
- Test: `tests/test_blender_render_scene.py` (extend)

**Interfaces:**
- Consumes: `blender_scene_io.iter_player_fbx_entries(output_dir, np)` (per-player `frames/thetas/root_R/root_t/shot_id`); `blender_scene_io.load_smpl_body_data(_REPO_ROOT, np)`; `smpl_skeleton.SMPL_JOINT_NAMES / SMPL_PARENTS / SMPL_REST_JOINTS_YUP / axis_angle_to_quaternion`; `render_look.kit_zone_for_height_fraction / resolve_player_colors`; `src.stages.export._player_team_class_map(output_dir)` (module-level, no stage instance needed).
- Produces: `_build_players(output_dir: Path, shot_id: str, colors: dict, smpl_data, pelvis_canon) -> list[object]` in the render script. Armature recipe (same convention as the FBX exporter's docstring): canonical rest pose from `SMPL_REST_JOINTS_YUP` (or `smpl_data["joint_positions"]` when the asset exists); per-frame pose-bone `rotation_quaternion` from `thetas` via `axis_angle_to_quaternion` — **bone 0 (pelvis) gets identity, `thetas[0]` is ignored**; per-frame armature object `matrix_world` from `root_R`/`root_t` (translation `root_t - root_R @ pelvis_canon` when the SMPL asset re-anchoring is active, matching the FBX exporter's comment at the old `blender_export_fbx.py:474`).

- [ ] **Step 1: Extend the smoke fixture with one synthetic player**

```python
def _add_player_fixture(root, n=3):
    (root / "refined_poses").mkdir()
    np.savez(root / "refined_poses" / "P001_refined.npz",
             player_id="P001",
             frames=np.arange(n),
             betas=np.zeros(10, dtype=np.float32),
             thetas=np.zeros((n, 24, 3), dtype=np.float32),
             root_R=np.tile(np.eye(3, dtype=np.float32), (n, 1, 1)),
             root_t=np.tile(np.array([52.5, 30.0, 0.95], dtype=np.float32),
                            (n, 1)),
             confidence=np.ones(n, dtype=np.float32))
```

First, open a real `output/refined_poses/*_refined.npz` (`.venv311/bin/python -c "import numpy as np; d=np.load('output/refined_poses/P001_refined.npz'); print(d.files)"`) and mirror its exact key set in the fixture — `iter_player_fbx_entries` dictates the required keys. Add a second smoke test `test_smoke_render_with_player` that renders 1 frame at 160×90 with the player fixture and asserts: exit 0, mp4 exists, and stdout contains `PLAYERS_BUILT 1` (have `_build_players` print that count).

- [ ] **Step 2: Run to verify the new smoke test fails**

Run: `.venv311/bin/python -m pytest tests/test_blender_render_scene.py -q -k player`
Expected: FAIL — `PLAYERS_BUILT` marker absent (players not implemented).

- [ ] **Step 3: Implement `_build_players`**

Per entry from `iter_player_fbx_entries` (filtered to `entry["shot_id"] in ("", shot_id)`):

1. **Armature**: `bpy.data.armatures.new(f"{pid}_arm")` in edit mode create 24 bones named per `SMPL_JOINT_NAMES`, head at rest joint position, tail = head + 0.05 z (leaf) or child-mean (internal), parent per `SMPL_PARENTS`. Rest joints: `smpl_data["joint_positions"]` if present else `SMPL_REST_JOINTS_YUP`. NOTE the rest joints are Y-up canonical; the armature object's per-frame `root_R` maps canonical→pitch-world, so build the armature in canonical axes and let the object matrix do all the work — do NOT pre-rotate the rest pose (this mirrors the FBX exporter's design described in its module docstring).
2. **Pose keys**: for each frame `i`, for joints 1..23: `pose.bones[name].rotation_quaternion = axis_angle_to_quaternion(thetas[i, j])`, keyframe insert; pelvis bone stays identity (`thetas[0]` IGNORED — repo convention).
3. **Object keys**: build `Matrix` from `root_R[i]` (3×3 → 4×4) with translation `root_t[i] - root_R[i] @ pelvis_canon` (asset case) or `root_t[i]` (fallback rest joints, whose pelvis is origin-anchored — verify by printing `SMPL_REST_JOINTS_YUP[0]` once and adjusting the same way if non-zero); keyframe `location` + `rotation_quaternion` on the object.
4. **Body**:
   - Asset present → copy the mesh recipe from the FBX exporter's `_add_smpl_skinned_mesh` (`scripts/blender_export_fbx.py:380`): `from_pydata(v_template, [], faces)`, one vertex group per joint from `weights`, Armature modifier. Then assign kit materials: compute per-vertex height fraction from `v_template[:, 1]` (canonical Y-up!) normalised min→max; bucket into 4 material slots via `kit_zone_for_height_fraction`; per-face zone = majority of its vertices' zones.
   - Asset absent → capsule-limb fallback: for each bone with length > 0.02, add a cylinder (r=0.055 for limbs, 0.10 for spine bones, 0.09 sphere for head) parented to the bone (`parent_type='BONE'`), material by the zone of the bone midpoint height fraction.
   - Materials: `bpy.data.materials.new(f"{pid}_{zone}")`, `use_nodes=True`, set Principled Base Color to `colors[pid][zone]` (zone `"skin"` uses a fixed `#c68863` linearised constant declared at module top). Toon conversion happens in Task 7 — flat Principled is the placeholder.

Wire into `main`: `team_class = _player_team_class_map(output_dir)`, `colors = resolve_player_colors(style_teams_cfg, team_class)` — pass the `render.teams` dict through `--style-json` by extending the stage in this task: in `src/stages/render.py::_blender_args`, change `"--style-json", json.dumps(cfg.get("style", {}))` to embed teams: `json.dumps({**cfg.get("style", {}), "teams": cfg.get("teams", {})})`, and update the Task 2 argv test to assert `"teams"` appears in the style JSON.

- [ ] **Step 4: Run tests**

Run: `.venv311/bin/python -m pytest tests/test_blender_render_scene.py tests/test_render_stage.py -q`
Expected: PASS (smoke renders with 1 player; stage argv test updated).

- [ ] **Step 5: Commit**

```bash
git add scripts/blender_render_scene.py src/stages/render.py tests/
git commit -m "feat: render players — SMPL armature + mesh/capsule body + kit zones"
```

---

### Task 7: Toon look — ramp materials, inverted-hull outlines, blob shadows

**Files:**
- Modify: `scripts/blender_render_scene.py`
- Test: `tests/test_blender_render_scene.py` (extend smoke assertions)

**Interfaces:**
- Consumes: `style` dict (`ramp_steps`, `outline_width_m`, `palette.outline`); existing `_build_players` / `_build_ball` / `_build_environment`.
- Produces: `_toon_material(name: str, rgba: tuple, ramp_steps: int) -> object` replacing every flat Principled material; `_add_outline(obj, width_m: float, rgba: tuple) -> None`; `_add_blob_shadow(target_obj, radius_m: float) -> object`.

- [ ] **Step 1: Extend the player smoke test**

Add to `test_smoke_render_with_player`: render with `--style-json '{"ramp_steps": 3, "outline_width_m": 0.03}'` and assert stdout contains `TOON_MATERIALS <n>` with n ≥ 4 and `OUTLINES <n>` with n ≥ 2 (script prints counters after scene build). Run; expected FAIL (markers absent).

- [ ] **Step 2: Implement**

```python
def _toon_material(name, rgba, ramp_steps):
    """Diffuse -> Shader-to-RGB -> constant ColorRamp -> Emission.

    The ramp quantises lighting into ``ramp_steps`` bands (classic cel
    shading). Emission output keeps the bands flat and print-like.
    """
    import bpy
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    diffuse = nt.nodes.new("ShaderNodeBsdfDiffuse")
    diffuse.inputs["Color"].default_value = rgba
    to_rgb = nt.nodes.new("ShaderNodeShaderToRGB")
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.interpolation = "CONSTANT"
    # evenly spaced constant stops from 35% to 100% brightness
    ramp.color_ramp.elements[0].position = 0.0
    ramp.color_ramp.elements[0].color = tuple(c * 0.35 for c in rgba[:3]) + (1.0,)
    ramp.color_ramp.elements[1].position = 0.55
    ramp.color_ramp.elements[1].color = rgba
    for k in range(1, ramp_steps - 1):
        el = ramp.color_ramp.elements.new(0.15 + 0.4 * k / max(1, ramp_steps - 1))
        f = 0.35 + 0.65 * k / max(1, ramp_steps - 1)
        el.color = tuple(c * f for c in rgba[:3]) + (1.0,)
    emit = nt.nodes.new("ShaderNodeEmission")
    nt.links.new(diffuse.outputs["BSDF"], to_rgb.inputs["Shader"])
    nt.links.new(to_rgb.outputs["Color"], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], emit.inputs["Color"])
    nt.links.new(emit.outputs["Emission"], out.inputs["Surface"])
    return mat


def _add_outline(obj, width_m, rgba):
    """Inverted-hull outline: Solidify with flipped normals + backface-culled
    emission black shell."""
    import bpy
    mat = bpy.data.materials.new(obj.name + "_outline")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    emit = nt.nodes.new("ShaderNodeEmission")
    emit.inputs["Color"].default_value = rgba
    nt.links.new(emit.outputs["Emission"], out.inputs["Surface"])
    mat.use_backface_culling = True
    obj.data.materials.append(mat)
    mod = obj.modifiers.new("Outline", "SOLIDIFY")
    mod.thickness = -abs(width_m)
    mod.use_flip_normals = True
    mod.material_offset = len(obj.data.materials) - 1


def _add_blob_shadow(target_obj, radius_m):
    """Soft dark disc at z=0.01 following the target's XY (drivers)."""
    import bpy
    bpy.ops.mesh.primitive_circle_add(vertices=24, radius=radius_m,
                                      fill_mode="NGON")
    disc = bpy.context.active_object
    disc.location.z = 0.01
    for axis in (0, 1):
        drv = disc.driver_add("location", axis).driver
        var = drv.variables.new()
        var.name = "src"
        var.type = "TRANSFORMS"
        var.targets[0].id = target_obj
        var.targets[0].transform_type = ("LOC_X", "LOC_Y")[axis]
        drv.expression = "src"
    mat = bpy.data.materials.new(disc.name + "_mat")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (0.0, 0.0, 0.0, 1.0)
    bsdf.inputs["Alpha"].default_value = 0.35
    mat.blend_method = "BLEND"
    disc.data.materials.append(mat)
    return disc
```

Apply: convert all player/ball materials through `_toon_material` (environment keeps flat emission/diffuse — grass bands already read as toon); `_add_outline` on player body meshes and the ball with `style["outline_width_m"]` and `palette.outline`; `_add_blob_shadow` per player armature (radius 0.4) and ball (radius 0.15). Print `TOON_MATERIALS <n>` / `OUTLINES <n>`. Guard every node/enum name behind the Blender version assert from Task 5; when `ShaderNodeShaderToRGB` is missing (Cycles-only builds) fall back to plain flat Emission and print `TOON_FALLBACK_FLAT`.

- [ ] **Step 3: Run tests**

Run: `.venv311/bin/python -m pytest tests/test_blender_render_scene.py -q`
Expected: PASS. Then render a real 50-frame gberch excerpt for eyeballing:

```bash
blender --background --python scripts/blender_render_scene.py -- \
  --output-dir output --shot "" --cameras broadcast \
  --width 960 --height 540 --samples 8 --frame-start 200 --frame-end 250 \
  --style-json '{"ramp_steps":3,"outline_width_m":0.02,"teams":{}}'
open output/render/clip/broadcast.mp4
```

(Use the real gberch shot id from `output/shots/shots_manifest.json` instead of `""` if the manifest exists.) Record s/frame — the spec's 2–10 s/frame @1080p planning number scales to ≲2.5 s/frame at 540p; investigate before proceeding if far off.

- [ ] **Step 4: Commit**

```bash
git add scripts/blender_render_scene.py tests/test_blender_render_scene.py
git commit -m "feat: toon look — cel ramps, inverted-hull outlines, blob shadows"
```

---

### Task 8: Virtual cameras wiring + 9:16 variants

**Files:**
- Modify: `src/stages/render.py`, `scripts/blender_render_scene.py`
- Test: `tests/test_render_stage.py` (extend), `tests/test_blender_render_scene.py` (extend)

**Interfaces:**
- Consumes: `src.stages.export._per_shot_smpl_tracks(output_dir, shot_id) -> list[SmplWorldTrack]` (module-level, `src/stages/export.py:116`); `vcam.build_pov_track / build_ots_track / build_drone_track`; `ExportStage._virtual_camera_cfg` recipe (`src/stages/export.py:233` — copy the RigConfig construction, don't instantiate ExportStage).
- Produces: stage method `_write_virtual_camera_tracks(shot_id: str, camera_ids: list[str]) -> list[str]` — writes `output/render/<shot|clip>/cameras/<safe_id>_camera_track.json` (CameraTrack serialised with the same field names as `camera/camera_track.json`; `safe_id` replaces `:` with `_`, e.g. `pov_P001`) and returns the ids it could satisfy. Script camera-id → filename mapping matches (`cam_id.replace(":", "_")`). Ball track loading for OTS: the same JSON the ball stage wrote (pass the parsed dict; `_ball_xyz_by_frame` handles shape).
- 9:16: script `--vertical` renders every non-broadcast camera a second time at `(height, width)` swapped resolution to `<camera>_9x16.mp4`, multiplying `cam.lens` by 0.8 for the portrait pass (wider reframe).

- [ ] **Step 1: Write failing stage test**

```python
@pytest.mark.integration
def test_virtual_camera_tracks_written(tmp_path):
    # fixture: shots manifest with shot01 + refined npz fixture from
    # tests/test_blender_render_scene.py (import the helper) + ball track
    _write_min_fixture(tmp_path)          # reuse via a conftest helper
    _add_player_fixture(tmp_path)
    cfg = _cfg(cameras=["broadcast", "drone", "pov:P001"])
    stage = RenderStage(cfg, tmp_path)
    written = stage._write_virtual_camera_tracks("", ["drone", "pov:P001"])
    assert set(written) == {"drone", "pov:P001"}
    cams = tmp_path / "render" / "clip" / "cameras"
    assert (cams / "drone_camera_track.json").exists()
    assert (cams / "pov_P001_camera_track.json").exists()
    track = json.loads((cams / "drone_camera_track.json").read_text())
    assert track["frames"] and "R" in track["frames"][0]


@pytest.mark.unit
def test_unknown_player_camera_skipped_with_warning(tmp_path, caplog):
    _write_min_fixture(tmp_path)
    stage = RenderStage(_cfg(), tmp_path)
    written = stage._write_virtual_camera_tracks("", ["pov:P999"])
    assert written == []
    assert any("P999" in r.message for r in caplog.records)
```

Move `_write_min_fixture` / `_add_player_fixture` into `tests/conftest.py` (or a `tests/_render_fixtures.py` imported by both test files) when reusing across files.

- [ ] **Step 2: Run to verify failure**

Run: `.venv311/bin/python -m pytest tests/test_render_stage.py -q -k virtual`
Expected: FAIL — `_write_virtual_camera_tracks` missing.

- [ ] **Step 3: Implement**

Stage: build `RigConfig` from `self.config["export"]["virtual_cameras"]` (copy the `_virtual_camera_cfg` construction from `src/stages/export.py:233`), load tracks via `_per_shot_smpl_tracks`, parse the shot's ball track JSON when present. For each requested camera id: `drone` → `build_drone_track(tracks, ball, ...)`; `pov:<pid>`/`ots:<pid>` → find the matching track by `player_id` (warn + skip when absent) → `build_pov_track` / `build_ots_track`. Serialise `CameraTrack` to JSON with exactly the on-disk `camera_track.json` field names — read one real file and copy the serialisation the camera stage uses (grep `camera_track.json` writers under `src/stages/camera.py`) rather than inventing one; `image_size` = render resolution; `fps` from the broadcast track. Call `_write_virtual_camera_tracks` inside `run()` before `subprocess.run`, and pass only the satisfied ids in `--cameras`.

Script: `--vertical` handling in the render loop:

```python
for cam_id, cam_obj, fps, frame_range in cameras:
    _render(cam_obj, out_dir / f"{safe(cam_id)}.mp4", fps, frame_range,
            args.width, args.height, args.samples)
    if args.vertical and cam_id != "broadcast":
        cam_obj.data.lens = cam_obj.data.lens * 0.8
        _render(cam_obj, out_dir / f"{safe(cam_id)}_9x16.mp4", fps,
                frame_range, args.height, args.width, args.samples)
        cam_obj.data.lens = cam_obj.data.lens / 0.8
```

(For keyed lenses, scale via a second scene-level trick instead: set `scene.render.resolution_x/y` swapped and `cam_obj.data.sensor_fit = "VERTICAL"` for the portrait pass — this reframes without touching keyed lens values; prefer this and drop the 0.8 factor if the framing looks acceptable in the eyeball check.)

Extend the fbx-marked smoke test: run with `--cameras broadcast,drone --vertical` on the player fixture (stage-level test writes the drone track first, or generate it in the fixture by calling `build_drone_track` directly and dumping JSON) and assert `drone.mp4` and `drone_9x16.mp4` exist.

- [ ] **Step 4: Run tests**

Run: `.venv311/bin/python -m pytest tests/test_render_stage.py tests/test_blender_render_scene.py tests/test_virtual_cameras_drone.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/stages/render.py scripts/blender_render_scene.py tests/
git commit -m "feat: render virtual cameras (drone/pov/ots) and 9:16 variants"
```

---

### Task 9: AOV passes, quality report, docs

**Files:**
- Modify: `scripts/blender_render_scene.py`, `src/pipeline/quality_report.py`, `CLAUDE.md`
- Test: `tests/test_blender_render_scene.py`, new `tests/test_quality_report_render.py`

**Interfaces:**
- Consumes: `write_quality_report(output_dir)` structure (`src/pipeline/quality_report.py:174` — read how `_ball_section` is registered and mirror it); `render/render_timings.json` written by the stage (Task 2).
- Produces: script `--aov` → `output/render/<shot>/aov/<camera>/####.exr` multilayer (Z + Normal + CryptoObject); `_render_section(output_dir) -> dict | None` in quality_report.

- [ ] **Step 1: Failing quality-report test**

```python
# tests/test_quality_report_render.py
import json
import pytest

from src.pipeline.quality_report import _render_section


@pytest.mark.unit
def test_render_section_lists_outputs(tmp_path):
    d = tmp_path / "render" / "shot01"
    d.mkdir(parents=True)
    (d / "broadcast.mp4").write_bytes(b"x" * 1000)
    (d / "drone.mp4").write_bytes(b"x" * 2000)
    (tmp_path / "render" / "render_timings.json").write_text(
        json.dumps({"shot01": 42.5}))
    section = _render_section(tmp_path)
    assert section["shots"]["shot01"]["cameras"] == ["broadcast", "drone"]
    assert section["shots"]["shot01"]["render_seconds"] == 42.5


@pytest.mark.unit
def test_render_section_none_when_absent(tmp_path):
    assert _render_section(tmp_path) is None
```

Run: `.venv311/bin/python -m pytest tests/test_quality_report_render.py -q` → FAIL.

- [ ] **Step 2: Implement**

`_render_section`: return `None` when `output/render` missing; else walk shot dirs, list `*.mp4` stems sorted, attach `render_seconds` from `render_timings.json` (key fallback: `"clip"`), and byte sizes. Register it in `write_quality_report` exactly the way `_ball_section` is registered (same None-skip idiom).

AOV in the script (`--aov`): on the scene's view layer set `use_pass_z = True`, `use_pass_normal = True`, `use_pass_cryptomatte_object = True`; add a File Output compositor node targeting `render/<shot>/aov/<camera>/` with `format.file_format = "OPEN_EXR_MULTILAYER"`, linked from RenderLayers Z/Normal/Image; enable `scene.use_nodes = True`. Extend the fbx-marked smoke test: run 1 frame with `--aov` and assert at least one `.exr` exists under `aov/broadcast/`.

CLAUDE.md updates (single edit):
- Pipeline table: add row `| 8 | render | refined_poses + ball + camera | render/<shot>/<camera>.mp4 (+ 9:16, AOV EXRs) |`.
- Commands block: `python recon.py run --input clip.mp4 --output ./output/ --stages render` with a one-line description.
- Configuration section: 3-line summary of `render.*` keys (cameras, style tokens, aov_passes) and the drone params under `export.virtual_cameras`.
- External Dependencies: note Blender now also serves the render stage (still optional; stage skips without it).

- [ ] **Step 3: Run tests**

Run: `.venv311/bin/python -m pytest tests/test_quality_report_render.py tests/test_blender_render_scene.py -q`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add src/pipeline/quality_report.py scripts/blender_render_scene.py CLAUDE.md tests/
git commit -m "feat: render AOV passes, quality-report section, docs"
```

---

### Task 10: gberch end-to-end validation (manual gate)

**Files:** none created — verification checklist. Output artifacts land in `output/render/`.

**Interfaces:** consumes the full stage; produces the go/no-go evidence for the spec's "first publishable clip" milestone.

- [ ] **Step 1: Full default suite**

Run: `.venv311/bin/python -m pytest tests/ -q`
Expected: green except the two pre-existing known failures documented in CLAUDE.md (`test_aerial_arc_promotes_grounded_run_to_flight`, `test_player_fbx_has_24_bones_and_full_keyframes`). Any OTHER failure blocks sign-off.

- [ ] **Step 2: Real-clip render**

```bash
.venv311/bin/python recon.py run --input test-media/<gberch-clip> \
  --output ./output/ --stages render
ls -la output/render/*/
open output/render/*/broadcast.mp4 output/render/*/drone.mp4
```

(Confirm the exact gberch input filename from `output/shots/shots_manifest.json` `source_file` before running.) Verify by eyeball: players upright (root-orient convention respected), on the pitch, kit colors split into two teams, ball tracks the play, drone framing keeps the action centred, no z-fighting on lines, outlines read at 1080p.

- [ ] **Step 3: Record the numbers**

Append to `output/CROSS_CLIP_EVALUATION.md`: render date, s/frame at 1080p for broadcast + drone from `render/render_timings.json`, file sizes, and any visual defects found (as the punch list for look-dev iteration).

- [ ] **Step 4: Commit any fixes made during validation**

```bash
git add -A && git commit -m "fix: render stage issues found in gberch e2e validation"
```

(Skip if nothing changed.)

---

## Self-Review Notes

- **Spec coverage:** §2 stage contract → Tasks 2, 8; §3 readers → Task 1; §4 scene (pitch/stadium/players/kits/ball/toon/lighting) → Tasks 5–7; §5 cameras incl. drone + 9:16 → Tasks 3, 8; §6 config incl. `aov_passes` → Tasks 2, 9; §7 testing incl. perf gate → Tasks 5 (gate step), 10; §8 look-dev split (`save_blend`) → Tasks 2, 5; §9 out-of-scope honored (no crowd/cloth/overlay tasks); §10 risks: version assert (Task 5), SMPL fallback (Task 6), outline width tunable (Task 7).
- **Known deviation from spec §3:** `load_smpl_body_data` also moves to `blender_scene_io` (it is bpy-free and both scripts need it) — within the spirit of "readers move".
- **Fixture-vs-reality checks are explicit steps** (ball-track field names, refined-NPZ keys, camera-track serialisation, manifest status fields) because the plan author verified structure but not every field name; the executor must mirror real files, not the plan's guesses.
- **Type consistency check:** `RenderStage._blender_args` / script `_parse_args` flag names match (`--output-dir --shot --cameras --width --height --samples --style-json --vertical --aov --save-blend --frame-start --frame-end`); `build_drone_track` signature matches its Task 8 call site; `kit_zone_for_height_fraction` zone strings match the material-slot names in Task 6; camera-id sanitisation (`:` → `_`) matches between stage writer and script reader.
