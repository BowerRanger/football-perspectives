"""Tests for scripts/blender_render_scene.py.

Split into pure arg-parsing tests (run everywhere) and a Blender smoke
test (``@pytest.mark.fbx`` — skipped when Blender isn't on PATH, same
posture as tests/test_blender_export_smpl_skeleton.py).
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess

import numpy as np
import pytest

from scripts.blender_render_scene import _parse_args
from src.stages.render import RenderStage
from tests.conftest import _add_player_fixture, _write_min_fixture


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


@pytest.mark.unit
def test_parse_args_defaults_and_flags():
    ns = _parse_args([
        "--output-dir", "/tmp/o",
        "--vertical", "--aov", "--save-blend",
        "--frame-start", "10", "--frame-end", "20",
    ])
    # No --shot / --cameras / --width / --height / --samples / --style-json
    # supplied: defaults from the brief's skeleton.
    assert ns.shot == ""
    assert ns.cameras == ["broadcast"]
    assert (ns.width, ns.height) == (1920, 1080)
    assert ns.samples == 16
    assert ns.style_json == "{}"
    assert ns.vertical is True and ns.aov is True and ns.save_blend is True
    assert (ns.frame_start, ns.frame_end) == (10, 20)


def _add_hostile_pose_fixture(root, n=1):
    """Regression fixture for the CLAUDE.md-documented pitfall: 'thetas[0]
    is IGNORED; root_R carries root world orientation. Applying both flips
    the body upside down.' An all-zero-thetas fixture (as in
    ``_add_player_fixture``) can't distinguish correctly-ignoring
    ``thetas[0]`` from wrongly-applying it — both produce an identity
    pelvis rotation. This fixture is deliberately hostile: a large
    non-zero pelvis ``thetas[0]`` (must be ignored) plus a realistic
    90-degrees-about-X ``root_R`` (canonical Y-up -> pitch Z-up, matching
    the real pipeline's convention) so a regression is directly
    observable in the rendered scene — see
    ``test_player_pelvis_ignores_thetas0_and_stays_upright``.
    """
    (root / "refined_poses").mkdir()
    theta = np.pi / 2
    root_R_x90 = np.array([
        [1.0, 0.0, 0.0],
        [0.0, np.cos(theta), -np.sin(theta)],
        [0.0, np.sin(theta), np.cos(theta)],
    ], dtype=np.float32)
    thetas = np.zeros((n, 24, 3), dtype=np.float32)
    thetas[:, 0] = [0.3, 1.2, -0.7]  # non-trivial; must be IGNORED
    np.savez(root / "refined_poses" / "P001_refined.npz",
             player_id="P001",
             frames=np.arange(n),
             betas=np.zeros(10, dtype=np.float32),
             thetas=thetas,
             root_R=np.tile(root_R_x90, (n, 1, 1)),
             root_t=np.tile(np.array([52.5, 30.0, 0.95], dtype=np.float32),
                            (n, 1)),
             confidence=np.ones(n, dtype=np.float32),
             view_count=np.ones(n, dtype=np.int32),
             contributing_shots=np.array([], dtype="<U6"))


# Run inside a SECOND headless Blender process against the saved
# scene.blend (opened directly, not via `--python <script> --`) to dump
# the pelvis pose-bone rotation and the evaluated body mesh's world-space
# bounding box as one JSON line prefixed with a marker so it's easy to
# find among Blender's own log noise. Matches "P001_" for both body
# types (SMPL mesh `P001_body` or capsule-fallback `P001_<bone>_capsule`)
# so this works regardless of whether data/models/smpl_neutral.npz is
# present on the machine running the suite.
_INSPECT_BLEND_SCRIPT = """
import bpy
import json

arm = bpy.data.objects["P001_arm"]
pelvis = arm.pose.bones["pelvis"]

depsgraph = bpy.context.evaluated_depsgraph_get()
world_min = [1e9, 1e9, 1e9]
world_max = [-1e9, -1e9, -1e9]
for obj in bpy.data.objects:
    if obj.type != "MESH" or not obj.name.startswith("P001_"):
        continue
    obj_eval = obj.evaluated_get(depsgraph)
    mesh_eval = obj_eval.to_mesh()
    mw = obj_eval.matrix_world
    for v in mesh_eval.vertices:
        wp = mw @ v.co
        for i in range(3):
            world_min[i] = min(world_min[i], wp[i])
            world_max[i] = max(world_max[i], wp[i])
    obj_eval.to_mesh_clear()

result = {
    "pelvis_quat": [pelvis.rotation_quaternion[i] for i in range(4)],
    "world_min": world_min,
    "world_max": world_max,
}
print("INSPECT_JSON " + json.dumps(result))
"""


_BLENDER = shutil.which("blender")


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
    assert "RENDER_TIMING broadcast" in res.stdout


@pytest.mark.fbx
@pytest.mark.skipif(_BLENDER is None, reason="blender not on PATH")
def test_smoke_render_aov_writes_exr(tmp_path):
    """Task 9: --aov wires a multilayer-EXR compositor File Output node
    alongside the normal mp4 render. One frame is enough to prove the
    wiring works — this isn't validating render content, just that the
    per-camera ``aov/<safe_camera>/####.exr`` path gets populated.
    """
    _write_min_fixture(tmp_path)
    script = "scripts/blender_render_scene.py"
    res = subprocess.run(
        [_BLENDER, "--background", "--python", script, "--",
         "--output-dir", str(tmp_path), "--shot", "",
         "--cameras", "broadcast", "--width", "160", "--height", "90",
         "--samples", "1", "--style-json", "{}",
         "--frame-start", "0", "--frame-end", "0", "--aov"],
        capture_output=True, text=True, timeout=600)
    assert res.returncode == 0, res.stderr[-3000:]
    out = tmp_path / "render" / "clip" / "broadcast.mp4"
    assert out.exists() and out.stat().st_size > 0

    aov_dir = tmp_path / "render" / "clip" / "aov" / "broadcast"
    exrs = list(aov_dir.glob("*.exr"))
    assert len(exrs) >= 1, res.stdout[-3000:]
    assert exrs[0].stat().st_size > 0


@pytest.mark.fbx
@pytest.mark.skipif(_BLENDER is None, reason="blender not on PATH")
def test_smoke_render_with_player(tmp_path):
    """One player fixture on top of the min fixture; 1 frame at 160x90.

    Asserts the PLAYERS_BUILT marker + render success — never body type
    (SMPL-mesh vs capsule fallback) since that depends on whether
    data/models/smpl_neutral.npz happens to exist on the machine running
    the suite (it's gitignored; absent by default in a fresh worktree).

    Also exercises the Task 7 toon look (cel-ramp materials + inverted-
    hull outlines) via a non-default ``--style-json``: the script must
    print ``TOON_MATERIALS <n>`` (n >= 4 — the 4 kit zones for one
    player, socks/skin/shorts/shirt, plus the ball) and ``OUTLINES <n>``
    (n >= 2 — at least one player body part + the ball) after building
    the scene, regardless of which body-fallback path ran.
    """
    _write_min_fixture(tmp_path)
    _add_player_fixture(tmp_path)
    script = "scripts/blender_render_scene.py"
    res = subprocess.run(
        [_BLENDER, "--background", "--python", script, "--",
         "--output-dir", str(tmp_path), "--shot", "",
         "--cameras", "broadcast", "--width", "160", "--height", "90",
         "--samples", "1",
         "--style-json", '{"ramp_steps": 3, "outline_width_m": 0.03}',
         "--frame-start", "0", "--frame-end", "0"],
        capture_output=True, text=True, timeout=600)
    assert res.returncode == 0, res.stderr[-3000:]
    out = tmp_path / "render" / "clip" / "broadcast.mp4"
    assert out.exists() and out.stat().st_size > 0
    assert "PLAYERS_BUILT 1" in res.stdout

    toon_m = re.search(r"^TOON_MATERIALS (\d+)$", res.stdout, re.MULTILINE)
    assert toon_m is not None, res.stdout[-3000:]
    assert int(toon_m.group(1)) >= 4

    outlines_m = re.search(r"^OUTLINES (\d+)$", res.stdout, re.MULTILINE)
    assert outlines_m is not None, res.stdout[-3000:]
    assert int(outlines_m.group(1)) >= 2


@pytest.mark.fbx
@pytest.mark.skipif(_BLENDER is None, reason="blender not on PATH")
def test_player_pelvis_ignores_thetas0_and_stays_upright(tmp_path):
    """Regression test for the thetas[0]/root_R convention (CLAUDE.md:
    'thetas[0] is IGNORED; root_R carries root world orientation.
    Applying both flips the body upside down').

    Renders the hostile fixture with ``--save-blend``, then shells a
    SECOND headless Blender process to open the saved scene and dump the
    pelvis pose-bone rotation plus the body's evaluated world bounding
    box. Two assertions pin down the convention directly rather than by
    proxy:

    1. The pelvis pose-bone rotation stays identity DESPITE the fixture's
       non-zero ``thetas[0]`` — proves ``thetas[0]`` is never written to
       the pelvis bone.
    2. The body's world-space Z extent (height) is clearly larger than
       its Y extent — proves ``root_R``'s 90-about-X reorientation
       (canonical Y-up -> pitch Z-up) actually took effect. X isn't
       checked: it's the rotation axis itself, so the T-pose arm span
       dominating X is invariant to this bug and not informative here.
       If ``thetas[0]`` leaked onto the pelvis (root of the bone
       hierarchy), the extra rotation would perturb this Y/Z relationship
       — an all-zero-thetas fixture can't distinguish the two cases, only
       this hostile one can.
    """
    _write_min_fixture(tmp_path)
    _add_hostile_pose_fixture(tmp_path)
    script = "scripts/blender_render_scene.py"
    res = subprocess.run(
        [_BLENDER, "--background", "--python", script, "--",
         "--output-dir", str(tmp_path), "--shot", "",
         "--cameras", "broadcast", "--width", "160", "--height", "90",
         "--samples", "1", "--style-json", "{}",
         "--frame-start", "0", "--frame-end", "0", "--save-blend"],
        capture_output=True, text=True, timeout=600)
    assert res.returncode == 0, res.stderr[-3000:]
    assert "PLAYERS_BUILT 1" in res.stdout

    blend_path = tmp_path / "render" / "clip" / "scene.blend"
    assert blend_path.exists()

    inspect_script = tmp_path / "inspect_blend.py"
    inspect_script.write_text(_INSPECT_BLEND_SCRIPT)
    res2 = subprocess.run(
        [_BLENDER, "--background", str(blend_path), "--python", str(inspect_script)],
        capture_output=True, text=True, timeout=600)
    assert res2.returncode == 0, res2.stderr[-3000:]
    line = next(
        (ln for ln in res2.stdout.splitlines() if ln.startswith("INSPECT_JSON ")),
        None)
    assert line is not None, res2.stdout[-3000:]
    data = json.loads(line[len("INSPECT_JSON "):])

    pelvis_quat = data["pelvis_quat"]
    assert pelvis_quat == pytest.approx([1.0, 0.0, 0.0, 0.0], abs=1e-4), (
        f"pelvis pose-bone rotation must stay identity despite the "
        f"fixture's non-zero thetas[0]; got {pelvis_quat}"
    )

    world_min, world_max = data["world_min"], data["world_max"]
    z_extent = world_max[2] - world_min[2]
    y_extent = world_max[1] - world_min[1]
    assert z_extent > 2 * y_extent, (
        f"body must stand upright (tall along world Z, not Y) — "
        f"z_extent={z_extent:.3f} y_extent={y_extent:.3f}"
    )


@pytest.mark.fbx
@pytest.mark.skipif(_BLENDER is None, reason="blender not on PATH")
def test_smoke_render_drone_and_vertical(tmp_path):
    """Virtual-camera wiring + 9:16 variants (Task 8) smoke test.

    The drone CameraTrack is generated via
    ``RenderStage._write_virtual_camera_tracks`` — the real writer the
    render stage uses in ``run()`` — rather than hand-dumping JSON, so
    this exercises the actual serialisation path end to end. Renders
    ``--cameras broadcast,drone --vertical``: broadcast (the id excluded
    from the vertical pass) must never get a 9x16 variant, while drone
    (non-broadcast) gets both the landscape and portrait renders.
    """
    _write_min_fixture(tmp_path)
    _add_player_fixture(tmp_path)
    cfg = {
        "render": {"resolution": [160, 90]},
        "export": {"virtual_cameras": {}},
    }
    stage = RenderStage(cfg, tmp_path)
    written = stage._write_virtual_camera_tracks("", ["drone"])
    assert written == ["drone"]

    script = "scripts/blender_render_scene.py"
    res = subprocess.run(
        [_BLENDER, "--background", "--python", script, "--",
         "--output-dir", str(tmp_path), "--shot", "",
         "--cameras", "broadcast,drone", "--width", "160", "--height", "90",
         "--samples", "1", "--style-json", "{}", "--vertical",
         "--frame-start", "0", "--frame-end", "0"],
        capture_output=True, text=True, timeout=600)
    assert res.returncode == 0, res.stderr[-3000:]

    render_dir = tmp_path / "render" / "clip"
    assert (render_dir / "broadcast.mp4").exists()
    assert (render_dir / "drone.mp4").exists()
    assert (render_dir / "drone_9x16.mp4").exists()
    assert not (render_dir / "broadcast_9x16.mp4").exists()


@pytest.mark.fbx
@pytest.mark.skipif(_BLENDER is None, reason="blender not on PATH")
def test_smoke_render_vertical_and_aov_together(tmp_path):
    """Regression test: --vertical --aov together with a non-broadcast
    camera used to leave a corrupt extension-less EXR-named file under
    aov/<camera>/.

    ``scene.compositing_node_group`` stays attached to the scene once
    ``_setup_aov_compositor`` assigns it, and Blender keeps running the
    compositor on every later render call — including drone's 9:16
    vertical pass, which never receives an ``aov_dir`` and so never runs
    the rename-to-``.exr`` step. Without gating ``use_compositing`` per
    call, that vertical pass would still fire the File Output node at
    its stale directory/frame, dropping a wrong-resolution file named
    plain ``0000`` (no extension) next to the real ``0000.exr``.

    Asserts every file under ``aov/<camera>/`` (for every requested
    camera, not just broadcast) carries the ``.exr`` extension, and that
    the drone 9:16 mp4 still renders normally alongside it.
    """
    _write_min_fixture(tmp_path)
    _add_player_fixture(tmp_path)
    cfg = {
        "render": {"resolution": [160, 90]},
        "export": {"virtual_cameras": {}},
    }
    stage = RenderStage(cfg, tmp_path)
    written = stage._write_virtual_camera_tracks("", ["drone"])
    assert written == ["drone"]

    script = "scripts/blender_render_scene.py"
    res = subprocess.run(
        [_BLENDER, "--background", "--python", script, "--",
         "--output-dir", str(tmp_path), "--shot", "",
         "--cameras", "broadcast,drone", "--width", "160", "--height", "90",
         "--samples", "1", "--style-json", "{}", "--vertical", "--aov",
         "--frame-start", "0", "--frame-end", "0"],
        capture_output=True, text=True, timeout=600)
    assert res.returncode == 0, res.stderr[-3000:]

    render_dir = tmp_path / "render" / "clip"
    assert (render_dir / "drone_9x16.mp4").exists()
    assert (render_dir / "drone_9x16.mp4").stat().st_size > 0

    for cam in ("broadcast", "drone"):
        aov_dir = render_dir / "aov" / cam
        files = sorted(aov_dir.iterdir())
        assert files, f"no AOV output for {cam}: {res.stdout[-3000:]}"
        for f in files:
            assert f.suffix == ".exr", (
                f"non-.exr file leaked into {aov_dir}: {[p.name for p in files]}"
            )
