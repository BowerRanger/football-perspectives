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
from pathlib import Path

import numpy as np
import pytest

from scripts.blender_render_scene import _DEFAULT_STYLE, _parse_args, _resolve_style
from src.stages.render import RenderStage
from src.utils.blender_scene_io import load_smpl_body_data
from src.utils.smpl_skeleton import SMPL_JOINT_NAMES, compute_all_joint_worlds
from tests.conftest import _add_player_fixture, _write_min_fixture

# Absolute path, matching tests/test_blender_export_smpl_skeleton.py's
# convention — a relative "scripts/blender_render_scene.py" only resolves
# when pytest's cwd happens to be the repo root, which isn't guaranteed.
_SCRIPT = str(Path(__file__).resolve().parents[1] / "scripts" / "blender_render_scene.py")


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
    assert ns.render_root == "render"


@pytest.mark.unit
def test_parse_args_render_root_override():
    ns = _parse_args([
        "--output-dir", "/tmp/o", "--render-root", "render_experiments/variant_a",
    ])
    assert ns.render_root == "render_experiments/variant_a"


# --- _resolve_style: new style keys (render-experiments task) -----------
# The frozen-interface guarantee: every new key absent from --style-json
# must reproduce byte-identical current behaviour. These pin the exact
# defaults _build_world/_build_environment/_render read when a key is
# missing, independent of whichever hardcoded constants the script uses
# internally (a future refactor could rename those constants; these
# tests only care about the values _resolve_style actually returns).

@pytest.mark.unit
def test_resolve_style_no_new_keys_reproduces_exact_prior_defaults():
    resolved = _resolve_style({})
    assert resolved["sun_energy"] == 3.0
    assert tuple(resolved["sun_rotation_deg"]) == (50.0, 0.0, -30.0)
    assert resolved["world_strength"] == 1.0
    assert resolved["lines_emission_strength"] == 1.0
    # Every post effect must resolve to its neutral/off value — an
    # empty style-json must never attach a post compositor.
    post = resolved["post"]
    assert post["glare"] == 0.0
    assert post["grain"] == 0.0
    assert post["vignette"] == 0.0
    assert post["posterize"] == 0
    assert post["saturation"] == 1.0
    assert post["duotone"] == {"shadow": None, "highlight": None}
    # Task 2/7 defaults (palette/ramp_steps/outline/stripes) must be
    # completely untouched by this task's additions.
    assert resolved["ramp_steps"] == _DEFAULT_STYLE["ramp_steps"]
    assert resolved["outline_width_m"] == _DEFAULT_STYLE["outline_width_m"]
    assert resolved["grass_stripes"] == _DEFAULT_STYLE["grass_stripes"]
    assert resolved["palette"] == _DEFAULT_STYLE["palette"]


@pytest.mark.unit
def test_resolve_style_top_level_new_keys_overridable_independently():
    resolved = _resolve_style({"sun_energy": 0.4, "world_strength": 0.1})
    assert resolved["sun_energy"] == 0.4
    assert resolved["world_strength"] == 0.1
    # Untouched siblings keep their defaults.
    assert tuple(resolved["sun_rotation_deg"]) == (50.0, 0.0, -30.0)
    assert resolved["lines_emission_strength"] == 1.0


@pytest.mark.unit
def test_resolve_style_post_partial_override_keeps_other_effects_off():
    resolved = _resolve_style({"post": {"vignette": 0.6}})
    post = resolved["post"]
    assert post["vignette"] == 0.6
    assert post["glare"] == 0.0
    assert post["grain"] == 0.0
    assert post["posterize"] == 0
    assert post["saturation"] == 1.0
    assert post["duotone"] == {"shadow": None, "highlight": None}


@pytest.mark.unit
def test_resolve_style_post_duotone_partial_override():
    resolved = _resolve_style(
        {"post": {"duotone": {"shadow": "#001122"}, "saturation": 0.0}})
    post = resolved["post"]
    assert post["duotone"] == {"shadow": "#001122", "highlight": None}
    assert post["saturation"] == 0.0
    # Sibling post effects still default to off.
    assert post["glare"] == 0.0


@pytest.mark.unit
def test_resolve_style_full_look_dict_round_trips():
    style = {
        "sun_energy": 0.3,
        "sun_rotation_deg": [20.0, 0.0, 10.0],
        "world_strength": 0.05,
        "lines_emission_strength": 4.0,
        "post": {
            "glare": 1.2,
            "grain": 0.25,
            "vignette": 0.4,
            "posterize": 6,
            "duotone": {"shadow": "#0a0a2a", "highlight": "#ffb347"},
            "saturation": 1.6,
        },
        "palette": {"grass_light": "#123456"},
    }
    resolved = _resolve_style(style)
    assert resolved["sun_energy"] == 0.3
    assert resolved["sun_rotation_deg"] == [20.0, 0.0, 10.0]
    assert resolved["world_strength"] == 0.05
    assert resolved["lines_emission_strength"] == 4.0
    assert resolved["post"] == style["post"]
    assert resolved["palette"]["grass_light"] == "#123456"
    # Untouched palette entries keep their defaults.
    assert resolved["palette"]["outline"] == _DEFAULT_STYLE["palette"]["outline"]


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


def _add_limb_pose_fixture(root, root_t=(52.5, 30.0, 0.95)):
    """Regression fixture for the bone-space FK bug (task brief): per-frame
    pose keying applies raw SMPL theta quaternions to
    ``pose.bones[...].rotation_quaternion``, which Blender interprets in
    the bone's LOCAL REST frame. SMPL thetas are only directly applicable
    there when bone-local == canonical (uniform +Y tail, zero roll, as in
    ``blender_export_fbx.py``'s ``_build_smpl_armature``); the pre-fix
    code instead gave internal bones child-mean tails and leaf bones a +Z
    tail, so limb bones rotate about the WRONG axes.

    Non-zero, differing-axis ~90-degree bends on l_shoulder (16), r_elbow
    (19) and l_knee (4) — joints deep enough in the chain (r_elbow sits
    past two non-identity ancestor rotations once posed) that a wrong
    bone-local frame produces metre-scale world-position errors, while an
    all-zero-thetas fixture (``_add_player_fixture``) can't distinguish a
    correct FK from a buggy one (both give the T-pose). Identity root_R /
    root_t at a known pitch point isolates the bug to the per-bone pose
    application, independent of the thetas[0]/root_R convention already
    covered by ``_add_hostile_pose_fixture``.

    Returns ``(thetas, root_R, root_t)`` (frame 0 only) so the caller can
    feed the identical values into
    ``src.utils.smpl_skeleton.compute_all_joint_worlds`` as ground truth.
    """
    (root / "refined_poses").mkdir()
    thetas = np.zeros((1, 24, 3), dtype=np.float32)
    thetas[0, 16] = [np.pi / 2, 0.0, 0.0]   # l_shoulder: 90 deg about X
    thetas[0, 19] = [0.0, np.pi / 2, 0.0]   # r_elbow: 90 deg about Y
    thetas[0, 4] = [0.0, 0.0, np.pi / 2]    # l_knee: 90 deg about Z
    root_R = np.eye(3, dtype=np.float32)
    root_t_arr = np.array(root_t, dtype=np.float32)
    np.savez(root / "refined_poses" / "P001_refined.npz",
             player_id="P001",
             frames=np.arange(1),
             betas=np.zeros(10, dtype=np.float32),
             thetas=thetas,
             root_R=root_R[np.newaxis, ...],
             root_t=root_t_arr[np.newaxis, ...],
             confidence=np.ones(1, dtype=np.float32),
             view_count=np.ones(1, dtype=np.int32),
             contributing_shots=np.array([], dtype="<U6"))
    return thetas, root_R, root_t_arr


_PARITY_INSPECT_SCRIPT = """
import bpy
import json
from pathlib import Path

blend_dir = Path(bpy.data.filepath).parent
joint_names = json.loads((blend_dir / "joint_order.json").read_text())

arm = bpy.data.objects["P001_arm"]
bpy.context.scene.frame_set(0)
depsgraph = bpy.context.evaluated_depsgraph_get()
arm_eval = arm.evaluated_get(depsgraph)

heads = []
for jname in joint_names:
    pb = arm_eval.pose.bones[jname]
    world_head = arm_eval.matrix_world @ pb.head
    heads.append([world_head.x, world_head.y, world_head.z])

print("PARITY_JSON " + json.dumps({"heads": heads}))
"""


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
    script = _SCRIPT
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

    # Pixel-level smoke assertion (spec §7): a corrupt/black/blank/wrong
    # -size render must not slip through just because ffmpeg exited 0 and
    # wrote a nonzero-size file. ffprobe pins the container's declared
    # resolution to the exact --width/--height requested; a decoded frame's
    # channel means confirm it's neither black (grossly under-lit) nor
    # some other flat failure mode — the camera looks at the grass pitch
    # from above, so green should clearly dominate red and blue.
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "json", str(out)],
        capture_output=True, text=True, timeout=60)
    assert probe.returncode == 0, probe.stderr
    dims = json.loads(probe.stdout)["streams"][0]
    assert (dims["width"], dims["height"]) == (160, 90)

    frame_path = tmp_path / "frame0.png"
    extract = subprocess.run(
        ["ffmpeg", "-y", "-i", str(out), "-frames:v", "1", str(frame_path)],
        capture_output=True, text=True, timeout=60)
    assert extract.returncode == 0, extract.stderr[-2000:]

    from PIL import Image
    img = Image.open(frame_path).convert("RGB")
    arr = np.asarray(img, dtype=np.float64)
    r_mean = float(arr[..., 0].mean())
    g_mean = float(arr[..., 1].mean())
    b_mean = float(arr[..., 2].mean())
    luminance = 0.299 * r_mean + 0.587 * g_mean + 0.114 * b_mean
    assert luminance > 15.0, (
        f"frame looks black (mean luminance {luminance:.1f}); "
        f"r={r_mean:.1f} g={g_mean:.1f} b={b_mean:.1f}"
    )
    assert g_mean > r_mean and g_mean > b_mean, (
        f"expected a green-dominant grass-pitch frame, got "
        f"r={r_mean:.1f} g={g_mean:.1f} b={b_mean:.1f}"
    )


@pytest.mark.fbx
@pytest.mark.skipif(_BLENDER is None, reason="blender not on PATH")
def test_smoke_render_aov_writes_exr(tmp_path):
    """Task 9: --aov wires a multilayer-EXR compositor File Output node
    alongside the normal mp4 render. One frame is enough to prove the
    wiring works — this isn't validating render content, just that the
    per-camera ``aov/<safe_camera>/####.exr`` path gets populated.
    """
    _write_min_fixture(tmp_path)
    script = _SCRIPT
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
    script = _SCRIPT
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
    script = _SCRIPT
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
def test_player_limb_joints_match_pure_python_fk(tmp_path):
    """Ground-truth FK parity test for the bone-space bug.

    Builds a hostile-limb-pose fixture (see ``_add_limb_pose_fixture``),
    renders it with ``--save-blend``, then shells a SECOND headless
    Blender process to dump every pose-bone's evaluated world-space HEAD
    position from the saved .blend. Compares against the pure-python
    ground truth ``compute_all_joint_worlds`` — the same FK the export/
    glTF preview path uses — using whichever rest-joint table the render
    actually built the armature from (the real SMPL asset when
    ``data/models/smpl_neutral.npz`` is present on this machine, else the
    hand-typed fallback table; see ``load_smpl_body_data``), so the
    comparison is exact to float64 precision regardless of which body
    path is active.

    Also exercises the SMPL-mesh skinning path when the asset is present
    (task brief item 5): the rendered frame must show non-trivial pixel
    variance, ruling out a degenerate/invisible skinned body as a silent
    regression alongside the armature fix. Skipped gracefully when the
    asset is absent (e.g. CI).
    """
    _write_min_fixture(tmp_path)
    thetas, root_R, root_t = _add_limb_pose_fixture(tmp_path)
    script = _SCRIPT
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
    (blend_path.parent / "joint_order.json").write_text(
        json.dumps(list(SMPL_JOINT_NAMES)))

    inspect_script = tmp_path / "inspect_parity.py"
    inspect_script.write_text(_PARITY_INSPECT_SCRIPT)
    res2 = subprocess.run(
        [_BLENDER, "--background", str(blend_path), "--python", str(inspect_script)],
        capture_output=True, text=True, timeout=600)
    assert res2.returncode == 0, res2.stderr[-3000:]
    line = next(
        (ln for ln in res2.stdout.splitlines() if ln.startswith("PARITY_JSON ")),
        None)
    assert line is not None, res2.stdout[-3000:]
    actual = np.asarray(json.loads(line[len("PARITY_JSON "):])["heads"])

    repo_root = Path(__file__).resolve().parents[1]
    smpl_data, pelvis_canon = load_smpl_body_data(repo_root, np)
    # compute_all_joint_worlds treats rest_joints[0] as the canonical
    # origin (world_pos[0] = root_R @ rest[0] + root_t); the render
    # script instead re-anchors the armature object so the pelvis BONE's
    # (possibly off-origin, asset-derived) rest head lands exactly at
    # root_t (see _build_players' `offset = R @ pelvis_canon` comment).
    # Re-centre the asset's rest table on its own pelvis row so both
    # sides agree on where "the origin" is — the FK recursion only ever
    # consumes *differences* between rows past the base case, so this
    # shift doesn't change anything else about the comparison.
    rest_joints = (
        np.asarray(smpl_data["joint_positions"], dtype=np.float64) - pelvis_canon
        if smpl_data is not None else None
    )
    expected = compute_all_joint_worlds(
        thetas[0], root_R, root_t, rest_joints=rest_joints)

    errors = np.linalg.norm(actual - expected, axis=1)
    max_error = float(errors.max())
    assert max_error < 1e-3, (
        f"max per-joint world-position error {max_error:.4f} m "
        f"(per-joint: {dict(zip(SMPL_JOINT_NAMES, errors.round(4).tolist()))})"
    )

    smpl_asset = repo_root / "data" / "models" / "smpl_neutral.npz"
    if smpl_asset.exists():
        out = tmp_path / "render" / "clip" / "broadcast.mp4"
        assert out.exists() and out.stat().st_size > 0
        frame_path = tmp_path / "parity_frame.png"
        extract = subprocess.run(
            ["ffmpeg", "-y", "-i", str(out), "-frames:v", "1", str(frame_path)],
            capture_output=True, text=True, timeout=60)
        assert extract.returncode == 0, extract.stderr[-2000:]
        from PIL import Image
        img = Image.open(frame_path).convert("RGB")
        arr = np.asarray(img, dtype=np.float64)
        assert arr.std() > 5.0, (
            f"rendered frame looks flat (std={arr.std():.2f}); SMPL mesh "
            "skinning may be degenerate"
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

    script = _SCRIPT
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

    script = _SCRIPT
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


def _render_one_frame(tmp_path: Path, style: dict, out_name: str,
                       extra_args: list[str] | None = None):
    """Shared helper for the style.post smoke tests below: renders one
    frame of the broadcast camera with the given style, extracts frame 0
    to a PNG via ffmpeg, and returns (subprocess result, PNG path)."""
    res = subprocess.run(
        [_BLENDER, "--background", "--python", _SCRIPT, "--",
         "--output-dir", str(tmp_path), "--shot", "",
         "--cameras", "broadcast", "--width", "160", "--height", "90",
         "--samples", "1", "--style-json", json.dumps(style),
         "--frame-start", "0", "--frame-end", "0", *(extra_args or [])],
        capture_output=True, text=True, timeout=600)
    out = tmp_path / "render" / "clip" / "broadcast.mp4"
    frame_path = tmp_path / out_name
    if res.returncode == 0 and out.exists():
        extract = subprocess.run(
            ["ffmpeg", "-y", "-i", str(out), "-frames:v", "1", str(frame_path)],
            capture_output=True, text=True, timeout=60)
        assert extract.returncode == 0, extract.stderr[-2000:]
    return res, frame_path


@pytest.mark.fbx
@pytest.mark.skipif(_BLENDER is None, reason="blender not on PATH")
def test_smoke_render_style_post_vignette_darkens_corners(tmp_path):
    """style.post.vignette (render-experiments task) must visibly darken
    the frame's corners relative to its centre in a REAL headless
    render — the compositor node-graph math (EllipseMask -> Blur ->
    RGBToBW -> Math -> MixRGB) was validated in isolation via a probe
    script before this task shipped; this is the integration check that
    it's wired correctly end to end through --style-json.
    """
    _write_min_fixture(tmp_path)

    res_base, base_png = _render_one_frame(tmp_path, {}, "base_frame.png")
    assert res_base.returncode == 0, res_base.stderr[-3000:]

    res_vig, vig_png = _render_one_frame(
        tmp_path, {"post": {"vignette": 1.0}}, "vig_frame.png")
    assert res_vig.returncode == 0, res_vig.stderr[-3000:]

    from PIL import Image
    base_arr = np.asarray(Image.open(base_png).convert("RGB"), dtype=np.float64)
    vig_arr = np.asarray(Image.open(vig_png).convert("RGB"), dtype=np.float64)

    base_corner = base_arr[:10, :10].mean()
    vig_corner = vig_arr[:10, :10].mean()
    base_center = base_arr[35:55, 65:95].mean()
    vig_center = vig_arr[35:55, 65:95].mean()

    assert vig_corner < base_corner - 10, (
        f"expected vignette to darken corners; base={base_corner:.1f} "
        f"vig={vig_corner:.1f}"
    )
    # Centre sits inside the EllipseMask's unblurred core (mask ~= 1
    # there), so it should stay close to the baseline's centre brightness.
    assert abs(vig_center - base_center) < base_center * 0.5 + 10, (
        f"expected centre roughly unaffected; base={base_center:.1f} "
        f"vig={vig_center:.1f}"
    )


@pytest.mark.fbx
@pytest.mark.skipif(_BLENDER is None, reason="blender not on PATH")
def test_smoke_render_style_post_all_effects_combined(tmp_path):
    """Every style.post effect active at once must not crash the
    compositor graph build. Each effect (saturation/duotone/posterize/
    glare/vignette/grain) was probe-validated in isolation before this
    task shipped; this is the integration check that chaining all six
    together in one graph — several reusing `ShaderNode*` node types
    inside the `CompositorNodeTree`, per the Blender 5.1.1 adaptation
    note in ``_setup_post_compositor`` — doesn't hit a socket/link
    conflict.
    """
    _write_min_fixture(tmp_path)
    style = {
        "post": {
            "glare": 1.0,
            "grain": 0.3,
            "vignette": 0.5,
            "posterize": 4,
            "duotone": {"shadow": "#101018", "highlight": "#ffcc66"},
            "saturation": 0.2,
        }
    }
    res, _ = _render_one_frame(tmp_path, style, "combined_frame.png")
    assert res.returncode == 0, res.stderr[-3000:]
    out = tmp_path / "render" / "clip" / "broadcast.mp4"
    assert out.exists() and out.stat().st_size > 0


@pytest.mark.fbx
@pytest.mark.skipif(_BLENDER is None, reason="blender not on PATH")
def test_smoke_render_aov_and_post_together_warns_and_prefers_aov(tmp_path):
    """--aov and a configured style.post are mutually exclusive (task
    decision, documented in ``_render``): AOV always wins, with a
    one-time warning printed to stdout, and the AOV EXRs still come out
    exactly as they would without any style.post configured.
    """
    _write_min_fixture(tmp_path)
    res, _ = _render_one_frame(
        tmp_path, {"post": {"vignette": 1.0}}, "aov_post_frame.png",
        extra_args=["--aov"])
    assert res.returncode == 0, res.stderr[-3000:]
    assert "style.post effects are configured but --aov was requested" in res.stdout

    out = tmp_path / "render" / "clip" / "broadcast.mp4"
    assert out.exists() and out.stat().st_size > 0
    aov_dir = tmp_path / "render" / "clip" / "aov" / "broadcast"
    exrs = list(aov_dir.glob("*.exr"))
    assert len(exrs) >= 1, res.stdout[-3000:]


@pytest.mark.fbx
@pytest.mark.skipif(_BLENDER is None, reason="blender not on PATH")
def test_render_root_redirects_output_and_camera_track_lookup(tmp_path):
    """--render-root substitutes the "render" path component for BOTH
    the render output dir and the non-broadcast camera-track lookup dir.

    RenderStage._write_virtual_camera_tracks (the real writer) always
    writes under the literal "render" root; a render-experiments matrix
    run instead pre-stages the track at the alternate root and expects
    this script to look for it there too. The drone track is MOVED (not
    copied) off the default "render" root before the render, so if
    --render-root failed to redirect the lookup the script would fail to
    find it and error out — a real regression guard, not just "the
    custom path also happens to work".
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

    default_track = tmp_path / "render" / "clip" / "cameras" / "drone_camera_track.json"
    assert default_track.exists()

    custom_root = "render_experiment_x"
    custom_track_dir = tmp_path / custom_root / "clip" / "cameras"
    custom_track_dir.mkdir(parents=True)
    default_track.replace(custom_track_dir / "drone_camera_track.json")

    res = subprocess.run(
        [_BLENDER, "--background", "--python", _SCRIPT, "--",
         "--output-dir", str(tmp_path), "--shot", "",
         "--cameras", "drone", "--width", "160", "--height", "90",
         "--samples", "1", "--style-json", "{}",
         "--frame-start", "0", "--frame-end", "0",
         "--render-root", custom_root],
        capture_output=True, text=True, timeout=600)
    assert res.returncode == 0, res.stderr[-3000:]

    out = tmp_path / custom_root / "clip" / "drone.mp4"
    assert out.exists() and out.stat().st_size > 0
    # The protected default-root baseline must never be touched.
    assert not (tmp_path / "render" / "clip" / "drone.mp4").exists()


@pytest.mark.fbx
@pytest.mark.skipif(_BLENDER is None, reason="blender not on PATH")
def test_saved_stadium_has_dressing_and_opens_with_requested_render_settings(tmp_path):
    _write_min_fixture(tmp_path)
    result, _ = _render_one_frame(tmp_path, {}, 'stadium.png',
                                  extra_args=['--save-blend'])
    assert result.returncode == 0, result.stderr[-3000:]
    probe = tmp_path / 'inspect_stadium.py'
    probe.write_text('''import bpy
scene = bpy.context.scene
assert scene.render.engine == 'BLENDER_EEVEE'
assert (scene.render.resolution_x, scene.render.resolution_y) == (160,90)
assert scene.frame_start == scene.frame_end == scene.frame_current == 0
assert scene.camera.name == 'broadcast'
for name in ['Stadium_Seats','Stadium_Roof','Stadium_Boards','Stadium_CrowdSkin']:
    obj = bpy.data.objects[name]
    assert len(obj.data.polygons) > 0, name
# Stand seating is outside the playing rectangle, including both end stands.
for vertex in bpy.data.objects['Stadium_Seats'].data.vertices:
    x,y,z = vertex.co
    assert x < 0 or x > 105 or y < 0 or y > 68
nets = [o for o in scene.objects if o.name.startswith('GoalNet')]
assert len(nets) == 2
assert all(len(o.data.splines) > 50 for o in nets)
assert bpy.data.objects.get('L_PenaltyArc_Left') is not None
assert bpy.data.objects.get('L_PenaltyArc_Right') is not None
''')
    checked = subprocess.run([_BLENDER, '--background',
        str(tmp_path / 'render/clip/scene.blend'), '--python-exit-code', '1',
        '--python', str(probe)], capture_output=True,text=True,timeout=120)
    assert checked.returncode == 0, checked.stdout[-3000:] + checked.stderr[-3000:]
