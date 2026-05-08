"""Integration test: SMPL skeleton bake through Blender FBX export.

Marked ``fbx`` — requires Blender on PATH. Uses two Blender invocations:
one to run the exporter on a synthetic SmplWorldTrack, a second to
reopen the FBX and emit a JSON snapshot we can assert against.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from src.schemas.smpl_world import SmplWorldTrack
from src.utils.smpl_skeleton import SMPL_JOINT_NAMES, SMPL_PARENTS


pytestmark = pytest.mark.fbx


REOPEN_SCRIPT = """
import bpy, json, sys
from pathlib import Path
fbx_path = sys.argv[sys.argv.index('--') + 1]
out_path = sys.argv[sys.argv.index('--') + 2]
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.fbx(filepath=fbx_path)
arm = next(o for o in bpy.data.objects if o.type == 'ARMATURE')
bones = [
    {'name': b.name, 'parent': b.parent.name if b.parent else None}
    for b in arm.data.bones
]
fcurves = []
if arm.animation_data and arm.animation_data.action:
    fcurves = arm.animation_data.action.fcurves
n_kfs = max((len(fc.keyframe_points) for fc in fcurves), default=0)
out = {
    'bones': bones,
    'fcurve_count': len(fcurves),
    'max_keyframes': n_kfs,
}
Path(out_path).write_text(json.dumps(out))
"""


def _have_blender() -> bool:
    return shutil.which("blender") is not None


@pytest.mark.skipif(not _have_blender(), reason="Blender not on PATH")
def test_player_fbx_has_24_bones_and_full_keyframes(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    hmr_dir = output_dir / "hmr_world"
    hmr_dir.mkdir(parents=True)
    cam_dir = output_dir / "camera"
    cam_dir.mkdir(parents=True)
    (cam_dir / "camera_track.json").write_text(
        json.dumps({"fps": 30.0, "frames": []})
    )

    n_frames = 5
    track = SmplWorldTrack(
        player_id="P001",
        frames=np.arange(n_frames, dtype=np.int64),
        betas=np.zeros(10, dtype=np.float64),
        thetas=np.zeros((n_frames, 24, 3), dtype=np.float64),
        root_R=np.tile(np.eye(3), (n_frames, 1, 1)),
        root_t=np.array(
            [[float(i), 0.0, 0.0] for i in range(n_frames)],
            dtype=np.float64,
        ),
        confidence=np.ones(n_frames, dtype=np.float64),
    )
    track.save(hmr_dir / "P001_smpl_world.npz")

    repo = Path(__file__).resolve().parents[1]
    subprocess.run(
        [
            "blender",
            "--background",
            "--python",
            str(repo / "scripts" / "blender_export_fbx.py"),
            "--",
            "--output-dir",
            str(output_dir),
        ],
        check=True,
        capture_output=True,
    )

    fbx_path = output_dir / "export" / "fbx" / "P001.fbx"
    assert fbx_path.exists(), "player FBX not written"

    snapshot_path = tmp_path / "snapshot.json"
    reopen_script = tmp_path / "reopen.py"
    reopen_script.write_text(REOPEN_SCRIPT)
    subprocess.run(
        [
            "blender",
            "--background",
            "--python",
            str(reopen_script),
            "--",
            str(fbx_path),
            str(snapshot_path),
        ],
        check=True,
        capture_output=True,
    )

    snap = json.loads(snapshot_path.read_text())
    bone_names = {b["name"] for b in snap["bones"]}
    for name in SMPL_JOINT_NAMES:
        assert name in bone_names, f"missing bone {name}"
    parent_map = {b["name"]: b["parent"] for b in snap["bones"]}
    for j, name in enumerate(SMPL_JOINT_NAMES):
        if SMPL_PARENTS[j] == -1:
            assert parent_map[name] is None
        else:
            assert parent_map[name] == SMPL_JOINT_NAMES[SMPL_PARENTS[j]]
    assert snap["max_keyframes"] >= n_frames
