"""Blender headless FBX exporter for the broadcast-mono pipeline.

Invoked by ``ExportStage._export_fbx`` via::

    blender --background --python scripts/blender_export_fbx.py -- \
        --output-dir /path/to/output

Reads the same artefacts as ``ExportStage`` (camera_track, hmr_world NPZs,
ball_track) and emits ``output/export/fbx/PXXX.fbx`` (per player),
``ball.fbx`` and ``camera.fbx`` (UE5 conventions: scale 1.0 m, forward -Y,
up Z).

This v1 script is intentionally minimal: it produces single-bone
armatures animated at root_t/root_R per player (no full SMPL skinning).
The glTF export is the source of truth for the web viewer; FBX exists
only as a UE5 hand-off and is wired to be improved later.

The script is split into a CLI entry-point and a ``main`` that does the
actual work, so it stays readable without ``bpy`` available at lint time.

Run requirements:
    - Blender >= 3.6 on PATH (or configured via ``export.blender_path``).
    - The python interpreter inside Blender must have NumPy (Blender ships
      with NumPy by default).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _parse_args(argv: list[str]) -> argparse.Namespace:
    # Blender forwards arguments after ``--`` to the script.
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    parser = argparse.ArgumentParser(description="Export FBX bundle for UE5 from broadcast-mono outputs")
    parser.add_argument("--output-dir", required=True, help="Pipeline output directory")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = _parse_args(argv)
    output_dir = Path(args.output_dir).resolve()
    fbx_dir = output_dir / "export" / "fbx"
    fbx_dir.mkdir(parents=True, exist_ok=True)

    # Lazy import bpy — only available inside Blender.
    try:
        import bpy  # type: ignore
    except ImportError:
        sys.stderr.write(
            "blender_export_fbx.py must be run inside Blender (bpy unavailable)\n"
        )
        return 2

    import numpy as np

    # --- Helpers -----------------------------------------------------

    def _reset_scene() -> None:
        bpy.ops.wm.read_factory_settings(use_empty=True)

    def _set_unit_scale_metres() -> None:
        scene = bpy.context.scene
        scene.unit_settings.system = "METRIC"
        scene.unit_settings.scale_length = 1.0

    def _export_fbx(filepath: Path) -> None:
        bpy.ops.export_scene.fbx(
            filepath=str(filepath),
            use_selection=True,
            apply_unit_scale=True,
            global_scale=1.0,
            axis_forward="-Y",
            axis_up="Z",
            bake_anim=True,
            bake_anim_use_all_actions=False,
            bake_anim_use_nla_strips=False,
            bake_anim_simplify_factor=0.0,
        )

    def _add_armature(name: str) -> object:
        bpy.ops.object.armature_add(enter_editmode=False)
        arm = bpy.context.active_object
        arm.name = name
        arm.data.name = f"{name}_data"
        return arm

    # --- Player FBX --------------------------------------------------

    hmr_dir = output_dir / "hmr_world"
    fps = 30.0
    if (output_dir / "camera" / "camera_track.json").exists():
        cam_meta = json.loads((output_dir / "camera" / "camera_track.json").read_text())
        fps = float(cam_meta.get("fps", 30.0)) or 30.0

    if hmr_dir.exists():
        for npz_path in sorted(hmr_dir.glob("*_smpl_world.npz")):
            data = np.load(npz_path)
            player_id = str(data["player_id"])
            root_t = data["root_t"]
            frames = data["frames"]
            n_frames = int(frames.shape[0])
            if n_frames == 0:
                continue
            _reset_scene()
            _set_unit_scale_metres()
            scene = bpy.context.scene
            scene.frame_start = int(frames[0])
            scene.frame_end = int(frames[-1])
            scene.render.fps = int(round(fps))
            arm = _add_armature(player_id)
            for i, fi in enumerate(frames.tolist()):
                arm.location = (float(root_t[i, 0]), float(root_t[i, 1]), float(root_t[i, 2]))
                arm.keyframe_insert(data_path="location", frame=int(fi))
            bpy.ops.object.select_all(action="DESELECT")
            arm.select_set(True)
            _export_fbx(fbx_dir / f"{player_id}.fbx")

    # --- Ball FBX ----------------------------------------------------

    ball_path = output_dir / "ball" / "ball_track.json"
    if ball_path.exists():
        ball = json.loads(ball_path.read_text())
        ball_frames = [f for f in ball["frames"] if f.get("world_xyz")]
        if ball_frames:
            _reset_scene()
            _set_unit_scale_metres()
            scene = bpy.context.scene
            scene.frame_start = int(ball_frames[0]["frame"])
            scene.frame_end = int(ball_frames[-1]["frame"])
            scene.render.fps = int(round(fps))
            obj = _add_armature("ball")
            for f in ball_frames:
                obj.location = tuple(f["world_xyz"])
                obj.keyframe_insert(data_path="location", frame=int(f["frame"]))
            bpy.ops.object.select_all(action="DESELECT")
            obj.select_set(True)
            _export_fbx(fbx_dir / "ball.fbx")

    # --- Camera FBX --------------------------------------------------

    cam_path = output_dir / "camera" / "camera_track.json"
    if cam_path.exists():
        cam = json.loads(cam_path.read_text())
        frames = cam.get("frames", [])
        image_w, image_h = cam.get("image_size", [1920, 1080])
        if frames:
            _reset_scene()
            _set_unit_scale_metres()
            scene = bpy.context.scene
            scene.frame_start = int(frames[0]["frame"])
            scene.frame_end = int(frames[-1]["frame"])
            scene.render.fps = int(round(fps))
            bpy.ops.object.camera_add(location=tuple(cam.get("t_world", [0, 0, 0])))
            cam_obj = bpy.context.active_object
            cam_obj.name = "broadcast_camera"
            cam_data = cam_obj.data
            cam_data.sensor_width = float(image_w) / 100.0  # arbitrary; consumer rescales
            for f in frames:
                fx = float(f["K"][0][0])
                cam_data.lens = fx * (cam_data.sensor_width / float(image_w))
                cam_data.keyframe_insert(data_path="lens", frame=int(f["frame"]))
            bpy.ops.object.select_all(action="DESELECT")
            cam_obj.select_set(True)
            _export_fbx(fbx_dir / "camera.fbx")

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
