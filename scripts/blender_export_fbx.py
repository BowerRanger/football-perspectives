"""Blender headless FBX exporter for the broadcast-mono pipeline.

Invoked by ``ExportStage._export_fbx`` via::

    blender --background --python scripts/blender_export_fbx.py -- \
        --output-dir /path/to/output

Reads the same artefacts as ``ExportStage`` (camera_track, hmr_world NPZs,
ball_track) and emits ``output/export/fbx/PXXX.fbx`` (per player),
``ball.fbx`` and ``camera.fbx`` (UE5 conventions: scale 1.0 m, forward -Y,
up Z).

This script bakes a 24-joint SMPL armature per player (canonical rest
pose; per-frame pose-bone rotations from ``thetas``; armature object
transform carries ``root_R`` and ``root_t``). Ball and camera branches
use single-object animation (transform-only).

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
    parser = argparse.ArgumentParser(
        description="Export FBX bundle for UE5 from broadcast-mono outputs"
    )
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

    # Ensure the repo's ``src`` package is importable inside Blender.
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from src.utils.smpl_skeleton import (  # noqa: E402
        SMPL_JOINT_NAMES,
        SMPL_PARENTS,
        SMPL_REST_JOINTS_YUP,
        axis_angle_to_quaternion,
    )
    from src.utils.player_names import (  # noqa: E402
        display_name_for,
        load_player_names,
    )
    from mathutils import Matrix, Quaternion  # type: ignore  # noqa: E402

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

    def _add_simple_armature(name: str) -> object:
        """Single-bone armature, used for ball + camera branches."""
        bpy.ops.object.armature_add(enter_editmode=False)
        arm = bpy.context.active_object
        arm.name = name
        arm.data.name = f"{name}_data"
        return arm

    def _build_smpl_armature(name: str) -> object:
        """Create a 24-bone SMPL armature in canonical y-up rest pose."""
        bpy.ops.object.armature_add(enter_editmode=True)
        arm = bpy.context.active_object
        arm.name = name
        arm.data.name = f"{name}_data"
        arm.rotation_mode = "QUATERNION"
        edit_bones = arm.data.edit_bones
        # Remove the default "Bone" created by armature_add.
        for eb in list(edit_bones):
            edit_bones.remove(eb)
        bones: list[object] = []
        for j, jname in enumerate(SMPL_JOINT_NAMES):
            eb = edit_bones.new(jname)
            head = SMPL_REST_JOINTS_YUP[j]
            # Tail must differ from head; pick a 5cm offset along +y so
            # bones are visible in viewport. Direction is irrelevant for
            # FBX export of pose-bone rotations.
            eb.head = (float(head[0]), float(head[1]), float(head[2]))
            eb.tail = (float(head[0]), float(head[1] + 0.05), float(head[2]))
            if SMPL_PARENTS[j] != -1:
                eb.parent = bones[SMPL_PARENTS[j]]
                eb.use_connect = False
            bones.append(eb)
        bpy.ops.object.mode_set(mode="OBJECT")
        for pb in arm.pose.bones:
            pb.rotation_mode = "QUATERNION"
        return arm

    def _add_placeholder_skinned_mesh(arm: object, name: str) -> object:
        """Attach a minimal mesh skinned to ``pelvis`` so UE's Skeletal
        Mesh importer accepts the FBX.

        UE5 has no "skeleton-only" import path; an armature-only FBX is
        rejected with "no data to import". A 3-vertex 1mm triangle bound
        100% to pelvis is enough to satisfy the importer; the mesh
        itself is incidental and can be ignored on the UE side — what
        matters is the imported ``SK_SMPL`` skeleton asset.
        """
        mesh = bpy.data.meshes.new(f"{name}_placeholder_mesh")
        verts = [(0.001, 0.0, 0.0), (-0.001, 0.0, 0.0), (0.0, 0.001, 0.0)]
        faces = [(0, 1, 2)]
        mesh.from_pydata(verts, [], faces)
        mesh.update()
        obj = bpy.data.objects.new(f"{name}_placeholder", mesh)
        bpy.context.collection.objects.link(obj)
        obj.parent = arm
        vg = obj.vertex_groups.new(name="pelvis")
        vg.add([0, 1, 2], 1.0, "REPLACE")
        mod = obj.modifiers.new(name="Armature", type="ARMATURE")
        mod.object = arm
        mod.use_vertex_groups = True
        return obj

    def _add_smpl_skinned_mesh(arm: object, name: str, smpl_data) -> object:
        """Attach the full SMPL body mesh (mean shape) skinned to all 24
        joints. Used for previewing animations on ``SK_SMPL`` in UE
        before retargeting to a mannequin.

        ``smpl_data`` is a numpy NpzFile with keys:
            v_template (6890, 3) — y-up canonical rest vertices
            faces      (13776, 3) int32
            weights    (6890, 24) — per-vertex skinning weights
        """
        v_template = smpl_data["v_template"]
        faces = smpl_data["faces"]
        weights = smpl_data["weights"]

        mesh = bpy.data.meshes.new(f"{name}_smpl_mesh")
        verts = [(float(v[0]), float(v[1]), float(v[2])) for v in v_template]
        face_list = [(int(t[0]), int(t[1]), int(t[2])) for t in faces]
        mesh.from_pydata(verts, [], face_list)
        mesh.update()
        obj = bpy.data.objects.new(f"{name}_body", mesh)
        bpy.context.collection.objects.link(obj)
        obj.parent = arm

        # One vertex group per SMPL joint; populate from the per-vertex
        # weight matrix. Skip near-zero weights to keep the FBX small.
        weight_threshold = 1e-5
        for j, jname in enumerate(SMPL_JOINT_NAMES):
            vg = obj.vertex_groups.new(name=jname)
            col = weights[:, j]
            nonzero = np.where(col > weight_threshold)[0]
            for vi in nonzero:
                vg.add([int(vi)], float(col[vi]), "REPLACE")

        mod = obj.modifiers.new(name="Armature", type="ARMATURE")
        mod.object = arm
        mod.use_vertex_groups = True
        return obj

    # --- Player FBX --------------------------------------------------

    hmr_dir = output_dir / "hmr_world"
    fps = 30.0
    if (output_dir / "camera" / "camera_track.json").exists():
        cam_meta = json.loads((output_dir / "camera" / "camera_track.json").read_text())
        fps = float(cam_meta.get("fps", 30.0)) or 30.0

    name_mapping = load_player_names(output_dir)

    # Optional: real SMPL body mesh for preview before retargeting.
    smpl_npz_path = repo_root / "data" / "models" / "smpl_neutral.npz"
    smpl_data = None
    if smpl_npz_path.exists():
        smpl_data = dict(np.load(smpl_npz_path))
        sys.stdout.write(
            f"[player-fbx] using real SMPL body mesh from {smpl_npz_path}\n"
        )
    else:
        sys.stdout.write(
            "[player-fbx] no SMPL body npz found; falling back to placeholder triangle. "
            f"Run scripts/extract_smpl_neutral.py to enable.\n"
        )

    if hmr_dir.exists():
        for npz_path in sorted(hmr_dir.glob("*_smpl_world.npz")):
            data = np.load(npz_path)
            player_id = str(data["player_id"])
            display_name = display_name_for(player_id, name_mapping)
            frames = data["frames"]
            thetas = data["thetas"]      # (N, 24, 3)
            root_R = data["root_R"]      # (N, 3, 3)
            root_t = data["root_t"]      # (N, 3)
            n_frames = int(frames.shape[0])
            if n_frames == 0:
                continue
            _reset_scene()
            _set_unit_scale_metres()
            scene = bpy.context.scene
            scene.frame_start = int(frames[0])
            scene.frame_end = int(frames[-1])
            scene.render.fps = int(round(fps))
            arm = _build_smpl_armature(display_name)
            arm.rotation_mode = "QUATERNION"
            if smpl_data is not None:
                placeholder = _add_smpl_skinned_mesh(arm, display_name, smpl_data)
            else:
                placeholder = _add_placeholder_skinned_mesh(arm, display_name)

            for i, fi in enumerate(frames.tolist()):
                # Armature object: root translation in pitch z-up + root_R
                # rotation. ``root_R`` maps SMPL canonical y-up to pitch
                # z-up, so applying it as the armature's world rotation
                # lifts the y-up rest skeleton into pitch z-up at runtime.
                arm.location = (
                    float(root_t[i, 0]),
                    float(root_t[i, 1]),
                    float(root_t[i, 2]),
                )
                R = root_R[i]
                m = Matrix((
                    (float(R[0, 0]), float(R[0, 1]), float(R[0, 2]), 0.0),
                    (float(R[1, 0]), float(R[1, 1]), float(R[1, 2]), 0.0),
                    (float(R[2, 0]), float(R[2, 1]), float(R[2, 2]), 0.0),
                    (0.0, 0.0, 0.0, 1.0),
                ))
                arm.rotation_quaternion = m.to_quaternion()
                arm.keyframe_insert(data_path="location", frame=int(fi))
                arm.keyframe_insert(data_path="rotation_quaternion", frame=int(fi))

                # Per-bone parent-relative rotations (in canonical y-up,
                # which is the bone's local rest frame).
                for j, jname in enumerate(SMPL_JOINT_NAMES):
                    pb = arm.pose.bones[jname]
                    q = axis_angle_to_quaternion(thetas[i, j])
                    pb.rotation_quaternion = Quaternion(
                        (float(q[0]), float(q[1]), float(q[2]), float(q[3]))
                    )
                    pb.keyframe_insert(data_path="rotation_quaternion", frame=int(fi))

            bpy.ops.object.select_all(action="DESELECT")
            arm.select_set(True)
            placeholder.select_set(True)
            bpy.context.view_layer.objects.active = arm
            _export_fbx(fbx_dir / f"{display_name}.fbx")

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
            obj = _add_simple_armature("ball")
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
        image_w, _ = cam.get("image_size", [1920, 1080])
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
