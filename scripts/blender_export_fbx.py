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
    parser.add_argument(
        "--apose-only",
        action="store_true",
        help=(
            "Skip per-clip player/ball/camera export. Emit only "
            "SMPL_APose.fbx — a single-frame SMPL skeleton in A-pose "
            "(shoulders rotated 45° down) for use as the IK Retargeter "
            "retarget pose in UE5."
        ),
    )
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
        """Wipe scene + orphan data blocks between exports.

        ``read_factory_settings(use_empty=True)`` clears the scene but
        leaves orphaned datablocks (armatures, meshes, actions) in
        ``bpy.data``. Without an explicit purge those orphans get
        included in the next FBX, which leaks the previous player's
        armature object name in as a root bone — UE then fails the
        import with "Mesh contains <prev_id> bone as root" because the
        animation tracks don't carry that orphan-derived root.

        ``bpy.ops.outliner.orphans_purge`` is unreliable in headless
        Blender (needs an active outliner area). Iterate the data
        collections directly: repeat until stable so chained orphans
        (mesh → material → texture) all go.
        """
        bpy.ops.wm.read_factory_settings(use_empty=True)
        _purge_orphans()

    def _purge_orphans() -> None:
        collections = (
            bpy.data.objects,
            bpy.data.meshes,
            bpy.data.armatures,
            bpy.data.actions,
            bpy.data.materials,
            bpy.data.images,
            bpy.data.textures,
            bpy.data.cameras,
            bpy.data.lights,
            bpy.data.curves,
            bpy.data.node_groups,
        )
        for _ in range(8):  # bounded fixed-point iteration
            removed = 0
            for col in collections:
                for item in list(col):
                    if item.users == 0:
                        col.remove(item)
                        removed += 1
            if removed == 0:
                break

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

    # Constant armature object name shared by every per-player export.
    # Blender's FBX exporter encodes the armature object as the root of
    # the FBX bone hierarchy. If we used a player-specific name, every
    # FBX would have a different root and could never bind cleanly to a
    # single ``SK_SMPL`` skeleton in UE. A constant name lets one
    # ``SK_SMPL`` import bind every player's animation.
    SMPL_ARMATURE_NAME = "SMPLArmature"

    def _build_smpl_armature(name: str, joint_positions=None) -> object:
        """Create a 24-bone SMPL armature in canonical y-up rest pose.

        ``joint_positions``: (24, 3) array of joint positions in canonical
        y-up. When supplied, these (typically ``J_regressor @ v_template``
        from the bundled SMPL pkl) are used so the bones align with the
        SMPL mesh. Falls back to the hand-typed ``SMPL_REST_JOINTS_YUP``
        table when ``None`` — that table has the pelvis re-centred to the
        origin and so sits ~22cm above the real joint positions, which
        is fine for the JS viewer's bone overlay but mis-aligns the
        skeleton against the SMPL mesh in UE.

        ``name`` is retained in the signature only for backwards
        compatibility with callers that also use it to name the skinned
        mesh; the armature itself is always ``SMPLArmature``.
        """
        del name  # see SMPL_ARMATURE_NAME above
        joints = (
            joint_positions
            if joint_positions is not None
            else SMPL_REST_JOINTS_YUP
        )
        bpy.ops.object.armature_add(enter_editmode=True)
        arm = bpy.context.active_object
        arm.name = SMPL_ARMATURE_NAME
        arm.data.name = f"{SMPL_ARMATURE_NAME}_data"
        arm.rotation_mode = "QUATERNION"
        edit_bones = arm.data.edit_bones
        # Remove the default "Bone" created by armature_add.
        for eb in list(edit_bones):
            edit_bones.remove(eb)
        bones: list[object] = []
        for j, jname in enumerate(SMPL_JOINT_NAMES):
            eb = edit_bones.new(jname)
            head = joints[j]
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

    def _add_placeholder_skinned_mesh(
        arm: object, name: str, bone_name: str = "pelvis"
    ) -> object:
        """Attach a minimal mesh skinned to ``bone_name`` so UE's
        Skeletal Mesh importer accepts the FBX.

        UE5 has no "skeleton-only" import path; an armature-only FBX is
        rejected with "no data to import". A 3-vertex 1mm triangle bound
        100% to one bone is enough to satisfy the importer; the mesh
        itself is incidental and can be ignored on the UE side — what
        matters is the imported skeleton asset and its animation.
        """
        mesh = bpy.data.meshes.new(f"{name}_placeholder_mesh")
        verts = [(0.001, 0.0, 0.0), (-0.001, 0.0, 0.0), (0.0, 0.001, 0.0)]
        faces = [(0, 1, 2)]
        mesh.from_pydata(verts, [], faces)
        mesh.update()
        obj = bpy.data.objects.new(f"{name}_placeholder", mesh)
        bpy.context.collection.objects.link(obj)
        obj.parent = arm
        vg = obj.vertex_groups.new(name=bone_name)
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
    smpl_joint_positions = None
    # Pelvis position in canonical-space (after the foot-midpoint shift
    # below). The per-clip branch needs this to compute per-frame
    # ``arm.location = root_t - root_R @ pelvis_canon_shifted`` so the
    # pelvis still lands at root_t even though the canonical layout
    # has been re-anchored on the foot midpoint.
    pelvis_canon_shifted = np.zeros(3, dtype=np.float64)
    if smpl_npz_path.exists():
        smpl_data = dict(np.load(smpl_npz_path))
        if "joint_positions" in smpl_data and "v_template" in smpl_data:
            # Re-anchor canonical space so the foot midpoint sits at
            # (0, 0, 0). Both the bone hierarchy and the mesh vertices
            # are shifted; the armature object's origin then lands
            # between the feet at ground level in UE.
            jp = np.asarray(smpl_data["joint_positions"], dtype=np.float64)
            l_foot_idx = SMPL_JOINT_NAMES.index("l_foot")
            r_foot_idx = SMPL_JOINT_NAMES.index("r_foot")
            foot_midpoint = (jp[l_foot_idx] + jp[r_foot_idx]) / 2.0
            shift = -foot_midpoint
            smpl_data["joint_positions"] = (jp + shift).astype(np.float32)
            smpl_data["v_template"] = (
                np.asarray(smpl_data["v_template"], dtype=np.float64) + shift
            ).astype(np.float32)
            smpl_joint_positions = smpl_data["joint_positions"]
            pelvis_canon_shifted = np.asarray(
                smpl_joint_positions[0], dtype=np.float64
            )
            sys.stdout.write(
                f"[player-fbx] using real SMPL body mesh from {smpl_npz_path}"
                f" (foot-midpoint anchored; pelvis canon = "
                f"{tuple(float(x) for x in pelvis_canon_shifted)})\n"
            )
        else:
            sys.stdout.write(
                f"[player-fbx] {smpl_npz_path} missing joint_positions or "
                "v_template; re-run scripts/extract_smpl_neutral.py.\n"
            )
            smpl_data = None
    else:
        sys.stdout.write(
            "[player-fbx] no SMPL body npz found; falling back to placeholder triangle. "
            f"Run scripts/extract_smpl_neutral.py to enable.\n"
        )

    # --- A-pose only mode -------------------------------------------------
    # Emit a single-frame SMPL_APose.fbx for use as the IK Retargeter
    # retarget pose in UE5. Shoulders rotated 45° down to match Manny's
    # A-pose bind. No per-clip data; bypasses the player/ball/camera
    # branches.
    if args.apose_only:
        _reset_scene()
        _set_unit_scale_metres()
        scene = bpy.context.scene
        scene.frame_start = 0
        scene.frame_end = 0
        scene.render.fps = int(round(fps))
        arm = _build_smpl_armature("SMPL_APose", joint_positions=smpl_joint_positions)
        arm.rotation_mode = "QUATERNION"
        if smpl_data is not None:
            mesh_obj = _add_smpl_skinned_mesh(arm, "SMPL_APose", smpl_data)
        else:
            mesh_obj = _add_placeholder_skinned_mesh(arm, "SMPL_APose")

        # Armature object: rotate canonical y-up rest skeleton into pitch
        # z-up world (90° about +X). No translation — the canonical-space
        # shift above already re-anchored the bone hierarchy on the foot
        # midpoint, so the asset origin in UE lands between the feet at
        # ground level when arm.location = (0, 0, 0).
        from math import cos, sin, pi
        arm.location = (0.0, 0.0, 0.0)
        # Quaternion for 90° about +X: w=cos(45°), x=sin(45°).
        arm.rotation_quaternion = Quaternion(
            (float(cos(pi / 4)), float(sin(pi / 4)), 0.0, 0.0)
        )
        arm.keyframe_insert(data_path="location", frame=0)
        arm.keyframe_insert(data_path="rotation_quaternion", frame=0)

        # Per-bone rotations: identity everywhere except the shoulders.
        # In SMPL canonical (y-up), the arm at rest points along +X (left
        # arm) or -X (right arm). Rotating around the canonical +Z axis
        # tilts the arm in the X-Y plane (i.e. up/down).
        #   l_shoulder: -45° around +Z brings the arm down and in.
        #   r_shoulder: +45° around +Z mirrors that.
        import numpy as _np
        APOSE_THETAS = _np.zeros((24, 3), dtype=_np.float64)
        APOSE_THETAS[SMPL_JOINT_NAMES.index("l_shoulder")] = (0.0, 0.0, -pi / 4)
        APOSE_THETAS[SMPL_JOINT_NAMES.index("r_shoulder")] = (0.0, 0.0, +pi / 4)
        # Skip pelvis (j=0) — its world rotation is on the armature object.
        pelvis_pb = arm.pose.bones[SMPL_JOINT_NAMES[0]]
        pelvis_pb.rotation_quaternion = Quaternion((1.0, 0.0, 0.0, 0.0))
        pelvis_pb.keyframe_insert(data_path="rotation_quaternion", frame=0)
        for j in range(1, len(SMPL_JOINT_NAMES)):
            pb = arm.pose.bones[SMPL_JOINT_NAMES[j]]
            q = axis_angle_to_quaternion(APOSE_THETAS[j])
            pb.rotation_quaternion = Quaternion(
                (float(q[0]), float(q[1]), float(q[2]), float(q[3]))
            )
            pb.keyframe_insert(data_path="rotation_quaternion", frame=0)

        bpy.ops.object.select_all(action="DESELECT")
        arm.select_set(True)
        mesh_obj.select_set(True)
        bpy.context.view_layer.objects.active = arm
        _export_fbx(fbx_dir / "SMPL_APose.fbx")
        sys.stdout.write(
            f"[apose] wrote {fbx_dir / 'SMPL_APose.fbx'}\n"
        )
        return 0

    if hmr_dir.exists():
        for npz_path in sorted(hmr_dir.glob("*_smpl_world.npz")):
            data = np.load(npz_path)
            player_id = str(data["player_id"])
            shot_id = str(data["shot_id"]) if "shot_id" in data.files else ""
            display_name = display_name_for(player_id, name_mapping)
            # FBX filenames must be unique per (shot, player) — two shots
            # with the same player_id (e.g. after Merge by Name) would
            # otherwise overwrite each other on disk.
            fbx_name = (
                f"{shot_id}__{display_name}" if shot_id else display_name
            )
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
            arm = _build_smpl_armature(display_name, joint_positions=smpl_joint_positions)
            arm.rotation_mode = "QUATERNION"
            if smpl_data is not None:
                placeholder = _add_smpl_skinned_mesh(arm, display_name, smpl_data)
            else:
                placeholder = _add_placeholder_skinned_mesh(arm, display_name)

            for i, fi in enumerate(frames.tolist()):
                # Armature object: rotation = root_R[i] (SMPL canonical
                # y-up → pitch z-up + player body orientation). Location
                # is set so the pelvis ends up at root_t[i] in world,
                # accounting for the canonical-space foot-midpoint
                # anchor — pelvis sits at canonical pelvis_canon_shifted
                # in armature local space, so its world position is
                # arm.location + root_R[i] @ pelvis_canon_shifted.
                pelvis_world_offset_i = (
                    root_R[i] @ pelvis_canon_shifted
                )
                arm.location = (
                    float(root_t[i, 0] - pelvis_world_offset_i[0]),
                    float(root_t[i, 1] - pelvis_world_offset_i[1]),
                    float(root_t[i, 2] - pelvis_world_offset_i[2]),
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

                # Per-bone parent-relative rotations. The pelvis (j=0)
                # is intentionally skipped — its world rotation is
                # already carried by the armature object's root_R, just
                # like the web viewer's smplFK (viewer.html:smplFK
                # explicitly ignores thetas[0]). Applying both produces
                # a double-rotated, flipping pelvis.
                pelvis_pb = arm.pose.bones[SMPL_JOINT_NAMES[0]]
                pelvis_pb.rotation_quaternion = Quaternion((1.0, 0.0, 0.0, 0.0))
                pelvis_pb.keyframe_insert(data_path="rotation_quaternion", frame=int(fi))
                for j in range(1, len(SMPL_JOINT_NAMES)):
                    pb = arm.pose.bones[SMPL_JOINT_NAMES[j]]
                    q = axis_angle_to_quaternion(thetas[i, j])
                    pb.rotation_quaternion = Quaternion(
                        (float(q[0]), float(q[1]), float(q[2]), float(q[3]))
                    )
                    pb.keyframe_insert(data_path="rotation_quaternion", frame=int(fi))

            bpy.ops.object.select_all(action="DESELECT")
            arm.select_set(True)
            placeholder.select_set(True)
            bpy.context.view_layer.objects.active = arm
            # Diagnostic: confirms _reset_scene + _purge_orphans actually
            # gave us a clean slate per iteration. Any divergence between
            # arm.name and fbx_name in the log means state leaked.
            print(
                f"[blender_export_fbx] exporting {fbx_name}.fbx "
                f"with arm.name={arm.name!r} arm.data.name={arm.data.name!r} "
                f"objects_in_scene={[o.name for o in bpy.data.objects]}"
            )
            _export_fbx(fbx_dir / f"{fbx_name}.fbx")

    # --- Ball FBX (per shot) -----------------------------------------
    # Multi-shot layout writes ``ball/{shot_id}_ball_track.json`` and
    # this script emits ``fbx/{shot_id}_ball.fbx`` for each one. The
    # legacy unprefixed ``ball/ball_track.json`` is still picked up for
    # back-compat with old single-shot runs.

    ball_dir = output_dir / "ball"
    if ball_dir.exists():
        ball_track_paths: list[tuple[str, Path]] = []
        for path in sorted(ball_dir.glob("*_ball_track.json")):
            shot_id = path.stem[: -len("_ball_track")]
            ball_track_paths.append((shot_id, path))
        legacy = ball_dir / "ball_track.json"
        if legacy.exists():
            ball_track_paths.append(("", legacy))
        for shot_id, ball_path in ball_track_paths:
            ball = json.loads(ball_path.read_text())
            ball_frames = [f for f in ball["frames"] if f.get("world_xyz")]
            if not ball_frames:
                continue
            _reset_scene()
            _set_unit_scale_metres()
            scene = bpy.context.scene
            scene.frame_start = int(ball_frames[0]["frame"])
            scene.frame_end = int(ball_frames[-1]["frame"])
            scene.render.fps = int(round(fps))
            obj = _add_simple_armature("ball")
            # UE rejects skeleton-only FBX imports — bind a tiny
            # placeholder mesh to the armature's default bone ("Bone",
            # from bpy.ops.object.armature_add) so the FBX is a valid
            # SkeletalMesh+Anim asset on the UE side. The placeholder
            # mesh is discarded; BP_BallActor provides the real sphere.
            placeholder = _add_placeholder_skinned_mesh(obj, "ball", bone_name="Bone")
            for f in ball_frames:
                obj.location = tuple(f["world_xyz"])
                obj.keyframe_insert(data_path="location", frame=int(f["frame"]))
            bpy.ops.object.select_all(action="DESELECT")
            obj.select_set(True)
            placeholder.select_set(True)
            bpy.context.view_layer.objects.active = obj
            fbx_name = f"{shot_id}_ball.fbx" if shot_id else "ball.fbx"
            _export_fbx(fbx_dir / fbx_name)

    # --- Camera FBX (per shot) ---------------------------------------
    # Same per-shot pattern as ball above.

    cam_dir = output_dir / "camera"
    if cam_dir.exists():
        cam_track_paths: list[tuple[str, Path]] = []
        for path in sorted(cam_dir.glob("*_camera_track.json")):
            shot_id = path.stem[: -len("_camera_track")]
            cam_track_paths.append((shot_id, path))
        legacy_cam = cam_dir / "camera_track.json"
        if legacy_cam.exists():
            cam_track_paths.append(("", legacy_cam))
        for shot_id, cam_path in cam_track_paths:
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
            fbx_name = f"{shot_id}_camera.fbx" if shot_id else "camera.fbx"
            _export_fbx(fbx_dir / fbx_name)

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
