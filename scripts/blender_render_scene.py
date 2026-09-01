"""Headless Blender toon renderer for the broadcast-mono pipeline.

Invoked by RenderStage via:

    blender --background --python scripts/blender_render_scene.py -- \
        --output-dir OUT --shot SHOT --cameras broadcast,drone ...

Assembles a fully procedural scene (no binary assets): pitch + lines
from src/utils/pitch.py geometry, procedural stadium bowl, players from
refined_poses NPZs (Task 6+), ball from the dense ball track. Renders
EEVEE to output/render/<shot>/<camera>.mp4.

Split into module-level pure helpers (importable/testable without
``bpy``) and a ``main()`` that lazily imports ``bpy`` — same structure
as ``scripts/blender_export_fbx.py``. The bpy-dependent scene builders
(``_build_environment``, ``_build_ball``, ``_add_camera_from_track``,
``_render``) are nested inside ``main()`` since they close over the
lazily-imported ``bpy``/``bmesh``/``mathutils`` modules; later tasks
(players, toon materials, virtual-camera renders, vertical/AOV
variants) extend those nested functions in place.
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

from src.utils.pitch import PITCH_LENGTH, PITCH_WIDTH  # noqa: E402

# --- Pitch-geometry constants -------------------------------------------
# Mirrors src/utils/pitch.py's private constants (not importable — they
# are prefixed with `_` there). Kept in sync by hand; pitch.py is the
# authority for FIFA landmark coordinates this geometry should agree
# with (see FIFA_LANDMARKS in that module).
CENTRE_CIRCLE_R = 9.15          # mirrors pitch._CIRCLE_R
PENALTY_BOX_DEPTH_M = 16.5
PENALTY_BOX_WIDTH_M = 40.32
SIX_YARD_BOX_DEPTH_M = 5.5
SIX_YARD_BOX_WIDTH_M = 18.32
GOAL_HALF_WIDTH_M = 3.66        # mirrors pitch._GOAL_HALF (7.32 / 2)
GOAL_HEIGHT_M = 2.44            # mirrors pitch._GOAL_HEIGHT
GOAL_POST_RADIUS_M = 0.06

# --- Render-scene constants ----------------------------------------------
LINE_Z = 0.02
LINE_BEVEL_DEPTH = 0.06
STADIUM_SEGMENTS = 48
STADIUM_INNER_R = 75.0
STADIUM_INNER_H = 2.0
STADIUM_OUTER_R = 95.0
STADIUM_OUTER_H = 18.0
# Two-tone stand shading: both derived from palette["outline"], alternated
# per angular segment (see _build_stadium) for a flat dark two-tone look.
STADIUM_DARK_FACTOR = 0.55
STADIUM_LIGHT_FACTOR = 0.8
BALL_RADIUS_M = 0.11
SENSOR_WIDTH_MM = 36.0
DEFAULT_FPS = 25.0
DEFAULT_SUN_ROTATION_DEG = (50.0, 0.0, -30.0)
DEFAULT_SUN_ENERGY = 3.0

# Task 2's config/default.yaml `render.style` block, verbatim — the
# fallback whenever `--style-json` omits a key (RenderStage always
# passes `render.style`, but a bare `{}` — as in the smoke test — must
# still produce a fully populated style).
_DEFAULT_STYLE: dict = {
    "palette": {
        "grass_light": "#4d9e46",
        "grass_dark": "#3f8a3a",
        "lines": "#f5f5f0",
        "sky_top": "#9ecfe8",
        "sky_bottom": "#e8f4d8",
        "outline": "#1a1a1a",
    },
    "ramp_steps": 3,
    "outline_width_m": 0.02,
    "grass_stripes": 10,
}


def _resolve_style(style: dict) -> dict:
    """Merge a (possibly partial) style dict over the Task 2 defaults.

    Pure — no ``bpy`` — so it's unit-testable on its own. Top-level
    keys and the nested ``palette`` dict are merged independently so a
    caller can override e.g. only ``palette.grass_light`` without
    having to repeat every other palette entry.
    """
    merged = {k: v for k, v in _DEFAULT_STYLE.items() if k != "palette"}
    merged.update({k: v for k, v in style.items() if k != "palette"})
    palette = dict(_DEFAULT_STYLE["palette"])
    palette.update(style.get("palette") or {})
    merged["palette"] = palette
    return merged


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


def main(argv: list[str]) -> int:
    args = _parse_args(argv)
    style = _resolve_style(json.loads(args.style_json))

    try:
        import bpy  # type: ignore
    except ImportError:
        sys.stderr.write(
            "blender_render_scene.py must be run inside Blender (bpy unavailable)\n"
        )
        return 2

    if tuple(bpy.app.version) < (3, 6, 0):
        sys.stderr.write(
            f"Blender >= 3.6 required, got {bpy.app.version}\n"
        )
        return 2

    import bmesh  # type: ignore
    from math import radians

    from mathutils import Matrix, Quaternion  # type: ignore

    from src.utils import render_look
    from src.utils.blender_scene_io import load_camera_track, prepare_ball_keys

    if args.vertical:
        sys.stdout.write(
            "[render] --vertical parsed but not yet implemented (Task 9); ignoring\n"
        )
    if args.aov:
        sys.stdout.write(
            "[render] --aov parsed but not yet implemented (Task 9); ignoring\n"
        )

    output_dir = Path(args.output_dir).resolve()
    shot = args.shot
    # Legacy empty shot id renders under a fixed "clip" directory name —
    # keeps output/render/<dir>/<camera>.mp4 stable for single-shot runs.
    shot_dir = shot or "clip"
    out_dir = output_dir / "render" / shot_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    bpy.ops.wm.read_factory_settings(use_empty=True)

    # --- Materials ---------------------------------------------------

    def _new_diffuse_material(name: str, rgba) -> object:
        mat = bpy.data.materials.new(name)
        mat.use_nodes = True
        nt = mat.node_tree
        nt.nodes.clear()
        out = nt.nodes.new("ShaderNodeOutputMaterial")
        bsdf = nt.nodes.new("ShaderNodeBsdfDiffuse")
        bsdf.inputs["Color"].default_value = rgba
        nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
        return mat

    def _new_emission_material(name: str, rgba, strength: float = 1.0) -> object:
        mat = bpy.data.materials.new(name)
        mat.use_nodes = True
        nt = mat.node_tree
        nt.nodes.clear()
        out = nt.nodes.new("ShaderNodeOutputMaterial")
        emis = nt.nodes.new("ShaderNodeEmission")
        emis.inputs["Color"].default_value = rgba
        emis.inputs["Strength"].default_value = strength
        nt.links.new(emis.outputs["Emission"], out.inputs["Surface"])
        return mat

    def _grass_material(style: dict, palette: dict) -> object:
        """Mown-stripe grass: alternating light/dark bands across the
        pitch length. Uses the mesh's normalized Generated coordinates
        (0..1 across the plane's bounding box, stable under the
        object's own Scale) so `grass_stripes` directly sets the
        number of visible bands regardless of pitch padding.
        """
        light = render_look.hex_to_linear_rgba(palette["grass_light"])
        dark = render_look.hex_to_linear_rgba(palette["grass_dark"])
        stripes = float(style.get("grass_stripes", 10))
        mat = bpy.data.materials.new("M_Grass")
        mat.use_nodes = True
        nt = mat.node_tree
        nt.nodes.clear()
        out = nt.nodes.new("ShaderNodeOutputMaterial")
        diffuse = nt.nodes.new("ShaderNodeBsdfDiffuse")
        mix = nt.nodes.new("ShaderNodeMixRGB")
        mix.inputs["Color1"].default_value = dark
        mix.inputs["Color2"].default_value = light
        mfloor = nt.nodes.new("ShaderNodeMath")
        mfloor.operation = "FLOOR"
        mmod = nt.nodes.new("ShaderNodeMath")
        mmod.operation = "MODULO"
        mmod.inputs[1].default_value = 2.0
        mmul = nt.nodes.new("ShaderNodeMath")
        mmul.operation = "MULTIPLY"
        mmul.inputs[1].default_value = stripes
        sep = nt.nodes.new("ShaderNodeSeparateXYZ")
        texc = nt.nodes.new("ShaderNodeTexCoord")
        nt.links.new(texc.outputs["Generated"], sep.inputs["Vector"])
        nt.links.new(sep.outputs["X"], mmul.inputs[0])
        nt.links.new(mmul.outputs[0], mfloor.inputs[0])
        nt.links.new(mfloor.outputs[0], mmod.inputs[0])
        nt.links.new(mmod.outputs[0], mix.inputs["Fac"])
        nt.links.new(mix.outputs["Color"], diffuse.inputs["Color"])
        nt.links.new(diffuse.outputs["BSDF"], out.inputs["Surface"])
        return mat

    # --- Environment builders -----------------------------------------

    def _build_pitch(style: dict, palette: dict) -> None:
        bpy.ops.mesh.primitive_plane_add(size=1)
        plane = bpy.context.active_object
        plane.name = "Pitch"
        plane.scale = (PITCH_LENGTH + 10, PITCH_WIDTH + 10, 1.0)
        plane.location = (PITCH_LENGTH / 2, PITCH_WIDTH / 2, 0.0)
        plane.data.materials.append(_grass_material(style, palette))

    def _line_object(name: str, points: list[tuple[float, float]],
                      mat: object, cyclic: bool = False) -> object:
        curve_data = bpy.data.curves.new(name, type="CURVE")
        curve_data.dimensions = "3D"
        curve_data.bevel_depth = LINE_BEVEL_DEPTH
        spline = curve_data.splines.new("POLY")
        spline.points.add(len(points) - 1)
        for i, (x, y) in enumerate(points):
            spline.points[i].co = (x, y, LINE_Z, 1.0)
        spline.use_cyclic_u = cyclic
        obj = bpy.data.objects.new(name, curve_data)
        obj.data.materials.append(mat)
        bpy.context.collection.objects.link(obj)
        return obj

    def _box_points(x0: float, x1: float, half_width: float
                     ) -> list[tuple[float, float]]:
        y_near = PITCH_WIDTH / 2 - half_width
        y_far = PITCH_WIDTH / 2 + half_width
        return [(x0, y_near), (x1, y_near), (x1, y_far), (x0, y_far)]

    def _build_lines(lines_mat: object) -> None:
        _line_object("L_Boundary", [
            (0.0, 0.0), (PITCH_LENGTH, 0.0),
            (PITCH_LENGTH, PITCH_WIDTH), (0.0, PITCH_WIDTH),
        ], lines_mat, cyclic=True)
        _line_object("L_Halfway", [
            (PITCH_LENGTH / 2, 0.0), (PITCH_LENGTH / 2, PITCH_WIDTH),
        ], lines_mat)

        bpy.ops.curve.primitive_bezier_circle_add(
            radius=CENTRE_CIRCLE_R,
            location=(PITCH_LENGTH / 2, PITCH_WIDTH / 2, LINE_Z))
        circle = bpy.context.active_object
        circle.name = "L_CentreCircle"
        circle.data.bevel_depth = LINE_BEVEL_DEPTH
        circle.data.materials.append(lines_mat)

        _line_object("L_PenaltyBox_Left", _box_points(
            0.0, PENALTY_BOX_DEPTH_M, PENALTY_BOX_WIDTH_M / 2),
            lines_mat, cyclic=True)
        _line_object("L_PenaltyBox_Right", _box_points(
            PITCH_LENGTH - PENALTY_BOX_DEPTH_M, PITCH_LENGTH,
            PENALTY_BOX_WIDTH_M / 2), lines_mat, cyclic=True)
        _line_object("L_SixYard_Left", _box_points(
            0.0, SIX_YARD_BOX_DEPTH_M, SIX_YARD_BOX_WIDTH_M / 2),
            lines_mat, cyclic=True)
        _line_object("L_SixYard_Right", _box_points(
            PITCH_LENGTH - SIX_YARD_BOX_DEPTH_M, PITCH_LENGTH,
            SIX_YARD_BOX_WIDTH_M / 2), lines_mat, cyclic=True)

    def _build_goals(lines_mat: object) -> None:
        for x in (0.0, PITCH_LENGTH):
            for y in (PITCH_WIDTH / 2 - GOAL_HALF_WIDTH_M,
                      PITCH_WIDTH / 2 + GOAL_HALF_WIDTH_M):
                bpy.ops.mesh.primitive_cylinder_add(
                    radius=GOAL_POST_RADIUS_M, depth=GOAL_HEIGHT_M,
                    location=(x, y, GOAL_HEIGHT_M / 2))
                bpy.context.active_object.data.materials.append(lines_mat)
            bpy.ops.mesh.primitive_cylinder_add(
                radius=GOAL_POST_RADIUS_M, depth=GOAL_HALF_WIDTH_M * 2,
                location=(x, PITCH_WIDTH / 2, GOAL_HEIGHT_M),
                rotation=(radians(90), 0.0, 0.0))
            bpy.context.active_object.data.materials.append(lines_mat)

    def _build_stadium(palette: dict) -> None:
        """Raked stadium bowl ring: a low inward-tucked riser wall from
        the r=95 ground footprint to r=75 at h=2, then the main bowl
        flaring back out to r=95 at h=18 — built directly with bmesh
        so the two extrude+scale stages are explicit.

        Flat dark two-tone stand: two diffuse materials (both shades of
        `palette["outline"]`) are alternated per angular segment across
        both extrusion bands, giving vertical light/dark bays around the
        bowl rather than a single flat tone.
        """
        mesh = bpy.data.meshes.new("StadiumBowl")
        bm = bmesh.new()
        bmesh.ops.create_circle(
            bm, cap_ends=False, segments=STADIUM_SEGMENTS,
            radius=STADIUM_OUTER_R)
        base_edges = list(bm.edges)

        ext1 = bmesh.ops.extrude_edge_only(bm, edges=base_edges)
        verts1 = [g for g in ext1["geom"] if isinstance(g, bmesh.types.BMVert)]
        edges1 = [g for g in ext1["geom"] if isinstance(g, bmesh.types.BMEdge)]
        faces1 = [g for g in ext1["geom"] if isinstance(g, bmesh.types.BMFace)]
        bmesh.ops.translate(bm, verts=verts1, vec=(0.0, 0.0, STADIUM_INNER_H))
        scale1 = STADIUM_INNER_R / STADIUM_OUTER_R
        bmesh.ops.scale(bm, verts=verts1, vec=(scale1, scale1, 1.0))

        ext2 = bmesh.ops.extrude_edge_only(bm, edges=edges1)
        verts2 = [g for g in ext2["geom"] if isinstance(g, bmesh.types.BMVert)]
        faces2 = [g for g in ext2["geom"] if isinstance(g, bmesh.types.BMFace)]
        bmesh.ops.translate(
            bm, verts=verts2, vec=(0.0, 0.0, STADIUM_OUTER_H - STADIUM_INNER_H))
        scale2 = STADIUM_OUTER_R / STADIUM_INNER_R
        bmesh.ops.scale(bm, verts=verts2, vec=(scale2, scale2, 1.0))

        # Alternate material slot 0/1 per angular segment, independently
        # in each band (both bands index their segments 0..N-1 in the
        # same order they were created from `base_edges`/`edges1`, so a
        # given segment gets the same tone in both bands — vertical bays).
        for i, f in enumerate(faces1):
            f.material_index = i % 2
        for i, f in enumerate(faces2):
            f.material_index = i % 2

        bm.to_mesh(mesh)
        bm.free()
        obj = bpy.data.objects.new("StadiumBowl", mesh)
        bpy.context.collection.objects.link(obj)
        obj.location = (PITCH_LENGTH / 2, PITCH_WIDTH / 2, 0.0)

        outline = render_look.hex_to_linear_rgba(palette["outline"])
        dark = tuple(c * STADIUM_DARK_FACTOR for c in outline[:3]) + (1.0,)
        light = tuple(c * STADIUM_LIGHT_FACTOR for c in outline[:3]) + (1.0,)
        obj.data.materials.append(_new_diffuse_material("M_Stand_Dark", dark))
        obj.data.materials.append(_new_diffuse_material("M_Stand_Light", light))

    def _build_world(palette: dict) -> None:
        top = render_look.hex_to_linear_rgba(palette["sky_top"])
        bottom = render_look.hex_to_linear_rgba(palette["sky_bottom"])
        world = bpy.data.worlds.new("W_Sky")
        world.use_nodes = True
        nt = world.node_tree
        nt.nodes.clear()
        out = nt.nodes.new("ShaderNodeOutputWorld")
        bg = nt.nodes.new("ShaderNodeBackground")
        mix = nt.nodes.new("ShaderNodeMixRGB")
        mix.inputs["Color1"].default_value = bottom
        mix.inputs["Color2"].default_value = top
        sep = nt.nodes.new("ShaderNodeSeparateXYZ")
        texc = nt.nodes.new("ShaderNodeTexCoord")
        nt.links.new(texc.outputs["Generated"], sep.inputs["Vector"])
        nt.links.new(sep.outputs["Z"], mix.inputs["Fac"])
        nt.links.new(mix.outputs["Color"], bg.inputs["Color"])
        nt.links.new(bg.outputs["Background"], out.inputs["Surface"])
        bpy.context.scene.world = world

        bpy.ops.object.light_add(type="SUN")
        sun = bpy.context.active_object
        sun.name = "Sun"
        sun.rotation_euler = tuple(radians(d) for d in DEFAULT_SUN_ROTATION_DEG)
        sun.data.energy = DEFAULT_SUN_ENERGY

    def _build_environment(style: dict) -> None:
        palette = style["palette"]
        lines_mat = _new_emission_material(
            "M_Lines", render_look.hex_to_linear_rgba(palette["lines"]))
        _build_pitch(style, palette)
        _build_lines(lines_mat)
        _build_goals(lines_mat)
        _build_stadium(palette)
        _build_world(palette)

    # --- Ball ----------------------------------------------------------

    def _build_ball(ball_keys: list[dict]) -> object:
        bpy.ops.mesh.primitive_uv_sphere_add(radius=BALL_RADIUS_M)
        obj = bpy.context.active_object
        obj.name = "Ball"
        obj.rotation_mode = "QUATERNION"
        for k in ball_keys:
            fr = int(k["frame"])
            obj.location = tuple(k["location"])
            obj.keyframe_insert(data_path="location", frame=fr)
            obj.rotation_quaternion = Quaternion(tuple(k["rotation_quaternion"]))
            obj.keyframe_insert(data_path="rotation_quaternion", frame=fr)
        return obj

    # --- Camera ----------------------------------------------------------

    def _add_camera_from_track(cam_id: str, track: dict, width: int,
                                height: int) -> object:
        cam_data = bpy.data.cameras.new(cam_id)
        cam_data.sensor_width = SENSOR_WIDTH_MM
        cam_data.sensor_fit = "HORIZONTAL"
        cam_obj = bpy.data.objects.new(cam_id, cam_data)
        bpy.context.collection.objects.link(cam_obj)
        for fr_data in track.get("frames", []):
            fr = int(fr_data["frame"])
            cam_obj.matrix_world = Matrix(
                render_look.blender_camera_world_matrix(fr_data["R"], fr_data["t"]))
            cam_data.lens = render_look.lens_mm_from_K(fr_data["K"], width)
            cam_obj.keyframe_insert(data_path="location", frame=fr)
            cam_obj.keyframe_insert(data_path="rotation_euler", frame=fr)
            cam_data.keyframe_insert(data_path="lens", frame=fr)
        return cam_obj

    # --- Render ----------------------------------------------------------

    def _render(camera_obj: object, out_path: Path, fps: float,
                frame_range: tuple[int, int], width: int, height: int,
                samples: int) -> None:
        scene = bpy.context.scene
        scene.camera = camera_obj
        scene.render.resolution_x = width
        scene.render.resolution_y = height
        scene.render.fps = int(round(fps))
        scene.frame_start, scene.frame_end = frame_range

        engines = [e.identifier for e in
                   scene.render.bl_rna.properties["engine"].enum_items]
        scene.render.engine = (
            "BLENDER_EEVEE_NEXT" if "BLENDER_EEVEE_NEXT" in engines
            else "BLENDER_EEVEE"
        )
        scene.eevee.taa_render_samples = samples

        imf = scene.render.image_settings
        # Blender >= 5.0 gates movie formats behind a new `media_type`
        # enum (IMAGE / MULTI_LAYER_IMAGE / VIDEO) — `file_format =
        # 'FFMPEG'` raises a TypeError until this is set. The attribute
        # doesn't exist pre-5.0, where FFMPEG is directly selectable.
        if hasattr(imf, "media_type"):
            imf.media_type = "VIDEO"
        imf.file_format = "FFMPEG"
        scene.render.ffmpeg.format = "MPEG4"
        scene.render.ffmpeg.codec = "H264"
        # Blender would otherwise append the frame range to the
        # filename (e.g. `broadcast0002.mp4`); the stage/tests expect
        # the exact path passed in.
        scene.render.use_file_extension = False
        out_path.parent.mkdir(parents=True, exist_ok=True)
        scene.render.filepath = str(out_path)

        t0 = time.time()
        bpy.ops.render.render(animation=True)
        elapsed = time.time() - t0
        n_frames = frame_range[1] - frame_range[0] + 1
        # Eyeballing aid for the stage log; the quality report parses
        # render/render_timings.json (written by RenderStage) instead.
        print(f"RENDER_TIMING {camera_obj.name} {elapsed:.2f} {n_frames}")

    # --- Orchestration ---------------------------------------------------

    _build_environment(style)

    ball_path = (
        output_dir / "ball" / (f"{shot}_ball_track.json" if shot else "ball_track.json")
    )
    if ball_path.exists():
        ball_raw = json.loads(ball_path.read_text())
        ball_keys = prepare_ball_keys(ball_raw.get("frames", []))
        if ball_keys:
            _build_ball(ball_keys)
    else:
        sys.stdout.write(f"[render] no ball track at {ball_path}; skipping ball\n")

    def _camera_track_path(cam_id: str) -> Path:
        if cam_id == "broadcast":
            return output_dir / "camera" / (
                f"{shot}_camera_track.json" if shot else "camera_track.json")
        return output_dir / "render" / shot_dir / "cameras" / f"{cam_id}_camera_track.json"

    broadcast_path = _camera_track_path("broadcast")
    fps = DEFAULT_FPS
    if broadcast_path.exists():
        fps = float(load_camera_track(broadcast_path).get("fps", DEFAULT_FPS)) or DEFAULT_FPS

    for cam_id in args.cameras:
        cam_path = _camera_track_path(cam_id)
        if not cam_path.exists():
            sys.stderr.write(
                f"[render] camera track not found for '{cam_id}' at {cam_path}\n"
            )
            return 2
        track = load_camera_track(cam_path)
        frames = track.get("frames", [])
        if not frames:
            sys.stderr.write(f"[render] camera track '{cam_id}' has no frames\n")
            return 2
        frame_start = (
            args.frame_start if args.frame_start is not None else int(frames[0]["frame"])
        )
        frame_end = (
            args.frame_end if args.frame_end is not None else int(frames[-1]["frame"])
        )

        cam_obj = _add_camera_from_track(cam_id, track, args.width, args.height)

        if args.save_blend:
            bpy.ops.wm.save_as_mainfile(filepath=str(out_dir / "scene.blend"))

        _render(cam_obj, out_dir / f"{cam_id}.mp4", fps,
                (frame_start, frame_end), args.width, args.height, args.samples)

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
