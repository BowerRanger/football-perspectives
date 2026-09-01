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
(``_build_environment``, ``_build_ball``, ``_build_players``,
``_add_camera_from_track``, ``_render``) are nested inside ``main()``
since they close over the lazily-imported ``bpy``/``bmesh``/
``mathutils`` modules; later tasks (toon materials, virtual-camera
renders, vertical/AOV variants) extend those nested functions in place.
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

# --- Player constants ------------------------------------------------------
# Fixed skin tone (not team-configurable) — declared once here, linearised
# via render_look.hex_to_linear_rgba wherever a material needs it.
SKIN_COLOR_HEX = "#c68863"
# The 4 distinct zones `render_look.kit_zone_for_height_fraction` returns
# (its "skin" zone covers both legs and head/neck rest-height bands).
_KIT_ZONES = ("socks", "skin", "shorts", "shirt")
# Axial spine-chain bones get a thicker capsule than limb bones in the
# no-SMPL-asset fallback body.
_SPINE_BONES = frozenset({"pelvis", "spine1", "spine2", "spine3", "neck"})
_LIMB_CAPSULE_RADIUS_M = 0.055
_SPINE_CAPSULE_RADIUS_M = 0.10
_HEAD_SPHERE_RADIUS_M = 0.09
_MIN_CAPSULE_BONE_LEN_M = 0.02
# Unclassified players (no tracks/*_tracks.json team label) still need a
# renderable kit — mirrors render_look.resolve_player_colors' own internal
# fallback so a missing team classification never crashes the build.
_FALLBACK_KIT_HEX = {"shirt": "#888888", "shorts": "#666666", "socks": "#888888"}

# --- Toon-look constants ----------------------------------------------
# Ball kit is not team-configurable (single shared texture) — same
# pattern as SKIN_COLOR_HEX above.
BALL_COLOR_HEX = "#f2f2f2"
# Blob-shadow disc radii (Task 7 brief / controller ruling): per-player
# large enough to read under a standing figure, ball tighter to its
# smaller footprint.
PLAYER_SHADOW_RADIUS_M = 0.4
BALL_SHADOW_RADIUS_M = 0.15

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
    import numpy as np
    from math import radians

    from mathutils import Matrix, Quaternion, Vector  # type: ignore

    from src.utils import render_look
    from src.utils.blender_scene_io import (
        iter_player_fbx_entries,
        load_camera_track,
        load_smpl_body_data,
        prepare_ball_keys,
    )
    from src.utils.smpl_skeleton import (
        SMPL_JOINT_NAMES,
        SMPL_PARENTS,
        SMPL_REST_JOINTS_YUP,
        axis_angle_to_quaternion,
    )
    from src.stages.export import _player_team_class_map

    # --vertical is implemented in the render loop below (Task 8): a
    # second pass per non-broadcast camera with resolution_x/y swapped
    # and sensor_fit="VERTICAL" — reframes to 9:16 portrait without
    # touching the camera's keyed lens values. --aov (Task 9) is wired
    # in _setup_aov_compositor/_render below.

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

    # --- Toon materials, outlines, blob shadows -------------------------
    # Cel-shaded look (Task 7): Diffuse -> Shader-to-RGB -> constant
    # ColorRamp -> Emission quantises lighting into `ramp_steps` bands.
    # ShaderNodeShaderToRGB is EEVEE-only; guarded here (rather than
    # assumed) since a Cycles-only Blender build would otherwise raise on
    # node creation — falls back to flat Emission and prints
    # TOON_FALLBACK_FLAT once. Confirmed present on the Blender 5.1.1
    # build this task runs against.
    _shader_to_rgb_available = hasattr(bpy.types, "ShaderNodeShaderToRGB")
    _toon_material_count = 0
    _outline_count = 0
    _printed_toon_fallback = False

    def _toon_material(name: str, rgba, ramp_steps: int) -> object:
        """Diffuse -> Shader-to-RGB -> constant ColorRamp -> Emission.

        The ramp quantises lighting into ``ramp_steps`` bands (classic cel
        shading). Emission output keeps the bands flat and print-like.
        """
        nonlocal _toon_material_count, _printed_toon_fallback
        mat = bpy.data.materials.new(name)
        mat.use_nodes = True
        nt = mat.node_tree
        nt.nodes.clear()
        out = nt.nodes.new("ShaderNodeOutputMaterial")

        if not _shader_to_rgb_available:
            if not _printed_toon_fallback:
                print("TOON_FALLBACK_FLAT")
                _printed_toon_fallback = True
            emit = nt.nodes.new("ShaderNodeEmission")
            emit.inputs["Color"].default_value = rgba
            nt.links.new(emit.outputs["Emission"], out.inputs["Surface"])
            _toon_material_count += 1
            return mat

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
        # Blender 5.1.1 adaptation: ValToRGB's factor input socket is
        # named "Factor", not "Fac" as in the brief's snippet — renamed in
        # Blender's node socket-name pass. (MixRGB's "Fac" input used
        # elsewhere in this file is a different node type and unaffected;
        # verified both against the running Blender before wiring this.)
        nt.links.new(to_rgb.outputs["Color"], ramp.inputs["Factor"])
        nt.links.new(ramp.outputs["Color"], emit.inputs["Color"])
        nt.links.new(emit.outputs["Emission"], out.inputs["Surface"])
        _toon_material_count += 1
        return mat

    def _add_outline(obj: object, width_m: float, rgba) -> None:
        """Inverted-hull outline: Solidify with flipped normals +
        backface-culled emission black shell."""
        nonlocal _outline_count
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
        _outline_count += 1

    def _add_blob_shadow(target_obj: object, radius_m: float) -> object:
        """Soft dark disc at z=0.01 following the target's XY (drivers)."""
        bpy.ops.mesh.primitive_circle_add(
            vertices=24, radius=radius_m,
            # Blender 5.1.1 adaptation: the operator's fill kwarg is
            # `fill_type`, not `fill_mode` as in the brief's snippet
            # (confirmed via bpy.ops.mesh.primitive_circle_add's rna
            # property list before wiring this in); "NGON" is unchanged.
            fill_type="NGON")
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

    # --- Players ---------------------------------------------------------
    # Armature recipe mirrors scripts/blender_export_fbx.py's docstring
    # convention: canonical rest pose built in y-up axes (no pre-rotation —
    # the per-frame armature object matrix does the canonical->pitch-world
    # mapping), pose-bone quaternions from thetas[1:] (pelvis/thetas[0] is
    # IGNORED — root_R carries the root world orientation; see
    # src/utils/smpl_skeleton.py's compute_joint_world_pose docstring).

    _fallback_kit_rgba = {
        part: render_look.hex_to_linear_rgba(hexval)
        for part, hexval in _FALLBACK_KIT_HEX.items()
    }
    _skin_rgba = render_look.hex_to_linear_rgba(SKIN_COLOR_HEX)

    def _bone_children_map() -> dict:
        children: dict = {j: [] for j in range(24)}
        for j in range(1, 24):
            children[SMPL_PARENTS[j]].append(j)
        return children

    def _bone_rest_endpoints(rest_joints, children: dict) -> list:
        """(head, tail) per bone in canonical rest space.

        Internal bones (with children) get a tail at the mean of their
        children's rest positions; leaf bones (hands, feet, head) get a
        fixed +0.05 z tail so they stay visible. Same table drives both
        the armature's edit bones and the capsule-fallback geometry.
        """
        endpoints = []
        for j in range(24):
            head = np.asarray(rest_joints[j], dtype=np.float64)
            kids = children[j]
            if kids:
                tail = np.mean(
                    [np.asarray(rest_joints[k], dtype=np.float64) for k in kids],
                    axis=0)
            else:
                tail = head + np.array([0.0, 0.0, 0.05])
            endpoints.append((head, tail))
        return endpoints

    def _material_for(materials_cache: dict, colors: dict, pid: str,
                       zone: str, ramp_steps: int) -> object:
        key = (pid, zone)
        mat = materials_cache.get(key)
        if mat is not None:
            return mat
        if zone == "skin":
            rgba = _skin_rgba
        else:
            kit = colors.get(pid) or _fallback_kit_rgba
            rgba = kit.get(zone, _fallback_kit_rgba[zone])
        mat = _toon_material(f"{pid}_{zone}", rgba, ramp_steps)
        materials_cache[key] = mat
        return mat

    def _height_fraction(y: float, y_min: float, y_max: float) -> float:
        span = (y_max - y_min) or 1.0
        return (y - y_min) / span

    def _add_smpl_mesh_body(arm: object, pid: str, smpl_data, colors: dict,
                             materials_cache: dict, ramp_steps: int) -> object:
        """Full SMPL body mesh skinned to all 24 joints — mirrors
        blender_export_fbx.py's _add_smpl_skinned_mesh, plus per-face kit
        material slots (majority zone of the face's 3 vertices)."""
        v_template = smpl_data["v_template"]
        faces = smpl_data["faces"]
        weights = smpl_data["weights"]

        mesh = bpy.data.meshes.new(f"{pid}_smpl_mesh")
        verts = [(float(v[0]), float(v[1]), float(v[2])) for v in v_template]
        face_list = [(int(t[0]), int(t[1]), int(t[2])) for t in faces]
        mesh.from_pydata(verts, [], face_list)
        mesh.update()
        obj = bpy.data.objects.new(f"{pid}_body", mesh)
        bpy.context.collection.objects.link(obj)
        obj.parent = arm

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

        y = v_template[:, 1].astype(float)
        y_min, y_max = float(y.min()), float(y.max())
        vertex_zones = [
            render_look.kit_zone_for_height_fraction(_height_fraction(v, y_min, y_max))
            for v in y
        ]
        slot_index = {}
        for zone in _KIT_ZONES:
            obj.data.materials.append(
                _material_for(materials_cache, colors, pid, zone, ramp_steps))
            slot_index[zone] = len(obj.data.materials) - 1
        for poly in mesh.polygons:
            counts: dict = {}
            for vi in poly.vertices:
                z = vertex_zones[vi]
                counts[z] = counts.get(z, 0) + 1
            majority = max(counts.items(), key=lambda kv: kv[1])[0]
            poly.material_index = slot_index[majority]
        return obj

    def _parent_to_bone(obj: object, arm: object, bone_name: str) -> None:
        """Bone-parent ``obj`` to ``bone_name`` while preserving its
        current world transform (the scripted equivalent of Ctrl-P > Bone,
        Keep Transform). Must run while the armature is still at rest
        (identity object transform, identity pose) — capsule bodies are
        built before any per-frame posing in ``_build_players``.
        """
        obj.parent = arm
        obj.parent_type = "BONE"
        obj.parent_bone = bone_name
        pbone = arm.pose.bones[bone_name]
        bone_len = arm.data.bones[bone_name].length
        tail_mat = (
            arm.matrix_world @ pbone.matrix @ Matrix.Translation((0.0, bone_len, 0.0))
        )
        obj.matrix_parent_inverse = tail_mat.inverted()

    def _add_capsule_body(arm: object, pid: str, endpoints: list, colors: dict,
                           materials_cache: dict, ramp_steps: int) -> list:
        """Capsule-limb fallback body: one primitive per bone with a
        non-degenerate rest length, bone-parented so it follows the
        armature's per-frame pose."""
        y_all = np.array(
            [pt[1] for head, tail in endpoints for pt in (head, tail)])
        y_min, y_max = float(y_all.min()), float(y_all.max())
        objs = []
        for j, jname in enumerate(SMPL_JOINT_NAMES):
            head, tail = endpoints[j]
            direction = tail - head
            length = float(np.linalg.norm(direction))
            if length <= _MIN_CAPSULE_BONE_LEN_M:
                continue
            mid = (head + tail) / 2.0
            zone = render_look.kit_zone_for_height_fraction(
                _height_fraction(float(mid[1]), y_min, y_max))
            mat = _material_for(materials_cache, colors, pid, zone, ramp_steps)

            z_axis = Vector((0.0, 0.0, 1.0))
            dir_vec = Vector((float(direction[0]), float(direction[1]),
                               float(direction[2])))
            quat = z_axis.rotation_difference(dir_vec)

            if jname == "head":
                bpy.ops.mesh.primitive_uv_sphere_add(radius=_HEAD_SPHERE_RADIUS_M)
            elif jname in _SPINE_BONES:
                bpy.ops.mesh.primitive_cylinder_add(
                    radius=_SPINE_CAPSULE_RADIUS_M, depth=length)
            else:
                bpy.ops.mesh.primitive_cylinder_add(
                    radius=_LIMB_CAPSULE_RADIUS_M, depth=length)
            obj = bpy.context.active_object
            obj.name = f"{pid}_{jname}_capsule"
            obj.rotation_mode = "QUATERNION"
            obj.location = (float(mid[0]), float(mid[1]), float(mid[2]))
            obj.rotation_quaternion = quat
            obj.data.materials.append(mat)

            _parent_to_bone(obj, arm, jname)
            objs.append(obj)
        return objs

    def _build_players(output_dir: Path, shot_id: str, colors: dict,
                        smpl_data, pelvis_canon, style: dict) -> list:
        rest_joints = (
            np.asarray(smpl_data["joint_positions"], dtype=np.float64)
            if smpl_data is not None
            else SMPL_REST_JOINTS_YUP
        )
        endpoints = _bone_rest_endpoints(rest_joints, _bone_children_map())
        materials_cache: dict = {}
        armatures: list = []
        ramp_steps = style["ramp_steps"]
        outline_width = style["outline_width_m"]
        outline_rgba = render_look.hex_to_linear_rgba(style["palette"]["outline"])

        for entry in iter_player_fbx_entries(output_dir, np):
            if entry["shot_id"] not in ("", shot_id):
                continue
            pid = entry["player_id"]
            frames = np.asarray(entry["frames"])
            thetas = np.asarray(entry["thetas"])
            root_R = np.asarray(entry["root_R"])
            root_t = np.asarray(entry["root_t"])
            n_frames = int(frames.shape[0])
            if n_frames == 0:
                continue

            bpy.ops.object.armature_add(enter_editmode=True)
            arm = bpy.context.active_object
            arm.name = f"{pid}_arm"
            arm.data.name = f"{pid}_arm_data"
            arm.rotation_mode = "QUATERNION"
            edit_bones = arm.data.edit_bones
            for eb in list(edit_bones):
                edit_bones.remove(eb)
            bones = []
            for j, jname in enumerate(SMPL_JOINT_NAMES):
                eb = edit_bones.new(jname)
                head, tail = endpoints[j]
                eb.head = (float(head[0]), float(head[1]), float(head[2]))
                eb.tail = (float(tail[0]), float(tail[1]), float(tail[2]))
                if SMPL_PARENTS[j] != -1:
                    eb.parent = bones[SMPL_PARENTS[j]]
                    eb.use_connect = False
                bones.append(eb)
            bpy.ops.object.mode_set(mode="OBJECT")
            for pb in arm.pose.bones:
                pb.rotation_mode = "QUATERNION"

            # Body — built while the armature is still at rest (identity
            # object transform, identity pose), then posed below. Every
            # body part gets an inverted-hull outline (Task 7 toon look).
            if smpl_data is not None:
                body_objs = [_add_smpl_mesh_body(
                    arm, pid, smpl_data, colors, materials_cache, ramp_steps)]
            else:
                body_objs = _add_capsule_body(
                    arm, pid, endpoints, colors, materials_cache, ramp_steps)
            for body_obj in body_objs:
                _add_outline(body_obj, outline_width, outline_rgba)

            for i, fi in enumerate(frames.tolist()):
                fr = int(fi)
                # Joints 1..23 only — pelvis (bone 0) stays IDENTITY.
                # thetas[i, 0] is ignored: root_R carries the root's world
                # orientation (repo convention — see smpl_skeleton.py).
                for j in range(1, 24):
                    pb = arm.pose.bones[SMPL_JOINT_NAMES[j]]
                    q = axis_angle_to_quaternion(thetas[i, j])
                    pb.rotation_quaternion = Quaternion(
                        (float(q[0]), float(q[1]), float(q[2]), float(q[3])))
                    pb.keyframe_insert(data_path="rotation_quaternion", frame=fr)

                R = root_R[i]
                # Translation accounts for the foot-midpoint canonical
                # re-anchor: pelvis_canon is the shifted canonical pelvis
                # position (zero when smpl_data is None — verified
                # SMPL_REST_JOINTS_YUP[0] == (0, 0, 0) — so this formula
                # is exact in both the asset and fallback cases).
                offset = R @ pelvis_canon
                loc = root_t[i] - offset
                arm.location = (float(loc[0]), float(loc[1]), float(loc[2]))
                m = Matrix((
                    (float(R[0, 0]), float(R[0, 1]), float(R[0, 2]), 0.0),
                    (float(R[1, 0]), float(R[1, 1]), float(R[1, 2]), 0.0),
                    (float(R[2, 0]), float(R[2, 1]), float(R[2, 2]), 0.0),
                    (0.0, 0.0, 0.0, 1.0),
                ))
                arm.rotation_quaternion = m.to_quaternion()
                arm.keyframe_insert(data_path="location", frame=fr)
                arm.keyframe_insert(data_path="rotation_quaternion", frame=fr)

            armatures.append(arm)
            _add_blob_shadow(arm, PLAYER_SHADOW_RADIUS_M)

        print(f"PLAYERS_BUILT {len(armatures)}")
        return armatures

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

    # --- AOV (Task 9) -----------------------------------------------------
    # Blender 5.1.1 adaptation: the pre-5.x compositor API this brief was
    # written against (`scene.use_nodes = True` + `scene.node_tree`) no
    # longer exists — `scene.node_tree` raises AttributeError, and
    # `scene.use_nodes` is a deprecated no-op stub (warns, slated for
    # removal in 6.0) that does NOT create or assign a node group. The
    # compositor now lives behind `scene.compositing_node_group`, a plain
    # `CompositorNodeTree` datablock (`bpy.data.node_groups.new(name,
    # "CompositorNodeTree")`) that must be explicitly assigned to the
    # scene. The File Output node's socket API changed too — no
    # `file_slots`/`layer_slots`/`base_path` — sockets are declared via
    # `node.file_output_items.new(socket_type, name)` where `socket_type`
    # is a fixed enum (RGBA/FLOAT/VECTOR/...), which creates a same-named
    # input socket to link into. The Render Layers node's Z-pass output
    # socket is named "Depth", not "Z". All verified against the running
    # Blender via a probe script (raw EXR-header dump of a real render)
    # before wiring any of this — see task-9-report.md.
    _AOV_PASSES = (
        ("RGBA", "Image"),
        ("FLOAT", "Depth"),
        ("VECTOR", "Normal"),
        ("RGBA", "CryptoObject00"),
        ("RGBA", "CryptoObject01"),
        ("RGBA", "CryptoObject02"),
    )
    _aov_file_output_node = None

    def _setup_aov_compositor() -> object:
        """Enable Z/Normal/Cryptomatte view-layer passes and wire a
        multilayer-EXR File Output node fed by the Render Layers node.

        Built once per process (cached in ``_aov_file_output_node``) since
        the graph itself doesn't vary per camera — only the File Output
        node's ``directory`` changes, which callers set afterwards.
        """
        nonlocal _aov_file_output_node
        if _aov_file_output_node is not None:
            return _aov_file_output_node
        vl = bpy.context.view_layer
        vl.use_pass_z = True
        vl.use_pass_normal = True
        vl.use_pass_cryptomatte_object = True
        group = bpy.data.node_groups.new("Compositing", "CompositorNodeTree")
        bpy.context.scene.compositing_node_group = group
        rl = group.nodes.new("CompositorNodeRLayers")
        rl.layer = vl.name
        fo = group.nodes.new("CompositorNodeOutputFile")
        for socket_type, pass_name in _AOV_PASSES:
            fo.file_output_items.new(socket_type, pass_name)
        for _, pass_name in _AOV_PASSES:
            group.links.new(rl.outputs[pass_name], fo.inputs[pass_name])
        if hasattr(fo.format, "media_type"):
            fo.format.media_type = "MULTI_LAYER_IMAGE"
        fo.format.file_format = "OPEN_EXR_MULTILAYER"
        fo.file_name = "####"
        _aov_file_output_node = fo
        return fo

    # --- Render ----------------------------------------------------------

    def _render(camera_obj: object, out_path: Path, fps: float,
                frame_range: tuple[int, int], width: int, height: int,
                samples: int, aov_dir: Path | None = None) -> None:
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
        # the exact path passed in. This scene-level flag also governs
        # whether the AOV File Output node below appends its own format
        # extension (".exr") — rather than flip it per-render (risking
        # the main-output filename regression this comment describes),
        # the AOV frames are renamed to add ".exr" after the render
        # completes, below.
        scene.render.use_file_extension = False
        out_path.parent.mkdir(parents=True, exist_ok=True)
        scene.render.filepath = str(out_path)

        # `scene.compositing_node_group` (once assigned by
        # _setup_aov_compositor, below) stays attached to the scene for
        # every subsequent render call — Blender doesn't clear it, and
        # `scene.render.use_compositing` (a separate flag, default True)
        # is never touched elsewhere, so the compositor keeps running on
        # every later render regardless of this call's own `aov_dir`. Left
        # ungated, a landscape pass with `--aov` "poisons" every following
        # render (e.g. the 9:16 vertical pass, or a next camera without
        # AOV) into re-firing the File Output node at its stale directory
        # and stale (wrong-resolution) frame — and since the rename-to-
        # `.exr` loop below is itself gated on `aov_dir`, that stray write
        # is left as a corrupt extension-less file. Explicitly gating
        # `use_compositing` on *this* call's `aov_dir` on every call (not
        # just when AOV is requested) is what actually scopes the
        # compositor to the calls that asked for it.
        scene.render.use_compositing = aov_dir is not None
        if aov_dir is not None:
            fo = _setup_aov_compositor()
            aov_dir.mkdir(parents=True, exist_ok=True)
            fo.directory = str(aov_dir) + "/"

        t0 = time.time()
        bpy.ops.render.render(animation=True)
        elapsed = time.time() - t0
        n_frames = frame_range[1] - frame_range[0] + 1
        # Eyeballing aid for the stage log; the quality report parses
        # render/render_timings.json (written by RenderStage) instead.
        print(f"RENDER_TIMING {camera_obj.name} {elapsed:.2f} {n_frames}")

        if aov_dir is not None:
            for fr in range(frame_range[0], frame_range[1] + 1):
                raw = aov_dir / f"{fr:04d}"
                if raw.exists():
                    raw.rename(raw.with_suffix(".exr"))
            print(f"AOV_RENDER {camera_obj.name} {aov_dir}")

    # --- Orchestration ---------------------------------------------------

    _build_environment(style)

    outline_rgba = render_look.hex_to_linear_rgba(style["palette"]["outline"])

    ball_path = (
        output_dir / "ball" / (f"{shot}_ball_track.json" if shot else "ball_track.json")
    )
    if ball_path.exists():
        ball_raw = json.loads(ball_path.read_text())
        ball_keys = prepare_ball_keys(ball_raw.get("frames", []))
        if ball_keys:
            ball_obj = _build_ball(ball_keys)
            ball_obj.data.materials.append(_toon_material(
                "M_Ball", render_look.hex_to_linear_rgba(BALL_COLOR_HEX),
                style["ramp_steps"]))
            _add_outline(ball_obj, style["outline_width_m"], outline_rgba)
            _add_blob_shadow(ball_obj, BALL_SHADOW_RADIUS_M)
    else:
        sys.stdout.write(f"[render] no ball track at {ball_path}; skipping ball\n")

    team_class = _player_team_class_map(output_dir)
    player_colors = render_look.resolve_player_colors(
        style.get("teams", {}) or {}, team_class)
    smpl_data, pelvis_canon = load_smpl_body_data(_REPO_ROOT, np)
    if smpl_data is not None:
        sys.stdout.write(
            f"[render] using real SMPL body mesh (pelvis canon = "
            f"{tuple(float(x) for x in pelvis_canon)})\n")
    else:
        sys.stdout.write(
            "[render] no SMPL body asset at data/models/smpl_neutral.npz; "
            "using capsule-limb fallback bodies\n")
    _build_players(output_dir, shot, player_colors, smpl_data, pelvis_canon, style)

    print(f"TOON_MATERIALS {_toon_material_count}")
    print(f"OUTLINES {_outline_count}")

    def _safe_cam_id(cam_id: str) -> str:
        # Matches RenderStage._write_virtual_camera_tracks's safe_id
        # (src/stages/render.py): ":" -> "_" so player-scoped ids like
        # "pov:P001" become filesystem/ffmpeg-safe "pov_P001".
        return cam_id.replace(":", "_")

    def _camera_track_path(cam_id: str) -> Path:
        if cam_id == "broadcast":
            return output_dir / "camera" / (
                f"{shot}_camera_track.json" if shot else "camera_track.json")
        return (output_dir / "render" / shot_dir / "cameras"
                / f"{_safe_cam_id(cam_id)}_camera_track.json")

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
        safe_id = _safe_cam_id(cam_id)

        if args.save_blend:
            bpy.ops.wm.save_as_mainfile(filepath=str(out_dir / "scene.blend"))

        # AOV EXRs (Task 9) are only rendered for the landscape pass, one
        # subdirectory per camera — never for the 9:16 vertical pass below.
        aov_dir = (out_dir / "aov" / safe_id) if args.aov else None
        _render(cam_obj, out_dir / f"{safe_id}.mp4", fps,
                (frame_start, frame_end), args.width, args.height, args.samples,
                aov_dir=aov_dir)

        # 9:16 portrait pass (Task 8): every non-broadcast camera gets a
        # second render at swapped (height, width) resolution. Reframed
        # via sensor_fit="VERTICAL" rather than scaling cam.lens — the
        # lens values are keyframed (see _add_camera_from_track), so
        # scaling them would require re-keying every frame; sensor_fit
        # instead changes how the *existing* keyed lens maps to FOV for
        # this pass only, then is restored to "HORIZONTAL" for the next
        # camera's landscape render.
        if args.vertical and cam_id != "broadcast":
            cam_obj.data.sensor_fit = "VERTICAL"
            _render(cam_obj, out_dir / f"{safe_id}_9x16.mp4", fps,
                    (frame_start, frame_end), args.height, args.width, args.samples)
            cam_obj.data.sensor_fit = "HORIZONTAL"

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
