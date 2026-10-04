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
since they close over the lazily-imported ``bpy``/
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

from src.utils import render_look  # noqa: E402
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
BALL_RADIUS_M = 0.11
SENSOR_WIDTH_MM = 36.0
DEFAULT_FPS = 25.0
DEFAULT_SUN_ROTATION_DEG = (50.0, 0.0, -30.0)
DEFAULT_SUN_ENERGY = 3.0

# --- Player constants ------------------------------------------------------
# Fixed skin tone (not team-configurable) — declared once here, linearised
# via render_look.hex_to_linear_rgba wherever a material needs it.
SKIN_COLOR_HEX = "#c68863"  # == render_look.DEFAULT_SKIN_HEX
# Axial spine-chain bones get a thicker capsule than limb bones in the
# no-SMPL-asset fallback body.
_SPINE_BONES = frozenset({"pelvis", "spine1", "spine2", "spine3", "neck"})
_LIMB_CAPSULE_RADIUS_M = 0.055
_SPINE_CAPSULE_RADIUS_M = 0.10
_HEAD_SPHERE_RADIUS_M = 0.09
_MIN_CAPSULE_BONE_LEN_M = 0.02
# Every armature EDIT bone gets this fixed +Y rest tail (uniform
# direction, zero roll) — see the bone-space-bug note above
# ``_bone_rest_endpoints`` and ``blender_export_fbx.py:215-217``, whose
# comment this mirrors verbatim ("Direction is irrelevant for FBX export
# of pose-bone rotations"). Because every bone's rest orientation is then
# identical (== the armature/canonical frame), pose-bone
# ``rotation_quaternion`` values (SMPL thetas, expressed in that same
# canonical frame) apply correctly. Also used to cancel the same fixed
# offset when bone-parenting capsule geometry (Blender's BONE parent
# type anchors children at the bone's TAIL, not its head).
_ARMATURE_BONE_TAIL_M = 0.05
# Unclassified players (no tracks/*_tracks.json team label) still need a
# renderable kit — mirrors render_look.resolve_player_colors' own internal
# fallback so a missing team classification never crashes the build.
_FALLBACK_KIT_HEX = {"shirt": "#888888", "shorts": "#666666", "socks": "#888888",
                     "boots": "#1c1c1c", "gloves": SKIN_COLOR_HEX,
                     "hair": "#2b1d14"}

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
# still produce a fully populated style). Extended (render-experiments
# task) with sun/world/lines look knobs and the `post` compositor-effects
# block — every new key's default here reproduces the exact hardcoded
# value the script used before this task (DEFAULT_SUN_ENERGY,
# DEFAULT_SUN_ROTATION_DEG, and the implicit 1.0 Blender defaults for
# world Background strength / emission Strength), and every `post`
# sub-effect defaults to its own no-op value — see
# render_look.post_style_is_active, which a fully-defaulted `post` block
# must report as inactive.
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
    "sun_energy": DEFAULT_SUN_ENERGY,
    "sun_rotation_deg": DEFAULT_SUN_ROTATION_DEG,
    "world_strength": 1.0,
    "lines_emission_strength": 1.0,
    # "height_bands" (legacy v1 rest-height kit bands) or "anatomical"
    # (skinning-weight garments: sleeves, boots, gloves, hair —
    # render_look.anatomical_kit_zones). config/default.yaml ships
    # "anatomical"; the bare-style default keeps the v1 look.
    "body_zones": "height_bands",
    "stadium": {"enabled": True, "roof": True, "crowd_density": 0.65,
                "seat_color": "#294b65", "accent_color": "#d6b66e"},
    "post": {
        "glare": 0.0,
        "grain": 0.0,
        "vignette": 0.0,
        "posterize": 0,
        "duotone": {"shadow": None, "highlight": None},
        "saturation": 1.0,
    },
}


def _resolve_style(style: dict) -> dict:
    """Merge a (possibly partial) style dict over the defaults above.

    Pure — no ``bpy`` — so it's unit-testable on its own. Top-level
    keys and the nested ``palette``/``post``/``post.duotone`` dicts are
    merged independently (via ``render_look.merge_partial``) so a
    caller can override e.g. only ``post.grain`` without having to
    repeat every sibling key, at any nesting level.
    """
    merged = render_look.merge_partial(
        {k: v for k, v in _DEFAULT_STYLE.items() if k not in ("palette", "post")},
        {k: v for k, v in style.items() if k not in ("palette", "post")},
    )
    merged["palette"] = render_look.merge_partial(
        _DEFAULT_STYLE["palette"], style.get("palette"))

    post_defaults = _DEFAULT_STYLE["post"]
    post_in = style.get("post") or {}
    post = render_look.merge_partial(
        {k: v for k, v in post_defaults.items() if k != "duotone"},
        {k: v for k, v in post_in.items() if k != "duotone"},
    )
    post["duotone"] = render_look.merge_partial(
        post_defaults["duotone"], post_in.get("duotone"))
    merged["post"] = post
    merged["stadium"] = render_look.merge_partial(
        _DEFAULT_STYLE["stadium"], style.get("stadium"))
    return merged


# Per-vertex rest-pose position attribute baked on the SMPL body mesh; kit
# pattern materials read it so stripes/hoops follow the skinned torso.
REST_CO_ATTRIBUTE = "rest_co"


def _parse_args(argv: list[str]) -> argparse.Namespace:
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    p = argparse.ArgumentParser(description="Toon render of one shot")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--shot", default="")
    p.add_argument(
        "--render-root", default="render",
        help="Path component substituted for 'render' in both the render "
             "output dir (output-dir/<render-root>/<shot>/...) and the "
             "non-broadcast camera-track lookup dir (same location, "
             ".../cameras/). Default 'render' reproduces today's exact "
             "paths; render-experiments passes a distinct value so a "
             "matrix run never clobbers the protected baseline under "
             "output/render/<shot>/.")
    p.add_argument("--cameras", type=lambda s: s.split(","),
                    default=["broadcast"])
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--height", type=int, default=1080)
    p.add_argument("--samples", type=int, default=16)
    p.add_argument("--style-json", default="{}")
    p.add_argument("--vertical", action="store_true")
    p.add_argument(
        "--vertical-only", action="store_true",
        help="Render ONLY the 9:16 portrait pass per camera (skips the "
             "landscape pass and AOVs) - for shorts where the landscape "
             "mp4 is never used.")
    p.add_argument(
        "--allow-capsule-fallback", action="store_true",
        help="Accept capsule-limb bodies when the SMPL body asset "
             "(data/models/smpl_neutral.npz) is missing/unusable. Without "
             "this flag a missing asset exits non-zero rather than "
             "silently rendering the wrong bodies.")
    p.add_argument("--aov", action="store_true")
    p.add_argument("--save-blend", action="store_true")
    p.add_argument("--frame-start", type=int, default=None)
    p.add_argument("--frame-end", type=int, default=None)
    p.add_argument(
        "--time-stretch", type=int, default=1,
        help="Render-native slow motion: Blender time-stretching "
             "(frame_map_old/new) renders N interpolated frames per source "
             "frame, so the mp4 plays N x slower at the same fps with real "
             "in-between poses — unlike post-hoc ffmpeg setpts/minterpolate. "
             "1-9 (Blender caps frame_map_new at 900).")
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

    if tuple(bpy.app.version) < (5, 0, 0):
        sys.stderr.write(
            "Blender >= 5.0 required (this script hard-requires 5.x-only "
            "APIs: the compositor's scene.compositing_node_group, the "
            "image-settings media_type enum, and the ColorRamp 'Factor' "
            f"socket name), got {bpy.app.version}\n"
        )
        return 2

    import numpy as np
    from math import radians

    from mathutils import Matrix, Quaternion, Vector  # type: ignore

    from src.utils.blender_scene_io import (
        iter_player_fbx_entries,
        load_camera_track,
        load_smpl_body_data,
        shape_smpl_body_data,
        prepare_ball_keys,
    )
    from src.utils.smpl_skeleton import (
        SMPL_JOINT_NAMES,
        SMPL_PARENTS,
        SMPL_REST_JOINTS_YUP,
        axis_angle_to_quaternion,
    )
    from src.stages.export import _player_team_class_map
    from src.utils.player_names import load_kit_roles, load_player_appearance

    # --vertical is implemented in the render loop below (Task 8): a
    # second pass per non-broadcast camera with resolution_x/y swapped
    # and sensor_fit="VERTICAL" — reframes to 9:16 portrait without
    # touching the camera's keyed lens values. --aov (Task 9) is wired
    # in _setup_aov_compositor/_render below.

    output_dir = Path(args.output_dir).resolve()
    shot = args.shot
    # --render-root substitutes the "render" path component below (and
    # in _camera_track_path's non-broadcast branch, further down) so a
    # render-experiments matrix run can redirect its entire output tree
    # — including where it looks up virtual-camera tracks — away from
    # the protected output/render/<shot>/ baseline. Default "render"
    # reproduces every path byte-identically.
    render_root = args.render_root
    # Legacy empty shot id renders under a fixed "clip" directory name —
    # keeps output/render/<dir>/<camera>.mp4 stable for single-shot runs.
    shot_dir = shot or "clip"
    out_dir = output_dir / render_root / shot_dir
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
        noise = nt.nodes.new("ShaderNodeTexNoise")
        noise.inputs["Scale"].default_value = 2.5
        noise.inputs["Detail"].default_value = 2.0
        nt.links.new(texc.outputs["Object"], noise.inputs["Vector"])
        variation = nt.nodes.new("ShaderNodeMapRange")
        variation.inputs["To Min"].default_value = 0.88
        variation.inputs["To Max"].default_value = 1.04
        nt.links.new(noise.outputs["Fac"], variation.inputs["Value"])
        tint = nt.nodes.new("ShaderNodeMixRGB")
        tint.blend_type = "MULTIPLY"
        tint.inputs["Fac"].default_value = 1.0
        nt.links.new(mix.outputs["Color"], tint.inputs["Color1"])
        nt.links.new(variation.outputs["Result"], tint.inputs["Color2"])
        nt.links.new(tint.outputs["Color"], diffuse.inputs["Color"])
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

    def _build_pitch_details(lines_mat):
        from math import acos, cos, sin, pi
        def arc(name, x, y, radius, start, end):
            pts = [(x+radius*cos(start+(end-start)*i/64),
                    y+radius*sin(start+(end-start)*i/64)) for i in range(65)]
            _line_object(name, pts, lines_mat)
        theta = acos((PENALTY_BOX_DEPTH_M-11)/CENTRE_CIRCLE_R)
        arc("L_PenaltyArc_Left",11,PITCH_WIDTH/2,CENTRE_CIRCLE_R,-theta,theta)
        arc("L_PenaltyArc_Right",PITCH_LENGTH-11,PITCH_WIDTH/2,
            CENTRE_CIRCLE_R,pi-theta,pi+theta)
        for x,y,start in [(0,0,0),(PITCH_LENGTH,0,pi/2),
                          (PITCH_LENGTH,PITCH_WIDTH,pi),(0,PITCH_WIDTH,3*pi/2)]:
            arc("L_CornerArc",x,y,1.,start,start+pi/2)
        for x in (11,PITCH_LENGTH/2,PITCH_LENGTH-11):
            bpy.ops.mesh.primitive_circle_add(vertices=24, radius=0.11,
                fill_type="NGON", location=(x,PITCH_WIDTH/2,LINE_Z))
            bpy.context.object.name = "L_Spot"
            bpy.context.object.data.materials.append(lines_mat)

    def _build_nets(lines_mat):
        net_mat = _new_diffuse_material("M_GoalNet", (0.58,0.64,0.62,1))
        for x, sign in [(0.,-1.),(PITCH_LENGTH,1.)]:
            curve = bpy.data.curves.new("GoalNet", "CURVE")
            curve.dimensions = "3D"
            curve.bevel_depth = 0.009
            curve.bevel_resolution = 0
            def strand(points):
                sp = curve.splines.new("POLY")
                sp.points.add(len(points)-1)
                for p, co in zip(sp.points, points):
                    p.co = (*co,1)
            y0,y1 = PITCH_WIDTH/2-GOAL_HALF_WIDTH_M,PITCH_WIDTH/2+GOAL_HALF_WIDTH_M
            back=x+sign*2.0
            for y in np.linspace(y0,y1,42):
                strand([(x,y,GOAL_HEIGHT_M),(back,y,GOAL_HEIGHT_M),(back,y,0.05)])
            for z in np.linspace(0.05,GOAL_HEIGHT_M,15):
                strand([(x,y0,z),(back,y0,z),(back,y1,z),(x,y1,z)])
            for xx in np.linspace(x,back,12):
                strand([(xx,y0,0.05),(xx,y0,GOAL_HEIGHT_M),
                        (xx,y1,GOAL_HEIGHT_M),(xx,y1,0.05)])
            obj = bpy.data.objects.new("GoalNet",curve)
            bpy.context.collection.objects.link(obj)
            curve.materials.append(net_mat)
            support = bpy.data.curves.new("GoalRearSupport", "CURVE")
            support.dimensions = "3D"
            support.bevel_depth = 0.035
            sp = support.splines.new("POLY")
            points=[(x,y0,0.04),(back,y0,0.04),(back,y0,GOAL_HEIGHT_M),
                    (back,y1,GOAL_HEIGHT_M),(back,y1,0.04),(x,y1,0.04)]
            sp.points.add(len(points)-1)
            for p,co in zip(sp.points,points):
                p.co=(*co,1)
            obj=bpy.data.objects.new("GoalRearSupport",support)
            bpy.context.collection.objects.link(obj)
            support.materials.append(lines_mat)

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

    def _build_world(style: dict, palette: dict) -> None:
        top = render_look.hex_to_linear_rgba(palette["sky_top"])
        bottom = render_look.hex_to_linear_rgba(palette["sky_bottom"])
        world = bpy.data.worlds.new("W_Sky")
        world.use_nodes = True
        nt = world.node_tree
        nt.nodes.clear()
        out = nt.nodes.new("ShaderNodeOutputWorld")
        bg = nt.nodes.new("ShaderNodeBackground")
        # world_strength (default 1.0 — Blender's own Background node
        # default) dims/brightens the whole sky+ambient contribution;
        # night/floodlit looks pair a low value here with a low sun_energy.
        bg.inputs["Strength"].default_value = float(style.get("world_strength", 1.0))
        mix = nt.nodes.new("ShaderNodeMixRGB")
        mix.inputs["Color1"].default_value = bottom
        mix.inputs["Color2"].default_value = top
        sep = nt.nodes.new("ShaderNodeSeparateXYZ")
        texc = nt.nodes.new("ShaderNodeTexCoord")
        nt.links.new(texc.outputs["Normal"], sep.inputs["Vector"])
        nt.links.new(sep.outputs["Z"], mix.inputs["Fac"])
        nt.links.new(mix.outputs["Color"], bg.inputs["Color"])
        nt.links.new(bg.outputs["Background"], out.inputs["Surface"])
        bpy.context.scene.world = world

        bpy.ops.object.light_add(type="SUN")
        sun = bpy.context.active_object
        sun.name = "Sun"
        sun_rotation_deg = style.get("sun_rotation_deg", DEFAULT_SUN_ROTATION_DEG)
        sun.rotation_euler = tuple(radians(d) for d in sun_rotation_deg)
        sun.data.energy = float(style.get("sun_energy", DEFAULT_SUN_ENERGY))

    def _build_environment(style: dict) -> None:
        palette = style["palette"]
        lines_mat = _new_emission_material(
            "M_Lines", render_look.hex_to_linear_rgba(palette["lines"]),
            strength=float(style.get("lines_emission_strength", 1.0)))
        _build_pitch(style, palette)
        _build_lines(lines_mat)
        _build_pitch_details(lines_mat)
        _build_goals(lines_mat)
        _build_nets(lines_mat)
        from scripts.blender_stadium import build_stadium
        build_stadium(bpy, style)
        _build_world(style, palette)

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

    def _toon_material(name: str, rgba, ramp_steps: int,
                       pattern: dict | None = None) -> object:
        """Diffuse -> Shader-to-RGB -> constant ColorRamp -> Emission.

        The ramp quantises lighting into ``ramp_steps`` bands (classic cel
        shading). Emission output keeps the bands flat and print-like.

        ``pattern`` (``{type, colors: (rgba, rgba), width_m}``) switches to
        :func:`_toon_pattern_material`: the stripe/hoop colour is chosen
        from the baked ``rest_co`` attribute BEFORE the toon ramp, so the
        cel bands still shade the pattern.
        """
        if pattern is not None and _shader_to_rgb_available:
            return _toon_pattern_material(name, ramp_steps, pattern)
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

        # The ramp is driven by LIGHTING ONLY (a white diffuse), then
        # multiplied by the colour. Feeding the coloured diffuse into the
        # ramp (the v1 wiring) made the band depend on the colour's own
        # luminance: saturated reds (low luminance) never left the 35% band
        # and rendered maroon, yellow dropped a band and went olive — kits
        # never showed their true colour (origi01: Barcelona yellow → khaki).
        # Same maths as _toon_pattern_material.
        diffuse = nt.nodes.new("ShaderNodeBsdfDiffuse")
        diffuse.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
        to_rgb = nt.nodes.new("ShaderNodeShaderToRGB")
        ramp = nt.nodes.new("ShaderNodeValToRGB")
        ramp.color_ramp.interpolation = "CONSTANT"
        # evenly spaced constant stops from 35% to 100% brightness
        ramp.color_ramp.elements[0].position = 0.0
        ramp.color_ramp.elements[0].color = (0.35, 0.35, 0.35, 1.0)
        ramp.color_ramp.elements[1].position = 0.55
        ramp.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)
        for k in range(1, ramp_steps - 1):
            el = ramp.color_ramp.elements.new(0.15 + 0.4 * k / max(1, ramp_steps - 1))
            f = 0.35 + 0.65 * k / max(1, ramp_steps - 1)
            el.color = (f, f, f, 1.0)
        shade = nt.nodes.new("ShaderNodeMixRGB")
        shade.blend_type = "MULTIPLY"
        shade.inputs["Fac"].default_value = 1.0
        shade.inputs["Color1"].default_value = rgba
        emit = nt.nodes.new("ShaderNodeEmission")
        nt.links.new(diffuse.outputs["BSDF"], to_rgb.inputs["Shader"])
        # Blender 5.1.1 adaptation: ValToRGB's factor input socket is
        # named "Factor", not "Fac" as in the brief's snippet — renamed in
        # Blender's node socket-name pass. (MixRGB's "Fac" input used
        # elsewhere in this file is a different node type and unaffected;
        # verified both against the running Blender before wiring this.)
        nt.links.new(to_rgb.outputs["Color"], ramp.inputs["Factor"])
        nt.links.new(ramp.outputs["Color"], shade.inputs["Color2"])
        nt.links.new(shade.outputs["Color"], emit.inputs["Color"])
        nt.links.new(emit.outputs["Emission"], out.inputs["Surface"])
        _toon_material_count += 1
        return mat

    def _toon_pattern_material(name: str, ramp_steps: int, pattern: dict) -> object:
        """Pattern colour (rest-pose stripe mask) x grey toon ramp -> Emission.

        Stripe index = mod(floor(rest_co.<axis> / width + 0.5), 2): x for
        vertical stripes, y (height of the Y-up rest mesh) for hoops.
        Mathematically the same as the solid path: ramp(lighting) * colour.
        """
        nonlocal _toon_material_count
        mat = bpy.data.materials.new(name)
        mat.use_nodes = True
        nt = mat.node_tree
        nt.nodes.clear()
        out = nt.nodes.new("ShaderNodeOutputMaterial")
        diffuse = nt.nodes.new("ShaderNodeBsdfDiffuse")
        diffuse.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
        to_rgb = nt.nodes.new("ShaderNodeShaderToRGB")
        ramp = nt.nodes.new("ShaderNodeValToRGB")
        ramp.color_ramp.interpolation = "CONSTANT"
        ramp.color_ramp.elements[0].position = 0.0
        ramp.color_ramp.elements[0].color = (0.35, 0.35, 0.35, 1.0)
        ramp.color_ramp.elements[1].position = 0.55
        ramp.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)
        for k in range(1, ramp_steps - 1):
            el = ramp.color_ramp.elements.new(0.15 + 0.4 * k / max(1, ramp_steps - 1))
            f = 0.35 + 0.65 * k / max(1, ramp_steps - 1)
            el.color = (f, f, f, 1.0)
        attr = nt.nodes.new("ShaderNodeAttribute")
        attr.attribute_type = "GEOMETRY"
        attr.attribute_name = REST_CO_ATTRIBUTE
        sep = nt.nodes.new("ShaderNodeSeparateXYZ")
        nt.links.new(attr.outputs["Vector"], sep.inputs["Vector"])
        axis_out = ("X", "Y")[render_look.pattern_axis(pattern["type"])]
        div = nt.nodes.new("ShaderNodeMath")
        div.operation = "DIVIDE"
        div.inputs[1].default_value = float(pattern["width_m"])
        nt.links.new(sep.outputs[axis_out], div.inputs[0])
        shift = nt.nodes.new("ShaderNodeMath")
        shift.operation = "ADD"
        shift.inputs[1].default_value = 0.5
        nt.links.new(div.outputs[0], shift.inputs[0])
        floor = nt.nodes.new("ShaderNodeMath")
        floor.operation = "FLOOR"
        nt.links.new(shift.outputs[0], floor.inputs[0])
        # PINGPONG(x, 1) is |x mod 2 - 1| and stays well-defined for
        # negative x (plain MODULO truncates toward zero on Blender's
        # Math node): floor index -> 0/1 stripe parity.
        parity = nt.nodes.new("ShaderNodeMath")
        parity.operation = "PINGPONG"
        parity.inputs[1].default_value = 1.0
        nt.links.new(floor.outputs[0], parity.inputs[0])
        colour = nt.nodes.new("ShaderNodeMixRGB")
        colour.inputs["Color1"].default_value = pattern["colors"][0]
        colour.inputs["Color2"].default_value = pattern["colors"][1]
        nt.links.new(parity.outputs[0], colour.inputs["Fac"])
        shade = nt.nodes.new("ShaderNodeMixRGB")
        shade.blend_type = "MULTIPLY"
        shade.inputs["Fac"].default_value = 1.0
        nt.links.new(colour.outputs["Color"], shade.inputs["Color1"])
        nt.links.new(diffuse.outputs["BSDF"], to_rgb.inputs["Shader"])
        nt.links.new(to_rgb.outputs["Color"], ramp.inputs["Factor"])
        nt.links.new(ramp.outputs["Color"], shade.inputs["Color2"])
        emit = nt.nodes.new("ShaderNodeEmission")
        nt.links.new(shade.outputs["Color"], emit.inputs["Color"])
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
        """(head, tail) per bone in canonical rest space — the GEOMETRY
        table for capsule-body placement (direction, length, midpoint).

        Internal bones (with children) get a tail at the mean of their
        children's rest positions; leaf bones (hands, feet, head) get a
        fixed +0.05 z tail so they stay visible. NOT used for the
        armature's own edit-bone tails (see ``_ARMATURE_BONE_TAIL_M`):
        Blender interprets a pose-bone's ``rotation_quaternion`` in that
        bone's own LOCAL REST frame, which only matches SMPL's canonical
        joint-rotation convention when every bone's rest orientation is
        identical (a uniform-direction, zero-roll tail) — a per-bone
        direction here would rotate the SMPL thetas about the wrong axes.
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
        kit = colors.get(pid) or _fallback_kit_rgba
        if zone == "skin":
            rgba = kit.get("skin", _skin_rgba)
        else:
            # sleeve / collar fall back to the shirt colour.
            rgba = kit.get(zone) or _fallback_kit_rgba.get(
                zone, kit.get("shirt", _fallback_kit_rgba["shirt"]))
        pattern = None
        opts = player_kit_opts.get(pid, {})
        if zone in opts.get("pattern_zones", ()):
            pattern = opts.get("pattern")
        mat = _toon_material(f"{pid}_{zone}", rgba, ramp_steps, pattern=pattern)
        materials_cache[key] = mat
        return mat

    def _height_fraction(y: float, y_min: float, y_max: float) -> float:
        span = (y_max - y_min) or 1.0
        return (y - y_min) / span

    def _add_smpl_mesh_body(arm: object, pid: str, smpl_data, colors: dict,
                             materials_cache: dict, ramp_steps: int,
                             endpoints: list) -> list:
        """Full SMPL body mesh skinned to all 24 joints — mirrors
        blender_export_fbx.py's _add_smpl_skinned_mesh, plus per-face kit
        material slots (majority zone of the face's 3 vertices).

        ``load_smpl_body_data`` only guarantees ``joint_positions``/
        ``v_template`` are present (it returns ``None`` outright when
        those are missing) — ``faces``/``weights`` aren't checked there,
        so an SMPL asset variant missing either key would otherwise
        KeyError here. Guard them explicitly and fall back to the
        capsule-limb body (same as the ``smpl_data is None`` path)
        rather than crash the whole render.
        """
        missing = [k for k in ("faces", "weights") if k not in smpl_data]
        if missing:
            print(f"[render] SMPL body asset for {pid} missing key(s) "
                  f"{missing}; falling back to capsule body")
            return _add_capsule_body(
                arm, pid, endpoints, colors, materials_cache, ramp_steps)

        v_template = smpl_data["v_template"]
        faces = smpl_data["faces"]
        weights = smpl_data["weights"]

        mesh = bpy.data.meshes.new(f"{pid}_smpl_mesh")
        verts = [(float(v[0]), float(v[1]), float(v[2])) for v in v_template]
        face_list = [(int(t[0]), int(t[1]), int(t[2])) for t in faces]
        mesh.from_pydata(verts, [], face_list)
        mesh.update()
        # Rest-pose positions as a per-vertex vector attribute: pattern
        # materials read it (skinning never touches attributes), so stripes
        # and hoops stay glued to the torso as the body runs and turns.
        rest_attr = mesh.attributes.new(
            name=REST_CO_ATTRIBUTE, type="FLOAT_VECTOR", domain="POINT")
        rest_attr.data.foreach_set(
            "vector", render_look.rest_coords(v_template).ravel().tolist())
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

        if style.get("body_zones") == "anatomical":
            opts = player_kit_opts.get(pid, {})
            vertex_zones = render_look.anatomical_kit_zones(
                v_template, weights, smpl_data["joint_positions"],
                sleeves=opts.get("sleeves", "short"),
                gloves=bool(opts.get("gloves", False)))
        else:
            y = v_template[:, 1].astype(float)
            y_min, y_max = float(y.min()), float(y.max())
            vertex_zones = [
                render_look.kit_zone_for_height_fraction(
                    _height_fraction(v, y_min, y_max))
                for v in y
            ]
        slot_index = {}
        for zone in sorted(set(vertex_zones)):
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
        return [obj]

    def _parent_to_bone(obj: object, arm: object, bone_name: str,
                         local_offset, local_rotation: object) -> None:
        """Bone-parent ``obj`` to ``bone_name`` with an explicit LOCAL
        transform, given relative to the bone's HEAD in canonical rest
        space (``local_rotation`` likewise canonical — a rotation from a
        primitive's default axis to a direction expressed in that same
        space).

        Every armature bone's rest tail is now a fixed
        ``_ARMATURE_BONE_TAIL_M`` along local +Y from its head (uniform
        direction, zero roll — see the edit-bone loop above), which makes
        bone-local space identical to canonical rest space for every
        bone. Blender's BONE parent type anchors the child at the bone's
        TAIL, not its head, so that fixed offset is cancelled out of
        ``local_offset``'s Y component here — a plain constant
        subtraction, unlike the old Keep-Transform snapshot this
        replaced (which queried the CURRENT pose-bone matrix and so
        needed the armature to be at an identity pose to be correct).
        """
        obj.parent = arm
        obj.parent_type = "BONE"
        obj.parent_bone = bone_name
        obj.location = (
            float(local_offset[0]),
            float(local_offset[1] - _ARMATURE_BONE_TAIL_M),
            float(local_offset[2]),
        )
        obj.rotation_quaternion = local_rotation

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
            obj.data.materials.append(mat)

            # (mid - head): capsule centre relative to the bone's HEAD in
            # canonical rest space — the local offset _parent_to_bone
            # expects (see its docstring for the tail-cancellation math).
            _parent_to_bone(obj, arm, jname, mid - head, quat)
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
            player_smpl, player_pelvis = shape_smpl_body_data(smpl_data, entry["betas"], np)
            player_rest = player_smpl["joint_positions"] if player_smpl is not None else rest_joints
            endpoints = _bone_rest_endpoints(player_rest, _bone_children_map())
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
                head, _geom_tail = endpoints[j]
                eb.head = (float(head[0]), float(head[1]), float(head[2]))
                # Uniform +Y tail, zero roll (never the GEOMETRY table's
                # per-bone tail) — see _ARMATURE_BONE_TAIL_M.
                eb.tail = (float(head[0]), float(head[1] + _ARMATURE_BONE_TAIL_M),
                           float(head[2]))
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
                body_objs = _add_smpl_mesh_body(
                    arm, pid, player_smpl, colors, materials_cache, ramp_steps,
                    endpoints)
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
                offset = R @ player_pelvis
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

    # --- Style-post compositor (render-experiments task) ------------------
    # style.post's {glare, grain, vignette, posterize, duotone, saturation}
    # become a SECOND compositor graph, mutually exclusive with AOV (see
    # _render's gating below: an --aov request always wins, with a
    # one-time warning, over a configured post block — running both
    # would mean the AOV File Output node and the post effects fight over
    # the scene's single `compositing_node_group` slot, and AOV's raw
    # passes are the more valuable artifact to protect for downstream
    # tooling). Unlike _setup_aov_compositor this is REBUILT (not cached)
    # on every call that needs it: the grain effect bakes a noise image at
    # the calling resolution, and the 9:16 vertical pass renders at
    # swapped (height, width) from its landscape counterpart — a cached
    # graph would leave grain misaligned/wrong-aspect on whichever pass
    # didn't build it.
    #
    # Blender 5.1.1 adaptation (verified via a probe script rendering
    # actual stills and diffing pixels before wiring this — same
    # discipline as the AOV section above): a `CompositorNodeTree`
    # assigned to `scene.compositing_node_group` has NO "Composite"
    # output node in this version (`CompositorNodeComposite` doesn't
    # exist) — the redesigned compositor instead uses the generic
    # node-group interface: an OUTPUT socket declared via
    # `group.interface.new_socket(name=..., in_out='OUTPUT',
    # socket_type='NodeSocketColor')` plus a `NodeGroupOutput` node whose
    # matching input socket is what actually feeds the final rendered
    # image. Skipping the interface-socket step leaves the graph
    # silently inert — a `NodeGroupOutput` node's un-declared socket
    # links but has no effect on the render (confirmed by a probe render
    # with a naive Invert-only graph coming back pixel-identical to the
    # uncomposited baseline until the interface socket was added).
    # Also note several node types below are `ShaderNode*`, not
    # `CompositorNode*` (no compositor-native ColorRamp/Math/MixRGB exist
    # in this version) — cross-tree node reuse confirmed working via the
    # same probe.
    _post_compositor_group = None
    _post_grain_image = None

    def _teardown_post_compositor() -> None:
        nonlocal _post_compositor_group, _post_grain_image
        if _post_grain_image is not None:
            bpy.data.images.remove(_post_grain_image)
            _post_grain_image = None
        if _post_compositor_group is not None:
            bpy.data.node_groups.remove(_post_compositor_group)
            _post_compositor_group = None

    def _setup_post_compositor(post: dict, width: int, height: int) -> object:
        """(Re)build the style.post effect chain for this call's exact
        (width, height); see the module comment above for why this
        rebuilds every time rather than caching like AOV does.

        Effect order — Render Layers -> saturation -> duotone -> posterize
        -> glare -> vignette -> grain -> (group output): duotone fully
        replaces color (by design — a 2-color gradient has no "hue" left
        for a prior saturation change to act on, so saturation must run
        first to have any visible effect when both are set); glare runs
        on the graded/posterized image so its bloom picks up the
        stylised bright regions rather than the raw pre-grade render;
        vignette darkens after glare so the glow itself falls off toward
        the edges too; grain is the final film-stock-like overlay, last
        in any real photochemical pipeline.
        """
        nonlocal _post_compositor_group, _post_grain_image
        _teardown_post_compositor()

        group = bpy.data.node_groups.new("PostFX", "CompositorNodeTree")
        bpy.context.scene.compositing_node_group = group
        rl = group.nodes.new("CompositorNodeRLayers")
        rl.layer = bpy.context.view_layer.name
        prev = rl.outputs["Image"]

        saturation = float(post.get("saturation", 1.0))
        if saturation != 1.0:
            hue_sat = group.nodes.new("CompositorNodeHueSat")
            hue_sat.inputs["Hue"].default_value = 0.5
            hue_sat.inputs["Value"].default_value = 1.0
            hue_sat.inputs["Factor"].default_value = 1.0
            hue_sat.inputs["Saturation"].default_value = saturation
            group.links.new(prev, hue_sat.inputs["Image"])
            prev = hue_sat.outputs["Image"]

        duotone_pair = render_look.duotone_colors(post.get("duotone"))
        if duotone_pair is not None:
            shadow_rgba, highlight_rgba = duotone_pair
            to_bw = group.nodes.new("CompositorNodeRGBToBW")
            group.links.new(prev, to_bw.inputs["Image"])
            ramp = group.nodes.new("ShaderNodeValToRGB")
            ramp.color_ramp.interpolation = "LINEAR"
            ramp.color_ramp.elements[0].position = 0.0
            ramp.color_ramp.elements[0].color = shadow_rgba
            ramp.color_ramp.elements[1].position = 1.0
            ramp.color_ramp.elements[1].color = highlight_rgba
            group.links.new(to_bw.outputs["Val"], ramp.inputs["Factor"])
            prev = ramp.outputs["Color"]

        posterize_steps = int(post.get("posterize", 0) or 0)
        if posterize_steps >= 2:
            poster = group.nodes.new("CompositorNodePosterize")
            poster.inputs["Steps"].default_value = float(posterize_steps)
            group.links.new(prev, poster.inputs["Image"])
            prev = poster.outputs["Image"]

        glare_strength = float(post.get("glare", 0.0))
        if glare_strength > 0.0:
            glare = group.nodes.new("CompositorNodeGlare")
            # "Fog Glow" is the closest analogue to legacy bloom (EEVEE
            # Next has none natively — see the CLAUDE.md/task note this
            # mirrors); Type is a MENU input socket taking these exact
            # title-case strings, not an enum property on the node.
            glare.inputs["Type"].default_value = "Fog Glow"
            glare.inputs["Threshold"].default_value = 0.5
            glare.inputs["Strength"].default_value = glare_strength
            group.links.new(prev, glare.inputs["Image"])
            prev = glare.outputs["Image"]

        vignette_strength = float(post.get("vignette", 0.0))
        if vignette_strength > 0.0:
            mask = group.nodes.new("CompositorNodeEllipseMask")
            mask.inputs["Position"].default_value = (0.5, 0.5)
            # Both Size axes are fractions of the frame WIDTH, so a square
            # size draws a circle — on a 9:16 pass that is a small disc.
            # Scale y by the aspect to get a frame-filling ellipse.
            mask.inputs["Size"].default_value = (0.9, 0.9 * height / width)
            blur = group.nodes.new("CompositorNodeBlur")
            blur.inputs["Type"].default_value = "Gaussian"
            # Blur Size is in PIXELS in Blender 5.x (a fractional size
            # was a no-op and left a hard-edged disc); feather by a
            # quarter of the short side for a soft falloff.
            feather_px = 0.25 * min(width, height)
            blur.inputs["Size"].default_value = (feather_px, feather_px)
            group.links.new(mask.outputs["Mask"], blur.inputs["Image"])
            mask_val = group.nodes.new("CompositorNodeRGBToBW")
            group.links.new(blur.outputs["Image"], mask_val.inputs["Image"])
            # darken_factor = vignette_strength * (1 - blurred_mask):
            # 0 at frame centre (mask==1, no darkening), ramping up
            # toward the edges (mask->0).
            invert = group.nodes.new("ShaderNodeMath")
            invert.operation = "SUBTRACT"
            invert.inputs[0].default_value = 1.0
            group.links.new(mask_val.outputs["Val"], invert.inputs[1])
            darken = group.nodes.new("ShaderNodeMath")
            darken.operation = "MULTIPLY"
            darken.inputs[1].default_value = vignette_strength
            group.links.new(invert.outputs[0], darken.inputs[0])
            black = group.nodes.new("CompositorNodeRGB")
            black.outputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)
            vignette_mix = group.nodes.new("ShaderNodeMixRGB")
            vignette_mix.blend_type = "MIX"
            group.links.new(prev, vignette_mix.inputs["Color1"])
            group.links.new(black.outputs["Color"], vignette_mix.inputs["Color2"])
            group.links.new(darken.outputs[0], vignette_mix.inputs["Fac"])
            prev = vignette_mix.outputs["Color"]

        grain_amount = float(post.get("grain", 0.0))
        if grain_amount > 0.0:
            pixels = render_look.grain_noise_pixels(width, height)
            img = bpy.data.images.new(
                "PostFX_Grain", width, height, alpha=True, float_buffer=True)
            # Non-Color: this is raw blend data, not a color to run
            # through Blender's sRGB/view-transform pipeline.
            img.colorspace_settings.name = "Non-Color"
            img.pixels.foreach_set(pixels)
            _post_grain_image = img
            img_node = group.nodes.new("CompositorNodeImage")
            img_node.image = img
            grain_mix = group.nodes.new("ShaderNodeMixRGB")
            grain_mix.blend_type = "OVERLAY"
            grain_mix.inputs["Fac"].default_value = grain_amount
            group.links.new(prev, grain_mix.inputs["Color1"])
            group.links.new(img_node.outputs["Image"], grain_mix.inputs["Color2"])
            prev = grain_mix.outputs["Color"]

        group.interface.new_socket(
            name="Image", in_out="OUTPUT", socket_type="NodeSocketColor")
        group_out = group.nodes.new("NodeGroupOutput")
        group.links.new(prev, group_out.inputs["Image"])

        _post_compositor_group = group
        return group

    _printed_post_aov_warning = False

    # --- Render ----------------------------------------------------------

    def _render(camera_obj: object, out_path: Path, fps: float,
                frame_range: tuple[int, int], width: int, height: int,
                samples: int, aov_dir: Path | None = None,
                post: dict | None = None) -> None:
        scene = bpy.context.scene
        scene.camera = camera_obj
        scene.render.resolution_x = width
        scene.render.resolution_y = height
        scene.render.fps = int(round(fps))
        # Time stretching maps scene frame f -> animation time f / stretch,
        # so the stretched range [start*N, end*N] covers the same source
        # frames with every keyed channel (poses, ball, camera) interpolated.
        stretch = min(9, max(1, int(args.time_stretch)))
        scene.render.frame_map_old = 100
        scene.render.frame_map_new = 100 * stretch
        scene.frame_start = frame_range[0] * stretch
        scene.frame_end = frame_range[1] * stretch

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

        # `scene.compositing_node_group` (once assigned by either
        # _setup_aov_compositor or _setup_post_compositor, below) stays
        # attached to the scene for every subsequent render call —
        # Blender doesn't clear it, and `scene.render.use_compositing`
        # (a separate flag, default True) is never touched elsewhere, so
        # the compositor keeps running on every later render regardless
        # of this call's own `aov_dir`/`post`. Left ungated, a landscape
        # pass with `--aov` (or an active style.post) "poisons" every
        # following render (e.g. the 9:16 vertical pass, or a next camera
        # without AOV) into re-firing at a stale directory/resolution —
        # and since the AOV rename-to-`.exr` loop below is itself gated
        # on `aov_dir`, that stray write is left as a corrupt extension-
        # less file. Explicitly (re)deciding `use_compositing` AND which
        # graph (if any) is attached on *every* call — not just when a
        # request is present — is what actually scopes each to the calls
        # that asked for it.
        #
        # AOV and style.post are mutually exclusive for now: both would
        # otherwise need to share the scene's single
        # `compositing_node_group` slot, and AOV's raw Z/Normal/
        # Cryptomatte passes are the more valuable artifact for
        # downstream tooling to protect, so an `--aov` request always
        # wins — with a one-time warning — over a configured post block.
        active_post = post if post and render_look.post_style_is_active(post) else None
        if active_post is not None and aov_dir is not None:
            nonlocal _printed_post_aov_warning
            if not _printed_post_aov_warning:
                print(
                    "[render] style.post effects are configured but --aov "
                    "was requested; AOV wins (style.post is disabled for "
                    "this run) — see blender_render_scene.py's AOV/post "
                    "mutual-exclusion note."
                )
                _printed_post_aov_warning = True
            active_post = None

        if aov_dir is not None:
            scene.render.use_compositing = True
            _teardown_post_compositor()
            fo = _setup_aov_compositor()
            aov_dir.mkdir(parents=True, exist_ok=True)
            fo.directory = str(aov_dir) + "/"
        elif active_post is not None:
            scene.render.use_compositing = True
            _setup_post_compositor(active_post, width, height)
        else:
            scene.render.use_compositing = False
            _teardown_post_compositor()

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
    # players.json kit roles + skin/hair (operator data) — the same
    # role source the export stage honours (load_kit_roles).
    player_looks = render_look.resolve_player_looks(
        style.get("teams", {}) or {}, team_class,
        role_overrides=load_kit_roles(output_dir),
        appearance=load_player_appearance(output_dir))
    player_colors = {pid: look["colors"] for pid, look in player_looks.items()}
    player_kit_opts = {
        pid: {"sleeves": look["sleeves"], "gloves": look["gloves"],
              "pattern": look["pattern"],
              "pattern_zones": look["pattern_zones"]}
        for pid, look in player_looks.items()}
    smpl_data, pelvis_canon = load_smpl_body_data(_REPO_ROOT, np)
    smpl_problem = render_look.smpl_asset_problem(
        smpl_data, args.allow_capsule_fallback)
    if smpl_problem:
        sys.stderr.write(f"[render] {smpl_problem}\n")
        return 3
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
        return (output_dir / render_root / shot_dir / "cameras"
                / f"{_safe_cam_id(cam_id)}_camera_track.json")

    broadcast_path = _camera_track_path("broadcast")
    fps = DEFAULT_FPS
    if broadcast_path.exists():
        fps = float(load_camera_track(broadcast_path).get("fps", DEFAULT_FPS)) or DEFAULT_FPS

    # Build every requested camera FIRST (bailing out on any missing/empty
    # track before rendering anything), then save the .blend ONCE — rather
    # than once per camera, which just re-wrote the same file N times and
    # left scene.camera unset (the file only "usefully" opened on whatever
    # camera a later render call happened to set). scene.camera is pointed
    # at the first built camera so the saved file opens on something.
    cam_entries: list[tuple[str, object, int, int]] = []
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
        cam_entries.append((cam_id, cam_obj, frame_start, frame_end))

    if args.save_blend and cam_entries:
        scene = bpy.context.scene
        scene.camera = cam_entries[0][1]
        scene.render.engine = "BLENDER_EEVEE"
        scene.render.resolution_x = args.width
        scene.render.resolution_y = args.height
        scene.render.resolution_percentage = 100
        scene.render.fps = int(round(fps))
        scene.render.fps_base = int(round(fps)) / fps
        scene.frame_start = min(entry[2] for entry in cam_entries)
        scene.frame_end = max(entry[3] for entry in cam_entries)
        scene.frame_set(scene.frame_start)
        bpy.ops.wm.save_as_mainfile(filepath=str(out_dir / "scene.blend"))

    def _hide_player_for_eyes(cam_id: str) -> list:
        """``eyes:<PID>``: hide that player's armature + body/capsule
        objects for this camera's renders (the camera sits inside the
        head, so the own body would clip the lens); returns the objects
        hidden, for :func:`_restore_hidden`."""
        pid = render_look.eyes_hidden_pid(cam_id)
        if pid is None:
            return []
        arm = bpy.data.objects.get(f"{pid}_arm")
        if arm is None:
            print(f"[render] {cam_id}: no armature for {pid}; nothing to hide")
            return []
        objs = [arm, *arm.children_recursive]
        for o in objs:
            o.hide_render = True
        return objs

    def _restore_hidden(objs: list) -> None:
        for o in objs:
            o.hide_render = False

    for cam_id, cam_obj, frame_start, frame_end in cam_entries:
        safe_id = _safe_cam_id(cam_id)

        # AOV EXRs (Task 9) are only rendered for the landscape pass, one
        # subdirectory per camera — never for the 9:16 vertical pass below.
        hidden = _hide_player_for_eyes(cam_id)
        passes = render_look.plan_passes(
            cam_id, args.vertical, args.vertical_only)
        if args.vertical_only and args.aov:
            print("[render] --aov ignored under --vertical-only "
                  "(AOVs are landscape-pass only)")
        aov_dir = (out_dir / "aov" / safe_id) if (
            args.aov and not args.vertical_only) else None
        if ("", False) in passes:
            _render(cam_obj, out_dir / f"{safe_id}.mp4", fps,
                    (frame_start, frame_end), args.width, args.height,
                    args.samples, aov_dir=aov_dir, post=style.get("post"))

        # 9:16 portrait pass (Task 8): every non-broadcast camera gets a
        # second render at swapped (height, width) resolution. Reframed
        # via sensor_fit="VERTICAL" rather than scaling cam.lens — the
        # lens values are keyframed (see _add_camera_from_track), so
        # scaling them would require re-keying every frame; sensor_fit
        # instead changes how the *existing* keyed lens maps to FOV for
        # this pass only, then is restored to "HORIZONTAL" for the next
        # camera's landscape render. style.post still applies here (it's
        # never gated on --vertical the way AOV is) — _setup_post_compositor
        # rebuilds its grain bake at this call's swapped resolution, so it
        # stays aligned rather than reusing the landscape pass's tile.
        if ("_9x16", True) in passes:
            cam_obj.data.sensor_fit = "VERTICAL"
            _render(cam_obj, out_dir / f"{safe_id}_9x16.mp4", fps,
                    (frame_start, frame_end), args.height, args.width, args.samples,
                    post=style.get("post"))
            cam_obj.data.sensor_fit = "HORIZONTAL"
        _restore_hidden(hidden)

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
