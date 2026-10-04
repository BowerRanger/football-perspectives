"""Procedural stadium dressing, batched by material to keep EEVEE affordable.

Imported only inside Blender. Dimensions use the pipeline's pitch coordinates:
X along the pitch, Y across it, Z up. No downloaded assets or random state.
"""
from __future__ import annotations

import math
import random

from src.utils.pitch import PITCH_LENGTH as L, PITCH_WIDTH as W
from src.utils.render_look import hex_to_linear_rgba
from src.utils.stadium_dressing import crowd_palette_for_stand, structural_tones


DEFAULT_CROWD_COLORS = ("#283e50", "#b1b8af", "#a64039", "#ceac78", "#476780", "#d2c9b5")
DEFAULT_BOARD_TEXT = "FOOTBALL / PERSPECTIVES"


def build_stadium(bpy, style):
    cfg = style.get("stadium", {})
    if not cfg.get("enabled", True):
        return
    collection = bpy.data.collections.new("Stadium")
    bpy.context.scene.collection.children.link(collection)
    batches = {}

    def material(name, color, emission=False):
        mat = bpy.data.materials.new(name)
        mat.diffuse_color = hex_to_linear_rgba(color)
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        bsdf.inputs["Base Color"].default_value = mat.diffuse_color
        bsdf.inputs["Roughness"].default_value = 0.78
        if emission:
            bsdf.inputs["Emission Color"].default_value = mat.diffuse_color
            bsdf.inputs["Emission Strength"].default_value = 0.6
        batches[name] = [[], [], mat]
        return name

    # stand_tone seeds concrete/steel/tunnels/boards, all lifted to a minimum
    # lightness (stadium_dressing.structural_tones) so low cameras don't see
    # a black band where the toon ramp's shadow band hits dark fascia.
    tones = structural_tones(cfg)
    concrete = material("Stadium_Concrete", tones["concrete"])
    steel = material("Stadium_Steel", tones["steel"])
    roof = material("Stadium_Roof", "#c5ced2")
    seat = material("Stadium_Seats", cfg.get("seat_color", "#294b65"))
    accent = material("Stadium_SeatAccent", cfg.get("accent_color", "#d6b66e"))
    dark = material("Stadium_Tunnels", tones["tunnels"])
    board = material("Stadium_Boards", tones["board"])
    board_ink = material("Stadium_BoardText", tones["board_text"], True)
    white = material("Stadium_White", "#e1e9e6", True)
    turf = material("Stadium_Runoff", "#355f43")
    ground = material("Stadium_Concourse", "#414e55")
    # crowd_colors: per-venue shirt palette (repeat a colour to weight it);
    # the away_end stand gets its own palette (the away kit by default).
    stand_names = ("North", "South", "East", "West")
    palettes = {n: tuple(crowd_palette_for_stand(cfg, n) or DEFAULT_CROWD_COLORS)
                for n in stand_names}
    crowd_mats = {}
    for palette in dict.fromkeys(palettes.values()):
        crowd_mats[palette] = [
            material(f"Stadium_Crowd_{len(crowd_mats)}_{i}", c)
            for i, c in enumerate(palette)]
    crowd_for = {n: crowd_mats[palettes[n]] for n in stand_names}
    skin = material("Stadium_CrowdSkin", "#b88667")

    # All cuboids for a material share one mesh; no thousands of Blender objects.
    def box(mat, center, size, angle=0):
        verts, faces, _ = batches[mat]
        n = len(verts)
        co, si = math.cos(angle), math.sin(angle)
        for x, y, z in [(-1,-1,-1), (1,-1,-1), (1,1,-1), (-1,1,-1),
                        (-1,-1,1), (1,-1,1), (1,1,1), (-1,1,1)]:
            x, y, z = x*size[0]/2, y*size[1]/2, z*size[2]/2
            verts.append((center[0]+co*x-si*y, center[1]+si*x+co*y, center[2]+z))
        faces.extend(tuple(n+i for i in f) for f in
                     [(0,3,2,1), (4,5,6,7), (0,1,5,4), (1,2,6,5), (2,3,7,6), (3,0,4,7)])

    def beam(mat, a, b, radius=0.06):
        # Curves grouped by purpose for slim rails, poles and trusses.
        key = f"{mat}_Rails"
        obj = bpy.data.objects.get(key)
        if obj is None:
            curve = bpy.data.curves.new(key, "CURVE")
            curve.dimensions = "3D"
            curve.bevel_depth = radius
            curve.bevel_resolution = 0
            curve.resolution_u = 1
            obj = bpy.data.objects.new(key, curve)
            collection.objects.link(obj)
            curve.materials.append(batches[mat][2])
        sp = obj.data.splines.new("POLY")
        sp.points.add(1)
        sp.points[0].co = (*a, 1)
        sp.points[1].co = (*b, 1)

    box(ground, (L/2, W/2, -0.22), (190, 148, 0.3))
    box(turf, (L/2, W/2, -0.05), (L+16, W+16, 0.06))
    rng = random.Random(17)
    density = max(0., min(1., float(cfg.get("crowd_density", 0.65))))
    # Local u along each stand, v points away from the pitch.
    stands = [("North", (L/2, W+10), 0., 118.),
              ("South", (L/2, -10), math.pi, 118.),
              ("West", (-12, W/2), math.pi/2, 82.),
              ("East", (L+12, W/2), -math.pi/2, 82.)]
    for name, origin, angle, length in stands:
        co, si = math.cos(angle), math.sin(angle)
        def point(u, v, z):
            return (origin[0]+co*u-si*v, origin[1]+si*u+co*v, z)
        def local(mat, u, v, z, size):
            box(mat, point(u, v, z), size, angle)
        local(concrete, 0, 10, 0.55, (length, 23, 1.1))
        for row in range(22):
            v = row*0.88 + (2.2 if row >= 11 else 0)
            z = 1.2 + row*0.55 + (1.3 if row >= 11 else 0)
            local(concrete, 0, v, z-0.28, (length, 0.88, 0.56))
            for col in range(int(length/0.72)):
                u = -length/2 + 0.5 + col*0.72
                # Aisles every 12 seats; central access tunnels at tier fronts.
                if col % 14 < 2 or (abs(u)<2 and row in (0,1,11,12)):
                    continue
                sm = accent if col % 42 >= 35 else seat
                local(sm, u, v, z+0.22, (0.48,0.43,0.12))
                local(sm, u, v+0.20, z+0.47, (0.48,0.09,0.50))
                if rng.random() < density:
                    local(rng.choice(crowd_for[name]), u, v, z+0.57, (0.36,0.28,0.46))
                    local(skin, u, v, z+0.94, (0.20,0.20,0.23))
        local(dark, 0, 0.1, 1.55, (3.8,0.18,2.4))
        local(dark, 0, 11.3, 8.0, (3.8,0.18,2.4))
        # Tier fascia and continuous safety rails.
        for v, z in [(-0.45,1.35), (10.0,7.8), (21.5,15.0)]:
            local(steel, 0,v,z, (length,0.15,0.5))
            beam(steel, point(-length/2,v,z+0.7), point(length/2,v,z+0.7))
            for u in range(-int(length/2), int(length/2), 4):
                beam(steel, point(u,v,z), point(u,v,z+0.7))
        if cfg.get("roof", True):
            local(roof,0,19,21.0,(length+1,10,0.28))
            local(steel,0,14,20.6,(length+1,0.3,0.65))
            for u in range(-int(length/2),int(length/2)+1,10):
                beam(steel,point(u,23,0),point(u,23,21))
                beam(steel,point(u,23,20.7),point(u,14,20.7))
                beam(steel,point(u,23,17),point(u,14,20.7))
                local(white,u,14.2,20.1,(2.8,0.4,0.2))
        # Inner advertising ribbon, behind the cameras' pitchside corridor.
        local(board,0,-3.5,0.60,(length,0.16,1.15))
        for u in range(-int(length/2)+4,int(length/2)-3,9):
            data = bpy.data.curves.new(f"{name}_BoardText", "FONT")
            data.body = cfg.get("board_text") or DEFAULT_BOARD_TEXT
            data.align_x = "CENTER"
            data.size = 0.29
            data.extrude = 0
            obj = bpy.data.objects.new(data.name, data)
            collection.objects.link(obj)
            obj.location = point(u,-3.60,0.51)
            obj.rotation_euler = (math.pi/2, 0, angle)
            data.materials.append(batches[board_ink][2])
            local(accent,u+4,-3.61,0.60,(0.12,0.02,0.8))

    # Dugouts sit outside the dolly path (y=-3), with an open pitch-facing side.
    for x in (L/2-14,L/2+14):
        box(steel,(x,-7.7,1.2),(8,0.16,2.4))
        box(roof,(x,-6.9,2.5),(8.2,2.0,0.18))
        for dx in (-4,4):
            box(steel,(x+dx,-6.9,1.2),(0.12,1.8,2.4))
        for dx in range(-3,4):
            box(seat,(x+dx,-7.1,0.55),(0.6,0.5,0.14))
            box(seat,(x+dx,-7.35,0.9),(0.6,0.12,0.7))
    # Corner flags have a deliberately simple, lightly folded silhouette.
    for x in (0,L):
        for y in (0,W):
            beam(white,(x,y,0),(x,y,1.55),0.025)
            box(accent,(x+0.19,y,1.37),(0.38,0.035,0.28))
    # Four floodlight masts fill the otherwise open stadium corners.
    for x in (-14,L+14):
        for y in (-12,W+12):
            box(steel,(x,y,12),(0.38,0.38,24))
            box(steel,(x,y,24),(4.2,0.6,2.0))
            for dx in (-1.5,-0.5,0.5,1.5):
                for dz in (-0.5,0.5):
                    box(white,(x+dx,y+(0.34 if y<0 else -0.34),24+dz),(0.75,0.08,0.65))
    for name, (verts,faces,mat) in batches.items():
        if not verts:
            continue
        mesh = bpy.data.meshes.new(name)
        mesh.from_pydata(verts,[],faces)
        mesh.update()
        obj = bpy.data.objects.new(name,mesh)
        collection.objects.link(obj)
        mesh.materials.append(mat)
