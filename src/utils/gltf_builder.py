"""Build a glTF 2.0 scene from broadcast-mono pipeline outputs.

Inputs:
    - ``CameraTrack`` (per-frame K, R; clip-shared t_world).
    - List of ``SmplWorldTrack`` (per-player root_t/root_R/thetas/betas).
    - ``BallTrack`` (per-frame ball world_xyz).

Outputs:
    - A ``.glb`` byte string containing:
        * A pitch ground plane (105 m x 68 m green) with painted white lines
          (touchlines, halfway, centre circle, penalty boxes) emitted as line
          primitives.
        * One animated capsule per player at root_t (with root rotation).
        * One animated sphere ball at ball_track world_xyz.
        * One animated camera reflecting per-frame K/R/t_world.

v1 simplification: players are emitted as oriented capsules at root_t rather
than fully-skinned SMPL meshes.  Body deformation by ``thetas`` and full
SMPL skinning is a future enhancement; the GLB contains everything needed
to wire that up later (animation samplers and node hierarchy are already
in place; only the mesh geometry is simplified).

Coordinate convention: pitch-world is z-up (z=0 is the pitch surface),
x along the nearside touchline, y toward the far side.  glTF nodes are
authored in the pitch frame directly — clients that prefer y-up should
add a root rotation; this avoids lossy conversion at export time.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from typing import Iterable

import numpy as np

# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SceneBundle:
    """Inputs to ``build_glb``."""

    camera_track: object       # CameraTrack
    players: tuple             # tuple[SmplWorldTrack, ...]
    ball_track: object | None  # BallTrack | None
    pitch_length_m: float = 105.0
    pitch_width_m: float = 68.0
    ball_radius_m: float = 0.11
    player_height_m: float = 1.85
    player_radius_m: float = 0.30


# ---------------------------------------------------------------------------
# Pitch geometry
# ---------------------------------------------------------------------------


def _pitch_line_segments(length: float, width: float) -> list[tuple[tuple[float, float, float], tuple[float, float, float]]]:
    """Return (start, end) segments approximating FIFA pitch markings."""
    z = 0.001  # avoid z-fighting with the green plane
    segs: list[tuple[tuple[float, float, float], tuple[float, float, float]]] = []
    # Touchlines + goal lines (rectangle).
    segs.append(((0, 0, z), (length, 0, z)))
    segs.append(((0, width, z), (length, width, z)))
    segs.append(((0, 0, z), (0, width, z)))
    segs.append(((length, 0, z), (length, width, z)))
    # Halfway line.
    segs.append(((length / 2, 0, z), (length / 2, width, z)))
    # Centre circle (approximate with 36-segment polyline, radius 9.15 m).
    cx, cy, r = length / 2, width / 2, 9.15
    n_seg = 36
    for i in range(n_seg):
        a0 = 2 * np.pi * i / n_seg
        a1 = 2 * np.pi * (i + 1) / n_seg
        segs.append((
            (cx + r * np.cos(a0), cy + r * np.sin(a0), z),
            (cx + r * np.cos(a1), cy + r * np.sin(a1), z),
        ))
    # Penalty boxes (16.5 m deep, 40.32 m wide; centred on width).
    pb_d = 16.5
    pb_half = 40.32 / 2
    cy = width / 2
    # Left penalty box
    segs.append(((0, cy - pb_half, z), (pb_d, cy - pb_half, z)))
    segs.append(((0, cy + pb_half, z), (pb_d, cy + pb_half, z)))
    segs.append(((pb_d, cy - pb_half, z), (pb_d, cy + pb_half, z)))
    # Right penalty box
    segs.append(((length, cy - pb_half, z), (length - pb_d, cy - pb_half, z)))
    segs.append(((length, cy + pb_half, z), (length - pb_d, cy + pb_half, z)))
    segs.append(((length - pb_d, cy - pb_half, z), (length - pb_d, cy + pb_half, z)))
    # Centre spot omitted (single point); centre line covers it.
    return segs


def _build_pitch_geometry(length: float, width: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (plane_positions, plane_indices, line_positions, line_indices)."""
    # Plane: two triangles covering [0, length] x [0, width] at z=0.
    plane_positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [length, 0.0, 0.0],
            [length, width, 0.0],
            [0.0, width, 0.0],
        ],
        dtype=np.float32,
    )
    plane_indices = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)

    segs = _pitch_line_segments(length, width)
    line_positions = np.array([p for seg in segs for p in seg], dtype=np.float32)
    line_indices = np.arange(len(line_positions), dtype=np.uint32)
    return plane_positions, plane_indices, line_positions, line_indices


# ---------------------------------------------------------------------------
# Capsule + sphere primitives (very low-poly).
# ---------------------------------------------------------------------------


def _build_capsule(height: float, radius: float, lat: int = 6, lon: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """Capsule along +z, base at z=0, top at z=height. Low-poly."""
    body_h = max(height - 2 * radius, 0.0)
    rings: list[np.ndarray] = []
    # Bottom hemisphere (lat rings, latitude 0..pi/2).
    for i in range(lat + 1):
        theta = (np.pi / 2) * (i / lat)  # 0..pi/2
        z = radius - radius * np.cos(theta)  # 0..radius
        r = radius * np.sin(theta)
        ring = []
        for j in range(lon):
            phi = 2 * np.pi * j / lon
            ring.append([r * np.cos(phi), r * np.sin(phi), z])
        rings.append(np.array(ring, dtype=np.float32))
    # Top hemisphere offset by body_h.
    for i in range(lat + 1):
        theta = (np.pi / 2) * (i / lat)
        z = radius + body_h + radius * np.sin(theta)
        r = radius * np.cos(theta)
        ring = []
        for j in range(lon):
            phi = 2 * np.pi * j / lon
            ring.append([r * np.cos(phi), r * np.sin(phi), z])
        rings.append(np.array(ring, dtype=np.float32))

    positions = np.concatenate(rings, axis=0).astype(np.float32)
    n_rings = len(rings)
    indices: list[int] = []
    for i in range(n_rings - 1):
        for j in range(lon):
            a = i * lon + j
            b = i * lon + (j + 1) % lon
            c = (i + 1) * lon + j
            d = (i + 1) * lon + (j + 1) % lon
            indices.extend([a, c, b, b, c, d])
    return positions, np.array(indices, dtype=np.uint32)


def _build_sphere(radius: float, lat: int = 6, lon: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """Low-poly sphere centred at origin."""
    rings: list[np.ndarray] = []
    for i in range(lat + 1):
        theta = np.pi * i / lat   # 0..pi
        z = radius * np.cos(theta) * -1.0 + 0.0
        # Use sin for ring radius
        rr = radius * np.sin(theta)
        zz = radius * np.cos(theta)
        ring = []
        for j in range(lon):
            phi = 2 * np.pi * j / lon
            ring.append([rr * np.cos(phi), rr * np.sin(phi), zz])
        rings.append(np.array(ring, dtype=np.float32))
    positions = np.concatenate(rings, axis=0).astype(np.float32)
    indices: list[int] = []
    for i in range(lat):
        for j in range(lon):
            a = i * lon + j
            b = i * lon + (j + 1) % lon
            c = (i + 1) * lon + j
            d = (i + 1) * lon + (j + 1) % lon
            indices.extend([a, c, b, b, c, d])
    return positions, np.array(indices, dtype=np.uint32)


# ---------------------------------------------------------------------------
# GLB writer (manual).
# ---------------------------------------------------------------------------


class _GlbBuilder:
    """Helper that accumulates buffer views, accessors, and binary blobs."""

    def __init__(self) -> None:
        self.json: dict = {
            "asset": {"version": "2.0", "generator": "football-perspectives"},
            "scenes": [{"nodes": []}],
            "scene": 0,
            "nodes": [],
            "meshes": [],
            "materials": [],
            "animations": [],
            "accessors": [],
            "bufferViews": [],
            "buffers": [],
            "cameras": [],
        }
        self.bin = bytearray()

    # -- low-level helpers -------------------------------------------------

    def _pad(self) -> None:
        while len(self.bin) % 4 != 0:
            self.bin.append(0)

    def _add_buffer_view(self, data: bytes, target: int | None = None) -> int:
        offset = len(self.bin)
        self.bin.extend(data)
        self._pad()
        bv = {"buffer": 0, "byteOffset": offset, "byteLength": len(data)}
        if target is not None:
            bv["target"] = target
        self.json["bufferViews"].append(bv)
        return len(self.json["bufferViews"]) - 1

    def add_accessor_scalar_f32(self, values: np.ndarray) -> int:
        v = values.astype(np.float32, copy=False).ravel()
        bv = self._add_buffer_view(v.tobytes())
        acc = {
            "bufferView": bv,
            "componentType": 5126,  # FLOAT
            "count": int(v.size),
            "type": "SCALAR",
            "min": [float(v.min()) if v.size else 0.0],
            "max": [float(v.max()) if v.size else 0.0],
        }
        self.json["accessors"].append(acc)
        return len(self.json["accessors"]) - 1

    def add_accessor_vec3_f32(self, values: np.ndarray) -> int:
        v = np.asarray(values, dtype=np.float32).reshape(-1, 3)
        bv = self._add_buffer_view(v.tobytes(), target=34962)  # ARRAY_BUFFER
        mn = v.min(axis=0).tolist() if v.size else [0.0, 0.0, 0.0]
        mx = v.max(axis=0).tolist() if v.size else [0.0, 0.0, 0.0]
        acc = {
            "bufferView": bv,
            "componentType": 5126,
            "count": int(v.shape[0]),
            "type": "VEC3",
            "min": mn,
            "max": mx,
        }
        self.json["accessors"].append(acc)
        return len(self.json["accessors"]) - 1

    def add_accessor_vec4_f32(self, values: np.ndarray) -> int:
        v = np.asarray(values, dtype=np.float32).reshape(-1, 4)
        bv = self._add_buffer_view(v.tobytes())
        mn = v.min(axis=0).tolist() if v.size else [0.0, 0.0, 0.0, 0.0]
        mx = v.max(axis=0).tolist() if v.size else [0.0, 0.0, 0.0, 0.0]
        acc = {
            "bufferView": bv,
            "componentType": 5126,
            "count": int(v.shape[0]),
            "type": "VEC4",
            "min": mn,
            "max": mx,
        }
        self.json["accessors"].append(acc)
        return len(self.json["accessors"]) - 1

    def add_accessor_indices_u32(self, indices: np.ndarray) -> int:
        v = indices.astype(np.uint32, copy=False).ravel()
        bv = self._add_buffer_view(v.tobytes(), target=34963)  # ELEMENT_ARRAY_BUFFER
        acc = {
            "bufferView": bv,
            "componentType": 5125,  # UNSIGNED_INT
            "count": int(v.size),
            "type": "SCALAR",
            "min": [int(v.min()) if v.size else 0],
            "max": [int(v.max()) if v.size else 0],
        }
        self.json["accessors"].append(acc)
        return len(self.json["accessors"]) - 1

    # -- high-level builders ----------------------------------------------

    def add_material(self, base_color_rgba: tuple[float, float, float, float]) -> int:
        self.json["materials"].append({
            "pbrMetallicRoughness": {
                "baseColorFactor": list(base_color_rgba),
                "metallicFactor": 0.0,
                "roughnessFactor": 1.0,
            },
        })
        return len(self.json["materials"]) - 1

    def add_mesh_triangles(self, positions: np.ndarray, indices: np.ndarray, material: int, name: str) -> int:
        pos_acc = self.add_accessor_vec3_f32(positions)
        idx_acc = self.add_accessor_indices_u32(indices)
        self.json["meshes"].append({
            "name": name,
            "primitives": [{
                "attributes": {"POSITION": pos_acc},
                "indices": idx_acc,
                "mode": 4,  # TRIANGLES
                "material": material,
            }],
        })
        return len(self.json["meshes"]) - 1

    def add_mesh_lines(self, positions: np.ndarray, indices: np.ndarray, material: int, name: str) -> int:
        pos_acc = self.add_accessor_vec3_f32(positions)
        idx_acc = self.add_accessor_indices_u32(indices)
        self.json["meshes"].append({
            "name": name,
            "primitives": [{
                "attributes": {"POSITION": pos_acc},
                "indices": idx_acc,
                "mode": 1,  # LINES
                "material": material,
            }],
        })
        return len(self.json["meshes"]) - 1

    def add_node(self, node: dict) -> int:
        self.json["nodes"].append(node)
        idx = len(self.json["nodes"]) - 1
        self.json["scenes"][0]["nodes"].append(idx)
        return idx

    def add_orphan_node(self, node: dict) -> int:
        """Add a node that is *not* attached to the scene root."""
        self.json["nodes"].append(node)
        return len(self.json["nodes"]) - 1

    def add_animation(self, name: str, channels: list[dict], samplers: list[dict]) -> None:
        self.json["animations"].append({
            "name": name,
            "channels": channels,
            "samplers": samplers,
        })

    def add_camera(self, perspective: dict, name: str) -> int:
        self.json["cameras"].append({"type": "perspective", "perspective": perspective, "name": name})
        return len(self.json["cameras"]) - 1

    # -- finalise ----------------------------------------------------------

    def to_glb(self) -> bytes:
        # Buffer is a single binary buffer.
        self.json["buffers"] = [{"byteLength": len(self.bin)}]
        gltf_json = json.dumps(self.json, separators=(",", ":")).encode("utf-8")
        # Pad JSON chunk to 4-byte alignment with spaces.
        while len(gltf_json) % 4 != 0:
            gltf_json += b" "
        bin_payload = bytes(self.bin)
        # Pad bin chunk too (already 4-byte aligned by our _pad).
        total = 12 + 8 + len(gltf_json) + 8 + len(bin_payload)
        header = struct.pack("<III", 0x46546C67, 2, total)
        json_chunk = struct.pack("<II", len(gltf_json), 0x4E4F534A) + gltf_json
        bin_chunk = struct.pack("<II", len(bin_payload), 0x004E4942) + bin_payload
        return header + json_chunk + bin_chunk


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rotmat_to_quat(R: np.ndarray) -> np.ndarray:
    """Convert (3, 3) rotation matrix to (x, y, z, w) quaternion (glTF order)."""
    R = np.asarray(R, dtype=np.float64)
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([x, y, z, w], dtype=np.float32)
    return q / max(float(np.linalg.norm(q)), 1e-12)


def _camera_world_position(R: np.ndarray, t_world: np.ndarray) -> np.ndarray:
    """t_world is the world-space camera centre (consistent with rest of pipeline)."""
    return np.asarray(t_world, dtype=np.float64).reshape(3)


def _camera_orientation_quat(R: np.ndarray) -> np.ndarray:
    """Convert pipeline R (world->camera) into a glTF node rotation.

    glTF cameras look along -Z in their local frame.  Pipeline R maps
    world points into a camera frame whose +Z axis points along the
    optical ray (into the scene).  Therefore the glTF node rotation is
    the inverse of R, with a flip of the Z axis so -Z becomes the
    optical ray.
    """
    R = np.asarray(R, dtype=np.float64)
    # Flip the camera-frame Z axis so glTF -Z aligns with pipeline +Z.
    flip = np.diag([1.0, -1.0, -1.0])
    M = R.T @ flip
    return _rotmat_to_quat(M)


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------


def build_glb(bundle: SceneBundle) -> tuple[bytes, dict]:
    """Build a GLB binary + a metadata sidecar dict."""
    g = _GlbBuilder()

    # Materials.
    pitch_mat = g.add_material((0.10, 0.55, 0.18, 1.0))   # green
    line_mat = g.add_material((1.0, 1.0, 1.0, 1.0))       # white
    player_mat = g.add_material((0.85, 0.20, 0.20, 1.0))  # red player
    ball_mat = g.add_material((1.0, 0.90, 0.20, 1.0))     # yellow ball

    # Pitch.
    plane_pos, plane_idx, line_pos, line_idx = _build_pitch_geometry(
        bundle.pitch_length_m, bundle.pitch_width_m
    )
    pitch_mesh = g.add_mesh_triangles(plane_pos, plane_idx, pitch_mat, "pitch_plane")
    line_mesh = g.add_mesh_lines(line_pos, line_idx, line_mat, "pitch_lines")
    g.add_node({"name": "pitch", "mesh": pitch_mesh})
    g.add_node({"name": "pitch_lines", "mesh": line_mesh})

    # Players.
    player_meta: list[dict] = []
    cap_pos, cap_idx = _build_capsule(bundle.player_height_m, bundle.player_radius_m)
    player_mesh = g.add_mesh_triangles(cap_pos, cap_idx, player_mat, "player_capsule")
    fps = float(bundle.camera_track.fps) if hasattr(bundle.camera_track, "fps") else 30.0
    for player in bundle.players:
        n_frames = int(player.frames.shape[0])
        if n_frames == 0:
            continue
        # Initial pose.
        init_t = np.asarray(player.root_t[0], dtype=np.float32).tolist()
        init_q = _rotmat_to_quat(player.root_R[0]).tolist()
        node_idx = g.add_node({
            "name": str(player.player_id),
            "mesh": player_mesh,
            "translation": [float(init_t[0]), float(init_t[1]), float(init_t[2])],
            "rotation": [float(q) for q in init_q],
        })
        # Animation samplers.
        times = np.asarray(player.frames, dtype=np.float32) / max(fps, 1e-6)
        time_acc = g.add_accessor_scalar_f32(times)
        trans_acc = g.add_accessor_vec3_f32(np.asarray(player.root_t, dtype=np.float32))
        quats = np.array([_rotmat_to_quat(R) for R in player.root_R], dtype=np.float32)
        rot_acc = g.add_accessor_vec4_f32(quats)
        g.add_animation(
            name=f"{player.player_id}_anim",
            samplers=[
                {"input": time_acc, "output": trans_acc, "interpolation": "LINEAR"},
                {"input": time_acc, "output": rot_acc, "interpolation": "LINEAR"},
            ],
            channels=[
                {"sampler": 0, "target": {"node": node_idx, "path": "translation"}},
                {"sampler": 1, "target": {"node": node_idx, "path": "rotation"}},
            ],
        )
        player_meta.append({
            "player_id": str(player.player_id),
            "num_frames": n_frames,
            "frame_range": [int(player.frames.min()), int(player.frames.max())],
            "mean_confidence": float(player.confidence.mean()) if player.confidence.size else 0.0,
        })
        # NOTE: full SMPL skinning + per-joint pose application from
        # ``player.thetas`` is intentionally deferred — v1 emits an
        # oriented capsule per player.  Future work: emit a 24-joint
        # skin with skinned mesh and per-joint rotation channels.

    # Ball.
    ball_meta: dict | None = None
    if bundle.ball_track is not None:
        ball_track = bundle.ball_track
        if hasattr(ball_track, "frames") and len(ball_track.frames) > 0:
            sphere_pos, sphere_idx = _build_sphere(bundle.ball_radius_m)
            ball_mesh = g.add_mesh_triangles(sphere_pos, sphere_idx, ball_mat, "ball")
            # Build animation: only frames with non-null world_xyz.
            ball_frames = [f for f in ball_track.frames if f.world_xyz is not None]
            if ball_frames:
                ball_fps = float(getattr(ball_track, "fps", fps) or fps)
                times = np.array([f.frame for f in ball_frames], dtype=np.float32) / max(ball_fps, 1e-6)
                positions = np.array([f.world_xyz for f in ball_frames], dtype=np.float32)
                init_pos = positions[0].tolist()
                node_idx = g.add_node({
                    "name": "ball",
                    "mesh": ball_mesh,
                    "translation": [float(init_pos[0]), float(init_pos[1]), float(init_pos[2])],
                })
                time_acc = g.add_accessor_scalar_f32(times)
                trans_acc = g.add_accessor_vec3_f32(positions)
                g.add_animation(
                    name="ball_anim",
                    samplers=[{"input": time_acc, "output": trans_acc, "interpolation": "LINEAR"}],
                    channels=[{"sampler": 0, "target": {"node": node_idx, "path": "translation"}}],
                )
                ball_meta = {
                    "num_frames": int(len(ball_frames)),
                    "frame_range": [int(ball_frames[0].frame), int(ball_frames[-1].frame)],
                }

    # Camera.
    camera_meta: dict | None = None
    cam_track = bundle.camera_track
    if hasattr(cam_track, "frames") and len(cam_track.frames) > 0:
        first = cam_track.frames[0]
        K0 = np.asarray(first.K, dtype=np.float64)
        img_w, img_h = cam_track.image_size
        fx = float(K0[0, 0])
        # glTF perspective: yfov + aspectRatio.
        yfov = 2.0 * np.arctan2(float(img_h) / 2.0, fx)
        aspect = float(img_w) / float(img_h) if img_h else 1.0
        cam_idx = g.add_camera(
            {
                "yfov": float(yfov),
                "aspectRatio": float(aspect),
                "znear": 0.1,
                "zfar": 1000.0,
            },
            "broadcast_camera",
        )
        t_world = np.asarray(cam_track.t_world, dtype=np.float64)
        init_q = _camera_orientation_quat(np.asarray(first.R, dtype=np.float64))
        node_idx = g.add_node({
            "name": "broadcast_camera",
            "camera": cam_idx,
            "translation": [float(t_world[0]), float(t_world[1]), float(t_world[2])],
            "rotation": [float(q) for q in init_q],
        })
        # Animate rotation only — t_world is clip-shared in the spec.
        cam_fps = float(cam_track.fps) if cam_track.fps else fps
        times = np.array([f.frame for f in cam_track.frames], dtype=np.float32) / max(cam_fps, 1e-6)
        quats = np.array([_camera_orientation_quat(np.asarray(f.R, dtype=np.float64)) for f in cam_track.frames], dtype=np.float32)
        time_acc = g.add_accessor_scalar_f32(times)
        rot_acc = g.add_accessor_vec4_f32(quats)
        g.add_animation(
            name="camera_anim",
            samplers=[{"input": time_acc, "output": rot_acc, "interpolation": "LINEAR"}],
            channels=[{"sampler": 0, "target": {"node": node_idx, "path": "rotation"}}],
        )
        camera_meta = {
            "image_size": [int(img_w), int(img_h)],
            "num_frames": int(len(cam_track.frames)),
            "frame_range": [int(cam_track.frames[0].frame), int(cam_track.frames[-1].frame)],
            "fps": float(cam_fps),
        }

    metadata = {
        "fps": fps,
        "pitch": {
            "length_m": float(bundle.pitch_length_m),
            "width_m": float(bundle.pitch_width_m),
        },
        "players": player_meta,
        "ball": ball_meta,
        "camera": camera_meta,
        "notes": {
            "player_geometry": "v1 capsule; full SMPL skinning is a future enhancement",
            "coordinate_frame": "pitch z-up; x along nearside, y toward far side",
        },
    }
    return g.to_glb(), metadata
