"""Build a glTF 2.0 scene from SMPL results for the 3D viewer.

Creates a .glb file containing:
- A skeleton (24 joints) per player with animation
- A pitch ground plane (105m × 68m)
- Metadata sidecar JSON for the viewer
"""

import json
import struct
from pathlib import Path

import numpy as np

# SMPL 24-joint skeleton parent indices (-1 = root)
_SMPL_PARENTS = [
    -1,  # 0  pelvis
    0,   # 1  left_hip
    0,   # 2  right_hip
    0,   # 3  spine1
    1,   # 4  left_knee
    2,   # 5  right_knee
    3,   # 6  spine2
    4,   # 7  left_ankle
    5,   # 8  right_ankle
    6,   # 9  spine3
    7,   # 10 left_foot
    8,   # 11 right_foot
    9,   # 12 neck
    9,   # 13 left_collar
    9,   # 14 right_collar
    12,  # 15 head
    13,  # 16 left_shoulder
    14,  # 17 right_shoulder
    16,  # 18 left_elbow
    17,  # 19 right_elbow
    18,  # 20 left_wrist
    19,  # 21 right_wrist
    20,  # 22 left_hand
    21,  # 23 right_hand
]

_SMPL_JOINT_NAMES = [
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee",
    "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot",
    "neck", "left_collar", "right_collar", "head", "left_shoulder",
    "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
    "left_hand", "right_hand",
]


def build_scene_metadata(
    smpl_results: list,
    player_teams: dict[str, str],
    fps: float,
) -> dict:
    """Build scene_metadata.json content.

    Args:
        smpl_results: list of SmplResult objects.
        player_teams: fallback ``{player_id: team}`` mapping (used only when
            the SmplResult itself has no team).
        fps: frame rate.

    Returns:
        dict ready for JSON serialization.
    """
    players = []
    max_frames = 0
    for r in smpl_results:
        n_frames = r.poses.shape[0]
        max_frames = max(max_frames, n_frames)
        team = getattr(r, "team", "") or player_teams.get(r.player_id, "unknown")
        player_name = getattr(r, "player_name", "") or r.player_id
        players.append({
            "player_id": r.player_id,
            "player_name": player_name,
            "team": team,
            "num_frames": n_frames,
        })

    return {
        "fps": fps,
        "total_frames": max_frames,
        "num_players": len(players),
        "players": players,
        "joint_names": _SMPL_JOINT_NAMES,
        "pitch": {"length": 105.0, "width": 68.0},
    }


def build_minimal_glb(
    smpl_results: list,
    player_teams: dict[str, str],
) -> bytes:
    """Build a minimal GLB containing per-player translation animation.

    This is a simplified GLB that encodes player root translations as
    animated nodes. The full 3D viewer reads both this and the metadata
    JSON to render skeletons.

    For a production pipeline, this would include full skeleton hierarchies
    and quaternion rotation channels. The current version stores translation
    keyframes so the viewer can at minimum show player movement.
    """
    # Build a minimal valid glTF JSON
    nodes = []
    animations = []
    buffers_data = bytearray()
    buffer_views = []
    accessors = []

    for pi, result in enumerate(smpl_results):
        n_frames = result.transl.shape[0]
        fps = result.fps if result.fps > 0 else 25.0

        # Node for this player
        nodes.append({
            "name": result.player_id,
            "translation": [float(result.transl[0, 0]), float(result.transl[0, 2]), float(-result.transl[0, 1])],
        })

        # Time keyframes
        times = np.arange(n_frames, dtype=np.float32) / fps
        time_bytes = times.tobytes()
        time_bv_idx = len(buffer_views)
        time_offset = len(buffers_data)
        buffers_data.extend(time_bytes)
        # Pad to 4-byte alignment
        while len(buffers_data) % 4 != 0:
            buffers_data.append(0)
        buffer_views.append({
            "buffer": 0,
            "byteOffset": time_offset,
            "byteLength": len(time_bytes),
        })
        time_acc_idx = len(accessors)
        accessors.append({
            "bufferView": time_bv_idx,
            "componentType": 5126,  # FLOAT
            "count": n_frames,
            "type": "SCALAR",
            "min": [float(times[0])],
            "max": [float(times[-1])],
        })

        # Translation keyframes (convert to glTF Y-up: x, z, -y)
        translations = np.column_stack([
            result.transl[:, 0],
            result.transl[:, 2],
            -result.transl[:, 1],
        ]).astype(np.float32)
        trans_bytes = translations.tobytes()
        trans_bv_idx = len(buffer_views)
        trans_offset = len(buffers_data)
        buffers_data.extend(trans_bytes)
        while len(buffers_data) % 4 != 0:
            buffers_data.append(0)
        buffer_views.append({
            "buffer": 0,
            "byteOffset": trans_offset,
            "byteLength": len(trans_bytes),
        })
        trans_acc_idx = len(accessors)
        accessors.append({
            "bufferView": trans_bv_idx,
            "componentType": 5126,
            "count": n_frames,
            "type": "VEC3",
            "min": translations.min(axis=0).tolist(),
            "max": translations.max(axis=0).tolist(),
        })

        animations.append({
            "name": f"{result.player_id}_anim",
            "channels": [{
                "sampler": 0,
                "target": {"node": pi, "path": "translation"},
            }],
            "samplers": [{
                "input": time_acc_idx,
                "output": trans_acc_idx,
                "interpolation": "LINEAR",
            }],
        })

    gltf = {
        "asset": {"version": "2.0", "generator": "football-perspectives"},
        "scene": 0,
        "scenes": [{"name": "scene", "nodes": list(range(len(nodes)))}],
        "nodes": nodes,
        "animations": animations,
        "accessors": accessors,
        "bufferViews": buffer_views,
        "buffers": [{"byteLength": len(buffers_data)}],
    }

    # Encode as GLB
    gltf_json = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    # Pad JSON to 4-byte alignment
    while len(gltf_json) % 4 != 0:
        gltf_json += b" "

    # GLB header: magic + version + total length
    total_length = 12 + 8 + len(gltf_json) + 8 + len(buffers_data)
    header = struct.pack("<III", 0x46546C67, 2, total_length)  # glTF magic
    json_chunk = struct.pack("<II", len(gltf_json), 0x4E4F534A) + gltf_json  # JSON chunk
    bin_chunk = struct.pack("<II", len(buffers_data), 0x004E4942) + bytes(buffers_data)  # BIN chunk

    return header + json_chunk + bin_chunk
