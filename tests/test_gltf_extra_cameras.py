from __future__ import annotations

import json
import struct

import numpy as np
import pytest

from src.schemas.camera_track import CameraFrame, CameraTrack
from src.utils.gltf_builder import SceneBundle, build_glb


def _parse_gltf_json(glb: bytes) -> dict:
    json_len = struct.unpack_from("<I", glb, 12)[0]
    return json.loads(glb[20 : 20 + json_len])


def _track(clip_id: str, with_per_frame_t: bool) -> CameraTrack:
    frames = tuple(
        CameraFrame(
            frame=i, K=[[1000, 0, 960], [0, 1000, 540], [0, 0, 1]],
            R=[[1, 0, 0], [0, 1, 0], [0, 0, 1]], confidence=1.0, is_anchor=False,
            t=[0.0, 0.0, float(-i)] if with_per_frame_t else None,
        )
        for i in range(3)
    )
    return CameraTrack(clip_id=clip_id, fps=30.0, image_size=(1920, 1080),
                       t_world=[0, 0, 30], frames=frames)


@pytest.mark.unit
def test_extra_cameras_appear_in_metadata() -> None:
    bundle = SceneBundle(
        camera_track=_track("broadcast", False),
        players=(),
        ball_track=None,
        pitch_length_m=105.0,
        pitch_width_m=68.0,
        ball_radius_m=0.11,
        extra_cameras=(("P003_pov", _track("P003_pov", True)),),
    )
    glb, meta = build_glb(bundle)
    assert glb[:4] == b"glTF"
    names = {c["name"] for c in meta.get("cameras", [])}
    assert "P003_pov" in names


@pytest.mark.unit
def test_extra_camera_has_translation_and_rotation_channels() -> None:
    bundle = SceneBundle(
        camera_track=_track("broadcast", False),
        players=(),
        ball_track=None,
        pitch_length_m=105.0,
        pitch_width_m=68.0,
        ball_radius_m=0.11,
        extra_cameras=(("P003_pov", _track("P003_pov", True)),),
    )
    glb, meta = build_glb(bundle)
    gltf = _parse_gltf_json(glb)
    anim = next(a for a in gltf["animations"] if "P003_pov" in a["name"])
    paths = {ch["target"]["path"] for ch in anim["channels"]}
    assert paths == {"rotation", "translation"}
    rot_ch = next(c for c in anim["channels"] if c["target"]["path"] == "rotation")
    trans_ch = next(c for c in anim["channels"] if c["target"]["path"] == "translation")
    rot_acc = gltf["accessors"][anim["samplers"][rot_ch["sampler"]]["output"]]
    trans_acc = gltf["accessors"][anim["samplers"][trans_ch["sampler"]]["output"]]
    assert rot_acc["type"] == "VEC4"
    assert trans_acc["type"] == "VEC3"
    assert trans_acc["count"] == 3
