"""glTF ball rotation sampler tests.

The ball node used to animate translation only; Phase 3 adds a rotation
channel driven by the per-frame ``BallFrame.quat_wxyz``.  glTF stores
quaternions as (x, y, z, w) so the builder must convert from our
(w, x, y, z) convention — this is the load-bearing, explicitly-tested
behaviour.
"""

from __future__ import annotations

import json
import struct

import numpy as np
import pytest

from src.schemas.ball_track import BallFrame, BallTrack, FlightSegment
from src.schemas.camera_track import CameraFrame, CameraTrack
from src.utils.gltf_builder import SceneBundle, build_glb


def _parse_gltf_json(glb: bytes) -> dict:
    json_len = struct.unpack_from("<I", glb, 12)[0]
    return json.loads(glb[20 : 20 + json_len])


def _bin_chunk(glb: bytes) -> bytes:
    json_len = struct.unpack_from("<I", glb, 12)[0]
    bin_start = 20 + json_len
    bin_len = struct.unpack_from("<I", glb, bin_start)[0]
    return glb[bin_start + 8 : bin_start + 8 + bin_len]


def _read_accessor(gltf: dict, binary: bytes, acc_idx: int) -> np.ndarray:
    acc = gltf["accessors"][acc_idx]
    bv = gltf["bufferViews"][acc["bufferView"]]
    offset = bv.get("byteOffset", 0)
    count = acc["count"]
    ncomp = {"SCALAR": 1, "VEC3": 3, "VEC4": 4}[acc["type"]]
    raw = binary[offset : offset + count * ncomp * 4]
    return np.frombuffer(raw, dtype="<f4").reshape(count, ncomp)


def _camera_track(n: int) -> CameraTrack:
    return CameraTrack(
        clip_id="play", fps=30.0, image_size=(1920, 1080), t_world=[0, 0, 30],
        frames=tuple(
            CameraFrame(frame=i, K=[[1000, 0, 960], [0, 1000, 540], [0, 0, 1]],
                        R=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                        confidence=1.0, is_anchor=False)
            for i in range(n)
        ),
    )


# A known wxyz quaternion: 90° about world +Z.
# wxyz = (cos45, 0, 0, sin45). glTF xyz,w = (0, 0, sin45, cos45).
_C = float(np.cos(np.pi / 4))
_S = float(np.sin(np.pi / 4))
_KNOWN_WXYZ = (_C, 0.0, 0.0, _S)


def _ball_track() -> BallTrack:
    frames = (
        BallFrame(frame=0, world_xyz=(10.0, 5.0, 0.11), state="grounded",
                  confidence=0.9, quat_wxyz=(1.0, 0.0, 0.0, 0.0)),
        BallFrame(frame=1, world_xyz=(11.0, 5.0, 1.0), state="flight",
                  confidence=0.9, flight_segment_id=0, quat_wxyz=_KNOWN_WXYZ),
        BallFrame(frame=2, world_xyz=(12.0, 5.0, 0.11), state="grounded",
                  confidence=0.9, quat_wxyz=(0.0, 1.0, 0.0, 0.0)),
    )
    return BallTrack(
        clip_id="play", fps=30.0, frames=frames,
        flight_segments=(
            FlightSegment(id=0, frame_range=(1, 1),
                          parabola={"p0": [11, 5, 1], "v0": [1, 0, 0], "g": -9.81},
                          fit_residual_px=1.0),
        ),
    )


def _bundle() -> SceneBundle:
    return SceneBundle(
        camera_track=_camera_track(3), players=(),
        ball_track=_ball_track(),
        pitch_length_m=105.0, pitch_width_m=68.0, ball_radius_m=0.11,
    )


@pytest.mark.unit
def test_ball_has_rotation_channel() -> None:
    glb, _ = build_glb(_bundle())
    gltf = _parse_gltf_json(glb)
    anim = next(a for a in gltf["animations"] if a["name"] == "ball_anim")
    paths = {ch["target"]["path"] for ch in anim["channels"]}
    assert "rotation" in paths
    assert "translation" in paths
    # Both channels point at the same ball node.
    nodes = {ch["target"]["node"] for ch in anim["channels"]}
    assert len(nodes) == 1


@pytest.mark.unit
def test_ball_rotation_quaternions_unit_norm() -> None:
    glb, _ = build_glb(_bundle())
    gltf = _parse_gltf_json(glb)
    binary = _bin_chunk(glb)
    anim = next(a for a in gltf["animations"] if a["name"] == "ball_anim")
    rot_ch = next(c for c in anim["channels"] if c["target"]["path"] == "rotation")
    rot_acc_idx = anim["samplers"][rot_ch["sampler"]]["output"]
    assert gltf["accessors"][rot_acc_idx]["type"] == "VEC4"
    quats = _read_accessor(gltf, binary, rot_acc_idx)
    assert quats.shape == (3, 4)
    norms = np.linalg.norm(quats, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)


@pytest.mark.unit
def test_ball_rotation_wxyz_to_xyzw_conversion() -> None:
    """A known (w,x,y,z) must appear as (x,y,z,w) in the glTF buffer."""
    glb, _ = build_glb(_bundle())
    gltf = _parse_gltf_json(glb)
    binary = _bin_chunk(glb)
    anim = next(a for a in gltf["animations"] if a["name"] == "ball_anim")
    rot_ch = next(c for c in anim["channels"] if c["target"]["path"] == "rotation")
    rot_acc_idx = anim["samplers"][rot_ch["sampler"]]["output"]
    quats = _read_accessor(gltf, binary, rot_acc_idx)

    # Frame index 1 carries the known wxyz quaternion.
    w, x, y, z = _KNOWN_WXYZ
    expected_xyzw = np.array([x, y, z, w], dtype=np.float32)
    np.testing.assert_allclose(quats[1], expected_xyzw, atol=1e-6)

    # Frame 0 wxyz=(1,0,0,0) -> xyzw=(0,0,0,1).
    np.testing.assert_allclose(quats[0], [0.0, 0.0, 0.0, 1.0], atol=1e-6)
    # Frame 2 wxyz=(0,1,0,0) -> xyzw=(1,0,0,0).
    np.testing.assert_allclose(quats[2], [1.0, 0.0, 0.0, 0.0], atol=1e-6)


@pytest.mark.unit
def test_ball_rotation_holds_previous_for_none() -> None:
    """A None quat holds the previous quaternion; leading None -> identity."""
    frames = (
        BallFrame(frame=0, world_xyz=(0.0, 0.0, 0.11), state="grounded",
                  confidence=0.9, quat_wxyz=None),
        BallFrame(frame=1, world_xyz=(1.0, 0.0, 0.11), state="grounded",
                  confidence=0.9, quat_wxyz=_KNOWN_WXYZ),
        BallFrame(frame=2, world_xyz=(2.0, 0.0, 0.11), state="grounded",
                  confidence=0.9, quat_wxyz=None),
    )
    track = BallTrack(clip_id="play", fps=30.0, frames=frames,
                      flight_segments=())
    bundle = SceneBundle(camera_track=_camera_track(3), players=(),
                         ball_track=track, ball_radius_m=0.11)
    glb, _ = build_glb(bundle)
    gltf = _parse_gltf_json(glb)
    binary = _bin_chunk(glb)
    anim = next(a for a in gltf["animations"] if a["name"] == "ball_anim")
    rot_ch = next(c for c in anim["channels"] if c["target"]["path"] == "rotation")
    rot_acc_idx = anim["samplers"][rot_ch["sampler"]]["output"]
    quats = _read_accessor(gltf, binary, rot_acc_idx)
    # Leading None -> identity (0,0,0,1) in xyzw.
    np.testing.assert_allclose(quats[0], [0.0, 0.0, 0.0, 1.0], atol=1e-6)
    # Frame 1 is the known quat.
    w, x, y, z = _KNOWN_WXYZ
    np.testing.assert_allclose(quats[1], [x, y, z, w], atol=1e-6)
    # Frame 2 None -> holds frame 1.
    np.testing.assert_allclose(quats[2], [x, y, z, w], atol=1e-6)
