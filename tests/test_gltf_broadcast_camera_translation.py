"""The main (broadcast) glTF camera node must animate translation, not
just rotation — it hardcoded a single ``t_world`` translation and
animated rotation only, which is correct for a static broadcast rig but
silently wrong for a moving-camera clip (spidercam/wirecam) where every
frame's own recovered centre differs. The sibling ``extra_cameras`` path
(POV/OTS virtual cameras) already animates both channels via per-frame
``t``; this brings the main camera in line with that existing pattern.

See docs/superpowers/specs/2026-09-09-moving-camera-support.md.
"""

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


def _identity_R() -> list[list[float]]:
    return [[1, 0, 0], [0, 1, 0], [0, 0, 1]]


def _moving_track() -> CameraTrack:
    """A camera whose per-frame t genuinely differs frame to frame —
    camera_centre=None, mirroring what the camera stage now persists for
    a moving-camera (auto-diagnosed or explicit) shot."""
    frames = tuple(
        CameraFrame(
            frame=i, K=[[1000, 0, 960], [0, 1000, 540], [0, 0, 1]],
            R=_identity_R(), confidence=1.0, is_anchor=False,
            t=[float(i) * 2.0, 0.0, -30.0],   # -R^T@t = (-2i, 0, 30): moves
        )
        for i in range(3)
    )
    return CameraTrack(
        clip_id="spidercam", fps=30.0, image_size=(1920, 1080),
        t_world=[0.0, 0.0, 30.0], frames=frames, camera_centre=None,
    )


def _static_track() -> CameraTrack:
    """A camera whose per-frame t is constant — camera_centre set,
    mirroring a static broadcast shot."""
    frames = tuple(
        CameraFrame(
            frame=i, K=[[1000, 0, 960], [0, 1000, 540], [0, 0, 1]],
            R=_identity_R(), confidence=1.0, is_anchor=False,
            t=[0.0, 0.0, -30.0],
        )
        for i in range(3)
    )
    return CameraTrack(
        clip_id="broadcast", fps=30.0, image_size=(1920, 1080),
        t_world=[0.0, 0.0, 30.0], frames=frames,
        camera_centre=(0.0, 0.0, 30.0),
    )


def _bundle(track: CameraTrack) -> SceneBundle:
    return SceneBundle(
        camera_track=track, players=(), ball_track=None,
        pitch_length_m=105.0, pitch_width_m=68.0, ball_radius_m=0.11,
    )


@pytest.mark.unit
def test_broadcast_camera_animates_translation_for_a_moving_clip() -> None:
    glb, _meta = build_glb(_bundle(_moving_track()))
    gltf = _parse_gltf_json(glb)
    anim = next(a for a in gltf["animations"] if a["name"] == "camera_anim")
    paths = {ch["target"]["path"] for ch in anim["channels"]}
    assert paths == {"rotation", "translation"}, (
        "moving-camera clip's broadcast camera must animate translation, "
        "not just rotation"
    )
    trans_ch = next(c for c in anim["channels"] if c["target"]["path"] == "translation")
    trans_acc = gltf["accessors"][anim["samplers"][trans_ch["sampler"]]["output"]]
    assert trans_acc["type"] == "VEC3"
    assert trans_acc["count"] == 3


@pytest.mark.unit
def test_broadcast_camera_translation_tracks_per_frame_centre() -> None:
    """The animated translation values must be the per-frame recovered
    centre (-R^T @ t), not a single repeated value."""
    glb, _meta = build_glb(_bundle(_moving_track()))
    gltf = _parse_gltf_json(glb)
    anim = next(a for a in gltf["animations"] if a["name"] == "camera_anim")
    trans_ch = next(c for c in anim["channels"] if c["target"]["path"] == "translation")
    trans_acc_idx = anim["samplers"][trans_ch["sampler"]]["output"]
    trans_acc = gltf["accessors"][trans_acc_idx]
    bv = gltf["bufferViews"][trans_acc["bufferView"]]
    buf = gltf["buffers"][bv["buffer"]]
    # Reconstruct the raw bytes the same way the other GLB tests do: the
    # binary chunk directly follows the JSON chunk in a single-buffer GLB.
    json_len = struct.unpack_from("<I", glb, 12)[0]
    bin_chunk_start = 20 + json_len + 8
    offset = bin_chunk_start + bv.get("byteOffset", 0) + trans_acc.get("byteOffset", 0)
    values = np.frombuffer(
        glb[offset : offset + 12 * trans_acc["count"]], dtype="<f4",
    ).reshape(-1, 3)
    xs = values[:, 0]
    assert xs[0] != pytest.approx(xs[-1]), (
        f"translation x must vary across frames for a moving camera, got {xs}"
    )
    # Frame i's centre is (-2i, 0, 30) per _moving_track's construction.
    assert xs[1] == pytest.approx(-2.0, abs=1e-3)
    assert xs[2] == pytest.approx(-4.0, abs=1e-3)


@pytest.mark.unit
def test_broadcast_camera_translation_is_constant_for_a_static_clip() -> None:
    """Regression guard: a static broadcast clip's animated translation
    must still be constant (harmless no-op change for the common case)."""
    glb, _meta = build_glb(_bundle(_static_track()))
    gltf = _parse_gltf_json(glb)
    anim = next(a for a in gltf["animations"] if a["name"] == "camera_anim")
    trans_ch = next(c for c in anim["channels"] if c["target"]["path"] == "translation")
    trans_acc_idx = anim["samplers"][trans_ch["sampler"]]["output"]
    trans_acc = gltf["accessors"][trans_acc_idx]
    bv = gltf["bufferViews"][trans_acc["bufferView"]]
    json_len = struct.unpack_from("<I", glb, 12)[0]
    bin_chunk_start = 20 + json_len + 8
    offset = bin_chunk_start + bv.get("byteOffset", 0) + trans_acc.get("byteOffset", 0)
    values = np.frombuffer(
        glb[offset : offset + 12 * trans_acc["count"]], dtype="<f4",
    ).reshape(-1, 3)
    assert np.allclose(values[0], values[-1], atol=1e-4)
    assert np.allclose(values[0], [0.0, 0.0, 30.0], atol=1e-4)
