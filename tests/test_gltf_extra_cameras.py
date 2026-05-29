from __future__ import annotations

import numpy as np
import pytest

from src.schemas.camera_track import CameraFrame, CameraTrack
from src.utils.gltf_builder import SceneBundle, build_glb


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
