"""Unit tests for the detect-all-frames orchestration helper."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.schemas.anchor import LineObservation
from src.utils.camera_projection import project_world_to_image
from src.utils.line_camera_refine import PITCH_LINE_CATALOGUE, detect_lines_for_frames

_LOOK = np.array([0.0, 64.0, -30.0])
_LOOK = _LOOK / np.linalg.norm(_LOOK)
_RIGHT = np.array([1.0, 0.0, 0.0])
_DOWN = np.cross(_LOOK, _RIGHT)
R_BASE = np.array([_RIGHT, _DOWN, _LOOK], dtype=float)
C_TRUE = np.array([52.5, -30.0, 30.0])
IMAGE_SIZE = (1280, 720)


def _camera(fx=900.0):
    w, h = IMAGE_SIZE
    K = np.array([[fx, 0, w / 2], [0, fx, h / 2], [0, 0, 1.0]])
    R = R_BASE
    t = -R @ C_TRUE
    return K, R, t


def _draw_pitch_lines_frame(K, R, t, line_names):
    """Render a green frame with the named catalogue lines painted as
    ~5 px white stripes — enough for the ridge detector to lock on."""
    w, h = IMAGE_SIZE
    img = np.full((h, w, 3), (60, 110, 60), dtype=np.uint8)
    for name in line_names:
        seg = np.array(PITCH_LINE_CATALOGUE[name], dtype=float)
        cam = seg @ R.T + t
        if (cam[:, 2] <= 0.1).any():
            continue
        proj = project_world_to_image(K, R, t, (0.0, 0.0), seg)
        a = tuple(int(round(v)) for v in proj[0])
        b = tuple(int(round(v)) for v in proj[1])
        cv2.line(img, a, b, (255, 255, 255), thickness=5)
    return img


@pytest.mark.unit
def test_detect_lines_for_frames_returns_only_well_covered_frames():
    K, R, t = _camera()
    line_names = [
        "left_18yd_front", "left_18yd_near_edge", "left_18yd_far_edge",
        "left_6yd_front", "near_touchline",
    ]
    good = _draw_pitch_lines_frame(K, R, t, line_names)
    blank = np.full((IMAGE_SIZE[1], IMAGE_SIZE[0], 3), (60, 110, 60), dtype=np.uint8)

    frames_bgr = {0: good, 1: blank}
    cameras = {
        0: {"K": K, "R": R, "t": t},
        1: {"K": K, "R": R, "t": t},
        # frame 2 has a camera but no frame image — must be skipped silently
        2: {"K": K, "R": R, "t": t},
    }
    out = detect_lines_for_frames(frames_bgr, cameras, (0.0, 0.0))

    # The blank frame yields no lines and is excluded; frame 2 has no image.
    assert 1 not in out
    assert 2 not in out
    # The well-drawn frame yields >= 2 LineObservations.
    assert 0 in out
    assert len(out[0]) >= 2
    assert all(isinstance(ln, LineObservation) for ln in out[0])
    assert all(ln.world_segment is not None for ln in out[0])
