"""Unit tests for the camera-centre profile diagnostic."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.schemas.anchor import LineObservation
from src.utils.pitch_lines_catalogue import LINE_CATALOGUE
from src.utils.static_c_profile import (
    CProfileResult,
    make_c_grid,
    profile_camera_centre,
)

_LOOK = np.array([0.0, 64.0, -30.0])
_LOOK = _LOOK / np.linalg.norm(_LOOK)
_RIGHT = np.array([1.0, 0.0, 0.0])
_DOWN = np.cross(_LOOK, _RIGHT)
R_BASE = np.array([_RIGHT, _DOWN, _LOOK], dtype=float)
C_TRUE = np.array([52.5, -30.0, 30.0])
IMAGE_SIZE = (1920, 1080)
CX, CY = IMAGE_SIZE[0] / 2.0, IMAGE_SIZE[1] / 2.0
_LINE_NAMES = [
    "left_18yd_front", "left_18yd_near_edge", "left_18yd_far_edge",
    "left_6yd_front", "near_touchline", "left_goal_line", "halfway_line",
]


def _yaw(angle_deg):
    a = np.deg2rad(angle_deg)
    Ry = np.array([[np.cos(a), -np.sin(a), 0.0],
                   [np.sin(a), np.cos(a), 0.0],
                   [0.0, 0.0, 1.0]])
    return R_BASE @ Ry.T


def _project(K, R, t, world):
    cam = R @ np.asarray(world, float) + t
    pix = K @ cam
    return float(pix[0] / pix[2]), float(pix[1] / pix[2])


def _synthetic_clip(yaws, fx=1500.0):
    per_frame_lines = {}
    per_frame_bootstrap = {}
    for i, yaw in enumerate(yaws):
        R = _yaw(yaw)
        t = -R @ C_TRUE
        K = np.array([[fx, 0, CX], [0, fx, CY], [0, 0, 1.0]])
        lines = []
        for name in _LINE_NAMES:
            seg = LINE_CATALOGUE[name]
            cam = np.asarray(seg, float) @ R.T + t
            if (cam[:, 2] > 0.1).all():
                pa = np.asarray(seg[0]) + 0.2 * (np.asarray(seg[1]) - np.asarray(seg[0]))
                pb = np.asarray(seg[0]) + 0.8 * (np.asarray(seg[1]) - np.asarray(seg[0]))
                lines.append(LineObservation(
                    name=name,
                    image_segment=(_project(K, R, t, pa), _project(K, R, t, pb)),
                    world_segment=seg,
                ))
        per_frame_lines[i] = lines
        rvec, _ = cv2.Rodrigues(R)
        per_frame_bootstrap[i] = (rvec.reshape(3), fx)
    return per_frame_lines, per_frame_bootstrap


@pytest.mark.unit
def test_make_c_grid_spans_the_requested_box():
    grid = make_c_grid(np.array([10.0, 20.0, 30.0]), extent_m=6.0, n_steps=5)
    assert grid.shape == (125, 3)
    assert np.isclose(grid[:, 0].min(), 4.0)
    assert np.isclose(grid[:, 0].max(), 16.0)
    # The centre is on the grid.
    assert np.any(np.all(np.isclose(grid, [10.0, 20.0, 30.0]), axis=1))


@pytest.mark.unit
def test_profile_argmin_lands_on_true_centre():
    per_frame_lines, bootstrap = _synthetic_clip([-6.0, -2.0, 2.0, 6.0])
    grid = make_c_grid(C_TRUE, extent_m=4.0, n_steps=5)
    result = profile_camera_centre(
        per_frame_lines, IMAGE_SIZE,
        c_grid=grid, lens_seed=(CX, CY, 0.0, 0.0),
        per_frame_bootstrap=bootstrap,
    )
    assert isinstance(result, CProfileResult)
    # Clean synthetic lines → the true C must be (near) the argmin.
    assert np.linalg.norm(result.argmin_c - C_TRUE) < 1.1
    # Mean RMS at the argmin is sub-pixel.
    best_idx = int(np.argmin(result.mean_rms))
    assert result.mean_rms[best_idx] < 0.1
    # Per-frame seeds at the argmin are returned for every frame.
    assert set(result.per_frame_seeds.keys()) == set(per_frame_lines.keys())
