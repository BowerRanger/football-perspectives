"""Unit tests for the static-camera bundle adjustment from detected lines."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.schemas.anchor import LineObservation
from src.utils.pitch_lines_catalogue import LINE_CATALOGUE
from src.utils.static_line_solver import (
    StaticCameraSolution,
    solve_static_camera_from_lines,
)

# Physically valid broadcast pose (mirrors tests/test_anchor_solver.py).
_LOOK = np.array([0.0, 64.0, -30.0])
_LOOK = _LOOK / np.linalg.norm(_LOOK)
_RIGHT = np.array([1.0, 0.0, 0.0])
_DOWN = np.cross(_LOOK, _RIGHT)
R_BASE = np.array([_RIGHT, _DOWN, _LOOK], dtype=float)
C_TRUE = np.array([52.5, -30.0, 30.0])
IMAGE_SIZE = (1920, 1080)
CX, CY = IMAGE_SIZE[0] / 2.0, IMAGE_SIZE[1] / 2.0
FX_TRUE = 1500.0

# Lines visible from the broadcast pose — left penalty box plus halfway,
# touchlines and goal line so the geometry is well spread.
_LINE_NAMES = [
    "left_18yd_front", "left_18yd_near_edge", "left_18yd_far_edge",
    "left_6yd_front", "near_touchline", "left_goal_line", "halfway_line",
]


def _yaw(angle_deg: float) -> np.ndarray:
    a = np.deg2rad(angle_deg)
    Ry = np.array([[np.cos(a), -np.sin(a), 0.0],
                   [np.sin(a), np.cos(a), 0.0],
                   [0.0, 0.0, 1.0]])
    return R_BASE @ Ry.T


def _project(K, R, t, world):
    cam = R @ np.asarray(world, float) + t
    pix = K @ cam
    return float(pix[0] / pix[2]), float(pix[1] / pix[2])


def _make_line(K, R, t, name, alpha=0.2, beta=0.8) -> LineObservation:
    seg = LINE_CATALOGUE[name]
    pa, pb = np.asarray(seg[0]), np.asarray(seg[1])
    A = pa + alpha * (pb - pa)
    B = pa + beta * (pb - pa)
    return LineObservation(
        name=name,
        image_segment=(_project(K, R, t, A), _project(K, R, t, B)),
        world_segment=seg,
    )


def _synthetic_clip(yaws, fxs):
    """Build per-frame clean line observations + per-frame seeds for a
    known static C."""
    per_frame_lines: dict[int, list[LineObservation]] = {}
    per_frame_seeds: dict[int, tuple[np.ndarray, float]] = {}
    for i, (yaw, fx) in enumerate(zip(yaws, fxs)):
        R = _yaw(yaw)
        t = -R @ C_TRUE
        K = np.array([[fx, 0, CX], [0, fx, CY], [0, 0, 1.0]])
        lines = []
        for name in _LINE_NAMES:
            seg = LINE_CATALOGUE[name]
            # only keep lines whose endpoints project in front of the camera
            cam = np.asarray(seg, float) @ R.T + t
            if (cam[:, 2] > 0.1).all():
                lines.append(_make_line(K, R, t, name))
        per_frame_lines[i] = lines
        rvec, _ = cv2.Rodrigues(R)
        per_frame_seeds[i] = (rvec.reshape(3), fx)
    return per_frame_lines, per_frame_seeds


@pytest.mark.unit
def test_solver_recovers_known_static_centre_from_clean_lines():
    yaws = [-6.0, -3.0, 0.0, 3.0, 6.0]
    fxs = [1500.0, 1510.0, 1520.0, 1530.0, 1540.0]
    per_frame_lines, per_frame_seeds = _synthetic_clip(yaws, fxs)

    # Seed C 1.5 m off truth, seeds rvec/fx slightly perturbed.
    c_seed = C_TRUE + np.array([1.5, -1.0, 0.8])
    perturbed = {
        f: (rv + np.deg2rad(1.5), fx * 1.02)
        for f, (rv, fx) in per_frame_seeds.items()
    }

    sol = solve_static_camera_from_lines(
        per_frame_lines, IMAGE_SIZE,
        c_seed=c_seed, lens_seed=(CX, CY, 0.0, 0.0),
        per_frame_seeds=perturbed, lens_model="pinhole_k1k2",
    )

    assert isinstance(sol, StaticCameraSolution)
    # Exactly one camera centre, recovered tight.
    assert np.linalg.norm(sol.camera_centre - C_TRUE) < 0.05
    # Every frame's t satisfies -R.T @ t == the single C.
    for fid, (K, R, t) in sol.per_frame_KRt.items():
        c_frame = -R.T @ t
        assert np.linalg.norm(c_frame - sol.camera_centre) < 1e-6
    # Clean synthetic lines → sub-pixel RMS.
    assert sol.line_rms_mean < 0.05


@pytest.mark.unit
def test_solver_returns_one_centre_for_every_frame():
    yaws = [-5.0, 0.0, 5.0]
    fxs = [1500.0, 1500.0, 1500.0]
    per_frame_lines, per_frame_seeds = _synthetic_clip(yaws, fxs)
    sol = solve_static_camera_from_lines(
        per_frame_lines, IMAGE_SIZE,
        c_seed=C_TRUE + 0.5, lens_seed=(CX, CY, 0.0, 0.0),
        per_frame_seeds=per_frame_seeds, lens_model="pinhole_k1k2",
    )
    assert set(sol.per_frame_KRt.keys()) == set(per_frame_lines.keys())
    assert sol.camera_centre.shape == (3,)
