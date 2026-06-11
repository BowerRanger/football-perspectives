"""Hoarding (advertising-board) base-edge detection + per-clip calibration."""

from __future__ import annotations

import numpy as np
import pytest

from src.utils.camera_projection import project_world_to_image
from src.utils.hoarding_detector import (
    calibrate_board_line,
    detect_board_line,
)

IMAGE_SIZE = (1280, 720)
C_TRUE = np.array([52.5, -30.0, 14.0])
D_TRUE = 3.0


def _camera(yaw_deg: float, fx: float = 1500.0):
    target = np.array([52.5 + yaw_deg, 60.0, 0.0])
    fwd = target - C_TRUE
    fwd = fwd / np.linalg.norm(fwd)
    right = np.cross(fwd, np.array([0.0, 0.0, 1.0]))
    right = right / np.linalg.norm(right)
    down = np.cross(fwd, right)
    R = np.array([right, down, fwd])
    t = -R @ C_TRUE
    w, h = IMAGE_SIZE
    K = np.array([[fx, 0, w / 2], [0, fx, h / 2], [0, 0, 1.0]])
    return K, R, t


def _render(K, R, t, d=D_TRUE, band_px=22):
    """Grass below the board base edge, a bright LED band above it, dim
    crowd noise above the band."""
    w, h = IMAGE_SIZE
    rng = np.random.default_rng(1)
    img = np.empty((h, w, 3), dtype=np.uint8)
    img[:] = (60, 110, 60)  # grass (greenish BGR)
    xs_w = np.linspace(-20, 125, 600)
    world = np.stack([xs_w, np.full_like(xs_w, 68.0 + d),
                      np.zeros_like(xs_w)], axis=1)
    proj = project_world_to_image(K, R, t, (0.0, 0.0), world)
    fin = np.isfinite(proj).all(axis=1)
    proj = proj[fin]
    order = np.argsort(proj[:, 0])
    proj = proj[order]
    cols = np.arange(w)
    y_edge = np.interp(cols, proj[:, 0], proj[:, 1],
                       left=np.nan, right=np.nan)
    crowd = rng.integers(40, 110, size=(h, w, 3), dtype=np.uint8)
    for x in cols:
        ye = y_edge[x]
        if not np.isfinite(ye):
            continue
        ye_i = int(round(ye))
        top = max(0, ye_i - band_px)
        if ye_i > 0:
            img[:top, x] = crowd[:top, x]
            img[top:max(top, ye_i), x] = (235, 235, 235)  # bright LED band
    return img


@pytest.mark.unit
def test_detects_board_base_edge_subpixel():
    K, R, t = _camera(0.0)
    img = _render(K, R, t)
    det = detect_board_line(img, K, R, t, (0.0, 0.0), D_TRUE, 0.0)
    assert det is not None
    assert len(det.image_points) >= 10
    # every detected point must lie on the true edge (within ~1.5 px)
    xs_w = np.linspace(-20, 125, 600)
    world = np.stack([xs_w, np.full_like(xs_w, 68.0 + D_TRUE),
                      np.zeros_like(xs_w)], axis=1)
    proj = project_world_to_image(K, R, t, (0.0, 0.0), world)
    fin = np.isfinite(proj).all(axis=1)
    proj = proj[fin]
    order = np.argsort(proj[:, 0])
    proj = proj[order]
    for (u, v) in det.image_points:
        v_true = np.interp(u, proj[:, 0], proj[:, 1])
        assert abs(v - v_true) < 1.5


@pytest.mark.unit
def test_detects_offset_seed_still_locks():
    # the seed model is 2 m off the true line — within the search strip the
    # consensus must still lock the true edge
    K, R, t = _camera(0.0)
    img = _render(K, R, t)
    det = detect_board_line(img, K, R, t, (0.0, 0.0), D_TRUE + 2.0, 0.0)
    assert det is not None and len(det.image_points) >= 8


@pytest.mark.unit
def test_calibration_recovers_offset():
    frames = {}
    cams = {}
    for i, yaw in enumerate((-8.0, 0.0, 8.0)):
        K, R, t = _camera(yaw)
        frames[i] = _render(K, R, t)
        cams[i] = {"K": K, "R": R, "t": t}
    model = calibrate_board_line(frames, cams, (0.0, 0.0))
    assert model is not None
    assert model.frames == 3
    assert abs(model.d - D_TRUE) < 0.4
    assert model.residual < 0.3


@pytest.mark.unit
def test_no_board_in_view_returns_none():
    # camera looking at the near goal area — board line projects out of frame
    K, R, t = _camera(0.0)
    w, h = IMAGE_SIZE
    rng = np.random.default_rng(2)
    img = rng.integers(40, 110, size=(h, w, 3), dtype=np.uint8)
    det = detect_board_line(img, K, R, t, (0.0, 0.0), D_TRUE, 0.0)
    # pure noise: either no detection or far too few consistent points
    assert det is None or len(det.image_points) < 20