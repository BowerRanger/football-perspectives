"""Tests for the centralised world<->image projection helper."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.utils.camera_projection import (
    project_world_to_image,
    undistort_pixel,
)


@pytest.mark.unit
def test_project_round_trip_zero_distortion():
    """With zero distortion, helper output matches the bare K @ (RX + t)."""
    K = np.array([[1500.0, 0.0, 960.0], [0.0, 1500.0, 540.0], [0.0, 0.0, 1.0]])
    R = np.eye(3)
    t = np.array([0.0, 0.0, 30.0])
    world = np.array([[0.0, 0.0, 0.0], [10.0, 5.0, 0.0]])
    proj = project_world_to_image(
        K, R, t, distortion=(0.0, 0.0), world_points=world,
    )
    cam = world @ R.T + t
    expected = (K @ cam.T).T
    expected = expected[:, :2] / expected[:, 2:]
    assert np.allclose(proj, expected, atol=1e-4)


@pytest.mark.unit
def test_project_with_distortion_matches_cv2():
    """Helper agrees with the underlying ``cv2.projectPoints`` for non-zero k."""
    K = np.array([[1500.0, 0.0, 960.0], [0.0, 1500.0, 540.0], [0.0, 0.0, 1.0]])
    R = np.eye(3)
    t = np.array([0.0, 0.0, 30.0])
    world = np.array([[0.0, 0.0, 0.0], [10.0, 5.0, 0.0]])
    proj = project_world_to_image(
        K, R, t, distortion=(0.1, -0.02), world_points=world,
    )
    rvec, _ = cv2.Rodrigues(R)
    expected, _ = cv2.projectPoints(
        world.reshape(-1, 1, 3), rvec, t.reshape(3, 1), K,
        np.array([0.1, -0.02, 0.0, 0.0, 0.0]),
    )
    assert np.allclose(proj, expected.reshape(-1, 2), atol=1e-4)


@pytest.mark.unit
def test_undistort_pixel_inverts_distortion():
    """Undistorting a distorted projection lands at the linear projection."""
    K = np.array([[1500.0, 0.0, 960.0], [0.0, 1500.0, 540.0], [0.0, 0.0, 1.0]])
    R = np.eye(3)
    t = np.array([0.0, 0.0, 30.0])
    world = np.array([[10.0, 5.0, 0.0]])
    distorted = project_world_to_image(
        K, R, t, distortion=(0.1, -0.02), world_points=world,
    )
    undist = undistort_pixel(distorted[0], K, distortion=(0.1, -0.02))
    cam = R @ world[0] + t
    linear = (K @ cam)[:2] / cam[2]
    assert np.allclose(undist, linear, atol=0.5)
