"""Unit tests for src/utils/pitch_lines.py and src/utils/calibration_debug.py."""

from __future__ import annotations

import cv2
import numpy as np

from src.utils.calibration_debug import draw_overlay, project_pitch_lines
from src.utils.pitch_lines import pitch_polylines


def _broadcast_camera() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    K = np.array([[1500.0, 0.0, 960.0],
                  [0.0, 1500.0, 540.0],
                  [0.0, 0.0, 1.0]], dtype=np.float64)
    cam_pos = np.array([52.5, -20.0, 25.0], dtype=np.float64)
    target = np.array([52.5, 34.0, 0.0], dtype=np.float64)
    forward = target - cam_pos
    forward /= np.linalg.norm(forward)
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    R = np.stack([right, -up, forward], axis=0)
    rvec, _ = cv2.Rodrigues(R)
    tvec = -R @ cam_pos
    return K, rvec.reshape(3), tvec.reshape(3)


class TestPitchPolylines:
    def test_returns_expected_count(self):
        polys = pitch_polylines()
        # 4 sidelines + halfway + circle + 2 18yd + 2 6yd + 2 arcs + 2 goals
        assert len(polys) == 14

    def test_all_polylines_on_or_near_pitch(self):
        polys = pitch_polylines()
        for poly in polys:
            assert poly.shape[1] == 3
            assert poly.shape[0] >= 2
            # x in [0, 105], y in [0, 68], z in [0, 2.44]
            assert np.all(poly[:, 0] >= -0.01) and np.all(poly[:, 0] <= 105.01)
            assert np.all(poly[:, 1] >= -0.01) and np.all(poly[:, 1] <= 68.01)
            assert np.all(poly[:, 2] >= -0.01) and np.all(poly[:, 2] <= 2.45)

    def test_centre_circle_radius(self):
        polys = pitch_polylines()
        circle = polys[5]
        centre = np.array([52.5, 34.0, 0.0])
        radii = np.linalg.norm(circle - centre, axis=1)
        assert np.allclose(radii, 9.15, atol=0.01)


class TestProjectPitchLines:
    def test_centre_circle_projects_inside_image(self):
        K, rvec, tvec = _broadcast_camera()
        polylines = project_pitch_lines(K, rvec, tvec, image_size=(1920, 1080))
        circle = polylines[5]
        assert circle.shape[0] > 0
        # Circle should land somewhere in the frame
        xs, ys = circle[:, 0], circle[:, 1]
        assert xs.min() > 0 and xs.max() < 1920
        assert ys.min() > 0 and ys.max() < 1080

    def test_behind_camera_polylines_skipped(self):
        # Camera behind the goal line looking down at the pitch from
        # behind near touchline — far touchline should still project,
        # but a polyline behind the camera plane should be empty.
        K = np.array([[1500.0, 0.0, 960.0],
                      [0.0, 1500.0, 540.0],
                      [0.0, 0.0, 1.0]], dtype=np.float64)
        cam_pos = np.array([52.5, -10.0, 5.0], dtype=np.float64)
        target = np.array([52.5, 34.0, 0.0], dtype=np.float64)
        forward = target - cam_pos
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        R = np.stack([right, -up, forward], axis=0)
        rvec, _ = cv2.Rodrigues(R)
        tvec = -R @ cam_pos
        polylines = project_pitch_lines(K, rvec, tvec, image_size=(1920, 1080))
        # Each polyline should be either empty or fully ahead of the
        # camera — no NaN/inf coordinates leaking through.
        for poly in polylines:
            assert poly.dtype == np.int32
            if poly.size > 0:
                assert np.all(np.isfinite(poly.astype(np.float64)))


class TestDrawOverlay:
    def test_returns_same_shape(self):
        K, rvec, tvec = _broadcast_camera()
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        out = draw_overlay(frame, K, rvec, tvec, label="test f0")
        assert out.shape == frame.shape
        assert out.dtype == np.uint8
        # Should have drawn some non-zero pixels for the lines + label
        assert np.any(out > 0)

    def test_input_not_mutated(self):
        K, rvec, tvec = _broadcast_camera()
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        original = frame.copy()
        _ = draw_overlay(frame, K, rvec, tvec)
        np.testing.assert_array_equal(frame, original)
