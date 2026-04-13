"""Unit tests for src/utils/vp_calibration.py and pitch_line_detector.py."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.utils.pitch_line_detector import DetectedLine
from src.utils.vp_calibration import (
    calibration_from_vanishing_points,
    cluster_lines_by_orientation,
    vanishing_point_from_lines,
)


def _line_through(p1: tuple[float, float], p2: tuple[float, float]) -> DetectedLine:
    return DetectedLine(p1[0], p1[1], p2[0], p2[1])


class TestVanishingPointFromLines:
    def test_intersecting_lines_recover_known_vp(self):
        # Lines that all pass through (1000, 500)
        vp = (1000.0, 500.0)
        lines = [
            _line_through((100, 100), vp),
            _line_through((100, 800), vp),
            _line_through((1900, 200), vp),
            _line_through((1900, 800), vp),
        ]
        recovered = vanishing_point_from_lines(lines)
        assert np.allclose(recovered, vp, atol=0.5)

    def test_too_few_lines_returns_nan(self):
        result = vanishing_point_from_lines([_line_through((0, 0), (10, 10))])
        assert np.all(np.isnan(result))


class TestClusterLinesByOrientation:
    def test_separates_horizontal_and_vertical(self):
        horizontals = [DetectedLine(100, 100 + i * 10, 1000, 100 + i * 10) for i in range(5)]
        verticals = [DetectedLine(100 + i * 10, 100, 100 + i * 10, 1000) for i in range(5)]
        clusters = cluster_lines_by_orientation(horizontals + verticals, n_clusters=2, angle_tol_deg=8.0)
        assert len(clusters) == 2
        sizes = sorted(len(c) for c in clusters)
        assert sizes == [5, 5]


class TestCalibrationFromVanishingPoints:
    def test_recovers_synthetic_camera(self):
        # Build a synthetic camera, project two known orthogonal world
        # directions to get the VPs, then check we recover the same K
        # and R.
        K = np.array([[1500.0, 0.0, 960.0],
                      [0.0, 1500.0, 540.0],
                      [0.0, 0.0, 1.0]], dtype=np.float64)
        # Off-axis target so the touchline (1,0,0) VP isn't at infinity.
        cam_pos = np.array([20.0, -20.0, 25.0], dtype=np.float64)
        target = np.array([60.0, 34.0, 0.0], dtype=np.float64)
        forward = target - cam_pos
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        R_truth = np.stack([right, -up, forward], axis=0)
        rvec_truth, _ = cv2.Rodrigues(R_truth)
        tvec_truth = -R_truth @ cam_pos

        # Touchline VP: project (1, 0, 0) into camera frame
        d_touch_cam = R_truth @ np.array([1.0, 0.0, 0.0])
        vp_touch_h = K @ d_touch_cam
        vp_touch = vp_touch_h[:2] / vp_touch_h[2]

        d_goal_cam = R_truth @ np.array([0.0, 1.0, 0.0])
        vp_goal_h = K @ d_goal_cam
        vp_goal = vp_goal_h[:2] / vp_goal_h[2]

        result = calibration_from_vanishing_points(
            vp_touch, vp_goal, image_size=(1920, 1080),
            camera_position_world=cam_pos,
        )
        assert result is not None
        assert abs(result.focal_length - 1500.0) < 5.0
        # Recovered R should match truth (within numerical error)
        assert np.allclose(result.R, R_truth, atol=1e-3)

    def test_non_orthogonal_vps_returns_none(self):
        # Two VPs both at the principal point — degenerate
        result = calibration_from_vanishing_points(
            np.array([960.0, 540.0]),
            np.array([960.0, 540.0]),
            image_size=(1920, 1080),
            camera_position_world=np.array([52.5, -20.0, 25.0]),
        )
        assert result is None
