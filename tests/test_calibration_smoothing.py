"""Unit tests for src/utils/calibration_refine.smooth_calibration_temporally."""

from __future__ import annotations

import cv2
import numpy as np

from src.schemas.calibration import CalibrationResult, CameraFrame
from src.utils.calibration_refine import (
    _hampel_filter_1d,
    _median_filter_1d,
    smooth_calibration_temporally,
)
from src.utils.camera import camera_world_position


def _camera_frame(
    frame: int,
    rvec: np.ndarray,
    tvec: np.ndarray,
    fx: float = 1500.0,
) -> CameraFrame:
    K = [[fx, 0.0, 960.0],
         [0.0, fx, 540.0],
         [0.0, 0.0, 1.0]]
    return CameraFrame(
        frame=frame,
        intrinsic_matrix=K,
        rotation_vector=rvec.tolist(),
        translation_vector=tvec.tolist(),
        reprojection_error=0.0,
        num_correspondences=0,
        confidence=1.0,
        tracked_landmark_types=[],
    )


def _build_calibration(
    rvecs: np.ndarray,
    fxs: np.ndarray,
    cam_pos: np.ndarray,
    frame_indices: list[int] | None = None,
) -> CalibrationResult:
    """Construct a CalibrationResult with a shared camera position.

    ``rvecs`` is ``(N, 3)``; ``fxs`` is ``(N,)``.  Translation is
    derived from each rvec + the shared position.
    """
    n = len(rvecs)
    if frame_indices is None:
        frame_indices = list(range(0, n * 6, 6))
    frames: list[CameraFrame] = []
    for i, fi in enumerate(frame_indices):
        rvec = rvecs[i]
        R, _ = cv2.Rodrigues(rvec)
        tvec = -R @ cam_pos
        frames.append(_camera_frame(fi, rvec, tvec, fx=float(fxs[i])))
    return CalibrationResult(shot_id="test", camera_type="static", frames=frames)


class TestMedianFilter1D:
    def test_window_1_is_identity(self):
        xs = np.array([1.0, 5.0, 3.0, 8.0, 2.0])
        out = _median_filter_1d(xs, window=1)
        np.testing.assert_array_equal(out, xs)

    def test_rejects_single_outlier(self):
        xs = np.array([3.0, 3.0, 3.0, 99.0, 3.0, 3.0, 3.0])
        out = _median_filter_1d(xs, window=5)
        # Outlier at index 3 should be replaced by the median of its
        # 5-window neighbourhood, which is 3.
        assert out[3] == 3.0

    def test_preserves_smooth_ramp(self):
        xs = np.arange(20, dtype=np.float64)
        out = _median_filter_1d(xs, window=5)
        # Median of a centred 5-window of a ramp is the centre value;
        # edge windows narrow but still hit the centre.
        np.testing.assert_allclose(out, xs)


class TestHampelFilter1D:
    def test_smooth_ramp_unchanged(self):
        # A linear ramp has zero local MAD around its centre values so
        # a Hampel filter shouldn't touch it.
        xs = np.arange(20, dtype=np.float64)
        out = _hampel_filter_1d(xs, window=5, k=3.0)
        np.testing.assert_array_equal(out, xs)

    def test_constant_unchanged(self):
        xs = np.full(10, 5.0)
        out = _hampel_filter_1d(xs, window=5, k=3.0)
        np.testing.assert_array_equal(out, xs)

    def test_replaces_single_spike(self):
        xs = np.array([3.0, 3.1, 2.9, 99.0, 3.0, 3.1, 2.9])
        out = _hampel_filter_1d(xs, window=5, k=3.0)
        # Spike at index 3 should be replaced; neighbours preserved
        assert out[3] != 99.0
        assert abs(out[3] - 3.0) < 0.5
        # Non-spike positions are kept exactly
        for i in (0, 1, 2, 4, 5, 6):
            assert out[i] == xs[i]

    def test_preserves_inliers_in_noisy_data(self):
        # Slowly drifting series — hampel should leave it alone
        rng = np.random.default_rng(0)
        xs = np.linspace(0, 10, 20) + rng.normal(0, 0.1, 20)
        out = _hampel_filter_1d(xs, window=5, k=3.0)
        # Most points should be unchanged (no clear outliers)
        unchanged = int(np.sum(out == xs))
        assert unchanged >= 18  # allow up to 2 borderline replacements


class TestSmoothCalibrationTemporally:
    def test_too_few_frames_passthrough(self):
        cam_pos = np.array([52.5, -25.0, 28.0])
        rvecs = np.tile(np.array([1.8, 0.0, 0.0]), (3, 1))
        fxs = np.full(3, 1500.0)
        cal = _build_calibration(rvecs, fxs, cam_pos)
        smoothed = smooth_calibration_temporally(cal, window=5)
        assert smoothed is cal  # unchanged

    def test_rejects_single_keyframe_outlier(self):
        # 7 keyframes, all rvecs ~equal except frame 3 has a wild outlier
        cam_pos = np.array([52.5, -25.0, 28.0])
        baseline = np.array([1.8, -0.05, 0.05])
        rvecs = np.tile(baseline, (7, 1))
        # Outlier at index 3 — 14° off in rvec_z
        rvecs[3] = baseline + np.array([0.0, 0.0, 0.25])
        fxs = np.full(7, 1500.0)
        fxs[3] = 5000.0  # also a focal length spike
        cal = _build_calibration(rvecs, fxs, cam_pos)

        smoothed = smooth_calibration_temporally(cal, window=5)
        # The outlier keyframe should be pulled back to ~baseline
        out_rvec = np.array(smoothed.frames[3].rotation_vector)
        out_fx = smoothed.frames[3].intrinsic_matrix[0][0]
        assert abs(out_rvec[2] - baseline[2]) < 0.01
        assert abs(out_fx - 1500.0) < 1.0

    def test_preserves_smooth_pan(self):
        # Camera panning smoothly: rvec_z increases linearly across keyframes
        cam_pos = np.array([52.5, -25.0, 28.0])
        n = 9
        rvecs = np.tile(np.array([1.8, 0.0, 0.0]), (n, 1))
        rvecs[:, 2] = np.linspace(-0.1, 0.1, n)
        fxs = np.full(n, 1500.0)
        cal = _build_calibration(rvecs, fxs, cam_pos)

        smoothed = smooth_calibration_temporally(cal, window=5)
        smoothed_z = np.array([cf.rotation_vector[2] for cf in smoothed.frames])
        # The smooth ramp should survive the median filter unchanged
        # (median of any 5-window of a linear ramp is the centre value).
        np.testing.assert_allclose(smoothed_z, rvecs[:, 2], atol=1e-9)

    def test_camera_position_preserved(self):
        # After smoothing, every keyframe's recovered camera world
        # position should still equal the input shared position.
        cam_pos = np.array([52.5, -25.0, 28.0])
        n = 7
        rng = np.random.default_rng(0)
        rvecs = np.array([1.8, 0.0, 0.0]) + rng.normal(0, 0.1, size=(n, 3))
        fxs = 1500.0 + rng.normal(0, 50, size=n)
        cal = _build_calibration(rvecs, fxs, cam_pos)

        smoothed = smooth_calibration_temporally(cal, window=5)
        for cf in smoothed.frames:
            recovered = camera_world_position(
                np.asarray(cf.rotation_vector),
                np.asarray(cf.translation_vector),
            )
            assert np.allclose(recovered, cam_pos, atol=1e-6)
