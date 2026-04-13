import numpy as np
import cv2
import pytest

from src.utils.triangulation import (
    weighted_dlt,
    ransac_triangulate,
    compute_reprojection_errors,
    temporal_smooth_savgol,
    enforce_bone_lengths,
    snap_feet_to_ground,
)
from src.stages.triangulation import _choose_methods
from src.utils.camera import build_projection_matrix


def _strict(*shot_ids):
    """Build a fake strict-views list with N entries (only shot_id matters here)."""
    return [(sid, None, None, None, None, None) for sid in shot_ids]


def _ext(*shot_ids):
    return [(sid, None, None, None, None, None) for sid in shot_ids]


class TestChooseMethods:
    def test_raw_eligibility(self):
        view_data = [
            (_strict("a", "b"), []),  # multi
            (_strict("a"), []),        # single (one strict shot)
            ([], _ext("a")),           # single (extrapolated only)
            ([], []),                  # nothing
        ]
        result = _choose_methods(view_data, allow_single_shot=True, hysteresis=1)
        assert result == ["multi", "single", "single", None]

    def test_single_shot_disabled(self):
        view_data = [
            (_strict("a", "b"), []),
            (_strict("a"), []),
        ]
        result = _choose_methods(view_data, allow_single_shot=False, hysteresis=1)
        assert result == ["multi", None]

    def test_suppresses_short_single_island_in_multi_run(self):
        view_data = [
            (_strict("a", "b"), []),
            (_strict("a", "b"), []),
            (_strict("a"), []),         # 1-frame single island
            (_strict("a", "b"), []),
            (_strict("a", "b"), []),
        ]
        result = _choose_methods(view_data, allow_single_shot=True, hysteresis=3)
        assert result == ["multi", "multi", None, "multi", "multi"]

    def test_suppresses_two_frame_single_island(self):
        view_data = [
            (_strict("a", "b"), []),
            (_strict("a"), []),
            (_strict("a"), []),
            (_strict("a", "b"), []),
        ]
        result = _choose_methods(view_data, allow_single_shot=True, hysteresis=3)
        assert result == ["multi", None, None, "multi"]

    def test_keeps_long_single_run(self):
        view_data = [
            (_strict("a", "b"), []),
            (_strict("a"), []),
            (_strict("a"), []),
            (_strict("a"), []),
            (_strict("a", "b"), []),
        ]
        # 3-frame run is NOT shorter than hysteresis=3 → kept
        result = _choose_methods(view_data, allow_single_shot=True, hysteresis=3)
        assert result == ["multi", "single", "single", "single", "multi"]

    def test_does_not_suppress_at_sequence_boundary(self):
        # Single island at start has no left neighbour → kept
        view_data = [
            (_strict("a"), []),
            (_strict("a", "b"), []),
            (_strict("a", "b"), []),
        ]
        result = _choose_methods(view_data, allow_single_shot=True, hysteresis=3)
        assert result == ["single", "multi", "multi"]


def _two_cameras():
    """Two synthetic cameras looking at the pitch from different angles."""
    K = np.array([[1500, 0, 960], [0, 1500, 540], [0, 0, 1]], dtype=np.float64)

    rvec1 = np.array([0.05, 0.15, 0.0], dtype=np.float64)
    tvec1 = np.array([-52.5, -34.0, 60.0], dtype=np.float64)
    P1 = build_projection_matrix(K, rvec1, tvec1)

    rvec2 = np.array([0.1, -0.2, 0.05], dtype=np.float64)
    tvec2 = np.array([-52.5, -100.0, 55.0], dtype=np.float64)
    P2 = build_projection_matrix(K, rvec2, tvec2)

    return [P1, P2], K, [(rvec1, tvec1), (rvec2, tvec2)]


def _project(P, pt_3d):
    """Project a 3D point through a 3x4 projection matrix."""
    pt_h = np.append(pt_3d, 1.0)
    proj = P @ pt_h
    return (proj[:2] / proj[2]).astype(np.float64)


class TestWeightedDLT:
    def test_triangulates_known_point(self):
        projections, _, _ = _two_cameras()
        pt_true = np.array([30.0, 20.0, 0.0])
        pts_2d = [_project(P, pt_true) for P in projections]

        result = weighted_dlt(projections, pts_2d, [1.0, 1.0])
        assert np.allclose(result, pt_true, atol=0.05)

    def test_higher_weight_pulls_result(self):
        projections, _, _ = _two_cameras()
        pt_true = np.array([50.0, 30.0, 1.0])
        pts_2d = [_project(P, pt_true) for P in projections]
        # Add noise to second view
        pts_2d[1] += np.array([5.0, 5.0])

        result_equal = weighted_dlt(projections, pts_2d, [1.0, 1.0])
        result_biased = weighted_dlt(projections, pts_2d, [10.0, 0.1])
        # Biased result should be closer to the clean first view
        err_equal = np.linalg.norm(result_equal - pt_true)
        err_biased = np.linalg.norm(result_biased - pt_true)
        assert err_biased < err_equal


class TestRANSACTriangulate:
    def test_two_clean_views(self):
        projections, _, _ = _two_cameras()
        pt_true = np.array([40.0, 25.0, 0.0])
        pts_2d = [_project(P, pt_true) for P in projections]

        pt, err, nv = ransac_triangulate(projections, pts_2d, [1.0, 1.0])
        assert np.allclose(pt, pt_true, atol=0.1)
        assert nv == 2
        assert err < 1.0

    def test_rejects_outlier_view(self):
        projections, _, _ = _two_cameras()
        # Add a third "camera" with garbage data
        P3 = projections[0].copy()
        pt_true = np.array([60.0, 34.0, 0.0])
        pts_2d = [_project(P, pt_true) for P in projections]
        pts_2d.append(np.array([100.0, 100.0]))  # outlier
        projections_3 = projections + [P3]

        pt, err, nv = ransac_triangulate(
            projections_3, pts_2d, [1.0, 1.0, 1.0], threshold=10.0
        )
        assert nv == 2  # outlier rejected
        assert np.allclose(pt, pt_true, atol=0.5)

    def test_returns_nan_for_single_view(self):
        projections, _, _ = _two_cameras()
        pt, err, nv = ransac_triangulate(
            [projections[0]], [np.array([500, 300])], [1.0]
        )
        assert np.all(np.isnan(pt))
        assert nv == 0


class TestReprojectionErrors:
    def test_zero_for_perfect_projection(self):
        projections, _, _ = _two_cameras()
        pt_true = np.array([50.0, 34.0, 0.0])
        pts_2d = [_project(P, pt_true) for P in projections]

        errs = compute_reprojection_errors(pt_true, projections, pts_2d)
        assert np.all(errs < 0.01)


class TestTemporalSmoothing:
    def test_smooths_noisy_trajectory(self):
        rng = np.random.default_rng(42)
        n_frames = 50
        positions = np.zeros((n_frames, 17, 3), dtype=np.float32)
        # Create a smooth ground truth
        for j in range(17):
            for d in range(3):
                positions[:, j, d] = np.sin(np.linspace(0, 2 * np.pi, n_frames)) * 10

        noisy = positions + rng.normal(0, 1.0, positions.shape).astype(np.float32)
        smoothed = temporal_smooth_savgol(noisy, window=7, order=3)

        # Smoothed should be closer to true than noisy
        err_noisy = np.nanmean(np.abs(noisy - positions))
        err_smooth = np.nanmean(np.abs(smoothed - positions))
        assert err_smooth < err_noisy

    def test_handles_nan_gaps(self):
        positions = np.ones((20, 17, 3), dtype=np.float32) * 5.0
        positions[5:8, :, :] = np.nan  # gap
        result = temporal_smooth_savgol(positions, window=7, order=3)
        # Non-NaN frames should still be valid
        assert not np.any(np.isnan(result[0]))
        assert not np.any(np.isnan(result[15]))

    def test_fills_short_gaps(self):
        positions = np.ones((30, 17, 3), dtype=np.float32) * 5.0
        positions[10:13, :, :] = np.nan  # 3-frame gap, bounded on both sides
        result = temporal_smooth_savgol(
            positions, window=11, order=2, max_gap_fill=5,
        )
        # 3-frame gap ≤ max_gap_fill → filled
        assert not np.any(np.isnan(result[10:13]))
        # Filled values should be close to surrounding constant 5.0
        assert np.allclose(result[10:13], 5.0, atol=0.01)

    def test_preserves_long_gaps(self):
        positions = np.ones((40, 17, 3), dtype=np.float32) * 5.0
        positions[10:25, :, :] = np.nan  # 15-frame gap, > max_gap_fill
        result = temporal_smooth_savgol(
            positions, window=11, order=2, max_gap_fill=5,
        )
        # Long gap stays NaN
        assert np.all(np.isnan(result[10:25]))
        # Surrounding frames still smoothed (non-NaN)
        assert not np.any(np.isnan(result[0:10]))
        assert not np.any(np.isnan(result[25:]))

    def test_does_not_extrapolate_leading_trailing_nans(self):
        positions = np.ones((30, 17, 3), dtype=np.float32) * 5.0
        positions[:3, :, :] = np.nan   # leading
        positions[-3:, :, :] = np.nan  # trailing
        result = temporal_smooth_savgol(
            positions, window=11, order=2, max_gap_fill=5,
        )
        assert np.all(np.isnan(result[:3]))
        assert np.all(np.isnan(result[-3:]))

    def test_short_sequence_unchanged(self):
        positions = np.ones((3, 17, 3), dtype=np.float32)
        result = temporal_smooth_savgol(positions, window=7, order=3)
        np.testing.assert_array_equal(result, positions)


class TestBoneLengthEnforcement:
    def test_clamps_stretched_bone(self):
        positions = np.zeros((10, 17, 3), dtype=np.float32)
        # Set up a consistent shoulder-to-elbow distance of 0.3m
        for f in range(10):
            positions[f, 5] = [50.0, 30.0, 1.0]  # left_shoulder
            positions[f, 7] = [50.3, 30.0, 1.0]  # left_elbow, 0.3m away

        # Stretch one frame's bone to 1.0m (>20% deviation)
        positions[5, 7] = [51.0, 30.0, 1.0]

        result = enforce_bone_lengths(positions, tolerance=0.2)
        bone_len = np.linalg.norm(result[5, 7] - result[5, 5])
        # Should be clamped back to median (~0.3)
        assert abs(bone_len - 0.3) < 0.01


class TestFootGroundSnap:
    def test_snaps_stationary_feet(self):
        positions = np.zeros((10, 17, 3), dtype=np.float32)
        # Left ankle (index 15) is stationary at z=0.05 (slightly above ground)
        for f in range(10):
            positions[f, 15] = [50.0, 30.0, 0.05]

        result = snap_feet_to_ground(positions, velocity_threshold=0.1)
        # Should be snapped to z=0
        for f in range(1, 9):  # edges don't have velocity
            assert result[f, 15, 2] == pytest.approx(0.0)

    def test_preserves_moving_feet(self):
        positions = np.zeros((10, 17, 3), dtype=np.float32)
        # Left ankle moving fast
        for f in range(10):
            positions[f, 15] = [50.0 + f * 0.5, 30.0, 0.3]

        result = snap_feet_to_ground(positions, velocity_threshold=0.1)
        # Should NOT be snapped (velocity > threshold)
        for f in range(1, 9):
            assert result[f, 15, 2] == pytest.approx(0.3)
