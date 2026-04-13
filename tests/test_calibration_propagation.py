"""Unit tests for src/utils/calibration_propagation.py."""

from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import numpy as np

from src.schemas.calibration import CalibrationResult, CameraFrame
from src.utils.calibration_propagation import (
    _camera_frame_to_homography,
    _find_gaps,
    _homography_to_camera_frame,
    propagate_calibration_across_gaps,
)


def _build_camera(
    cam_pos: tuple[float, float, float] = (52.5, -25.0, 28.0),
    target: tuple[float, float, float] = (52.5, 34.0, 0.0),
    fx: float = 1500.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    K = np.array([[fx, 0.0, 960.0],
                  [0.0, fx, 540.0],
                  [0.0, 0.0, 1.0]], dtype=np.float64)
    cam = np.array(cam_pos, dtype=np.float64)
    tgt = np.array(target, dtype=np.float64)
    forward = tgt - cam
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    R = np.stack([right, -up, forward], axis=0)
    rvec, _ = cv2.Rodrigues(R)
    tvec = -R @ cam
    return K, rvec.reshape(3), tvec.reshape(3)


def _camera_frame(frame_idx: int, K, rvec, tvec) -> CameraFrame:
    return CameraFrame(
        frame=frame_idx,
        intrinsic_matrix=K.tolist(),
        rotation_vector=rvec.reshape(3).tolist(),
        translation_vector=tvec.reshape(3).tolist(),
        reprojection_error=0.0,
        num_correspondences=0,
        confidence=1.0,
        tracked_landmark_types=[],
    )


def _write_synthetic_clip(
    path: Path,
    n_frames: int,
    width: int = 320,
    height: int = 240,
) -> None:
    """Write a short .mp4 with random-but-trackable content (noise patches
    stamped at different positions so optical flow has something to grab)."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 25.0, (width, height))
    rng = np.random.default_rng(0)
    base = rng.integers(0, 255, (height, width, 3), dtype=np.uint8)
    for i in range(n_frames):
        # Slight pan: roll the image horizontally by `i` pixels.
        shifted = np.roll(base, i, axis=1)
        writer.write(shifted)
    writer.release()


class TestFindGaps:
    def test_no_gaps(self):
        assert _find_gaps([0, 1, 2], 3) == []

    def test_interior_gap(self):
        assert _find_gaps([0, 1, 4, 5], 6) == [(2, 3)]

    def test_leading_gap(self):
        assert _find_gaps([3, 4], 5) == [(0, 2)]

    def test_trailing_gap(self):
        assert _find_gaps([0, 1], 5) == [(2, 4)]

    def test_multiple_gaps(self):
        assert _find_gaps([2, 5, 9], 12) == [(0, 1), (3, 4), (6, 8), (10, 11)]


class TestHomographyRoundTrip:
    def test_camera_frame_to_homography_and_back(self):
        """Decomposing a freshly-computed pitch homography must recover
        a CameraFrame whose ``[r1 | r2 | t]`` matches the original
        (up to the SVD orthogonalisation)."""
        K, rvec, tvec = _build_camera()
        cf_a = _camera_frame(0, K, rvec, tvec)
        H = _camera_frame_to_homography(cf_a)
        cf_b = _homography_to_camera_frame(H, anchor_cf=cf_a, frame_idx=0)
        # Rotation delta between the two frames should be small
        R_a = cv2.Rodrigues(rvec.reshape(3))[0]
        R_b = cv2.Rodrigues(np.asarray(cf_b.rotation_vector).reshape(3))[0]
        R_delta = R_a @ R_b.T
        angle = np.arccos(np.clip((np.trace(R_delta) - 1) / 2, -1.0, 1.0))
        assert float(np.degrees(angle)) < 0.5

    def test_homography_chain_composes(self):
        """Two homographies composed sequentially must match a single
        direct homography representing their combined transform."""
        K, rvec, tvec = _build_camera()
        cf_a = _camera_frame(0, K, rvec, tvec)
        H_a = _camera_frame_to_homography(cf_a)
        # Apply a small 2D affine shift (simulating a pan-induced
        # image-space transform) and compose.
        M = np.array([[1.0, 0.0, 3.0],
                      [0.0, 1.0, 2.0],
                      [0.0, 0.0, 1.0]], dtype=np.float64)
        H_b = M @ H_a
        cf_b = _homography_to_camera_frame(H_b, anchor_cf=cf_a, frame_idx=1)
        # The recovered CameraFrame should project a known pitch
        # point to approximately (original projection + shift).
        pitch_pt = np.array([[50.0, 30.0, 0.0]], dtype=np.float64)
        proj_a, _ = cv2.projectPoints(
            pitch_pt, rvec.reshape(3), tvec.reshape(3), K, None,
        )
        expected = proj_a.reshape(2) + np.array([3.0, 2.0])
        proj_b, _ = cv2.projectPoints(
            pitch_pt,
            np.asarray(cf_b.rotation_vector).reshape(3),
            np.asarray(cf_b.translation_vector).reshape(3),
            np.asarray(cf_b.intrinsic_matrix),
            None,
        )
        actual = proj_b.reshape(2)
        assert np.allclose(actual, expected, atol=1.0)


class TestPropagation:
    def test_empty_calibration_passthrough(self):
        cal = CalibrationResult(shot_id="t", camera_type="static", frames=[])
        clip = Path(tempfile.mkstemp(suffix=".mp4")[1])
        try:
            _write_synthetic_clip(clip, n_frames=5)
            new_cal, stats = propagate_calibration_across_gaps(cal, clip)
        finally:
            clip.unlink(missing_ok=True)
        assert new_cal.frames == []
        assert stats.n_gaps == 0

    def test_no_gaps_passthrough(self):
        K, rvec, tvec = _build_camera()
        cal = CalibrationResult(
            shot_id="t", camera_type="static",
            frames=[_camera_frame(i, K, rvec, tvec) for i in range(5)],
        )
        clip = Path(tempfile.mkstemp(suffix=".mp4")[1])
        try:
            _write_synthetic_clip(clip, n_frames=5)
            new_cal, stats = propagate_calibration_across_gaps(cal, clip)
        finally:
            clip.unlink(missing_ok=True)
        assert len(new_cal.frames) == 5
        assert stats.n_gaps == 0
        assert stats.n_filled == 0

    def test_missing_clip_returns_input(self):
        K, rvec, tvec = _build_camera()
        cal = CalibrationResult(
            shot_id="t", camera_type="static",
            frames=[_camera_frame(0, K, rvec, tvec)],
        )
        new_cal, stats = propagate_calibration_across_gaps(
            cal, Path("/nonexistent/clip.mp4"),
        )
        assert new_cal is cal
        assert stats.n_gaps == 0
