"""Unit tests for src/utils/calibration_refine.refine_per_frame."""

from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import numpy as np

from src.schemas.calibration import CalibrationResult, CameraFrame
from src.utils.calibration_refine import refine_per_frame


def _broadcast_camera(
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


def _make_keyframe_calibration(n_keyframes: int = 3) -> CalibrationResult:
    """Build a tiny synthetic CalibrationResult with keyframes at frames 0, 2, 4."""
    K, rvec, tvec = _broadcast_camera()
    frames: list[CameraFrame] = []
    for i in range(n_keyframes):
        frames.append(CameraFrame(
            frame=i * 2,
            intrinsic_matrix=K.tolist(),
            rotation_vector=rvec.tolist(),
            translation_vector=tvec.tolist(),
            reprojection_error=0.0,
            num_correspondences=0,
            confidence=1.0,
            tracked_landmark_types=[],
        ))
    return CalibrationResult(shot_id="synthetic", camera_type="static", frames=frames)


def _write_blank_clip(n_frames: int, width: int = 320, height: int = 240) -> Path:
    """Write a tiny all-black .mp4 clip for testing."""
    tmp = Path(tempfile.mkstemp(suffix=".mp4")[1])
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(tmp), fourcc, 25.0, (width, height))
    blank = np.zeros((height, width, 3), dtype=np.uint8)
    for _ in range(n_frames):
        writer.write(blank)
    writer.release()
    return tmp


class TestRefinePerFrame:
    def test_returns_one_frame_per_video_frame(self):
        cal = _make_keyframe_calibration(n_keyframes=3)
        clip_path = _write_blank_clip(n_frames=5)
        try:
            refined, diags = refine_per_frame(cal, clip_path)
        finally:
            clip_path.unlink(missing_ok=True)
        # Expect one CameraFrame per video frame
        assert len(refined.frames) == 5
        # Frame indices are 0..4 (video-frame indexing, not keyframe)
        assert [cf.frame for cf in refined.frames] == [0, 1, 2, 3, 4]
        # Each output frame is a valid CameraFrame
        for cf in refined.frames:
            assert len(cf.rotation_vector) == 3
            assert len(cf.translation_vector) == 3
            assert len(cf.intrinsic_matrix) == 3

    def test_seed_preserved_when_no_lines_detected(self):
        # Black frames have no lines, so refine_with_lines() will reject
        # the refinement and return the seed unchanged.
        cal = _make_keyframe_calibration(n_keyframes=2)
        clip_path = _write_blank_clip(n_frames=3)
        try:
            refined, diags = refine_per_frame(cal, clip_path)
        finally:
            clip_path.unlink(missing_ok=True)
        # All ICL diagnostics should report not accepted (no lines)
        for d in diags:
            assert d.accepted is False
        # The output rvec at frame 0 should equal the seed (which is
        # the keyframe at frame 0 since it's an exact match).
        seed_rvec = np.array(cal.frames[0].rotation_vector)
        out_rvec = np.array(refined.frames[0].rotation_vector)
        np.testing.assert_allclose(out_rvec, seed_rvec, atol=1e-9)

    def test_empty_calibration_passthrough(self):
        cal = CalibrationResult(shot_id="empty", camera_type="static", frames=[])
        clip_path = _write_blank_clip(n_frames=3)
        try:
            refined, diags = refine_per_frame(cal, clip_path)
        finally:
            clip_path.unlink(missing_ok=True)
        assert refined.frames == []
        assert diags == []

    def test_missing_clip_passthrough(self):
        cal = _make_keyframe_calibration(n_keyframes=2)
        refined, diags = refine_per_frame(cal, Path("/nonexistent/clip.mp4"))
        assert refined is cal
        assert diags == []

    def test_sample_every_skips_frames(self):
        cal = _make_keyframe_calibration(n_keyframes=3)
        clip_path = _write_blank_clip(n_frames=10)
        try:
            refined, _ = refine_per_frame(cal, clip_path, sample_every=3)
        finally:
            clip_path.unlink(missing_ok=True)
        # Every 3rd frame: 0, 3, 6, 9 — 4 frames
        assert len(refined.frames) == 4
        assert [cf.frame for cf in refined.frames] == [0, 3, 6, 9]
