"""Task 3 of the GVHMR inference-time campaign: calibrated camera R
(``R_w2c_per_frame`` on ``estimate_sequence``, ``per_frame_R`` on
``run_on_track``) as a drop-in replacement for GVHMR's internal
SimpleVO camera-rotation estimate.

``compute_cam_angvel`` only uses relative rotation between consecutive
frames (``R_w2c[1:] @ R_w2c[:-1].T``), so a calibrated pitch-frame
``R_w2c`` is a direct substitute for SimpleVO's own ``R_w2c`` estimate --
both are OpenCV world-to-camera rotations.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.utils.gvhmr_estimator import GVHMREstimator, run_on_track
from tests.test_gvhmr_estimator_timing import (
    _FakeBodyModel,
    _FakeExtractor,
    _FakeModel,
    _FakeVitPose,
    _frames_and_bboxes,
)


def build_fake_estimator(monkeypatch, *, static_cam: bool = False) -> GVHMREstimator:
    """static_cam=False by default here (opposite of the timing/decode
    test files' default) -- these tests are specifically about the
    R_w2c selection priority between calibrated R, SimpleVO, and the
    static-cam identity fallback, so SimpleVO must be reachable."""
    est = GVHMREstimator(checkpoint="unused", device="cpu", static_cam=static_cam)
    est._ensure_imports()

    def _fake_load_model() -> None:
        if est._model is not None:
            return
        est._ensure_imports()
        est._vitpose = _FakeVitPose()
        est._extractor = _FakeExtractor()
        est._model = _FakeModel()
        est._body_model = _FakeBodyModel()

    monkeypatch.setattr(est, "_load_model", _fake_load_model)
    return est


def _spy_simple_vo(monkeypatch, est: GVHMREstimator) -> dict:
    calls = {"n": 0}

    def _fake_vo(video_path, n_frames):
        calls["n"] += 1
        return torch.eye(3).repeat(n_frames, 1, 1)

    monkeypatch.setattr(est, "_estimate_camera_rotations", _fake_vo)
    return calls


@pytest.mark.unit
def test_calibrated_r_skips_simple_vo(monkeypatch) -> None:
    est = build_fake_estimator(monkeypatch, static_cam=False)
    vo_calls = _spy_simple_vo(monkeypatch, est)
    frames, bboxes = _frames_and_bboxes(4)
    R_w2c = np.tile(np.eye(3, dtype=np.float32), (4, 1, 1))

    est.estimate_sequence(frames, bboxes, R_w2c_per_frame=R_w2c)

    assert vo_calls["n"] == 0, "calibrated R_w2c_per_frame must skip SimpleVO entirely"
    assert est.timings[0]["simple_vo"] == 0.0


@pytest.mark.unit
def test_no_calibrated_r_falls_back_to_simple_vo_when_not_static(monkeypatch) -> None:
    est = build_fake_estimator(monkeypatch, static_cam=False)
    vo_calls = _spy_simple_vo(monkeypatch, est)
    frames, bboxes = _frames_and_bboxes(4)

    est.estimate_sequence(frames, bboxes, R_w2c_per_frame=None)

    assert vo_calls["n"] == 1, "no calibrated R + static_cam=False must run SimpleVO"


@pytest.mark.unit
def test_static_cam_semantics_unchanged_without_calibrated_r(monkeypatch) -> None:
    """static_cam=True must still skip SimpleVO (existing behaviour,
    unrelated to the new calibrated-R path) when no R is supplied."""
    est = build_fake_estimator(monkeypatch, static_cam=True)
    vo_calls = _spy_simple_vo(monkeypatch, est)
    frames, bboxes = _frames_and_bboxes(4)

    est.estimate_sequence(frames, bboxes, R_w2c_per_frame=None)

    assert vo_calls["n"] == 0
    assert est.timings[0]["simple_vo"] == 0.0


@pytest.mark.unit
def test_mismatched_length_calibrated_r_falls_back(monkeypatch) -> None:
    """A calibrated R whose length doesn't match n_frames (e.g. the
    caller passed the whole-track array instead of slicing per chunk)
    must NOT be used -- fall back to the existing SimpleVO/static path
    instead of silently truncating or erroring."""
    est = build_fake_estimator(monkeypatch, static_cam=False)
    vo_calls = _spy_simple_vo(monkeypatch, est)
    frames, bboxes = _frames_and_bboxes(4)
    wrong_length_R = np.tile(np.eye(3, dtype=np.float32), (7, 1, 1))  # 7 != 4

    est.estimate_sequence(frames, bboxes, R_w2c_per_frame=wrong_length_R)

    assert vo_calls["n"] == 1, "mismatched-length R must fall back to SimpleVO"


@pytest.mark.unit
def test_calibrated_r_feeds_cam_angvel_from_supplied_rotations(monkeypatch) -> None:
    """A non-trivial (rotating) calibrated R_w2c should produce a
    non-zero cam_angvel in data['cam_angvel'] fed to model.predict --
    proving the calibrated array actually drives cam_angvel instead of
    being computed-then-discarded."""
    est = build_fake_estimator(monkeypatch, static_cam=False)
    _spy_simple_vo(monkeypatch, est)
    n = 5
    frames, bboxes = _frames_and_bboxes(n)

    # A small per-frame yaw rotation about Z -> non-trivial cam_angvel.
    angles = np.linspace(0.0, 0.2, n)
    R_w2c = np.stack(
        [
            np.array(
                [[np.cos(a), -np.sin(a), 0.0], [np.sin(a), np.cos(a), 0.0], [0.0, 0.0, 1.0]],
                dtype=np.float32,
            )
            for a in angles
        ]
    )

    est.estimate_sequence(frames, bboxes, R_w2c_per_frame=R_w2c)

    data = est._model.calls[0]
    cam_angvel = data["cam_angvel"]
    assert cam_angvel.shape == (n, 6)
    identity_angvel = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    assert not torch.allclose(cam_angvel[0], identity_angvel, atol=1e-4), (
        "rotating calibrated R_w2c should yield non-identity cam_angvel"
    )


# ---------------------------------------------------------------------------
# run_on_track: per-chunk R slicing, mirroring per_frame_K's existing test
# coverage.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_run_on_track_slices_per_frame_r_per_chunk(monkeypatch, tmp_path) -> None:
    import cv2

    n = 7
    w, h = 64, 48
    video_path = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (w, h))
    for i in range(n):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[:, :, i % 3] = 150
        writer.write(frame)
    writer.release()

    track_frames = [(i, (5, 5, 40, 44)) for i in range(n)]
    per_frame_R = np.stack([np.eye(3, dtype=np.float32) * (i + 1) for i in range(n)])

    seen_R_lengths: list[int] = []
    seen_R_values: list[np.ndarray] = []

    def _fake_estimate_sequence(self, frames_bgr, bboxes, fps=30.0, K_per_frame=None,
                                 R_w2c_per_frame=None, legacy_decode=False):
        m = len(frames_bgr)
        seen_R_lengths.append(0 if R_w2c_per_frame is None else len(R_w2c_per_frame))
        if R_w2c_per_frame is not None:
            seen_R_values.append(np.asarray(R_w2c_per_frame).copy())
        return {
            "global_orient": np.zeros((m, 3), dtype=np.float32),
            "body_pose": np.zeros((m, 63), dtype=np.float32),
            "betas": np.zeros((m, 10), dtype=np.float32),
            "transl": np.zeros((m, 3), dtype=np.float32),
            "kp2d": np.zeros((m, 17, 3), dtype=np.float32),
        }

    monkeypatch.setattr(GVHMREstimator, "estimate_sequence", _fake_estimate_sequence)
    est = GVHMREstimator(checkpoint="unused", device="cpu")

    # run_on_track checks checkpoint.exists() unconditionally before ever
    # calling estimate_sequence, regardless of the estimator= override --
    # write a stub file so that gate passes without a real GVHMR checkpoint.
    fake_ckpt = tmp_path / "fake_ckpt.pt"
    fake_ckpt.write_bytes(b"stub")

    out = run_on_track(
        track_frames=track_frames,
        video_path=video_path,
        checkpoint=fake_ckpt,
        device="cpu",
        batch_size=16,
        max_sequence_length=3,  # forces 3 chunks: 3+3+1
        estimator=est,
        per_frame_R=per_frame_R,
    )

    assert out["thetas"].shape[0] == n
    assert seen_R_lengths == [3, 3, 1]
    np.testing.assert_array_equal(seen_R_values[0], per_frame_R[0:3])
    np.testing.assert_array_equal(seen_R_values[1], per_frame_R[3:6])
    np.testing.assert_array_equal(seen_R_values[2], per_frame_R[6:7])


@pytest.mark.unit
def test_run_on_track_none_per_frame_r_stays_none_per_chunk(monkeypatch, tmp_path) -> None:
    import cv2

    n = 4
    w, h = 64, 48
    video_path = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (w, h))
    for i in range(n):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()

    track_frames = [(i, (5, 5, 40, 44)) for i in range(n)]
    seen_R: list[object] = []

    def _fake_estimate_sequence(self, frames_bgr, bboxes, fps=30.0, K_per_frame=None,
                                 R_w2c_per_frame=None, legacy_decode=False):
        m = len(frames_bgr)
        seen_R.append(R_w2c_per_frame)
        return {
            "global_orient": np.zeros((m, 3), dtype=np.float32),
            "body_pose": np.zeros((m, 63), dtype=np.float32),
            "betas": np.zeros((m, 10), dtype=np.float32),
            "transl": np.zeros((m, 3), dtype=np.float32),
            "kp2d": np.zeros((m, 17, 3), dtype=np.float32),
        }

    monkeypatch.setattr(GVHMREstimator, "estimate_sequence", _fake_estimate_sequence)
    est = GVHMREstimator(checkpoint="unused", device="cpu")
    fake_ckpt = tmp_path / "fake_ckpt.pt"
    fake_ckpt.write_bytes(b"stub")

    run_on_track(
        track_frames=track_frames,
        video_path=video_path,
        checkpoint=fake_ckpt,
        device="cpu",
        batch_size=16,
        max_sequence_length=120,
        estimator=est,
        per_frame_R=None,
    )

    assert seen_R == [None]


@pytest.mark.unit
def test_run_on_track_legacy_decode_passthrough(monkeypatch, tmp_path) -> None:
    import cv2

    n = 3
    w, h = 64, 48
    video_path = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (w, h))
    for i in range(n):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()

    track_frames = [(i, (5, 5, 40, 44)) for i in range(n)]
    seen_legacy: list[bool] = []

    def _fake_estimate_sequence(self, frames_bgr, bboxes, fps=30.0, K_per_frame=None,
                                 R_w2c_per_frame=None, legacy_decode=False):
        m = len(frames_bgr)
        seen_legacy.append(legacy_decode)
        return {
            "global_orient": np.zeros((m, 3), dtype=np.float32),
            "body_pose": np.zeros((m, 63), dtype=np.float32),
            "betas": np.zeros((m, 10), dtype=np.float32),
            "transl": np.zeros((m, 3), dtype=np.float32),
            "kp2d": np.zeros((m, 17, 3), dtype=np.float32),
        }

    monkeypatch.setattr(GVHMREstimator, "estimate_sequence", _fake_estimate_sequence)
    est = GVHMREstimator(checkpoint="unused", device="cpu")
    fake_ckpt = tmp_path / "fake_ckpt.pt"
    fake_ckpt.write_bytes(b"stub")

    run_on_track(
        track_frames=track_frames,
        video_path=video_path,
        checkpoint=fake_ckpt,
        device="cpu",
        batch_size=16,
        max_sequence_length=120,
        estimator=est,
        legacy_decode=True,
    )

    assert seen_legacy == [True]
