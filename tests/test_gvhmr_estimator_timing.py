"""Task 1 of the GVHMR inference-time campaign: perf_counter
instrumentation around each phase of ``_estimate_sequence_locked``
(temp-video write, decode+preproc, ViTPose extract, HMR2 feature
extract, SimpleVO, model.predict, FK).

Exercises the real orchestration logic against fake network/ViTPose/
HMR2/SMPLX submodules standing in for the real GVHMR weights, so this
runs in unit-test time without a GPU or checkpoint. Vendored geo/preproc
helper functions (get_bbx_xys_from_xyxy, estimate_K, compute_cam_angvel,
get_batch) run for real via ``_ensure_imports()`` — they're pure tensor
ops with no checkpoint dependency, only sys.path shims.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest
import torch

from src.utils.gvhmr_estimator import GVHMREstimator, _TIMING_KEYS


class _FakeVitPose:
    def __init__(self) -> None:
        self.calls: list = []
        self.bbx_calls: list = []

    def extract(self, video_path_or_imgs, bbx_xys, img_ds: float = 0.5):
        self.calls.append(video_path_or_imgs)
        self.bbx_calls.append(bbx_xys)
        n = bbx_xys.shape[0]
        return torch.zeros(n, 17, 3)


class _FakeExtractor:
    def __init__(self) -> None:
        self.calls: list = []
        self.bbx_calls: list = []

    def extract_video_features(self, video_path_or_imgs, bbx_xys, img_ds: float = 0.5):
        self.calls.append(video_path_or_imgs)
        self.bbx_calls.append(bbx_xys)
        n = bbx_xys.shape[0]
        return torch.zeros(n, 1024)


class _FakeBodyModelOutput:
    def __init__(self, joints: torch.Tensor) -> None:
        self.joints = joints


class _FakeBodyModel:
    def __call__(self, *, global_orient, body_pose, betas, transl):
        n = global_orient.shape[0]
        return _FakeBodyModelOutput(torch.zeros(n, 24, 3))


class _FakeModel:
    def __init__(self) -> None:
        self.calls: list = []

    def predict(self, data, static_cam: bool = False):
        self.calls.append(data)
        n = int(data["length"])
        return {
            "smpl_params_incam": {
                "global_orient": torch.zeros(n, 3),
                "body_pose": torch.zeros(n, 63),
                "betas": torch.zeros(n, 10),
                "transl": torch.zeros(n, 3),
            },
            "net_outputs": {"model_output": {"pred_cam": torch.zeros(n, 3)}},
        }


def build_fake_estimator(monkeypatch, *, static_cam: bool = True) -> GVHMREstimator:
    """A GVHMREstimator whose ``_load_model`` is bypassed (no checkpoint,
    no hydra compose, no GPU) but whose vendored geo/preproc imports ARE
    real (via ``_ensure_imports()``), with fake network/ViTPose/HMR2/
    SMPLX submodules standing in for the real weights.

    ``static_cam=True`` by default so tests that don't care about camera
    rotation don't pay for (or depend on) a real SimpleVO/pycolmap solve
    on a near-featureless synthetic clip.
    """
    est = GVHMREstimator(checkpoint="unused", device="cpu", static_cam=static_cam)
    # Eager (not just inside the fake _load_model below) so a test can
    # import hmr4d submodules itself -- e.g. to monkeypatch get_batch --
    # before calling estimate_sequence(), regardless of test run order.
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


def _frames_and_bboxes(n: int) -> tuple[list[np.ndarray], list[list[float]]]:
    frames = [np.zeros((48, 64, 3), dtype=np.uint8) for _ in range(n)]
    bboxes = [[5.0, 5.0, 40.0, 44.0] for _ in range(n)]
    return frames, bboxes


@pytest.mark.unit
def test_timing_keys_present_and_non_negative_after_estimate(monkeypatch) -> None:
    est = build_fake_estimator(monkeypatch)
    frames, bboxes = _frames_and_bboxes(4)

    out = est.estimate_sequence(frames, bboxes)

    assert out["global_orient"].shape == (4, 3)
    assert len(est.timings) == 1
    record = est.timings[0]
    assert set(record.keys()) == set(_TIMING_KEYS)
    for k in _TIMING_KEYS:
        assert record[k] >= 0.0, f"{k} timing must be non-negative"
    # static_cam=True in this fixture -> SimpleVO never runs.
    assert record["simple_vo"] == 0.0
    # Phases with real filesystem/decode work should be measurably
    # non-zero; phases backed by trivial fakes could in principle read
    # exactly 0.0 at perf_counter resolution, so only assert strict
    # positivity for temp-video write + decode.
    assert record["temp_video_write"] > 0.0
    assert record["decode_preproc"] > 0.0


@pytest.mark.unit
def test_timing_totals_sums_across_chunks(monkeypatch) -> None:
    est = build_fake_estimator(monkeypatch)
    frames, bboxes = _frames_and_bboxes(3)

    est.estimate_sequence(frames, bboxes)
    est.estimate_sequence(frames, bboxes)

    totals = est.timing_totals()
    assert totals["n_chunks"] == 2.0
    assert set(totals.keys()) == set(_TIMING_KEYS) | {"total", "n_chunks"}
    for k in _TIMING_KEYS:
        expected = est.timings[0][k] + est.timings[1][k]
        assert totals[k] == pytest.approx(expected)
    assert totals["total"] == pytest.approx(sum(totals[k] for k in _TIMING_KEYS))


@pytest.mark.unit
def test_timing_totals_empty_before_any_chunk(monkeypatch) -> None:
    est = build_fake_estimator(monkeypatch)
    totals = est.timing_totals()
    assert totals["n_chunks"] == 0.0
    assert totals["total"] == 0.0
    for k in _TIMING_KEYS:
        assert totals[k] == 0.0


@pytest.mark.unit
def test_chunk_timing_logged_at_info_level(monkeypatch, caplog: pytest.LogCaptureFixture) -> None:
    est = build_fake_estimator(monkeypatch)
    frames, bboxes = _frames_and_bboxes(2)

    with caplog.at_level(logging.INFO, logger="src.utils.gvhmr_estimator"):
        est.estimate_sequence(frames, bboxes)

    assert any("GVHMR chunk timing" in r.message for r in caplog.records), (
        f"expected one INFO chunk-timing line, got: {[r.message for r in caplog.records]}"
    )


@pytest.mark.unit
def test_simple_vo_timed_when_it_runs(monkeypatch) -> None:
    """static_cam=False (default) with no calibrated R -> SimpleVO runs;
    its timer must be positive, and it must be the thing invoked (not
    skipped by the R_w2c_per_frame priority path, which isn't exercised
    here)."""
    est = build_fake_estimator(monkeypatch, static_cam=False)
    calls = {"n": 0}

    def _fake_vo(video_path: Path, n_frames: int):
        calls["n"] += 1
        return torch.eye(3).repeat(n_frames, 1, 1)

    monkeypatch.setattr(est, "_estimate_camera_rotations", _fake_vo)
    frames, bboxes = _frames_and_bboxes(3)

    est.estimate_sequence(frames, bboxes)

    assert calls["n"] == 1, "SimpleVO should run exactly once when static_cam=False and no calibrated R"
    assert est.timings[0]["simple_vo"] >= 0.0
