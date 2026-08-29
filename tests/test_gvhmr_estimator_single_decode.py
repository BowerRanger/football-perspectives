"""Task 2 of the GVHMR inference-time campaign: a single shared
``get_batch()`` decode feeding both ViTPose and the HMR2 feature
extractor, instead of each independently decoding the temp video.

``legacy_decode=True`` is kept as a bench/regression escape hatch that
restores the original per-extractor path-based decode (used by
``scripts/bench_gvhmr_inference.py`` for the "legacy" mode baseline).

CRITICAL invariant asserted here: ``data["bbx_xys"]`` fed to
``model.predict`` must stay the ORIGINAL ``get_bbx_xys_from_xyxy``
tensor, never the size/position-adjusted tensor ``get_batch()`` returns
alongside the cropped image batch — mixing them up would silently shift
every player's weak-perspective camera parameters.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.utils.gvhmr_estimator import GVHMREstimator
from tests.test_gvhmr_estimator_timing import (
    _FakeBodyModel,
    _FakeExtractor,
    _FakeModel,
    _FakeVitPose,
    _frames_and_bboxes,
)


def build_fake_estimator(monkeypatch, *, static_cam: bool = True) -> GVHMREstimator:
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


@pytest.mark.unit
def test_default_mode_calls_get_batch_exactly_once(monkeypatch) -> None:
    est = build_fake_estimator(monkeypatch)
    frames, bboxes = _frames_and_bboxes(5)

    call_count = {"n": 0}
    import src.utils.gvhmr_estimator as gvhmr_estimator_mod  # noqa: F401
    from hmr4d.utils.preproc import vitfeat_extractor as _ext_mod

    real_get_batch = _ext_mod.get_batch

    def _counting_get_batch(*args, **kwargs):
        call_count["n"] += 1
        return real_get_batch(*args, **kwargs)

    monkeypatch.setattr(_ext_mod, "get_batch", _counting_get_batch)

    est.estimate_sequence(frames, bboxes)

    assert call_count["n"] == 1, "shared-decode mode must call get_batch exactly once per chunk"


@pytest.mark.unit
def test_default_mode_feeds_same_tensor_object_to_both_extractors(monkeypatch) -> None:
    est = build_fake_estimator(monkeypatch)
    frames, bboxes = _frames_and_bboxes(4)

    est.estimate_sequence(frames, bboxes)

    vitpose_arg = est._vitpose.calls[0]
    extractor_arg = est._extractor.calls[0]
    assert isinstance(vitpose_arg, torch.Tensor)
    assert isinstance(extractor_arg, torch.Tensor)
    assert vitpose_arg is extractor_arg, (
        "both extractors should receive the identical decoded tensor, "
        "not independently-decoded copies"
    )


@pytest.mark.unit
def test_default_mode_extractors_receive_adjusted_bbx_xys(monkeypatch) -> None:
    """The bbx_xys handed to the extractors should be get_batch()'s
    ADJUSTED copy (matching the actual crop) -- i.e. exactly what a
    standalone get_batch() call on the same inputs returns -- not the
    original pre-get_batch tensor. Getting this wrong would decode
    ViTPose's heatmap->keypoint postprocessing against the wrong crop
    window."""
    est = build_fake_estimator(monkeypatch)
    frames, bboxes = _frames_and_bboxes(4)

    from hmr4d.utils.geo.hmr_cam import get_bbx_xys_from_xyxy
    from hmr4d.utils.preproc.vitfeat_extractor import get_batch as real_get_batch

    bbx_xyxy = torch.tensor(bboxes, dtype=torch.float32)
    original_bbx_xys = get_bbx_xys_from_xyxy(bbx_xyxy, base_enlarge=1.2).float()

    est.estimate_sequence(frames, bboxes)

    # Independently reproduce what get_batch() returns for these exact
    # inputs (same frames written to a fresh temp video via the same
    # helper the estimator itself uses).
    tmp_video = GVHMREstimator._write_temp_video(frames, 30.0)
    try:
        _, expected_adjusted = real_get_batch(str(tmp_video), original_bbx_xys, img_ds=0.5)
    finally:
        tmp_video.unlink(missing_ok=True)

    vitpose_bbx = est._vitpose.bbx_calls[0]
    extractor_bbx = est._extractor.bbx_calls[0]
    assert torch.allclose(vitpose_bbx, expected_adjusted, atol=1e-6)
    assert torch.allclose(extractor_bbx, expected_adjusted, atol=1e-6)


@pytest.mark.unit
def test_data_bbx_xys_stays_original_not_adjusted(monkeypatch) -> None:
    """CRITICAL invariant: model.predict's data["bbx_xys"] must be the
    original get_bbx_xys_from_xyxy tensor, not get_batch()'s adjusted
    copy -- in both shared-decode and legacy_decode modes."""
    from hmr4d.utils.geo.hmr_cam import get_bbx_xys_from_xyxy

    bboxes = [[5.0, 5.0, 40.0, 44.0]] * 4
    bbx_xyxy = torch.tensor(bboxes, dtype=torch.float32)
    expected_bbx_xys = get_bbx_xys_from_xyxy(bbx_xyxy, base_enlarge=1.2).float()

    for legacy in (False, True):
        est = build_fake_estimator(monkeypatch)
        frames, _ = _frames_and_bboxes(4)

        est.estimate_sequence(frames, bboxes, legacy_decode=legacy)

        seen_data = est._model.calls[0]
        assert torch.allclose(seen_data["bbx_xys"].cpu(), expected_bbx_xys), (
            f"legacy_decode={legacy}: data['bbx_xys'] drifted from the "
            "original get_bbx_xys_from_xyxy tensor"
        )


@pytest.mark.unit
def test_legacy_decode_restores_path_based_calls(monkeypatch) -> None:
    est = build_fake_estimator(monkeypatch)
    frames, bboxes = _frames_and_bboxes(3)

    est.estimate_sequence(frames, bboxes, legacy_decode=True)

    assert isinstance(est._vitpose.calls[0], str), "legacy_decode should pass a path, not a tensor"
    assert isinstance(est._extractor.calls[0], str)
    # decode_preproc timer stays 0.0 in legacy mode -- decode happens
    # inside each extractor's own (untimed-separately) call instead.
    assert est.timings[0]["decode_preproc"] == 0.0


@pytest.mark.unit
def test_legacy_decode_does_not_call_shared_get_batch(monkeypatch) -> None:
    """legacy_decode=True should NOT take the estimator's own single
    shared get_batch() shortcut at all -- decode is fully delegated back
    to each extractor's ``extract(str(path), bbx_xys)`` call, exactly as
    before Task 2. (The vendored VitPoseExtractor.extract /
    Extractor.extract_video_features each call get_batch() internally
    when given a path, but that's real-model-only vendored behaviour our
    fakes intentionally bypass here -- covered instead by
    test_legacy_decode_restores_path_based_calls, which pins the path-
    based call shape our code hands them.)"""
    est = build_fake_estimator(monkeypatch)
    frames, bboxes = _frames_and_bboxes(3)

    call_count = {"n": 0}
    from hmr4d.utils.preproc import vitfeat_extractor as _ext_mod

    real_get_batch = _ext_mod.get_batch

    def _counting_get_batch(*args, **kwargs):
        call_count["n"] += 1
        return real_get_batch(*args, **kwargs)

    monkeypatch.setattr(_ext_mod, "get_batch", _counting_get_batch)

    est.estimate_sequence(frames, bboxes, legacy_decode=True)

    assert call_count["n"] == 0, (
        "legacy_decode=True must not call the estimator's shared get_batch shortcut"
    )


@pytest.mark.unit
def test_legacy_and_shared_decode_produce_allclose_outputs(monkeypatch) -> None:
    """The bench harness's parity gate: legacy vs shared-decode kp2d and
    global_orient/etc. outputs must match within a tight tolerance --
    they run the exact same underlying computation, just via different
    call paths (path-per-extractor vs single shared decode)."""
    frames, bboxes = _frames_and_bboxes(5)

    est_legacy = build_fake_estimator(monkeypatch)
    out_legacy = est_legacy.estimate_sequence(frames, bboxes, legacy_decode=True)

    est_shared = build_fake_estimator(monkeypatch)
    out_shared = est_shared.estimate_sequence(frames, bboxes, legacy_decode=False)

    for key in ("kp2d", "bbx_xys"):
        np.testing.assert_allclose(
            out_legacy[key], out_shared[key], atol=1e-6,
            err_msg=f"{key} diverged between legacy_decode and shared-decode",
        )
