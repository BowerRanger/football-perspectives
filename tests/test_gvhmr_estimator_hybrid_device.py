"""Task 1 of the hybrid-device shim campaign: ViTPose/HMR2 feature
extraction can run on a device different from the main GVHMR transformer
(``extractor_device`` on ``GVHMREstimator``/``run_on_track``). On Apple
Silicon this lets extraction use MPS (~4-5x over CPU) while the
transformer -- whose RoPE implementation SIGABRTs on MPS -- stays on the
main device.

Exercises the real orchestration logic against fake network/ViTPose/
HMR2/SMPLX submodules standing in for the real GVHMR weights (same
pattern as ``test_gvhmr_estimator_timing.py`` /
``test_gvhmr_estimator_single_decode.py``), so this runs in unit-test
time without a GPU or checkpoint. This machine has a real MPS backend
(Apple Silicon), so the hybrid-split tests use genuine ``device="mps"``
tensors/redirects rather than mocking MPS itself -- only
``torch.backends.mps.is_available`` / ``torch.cuda.is_available`` are
monkeypatched, and only for the ``_normalize_extractor_device`` matrix
tests where unavailable-backend behaviour must be exercised
deterministically regardless of the host.
"""

from __future__ import annotations

import contextlib

import pytest
import torch

from src.utils.gvhmr_estimator import (
    GVHMREstimator,
    _TIMING_KEYS,
    _assert_predict_inputs_on,
    _normalize_device,
    _normalize_extractor_device,
)
from tests.test_gvhmr_estimator_timing import (
    _FakeBodyModel,
    _FakeExtractor,
    _FakeModel,
    _FakeVitPose,
    _frames_and_bboxes,
)
from tests.test_gvhmr_estimator_timing import build_fake_estimator as _build_plain_estimator

_MPS_AVAILABLE = torch.backends.mps.is_available()
_requires_mps = pytest.mark.skipif(
    not _MPS_AVAILABLE, reason="MPS backend not available on this machine"
)


# ---------------------------------------------------------------------------
# Hybrid-aware fakes: extend the shared campaign fixtures with .to()-able
# attrs (.pose / .extractor) and a settable _fp_device, mirroring what
# _load_model's explicit post-construction placement does on the real
# VitPoseExtractor/Extractor instances.
# ---------------------------------------------------------------------------


class _FakeDeviceHolder:
    """Minimal .to()/.eval()-able stand-in for a real nn.Module submodule
    (VitPoseExtractor.pose / Extractor.extractor), so _load_model's
    explicit placement code and _run_extractor_phase's fallback path have
    something real to call .to()/.eval() on."""

    def __init__(self, device: str) -> None:
        self.device_history: list[str] = [device]

    def to(self, device):
        self.device_history.append(str(device))
        return self

    def eval(self):
        return self

    @property
    def current_device(self) -> str:
        return self.device_history[-1]


class _FakeVitPoseHybrid(_FakeVitPose):
    """_FakeVitPose extended with a .pose holder, a settable _fp_device,
    and output tensors genuinely placed on _fp_device (so a broken
    to-main hop in the code under test would legitimately fail a test
    asserting device.type == main, rather than passing by accident)."""

    def __init__(self, device: str) -> None:
        super().__init__()
        self.pose = _FakeDeviceHolder(device)
        self._fp_device = device
        self.fail_next = False

    def extract(self, video_path_or_imgs, bbx_xys, img_ds: float = 0.5):
        self.calls.append(video_path_or_imgs)
        self.bbx_calls.append(bbx_xys)
        if self.fail_next:
            self.fail_next = False
            raise RuntimeError("simulated extractor-device failure")
        n = bbx_xys.shape[0]
        return torch.zeros(n, 17, 3, device=self._fp_device)


class _FakeExtractorHybrid(_FakeExtractor):
    def __init__(self, device: str) -> None:
        super().__init__()
        self.extractor = _FakeDeviceHolder(device)
        self._fp_device = device
        self.fail_next = False

    def extract_video_features(self, video_path_or_imgs, bbx_xys, img_ds: float = 0.5):
        self.calls.append(video_path_or_imgs)
        self.bbx_calls.append(bbx_xys)
        if self.fail_next:
            self.fail_next = False
            raise RuntimeError("simulated extractor-device failure")
        n = bbx_xys.shape[0]
        return torch.zeros(n, 1024, device=self._fp_device)


def build_fake_hybrid_estimator(
    monkeypatch,
    *,
    device: str = "cpu",
    extractor_device: str = "cpu",
    static_cam: bool = True,
    vitpose_cls=_FakeVitPoseHybrid,
    extractor_cls=_FakeExtractorHybrid,
) -> GVHMREstimator:
    """A GVHMREstimator whose ``_load_model`` is bypassed like the other
    campaign fixtures (no checkpoint, no hydra compose, no GPU), but
    which ALSO performs the real device-resolution step ``_load_model``
    would (``_normalize_extractor_device``) and wires the hybrid-aware
    fakes' ``.pose``/``.extractor``/``_fp_device`` the same way the real
    placement code in ``_load_model`` does -- so the fallback/placement
    logic under test runs exactly as it would against real preprocessors.
    """
    est = GVHMREstimator(
        checkpoint="unused", device=device, static_cam=static_cam,
        extractor_device=extractor_device,
    )
    est._ensure_imports()

    def _fake_load_model() -> None:
        if est._model is not None:
            return
        est._ensure_imports()
        est.resolved_extractor_device = _normalize_extractor_device(
            est._extractor_device_requested, est._device
        )
        est._vitpose = vitpose_cls(est.resolved_extractor_device)
        est._extractor = extractor_cls(est.resolved_extractor_device)
        est._model = _FakeModel()
        est._body_model = _FakeBodyModel()

    monkeypatch.setattr(est, "_load_model", _fake_load_model)
    return est


# ---------------------------------------------------------------------------
# 1. _normalize_extractor_device matrix
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_normalize_extractor_device_auto_prefers_mps_when_main_is_cpu(monkeypatch) -> None:
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert _normalize_extractor_device("auto", "cpu") == "mps"


@pytest.mark.unit
def test_normalize_extractor_device_auto_falls_back_to_main_without_mps_backend(monkeypatch) -> None:
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert _normalize_extractor_device("auto", "cpu") == "cpu"


@pytest.mark.unit
def test_normalize_extractor_device_auto_never_splits_on_cuda_main(monkeypatch) -> None:
    # Even with MPS "available", a cuda main device must win -- GPU boxes
    # never split extraction onto a second device.
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert _normalize_extractor_device("auto", "cuda:0") == "cuda:0"
    assert _normalize_extractor_device("auto", "cuda") == "cuda"


@pytest.mark.unit
def test_normalize_extractor_device_auto_resolves_against_real_cuda_probe(monkeypatch) -> None:
    """Covers the torch.cuda half of the matrix: feed
    _normalize_device('auto')'s own cuda resolution straight into
    _normalize_extractor_device."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    main = _normalize_device("auto")
    assert main == "cuda:0"
    assert _normalize_extractor_device("auto", main) == "cuda:0"


@pytest.mark.unit
def test_normalize_extractor_device_explicit_cpu_passes_through_unchanged() -> None:
    assert _normalize_extractor_device("cpu", "cpu") == "cpu"
    assert _normalize_extractor_device("cpu", "mps") == "cpu"
    assert _normalize_extractor_device("cpu", "cuda:0") == "cpu"


@pytest.mark.unit
def test_normalize_extractor_device_explicit_mps_raises_without_backend(monkeypatch) -> None:
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="mps"):
        _normalize_extractor_device("mps", "cpu")


@pytest.mark.unit
def test_normalize_extractor_device_explicit_mps_passes_when_backend_available(monkeypatch) -> None:
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert _normalize_extractor_device("mps", "cpu") == "mps"


# ---------------------------------------------------------------------------
# 2. extractor_device="cpu" -> bit-identical to pre-split behaviour
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_extractor_device_cpu_matches_pre_split_behaviour_with_original_fakes(monkeypatch) -> None:
    """Reuses the ORIGINAL (non-hybrid) fakes verbatim -- proving the new
    split logic doesn't even engage on the default extractor_device="cpu"
    path, since these fakes have no .pose/.extractor/_fp_device at all."""
    est = _build_plain_estimator(monkeypatch)  # device="cpu"; extractor_device defaults to "cpu"
    frames, bboxes = _frames_and_bboxes(4)

    out = est.estimate_sequence(frames, bboxes)

    assert out["global_orient"].shape == (4, 3)
    assert est.resolved_extractor_device == "cpu"
    assert est.extractor_fallback_count == 0
    assert set(est.timings[0].keys()) == set(_TIMING_KEYS), "no new timing keys were introduced"
    for k in _TIMING_KEYS:
        assert est.timings[0][k] >= 0.0


@pytest.mark.unit
def test_extractor_device_cpu_no_fallback_bookkeeping_across_chunks(monkeypatch) -> None:
    est = _build_plain_estimator(monkeypatch)
    frames, bboxes = _frames_and_bboxes(3)

    est.estimate_sequence(frames, bboxes)
    est.estimate_sequence(frames, bboxes)

    assert est.extractor_fallback_count == 0
    assert est.resolved_extractor_device == "cpu"
    assert len(est.timings) == 2


# ---------------------------------------------------------------------------
# 3. Crash-safety: _assert_predict_inputs_on + the "all on main" crossing
# ---------------------------------------------------------------------------


@pytest.mark.unit
@_requires_mps
def test_predict_inputs_all_on_main_after_hybrid_split(monkeypatch) -> None:
    """Integration: with extractor_device='mps' != main='cpu', the data
    dict that actually reaches model.predict() must have every tensor
    moved back to main -- the second crossing point in the binding
    design. The hybrid fakes genuinely place their output on 'mps', so
    this would fail for real if the .to(main) hop were missing."""
    est = build_fake_hybrid_estimator(monkeypatch, device="cpu", extractor_device="mps")
    frames, bboxes = _frames_and_bboxes(3)

    est.estimate_sequence(frames, bboxes)

    assert est.resolved_extractor_device == "mps"
    seen_data = est._model.calls[0]
    for key, value in seen_data.items():
        if isinstance(value, torch.Tensor):
            assert value.device.type == "cpu", f"{key} is still on {value.device}"


@pytest.mark.unit
def test_assert_predict_inputs_on_passes_when_everything_on_main() -> None:
    model = torch.nn.Linear(2, 2)  # real nn.Module -> .parameters() works
    data = {
        "length": torch.tensor(3),
        "kp2d": torch.zeros(3, 17, 3),
        "note": "not a tensor, must be ignored",
    }
    _assert_predict_inputs_on("cpu", data, model)  # must not raise


@pytest.mark.unit
def test_assert_predict_inputs_on_skips_model_check_for_fake_without_parameters() -> None:
    """A test double without .parameters() (e.g. _FakeModel) must not
    make the guard raise -- only tensors actually present in `data` are
    checked in that case."""
    data = {"kp2d": torch.zeros(3, 17, 3)}
    _assert_predict_inputs_on("cpu", data, _FakeModel())  # must not raise


@pytest.mark.unit
@_requires_mps
def test_assert_predict_inputs_on_raises_on_fabricated_tensor_mismatch() -> None:
    model = torch.nn.Linear(2, 2)  # stays on cpu
    data = {"kp2d": torch.zeros(3, 17, 3, device="mps")}

    with pytest.raises(RuntimeError, match="kp2d"):
        _assert_predict_inputs_on("cpu", data, model)


@pytest.mark.unit
@_requires_mps
def test_assert_predict_inputs_on_raises_on_fabricated_model_device_mismatch() -> None:
    model = torch.nn.Linear(2, 2).to("mps")
    data = {"kp2d": torch.zeros(3, 17, 3)}  # cpu

    with pytest.raises(RuntimeError, match="model parameters"):
        _assert_predict_inputs_on("cpu", data, model)


# ---------------------------------------------------------------------------
# 4. Fallback policy: RuntimeError from a split extract -> retry once on
#    main, permanent pin, no re-attempt on later chunks.
# ---------------------------------------------------------------------------


@pytest.mark.unit
@_requires_mps
def test_fallback_retries_once_and_permanently_pins_resolved_device_to_main(monkeypatch) -> None:
    est = build_fake_hybrid_estimator(monkeypatch, device="cpu", extractor_device="mps")
    frames, bboxes = _frames_and_bboxes(3)

    # Force _load_model (fake) to run so the hybrid fakes exist, then arm
    # vitpose to fail exactly once on its next call.
    est._load_model()
    est._vitpose.fail_next = True

    out = est.estimate_sequence(frames, bboxes)

    assert out["global_orient"].shape == (3, 3)
    assert est.extractor_fallback_count == 1
    assert est.resolved_extractor_device == "cpu"
    assert est._vitpose._fp_device == "cpu"
    assert est._extractor._fp_device == "cpu"
    assert len(est._vitpose.calls) == 2, "extract should run twice: the failure, then the retry"
    assert est._vitpose.pose.current_device == "cpu", "preprocessor model must move back to main"

    # Later chunk: the split is not re-attempted -- fallback count must
    # not grow again, and resolved_extractor_device stays pinned to main.
    est._vitpose.fail_next = False  # would succeed either way; proves no re-arm needed
    est.estimate_sequence(frames, bboxes)

    assert est.extractor_fallback_count == 1
    assert est.resolved_extractor_device == "cpu"


@pytest.mark.unit
@_requires_mps
def test_fallback_retry_failure_propagates(monkeypatch) -> None:
    """A second RuntimeError on the main-device retry must propagate --
    the fallback policy retries exactly once, it doesn't loop."""

    class _AlwaysFailingVitPose(_FakeVitPoseHybrid):
        def extract(self, video_path_or_imgs, bbx_xys, img_ds: float = 0.5):
            raise RuntimeError("simulated permanent extractor failure")

    est = build_fake_hybrid_estimator(
        monkeypatch, device="cpu", extractor_device="mps",
        vitpose_cls=_AlwaysFailingVitPose,
    )
    frames, bboxes = _frames_and_bboxes(3)

    with pytest.raises(RuntimeError, match="simulated permanent extractor failure"):
        est.estimate_sequence(frames, bboxes)

    assert est.extractor_fallback_count == 1, "one fallback attempt should still be recorded"


@pytest.mark.unit
def test_extractor_fallback_count_starts_at_zero(monkeypatch) -> None:
    est = build_fake_hybrid_estimator(monkeypatch, device="cpu", extractor_device="cpu")
    assert est.extractor_fallback_count == 0
    assert est.resolved_extractor_device == "cpu"


# ---------------------------------------------------------------------------
# 5. _redirect_cuda scoping spy: extract phase under extractor device,
#    predict phase under main.
# ---------------------------------------------------------------------------


@pytest.mark.unit
@_requires_mps
def test_redirect_cuda_scoping_extract_under_extractor_predict_under_main(monkeypatch) -> None:
    import src.utils.gvhmr_estimator as gvhmr_mod

    recorded: list[str] = []
    real_redirect_cuda = gvhmr_mod._redirect_cuda

    @contextlib.contextmanager
    def _spy_redirect_cuda(device):
        recorded.append(device)
        with real_redirect_cuda(device):
            yield

    monkeypatch.setattr(gvhmr_mod, "_redirect_cuda", _spy_redirect_cuda)

    est = build_fake_hybrid_estimator(monkeypatch, device="cpu", extractor_device="mps")
    frames, bboxes = _frames_and_bboxes(3)

    est.estimate_sequence(frames, bboxes)

    assert "mps" in recorded, f"extractor-device redirect never entered: {recorded}"
    assert "cpu" in recorded, f"main-device redirect never entered: {recorded}"
    assert recorded.index("mps") < recorded.index("cpu"), (
        f"extract phase must be entered under the extractor device before "
        f"predict's main-device redirect: {recorded}"
    )


@pytest.mark.unit
def test_redirect_cuda_not_entered_for_extract_when_extractor_matches_main(monkeypatch) -> None:
    """The bit-identical shortcut: when extractor_device == main, extract
    calls must not enter _redirect_cuda at all (only predict's own call
    does) -- proving _run_extractor_phase's fast path is really taken."""
    import src.utils.gvhmr_estimator as gvhmr_mod

    recorded: list[str] = []
    real_redirect_cuda = gvhmr_mod._redirect_cuda

    @contextlib.contextmanager
    def _spy_redirect_cuda(device):
        recorded.append(device)
        with real_redirect_cuda(device):
            yield

    monkeypatch.setattr(gvhmr_mod, "_redirect_cuda", _spy_redirect_cuda)

    est = _build_plain_estimator(monkeypatch)  # extractor_device="cpu" == device="cpu"
    frames, bboxes = _frames_and_bboxes(3)

    est.estimate_sequence(frames, bboxes)

    # Exactly one redirect_cuda entry: model.predict's own -- extract
    # phases skipped it entirely.
    assert recorded == ["cpu"], f"expected only predict's redirect_cuda('cpu'), got {recorded}"


# ---------------------------------------------------------------------------
# run_on_track passthrough
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_run_on_track_extractor_device_default_is_cpu() -> None:
    import inspect

    from src.utils.gvhmr_estimator import run_on_track

    sig = inspect.signature(run_on_track)
    assert "extractor_device" in sig.parameters
    assert sig.parameters["extractor_device"].default == "cpu"


@pytest.mark.unit
def test_run_on_track_passes_extractor_device_to_fresh_estimator(monkeypatch, tmp_path) -> None:
    import cv2
    import numpy as np

    from src.utils.gvhmr_estimator import run_on_track

    n = 3
    w, h = 64, 48
    video_path = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (w, h))
    for _ in range(n):
        writer.write(np.zeros((h, w, 3), dtype=np.uint8))
    writer.release()

    track_frames = [(i, (5, 5, 40, 44)) for i in range(n)]
    fake_ckpt = tmp_path / "fake_ckpt.pt"
    fake_ckpt.write_bytes(b"stub")

    seen_extractor_devices: list[str] = []
    orig_init = GVHMREstimator.__init__

    def _spy_init(self, *args, extractor_device="cpu", **kwargs):
        seen_extractor_devices.append(extractor_device)
        orig_init(self, *args, extractor_device=extractor_device, **kwargs)

    monkeypatch.setattr(GVHMREstimator, "__init__", _spy_init)

    def _fake_estimate_sequence(self, frames_bgr, bboxes, fps=30.0, K_per_frame=None,
                                 R_w2c_per_frame=None, legacy_decode=False):
        m = len(frames_bgr)
        return {
            "global_orient": np.zeros((m, 3), dtype=np.float32),
            "body_pose": np.zeros((m, 63), dtype=np.float32),
            "betas": np.zeros((m, 10), dtype=np.float32),
            "transl": np.zeros((m, 3), dtype=np.float32),
            "kp2d": np.zeros((m, 17, 3), dtype=np.float32),
        }

    monkeypatch.setattr(GVHMREstimator, "estimate_sequence", _fake_estimate_sequence)

    run_on_track(
        track_frames=track_frames,
        video_path=video_path,
        checkpoint=fake_ckpt,
        device="cpu",
        batch_size=16,
        max_sequence_length=120,
        estimator=None,  # forces run_on_track to construct a fresh estimator
        extractor_device="mps",
    )

    assert seen_extractor_devices == ["mps"]


@pytest.mark.unit
def test_run_on_track_ignores_extractor_device_when_estimator_supplied(monkeypatch, tmp_path) -> None:
    """extractor_device is a constructor-only knob -- when the caller
    supplies a pre-built estimator, run_on_track must not try to mutate
    it based on its own extractor_device argument."""
    import cv2
    import numpy as np

    from src.utils.gvhmr_estimator import run_on_track

    n = 2
    w, h = 64, 48
    video_path = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (w, h))
    for _ in range(n):
        writer.write(np.zeros((h, w, 3), dtype=np.uint8))
    writer.release()

    track_frames = [(i, (5, 5, 40, 44)) for i in range(n)]
    fake_ckpt = tmp_path / "fake_ckpt.pt"
    fake_ckpt.write_bytes(b"stub")

    est = GVHMREstimator(checkpoint=str(fake_ckpt), device="cpu", extractor_device="cpu")

    def _fake_estimate_sequence(self, frames_bgr, bboxes, fps=30.0, K_per_frame=None,
                                 R_w2c_per_frame=None, legacy_decode=False):
        m = len(frames_bgr)
        return {
            "global_orient": np.zeros((m, 3), dtype=np.float32),
            "body_pose": np.zeros((m, 63), dtype=np.float32),
            "betas": np.zeros((m, 10), dtype=np.float32),
            "transl": np.zeros((m, 3), dtype=np.float32),
            "kp2d": np.zeros((m, 17, 3), dtype=np.float32),
        }

    monkeypatch.setattr(GVHMREstimator, "estimate_sequence", _fake_estimate_sequence)

    run_on_track(
        track_frames=track_frames,
        video_path=video_path,
        checkpoint=fake_ckpt,
        device="cpu",
        batch_size=16,
        max_sequence_length=120,
        estimator=est,
        extractor_device="mps",  # must be ignored -- est already resolved to "cpu"
    )

    assert est._extractor_device_requested == "cpu"
    assert est.resolved_extractor_device == "cpu"
