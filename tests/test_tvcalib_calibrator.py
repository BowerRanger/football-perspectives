"""Tests for the TVCalib calibration backend wrapper.

The heavy-weight tests (segmenter inference + AdamW solve) are skipped
unless ``kornia`` and ``SoccerNet`` are importable — they only run in
the ``[tvcalib]`` extras environment.  The coordinate-frame and config
dispatch tests run on the base environment using mocks.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.utils.neural_calibrator import (
    NeuralCalibration,
    convert_pnlcalib_to_ours,
)


def _tvcalib_deps_available() -> bool:
    """Return True when kornia + SoccerNet are importable."""
    for name in ("kornia", "SoccerNet"):
        if importlib.util.find_spec(name) is None:
            return False
    return True


requires_tvcalib = pytest.mark.skipif(
    not _tvcalib_deps_available(),
    reason="kornia/SoccerNet extras not installed (see pyproject.toml [tvcalib])",
)


class TestCoordinateFrameConversion:
    """The TVCalib wrapper reuses ``convert_pnlcalib_to_ours`` because
    PnLCalib and TVCalib share the SoccerNet pitch convention.  These
    tests lock in the invariants the wrapper relies on: camera position
    transforms to the corner-origin z-up frame, and the rotation
    post-multiplies by ``diag(1, -1, -1)``.
    """

    def test_pitch_centre_maps_to_corner_origin_centre(self) -> None:
        R_sn = np.eye(3, dtype=np.float64)
        C_sn = np.array([0.0, 55.0, -12.0])  # behind-tribune cam
        _, _, world_position = convert_pnlcalib_to_ours(R_sn, C_sn)
        assert world_position == pytest.approx([52.5, 34.0 - 55.0, 12.0])

    def test_positive_z_above_pitch(self) -> None:
        """SN c_z<0 (above pitch) → ours c_z>0 (above pitch)."""
        R_sn = np.eye(3, dtype=np.float64)
        for c_z_sn in (-5.0, -12.0, -40.0):
            _, _, pos = convert_pnlcalib_to_ours(
                R_sn, np.array([0.0, 50.0, c_z_sn]),
            )
            assert pos[2] > 0, f"c_z_sn={c_z_sn} did not map above pitch"

    def test_camera_position_round_trip_through_tvec(self) -> None:
        """``-R_ours @ C_ours`` must equal ``t_ours`` from the conversion."""
        R_sn = np.array(
            [
                [0.8660254, 0.0, 0.5],
                [0.0, 1.0, 0.0],
                [-0.5, 0.0, 0.8660254],
            ],
            dtype=np.float64,
        )
        C_sn = np.array([2.0, 60.0, -15.0])
        rvec, tvec, world_position = convert_pnlcalib_to_ours(R_sn, C_sn)
        import cv2

        R_ours, _ = cv2.Rodrigues(rvec)
        expected_tvec = -R_ours @ world_position
        assert tvec == pytest.approx(expected_tvec, abs=1e-6)


class TestBackendDispatch:
    """The calibration stage must route to the right backend by config."""

    def test_unknown_backend_raises(self, tmp_path: Path) -> None:
        from src.stages.calibration import CameraCalibrationStage

        stage = CameraCalibrationStage(
            config={"calibration": {"backend": "bogus"}},
            output_dir=tmp_path,
        )
        with pytest.raises(ValueError, match="Unknown calibration backend"):
            stage._calibrator()

    def test_pnlcalib_backend_selected_by_default(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from src.stages import calibration as cal_mod

        instantiated: list[str] = []

        def fake_pnl(**kwargs):
            instantiated.append("pnl")
            return MagicMock(name="PnLCalibrator")

        monkeypatch.setattr(cal_mod, "PnLCalibrator", fake_pnl)
        stage = cal_mod.CameraCalibrationStage(
            config={"calibration": {}}, output_dir=tmp_path,
        )
        stage._calibrator()
        assert instantiated == ["pnl"]

    def test_tvcalib_backend_instantiates_wrapper(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Inject a fake src.utils.tvcalib_calibrator so the import in
        # _calibrator() resolves to our spy instead of loading the real
        # (heavy) module.
        fake_module = types.ModuleType("src.utils.tvcalib_calibrator")
        calls: list[dict] = []

        class FakeTVCalibrator:
            def __init__(self, **kwargs):
                calls.append(kwargs)

            def calibrate(self, frame_bgr):  # noqa: ARG002
                return None

        fake_module.TVCalibCalibrator = FakeTVCalibrator  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "src.utils.tvcalib_calibrator", fake_module)

        from src.stages.calibration import CameraCalibrationStage

        stage = CameraCalibrationStage(
            config={
                "calibration": {
                    "backend": "tvcalib",
                    "device": "cpu",
                    "tvcalib": {
                        "prior": "left",
                        "optim_steps": 321,
                        "ndc_loss_threshold": 0.02,
                    },
                },
            },
            output_dir=tmp_path,
        )
        stage._calibrator()
        assert len(calls) == 1
        kwargs = calls[0]
        assert kwargs["prior"] == "left"
        assert kwargs["optim_steps"] == 321
        assert kwargs["ndc_loss_threshold"] == pytest.approx(0.02)
        assert kwargs["device"] == "cpu"


class TestTVCalibCalibratorContract:
    """Construction-time behaviour that doesn't need the heavy deps."""

    def test_unknown_prior_rejected(self) -> None:
        from src.utils.tvcalib_calibrator import TVCalibCalibrator

        with pytest.raises(ValueError, match="Unknown TVCalib prior"):
            TVCalibCalibrator(prior="bogus")

    def test_construction_does_not_load_model(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Lazy loading: ``__init__`` must not touch the segmenter."""
        from src.utils import tvcalib_calibrator as mod

        def fail(*_args, **_kwargs):
            raise AssertionError("_import_tvcalib_modules called too eagerly")

        monkeypatch.setattr(mod, "_import_tvcalib_modules", fail)
        calibrator = mod.TVCalibCalibrator(device="cpu")
        assert calibrator._loaded is False

    def test_calibrate_returns_none_when_segmenter_empty(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """If the segmenter produces no buckets, return None (don't solve)."""
        import torch

        from src.utils import tvcalib_calibrator as mod

        calibrator = mod.TVCalibCalibrator(device="cpu")

        # Pretend the TVCalib modules are loaded with minimal stubs.
        segmenter = MagicMock()
        segmenter.inference.return_value = torch.zeros((1, 256, 455), dtype=torch.long)
        calibrator._segmenter = segmenter
        calibrator._solver = MagicMock()
        calibrator._solver.self_optim_batch.side_effect = AssertionError(
            "solver must not run when segmenter output is empty",
        )

        class _Object3D:
            segment_names = set()

        calibrator._object3d = _Object3D()
        calibrator._modules = {
            "generate_class_synthesis": lambda _mask, radius: {},
            "get_line_extremities": lambda *a, **k: {},
            "InferenceDatasetCalibration": MagicMock(),
        }
        calibrator._loaded = True

        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        assert calibrator.calibrate(frame) is None


@requires_tvcalib
class TestTVCalibEndToEnd:
    """Heavy integration tests run only with the ``[tvcalib]`` extras."""

    def test_modules_import_without_error(self) -> None:
        from src.utils.tvcalib_calibrator import _import_tvcalib_modules

        mods = _import_tvcalib_modules()
        assert "TVCalibModule" in mods
        assert "generate_class_synthesis" in mods

    def test_calibrator_builds_on_cpu(self) -> None:
        """A constructed calibrator reaches the segmenter-load step.

        Does not download weights — only asserts the import and the
        construction path before ``_ensure_loaded`` attempts the
        download.  The real inference path is covered by the
        integration run described in the plan file.
        """
        from src.utils.tvcalib_calibrator import TVCalibCalibrator

        cal = TVCalibCalibrator(device="cpu", optim_steps=10)
        assert cal._prior == "center"
        assert cal._optim_steps == 10
