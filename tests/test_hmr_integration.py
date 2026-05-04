"""Integration tests for GVHMR that require the downloaded checkpoints.

These tests are gated on checkpoint presence — they skip when the 5 files
in ``third_party/gvhmr/inputs/checkpoints/`` are missing, so CI without
model weights still passes.

Run ``bash scripts/setup_gvhmr.sh`` to verify the expected layout.
"""

from __future__ import annotations

from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[1]
_CKPT_DIR = _REPO_ROOT / "third_party" / "gvhmr" / "inputs" / "checkpoints"

_REQUIRED_CHECKPOINTS = {
    "gvhmr": _CKPT_DIR / "gvhmr" / "gvhmr_siga24_release.ckpt",
    "hmr2": _CKPT_DIR / "hmr2" / "epoch=10-step=25000.ckpt",
    "vitpose": _CKPT_DIR / "vitpose" / "vitpose-h-multi-coco.pth",
    "yolo": _CKPT_DIR / "yolo" / "yolov8x.pt",
    "smpl": _CKPT_DIR / "body_models" / "smpl" / "SMPL_NEUTRAL.pkl",
    "smplx": _CKPT_DIR / "body_models" / "smplx" / "SMPLX_NEUTRAL.npz",
}


def _missing_checkpoints() -> list[str]:
    return [name for name, path in _REQUIRED_CHECKPOINTS.items() if not path.exists()]


def _have_all_checkpoints() -> bool:
    return not _missing_checkpoints()


def _have_core_checkpoints() -> bool:
    """Core checkpoints needed for config resolution (not full inference)."""
    return _REQUIRED_CHECKPOINTS["smplx"].exists()


@pytest.mark.skipif(
    not _have_core_checkpoints(),
    reason=f"GVHMR SMPLX missing: {_missing_checkpoints()}",
)
def test_gvhmr_config_resolves_and_model_instantiates() -> None:
    """Verify the hydra compose path loads the model architecture end-to-end.

    Does not load the pretrained weights (that's the next test).  This ensures
    the minimal config registration is sufficient and the dead yaml isn't on
    the critical path.
    """
    import sys

    gvhmr_root = _REPO_ROOT / "third_party" / "gvhmr"
    shims_root = _REPO_ROOT / "third_party" / "gvhmr_shims"
    sys.path.insert(0, str(shims_root))
    sys.path.insert(1, str(gvhmr_root))

    import hydra
    from hydra import compose, initialize_config_module

    from src.utils.gvhmr_register import register_minimal_gvhmr

    with initialize_config_module(version_base="1.3", config_module="hmr4d.configs"):
        register_minimal_gvhmr()
        cfg = compose(
            config_name="demo",
            overrides=["video_name=_test_", "static_cam=True"],
        )

    model = hydra.utils.instantiate(cfg.model, _recursive_=False)
    assert model is not None
    assert hasattr(model, "pipeline")
    assert hasattr(model.pipeline, "denoiser3d")
    # Verify the network has expected hyperparameters
    assert model.pipeline.denoiser3d.__class__.__name__ == "NetworkEncoderRoPE"


@pytest.mark.skipif(
    not _have_all_checkpoints(),
    reason=f"GVHMR checkpoints missing: {_missing_checkpoints()}",
)
def test_gvhmr_estimator_loads_with_weights() -> None:
    """Load the full GVHMR model including pretrained weights.

    Skipped until all 5 checkpoints are downloaded (GVHMR, HMR2, ViTPose,
    SMPL, SMPLX).  YOLO is expected to be symlinked by the setup script.
    """
    from src.utils.gvhmr_estimator import GVHMREstimator

    estimator = GVHMREstimator(device="cpu")
    estimator._load_model()

    assert estimator._model is not None
    assert estimator._body_model is not None
    assert estimator._vitpose is not None
    assert estimator._extractor is not None
