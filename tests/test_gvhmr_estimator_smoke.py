"""Smoke test for the per-track GVHMR runner adapter.

Verifies the public ``run_on_track`` signature is callable. Full integration
(including a fake runner that bypasses the real GVHMR weights) is exercised
in ``tests/test_hmr_world_stage.py``.
"""

from __future__ import annotations

import inspect

import pytest


@pytest.mark.unit
def test_run_on_track_signature() -> None:
    from src.utils.gvhmr_estimator import run_on_track

    sig = inspect.signature(run_on_track)
    assert "track_frames" in sig.parameters
    assert "checkpoint" in sig.parameters
    assert "video_path" in sig.parameters
    assert "device" in sig.parameters
    assert "batch_size" in sig.parameters
    assert "max_sequence_length" in sig.parameters
    # estimator is optional: passing a pre-constructed GVHMREstimator
    # avoids reloading the model per player (Slice 1 speed-up).
    assert "estimator" in sig.parameters
    assert sig.parameters["estimator"].default is None
    # per_frame_K is optional: the calibrated per-frame K from
    # camera_track replaces GVHMR's default estimate_K() so the body
    # doesn't lean away from camera under telephoto focal lengths.
    assert "per_frame_K" in sig.parameters
    assert sig.parameters["per_frame_K"].default is None
