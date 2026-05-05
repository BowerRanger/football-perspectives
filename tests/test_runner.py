"""Tests for src.pipeline.runner.resolve_stages — single-mode stage resolution."""

import pytest

from src.pipeline.runner import resolve_stages


@pytest.mark.unit
def test_resolve_all():
    assert resolve_stages("all", None) == [
        "prepare_shots",
        "tracking",
        "camera",
        "pose_2d",
        "hmr_world",
        "ball",
        "export",
    ]


@pytest.mark.unit
def test_resolve_subset():
    assert resolve_stages("camera,pose_2d", None) == ["camera", "pose_2d"]


@pytest.mark.unit
def test_resolve_unknown_raises():
    with pytest.raises(ValueError):
        resolve_stages("calibration", None)


@pytest.mark.unit
def test_resolve_with_from_stage_skips_earlier():
    result = resolve_stages("all", "hmr_world")
    assert result == ["hmr_world", "ball", "export"]
