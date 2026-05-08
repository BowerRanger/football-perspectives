"""Tests for src.pipeline.runner.resolve_stages — single-mode stage resolution."""

import pytest

from src.pipeline.runner import resolve_stages


@pytest.mark.unit
def test_resolve_all():
    assert resolve_stages("all", None) == [
        "prepare_shots",
        "tracking",
        "camera",
        "hmr_world",
        "ball",
        "export",
    ]


@pytest.mark.unit
def test_resolve_subset():
    assert resolve_stages("camera,hmr_world", None) == ["camera", "hmr_world"]


@pytest.mark.unit
def test_resolve_unknown_raises():
    with pytest.raises(ValueError):
        resolve_stages("calibration", None)


@pytest.mark.unit
def test_resolve_with_from_stage_skips_earlier():
    result = resolve_stages("all", "hmr_world")
    assert result == ["hmr_world", "ball", "export"]


@pytest.mark.unit
def test_run_pipeline_shot_filter_propagates_to_stage(
    tmp_path: Path, monkeypatch,
) -> None:
    """run_pipeline(stages='camera', shot_filter='alpha') sets
    stage.shot_filter='alpha' on the constructed CameraStage."""
    from src.pipeline.runner import run_pipeline
    from src.pipeline.base import BaseStage

    captured: dict = {}

    class FakeCameraStage(BaseStage):
        name = "camera"

        def is_complete(self) -> bool:
            return False

        def run(self) -> None:
            captured["shot_filter"] = self.shot_filter

    def fake_stage_class(name: str):
        if name == "camera":
            return FakeCameraStage
        return None

    monkeypatch.setattr("src.pipeline.runner._stage_class", fake_stage_class)
    run_pipeline(
        output_dir=tmp_path,
        stages="camera",
        from_stage=None,
        config={},
        shot_filter="alpha",
    )
    assert captured["shot_filter"] == "alpha"
