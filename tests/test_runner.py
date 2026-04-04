from pathlib import Path
from src.pipeline.config import load_config
from src.pipeline.runner import resolve_stages, STAGE_ORDER


def test_load_config_returns_dict(tmp_path):
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text("shot_segmentation:\n  threshold: 42.0\n")
    cfg = load_config(cfg_path)
    assert cfg["shot_segmentation"]["threshold"] == 42.0


def test_resolve_stages_all():
    names = resolve_stages("all", from_stage=None)
    assert names == [name for name, _ in STAGE_ORDER]


def test_resolve_stages_from():
    names = resolve_stages("all", from_stage="calibration")
    assert names == ["calibration", "sync"]


def test_resolve_stages_explicit():
    names = resolve_stages("1,3", from_stage=None)
    assert names == ["segmentation", "sync"]


def test_resolve_stages_from_numeric_alias():
    """--from-stage 2 should be equivalent to --from-stage calibration."""
    names_numeric = resolve_stages("all", from_stage="2")
    names_canonical = resolve_stages("all", from_stage="calibration")
    assert names_numeric == names_canonical
