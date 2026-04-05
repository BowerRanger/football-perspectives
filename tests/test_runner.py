from pathlib import Path
import pytest
from src.pipeline.config import load_config
from src.pipeline.runner import resolve_stages, STAGE_ORDER
from src.pipeline.runner import _create_pitch_detector
from src.utils.pitch_detector import ManualJsonPitchDetector


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
    assert names == ["calibration", "sync", "tracking", "pose", "matching"]


def test_resolve_stages_explicit():
    names = resolve_stages("1,3", from_stage=None)
    assert names == ["segmentation", "sync"]


def test_resolve_stages_from_numeric_alias():
    """--from-stage 2 should be equivalent to --from-stage calibration."""
    names_numeric = resolve_stages("all", from_stage="2")
    names_canonical = resolve_stages("all", from_stage="calibration")
    assert names_numeric == names_canonical


def test_stage_order_includes_stages_4_to_6():
    names = [name for name, _ in STAGE_ORDER]
    assert "tracking" in names
    assert "pose" in names
    assert "matching" in names


def test_aliases_include_stages_4_to_6():
    from src.pipeline.runner import _ALIASES
    assert _ALIASES["4"] == "tracking"
    assert _ALIASES["5"] == "pose"
    assert _ALIASES["6"] == "matching"


def test_resolve_stages_from_tracking():
    names = resolve_stages("all", from_stage="tracking")
    assert names == ["tracking", "pose", "matching"]


def test_resolve_stages_explicit_4_5():
    names = resolve_stages("4,5", from_stage=None)
    assert names == ["tracking", "pose"]


def test_create_pitch_detector_none_by_default(tmp_path):
    detector = _create_pitch_detector(config={}, output_dir=tmp_path)
    assert detector is None


def test_create_pitch_detector_manual_json(tmp_path):
    annotations_dir = tmp_path / "landmarks"
    annotations_dir.mkdir(parents=True)

    cfg = {
        "calibration": {
            "detector_type": "manual_json",
            "manual_landmarks_dir": "landmarks",
        }
    }
    detector = _create_pitch_detector(config=cfg, output_dir=tmp_path)
    assert isinstance(detector, ManualJsonPitchDetector)


def test_create_pitch_detector_unknown_raises(tmp_path):
    cfg = {"calibration": {"detector_type": "mystery"}}
    with pytest.raises(ValueError):
        _create_pitch_detector(config=cfg, output_dir=tmp_path)
