from pathlib import Path
import pytest
from src.pipeline.config import load_config
from src.pipeline.runner import resolve_stages, STAGE_ORDER, run_pipeline
from src.pipeline.runner import _create_pitch_detector
from src.utils.pitch_detector import HeuristicPitchDetector, HybridPitchDetector, ManualJsonPitchDetector


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
    assert names == ["calibration", "sync", "pose", "triangulation", "smpl_fitting", "export"]


def test_resolve_stages_explicit():
    names = resolve_stages("1,2", from_stage=None)
    assert names == ["segmentation", "tracking"]


def test_resolve_stages_from_numeric_alias():
    """--from-stage 3 should be equivalent to --from-stage calibration."""
    names_numeric = resolve_stages("all", from_stage="3")
    names_canonical = resolve_stages("all", from_stage="calibration")
    assert names_numeric == names_canonical


def test_stage_order_includes_tracking_and_pose():
    names = [name for name, _ in STAGE_ORDER]
    assert "tracking" in names
    assert "pose" in names
    assert "matching" not in names


def test_aliases_include_stages_3_to_6():
    from src.pipeline.runner import _ALIASES
    assert _ALIASES["2"] == "tracking"
    assert _ALIASES["3"] == "calibration"
    assert _ALIASES["4"] == "sync"
    assert _ALIASES["5"] == "pose"


def test_resolve_stages_from_tracking():
    names = resolve_stages("all", from_stage="tracking")
    assert names == ["tracking", "calibration", "sync", "pose", "triangulation", "smpl_fitting", "export"]


def test_resolve_stages_explicit_3_4():
    names = resolve_stages("3,4", from_stage=None)
    assert names == ["calibration", "sync"]


def test_aliases_remapped_to_new_order():
    from src.pipeline.runner import _ALIASES

    assert _ALIASES["2"] == "tracking"
    assert _ALIASES["3"] == "calibration"
    assert _ALIASES["4"] == "sync"
    assert _ALIASES["5"] == "pose"


def test_create_pitch_detector_hybrid_by_default(tmp_path):
    detector = _create_pitch_detector(config={}, output_dir=tmp_path)
    assert isinstance(detector, HybridPitchDetector)


def test_create_pitch_detector_none_when_configured(tmp_path):
    cfg = {"calibration": {"detector_type": "none"}}
    detector = _create_pitch_detector(config=cfg, output_dir=tmp_path)
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


def test_create_pitch_detector_heuristic(tmp_path):
    cfg = {"calibration": {"detector_type": "heuristic"}}
    detector = _create_pitch_detector(config=cfg, output_dir=tmp_path)
    assert isinstance(detector, HeuristicPitchDetector)


def test_create_pitch_detector_unknown_raises(tmp_path):
    cfg = {"calibration": {"detector_type": "mystery"}}
    with pytest.raises(ValueError):
        _create_pitch_detector(config=cfg, output_dir=tmp_path)


def test_run_pipeline_passes_device_to_pose_stage(tmp_path, monkeypatch):
    cfg = load_config()
    captured = {}

    class FakePoseStage:
        def __init__(self, config, output_dir, **kwargs):
            captured.update(kwargs)

        def is_complete(self):
            return False

        def run(self):
            return None

    monkeypatch.setattr("src.pipeline.runner.STAGE_ORDER", [("pose", FakePoseStage)])

    run_pipeline(
        output_dir=tmp_path,
        stages="pose",
        from_stage=None,
        config=cfg,
        device="mps",
    )

    assert captured["device"] == "mps"
