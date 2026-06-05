import yaml
from pathlib import Path


def test_default_config_has_auto_anchors_block():
    cfg = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "config" / "default.yaml").read_text()
    )
    aa = cfg["camera"]["auto_anchors"]
    assert aa["enabled"] is False
    assert aa["mode"] == "replace_when_empty"
    assert aa["keyframe_interval"] == 30
    assert "min_points_per_anchor" in aa
    assert aa["model"]["device"] == "auto"
