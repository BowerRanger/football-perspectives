import yaml
from pathlib import Path


def test_default_config_has_auto_anchors_block():
    cfg = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "config" / "default.yaml").read_text()
    )
    aa = cfg["camera"]["auto_anchors"]
    # On by default; with replace_when_empty, manual anchors still take
    # precedence so this only adds a cold-start for un-anchored shots.
    assert aa["enabled"] is True
    assert aa["mode"] == "replace_when_empty"
    assert aa["keyframe_interval"] == 18
    assert "min_points_per_anchor" in aa
    assert aa["model"]["device"] == "auto"
