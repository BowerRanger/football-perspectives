"""config/default.yaml structure guard: a block spliced into the wrong place
still parses (YAML is indentation-scoped), it just moves keys under another
block. 7fb8499 fixed `appearance:` landing inside export.virtual_cameras and
swallowing 14 camera keys — this pins the shape."""
from pathlib import Path

import pytest
import yaml

CFG = yaml.safe_load((Path(__file__).resolve().parents[1] / "config" / "default.yaml").read_text())


@pytest.mark.unit
def test_virtual_camera_keys_live_under_export():
    vc = CFG["export"]["virtual_cameras"]
    for key in ("chase_back_m", "chase_fov_deg", "dolly_fov_deg", "orbit_sweep_deg",
                "tactical_fov_deg", "corner_fov_deg", "sideline_fov_deg", "drone_fov_deg"):
        assert key in vc, key


@pytest.mark.unit
def test_stage_blocks_hold_only_their_own_keys():
    assert set(CFG["appearance"]) <= {"enabled", "sample_frames_per_shot", "min_keypoint_conf",
                                      "white_balance", "clustering", "library", "match", "kits"}
    assert set(CFG["appearance"]["library"]) <= {"snap_de", "lightness_weight"}


@pytest.mark.unit
def test_pipeline_stage_blocks_present_at_top_level():
    for block in ("pipeline", "tracking", "camera", "hmr_world", "refined_poses", "ball",
                  "appearance", "export", "render", "shorts"):
        assert isinstance(CFG.get(block), dict), block
