from src.stages.ball import _kinematic_touch_cfg
from src.utils.ball_kinematic_touch import KinematicTouchCfg


def test_empty_dict_gives_defaults():
    assert _kinematic_touch_cfg({}) == KinematicTouchCfg()


def test_overrides_are_applied():
    cfg = _kinematic_touch_cfg({"enabled": False, "contact_gap_m": 0.5,
                                "kin_min_foot_speed": 10.0})
    assert cfg.enabled is False
    assert cfg.contact_gap_m == 0.5
    assert cfg.kin_min_foot_speed == 10.0
    # untouched keys keep defaults
    assert cfg.nms_window == KinematicTouchCfg().nms_window
