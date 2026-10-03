"""KitSpec normalisation, sleeve/collar zones, kit pattern maths and the
pure helpers behind --vertical-only / eyes:<PID> / SMPL preflight."""
import numpy as np
import pytest

from src.utils import render_look as rl
from src.utils.smpl_skeleton import SMPL_REST_JOINTS_YUP as J

_ROLE = {"P001": ("unknown", "player")}


def _look(kit, role="home"):
    teams = {"defaults": {role: kit}, "by_player": {}}
    return rl.resolve_player_looks(teams, _ROLE, role_overrides={"P001": role})["P001"]


@pytest.mark.unit
def test_normalize_kit_legacy_shape_is_solid_and_unchanged():
    kit = rl.normalize_kit({"shirt": "#c8102e", "shorts": "#fff", "socks": "#c8102e"})
    assert kit["pattern"] is None and kit["sleeves"] == "short"
    assert kit["sleeve_color"] is None and kit["collar"] is None
    assert kit["shorts"] == "#ffffff"          # 3-digit hex expanded


@pytest.mark.unit
def test_normalize_kit_long_sleeves_keeps_length_meaning():
    kit = rl.normalize_kit({"shirt": "#111111", "sleeves": "long"})
    assert kit["sleeves"] == "long"
    assert _look({"shirt": "#111111", "sleeves": "long"})["sleeves"] == "long"
    assert rl.normalize_kit({"shirt": "#111111", "sleeves": "bogus"})["sleeves"] == "short"


@pytest.mark.unit
def test_normalize_kit_pattern_defaults_and_validation():
    kit = rl.normalize_kit({"shirt": "#d71920", "shorts": "#000000", "socks": "#000000",
                            "pattern": {"type": "vertical_stripes",
                                        "colors": ["#d71920", "#000000"]}})
    assert kit["pattern"] == {"type": "vertical_stripes",
                              "colors": ["#d71920", "#000000"], "width_m": 0.07}
    solid = rl.normalize_kit({"shirt": "#d71920", "pattern": {"type": "solid"}})
    assert solid["pattern"] is None
    with pytest.raises(ValueError):
        rl.normalize_kit({"shirt": "#d71920", "pattern": {"type": "paisley",
                                                          "colors": ["#000000", "#ffffff"]}})
    with pytest.raises(ValueError):
        rl.normalize_kit({"shirt": "#d71920", "pattern": {"type": "hoops",
                                                          "colors": ["#000000"]}})
    with pytest.raises(ValueError):
        rl.normalize_kit({"shirt": "#d71920", "pattern": {"type": "hoops",
                                                          "colors": ["#000000", "#fff"],
                                                          "width_m": 0}})


@pytest.mark.unit
def test_looks_carry_sleeve_collar_and_pattern():
    look = _look({"shirt": "#d71920", "sleeve_color": "#ffffff", "collar": "#222222",
                  "shorts": "#ffffff", "socks": "#d71920"})
    c = look["colors"]
    assert c["sleeve"] == rl.hex_to_linear_rgba("#ffffff")
    assert c["collar"] == rl.hex_to_linear_rgba("#222222")
    assert look["pattern"] is None


@pytest.mark.unit
def test_sleeve_and_collar_default_to_shirt():
    look = _look({"shirt": "#d71920", "shorts": "#000000", "socks": "#000000"})
    assert look["colors"]["sleeve"] == look["colors"]["shirt"]
    assert look["colors"]["collar"] == look["colors"]["shirt"]


@pytest.mark.unit
def test_pattern_zones_sleeves_inherit_unless_sleeve_color():
    pat = {"type": "vertical_stripes", "colors": ["#d71920", "#000000"]}
    inherit = _look({"shirt": "#d71920", "pattern": pat})
    assert inherit["pattern"]["type"] == "vertical_stripes"
    assert inherit["pattern"]["colors"][1] == rl.hex_to_linear_rgba("#000000")
    assert set(inherit["pattern_zones"]) == {"shirt", "sleeve"}
    own = _look({"shirt": "#d71920", "sleeve_color": "#ffffff", "pattern": pat})
    assert set(own["pattern_zones"]) == {"shirt"}
    plain = _look({"shirt": "#d71920"})
    assert plain["pattern"] is None and plain["pattern_zones"] == ()


@pytest.mark.unit
def test_pattern_mask_vertical_stripes_follow_lateral_x_and_centre():
    x = np.array([0.0, 0.03, 0.05, 0.10, 0.17, -0.05, -0.10])
    co = np.stack([x, np.zeros_like(x), np.zeros_like(x)], axis=1)
    v = rl.pattern_mask("vertical_stripes", co, 0.07)
    # stripe 0 is centred on x=0 (|x|<0.035); alternates either side
    assert v.tolist() == [0, 0, 1, 1, 0, 1, 1]


@pytest.mark.unit
def test_pattern_mask_hoops_follow_height():
    y = np.array([0.0, 0.05, 0.10, 0.15, 0.22])
    co = np.stack([np.zeros_like(y), y, np.zeros_like(y)], axis=1)
    h = rl.pattern_mask("hoops", co, 0.07)
    assert h.tolist() == [0, 1, 1, 0, 1]
    assert rl.pattern_axis("vertical_stripes") == 0 and rl.pattern_axis("hoops") == 1


@pytest.mark.unit
def test_zones_sleeve_and_collar():
    cases = [
        ("upper_arm_in", J[16] + [0.03, 0, 0], 16, "sleeve"),
        ("upper_arm_out", J[16] + (J[18] - J[16]) * 0.85, 16, "skin"),
        ("chest", J[9] + [0, 0, 0.08], 9, "shirt"),
        ("collar_front", J[12] + [0, 0.0, 0.05], 9, "collar"),
        ("collar_neck_base", J[12] + [0, 0.01, 0.03], 12, "collar"),
        ("neck_high", J[12] + [0, 0.05, 0.0], 12, "skin"),
        ("shoulder_top_wide", J[12] + [0.15, 0.0, 0.0], 13, "shirt"),
    ]
    verts = np.array([c[1] for c in cases])
    w = np.zeros((len(cases), 24))
    w[np.arange(len(cases)), [c[2] for c in cases]] = 1.0
    zones = rl.anatomical_kit_zones(verts, w, J, sleeves="short")
    assert dict(zip([c[0] for c in cases], zones)) == {c[0]: c[3] for c in cases}


@pytest.mark.unit
def test_zones_long_sleeves_are_sleeve_zone():
    verts = np.array([J[16] + (J[18] - J[16]) * 0.85, (J[18] + J[20]) / 2])
    w = np.zeros((2, 24))
    w[0, 16] = 1
    w[1, 18] = 1
    assert rl.anatomical_kit_zones(verts, w, J, sleeves="long") == ["sleeve", "sleeve"]


@pytest.mark.unit
def test_rest_coords_preserve_vertex_positions():
    v = np.array([[0.1, 1.2, -0.3], [0.0, 0.5, 0.1]])
    co = rl.rest_coords(v)
    assert co.dtype == np.float32 and co.shape == (2, 3)
    np.testing.assert_allclose(co, v, atol=1e-6)


@pytest.mark.unit
def test_eyes_hidden_pid():
    assert rl.eyes_hidden_pid("eyes:P006") == "P006"
    assert rl.eyes_hidden_pid("pov:P006") is None
    assert rl.eyes_hidden_pid("orbit") is None


@pytest.mark.unit
def test_plan_passes():
    assert rl.plan_passes("drone", vertical=False, vertical_only=False) == [("", False)]
    assert rl.plan_passes("drone", vertical=True, vertical_only=False) == [
        ("", False), ("_9x16", True)]
    assert rl.plan_passes("broadcast", vertical=True, vertical_only=False) == [("", False)]
    assert rl.plan_passes("drone", vertical=False, vertical_only=True) == [("_9x16", True)]
    assert rl.plan_passes("broadcast", vertical=False, vertical_only=True) == [("_9x16", True)]


@pytest.mark.unit
def test_smpl_asset_problem():
    ok = {"v_template": 1, "faces": 1, "weights": 1, "joint_positions": 1}
    assert rl.smpl_asset_problem(ok, allow_capsule_fallback=False) is None
    msg = rl.smpl_asset_problem(None, allow_capsule_fallback=False)
    assert msg and "smpl_neutral" in msg and "--allow-capsule-fallback" in msg
    assert rl.smpl_asset_problem(None, allow_capsule_fallback=True) is None
    partial = {"v_template": 1, "weights": 1, "joint_positions": 1}
    assert "faces" in rl.smpl_asset_problem(partial, False)
