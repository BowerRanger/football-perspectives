"""Kit-safety lint: post-stack simulation on team kits (pure, no bpy)."""
import numpy as np

from src.utils.kit_safety import _lab_after, apply_post, delta_e, lint_kit_safety

LIVERPOOL = {"shirt": "#c8102e", "shorts": "#c8102e", "socks": "#c8102e"}
CHELSEA = {"shirt": "#034694", "shorts": "#034694", "socks": "#f2f2f2"}
KEEPER = {"shirt": "#33373b", "shorts": "#2a2d30", "socks": "#2a2d30"}
REF = {"shirt": "#e8df3a", "shorts": "#151515", "socks": "#151515"}
GBERCH = {"home": LIVERPOOL, "away": CHELSEA, "away_gk": KEEPER, "referee": REF}
COMIC = {"saturation": 1.3}  # shipped comic preset: no posterize (gberch_experiments.yaml)


def codes(findings):
    return {f["code"] for f in findings}


def test_no_active_post_is_clean():
    assert lint_kit_safety(GBERCH, {}, 12.0) == []
    assert lint_kit_safety(GBERCH, None, 12.0) == []


def test_shipped_comic_preset_passes():
    assert lint_kit_safety(GBERCH, COMIC, 12.0) == []


def test_posterize_5_warns_on_chelsea_blue():
    findings = lint_kit_safety(GBERCH, {"posterize": 5}, 12.0)
    away = [f for f in findings if "away" in f["roles"]]
    assert away, findings
    assert all(f["severity"] == "warn" for f in findings)


def test_saturation_zero_merges_equal_luma_teams():
    kits = {"home": {"shirt": "#c0392b"}, "away": {"shirt": "#2980b9"}}
    assert delta_e(_lab_after("#c0392b", None), _lab_after("#2980b9", None)) > 12
    findings = lint_kit_safety(kits, {"saturation": 0.0}, 12.0)
    assert "team_merge" in codes(findings), findings


def test_duotone_merges_teams_of_similar_luma():
    kits = {"home": {"shirt": "#c0392b"}, "away": {"shirt": "#2980b9"}}
    post = {"duotone": {"shadow": "#0b1d3a", "highlight": "#f4c95d"}}
    assert "team_merge" in codes(lint_kit_safety(kits, post, 12.0))


def test_striped_kit_collapses_under_saturation_zero():
    kits = {"home": {"shirt": "#c0392b", "pattern": {
        "type": "vertical_stripes", "colors": ["#c0392b", "#2980b9"]}}}
    assert "intra_kit_collapse" in codes(lint_kit_safety(kits, {"saturation": 0.0}, 12.0))


def test_white_sleeves_on_red_survive_comic():
    arsenal = {"home": {"shirt": "#ef0107", "sleeve_color": "#ffffff", "shorts": "#ffffff"}}
    assert lint_kit_safety(arsenal, COMIC, 12.0) == []


def test_posterize_flags_skin_crush():
    assert "skin_crush" in codes(lint_kit_safety({"home": LIVERPOOL}, {"posterize": 4}, 12.0))


def test_apply_post_posterize_floors_linear():
    out = apply_post(np.array([0.5, 0.26, 0.9]), {"posterize": 4})
    assert list(out) == [0.5, 0.25, 0.75]
