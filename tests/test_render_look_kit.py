"""players.json kit roles, per-player appearance and anatomical kit zones
for the render stage (gberch shorts, 2026-10)."""
import json

import numpy as np
import pytest

from src.utils import render_look as rl
from src.utils.smpl_skeleton import SMPL_REST_JOINTS_YUP as J

_TEAMS = {
    "defaults": {
        "home": {"shirt": "#c8102e", "shorts": "#c8102e", "socks": "#c8102e"},
        "away": {"shirt": "#034694", "shorts": "#034694", "socks": "#f4f4f4"},
        "away_gk": {"shirt": "#2b2e31", "shorts": "#2b2e31", "socks": "#2b2e31",
                    "sleeves": "long", "gloves": "#e8e8e8", "boots": "#f0f0f0"},
    },
    "by_player": {},
}


@pytest.mark.unit
def test_resolve_player_colors_honours_players_json_roles():
    """Tracks label every player ``unknown`` under the default config, so
    without the hand-authored players.json roles the render stage painted
    every kit gray. players.json roles beat the derived role; a config
    ``by_player`` entry still beats players.json."""
    team_class = {"P001": ("unknown", "player"), "P002": ("unknown", "player"),
                  "P003": ("unknown", "player")}
    roles = {"P001": "home", "P002": "away", "P003": "home"}
    teams = {**_TEAMS, "by_player": {"P003": "away"}}
    colors = rl.resolve_player_colors(teams, team_class, role_overrides=roles)
    assert colors["P001"]["shirt"] == rl.hex_to_linear_rgba("#c8102e")
    assert colors["P002"]["socks"] == rl.hex_to_linear_rgba("#f4f4f4")
    assert colors["P003"]["shirt"] == rl.hex_to_linear_rgba("#034694")


@pytest.mark.unit
def test_resolve_player_colors_includes_role_only_players():
    """A player present only in players.json (no tracks entry) still
    resolves — the roles map is a source of player ids too."""
    colors = rl.resolve_player_colors(_TEAMS, {}, role_overrides={"P009": "away"})
    assert colors["P009"]["shirt"] == rl.hex_to_linear_rgba("#034694")


@pytest.mark.unit
def test_resolve_player_looks_kit_options_and_appearance():
    team_class = {"P005": ("unknown", "player"), "P006": ("unknown", "player")}
    roles = {"P005": "away_gk", "P006": "home"}
    appearance = {"P006": {"skin": "#5b3a29", "hair": "#111111"}}
    looks = rl.resolve_player_looks(_TEAMS, team_class, role_overrides=roles,
                                    appearance=appearance)
    gk, mid = looks["P005"], looks["P006"]
    assert gk["sleeves"] == "long" and gk["gloves"] is True
    assert gk["colors"]["gloves"] == rl.hex_to_linear_rgba("#e8e8e8")
    assert gk["colors"]["boots"] == rl.hex_to_linear_rgba("#f0f0f0")
    assert mid["sleeves"] == "short" and mid["gloves"] is False
    assert mid["colors"]["skin"] == rl.hex_to_linear_rgba("#5b3a29")
    assert mid["colors"]["hair"] == rl.hex_to_linear_rgba("#111111")
    assert gk["colors"]["skin"] == rl.hex_to_linear_rgba(rl.DEFAULT_SKIN_HEX)
    assert mid["colors"]["boots"] == rl.hex_to_linear_rgba(rl.DEFAULT_BOOTS_HEX)


def _one_hot(n_verts: int, joint_idx: list[int]) -> np.ndarray:
    w = np.zeros((n_verts, 24))
    w[np.arange(n_verts), joint_idx] = 1.0
    return w


@pytest.mark.unit
def test_anatomical_kit_zones_short_sleeves_and_legs():
    cases = [
        ("pelvis_hi", J[0] + [0, 0.08, 0.05], 0, "shirt"),
        ("pelvis_lo", J[0] + [0, -0.03, 0.05], 0, "shorts"),
        ("upper_arm_in", J[16] + [0.03, 0, 0], 16, "shirt"),
        ("upper_arm_out", J[16] + (J[18] - J[16]) * 0.85, 16, "skin"),
        ("forearm", (J[18] + J[20]) / 2, 18, "skin"),
        ("hand", J[22], 22, "skin"),
        ("thigh_hi", J[1] + (J[4] - J[1]) * 0.2, 1, "shorts"),
        ("thigh_lo", J[1] + (J[4] - J[1]) * 0.85, 1, "skin"),
        ("knee", J[4] + [0, -0.02, 0], 4, "skin"),
        ("shin", J[4] + (J[7] - J[4]) * 0.6, 4, "socks"),
        ("foot", J[10], 10, "boots"),
        ("chest", J[9] + [0, 0, 0.08], 9, "shirt"),
        ("face", J[15] + [0, 0.02, 0.09], 15, "skin"),
        ("crown", J[15] + [0, 0.16, 0], 15, "hair"),
        ("back_of_head", J[15] + [0, 0.09, -0.08], 15, "hair"),
        ("nape", J[15] + [0, 0.02, -0.08], 15, "skin"),
    ]
    verts = np.array([c[1] for c in cases])
    weights = _one_hot(len(cases), [c[2] for c in cases])
    zones = rl.anatomical_kit_zones(verts, weights, J, sleeves="short", gloves=False)
    assert dict(zip([c[0] for c in cases], zones)) == {c[0]: c[3] for c in cases}


@pytest.mark.unit
def test_anatomical_kit_zones_long_sleeves_and_gloves():
    verts = np.array([J[16] + (J[18] - J[16]) * 0.85, (J[18] + J[20]) / 2, J[22]])
    zones = rl.anatomical_kit_zones(verts, _one_hot(3, [16, 18, 22]), J,
                                    sleeves="long", gloves=True)
    assert zones == ["shirt", "shirt", "gloves"]


@pytest.mark.unit
def test_load_player_appearance(tmp_path):
    from src.utils.player_names import load_player_appearance
    (tmp_path / "players.json").write_text(json.dumps({
        "P001": {"name": "A", "skin": "#5b3a29", "hair": "#111111"},
        "P002": {"name": "B", "skin": "not-a-colour"},
        "P003": "Plain",
    }))
    assert load_player_appearance(tmp_path) == {
        "P001": {"skin": "#5b3a29", "hair": "#111111"}}
    assert load_player_appearance(tmp_path / "missing") == {}
