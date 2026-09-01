import numpy as np
import pytest

from src.utils import render_look as rl
from src.utils.virtual_cameras import intrinsics_from_fov, look_at_view


@pytest.mark.unit
@pytest.mark.parametrize("frac,zone", [
    (0.05, "socks"), (0.14, "socks"),
    (0.20, "skin"), (0.47, "skin"),
    (0.50, "shorts"), (0.57, "shorts"),
    (0.60, "shirt"), (0.85, "shirt"),
    (0.90, "skin"), (1.00, "skin"),      # head
])
def test_kit_zones(frac, zone):
    assert rl.kit_zone_for_height_fraction(frac) == zone


@pytest.mark.unit
def test_hex_to_linear_rgba():
    r, g, b, a = rl.hex_to_linear_rgba("#ffffff")
    assert (r, g, b, a) == (1.0, 1.0, 1.0, 1.0)
    r, g, b, a = rl.hex_to_linear_rgba("#000000")
    assert (r, g, b) == (0.0, 0.0, 0.0)
    # mid-grey: linearised value must be < srgb value (gamma expansion)
    r, _, _, _ = rl.hex_to_linear_rgba("#808080")
    assert 0.15 < r < 0.25


@pytest.mark.unit
def test_resolve_player_colors_by_player_override():
    """``by_player`` overrides are keyed by kit ROLE (a ``render.teams.
    defaults`` key), not by the tracking ``team`` value — this is
    orthogonal to the team_class -> role derivation covered by
    ``test_resolve_player_colors_real_producer_vocabulary`` below, so
    ``team_class`` here uses the real "A"/"B" vocabulary too."""
    teams = {
        "defaults": {
            "home": {"shirt": "#ff0000", "shorts": "#ffffff", "socks": "#ff0000"},
            "away": {"shirt": "#0000ff", "shorts": "#000000", "socks": "#0000ff"},
        },
        "by_player": {"P009": "away"},
    }
    team_class = {"P001": ("A", "player"), "P009": ("A", "player")}
    colors = rl.resolve_player_colors(teams, team_class)
    assert colors["P001"]["shirt"] == rl.hex_to_linear_rgba("#ff0000")
    assert colors["P009"]["shirt"] == rl.hex_to_linear_rgba("#0000ff")  # override wins


@pytest.mark.unit
def test_resolve_player_colors_real_producer_vocabulary():
    """``team_class`` values come from the REAL producer vocabulary
    (``_player_team_class_map`` reading ``src/schemas/tracks.py`` Track
    fields): ``team`` in "A"|"B"|"referee"|"unknown", ``class_name`` in
    "player"|"goalkeeper"|"referee"|"ball". Pre-fix, ``resolve_player_colors``
    did ``defaults.get(team)`` directly against that vocabulary while
    ``render.teams.defaults`` is keyed "home"/"away"/"referee" — so every
    player without a ``by_player`` override fell through to plain gray.
    This pins the fix: routing through
    ``src.utils.team_roles.derive_kit_role`` so A/B map to home/away,
    goalkeeper promotes to the *_gk role, and referee is recognised from
    either team or class_name.
    """
    teams = {
        "defaults": {
            "home": {"shirt": "#c0392b", "shorts": "#ffffff", "socks": "#c0392b"},
            "away": {"shirt": "#2980b9", "shorts": "#2c3e50", "socks": "#2980b9"},
            "home_gk": {"shirt": "#f1c40f", "shorts": "#2c3e50", "socks": "#f1c40f"},
            "away_gk": {"shirt": "#27ae60", "shorts": "#2c3e50", "socks": "#27ae60"},
            "referee": {"shirt": "#222222", "shorts": "#222222", "socks": "#222222"},
        },
        "by_player": {},
    }
    team_class = {
        "P001": ("A", "player"),
        "P002": ("B", "goalkeeper"),
        "P003": ("unknown", "referee"),
    }
    colors = rl.resolve_player_colors(teams, team_class)
    assert colors["P001"]["shirt"] == rl.hex_to_linear_rgba("#c0392b")  # home
    assert colors["P002"]["shirt"] == rl.hex_to_linear_rgba("#27ae60")  # away_gk
    assert colors["P003"]["shirt"] == rl.hex_to_linear_rgba("#222222")  # referee


@pytest.mark.unit
def test_blender_camera_matrix_position_and_forward():
    centre = np.array([10.0, 5.0, 20.0])
    target = np.array([50.0, 34.0, 0.0])
    R, t = look_at_view(centre, target)
    M = np.asarray(rl.blender_camera_world_matrix(
        [list(r) for r in R], list(t)))
    assert M[:3, 3] == pytest.approx(centre, abs=1e-9)   # translation = C
    # Blender cameras look down local -Z: -M[:3,2] must point at target.
    fwd = -M[:3, 2]
    expect = (target - centre) / np.linalg.norm(target - centre)
    assert fwd == pytest.approx(expect, abs=1e-9)


@pytest.mark.unit
def test_lens_mm_from_K():
    K = intrinsics_from_fov(46.8, (1920, 1080))  # ≈ 36mm-equiv horizontal fov
    lens = rl.lens_mm_from_K(K, 1920)
    assert lens == pytest.approx(41.6, abs=1.0)
