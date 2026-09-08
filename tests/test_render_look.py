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


# --- merge_partial ----------------------------------------------------

@pytest.mark.unit
def test_merge_partial_none_overrides_returns_defaults():
    defaults = {"a": 1, "b": 2}
    assert rl.merge_partial(defaults, None) == defaults
    # Must return a fresh dict, not the same object (caller mutation
    # safety — _resolve_style builds several of these per call).
    assert rl.merge_partial(defaults, None) is not defaults


@pytest.mark.unit
def test_merge_partial_overrides_only_given_keys():
    defaults = {"a": 1, "b": 2, "c": 3}
    merged = rl.merge_partial(defaults, {"b": 20})
    assert merged == {"a": 1, "b": 20, "c": 3}


@pytest.mark.unit
def test_merge_partial_empty_dict_overrides_nothing():
    defaults = {"a": 1}
    assert rl.merge_partial(defaults, {}) == defaults


# --- duotone_colors -----------------------------------------------------

@pytest.mark.unit
def test_duotone_colors_none_when_absent():
    assert rl.duotone_colors(None) is None
    assert rl.duotone_colors({}) is None


@pytest.mark.unit
@pytest.mark.parametrize("duotone", [
    {"shadow": "#000033"},
    {"highlight": "#ffdd00"},
    {"shadow": None, "highlight": "#ffdd00"},
    {"shadow": "#000033", "highlight": None},
])
def test_duotone_colors_none_when_only_one_endpoint_set(duotone):
    assert rl.duotone_colors(duotone) is None


@pytest.mark.unit
def test_duotone_colors_resolves_both_endpoints():
    shadow, highlight = rl.duotone_colors(
        {"shadow": "#000033", "highlight": "#ffdd00"})
    assert shadow == rl.hex_to_linear_rgba("#000033")
    assert highlight == rl.hex_to_linear_rgba("#ffdd00")


# --- post_style_is_active ------------------------------------------------

_NEUTRAL_POST = {
    "glare": 0.0,
    "grain": 0.0,
    "vignette": 0.0,
    "posterize": 0,
    "duotone": {"shadow": None, "highlight": None},
    "saturation": 1.0,
}


@pytest.mark.unit
def test_post_style_is_active_false_for_absent_or_neutral():
    assert rl.post_style_is_active(None) is False
    assert rl.post_style_is_active({}) is False
    assert rl.post_style_is_active(_NEUTRAL_POST) is False


@pytest.mark.unit
@pytest.mark.parametrize("overrides", [
    {"glare": 0.5},
    {"grain": 0.2},
    {"vignette": 0.3},
    {"posterize": 4},
    {"saturation": 0.0},
    {"saturation": 1.5},
    {"duotone": {"shadow": "#000000", "highlight": "#ffffff"}},
])
def test_post_style_is_active_true_when_one_effect_deviates(overrides):
    post = {**_NEUTRAL_POST, **overrides}
    assert rl.post_style_is_active(post) is True


@pytest.mark.unit
def test_post_style_is_active_false_for_posterize_one():
    # A single "step" collapses everything to one flat level (an
    # extreme, weird posterize) but 0/1 are both treated as the "off"
    # sentinel — only >= 2 is considered an active request.
    post = {**_NEUTRAL_POST, "posterize": 1}
    assert rl.post_style_is_active(post) is False


@pytest.mark.unit
def test_post_style_is_active_false_when_duotone_partial():
    post = {**_NEUTRAL_POST, "duotone": {"shadow": "#000000", "highlight": None}}
    assert rl.post_style_is_active(post) is False


# --- grain_noise_pixels ---------------------------------------------------

@pytest.mark.unit
def test_grain_noise_pixels_shape_and_range():
    w, h = 8, 5
    pixels = rl.grain_noise_pixels(w, h)
    assert pixels.shape == (w * h * 4,)
    assert pixels.dtype == np.float32
    assert float(pixels.min()) >= 0.0
    assert float(pixels.max()) <= 1.0
    # Alpha channel (every 4th value) must be fully opaque.
    rgba = pixels.reshape(h, w, 4)
    assert np.all(rgba[..., 3] == 1.0)
    # Grayscale: R == G == B at every pixel.
    assert np.array_equal(rgba[..., 0], rgba[..., 1])
    assert np.array_equal(rgba[..., 0], rgba[..., 2])


@pytest.mark.unit
def test_grain_noise_pixels_centered_near_neutral():
    pixels = rl.grain_noise_pixels(64, 64, seed=1)
    rgba = pixels.reshape(64, 64, 4)
    # OVERLAY-blend neutral point is 0.5 — the mean of a large enough
    # tile should land close to it (not proof of distribution shape,
    # just that this isn't some wildly off-center noise generator).
    assert rgba[..., 0].mean() == pytest.approx(0.5, abs=0.05)
    # Actually varies (not a flat/degenerate buffer).
    assert rgba[..., 0].std() > 0.01


@pytest.mark.unit
def test_grain_noise_pixels_deterministic_per_seed():
    a = rl.grain_noise_pixels(16, 16, seed=42)
    b = rl.grain_noise_pixels(16, 16, seed=42)
    assert np.array_equal(a, b)


@pytest.mark.unit
def test_grain_noise_pixels_differs_across_seeds():
    a = rl.grain_noise_pixels(16, 16, seed=1)
    b = rl.grain_noise_pixels(16, 16, seed=2)
    assert not np.array_equal(a, b)


@pytest.mark.unit
def test_grain_noise_pixels_does_not_perturb_global_random_state():
    np.random.seed(1234)
    before = np.random.get_state()[1].copy()
    rl.grain_noise_pixels(32, 32, seed=7)
    after = np.random.get_state()[1]
    assert np.array_equal(before, after)
