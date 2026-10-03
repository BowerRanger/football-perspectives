import numpy as np
import pytest

from src.schemas.kit import KitSpecError, normalise_kit_spec
from src.utils import kit_palette as kp


def test_roundtrip_hex_lab():
    for h in ("#c8102e", "#034694", "#f5f5f0", "#000000", "#ffffff"):
        assert kp.lab_to_hex(kp.hex_to_lab(h)) == h


def test_ciede2000_sharma_reference():
    # Sharma et al. 2005 test pair 1 -> 2.0425
    d = kp.delta_e2000(np.array([50, 2.6772, -79.7751]), np.array([50, 0, -82.7485]))
    assert d == pytest.approx(2.0425, abs=1e-3)


def test_white_balance_recovers_cast():
    cast = np.array([0.8, 1.0, 1.2])
    line = np.tile(kp.hex_to_rgb("#f5f5f0") * cast, (100, 1))
    wb = kp.white_balance_gains(line)
    assert wb.applied
    fixed = kp.apply_gains(kp.hex_to_rgb("#c8102e") * cast, wb)
    assert kp.delta_e_hex(kp.rgb_to_hex(fixed), "#c8102e") < 2.0


def test_white_balance_identity_when_few_pixels():
    wb = kp.white_balance_gains(np.zeros((5, 3)) + 200)
    assert not wb.applied and wb.gains == (1.0, 1.0, 1.0)


def test_line_pixel_candidates_filters_dark_and_coloured():
    img = np.zeros((40, 40, 3), np.uint8) + np.array([40, 120, 40], np.uint8)
    img[20, 20] = (240, 240, 235)
    img[10, 10] = (240, 40, 40)
    px = kp.line_pixel_candidates(img, np.array([[20, 20], [10, 10], [30, 30], [0, 0]]), patch=2)
    assert px.shape == (1, 3)


CANDS = {
    "a/home": {"shirt": "#c8102e", "shorts": "#c8102e", "socks": "#c8102e", "sleeves": "short"},
    "b/home": {"shirt": "#034694", "shorts": "#034694", "socks": "#f2f2f2", "sleeves": "short"},
    "c/home": {"shirt": "#e62333", "shorts": "#000000", "socks": "#000000", "sleeves": "short",
               "pattern": {"type": "vertical_stripes", "colors": ["#e62333", "#000000"], "width_m": 0.07}},
}


def test_snap_dull_broadcast_sample_to_library_after_wb():
    cast = np.array([0.62, 0.75, 0.7])  # dark, slightly cyan-shifted broadcast grade
    wb = kp.white_balance_gains(np.tile(kp.hex_to_rgb("#f5f5f0") * cast, (80, 1)))
    raw = {k: kp.rgb_to_hex(kp.hex_to_rgb("#c8102e") * cast) for k in ("shirt", "shorts", "socks")}
    # uncorrected sample is too dull to snap; corrected one snaps
    assert not kp.snap_kit(raw, {"a/home": CANDS["a/home"]}, snap_de=3).snapped
    fixed = {k: kp.rgb_to_hex(kp.apply_gains(kp.hex_to_rgb(v), wb)) for k, v in raw.items()}
    r = kp.snap_kit(fixed, CANDS)
    assert r.snapped and r.ref == "a/home" and r.spec["source"] == "library"


def test_snap_uses_full_vector_socks():
    r = kp.snap_kit({"shirt": "#34506c", "shorts": "#124266", "socks": "#a7b8a0"}, CANDS)
    assert r.ref == "b/home"


def test_snap_falls_back_to_corrected_sample():
    r = kp.snap_kit({"shirt": "#20d020", "shorts": "#20d020", "socks": "#20d020"}, CANDS, snap_de=12)
    assert not r.snapped and r.spec["source"] == "sampled" and r.spec["shirt"] == "#20d020"
    normalise_kit_spec(r.spec)


def test_striped_candidate_matches_stripe_colour_or_blend():
    r = kp.snap_kit({"shirt": "#d02030", "shorts": "#101010", "socks": "#101010"},
                    {"c/home": CANDS["c/home"]})
    assert r.snapped


def test_kit_spec_validation():
    s = normalise_kit_spec({"shirt": "#C8102E", "shorts": "#ffffff", "socks": "#ffffff"})
    assert s["shirt"] == "#c8102e" and s["sleeves"] == "short"
    with pytest.raises(KitSpecError):
        normalise_kit_spec({"shirt": "red", "shorts": "#ffffff", "socks": "#ffffff"})
    with pytest.raises(KitSpecError):
        normalise_kit_spec({"shirt": "#ffffff", "shorts": "#ffffff", "socks": "#ffffff",
                            "pattern": {"type": "hoops", "colors": ["#ffffff"]}})
    with pytest.raises(KitSpecError):
        normalise_kit_spec({"shirt": "#ffffff", "shorts": "#ffffff", "socks": "#ffffff", "sleeves": "none"})
