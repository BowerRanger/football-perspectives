import json

import pytest
import yaml

from src.schemas.kit import KitSpecError, normalise_kit_spec
from src.utils import kit_library as kl
from src.utils.kit_resolution import effective_team_kits


def test_season_for_date():
    assert kl.season_for_date("2026-05-12") == "2025-26"
    assert kl.season_for_date("2025-09-01") == "2025-26"
    assert kl.season_for_date("2019-05-07") == "2018-19"
    assert kl.season_for_date("") is None


def test_shipped_library_all_valid_and_seeded():
    lib = kl.load_library()
    for ref in ("liverpool/2025-26/home", "chelsea/2025-26/home", "liverpool/2018-19/home",
                "barcelona/2018-19/away", "bournemouth/2025-26/home",
                "manchester_city/2025-26/away", "arsenal/2023-24/home",
                "coventry_city/2023-24/home", "chelsea/2025-26/gk", "referees/yellow"):
        assert ref in lib.kits, ref
        normalise_kit_spec(lib.kits[ref].spec)
    assert lib.get("bournemouth/2025-26/home")["pattern"]["type"] == "vertical_stripes"
    assert lib.get("arsenal/2023-24/home")["sleeve_color"] == "#ffffff"
    assert lib.get("chelsea/2025-26/home")["socks"] == "#f2f2f2"


def test_club_lookup_and_select():
    lib = kl.load_library()
    assert lib.club_slug("Man City") == "manchester_city"
    assert lib.club_slug("AFC Bournemouth") == "bournemouth"
    assert lib.club_slug("Nobody FC") is None
    pool = lib.select(slugs=["liverpool"], season="2025-26")
    assert set(pool) == {"liverpool/2025-26/home"}
    assert "chelsea/2025-26/gk" in lib.select(keepers=True)
    assert set(lib.select(referees=True)) >= {"referees/black", "referees/yellow"}


def test_resolve_kit_ref_and_inline():
    assert kl.resolve_kit("liverpool/2025-26/home")["shirt"] == "#c8102e"
    inline = kl.resolve_kit({"shirt": "#FFFFFF", "shorts": "#000000", "socks": "#000000"})
    assert inline["shirt"] == "#ffffff"
    with pytest.raises(KitSpecError):
        kl.resolve_kit("nope/2000-01/home")


def test_custom_library_dir(tmp_path):
    (tmp_path / "x_fc.yaml").write_text(yaml.safe_dump({
        "club": "X FC", "kits": {"2030-31/home": {"shirt": "#112233", "shorts": "#112233", "socks": "#112233"},
                                  "bad": {"shirt": "red"}}}))
    lib = kl.load_library(tmp_path)
    assert list(lib.kits) == ["x_fc/2030-31/home"]


def test_effective_team_kits_precedence(tmp_path):
    cfg = {
        "render": {"teams": {"defaults": {
            "home": {"shirt": "#111111", "shorts": "#111111", "socks": "#111111"},
            "away": {"shirt": "#222222", "shorts": "#222222", "socks": "#222222"},
            "referee": {"shirt": "#333333", "shorts": "#333333", "socks": "#333333"}}}},
        "appearance": {"kits": {"home": "liverpool/2025-26/home"}},
    }
    # defaults only for away/referee, clip kit for home
    kits = effective_team_kits(tmp_path, cfg)
    assert kits["home"]["shirt"] == "#c8102e"
    assert kits["away"]["shirt"] == "#222222"
    # auto kits.json beats defaults, loses to clip config
    ap = tmp_path / "appearance"
    ap.mkdir()
    (ap / "kits.json").write_text(json.dumps({"kits": {
        "home": {"shirt": "#aaaaaa", "shorts": "#aaaaaa", "socks": "#aaaaaa"},
        "away": {"shirt": "#bbbbbb", "shorts": "#bbbbbb", "socks": "#bbbbbb"}}}))
    kits = effective_team_kits(tmp_path, cfg)
    assert kits["home"]["shirt"] == "#c8102e"      # clip > auto
    assert kits["away"]["shirt"] == "#bbbbbb"      # auto > defaults
    assert kits["referee"]["shirt"] == "#333333"   # defaults survive
    # operator beats everything
    (ap / "kits_operator.json").write_text(json.dumps({
        "home": "chelsea/2025-26/home", "away": {"shirt": "#010203", "shorts": "#010203", "socks": "#010203"}}))
    kits = effective_team_kits(tmp_path, cfg)
    assert kits["home"]["shirt"] == "#034694"
    assert kits["away"]["shirt"] == "#010203"


def test_effective_team_kits_no_appearance_is_defaults(tmp_path):
    cfg = {"render": {"teams": {"defaults": {"home": {"shirt": "#111111", "shorts": "#111111", "socks": "#111111"}}}}}
    assert effective_team_kits(tmp_path, cfg) == {"home": cfg["render"]["teams"]["defaults"]["home"]}
    assert effective_team_kits(tmp_path, None) == {}
