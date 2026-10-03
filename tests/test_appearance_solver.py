from types import SimpleNamespace

import numpy as np

from src.utils import kit_palette as kp
from src.utils.appearance_solver import solve_appearance
from src.utils.kit_library import load_library
from src.utils.team_clustering import cluster_teams

LIB = load_library()


def _parts(kit, rng):
    j = lambda: rng.normal(0, 1.0, 3)  # noqa: E731
    return {"torso": kp.hex_to_lab(kit["shirt"]) + j(), "shorts": kp.hex_to_lab(kit["shorts"]) + j(),
            "socks": kp.hex_to_lab(kit["socks"]) + j()}


def _world(home, away, away_gk, ref, seed=0, gk_home=None, away_x_low=False):
    rng = np.random.default_rng(seed)
    players, px = {}, {}
    for i in range(10):
        players[f"P{i:03d}"] = _parts(home, rng)
        px[f"P{i:03d}"] = 14 + 3 * i
    for i in range(10, 20):
        players[f"P{i:03d}"] = _parts(away, rng)
        px[f"P{i:03d}"] = 6 + 2 * (i - 10)                   # away (defending x=0) sits deeper
    players["P020"] = _parts(away_gk, rng)
    px["P020"] = 2.0
    players["P021"] = _parts(ref, rng)
    px["P021"] = 55.0
    return players, px


def _solve(players, px, **kw):
    c = cluster_teams(players, px)
    return c, solve_appearance(c, players, kp.IDENTITY_WB, library=LIB, **kw)


def test_match_pool_names_roles_from_matchinfo():
    home, away = LIB.get("liverpool/2025-26/home"), LIB.get("chelsea/2025-26/home")
    gk, ref = LIB.get("chelsea/2025-26/gk"), LIB.get("referees/yellow")
    players, px = _world(home, away, gk, ref)
    match = SimpleNamespace(home_team="Liverpool FC", away_team="Chelsea", date="2025-09-14")
    c, r = _solve(players, px, match=match)
    assert r.kits["home"]["ref"] == "liverpool/2025-26/home" and r.kits["home"]["source"] == "library"
    assert r.kits["away"]["ref"] == "chelsea/2025-26/home"
    assert r.kits["away_gk"]["ref"] == "chelsea/2025-26/gk"
    assert r.kits["referee"]["ref"] == "referees/yellow"
    assert r.player_roles["P000"] == "home" and r.player_roles["P015"] == "away"
    assert r.player_roles["P020"] == "away_gk" and r.player_roles["P021"] == "referee"
    assert "home_away_unassigned" not in r.needs_confirmation


def test_clip_pool_overrides_and_swaps_teams_correctly():
    # team 0 (smallest pid) wears blue here: clip kits must still name it correctly
    home, away = LIB.get("chelsea/2025-26/home"), LIB.get("liverpool/2025-26/home")
    gk, ref = LIB.get("liverpool/2025-26/gk"), LIB.get("referees/black")
    players, px = _world(home, away, gk, ref, seed=3)
    clip = {"home": LIB.get("liverpool/2025-26/home"), "away": LIB.get("chelsea/2025-26/home"),
            "away_gk": LIB.get("chelsea/2025-26/gk"), "home_gk": LIB.get("liverpool/2025-26/gk")}
    c, r = _solve(players, px, clip_kits=clip)
    assert r.player_roles["P000"] == "away" and r.player_roles["P015"] == "home"
    assert r.kits["home"]["source"] == "clip"
    assert r.player_roles["P020"] == "home_gk"
    assert r.kits["referee"]["ref"] == "referees/black"


def test_no_metadata_library_wide_snap_and_needs_confirmation():
    home, away = LIB.get("bournemouth/2025-26/home"), LIB.get("manchester_city/2025-26/away")
    gk, ref = LIB.get("manchester_city/2025-26/gk"), LIB.get("referees/black")
    players, px = _world(home, away, gk, ref, seed=5)
    c, r = _solve(players, px)
    assert "home_away_unassigned" in r.needs_confirmation
    refs = {r.kits["home"].get("ref"), r.kits["away"].get("ref")}
    assert refs == {"bournemouth/2025-26/home", "manchester_city/2025-26/away"}


def test_unknown_kits_fall_back_to_sampled_spec():
    odd1 = {"shirt": "#ff00ff", "shorts": "#ff00ff", "socks": "#ff00ff"}
    odd2 = {"shirt": "#00ffff", "shorts": "#00ffff", "socks": "#00ffff"}
    players, px = _world(odd1, odd2, odd1, odd2, seed=7)
    # make the odd keeper/ref clearly outliers
    players["P020"] = _parts({"shirt": "#ff8800", "shorts": "#ff8800", "socks": "#ff8800"}, np.random.default_rng(1))
    players["P021"] = _parts({"shirt": "#101010", "shorts": "#101010", "socks": "#101010"}, np.random.default_rng(2))
    c, r = _solve(players, px)
    sources = {k: v["source"] for k, v in r.kits.items()}
    assert sources["home"] == "sampled" and sources["away"] == "sampled"
    assert any(n.endswith("kit_not_in_library") for n in r.needs_confirmation)


def test_match_pool_accepts_loose_but_decisive_pairing_and_flags_it():
    # broadcast-dulled kits: far from the brand hexes, but clearly paired with their own club
    dull_home = {"shirt": "#623635", "shorts": "#3c3029", "socks": "#313123"}
    dull_away = {"shirt": "#9b9797", "shorts": "#aaa1aa", "socks": "#706f6e"}
    gk = LIB.get("manchester_city/2025-26/gk")
    players, px = _world(dull_home, dull_away, gk, LIB.get("referees/black"), seed=11)
    match = SimpleNamespace(home_team="Bournemouth", away_team="Man City", date=None)
    c, r = _solve(players, px, match=match)
    assert r.kits["home"]["ref"] == "bournemouth/2025-26/home"
    assert r.kits["away"]["ref"] == "manchester_city/2025-26/away"
    assert "away_snap_loose" in r.needs_confirmation


def test_library_wide_pool_does_not_accept_loose_snaps():
    dull_home = {"shirt": "#623635", "shorts": "#3c3029", "socks": "#313123"}
    dull_away = {"shirt": "#9b9797", "shorts": "#aaa1aa", "socks": "#706f6e"}
    players, px = _world(dull_home, dull_away, dull_away, dull_home, seed=12)
    players["P020"] = _parts({"shirt": "#ff8800", "shorts": "#ff8800", "socks": "#ff8800"},
                             np.random.default_rng(1))
    players["P021"] = _parts({"shirt": "#101010", "shorts": "#101010", "socks": "#101010"},
                             np.random.default_rng(2))
    c, r = _solve(players, px)
    assert r.kits["away"]["source"] == "sampled"
