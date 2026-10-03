"""Venue dressing library: lookup order, merge precedence, away-end split."""
import json
from pathlib import Path

import pytest
import yaml

from src.utils import stadium_dressing as sd
from src.utils.stadium_config import load_stadiums

REPO = Path(__file__).resolve().parents[1]
REGISTRY = yaml.safe_load((REPO / "config/stadiums.yaml").read_text())


def _out(tmp_path, *, anchors_stadium=None, match_venue=None, shot="s1"):
    (tmp_path / "camera").mkdir()
    (tmp_path / "shots").mkdir()
    if anchors_stadium is not None:
        (tmp_path / "camera" / f"{shot}_anchors.json").write_text(
            json.dumps({"stadium": anchors_stadium}))
    if match_venue is not None:
        (tmp_path / "shots" / "shots_manifest.json").write_text(
            json.dumps({"shots": [], "match": {"venue": match_venue}}))
    return tmp_path


def _cfg(venue=None, stadium=None, away=None):
    cfg = {"render": {"style": {"stadium": dict(stadium or {})}}}
    if venue:
        cfg["render"]["venue"] = venue
    if away:
        cfg["render"]["teams"] = {"defaults": {"away": away}}
    return cfg


def test_registry_still_loads_and_seeds_have_dressing():
    assert {"anfield", "vitality_stadium", "emirates_stadium"} <= set(load_stadiums())
    for sid in ("anfield", "vitality_stadium", "emirates_stadium"):
        assert REGISTRY["stadiums"][sid]["dressing"]["crowd_colors"]
    assert "dressing_default" in REGISTRY


def test_clip_venue_beats_anchor_stadium_beats_match_venue(tmp_path):
    out = _out(tmp_path, anchors_stadium="emirates_stadium", match_venue="Dean Court, Bournemouth")
    assert sd.resolve_dressing(_cfg("anfield"), out, "s1")["venue"] == "anfield"
    assert sd.resolve_dressing(_cfg(), out, "s1")["venue"] == "emirates_stadium"


def test_match_venue_alias(tmp_path):
    out = _out(tmp_path, match_venue="Vitality Stadium, Bournemouth")
    assert sd.resolve_dressing(_cfg(), out, "s1")["venue"] == "vitality_stadium"
    (tmp_path / "b").mkdir()
    out2 = _out(tmp_path / "b", match_venue="Anfield, Liverpool")
    assert sd.resolve_dressing(_cfg(), out2, "s1")["venue"] == "anfield"


def test_unknown_everything_falls_back_to_default(tmp_path):
    out = _out(tmp_path, match_venue="Some Park")
    d = sd.resolve_dressing(_cfg(), out, "s1")
    assert d["venue"] is None
    assert d["dressing_source"] == "default"
    assert d["seat_color"] and d["crowd_colors"]


def test_unknown_explicit_venue_warns_and_continues(tmp_path, caplog):
    out = _out(tmp_path, anchors_stadium="anfield")
    d = sd.resolve_dressing(_cfg("nowhere_park"), out, "s1")
    assert d["venue"] == "anfield"
    assert "nowhere_park" in caplog.text


def test_explicit_clip_keys_win(tmp_path):
    out = _out(tmp_path)
    d = sd.resolve_dressing(
        _cfg("anfield", stadium={"seat_color": "#123456", "roof": False}), out, "s1")
    assert d["seat_color"] == "#123456"
    assert d["roof"] is False
    assert d["crowd_colors"] == REGISTRY["stadiums"]["anfield"]["dressing"]["crowd_colors"]


def test_away_end_crowd_from_away_kit(tmp_path):
    out = _out(tmp_path)
    d = sd.resolve_dressing(_cfg("anfield", away={"shirt": "#034694", "shorts": "#034694"}), out, "s1")
    assert d["away_end"]["stand"] in ("North", "South", "East", "West")
    assert d["away_end"]["crowd_colors"].count("#034694") >= 2


def test_explicit_away_end_colours_beat_kit(tmp_path):
    out = _out(tmp_path)
    d = sd.resolve_dressing(
        _cfg("anfield", stadium={"away_end": {"stand": "West", "crowd_colors": ["#00ff00"]}},
             away={"shirt": "#034694"}), out, "s1")
    assert d["away_end"] == {"stand": "West", "crowd_colors": ["#00ff00"]}


def test_crowd_palette_for_stand():
    style = {"crowd_colors": ["#111111"], "away_end": {"stand": "East", "crowd_colors": ["#222222"]}}
    assert sd.crowd_palette_for_stand(style, "East") == ["#222222"]
    assert sd.crowd_palette_for_stand(style, "West") == ["#111111"]
    assert sd.crowd_palette_for_stand({}, "West") == list(sd.FALLBACK_CROWD_COLORS)


def test_tone_floor_lifts_near_black_only():
    assert sd.tone_floor("#000000", 38.0) != "#000000"
    assert sd.tone_floor("#c8d0d4", 38.0) == "#c8d0d4"
    lifted = sd.tone_floor("#102d39", 38.0)
    assert sd.relative_lightness(lifted) >= 37.0


def test_structural_tones_never_near_black():
    for style in ({}, {"stand_tone": "#000000", "board_color": "#050505"},
                  {"stand_tone": "#8a8f92", "board_color": "#3a4047"}):
        tones = sd.structural_tones(style)
        assert sd.relative_lightness(tones["concrete"]) >= sd.TONE_FLOOR_L - 1
        assert sd.relative_lightness(tones["board"]) >= sd.TONE_FLOOR_L - 1
        assert sd.relative_lightness(tones["steel"]) >= sd.TONE_FLOOR_L - 7
        assert sd.relative_lightness(tones["tunnels"]) >= sd.TONE_FLOOR_L - 15
    assert sd.structural_tones({"board_text_color": "#ffffff"})["board_text"] == "#ffffff"


def test_no_dressing_color_is_black_band_prone():
    for sid in ("anfield", "vitality_stadium", "emirates_stadium"):
        d = sd.resolve_dressing(_cfg(sid), Path("/nonexistent"), "s1")
        for key in ("stand_tone", "board_color"):
            assert sd.relative_lightness(sd.tone_floor(d[key], sd.TONE_FLOOR_L)) >= sd.TONE_FLOOR_L - 1
