"""Shipped clip configs resolve through the kit library."""

import json
from pathlib import Path

import pytest
import yaml

from src.schemas.kit import normalise_kit_spec
from src.utils.kit_palette import delta_e_hex
from src.utils.kit_resolution import effective_team_kits

ROOT = Path(__file__).resolve().parents[1]
CLIPS = ["gberch", "kroupi", "origi", "saka"]


def _cfg(clip: str) -> dict:
    return yaml.safe_load((ROOT / "config" / "clips" / f"{clip}.yaml").read_text())


@pytest.mark.parametrize("clip", CLIPS)
def test_clip_kits_resolve_and_validate(clip, tmp_path):
    cfg = _cfg(clip)
    kits = effective_team_kits(tmp_path, cfg)
    assert {"home", "away"} <= set(kits)
    for kit in kits.values():
        normalise_kit_spec(kit)
    assert cfg["ball"]["detection_cache"]["enabled"] is True


@pytest.mark.parametrize("clip,venue", [("kroupi", "vitality_stadium"), ("origi", "anfield"),
                                         ("saka", "emirates_stadium")])
def test_new_clip_configs_carry_venue(clip, venue):
    assert _cfg(clip)["render"]["venue"] == venue


def test_gberch_library_kits_match_frozen_hand_palette(tmp_path):
    hand = json.loads((ROOT / "tests/fixtures/appearance/gberch_hand_palette.json").read_text())
    kits = effective_team_kits(tmp_path, _cfg("gberch"))
    assert set(hand) == set(kits)
    for role, spec in hand.items():
        for part in ("shirt", "shorts", "socks", "boots"):
            assert delta_e_hex(kits[role][part], spec[part]) < 1.0, (role, part)
        assert kits[role]["sleeves"] == spec.get("sleeves", "short")
        assert kits[role].get("gloves") == spec.get("gloves")


def test_gberch_yaml_no_longer_carries_hand_teams():
    assert "teams" not in _cfg("gberch").get("render", {})


def test_default_yaml_appearance_block():
    cfg = yaml.safe_load((ROOT / "config" / "default.yaml").read_text())
    a = cfg["appearance"]
    assert a["library"]["snap_de"] == 12.0 and a["white_balance"]["target"] == "#f5f5f0"
    assert cfg["render"]["teams"]["defaults"]  # placeholder defaults remain the last-resort layer
