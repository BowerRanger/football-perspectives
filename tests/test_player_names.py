"""Tests for the player ID → display-name mapping helper."""

from __future__ import annotations

import json
from pathlib import Path

from src.utils.player_names import (
    display_name_for,
    load_player_names,
    safe_asset_name,
)


def test_load_missing_file_returns_empty(tmp_path: Path) -> None:
    assert load_player_names(tmp_path) == {}


def test_load_simple_mapping(tmp_path: Path) -> None:
    (tmp_path / "players.json").write_text(
        json.dumps({"P001": "Bellingham", "P002": "Saka"})
    )
    assert load_player_names(tmp_path) == {"P001": "Bellingham", "P002": "Saka"}


def test_load_extended_mapping(tmp_path: Path) -> None:
    (tmp_path / "players.json").write_text(
        json.dumps(
            {
                "P001": {"name": "Bellingham", "team": "England", "number": 22},
                "P002": {"name": "Saka"},
            }
        )
    )
    assert load_player_names(tmp_path) == {"P001": "Bellingham", "P002": "Saka"}


def test_load_skips_invalid_entries(tmp_path: Path) -> None:
    (tmp_path / "players.json").write_text(
        json.dumps({"P001": "Kane", "P002": 42, "P003": {"team": "no-name"}})
    )
    assert load_player_names(tmp_path) == {"P001": "Kane"}


def test_load_handles_invalid_json(tmp_path: Path) -> None:
    (tmp_path / "players.json").write_text("{not json")
    assert load_player_names(tmp_path) == {}


def test_safe_asset_name_alnum_pass_through() -> None:
    assert safe_asset_name("Bellingham") == "Bellingham"


def test_safe_asset_name_replaces_spaces_and_punctuation() -> None:
    assert safe_asset_name("Saka K.") == "Saka_K"
    assert safe_asset_name("Foden / Phil") == "Foden_Phil"


def test_safe_asset_name_prefixes_leading_digit() -> None:
    assert safe_asset_name("22Bellingham") == "P_22Bellingham"


def test_safe_asset_name_blank_falls_back() -> None:
    assert safe_asset_name("   ") == "player"
    assert safe_asset_name("...") == "player"


def test_display_name_for_uses_mapping() -> None:
    assert display_name_for("P001", {"P001": "Bellingham"}) == "Bellingham"


def test_display_name_for_falls_back_to_id() -> None:
    assert display_name_for("P001", {}) == "P001"


def test_display_name_for_sanitises_mapped_name() -> None:
    assert display_name_for("P001", {"P001": "Saka K."}) == "Saka_K"
