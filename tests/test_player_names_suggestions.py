"""load_kit_roles merges appearance suggestions UNDER operator players.json."""

from __future__ import annotations

import json
from pathlib import Path

from src.utils.player_names import load_kit_roles


def _suggest(tmp_path: Path, payload: dict) -> None:
    (tmp_path / "appearance").mkdir(exist_ok=True)
    (tmp_path / "appearance" / "players_suggested.json").write_text(json.dumps(payload))


def test_merge_suggestions_under_operator(tmp_path: Path) -> None:
    (tmp_path / "players.json").write_text(
        json.dumps({"P001": {"name": "A", "kit_role": "home"}})
    )
    _suggest(tmp_path, {
        "P001": {"kit_role": "away"},          # operator wins
        "P002": {"kit_role": "away_gk"},       # suggestion fills the gap
        "P003": {"kit_role": "striker"},       # invalid dropped
    })
    assert load_kit_roles(tmp_path) == {"P001": "home", "P002": "away_gk"}


def test_suggestions_only(tmp_path: Path) -> None:
    _suggest(tmp_path, {"P002": {"kit_role": "referee"}})
    assert load_kit_roles(tmp_path) == {"P002": "referee"}


def test_unchanged_without_suggestions(tmp_path: Path) -> None:
    (tmp_path / "players.json").write_text(
        json.dumps({"P001": {"name": "A", "kit_role": "home-gk"}})
    )
    assert load_kit_roles(tmp_path) == {"P001": "home_gk"}
    assert load_kit_roles(tmp_path / "missing") == {}


def test_malformed_suggestions_ignored(tmp_path: Path) -> None:
    (tmp_path / "players.json").write_text(json.dumps({"P001": {"kit_role": "home"}}))
    (tmp_path / "appearance").mkdir()
    (tmp_path / "appearance" / "players_suggested.json").write_text("{nope")
    assert load_kit_roles(tmp_path) == {"P001": "home"}
