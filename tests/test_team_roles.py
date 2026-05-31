"""Tests for kit-role derivation and the role -> colour palette."""

from __future__ import annotations

import pytest

from src.utils.team_roles import (
    KIT_ROLES,
    derive_kit_role,
    kit_role_from_override,
    kits_map,
    normalise_role,
)


@pytest.mark.unit
@pytest.mark.parametrize(
    "team,class_name,expected",
    [
        ("A", "player", "home"),
        ("A", "goalkeeper", "home_gk"),
        ("B", "player", "away"),
        ("B", "goalkeeper", "away_gk"),
        ("referee", "referee", "referee"),
        ("A", "referee", "referee"),  # referee class wins over team
        ("unknown", "player", "unknown"),
        ("", "", "unknown"),
        (None, None, "unknown"),
        ("home", "goalkeeper", "home_gk"),  # accepts "home"/"away" aliases
    ],
)
def test_derive_kit_role(team, class_name, expected) -> None:
    assert derive_kit_role(team, class_name) == expected


@pytest.mark.unit
@pytest.mark.parametrize(
    "team,is_gk,is_ref,expected",
    [
        ("home", False, False, "home"),
        ("home", True, False, "home_gk"),
        ("away", False, False, "away"),
        ("away", True, False, "away_gk"),
        # Referee toggle wins regardless of team / gk.
        ("home", False, True, "referee"),
        ("away", True, True, "referee"),
        (None, False, True, "referee"),
        # Accepts the tracking A/B aliases too.
        ("A", True, False, "home_gk"),
        ("B", False, False, "away"),
        # No usable team -> neutral fallback.
        ("", False, False, "unknown"),
        (None, False, False, "unknown"),
        ("bench", False, False, "unknown"),
    ],
)
def test_kit_role_from_override(team, is_gk, is_ref, expected) -> None:
    assert kit_role_from_override(team, is_gk=is_gk, is_ref=is_ref) == expected


@pytest.mark.unit
def test_kit_role_from_override_defaults() -> None:
    # gk/ref default to False.
    assert kit_role_from_override("home") == "home"
    assert kit_role_from_override("away") == "away"


@pytest.mark.unit
def test_normalise_role_aliases_and_invalid() -> None:
    assert normalise_role("home-gk") == "home_gk"
    assert normalise_role("Keeper_Away") == "away_gk"
    assert normalise_role("ref") == "referee"
    assert normalise_role("HOME") == "home"
    assert normalise_role("striker") is None
    assert normalise_role(None) is None


@pytest.mark.unit
def test_kits_map_defaults_when_no_match() -> None:
    palette = kits_map(None)
    assert set(palette) == set(KIT_ROLES)
    # Every role has a non-empty hex colour.
    assert all(v.startswith("#") for v in palette.values())


@pytest.mark.unit
def test_kits_map_overrides_from_match() -> None:
    match = {
        "kits": {
            "home_primary": "#0000ff",
            "away_primary": "#ff0000",
            "home_goalkeeper": "",  # blank -> keep default
            "referee": "#abcdef",
        }
    }
    palette = kits_map(match)
    assert palette["home"] == "#0000ff"
    assert palette["away"] == "#ff0000"
    assert palette["referee"] == "#abcdef"
    # Blank slot fell back to the default, not "".
    assert palette["home_gk"].startswith("#")
    assert palette["home_gk"] != ""
