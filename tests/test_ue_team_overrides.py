"""Tests for the UE-side ``team_overrides`` sidecar module.

That module lives in the (non-version-controlled) Unreal project at
``FootballPerspectives 5.8/Content/Python/football_perspectives/`` and has
no ``unreal`` dependency, so we load it by path and exercise its pure
resolution + JSON persistence. Skips cleanly when the UE project isn't
checked out next to this repo (e.g. CI).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_UE_MODULE = (
    Path(__file__).resolve().parents[2]
    / "FootballPerspectives 5.8"
    / "Content"
    / "Python"
    / "football_perspectives"
    / "team_overrides.py"
)


def _load_module():
    if not _UE_MODULE.exists():
        pytest.skip(f"UE project module not present at {_UE_MODULE}")
    spec = importlib.util.spec_from_file_location("ue_team_overrides", _UE_MODULE)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def to():
    return _load_module()


@pytest.mark.unit
@pytest.mark.parametrize(
    "team,is_gk,is_ref,expected",
    [
        ("home", False, False, "home"),
        ("home", True, False, "home_gk"),
        ("away", False, False, "away"),
        ("away", True, False, "away_gk"),
        ("home", False, True, "referee"),
        ("away", True, True, "referee"),
        ("A", True, False, "home_gk"),
        ("B", False, False, "away"),
        ("", False, False, "unknown"),
        (None, False, False, "unknown"),
    ],
)
def test_ue_mapping_matches_pipeline(to, team, is_gk, is_ref, expected) -> None:
    # Must agree with src.utils.team_roles.kit_role_from_override.
    from src.utils.team_roles import kit_role_from_override

    assert to.kit_role_from_override(team, is_gk=is_gk, is_ref=is_ref) == expected
    assert kit_role_from_override(team, is_gk=is_gk, is_ref=is_ref) == expected


@pytest.mark.unit
def test_load_missing_returns_empty(to, tmp_path) -> None:
    assert to.load_overrides(tmp_path, "gberch") == {}


@pytest.mark.unit
def test_load_garbled_returns_empty(to, tmp_path) -> None:
    path = to.override_path(tmp_path, "gberch")
    path.parent.mkdir(parents=True)
    path.write_text("{not json")
    assert to.load_overrides(tmp_path, "gberch") == {}


@pytest.mark.unit
def test_set_and_roundtrip(to, tmp_path) -> None:
    role = to.set_player_override(tmp_path, "gberch", "P001", team="away")
    assert role == "away"
    role = to.set_player_override(tmp_path, "gberch", "P003", team="home", is_gk=True)
    assert role == "home_gk"
    role = to.set_player_override(tmp_path, "gberch", "P007", is_ref=True)
    assert role == "referee"

    loaded = to.load_overrides(tmp_path, "gberch")
    assert loaded["P001"] == {"team": "away"}
    assert loaded["P003"] == {"team": "home", "is_gk": True}
    assert loaded["P007"] == {"is_ref": True}


@pytest.mark.unit
def test_referee_override_ignores_team(to, tmp_path) -> None:
    to.set_player_override(tmp_path, "gberch", "P007", team="home", is_gk=True, is_ref=True)
    loaded = to.load_overrides(tmp_path, "gberch")
    # Referee toggle drops team/gk from the stored entry.
    assert loaded["P007"] == {"is_ref": True}


@pytest.mark.unit
def test_role_for_player_falls_back(to, tmp_path) -> None:
    to.set_player_override(tmp_path, "gberch", "P001", team="away")
    overrides = to.load_overrides(tmp_path, "gberch")
    assert to.role_for_player("P001", overrides, "home") == "away"
    # No override -> pipeline fallback role is kept.
    assert to.role_for_player("P999", overrides, "home_gk") == "home_gk"


@pytest.mark.unit
def test_clear_override(to, tmp_path) -> None:
    to.set_player_override(tmp_path, "gberch", "P001", team="away")
    to.clear_player_override(tmp_path, "gberch", "P001")
    assert to.load_overrides(tmp_path, "gberch") == {}
