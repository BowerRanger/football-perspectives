"""Round-trip + backward/forward compat tests for the match metadata
block on ``ShotsManifest``."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest

from src.schemas.shots import (
    KitColors,
    MatchInfo,
    MomentInfo,
    RosterEntry,
    Shot,
    ShotsManifest,
)


def _full_match() -> MatchInfo:
    return MatchInfo(
        home_team="Liverpool",
        away_team="Real Madrid",
        home_score=0,
        away_score=1,
        venue="Stade de France",
        competition="UEFA Champions League",
        date="2022-05-28",
        moment=MomentInfo(
            minute=59,
            added_time=0,
            event_type="goal",
            description="Vinicius right-foot finish",
        ),
        kits=KitColors(
            home_primary="#c8102e",
            away_primary="#ffffff",
            home_goalkeeper="#005f3c",
            away_goalkeeper="#000000",
            referee="#ffeb3b",
        ),
        roster=[
            RosterEntry(name="Alisson", team="A", position="GK", shirt_number=1),
            RosterEntry(name="Salah", team="A", position="FWD", shirt_number=11),
            RosterEntry(name="Courtois", team="B", position="GK", shirt_number=1),
            RosterEntry(name="Vinicius", team="B", position="FWD", shirt_number=20),
        ],
    )


def _shot() -> Shot:
    return Shot(
        id="goal",
        start_frame=0,
        end_frame=120,
        start_time=0.0,
        end_time=4.0,
        clip_file="shots/goal.mp4",
    )


@pytest.mark.unit
def test_round_trip_with_full_match(tmp_path: Path) -> None:
    manifest = ShotsManifest(
        source_file="source.mp4",
        fps=30.0,
        total_frames=121,
        shots=[_shot()],
        match=_full_match(),
    )
    path = tmp_path / "shots_manifest.json"
    manifest.save(path)

    loaded = ShotsManifest.load(path)
    assert loaded == manifest
    assert loaded.match is not None
    assert loaded.match.moment is not None
    assert loaded.match.kits is not None
    assert loaded.match.roster[0].name == "Alisson"


@pytest.mark.unit
def test_legacy_manifest_without_match_loads(tmp_path: Path) -> None:
    """A manifest written before this feature has no ``match`` key.
    Loading it must succeed and return ``match=None``."""
    legacy = {
        "source_file": "source.mp4",
        "fps": 30.0,
        "total_frames": 121,
        "shots": [asdict(_shot())],
    }
    path = tmp_path / "shots_manifest.json"
    path.write_text(json.dumps(legacy))

    loaded = ShotsManifest.load(path)
    assert loaded.match is None
    assert loaded.shots[0].id == "goal"


@pytest.mark.unit
def test_future_manifest_with_unknown_top_level_key_loads(tmp_path: Path) -> None:
    """A manifest written by a newer version with extra top-level keys
    must not break loading. Unknown keys are dropped."""
    future = {
        "source_file": "source.mp4",
        "fps": 30.0,
        "total_frames": 121,
        "shots": [asdict(_shot())],
        "match": asdict(_full_match()),
        "tactical_phase": "high-press",
        "weather": {"wind_kph": 12},
    }
    path = tmp_path / "shots_manifest.json"
    path.write_text(json.dumps(future))

    loaded = ShotsManifest.load(path)
    assert loaded.match is not None
    assert loaded.match.home_team == "Liverpool"


@pytest.mark.unit
def test_match_without_optional_subblocks(tmp_path: Path) -> None:
    """A match block can omit moment / kits / roster individually."""
    partial = MatchInfo(
        home_team="A",
        away_team="B",
        home_score=0,
        away_score=0,
        venue="Stadium",
    )
    manifest = ShotsManifest(
        source_file="",
        fps=25.0,
        total_frames=0,
        shots=[],
        match=partial,
    )
    path = tmp_path / "shots_manifest.json"
    manifest.save(path)

    loaded = ShotsManifest.load(path)
    assert loaded.match is not None
    assert loaded.match.moment is None
    assert loaded.match.kits is None
    assert loaded.match.roster == []


@pytest.mark.unit
def test_load_filters_unknown_keys_within_match(tmp_path: Path) -> None:
    """Future fields added to ``match`` / ``moment`` / ``kits`` /
    ``roster[]`` must not break load. Same defensive filtering as Shot."""
    raw = {
        "source_file": "",
        "fps": 25.0,
        "total_frames": 0,
        "shots": [],
        "match": {
            "home_team": "A",
            "away_team": "B",
            "home_score": 1,
            "away_score": 2,
            "venue": "S",
            "tactical_block": "future",
            "moment": {
                "minute": 12,
                "weather_at_kick": "rain",  # unknown
            },
            "kits": {
                "home_primary": "#ff0000",
                "away_primary": "#0000ff",
                "alpha_channel": 1.0,  # unknown
            },
            "roster": [
                {"name": "X", "team": "A", "captain": True},  # unknown
            ],
        },
    }
    path = tmp_path / "shots_manifest.json"
    path.write_text(json.dumps(raw))

    loaded = ShotsManifest.load(path)
    assert loaded.match is not None
    assert loaded.match.moment is not None
    assert loaded.match.moment.minute == 12
    assert loaded.match.kits is not None
    assert loaded.match.kits.home_primary == "#ff0000"
    assert loaded.match.roster[0].name == "X"
