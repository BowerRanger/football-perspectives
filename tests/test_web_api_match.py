"""Integration tests for the match metadata endpoints
(``GET/PUT /api/match`` and ``POST /api/match/lookup``)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from src.pipeline.match_sources import MatchCandidate
from src.schemas.shots import MatchInfo, ShotsManifest
from src.web.server import create_app


@pytest.fixture
def client(tmp_path: Path):
    app = create_app(output_dir=tmp_path, config_path=None)
    return TestClient(app), tmp_path


def _full_match_body() -> dict:
    return {
        "home_team": "Liverpool",
        "away_team": "Real Madrid",
        "home_score": 0,
        "away_score": 1,
        "venue": "Stade de France",
        "competition": "UEFA Champions League",
        "date": "2022-05-28",
        "moment": {
            "minute": 59,
            "added_time": 0,
            "event_type": "goal",
            "description": "Vinicius",
        },
        "kits": {
            "home_primary": "#c8102e",
            "away_primary": "#ffffff",
            "home_goalkeeper": "",
            "away_goalkeeper": "",
            "referee": "#000000",
        },
        "roster": [
            {"name": "Salah", "team": "A", "position": "FWD", "shirt_number": 11},
            {"name": "Vinicius", "team": "B", "position": "FWD", "shirt_number": 20},
        ],
    }


@pytest.mark.integration
def test_get_match_on_fresh_output_returns_null(client) -> None:
    c, _ = client
    resp = c.get("/api/match")
    assert resp.status_code == 200
    assert resp.json() is None


@pytest.mark.integration
def test_put_match_creates_stub_manifest_when_missing(client) -> None:
    c, tmp = client
    body = _full_match_body()
    resp = c.put("/api/match", json=body)
    assert resp.status_code == 200, resp.text

    manifest_path = tmp / "shots" / "shots_manifest.json"
    assert manifest_path.exists()
    saved = json.loads(manifest_path.read_text())
    assert saved["shots"] == []
    assert saved["match"]["home_team"] == "Liverpool"
    assert saved["match"]["moment"]["minute"] == 59
    assert saved["match"]["kits"]["home_primary"] == "#c8102e"
    assert len(saved["match"]["roster"]) == 2


@pytest.mark.integration
def test_put_match_round_trips_via_get(client) -> None:
    c, _ = client
    body = _full_match_body()
    c.put("/api/match", json=body)
    resp = c.get("/api/match")
    assert resp.status_code == 200
    got = resp.json()
    assert got["home_team"] == "Liverpool"
    assert got["roster"][0]["name"] == "Salah"
    assert got["kits"]["away_primary"] == "#ffffff"


@pytest.mark.integration
def test_put_match_preserves_existing_shots(client) -> None:
    c, tmp = client
    shots_dir = tmp / "shots"
    shots_dir.mkdir()
    existing = ShotsManifest(
        source_file="src.mp4", fps=30.0, total_frames=120,
        shots=[],
    )
    # Seed a manifest with shots already present so the PUT preserves them.
    raw = {
        "source_file": "src.mp4",
        "fps": 30.0,
        "total_frames": 120,
        "shots": [
            {
                "id": "g1", "start_frame": 0, "end_frame": 60,
                "start_time": 0.0, "end_time": 2.0,
                "clip_file": "shots/g1.mp4", "speed_factor": 1.0,
            },
        ],
    }
    (shots_dir / "shots_manifest.json").write_text(json.dumps(raw))

    resp = c.put("/api/match", json=_full_match_body())
    assert resp.status_code == 200

    saved = json.loads((shots_dir / "shots_manifest.json").read_text())
    assert len(saved["shots"]) == 1
    assert saved["shots"][0]["id"] == "g1"
    assert saved["match"]["home_team"] == "Liverpool"


@pytest.mark.integration
def test_put_match_rejects_malformed_body(client) -> None:
    c, _ = client
    # Missing required field (home_team)
    bad = _full_match_body()
    del bad["home_team"]
    resp = c.put("/api/match", json=bad)
    assert resp.status_code == 422


@pytest.mark.integration
def test_put_match_without_optional_subblocks(client) -> None:
    c, _ = client
    body = {
        "home_team": "A", "away_team": "B",
        "home_score": 1, "away_score": 0,
        "venue": "Stadium",
    }
    resp = c.put("/api/match", json=body)
    assert resp.status_code == 200
    got = c.get("/api/match").json()
    assert got["moment"] is None
    assert got["kits"] is None
    assert got["roster"] == []


@pytest.mark.integration
def test_lookup_returns_candidates_from_registered_source(client) -> None:
    c, _ = client
    fake = MatchCandidate(
        match=MatchInfo(
            home_team="Liverpool", away_team="Real Madrid",
            home_score=0, away_score=1, venue="Stade de France",
            date="2022-05-28",
        ),
        provider="wikidata",
        provider_id="Q108546107",
        confidence=0.7,
        filled_fields=["venue"],
    )
    with patch(
        "src.pipeline.match_sources.wikidata.WikidataMatchSource.lookup",
        return_value=[fake],
    ), patch(
        "src.pipeline.match_sources.wikidata.fetch_kits_by_team_names",
        return_value=None,
    ):
        resp = c.post("/api/match/lookup", json={
            "season": "2021-22",
            "home_team": "Liverpool",
            "away_team": "Real Madrid",
            "provider": "wikidata",
        })
    assert resp.status_code == 200
    body = resp.json()
    assert len(body) == 1
    assert body[0]["provider"] == "wikidata"
    assert body[0]["provider_id"] == "Q108546107"
    assert body[0]["match"]["venue"] == "Stade de France"
    assert body[0]["filled_fields"] == ["venue"]


@pytest.mark.integration
def test_lookup_returns_empty_list_when_no_matches(client) -> None:
    c, _ = client
    with patch(
        "src.pipeline.match_sources.wikidata.WikidataMatchSource.lookup",
        return_value=[],
    ):
        resp = c.post("/api/match/lookup", json={
            "season": "2021-22",
            "home_team": "A",
            "away_team": "B",
            "provider": "wikidata",
        })
    assert resp.status_code == 200
    assert resp.json() == []


@pytest.mark.integration
def test_lookup_returns_502_on_provider_error(client) -> None:
    from src.pipeline.match_sources.wikidata import WikidataLookupError
    c, _ = client
    with patch(
        "src.pipeline.match_sources.wikidata.WikidataMatchSource.lookup",
        side_effect=WikidataLookupError("upstream 503"),
    ):
        resp = c.post("/api/match/lookup", json={
            "season": "2021-22",
            "home_team": "A",
            "away_team": "B",
            "provider": "wikidata",
        })
    assert resp.status_code == 502


@pytest.mark.integration
def test_lookup_returns_502_when_football_data_fails(client) -> None:
    from src.pipeline.match_sources.football_data_org import (
        FootballDataLookupError,
    )
    c, _ = client
    with patch(
        "src.pipeline.match_sources.football_data_org.FootballDataOrgSource.lookup",
        side_effect=FootballDataLookupError("API key missing"),
    ):
        resp = c.post("/api/match/lookup", json={
            "season": "2025-26",
            "home_team": "Liverpool",
            "away_team": "Chelsea",
            "provider": "football-data",
        })
    assert resp.status_code == 502
    assert "API key" in resp.json()["detail"]


@pytest.mark.integration
def test_lookup_default_provider_is_football_data(client) -> None:
    """Omitting ``provider`` in the body must route to football-data,
    not Wikidata (which has near-zero match coverage)."""
    c, _ = client
    with patch(
        "src.pipeline.match_sources.football_data_org.FootballDataOrgSource.lookup",
        return_value=[],
    ) as fd_lookup, patch(
        "src.pipeline.match_sources.wikidata.WikidataMatchSource.lookup",
    ) as wd_lookup:
        c.post("/api/match/lookup", json={
            "season": "2025-26",
            "home_team": "Liverpool",
            "away_team": "Chelsea",
        })
    fd_lookup.assert_called_once()
    wd_lookup.assert_not_called()


@pytest.mark.integration
def test_lookup_hydrates_kits_from_wikidata(client) -> None:
    """A football-data candidate without kits gets them filled in
    from Wikidata post-hoc, and ``kits`` is added to filled_fields."""
    from src.schemas.shots import KitColors

    c, _ = client
    fd_cand = MatchCandidate(
        match=MatchInfo(
            home_team="Liverpool", away_team="Chelsea",
            home_score=2, away_score=1, venue="Anfield",
            competition="Premier League", date="2025-08-15",
        ),
        provider="football-data",
        provider_id="999",
        confidence=0.9,
        filled_fields=["venue", "competition", "score"],
    )
    fake_kits = KitColors(home_primary="#c8102e", away_primary="#034694")
    with patch(
        "src.pipeline.match_sources.football_data_org.FootballDataOrgSource.lookup",
        return_value=[fd_cand],
    ), patch(
        "src.pipeline.match_sources.wikidata.fetch_kits_by_team_names",
        return_value=fake_kits,
    ):
        resp = c.post("/api/match/lookup", json={
            "season": "2025-26",
            "home_team": "Liverpool",
            "away_team": "Chelsea",
            "provider": "football-data",
        })
    assert resp.status_code == 200
    body = resp.json()
    assert body[0]["match"]["kits"]["home_primary"] == "#c8102e"
    assert "kits" in body[0]["filled_fields"]


@pytest.mark.integration
def test_lookup_kit_hydration_uses_candidate_home_away_not_user_input(
    client,
) -> None:
    """User types Liverpool then Chelsea, but football-data returns a
    fixture where Chelsea is the home team. Kit colours must be
    bound to the FIXTURE's home/away (so the viewer's TEAM_COLORS.A
    is Chelsea blue), not the user's input order."""
    from src.schemas.shots import KitColors

    c, _ = client
    chelsea_home_fixture = MatchCandidate(
        match=MatchInfo(
            home_team="Chelsea FC", away_team="Liverpool FC",
            home_score=0, away_score=3, venue="Stamford Bridge",
            competition="Premier League", date="2025-10-04",
        ),
        provider="football-data",
        provider_id="555",
        confidence=0.9,
        filled_fields=["venue", "competition", "score"],
    )

    def fake_kits(home: str, away: str):
        if home == "Chelsea FC" and away == "Liverpool FC":
            return KitColors(home_primary="#034694", away_primary="#c8102e")
        # Any other arg order would be the bug: surface as wrong hex.
        return KitColors(home_primary="#000000", away_primary="#000000")

    with patch(
        "src.pipeline.match_sources.football_data_org.FootballDataOrgSource.lookup",
        return_value=[chelsea_home_fixture],
    ), patch(
        "src.pipeline.match_sources.wikidata.fetch_kits_by_team_names",
        side_effect=fake_kits,
    ):
        resp = c.post("/api/match/lookup", json={
            "season": "2025-26",
            "home_team": "Liverpool",
            "away_team": "Chelsea",
            "provider": "football-data",
        })

    body = resp.json()
    assert body[0]["match"]["kits"]["home_primary"] == "#034694"
    assert body[0]["match"]["kits"]["away_primary"] == "#c8102e"


@pytest.mark.integration
def test_lookup_survives_kit_hydration_failure(client) -> None:
    """A Wikidata outage during kit-hydration must not fail the whole
    lookup — the user still wants their football-data candidates."""
    from src.pipeline.match_sources.wikidata import WikidataLookupError

    c, _ = client
    fd_cand = MatchCandidate(
        match=MatchInfo(
            home_team="Liverpool", away_team="Chelsea",
            home_score=2, away_score=1, venue="Anfield",
            competition="Premier League", date="2025-08-15",
        ),
        provider="football-data",
        provider_id="999",
        confidence=0.9,
        filled_fields=["venue"],
    )
    with patch(
        "src.pipeline.match_sources.football_data_org.FootballDataOrgSource.lookup",
        return_value=[fd_cand],
    ), patch(
        "src.pipeline.match_sources.wikidata.fetch_kits_by_team_names",
        side_effect=WikidataLookupError("wikidata 503"),
    ):
        resp = c.post("/api/match/lookup", json={
            "season": "2025-26",
            "home_team": "Liverpool",
            "away_team": "Chelsea",
            "provider": "football-data",
        })
    assert resp.status_code == 200
    body = resp.json()
    assert len(body) == 1
    assert body[0]["match"]["kits"] is None


@pytest.mark.integration
def test_lookup_returns_404_for_unknown_provider(client) -> None:
    c, _ = client
    resp = c.post("/api/match/lookup", json={
        "season": "2021-22",
        "home_team": "A",
        "away_team": "B",
        "provider": "wikipedia",  # not registered
    })
    assert resp.status_code == 404
