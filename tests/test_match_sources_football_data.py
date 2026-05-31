"""Unit tests for the football-data.org match source. All HTTP calls
are mocked via ``unittest.mock.patch``; no live network in CI."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import requests

from src.pipeline.match_sources import MatchCandidate
from src.pipeline.match_sources.football_data_org import (
    FootballDataLookupError,
    FootballDataOrgSource,
)


def _resp(payload: dict, status: int = 200) -> MagicMock:
    m = MagicMock(spec=requests.Response)
    m.status_code = status
    m.json.return_value = payload
    if status >= 400:
        m.raise_for_status.side_effect = requests.HTTPError(f"{status}")
    else:
        m.raise_for_status.return_value = None
    return m


def _pl_teams(*, liverpool_id: int = 64, chelsea_id: int = 61) -> dict:
    return {
        "count": 2,
        "teams": [
            {
                "id": liverpool_id,
                "name": "Liverpool FC",
                "shortName": "Liverpool",
                "tla": "LIV",
                "venue": "Anfield",
                "clubColors": "Red / White",
                "squad": [
                    {"id": 3754, "name": "Mohamed Salah", "position": "Offence", "shirtNumber": 11},
                    {"id": 8911, "name": "Virgil van Dijk", "position": "Defence", "shirtNumber": 4},
                ],
            },
            {
                "id": chelsea_id,
                "name": "Chelsea FC",
                "shortName": "Chelsea",
                "tla": "CHE",
                "venue": "Stamford Bridge",
                "clubColors": "Royal Blue / White",
                "squad": [
                    {"id": 100, "name": "Cole Palmer", "position": "Offence", "shirtNumber": 20},
                ],
            },
        ],
    }


def _team_matches(rows: list[dict]) -> dict:
    return {"count": len(rows), "matches": rows}


def _match_row(*, home_id: int, away_id: int, **overrides) -> dict:
    row = {
        "id": 999,
        "utcDate": "2025-08-15T19:00:00Z",
        "status": "FINISHED",
        "homeTeam": {"id": home_id, "name": "Liverpool FC", "shortName": "Liverpool"},
        "awayTeam": {"id": away_id, "name": "Chelsea FC", "shortName": "Chelsea"},
        "score": {"fullTime": {"home": 2, "away": 1}},
        "venue": "Anfield",
        "competition": {"id": 2021, "name": "Premier League", "code": "PL"},
    }
    row.update(overrides)
    return row


@pytest.mark.unit
def test_lookup_returns_candidate_for_matched_fixture() -> None:
    """Happy path: PL team list resolves both teams, fixtures call
    returns one match, fields populated."""
    src = FootballDataOrgSource(api_key="test-key")
    with patch("src.pipeline.match_sources.football_data_org.requests.get") as g:
        g.side_effect = [
            _resp(_pl_teams()),
            _resp(_team_matches([_match_row(home_id=64, away_id=61)])),
        ]
        cands = src.lookup(season="2025-26", home_team="Liverpool", away_team="Chelsea")

    assert len(cands) == 1
    c = cands[0]
    assert isinstance(c, MatchCandidate)
    assert c.provider == "football-data"
    assert c.provider_id == "999"
    assert c.match.home_team == "Liverpool FC"
    assert c.match.away_team == "Chelsea FC"
    assert c.match.home_score == 2
    assert c.match.away_score == 1
    assert c.match.venue == "Anfield"
    assert c.match.competition == "Premier League"
    assert c.match.date == "2025-08-15"
    assert "venue" in c.filled_fields
    assert "competition" in c.filled_fields
    assert "score" in c.filled_fields


@pytest.mark.unit
def test_lookup_populates_roster_from_squads() -> None:
    """Each team's squad (from the PL teams payload) lands in the
    candidate's roster, tagged with team A (home) / B (away)."""
    src = FootballDataOrgSource(api_key="test-key")
    with patch("src.pipeline.match_sources.football_data_org.requests.get") as g:
        g.side_effect = [
            _resp(_pl_teams()),
            _resp(_team_matches([_match_row(home_id=64, away_id=61)])),
        ]
        cands = src.lookup(season="2025-26", home_team="Liverpool", away_team="Chelsea")

    roster = cands[0].match.roster
    home_names = {r.name for r in roster if r.team == "A"}
    away_names = {r.name for r in roster if r.team == "B"}
    assert "Mohamed Salah" in home_names
    assert "Virgil van Dijk" in home_names
    assert "Cole Palmer" in away_names
    assert "roster" in cands[0].filled_fields


@pytest.mark.unit
def test_lookup_filters_matches_by_opponent() -> None:
    """When the team-matches call returns multiple matches, only the
    ones whose opponent is the requested away team are kept."""
    src = FootballDataOrgSource(api_key="test-key")
    extra_match = _match_row(home_id=64, away_id=57)  # vs Arsenal
    extra_match["id"] = 1234
    extra_match["awayTeam"] = {"id": 57, "name": "Arsenal FC"}
    with patch("src.pipeline.match_sources.football_data_org.requests.get") as g:
        g.side_effect = [
            _resp(_pl_teams()),
            _resp(_team_matches([
                _match_row(home_id=64, away_id=61),
                extra_match,
            ])),
        ]
        cands = src.lookup(season="2025-26", home_team="Liverpool", away_team="Chelsea")
    assert len(cands) == 1
    assert cands[0].provider_id == "999"


@pytest.mark.unit
def test_lookup_matches_when_away_team_is_home() -> None:
    """When the away team is actually at home in the fixture, the
    candidate still surfaces (Wikidata vs football-data home/away
    semantics can differ from the user's mental model)."""
    src = FootballDataOrgSource(api_key="test-key")
    # Liverpool is queried first → fetched matches are from Liverpool's
    # perspective. A "Chelsea at home" fixture has Liverpool as awayTeam.
    flipped = _match_row(home_id=61, away_id=64)
    flipped["id"] = 555
    flipped["homeTeam"] = {"id": 61, "name": "Chelsea FC", "shortName": "Chelsea"}
    flipped["awayTeam"] = {"id": 64, "name": "Liverpool FC", "shortName": "Liverpool"}
    flipped["venue"] = "Stamford Bridge"
    flipped["score"] = {"fullTime": {"home": 0, "away": 3}}
    with patch("src.pipeline.match_sources.football_data_org.requests.get") as g:
        g.side_effect = [
            _resp(_pl_teams()),
            _resp(_team_matches([flipped])),
        ]
        cands = src.lookup(season="2025-26", home_team="Liverpool", away_team="Chelsea")
    assert len(cands) == 1
    # The candidate reflects the actual home/away from the data.
    assert cands[0].match.home_team == "Chelsea FC"
    assert cands[0].match.away_team == "Liverpool FC"
    assert cands[0].match.home_score == 0
    assert cands[0].match.away_score == 3


@pytest.mark.unit
def test_lookup_returns_empty_when_team_not_in_competition() -> None:
    """If neither team is in the PL list, lookup falls through other
    competitions; if still not found, returns []."""
    src = FootballDataOrgSource(api_key="test-key")
    empty_pl = {"count": 0, "teams": []}
    with patch("src.pipeline.match_sources.football_data_org.requests.get") as g:
        # Empty results across every probed competition.
        g.return_value = _resp(empty_pl)
        cands = src.lookup(season="2025-26", home_team="Unknown", away_team="Nope")
    assert cands == []


@pytest.mark.unit
def test_lookup_raises_when_api_key_missing(monkeypatch) -> None:
    """Provider with no API key raises a clear error so the API layer
    can surface it as a 502 with an actionable message."""
    monkeypatch.delenv("FOOTBALL_DATA_ORG_API_KEY", raising=False)
    src = FootballDataOrgSource(api_key="")
    with pytest.raises(FootballDataLookupError, match="API key"):
        src.lookup(season="2025-26", home_team="A", away_team="B")


@pytest.mark.unit
def test_lookup_raises_on_http_error() -> None:
    src = FootballDataOrgSource(api_key="test-key")
    with patch("src.pipeline.match_sources.football_data_org.requests.get") as g:
        g.return_value = _resp({}, status=503)
        with pytest.raises(FootballDataLookupError):
            src.lookup(season="2025-26", home_team="A", away_team="B")


@pytest.mark.unit
def test_lookup_raises_on_rate_limit_with_retry_hint() -> None:
    """A 429 from football-data.org should surface the retry hint so
    the user knows to wait."""
    src = FootballDataOrgSource(api_key="test-key")
    rl = _resp({"message": "You reached your request limit"}, status=429)
    with patch("src.pipeline.match_sources.football_data_org.requests.get") as g:
        g.return_value = rl
        with pytest.raises(FootballDataLookupError, match="rate limit"):
            src.lookup(season="2025-26", home_team="A", away_team="B")


@pytest.mark.unit
def test_lookup_uses_correct_season_year() -> None:
    """For season "2025-26", the team-list and matches calls must use
    season=2025 (football-data uses the start year)."""
    src = FootballDataOrgSource(api_key="test-key")
    with patch("src.pipeline.match_sources.football_data_org.requests.get") as g:
        g.side_effect = [
            _resp(_pl_teams()),
            _resp(_team_matches([])),
        ]
        src.lookup(season="2025-26", home_team="Liverpool", away_team="Chelsea")
    teams_params = g.call_args_list[0].kwargs["params"]
    matches_params = g.call_args_list[1].kwargs["params"]
    assert teams_params["season"] == 2025
    assert matches_params["season"] == 2025


@pytest.mark.unit
def test_lookup_sends_api_key_header() -> None:
    src = FootballDataOrgSource(api_key="secret-token")
    with patch("src.pipeline.match_sources.football_data_org.requests.get") as g:
        g.side_effect = [
            _resp(_pl_teams()),
            _resp(_team_matches([])),
        ]
        src.lookup(season="2025-26", home_team="Liverpool", away_team="Chelsea")
    headers = g.call_args_list[0].kwargs["headers"]
    assert headers["X-Auth-Token"] == "secret-token"


@pytest.mark.unit
def test_lookup_skips_non_finished_matches() -> None:
    """Scheduled and postponed matches haven't happened yet, so they
    don't belong in the candidate list — the score is None and would
    confuse the form."""
    src = FootballDataOrgSource(api_key="test-key")
    scheduled = _match_row(home_id=64, away_id=61)
    scheduled["id"] = 222
    scheduled["status"] = "SCHEDULED"
    scheduled["score"] = {"fullTime": {"home": None, "away": None}}
    finished = _match_row(home_id=64, away_id=61)
    finished["id"] = 333
    with patch("src.pipeline.match_sources.football_data_org.requests.get") as g:
        g.side_effect = [
            _resp(_pl_teams()),
            _resp(_team_matches([scheduled, finished])),
        ]
        cands = src.lookup(season="2025-26", home_team="Liverpool", away_team="Chelsea")
    ids = {c.provider_id for c in cands}
    assert ids == {"333"}
