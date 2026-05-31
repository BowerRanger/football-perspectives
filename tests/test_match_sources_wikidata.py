"""Unit tests for the Wikidata match source.

All HTTP calls are mocked via ``unittest.mock.patch`` on
``requests.get``; no live network access in CI.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import requests

from src.pipeline.match_sources import MatchCandidate, season_to_date_window
from src.pipeline.match_sources.wikidata import (
    WikidataLookupError,
    WikidataMatchSource,
)


@pytest.mark.unit
def test_season_to_date_window_two_year_format() -> None:
    assert season_to_date_window("2021-22") == ("2021-07-01", "2022-06-30")


@pytest.mark.unit
def test_season_to_date_window_full_two_year_format() -> None:
    assert season_to_date_window("1999-2000") == ("1999-07-01", "2000-06-30")


@pytest.mark.unit
def test_season_to_date_window_single_year_format() -> None:
    assert season_to_date_window("2024") == ("2024-01-01", "2024-12-31")


@pytest.mark.unit
def test_season_to_date_window_rejects_garbage() -> None:
    for bad in ("", "abcd", "21-22", "2021-23", "2021-21", "20-21"):
        with pytest.raises(ValueError):
            season_to_date_window(bad)


@pytest.mark.unit
def test_lookup_passes_season_window_into_sparql() -> None:
    """The SPARQL query for ``_find_matches`` must be filtered by the
    full season's date range, not just one day."""
    src = WikidataMatchSource()
    with patch("src.pipeline.match_sources.wikidata.requests.get") as g:
        g.side_effect = [
            _resp(_wbsearch("Q1")),
            _resp(_wbsearch("Q2")),
            _resp({"results": {"bindings": []}}),
        ]
        src.lookup(season="2021-22", home_team="A", away_team="B")
    query = g.call_args_list[2].kwargs["params"]["query"]
    assert "2021-07-01" in query
    assert "2022-06-30" in query


def _resp(payload: dict, status: int = 200) -> MagicMock:
    """Build a minimal ``requests.Response``-like mock."""
    m = MagicMock(spec=requests.Response)
    m.status_code = status
    m.json.return_value = payload
    if status >= 400:
        m.raise_for_status.side_effect = requests.HTTPError(f"{status}")
    else:
        m.raise_for_status.return_value = None
    return m


def _wbsearch(qid: str, label: str = "Football Club", desc: str = "association football club") -> dict:
    return {"search": [{"id": qid, "label": label, "description": desc}]}


def _sparql_matches(rows: list[dict]) -> dict:
    return {"results": {"bindings": rows}}


def _binding(**fields: dict) -> dict:
    return {k: {"value": v} for k, v in fields.items()}


@pytest.mark.unit
def test_lookup_returns_candidate_with_venue_and_competition() -> None:
    """Happy path: team search succeeds for both, SPARQL returns one
    match, kits + lineup return empty (typical for Wikidata)."""
    src = WikidataMatchSource()

    search_resps = [
        _resp(_wbsearch("Q1130849", "Liverpool F.C.")),
        _resp(_wbsearch("Q8682", "Real Madrid CF")),
    ]
    matches_resp = _resp(_sparql_matches([
        _binding(
            match="http://www.wikidata.org/entity/Q108546107",
            date="2022-05-28T00:00:00Z",
            venueLabel="Stade de France",
            competitionLabel="2021–22 UEFA Champions League",
        ),
    ]))
    kits_resp = _resp(_sparql_matches([]))   # no colour data
    lineup_resp = _resp(_sparql_matches([]))  # no lineup data

    with patch("src.pipeline.match_sources.wikidata.requests.get") as g:
        g.side_effect = [*search_resps, matches_resp, kits_resp, lineup_resp]
        cands = src.lookup(season="2021-22", home_team="Liverpool", away_team="Real Madrid")

    assert len(cands) == 1
    c = cands[0]
    assert isinstance(c, MatchCandidate)
    assert c.provider == "wikidata"
    assert c.provider_id == "Q108546107"
    assert c.match.venue == "Stade de France"
    assert c.match.competition == "2021–22 UEFA Champions League"
    assert c.match.home_team == "Liverpool"
    assert c.match.away_team == "Real Madrid"
    assert "venue" in c.filled_fields
    assert "competition" in c.filled_fields


@pytest.mark.unit
def test_lookup_returns_empty_when_team_not_resolved() -> None:
    """If a team name doesn't resolve to a Q-ID, return []."""
    src = WikidataMatchSource()
    with patch("src.pipeline.match_sources.wikidata.requests.get") as g:
        g.side_effect = [
            _resp({"search": []}),  # home team not found
        ]
        cands = src.lookup(season="2021-22", home_team="Unknown", away_team="Other")
    assert cands == []


@pytest.mark.unit
def test_lookup_returns_empty_when_no_matches_found() -> None:
    src = WikidataMatchSource()
    with patch("src.pipeline.match_sources.wikidata.requests.get") as g:
        g.side_effect = [
            _resp(_wbsearch("Q1")),
            _resp(_wbsearch("Q2")),
            _resp(_sparql_matches([])),
        ]
        cands = src.lookup(season="2021-22", home_team="A", away_team="B")
    assert cands == []


@pytest.mark.unit
def test_lookup_raises_on_http_error() -> None:
    """A 502 or network error from Wikidata becomes WikidataLookupError
    so the API layer can surface it as a 502."""
    src = WikidataMatchSource()
    with patch("src.pipeline.match_sources.wikidata.requests.get") as g:
        g.side_effect = [_resp({}, status=503)]
        with pytest.raises(WikidataLookupError):
            src.lookup(season="2021-22", home_team="A", away_team="B")


@pytest.mark.unit
def test_lookup_raises_on_timeout() -> None:
    src = WikidataMatchSource()
    with patch("src.pipeline.match_sources.wikidata.requests.get") as g:
        g.side_effect = requests.Timeout("timed out")
        with pytest.raises(WikidataLookupError):
            src.lookup(season="2021-22", home_team="A", away_team="B")


@pytest.mark.unit
def test_team_search_prefers_football_descriptions() -> None:
    """When multiple Q-items match a name, prefer one whose description
    mentions 'football' / 'soccer' over an arbitrary first hit."""
    src = WikidataMatchSource()

    multi = {
        "search": [
            {"id": "Q999", "label": "Liverpool", "description": "city in England"},
            {"id": "Q1130849", "label": "Liverpool F.C.", "description": "English association football club"},
        ],
    }

    with patch("src.pipeline.match_sources.wikidata.requests.get") as g:
        g.side_effect = [
            _resp(multi),
            _resp(_wbsearch("Q8682")),
            _resp(_sparql_matches([])),
        ]
        src.lookup(season="2021-22", home_team="Liverpool", away_team="Real Madrid")

    # The SPARQL match query (3rd call) must reference Q1130849, not Q999.
    sparql_call_args = g.call_args_list[2]
    query_param = sparql_call_args.kwargs.get("params", {}).get("query", "")
    assert "Q1130849" in query_param
    assert "Q999" not in query_param


@pytest.mark.unit
def test_lookup_hydrates_kit_colors_when_present() -> None:
    src = WikidataMatchSource()

    kits_binding = _sparql_matches([
        _binding(team="http://www.wikidata.org/entity/Q1130849", color="C8102E"),
        _binding(team="http://www.wikidata.org/entity/Q8682", color="FFFFFF"),
    ])

    with patch("src.pipeline.match_sources.wikidata.requests.get") as g:
        g.side_effect = [
            _resp(_wbsearch("Q1130849")),
            _resp(_wbsearch("Q8682")),
            _resp(_sparql_matches([
                _binding(
                    match="http://www.wikidata.org/entity/Q108546107",
                    date="2022-05-28T00:00:00Z",
                ),
            ])),
            _resp(kits_binding),
            _resp(_sparql_matches([])),
        ]
        cands = src.lookup(season="2021-22", home_team="Liverpool", away_team="Real Madrid")

    assert cands[0].match.kits is not None
    assert cands[0].match.kits.home_primary == "#c8102e"
    assert cands[0].match.kits.away_primary == "#ffffff"
    assert "kits" in cands[0].filled_fields
