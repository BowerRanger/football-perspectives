"""football-data.org-backed match metadata source.

Pulls fixture data (date, score, venue, competition) and per-club
squads via the v4 REST API. Free tier covers 12 competitions including
the Premier League, La Liga, Bundesliga, Serie A, Ligue 1, and the
Champions League; for fixtures outside those leagues the lookup
returns ``[]`` and the user falls back to manual entry.

The API key is read from the ``FOOTBALL_DATA_ORG_API_KEY`` environment
variable. The free tier rate limit is 10 requests/minute — each
lookup uses 1 + N calls (one teams probe per competition until both
teams resolve, plus one matches fetch), so a single lookup of two PL
clubs costs 2 requests.
"""

from __future__ import annotations

import os
from typing import Any

import requests

from src.pipeline.match_sources import (
    MatchCandidate,
    register_source,
    season_to_date_window,
)
from src.schemas.shots import MatchInfo, RosterEntry


_BASE_URL = "https://api.football-data.org/v4"
_DEFAULT_TIMEOUT = 5.0

# Free-tier competitions in priority order. PL/PD/BL1/SA/FL1 are
# domestic top-flights, CL/EL are continental, EC/WC are international.
# Probed in this order until both team names resolve to IDs.
_COMPETITIONS: tuple[str, ...] = (
    "PL", "PD", "BL1", "SA", "FL1", "DED", "PPL", "ELC", "BSA",
    "CL", "EC", "WC",
)


class FootballDataLookupError(RuntimeError):
    """Raised when football-data.org is unreachable, returns an HTTP
    error, or rejects authentication.

    Carries the originating HTTP status code (when one is available)
    so callers can distinguish "this competition isn't on your tier"
    (403) from "the service is down" (5xx).
    """

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


def _season_start_year(season: str) -> int:
    """football-data identifies seasons by their start year as an
    integer (e.g. 2025 for "2025-26"). Reuses ``season_to_date_window``
    for canonical parsing/validation."""
    window_start, _ = season_to_date_window(season)
    return int(window_start[:4])


def _normalise(name: str) -> str:
    """Case-fold + strip for tolerant team-name matching."""
    return name.strip().lower()


def _team_name_matches(team: dict, query: str) -> bool:
    """A team in football-data.org's list matches the user's query if
    any of its name, shortName, or TLA contains/equals the query."""
    q = _normalise(query)
    if not q:
        return False
    for key in ("name", "shortName"):
        v = team.get(key) or ""
        if q in _normalise(v):
            return True
    tla = team.get("tla") or ""
    return q == _normalise(tla)


class FootballDataOrgSource:
    name = "football-data"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> None:
        # ``api_key=None`` reads from the env at construction so a
        # process started without the key still has a usable instance —
        # the error is deferred to ``lookup()`` where the API layer can
        # surface it.
        env_key = os.environ.get("FOOTBALL_DATA_ORG_API_KEY", "")
        self._api_key = api_key if api_key is not None else env_key
        self._timeout = timeout

    def lookup(
        self, *, season: str, home_team: str, away_team: str,
    ) -> list[MatchCandidate]:
        if not self._api_key:
            raise FootballDataLookupError(
                "football-data.org API key missing — set the "
                "FOOTBALL_DATA_ORG_API_KEY environment variable",
            )
        # Validates the season string and computes the start year.
        start_year = _season_start_year(season)

        home_team_obj, away_team_obj = self._resolve_teams(
            home_team=home_team,
            away_team=away_team,
            start_year=start_year,
        )
        if home_team_obj is None or away_team_obj is None:
            return []

        matches = self._fetch_team_matches(
            team_id=int(home_team_obj["id"]),
            start_year=start_year,
        )

        roster = self._squad_to_roster(
            home_squad=home_team_obj.get("squad") or [],
            away_squad=away_team_obj.get("squad") or [],
        )

        candidates: list[MatchCandidate] = []
        away_id = int(away_team_obj["id"])
        for m in matches:
            if m.get("status") != "FINISHED":
                continue
            home_id_m = (m.get("homeTeam") or {}).get("id")
            away_id_m = (m.get("awayTeam") or {}).get("id")
            if {home_id_m, away_id_m} != {int(home_team_obj["id"]), away_id}:
                continue
            info = self._match_to_info(m, roster=roster)
            filled: list[str] = []
            if info.venue:
                filled.append("venue")
            if info.competition:
                filled.append("competition")
            if info.home_score or info.away_score:
                filled.append("score")
            if roster:
                filled.append("roster")
            candidates.append(MatchCandidate(
                match=info,
                provider=self.name,
                provider_id=str(m.get("id", "")),
                confidence=0.9,
                filled_fields=filled,
            ))
        return candidates

    def _resolve_teams(
        self, *, home_team: str, away_team: str, start_year: int,
    ) -> tuple[dict | None, dict | None]:
        """Iterate free-tier competitions in priority order until both
        names resolve. Returns the **team objects** (not just IDs) so
        the caller can pull squad + venue from the same payload
        without an extra request."""
        home_obj: dict | None = None
        away_obj: dict | None = None
        for comp in _COMPETITIONS:
            try:
                data = self._get(
                    f"/competitions/{comp}/teams",
                    params={"season": start_year},
                )
            except FootballDataLookupError as exc:
                # A competition the key doesn't have access to returns
                # 403; skip it rather than failing the whole lookup.
                # Anything else (5xx, network) is fatal and propagates.
                if exc.status_code == 403:
                    continue
                raise
            teams = data.get("teams") or []
            for t in teams:
                if home_obj is None and _team_name_matches(t, home_team):
                    home_obj = t
                if away_obj is None and _team_name_matches(t, away_team):
                    away_obj = t
            if home_obj is not None and away_obj is not None:
                break
        return home_obj, away_obj

    def _fetch_team_matches(
        self, *, team_id: int, start_year: int,
    ) -> list[dict[str, Any]]:
        data = self._get(
            f"/teams/{team_id}/matches",
            params={"season": start_year},
        )
        return data.get("matches") or []

    @staticmethod
    def _squad_to_roster(
        *, home_squad: list[dict], away_squad: list[dict],
    ) -> list[RosterEntry]:
        roster: list[RosterEntry] = []
        for p in home_squad:
            if not p.get("name"):
                continue
            roster.append(RosterEntry(
                name=p["name"],
                team="A",
                position=p.get("position") or "",
                shirt_number=p.get("shirtNumber"),
            ))
        for p in away_squad:
            if not p.get("name"):
                continue
            roster.append(RosterEntry(
                name=p["name"],
                team="B",
                position=p.get("position") or "",
                shirt_number=p.get("shirtNumber"),
            ))
        return roster

    @staticmethod
    def _match_to_info(
        m: dict[str, Any], *, roster: list[RosterEntry],
    ) -> MatchInfo:
        home = m.get("homeTeam") or {}
        away = m.get("awayTeam") or {}
        score = (m.get("score") or {}).get("fullTime") or {}
        comp = (m.get("competition") or {}).get("name") or ""
        utc = m.get("utcDate") or ""
        return MatchInfo(
            home_team=home.get("name") or "",
            away_team=away.get("name") or "",
            home_score=int(score.get("home") or 0),
            away_score=int(score.get("away") or 0),
            venue=m.get("venue") or "",
            competition=comp,
            date=utc[:10] if utc else "",
            roster=list(roster),
        )

    def _get(self, path: str, *, params: dict[str, Any]) -> dict[str, Any]:
        url = _BASE_URL + path
        headers = {
            "X-Auth-Token": self._api_key,
            "User-Agent": "football-perspectives/0.1",
            "Accept": "application/json",
        }
        try:
            r = requests.get(
                url, params=params, headers=headers, timeout=self._timeout,
            )
        except requests.RequestException as exc:
            raise FootballDataLookupError(str(exc)) from exc
        if r.status_code == 429:
            raise FootballDataLookupError(
                "football-data.org rate limit reached — wait ~60s and retry",
                status_code=429,
            )
        try:
            r.raise_for_status()
        except requests.HTTPError as exc:
            raise FootballDataLookupError(
                str(exc), status_code=r.status_code,
            ) from exc
        return r.json()


# Register a default instance so the API can resolve provider name
# "football-data" without explicit setup. Reads the env var lazily on
# construction; a missing key surfaces at ``lookup()`` time so the
# server still starts cleanly during local dev.
register_source(FootballDataOrgSource())
