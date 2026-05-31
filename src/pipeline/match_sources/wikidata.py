"""Wikidata-backed match metadata source.

Resolves team names via ``wbsearchentities``, then looks up matches and
related facts (venue, competition, kits, lineup) via SPARQL against the
public Wikidata Query Service.

Coverage is uneven — comprehensive for major finals, sparse for routine
league fixtures. Missing fields are returned blank; callers inspect
``MatchCandidate.filled_fields`` to know what landed.
"""

from __future__ import annotations

from typing import Any

import requests

from src.pipeline.match_sources import (
    MatchCandidate,
    register_source,
    season_to_date_window,
)
from src.schemas.shots import KitColors, MatchInfo, RosterEntry

_USER_AGENT = (
    "football-perspectives/0.1 (https://github.com/joebower/football-perspectives; "
    "research project)"
)
_SEARCH_URL = "https://www.wikidata.org/w/api.php"
_SPARQL_URL = "https://query.wikidata.org/sparql"
_DEFAULT_TIMEOUT = 5.0


class WikidataLookupError(RuntimeError):
    """Raised when the Wikidata query service is unreachable, returns
    a 5xx, or otherwise fails in a way we can't recover from."""


def _binding_value(binding: dict, key: str, default: str = "") -> str:
    return binding.get(key, {}).get("value", default)


def _qid_from_uri(uri: str) -> str:
    return uri.rsplit("/", 1)[-1] if uri else ""


def _normalise_hex_colour(raw: str) -> str:
    """Wikidata stores colour as ``RRGGBB`` (no leading ``#``).
    Return a CSS-style ``#rrggbb`` or empty if the value isn't a 6-hex
    triplet."""
    s = raw.strip().lstrip("#").lower()
    if len(s) == 6 and all(c in "0123456789abcdef" for c in s):
        return f"#{s}"
    return ""


class WikidataMatchSource:
    """``MatchInfoSource`` implementation backed by Wikidata."""

    name = "wikidata"

    def __init__(
        self,
        *,
        timeout: float = _DEFAULT_TIMEOUT,
        user_agent: str = _USER_AGENT,
    ) -> None:
        self._timeout = timeout
        self._headers = {
            "User-Agent": user_agent,
            "Accept": "application/json",
        }

    def lookup(
        self, *, season: str, home_team: str, away_team: str,
    ) -> list[MatchCandidate]:
        # Resolve once so a malformed season fails fast before any HTTP.
        window_start, window_end = season_to_date_window(season)

        home_qid = self._resolve_team(home_team)
        if not home_qid:
            return []
        away_qid = self._resolve_team(away_team)
        if not away_qid:
            return []

        matches = self._find_matches(
            home_qid=home_qid,
            away_qid=away_qid,
            window_start=window_start,
            window_end=window_end,
        )
        if not matches:
            return []

        kits = self._hydrate_kits(home_qid=home_qid, away_qid=away_qid)

        candidates: list[MatchCandidate] = []
        for match_row in matches:
            roster = self._hydrate_lineup(
                match_qid=match_row["match_qid"],
                home_qid=home_qid,
                away_qid=away_qid,
            )
            info = MatchInfo(
                home_team=home_team,
                away_team=away_team,
                home_score=match_row.get("home_score", 0),
                away_score=match_row.get("away_score", 0),
                venue=match_row.get("venue", ""),
                competition=match_row.get("competition", ""),
                date=match_row.get("date", ""),
                kits=kits,
                roster=roster,
            )
            filled = []
            if info.venue:
                filled.append("venue")
            if info.competition:
                filled.append("competition")
            if info.home_score or info.away_score:
                filled.append("score")
            if kits is not None:
                filled.append("kits")
            if roster:
                filled.append("roster")
            candidates.append(MatchCandidate(
                match=info,
                provider=self.name,
                provider_id=match_row["match_qid"],
                confidence=0.7,
                filled_fields=filled,
            ))
        return candidates

    def _resolve_team(self, name: str) -> str | None:
        data = self._get(_SEARCH_URL, params={
            "action": "wbsearchentities",
            "search": name,
            "language": "en",
            "format": "json",
            "type": "item",
            "limit": "10",
        })
        results = data.get("search", []) or []
        for r in results:
            desc = (r.get("description") or "").lower()
            if "football" in desc or "soccer" in desc:
                return r.get("id")
        return results[0].get("id") if results else None

    def _find_matches(
        self, *, home_qid: str, away_qid: str,
        window_start: str, window_end: str,
    ) -> list[dict[str, Any]]:
        query = f"""
        SELECT ?match ?date ?venueLabel ?competitionLabel WHERE {{
          ?match wdt:P710 wd:{home_qid} .
          ?match wdt:P710 wd:{away_qid} .
          ?match wdt:P585 ?date .
          FILTER(?date >= "{window_start}T00:00:00Z"^^xsd:dateTime
                 && ?date <= "{window_end}T23:59:59Z"^^xsd:dateTime)
          OPTIONAL {{ ?match wdt:P276 ?venue . }}
          OPTIONAL {{ ?match wdt:P361 ?competition . }}
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
        }} LIMIT 20
        """
        data = self._get(_SPARQL_URL, params={"query": query, "format": "json"})
        rows = data.get("results", {}).get("bindings", []) or []
        out: list[dict[str, Any]] = []
        for b in rows:
            uri = _binding_value(b, "match")
            iso = _binding_value(b, "date")
            out.append({
                "match_qid": _qid_from_uri(uri),
                "date": iso[:10] if iso else "",
                "venue": _binding_value(b, "venueLabel"),
                "competition": _binding_value(b, "competitionLabel"),
            })
        return out

    def _hydrate_kits(
        self, *, home_qid: str, away_qid: str,
    ) -> KitColors | None:
        query = f"""
        SELECT ?team ?color WHERE {{
          VALUES ?team {{ wd:{home_qid} wd:{away_qid} }}
          ?team wdt:P462 ?colorItem .
          ?colorItem wdt:P465 ?color .
        }}
        """
        data = self._get(_SPARQL_URL, params={"query": query, "format": "json"})
        rows = data.get("results", {}).get("bindings", []) or []
        home_hex = ""
        away_hex = ""
        for b in rows:
            team_qid = _qid_from_uri(_binding_value(b, "team"))
            hex_val = _normalise_hex_colour(_binding_value(b, "color"))
            if not hex_val:
                continue
            if team_qid == home_qid and not home_hex:
                home_hex = hex_val
            elif team_qid == away_qid and not away_hex:
                away_hex = hex_val
        if not (home_hex or away_hex):
            return None
        return KitColors(
            home_primary=home_hex,
            away_primary=away_hex,
        )

    def _hydrate_lineup(
        self, *, match_qid: str, home_qid: str, away_qid: str,
    ) -> list[RosterEntry]:
        query = f"""
        SELECT ?player ?playerLabel ?team WHERE {{
          wd:{match_qid} wdt:P710 ?player .
          OPTIONAL {{ ?player wdt:P54 ?team .
                     VALUES ?team {{ wd:{home_qid} wd:{away_qid} }} }}
          FILTER NOT EXISTS {{ ?player wdt:P31/wdt:P279* wd:Q12973014 . }}
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
        }} LIMIT 30
        """
        data = self._get(_SPARQL_URL, params={"query": query, "format": "json"})
        rows = data.get("results", {}).get("bindings", []) or []
        roster: list[RosterEntry] = []
        for b in rows:
            name = _binding_value(b, "playerLabel")
            team_qid = _qid_from_uri(_binding_value(b, "team"))
            team = "A" if team_qid == home_qid else ("B" if team_qid == away_qid else "")
            if not name or not team:
                continue
            roster.append(RosterEntry(name=name, team=team))
        return roster

    def _get(self, url: str, *, params: dict[str, str]) -> dict[str, Any]:
        try:
            r = requests.get(
                url, params=params, headers=self._headers, timeout=self._timeout,
            )
            r.raise_for_status()
            return r.json()
        except requests.RequestException as exc:
            raise WikidataLookupError(str(exc)) from exc


def fetch_kits_by_team_names(
    home_team: str,
    away_team: str,
    *,
    source: WikidataMatchSource | None = None,
) -> KitColors | None:
    """Resolve two team names to Wikidata Q-IDs and pull their kit
    colours. Returns ``None`` if either team can't be resolved or
    Wikidata has no colour data for them.

    Surfaces ``WikidataLookupError`` only for transport-level failures;
    a successful response with no colour bindings just returns ``None``
    so the caller can decide whether to keep the candidate with blank
    kits.

    Used by the API's ``/api/match/lookup`` endpoint to hydrate kit
    colours regardless of which provider returned the candidate list
    — Wikidata is the only public source that carries kits as
    structured hex values.
    """
    src = source or WikidataMatchSource()
    home_qid = src._resolve_team(home_team)
    away_qid = src._resolve_team(away_team)
    if not home_qid or not away_qid:
        return None
    return src._hydrate_kits(home_qid=home_qid, away_qid=away_qid)


# Register the default Wikidata provider so the API can resolve "wikidata"
# without explicit setup at startup.
register_source(WikidataMatchSource())
