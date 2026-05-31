"""Pluggable external sources for match metadata lookups.

A ``MatchInfoSource`` resolves a (season, home_team, away_team) triple
into zero or more ``MatchCandidate``s. The dashboard's *Lookup* button
calls into the registered source(s); the user picks one candidate and
the form pre-fills with whatever fields the provider could hydrate.

Best-effort by design: providers are free to return ``MatchInfo`` with
many fields blank when the underlying data isn't available, and the UI
surfaces ``filled_fields`` so the user knows what still needs typing.

Season strings are accepted as either ``"YYYY-YY"`` (European-style
spanning two calendar years, e.g. ``"2021-22"``) or a single year
``"YYYY"`` (single-calendar-year leagues, e.g. MLS). Providers convert
this to whatever date window the underlying data store needs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from src.schemas.shots import MatchInfo


@dataclass
class MatchCandidate:
    """One hit from a lookup provider, packaging a ``MatchInfo`` with
    the provider's identifier and a record of which fields it filled."""

    match: MatchInfo
    provider: str
    provider_id: str
    confidence: float = 0.5
    filled_fields: list[str] = field(default_factory=list)


class MatchInfoSource(Protocol):
    """Provider interface. Implementations live alongside this module."""

    name: str

    def lookup(
        self, *, season: str, home_team: str, away_team: str,
    ) -> list[MatchCandidate]: ...


def season_to_date_window(season: str) -> tuple[str, str]:
    """Resolve a ``season`` string into an inclusive ISO date window.

    Accepts:

    * ``"YYYY-YY"`` — European-style two-year season; window covers
      ``YYYY-07-01`` to ``(YYYY+1)-06-30``.
    * ``"YYYY-YYYY"`` — same as above, full end year.
    * ``"YYYY"`` — single calendar year; window covers
      ``YYYY-01-01`` to ``YYYY-12-31``.

    Raises ``ValueError`` for anything else so providers can reject
    malformed input loudly rather than silently widening the search.
    """
    s = season.strip()
    if "-" in s:
        a, b = s.split("-", 1)
        if not a.isdigit() or not b.isdigit() or len(a) != 4:
            raise ValueError(f"invalid season: {season!r}")
        start = int(a)
        if len(b) == 2:
            end = (start // 100) * 100 + int(b)
            if end < start:
                end += 100
        elif len(b) == 4:
            end = int(b)
        else:
            raise ValueError(f"invalid season: {season!r}")
        if end != start + 1:
            raise ValueError(
                f"season {season!r} must span exactly one year boundary",
            )
        return f"{start:04d}-07-01", f"{end:04d}-06-30"
    if not s.isdigit() or len(s) != 4:
        raise ValueError(f"invalid season: {season!r}")
    y = int(s)
    return f"{y:04d}-01-01", f"{y:04d}-12-31"


_SOURCES: dict[str, MatchInfoSource] = {}


def register_source(source: MatchInfoSource) -> None:
    _SOURCES[source.name] = source


def get_source(name: str) -> MatchInfoSource | None:
    return _SOURCES.get(name)


def all_sources() -> dict[str, MatchInfoSource]:
    return dict(_SOURCES)
