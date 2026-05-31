"""Canonical kit roles and the role -> colour resolution used by export.

A *kit role* is the team/position bucket that decides which kit colour a
player wears: home outfield, away outfield, each side's goalkeeper, or the
referee. ``unknown`` is the neutral fallback when a track was never
classified.

The role is derived from the tracking output (``Track.team`` +
``Track.class_name``), with an optional per-player override authored in
``output/players.json`` (see ``src.utils.player_names``). The colours come
from the match ``KitColors`` block (``src.schemas.shots``); this module
mirrors those five hex strings onto the role keys and supplies sensible
defaults so the UE side always has a full palette.
"""

from __future__ import annotations

# Ordered so the five user-facing roles come first; ``unknown`` is the
# neutral fallback and is intentionally last.
KIT_ROLES: tuple[str, ...] = (
    "home",
    "away",
    "home_gk",
    "away_gk",
    "referee",
    "unknown",
)

# Neutral palette used when the match carries no kit colours (or a slot is
# blank). Distinct hues so players stay distinguishable even with no match
# metadata set.
_DEFAULT_KITS: dict[str, str] = {
    "home": "#3b82f6",      # blue
    "away": "#ef4444",      # red
    "home_gk": "#22c55e",   # green
    "away_gk": "#f59e0b",   # amber
    "referee": "#111827",   # near-black
    "unknown": "#9ca3af",   # grey
}


def derive_kit_role(team: str | None, class_name: str | None) -> str:
    """Map a track's ``team`` + ``class_name`` to a canonical kit role.

    ``team`` follows the tracking convention ``"A" | "B" | "referee" |
    "unknown"``; ``class_name`` is ``"player" | "goalkeeper" | "referee"
    | "ball"``. The home/away primaries map from team A/B, and the
    goalkeeper class promotes a side's outfield role to its keeper role.
    """
    team_norm = (team or "").strip().lower()
    cls_norm = (class_name or "").strip().lower()

    if team_norm in ("referee", "ref") or cls_norm == "referee":
        return "referee"

    is_keeper = cls_norm == "goalkeeper"
    if team_norm in ("a", "home"):
        return "home_gk" if is_keeper else "home"
    if team_norm in ("b", "away"):
        return "away_gk" if is_keeper else "away"
    return "unknown"


def kit_role_from_override(
    team: str | None, is_gk: bool = False, is_ref: bool = False
) -> str:
    """Map an explicit in-engine override to a canonical kit role.

    Unlike :func:`derive_kit_role` (which reads the tracking ``team`` +
    ``class_name``), this takes the controls a user sets by hand: the
    side (``team`` ∈ ``home``/``away``, or the ``A``/``B`` aliases) plus a
    goalkeeper and a referee toggle. The referee toggle wins over
    everything; otherwise the goalkeeper toggle promotes a side's outfield
    role to its keeper role. Falls through to ``derive_kit_role`` so the
    home/away normalisation and ``unknown`` fallback stay in one place.
    """
    if is_ref:
        return "referee"
    class_name = "goalkeeper" if is_gk else "player"
    return derive_kit_role(team, class_name)


def normalise_role(role: str | None) -> str | None:
    """Return ``role`` if it is a recognised kit role, else ``None``.

    Accepts a few friendly aliases so a hand-authored ``players.json``
    can use ``"gk_home"``/``"home-gk"``/``"keeper_home"`` etc.
    """
    if not role:
        return None
    r = role.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "gk_home": "home_gk",
        "home_keeper": "home_gk",
        "keeper_home": "home_gk",
        "gk_away": "away_gk",
        "away_keeper": "away_gk",
        "keeper_away": "away_gk",
        "ref": "referee",
    }
    r = aliases.get(r, r)
    return r if r in KIT_ROLES else None


def kits_map(match: dict | None) -> dict[str, str]:
    """Return the full role -> hex-colour palette for a match.

    ``match`` is the ``asdict``-ed ``MatchInfo`` (or ``None``). Reads the
    nested ``kits`` block and falls back to :data:`_DEFAULT_KITS` for any
    role whose colour is missing or blank.
    """
    out = dict(_DEFAULT_KITS)
    kits = (match or {}).get("kits") or {}
    slot_to_role = {
        "home_primary": "home",
        "away_primary": "away",
        "home_goalkeeper": "home_gk",
        "away_goalkeeper": "away_gk",
        "referee": "referee",
    }
    for slot, role in slot_to_role.items():
        value = kits.get(slot)
        if isinstance(value, str) and value.strip():
            out[role] = value.strip()
    return out
