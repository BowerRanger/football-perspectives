"""Venue dressing resolver: which seats / crowd / boards does this shot get?

Pure (no bpy). ``config/stadiums.yaml`` carries a per-venue ``dressing``
block next to the pitch metadata plus a top-level ``dressing_default``.
``resolve_dressing`` picks the venue, merges ``dressing_default`` <- venue
dressing <- explicit clip ``render.style.stadium`` keys (clip keys win), and
derives the away-end crowd palette from the away kit when the venue does not
pin one. ``scripts/blender_stadium.py`` consumes the result.

Venue lookup order: clip ``render.venue`` -> ``<out>/camera/<shot>_anchors.json``
``stadium`` -> ``shots_manifest.json`` ``match.venue`` (alias match) -> default.

Dressing keys (all optional):
  seat_color, accent_color     seat shell / aisle-block colours
  stand_tone                   concrete tier mass (also seeds steel + tunnels)
  board_color, board_text, board_text_color   perimeter advertising ribbon
  crowd_colors                 shirt palette (repeat a colour to weight it)
  away_end: {stand, crowd_colors}   North|South|East|West; colours default to
                               the away kit
Dark structural tones are lifted to ``TONE_FLOOR_L`` (CIE L*) so the toon
ramp's 35% shadow band cannot turn the stand fronts into a black band behind
low cameras.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Mapping

logger = logging.getLogger(__name__)

_REGISTRY_PATH = Path(__file__).resolve().parents[2] / "config" / "stadiums.yaml"
STANDS = ("North", "South", "East", "West")
DEFAULT_AWAY_STAND = "East"
# Mirrors blender_stadium's historic palette; used only if the YAML is absent.
FALLBACK_CROWD_COLORS = ("#283e50", "#b1b8af", "#a64039", "#ceac78", "#476780", "#d2c9b5")
_FALLBACK_DRESSING = {
    "seat_color": "#294b65", "accent_color": "#d6b66e", "stand_tone": "#707e87",
    "board_color": "#2b4a58", "board_text": "FOOTBALL / PERSPECTIVES",
    "crowd_colors": list(FALLBACK_CROWD_COLORS),
}
TONE_FLOOR_L = 38.0   # min CIE L* for stand_tone / board_color before the 0.35 shadow band


# --- colour helpers ---------------------------------------------------------

def _hex_rgb(h: str) -> tuple[float, float, float]:
    s = h.lstrip("#")
    return tuple(int(s[i:i + 2], 16) / 255.0 for i in (0, 2, 4))  # type: ignore[return-value]


def _to_hex(rgb) -> str:
    return "#" + "".join(f"{max(0, min(255, round(c * 255))):02x}" for c in rgb)


def _lin(c: float) -> float:
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def relative_lightness(h: str) -> float:
    """CIE L* of an sRGB hex (0 black .. 100 white)."""
    r, g, b = (_lin(c) for c in _hex_rgb(h))
    y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b
    f = y ** (1 / 3) if y > 216 / 24389 else (24389 / 27 * y + 16) / 116
    return 116 * f - 16


def tone_floor(h: str, min_l: float = TONE_FLOOR_L) -> str:
    """Lift ``h`` toward white just enough that its L* >= ``min_l``.

    Colours already at/above the floor are returned unchanged.
    """
    if relative_lightness(h) >= min_l:
        return h
    rgb = _hex_rgb(h)
    lo, hi = 0.0, 1.0
    for _ in range(24):
        mid = (lo + hi) / 2
        cand = tuple(c + (1.0 - c) * mid for c in rgb)
        if relative_lightness(_to_hex(cand)) >= min_l:
            hi = mid
        else:
            lo = mid
    return _to_hex(tuple(c + (1.0 - c) * hi for c in rgb))


# --- registry ---------------------------------------------------------------

def _load_registry(path: Path | None = None) -> dict:
    import yaml  # lazy: Blender's bundled Python has no PyYAML (blender_stadium imports this module)

    target = path or _REGISTRY_PATH
    if not target.exists():
        return {}
    return yaml.safe_load(target.read_text()) or {}


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def _match_venue_id(venue: str, stadiums: Mapping[str, Mapping]) -> str | None:
    """Map a free-text venue ("Anfield, Liverpool") onto a registry id."""
    nv = _norm(venue)
    if not nv:
        return None
    for sid, body in stadiums.items():
        names = [sid, *(body.get("aliases") or [])]
        if any(_norm(n) and (_norm(n) in nv or nv in _norm(n)) for n in names):
            return sid
    return None


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def resolve_venue_id(cfg: Mapping, output_dir: Path, shot: str,
                     stadiums: Mapping[str, Mapping]) -> str | None:
    explicit = (cfg.get("render") or {}).get("venue")
    if explicit:
        if explicit in stadiums:
            return explicit
        logger.warning("[stadium_dressing] render.venue %r is not in stadiums.yaml; "
                       "falling back to anchors/match venue", explicit)
    out = Path(output_dir)
    anchored = _read_json(out / "camera" / f"{shot}_anchors.json").get("stadium")
    if anchored and anchored in stadiums:
        return anchored
    match = (_read_json(out / "shots" / "shots_manifest.json").get("match") or {})
    return _match_venue_id(match.get("venue") or "", stadiums)


# --- resolution -------------------------------------------------------------

def _away_kit(cfg: Mapping, output_dir: Path) -> Mapping | None:
    try:
        from src.utils.kit_resolution import effective_team_kits
        kits = effective_team_kits(Path(output_dir), cfg)
    except Exception:  # noqa: BLE001 - kit lookup is a nicety, never fatal
        kits = ((cfg.get("render") or {}).get("teams") or {}).get("defaults") or {}
    return kits.get("away")


def _crowd_from_kit(kit: Mapping) -> list[str]:
    shirt = kit.get("shirt")
    if not shirt:
        return []
    second = kit.get("shorts") if kit.get("shorts") and kit.get("shorts") != shirt else "#e9e5dc"
    return [shirt, shirt, shirt, second, shirt, "#2a2a2a"]


def resolve_dressing(cfg: Mapping, output_dir: Path, shot: str,
                     registry_path: Path | None = None) -> dict[str, Any]:
    """Final ``render.style.stadium`` dict for ``shot`` (explicit clip keys win)."""
    raw = _load_registry(registry_path)
    stadiums = raw.get("stadiums") or {}
    base = {**_FALLBACK_DRESSING, **(raw.get("dressing_default") or {})}
    venue = resolve_venue_id(cfg, Path(output_dir), shot, stadiums)
    venue_dressing = dict((stadiums.get(venue) or {}).get("dressing") or {}) if venue else {}
    merged: dict[str, Any] = {**base, **venue_dressing}
    explicit = dict(((cfg.get("render") or {}).get("style") or {}).get("stadium") or {})
    away = {**dict(merged.get("away_end") or {}), **dict(explicit.get("away_end") or {})}
    merged.update({k: v for k, v in explicit.items() if k != "away_end"})
    if away or venue_dressing:
        away.setdefault("stand", DEFAULT_AWAY_STAND)
        if not away.get("crowd_colors"):
            kit = _away_kit(cfg, Path(output_dir))
            colors = _crowd_from_kit(kit) if kit else []
            if colors:
                away["crowd_colors"] = colors
        if away.get("crowd_colors"):
            merged["away_end"] = away
        else:
            merged.pop("away_end", None)
    merged["venue"] = venue
    merged["dressing_source"] = "venue" if venue_dressing else "default"
    return merged


def _shade(h: str, factor: float) -> str:
    return _to_hex(tuple(c * factor for c in _hex_rgb(h)))


def structural_tones(style_stadium: Mapping) -> dict[str, str]:
    """Concrete / steel / tunnel / board colours with the black-band floor applied.

    ``stand_tone`` seeds the concrete mass; steel and tunnel openings are
    darker derivatives, but all are lifted to a minimum lightness so the toon
    ramp's shadow band (35%) never collapses a low camera's horizon to black.
    """
    stand = style_stadium.get("stand_tone") or _FALLBACK_DRESSING["stand_tone"]
    board = style_stadium.get("board_color") or _FALLBACK_DRESSING["board_color"]
    return {
        "concrete": tone_floor(stand, TONE_FLOOR_L),
        "steel": tone_floor(_shade(stand, 0.55), TONE_FLOOR_L - 6.0),
        "tunnels": tone_floor(_shade(stand, 0.30), TONE_FLOOR_L - 14.0),
        "board": tone_floor(board, TONE_FLOOR_L),
        "board_text": style_stadium.get("board_text_color") or "#e1e9e6",
    }


def crowd_palette_for_stand(style_stadium: Mapping, stand: str) -> list[str]:
    """Crowd shirt palette for one stand (the away end gets the away palette)."""
    away = style_stadium.get("away_end") or {}
    if away.get("stand") == stand and away.get("crowd_colors"):
        return list(away["crowd_colors"])
    return list(style_stadium.get("crowd_colors") or FALLBACK_CROWD_COLORS)
