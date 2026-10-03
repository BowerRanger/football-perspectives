"""KitSpec — the one kit-description shape shared by the kit library,
clip configs (``appearance.kits`` / ``render.teams.defaults``), the
appearance sidecars and the renderer.

A KitSpec is a plain dict (YAML/JSON friendly)::

    {shirt: "#c8102e", sleeve_color?: hex, collar?: hex, shorts: hex,
     socks: hex, boots?: hex, gloves?: hex, sleeves: "short"|"long",
     pattern?: {type: "solid"|"vertical_stripes"|"hoops",
                colors: [hex, hex], width_m: 0.07}}

``sleeves`` keeps its legacy meaning (LENGTH). ``pattern`` paints the
torso (and the sleeves unless ``sleeve_color`` is set).
"""

from __future__ import annotations

import re
from typing import Any, Mapping

KIT_ROLES = ("home", "away", "home_gk", "away_gk", "referee")
SLEEVE_LENGTHS = ("short", "long")
PATTERN_TYPES = ("solid", "vertical_stripes", "hoops")
DEFAULT_STRIPE_WIDTH_M = 0.07

_HEX_RE = re.compile(r"^#[0-9A-Fa-f]{6}$")
_REQUIRED_HEX = ("shirt", "shorts", "socks")
_OPTIONAL_HEX = ("sleeve_color", "collar", "boots", "gloves")
# Provenance / bookkeeping keys that may ride along on a spec.
_PASSTHROUGH = ("source", "name", "ref", "delta_e", "pool")


class KitSpecError(ValueError):
    """A kit dict failed validation."""


def _hex(value: Any, key: str) -> str:
    if not isinstance(value, str) or not _HEX_RE.match(value):
        raise KitSpecError(f"kit.{key}: expected '#RRGGBB', got {value!r}")
    return value.lower()


def normalise_pattern(raw: Any) -> dict | None:
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise KitSpecError(f"kit.pattern: expected a mapping, got {raw!r}")
    ptype = raw.get("type", "solid")
    if ptype not in PATTERN_TYPES:
        raise KitSpecError(f"kit.pattern.type: {ptype!r} not in {PATTERN_TYPES}")
    colors = raw.get("colors")
    width = float(raw.get("width_m", DEFAULT_STRIPE_WIDTH_M))
    if ptype == "solid":
        return {"type": "solid",
                "colors": [_hex(c, "pattern.colors") for c in (colors or [])][:2],
                "width_m": width}
    if not isinstance(colors, (list, tuple)) or len(colors) != 2:
        raise KitSpecError("kit.pattern.colors: need exactly two hex colours")
    if not 0.005 <= width <= 0.5:
        raise KitSpecError(f"kit.pattern.width_m: {width} outside [0.005, 0.5]")
    return {"type": ptype, "colors": [_hex(c, "pattern.colors") for c in colors],
            "width_m": width}


def normalise_kit_spec(raw: Mapping[str, Any]) -> dict:
    """Validate ``raw`` and return a canonical copy (lower-case hex,
    defaults filled: ``sleeves: short``). Raises :class:`KitSpecError`."""
    if not isinstance(raw, Mapping):
        raise KitSpecError(f"kit spec must be a mapping, got {type(raw).__name__}")
    out: dict[str, Any] = {}
    for key in _REQUIRED_HEX:
        if key not in raw:
            raise KitSpecError(f"kit.{key}: required")
        out[key] = _hex(raw[key], key)
    for key in _OPTIONAL_HEX:
        if raw.get(key) is not None:
            out[key] = _hex(raw[key], key)
    sleeves = raw.get("sleeves", "short")
    if sleeves not in SLEEVE_LENGTHS:
        raise KitSpecError(f"kit.sleeves: {sleeves!r} not in {SLEEVE_LENGTHS}")
    out["sleeves"] = sleeves
    pattern = normalise_pattern(raw.get("pattern"))
    if pattern is not None:
        out["pattern"] = pattern
    for key in _PASSTHROUGH:
        if key in raw:
            out[key] = raw[key]
    return out


def is_kit_spec(raw: Any) -> bool:
    try:
        normalise_kit_spec(raw)
    except KitSpecError:
        return False
    return True
