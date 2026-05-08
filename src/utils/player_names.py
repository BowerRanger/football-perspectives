"""Player ID → display-name mapping.

User maintains a hand-authored ``output/players.json`` that maps the
pipeline's numeric player IDs (e.g. ``P001``) to readable names
(e.g. ``Bellingham``). The export stage consults this mapping when
naming FBX files, armatures, and manifest entries.

Accepted file formats::

    {"P001": "Bellingham", "P002": "Saka"}

    or, with optional metadata for forward compatibility::

    {"P001": {"name": "Bellingham", "team": "England", "number": 22}}

The helper normalises both shapes and returns just the name. Unmapped
IDs fall back to the pipeline ID itself.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Mapping

logger = logging.getLogger(__name__)


def load_player_names(output_dir: Path) -> dict[str, str]:
    """Load the ``output/players.json`` mapping, returning an empty dict
    when the file is absent or malformed."""
    path = output_dir / "players.json"
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        logger.warning("[player_names] %s is not valid JSON: %s", path, exc)
        return {}
    if not isinstance(raw, Mapping):
        logger.warning("[player_names] %s must be an object at the root", path)
        return {}
    out: dict[str, str] = {}
    for k, v in raw.items():
        if isinstance(v, str):
            out[k] = v
        elif isinstance(v, Mapping) and isinstance(v.get("name"), str):
            out[k] = v["name"]
    return out


_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9_]+")


def safe_asset_name(name: str) -> str:
    """Sanitise a free-form name for use as a UE asset / filename.

    Replaces runs of non-alphanumeric characters with ``_`` and strips
    leading digits (UE asset names cannot start with a number).
    """
    cleaned = _SAFE_NAME_RE.sub("_", name).strip("_")
    if not cleaned:
        return "player"
    if cleaned[0].isdigit():
        cleaned = f"P_{cleaned}"
    return cleaned


def display_name_for(player_id: str, mapping: Mapping[str, str]) -> str:
    """Return a UE-safe display name for ``player_id``.

    Uses the mapping when present; otherwise returns the pipeline ID.
    """
    mapped = mapping.get(player_id)
    return safe_asset_name(mapped) if mapped else player_id
