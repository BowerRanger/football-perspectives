"""Kit resolution — which team kits does a shot actually wear?

Precedence (highest first)::

    <out>/appearance/kits_operator.json
    > clip config ``appearance.kits`` (library refs or inline KitSpecs)
    > auto  <out>/appearance/kits.json
    > ``render.teams.defaults``
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Mapping

logger = logging.getLogger(__name__)

APPEARANCE_DIR = "appearance"


def _load_json_kits(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        logger.warning("[kit_resolution] %s is not valid JSON: %s", path, exc)
        return {}
    kits = raw.get("kits", raw) if isinstance(raw, Mapping) else {}
    return {k: dict(v) for k, v in kits.items() if isinstance(v, Mapping)}


def effective_team_kits(output_dir: Path, cfg: Mapping[str, Any] | None) -> dict[str, dict]:
    """``role -> KitSpec dict`` merged per ROLE by the precedence above."""
    cfg = cfg or {}
    out_dir = Path(output_dir)
    merged: dict[str, dict] = {}
    layers = [
        dict(((cfg.get("render") or {}).get("teams") or {}).get("defaults") or {}),
        _load_json_kits(out_dir / APPEARANCE_DIR / "kits.json"),
        dict((cfg.get("appearance") or {}).get("kits") or {}),
        _load_json_kits(out_dir / APPEARANCE_DIR / "kits_operator.json"),
    ]
    for layer in layers:
        for role, kit in layer.items():
            if isinstance(kit, Mapping):
                merged[role] = dict(kit)
    return merged
