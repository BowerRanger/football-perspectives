"""Sidecars written by the ``appearance`` stage under ``<out>/appearance/``.

* ``kits.json``              — suggested team kits (``{"kits": {role: KitSpec}}``)
                               plus provenance (teams, white balance, notes).
* ``players_suggested.json`` — ``{pid: {"kit_role": role, ...}}``, same keys as
                               the operator ``players.json``; merged UNDER it by
                               ``player_names.load_kit_roles``.
* ``kits_operator.json``     — operator kit overrides (never written by the
                               stage; read by ``kit_resolution``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from src.schemas.kit import KitSpecError, normalise_kit_spec

SCHEMA = "appearance_kits"
VERSION = 1
APPEARANCE_DIR = "appearance"
KITS_FILE = "kits.json"
PLAYERS_SUGGESTED_FILE = "players_suggested.json"
KITS_OPERATOR_FILE = "kits_operator.json"


def appearance_dir(output_dir: Path | str) -> Path:
    return Path(output_dir) / APPEARANCE_DIR


def build_kits_payload(
    kits: Mapping[str, Mapping],
    *,
    teams: list[dict],
    clustering: Mapping[str, Any],
    white_balance: Mapping[str, Any],
    needs_confirmation: list[str],
    shots: list[str],
) -> dict:
    """Validate every kit and assemble the ``kits.json`` payload."""
    clean: dict[str, dict] = {}
    for role, kit in kits.items():
        try:
            clean[role] = normalise_kit_spec(kit)
        except KitSpecError as exc:
            raise KitSpecError(f"{role}: {exc}") from exc
    return {
        "schema": SCHEMA,
        "version": VERSION,
        "kits": clean,
        "teams": teams,
        "clustering": dict(clustering),
        "white_balance": dict(white_balance),
        "needs_confirmation": sorted(set(needs_confirmation)),
        "shots": shots,
    }


def build_players_suggested(roles: Mapping[str, str]) -> dict[str, dict]:
    return {pid: {"kit_role": role, "source": "appearance"} for pid, role in sorted(roles.items())}


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default) + "\n")


def _json_default(obj: Any):
    import numpy as np

    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"not JSON serialisable: {type(obj).__name__}")


def load_kits(output_dir: Path | str) -> dict | None:
    path = appearance_dir(output_dir) / KITS_FILE
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None
