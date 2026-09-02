"""Foot-contact sidecar schema (``{shot}__{pid}_foot_contacts.json``).

Written by the ``hmr_world`` stage (and ``scripts/reanchor_hmr_world.py``)
next to each ``*_smpl_world.npz``; consumed by ``refined_poses`` so the
contact-aware ground snap and foot-lock finale reuse the extraction-time
stance spans instead of re-deriving them. Frame indices inside the
payload are hmr_world track-ARRAY positions (0..n_frames-1), not global
frame numbers — consumers re-base with ``FootContacts.shifted``.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.utils.foot_contact import FootContacts

SCHEMA = "foot_contacts"
VERSION = 1


def save_foot_contacts(
    path: Path | str,
    contacts: FootContacts,
    *,
    shot_id: str,
    player_id: str,
    anchor_mode: str,
) -> None:
    payload = {
        "schema": SCHEMA,
        "version": VERSION,
        "shot_id": str(shot_id),
        "player_id": str(player_id),
        "anchor_mode": str(anchor_mode),
        "contacts": contacts.to_json(),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def load_foot_contacts(path: Path | str) -> tuple[FootContacts, dict]:
    """Return ``(contacts, meta)`` where meta carries shot_id/player_id/
    anchor_mode/version. Raises ``ValueError`` on a non-sidecar file so
    callers can distinguish "absent" (they check existence first) from
    "corrupt/foreign" (they should warn, not silently fall back)."""
    d = json.loads(Path(path).read_text())
    if d.get("schema") != SCHEMA:
        raise ValueError(f"not a foot_contacts sidecar: {path}")
    meta = {
        "shot_id": str(d.get("shot_id", "")),
        "player_id": str(d.get("player_id", "")),
        "anchor_mode": str(d.get("anchor_mode", "")),
        "version": int(d.get("version", 1)),
    }
    return FootContacts.from_json(d["contacts"]), meta
