"""Per-shot virtual-camera selection, edited from the web Export panel.

Persisted at ``output/export/{shot_id}_camera_selection.json``. The export
stage reads it to decide which players get POV/OTS cameras.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

VALID_RIGS: tuple[str, ...] = ("pov", "ots")


class CameraSelectionError(ValueError):
    """Raised when a selection payload fails validation."""


@dataclass(frozen=True)
class RigSelection:
    player_id: str
    rigs: tuple[str, ...]


@dataclass(frozen=True)
class CameraSelection:
    shot_id: str
    selections: tuple[RigSelection, ...] = ()

    @classmethod
    def empty(cls, shot_id: str) -> "CameraSelection":
        return cls(shot_id=shot_id, selections=())

    @classmethod
    def from_dict(cls, data: dict) -> "CameraSelection":
        shot_id = str(data.get("shot_id", ""))
        if not shot_id:
            raise CameraSelectionError("shot_id must be non-empty")
        out: list[RigSelection] = []
        for entry in data.get("selections", []) or []:
            pid = str(entry.get("player_id", ""))
            if not pid:
                raise CameraSelectionError("each selection needs a player_id")
            raw_rigs = entry.get("rigs", []) or []
            for r in raw_rigs:
                if r not in VALID_RIGS:
                    raise CameraSelectionError(f"unknown rig {r!r}; valid: {VALID_RIGS}")
            ordered = tuple(r for r in VALID_RIGS if r in set(raw_rigs))
            if ordered:
                out.append(RigSelection(player_id=pid, rigs=ordered))
        return cls(shot_id=shot_id, selections=tuple(out))

    def to_dict(self) -> dict:
        return {
            "shot_id": self.shot_id,
            "selections": [
                {"player_id": s.player_id, "rigs": list(s.rigs)}
                for s in self.selections
            ],
        }

    @classmethod
    def load(cls, path: Path) -> "CameraSelection":
        return cls.from_dict(json.loads(Path(path).read_text()))

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=2))
        tmp.replace(path)  # atomic on POSIX
