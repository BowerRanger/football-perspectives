"""Per-shot render camera selection, edited from the web Render panel.

Persisted at ``output/render/{shot_id or "clip"}_render_selection.json``
— mirrors ``RenderStage._active_shot_ids``' ``""`` legacy sentinel (no
shots manifest on disk) mapping to ``"clip"`` for the on-disk name, the
same convention ``RenderStage`` already uses for
``render/<shot|clip>/cameras/``. Unlike ``CameraSelection`` (which
requires a non-empty ``shot_id``), an empty ``shot_id`` here is a valid,
expected legacy value rather than a validation error.

``RenderStage.run`` loads this sidecar (when present) and merges it over
``render.cameras`` / ``render.vertical_variant`` from config — operator
input always wins, matching the anchor / sync-map / ball-anchor
conventions elsewhere in the pipeline.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

# broadcast | drone | pov:<player_id> | ots:<player_id>
_CAMERA_ID_RE = re.compile(r"^(broadcast|drone|(?:pov|ots):[A-Za-z0-9_-]+)$")


class RenderSelectionError(ValueError):
    """Raised when a selection payload fails validation."""


@dataclass(frozen=True)
class RenderSelection:
    shot_id: str
    cameras: tuple[str, ...] = ()
    # None means "don't override config" — distinct from False, which
    # explicitly forces the landscape-only render for this shot.
    vertical_variant: bool | None = None

    @classmethod
    def empty(cls, shot_id: str) -> "RenderSelection":
        return cls(shot_id=shot_id, cameras=(), vertical_variant=None)

    @classmethod
    def from_dict(cls, data: dict) -> "RenderSelection":
        shot_id = str(data.get("shot_id", ""))
        raw_cameras = data.get("cameras", []) or []
        if not isinstance(raw_cameras, list):
            raise RenderSelectionError("cameras must be a list")
        cameras: list[str] = []
        for cam in raw_cameras:
            if not isinstance(cam, str) or not _CAMERA_ID_RE.match(cam):
                raise RenderSelectionError(
                    f"invalid camera id {cam!r}; expected 'broadcast', "
                    f"'drone', 'pov:<player_id>' or 'ots:<player_id>'"
                )
            cameras.append(cam)
        vertical_variant = data.get("vertical_variant", None)
        if vertical_variant is not None and not isinstance(vertical_variant, bool):
            raise RenderSelectionError("vertical_variant must be a bool or null")
        return cls(
            shot_id=shot_id,
            cameras=tuple(cameras),
            vertical_variant=vertical_variant,
        )

    def to_dict(self) -> dict:
        return {
            "shot_id": self.shot_id,
            "cameras": list(self.cameras),
            "vertical_variant": self.vertical_variant,
        }

    @classmethod
    def load(cls, path: Path) -> "RenderSelection":
        return cls.from_dict(json.loads(Path(path).read_text()))

    def save(self, path: Path) -> None:
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=2))
        tmp.replace(dest)  # atomic on POSIX
