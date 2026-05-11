"""Persisted ball anchor data — user-supplied per-frame ball positions
and state tags. Read by ``BallStage`` as a Layer 5 input before the
WASB detection loop runs.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal


BallAnchorState = Literal[
    "grounded",
    "airborne_low",
    "airborne_mid",
    "airborne_high",
    "kick",
    "catch",
    "bounce",
    "header",
    "off_screen_flight",
]

_VALID_STATES: frozenset[str] = frozenset({
    "grounded", "airborne_low", "airborne_mid", "airborne_high",
    "kick", "catch", "bounce", "header", "off_screen_flight",
})


@dataclass(frozen=True)
class BallAnchor:
    """One per-frame anchor."""
    frame: int
    # None only when state == "off_screen_flight".
    image_xy: tuple[float, float] | None
    state: BallAnchorState


@dataclass(frozen=True)
class BallAnchorSet:
    clip_id: str
    image_size: tuple[int, int]
    anchors: tuple[BallAnchor, ...]

    @classmethod
    def load(cls, path: Path) -> "BallAnchorSet":
        with path.open() as fh:
            data = json.load(fh)
        anchors = []
        for a in data.get("anchors", []):
            state = str(a["state"])
            if state not in _VALID_STATES:
                raise ValueError(f"unknown ball anchor state: {state!r}")
            raw_xy = a.get("image_xy")
            if raw_xy is None:
                if state != "off_screen_flight":
                    raise ValueError(
                        f"image_xy is required for state {state!r} "
                        f"(only off_screen_flight may omit it)"
                    )
                image_xy = None
            else:
                image_xy = (float(raw_xy[0]), float(raw_xy[1]))
            anchors.append(BallAnchor(
                frame=int(a["frame"]),
                image_xy=image_xy,
                state=state,  # type: ignore[arg-type]
            ))
        return cls(
            clip_id=str(data["clip_id"]),
            image_size=(int(data["image_size"][0]), int(data["image_size"][1])),
            anchors=tuple(anchors),
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as fh:
            json.dump(
                asdict(self), fh, indent=2,
                default=lambda v: list(v) if isinstance(v, tuple) else v,
            )
