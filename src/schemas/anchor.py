from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class LandmarkObservation:
    name: str
    image_xy: tuple[float, float]
    world_xyz: tuple[float, float, float]


@dataclass(frozen=True)
class LineObservation:
    """A line correspondence for camera calibration.

    The user draws a 2-point ``image_segment`` on a frame and selects a known
    pitch line from the catalogue. There are two flavours:

    1. **Position-known line** (``world_segment`` set) — e.g. ``near_touchline``
       at world ``((0,0,0),(105,0,0))``. The solver requires the user's two
       image points to lie on the *projection* of that specific world line.

    2. **Direction-only / vanishing-point line** (``world_direction`` set) —
       e.g. ``vertical_separator`` with direction ``(0,0,1)``. The world
       position is unknown (the user is marking a vertical LED-board seam at
       an arbitrary x position), but the line is parallel to a known world
       direction. The solver requires the user's image line to point at the
       vanishing point of that direction.

    Exactly one of ``world_segment`` or ``world_direction`` should be set.
    """

    name: str
    image_segment: tuple[tuple[float, float], tuple[float, float]]
    world_segment: tuple[tuple[float, float, float], tuple[float, float, float]] | None = None
    world_direction: tuple[float, float, float] | None = None


@dataclass(frozen=True)
class Anchor:
    frame: int
    landmarks: tuple[LandmarkObservation, ...]
    lines: tuple[LineObservation, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class AnchorSet:
    clip_id: str
    image_size: tuple[int, int]   # (width, height)
    anchors: tuple[Anchor, ...]

    @classmethod
    def load(cls, path: Path) -> "AnchorSet":
        with path.open() as fh:
            data = json.load(fh)
        anchors = tuple(
            Anchor(
                frame=int(a["frame"]),
                landmarks=tuple(
                    LandmarkObservation(
                        name=str(lm["name"]),
                        image_xy=tuple(lm["image_xy"]),
                        world_xyz=tuple(lm["world_xyz"]),
                    )
                    for lm in a.get("landmarks", [])
                ),
                lines=tuple(
                    LineObservation(
                        name=str(ln["name"]),
                        image_segment=(
                            tuple(ln["image_segment"][0]),
                            tuple(ln["image_segment"][1]),
                        ),
                        world_segment=(
                            (
                                tuple(ln["world_segment"][0]),
                                tuple(ln["world_segment"][1]),
                            )
                            if ln.get("world_segment") is not None
                            else None
                        ),
                        world_direction=(
                            tuple(ln["world_direction"])
                            if ln.get("world_direction") is not None
                            else None
                        ),
                    )
                    for ln in a.get("lines", [])
                ),
            )
            for a in data["anchors"]
        )
        return cls(
            clip_id=str(data["clip_id"]),
            image_size=tuple(data["image_size"]),
            anchors=anchors,
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as fh:
            json.dump(asdict(self), fh, indent=2, default=lambda v: list(v) if isinstance(v, tuple) else v)
