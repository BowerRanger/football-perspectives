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
    pitch line from the catalogue (e.g. ``near_touchline``). ``world_segment``
    captures both endpoints of that catalogue line in pitch coordinates.

    The solver does NOT require the user's image endpoints to align with the
    world segment's endpoints — only that the two image points lie on the
    *projection* of the world line through (K, R, t).
    """

    name: str
    image_segment: tuple[tuple[float, float], tuple[float, float]]
    world_segment: tuple[tuple[float, float, float], tuple[float, float, float]]


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
                            tuple(ln["world_segment"][0]),
                            tuple(ln["world_segment"][1]),
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
