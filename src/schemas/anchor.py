from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class LandmarkObservation:
    name: str
    image_xy: tuple[float, float]
    world_xyz: tuple[float, float, float]


@dataclass(frozen=True)
class Anchor:
    frame: int
    landmarks: tuple[LandmarkObservation, ...]


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
                    for lm in a["landmarks"]
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
