"""Ball-track schema: per-frame ball world position + flight segments.

Output of `BallStage`.  Every frame in the camera span is represented;
frames without a detection are emitted as ``state="missing"``.  Frames
classified as part of a parabolic flight are tagged with a non-null
``flight_segment_id`` and their ``world_xyz`` is the parabola evaluation
(rather than the ground projection).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

State = Literal["grounded", "flight", "occluded", "missing"]


@dataclass(frozen=True)
class BallFrame:
    frame: int
    world_xyz: tuple[float, float, float] | None
    state: State
    confidence: float
    flight_segment_id: int | None = None


@dataclass(frozen=True)
class FlightSegment:
    id: int
    frame_range: tuple[int, int]
    parabola: dict
    fit_residual_px: float


@dataclass(frozen=True)
class BallTrack:
    clip_id: str
    fps: float
    frames: tuple[BallFrame, ...]
    flight_segments: tuple[FlightSegment, ...]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as fh:
            json.dump(
                asdict(self),
                fh,
                indent=2,
                default=lambda v: list(v) if isinstance(v, tuple) else v,
            )

    @classmethod
    def load(cls, path: Path) -> "BallTrack":
        with path.open() as fh:
            data = json.load(fh)
        frames = tuple(
            BallFrame(
                frame=int(f["frame"]),
                world_xyz=(
                    tuple(f["world_xyz"]) if f["world_xyz"] is not None else None
                ),
                state=f["state"],
                confidence=float(f["confidence"]),
                flight_segment_id=f.get("flight_segment_id"),
            )
            for f in data["frames"]
        )
        segs = tuple(
            FlightSegment(
                id=int(s["id"]),
                frame_range=tuple(s["frame_range"]),
                parabola=s["parabola"],
                fit_residual_px=float(s["fit_residual_px"]),
            )
            for s in data["flight_segments"]
        )
        return cls(
            clip_id=data["clip_id"],
            fps=data["fps"],
            frames=frames,
            flight_segments=segs,
        )
