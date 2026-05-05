from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class CameraFrame:
    frame: int
    K: list[list[float]]      # 3x3
    R: list[list[float]]      # 3x3
    confidence: float
    is_anchor: bool
    # Per-frame world->camera translation. Optional for backward compatibility
    # with older camera_track.json files that stored only a clip-shared
    # ``t_world``; viewers/downstream should prefer per-frame ``t`` when
    # present and fall back to ``CameraTrack.t_world`` when absent.
    t: list[float] | None = None


@dataclass(frozen=True)
class CameraTrack:
    clip_id: str
    fps: float
    image_size: tuple[int, int]
    t_world: list[float]                   # length 3 — representative/median across anchors
    frames: tuple[CameraFrame, ...]
    # Shared principal point recovered by the bundle adjustment.
    principal_point: tuple[float, float] | None = None

    @classmethod
    def load(cls, path: Path) -> "CameraTrack":
        with path.open() as fh:
            data = json.load(fh)
        frames = tuple(
            CameraFrame(
                frame=int(f["frame"]),
                K=[list(r) for r in f["K"]],
                R=[list(r) for r in f["R"]],
                confidence=float(f["confidence"]),
                is_anchor=bool(f["is_anchor"]),
                t=list(f["t"]) if f.get("t") is not None else None,
            )
            for f in data["frames"]
        )
        pp_raw = data.get("principal_point")
        principal_point = tuple(pp_raw) if pp_raw is not None else None
        return cls(
            clip_id=str(data["clip_id"]),
            fps=float(data["fps"]),
            image_size=tuple(data["image_size"]),
            t_world=list(data["t_world"]),
            frames=frames,
            principal_point=principal_point,
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as fh:
            json.dump(asdict(self), fh, indent=2)
