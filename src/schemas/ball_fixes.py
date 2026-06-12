"""Persisted cross-replay triangulated ball fixes.

Written by the ball stage after the triangulation pass; consumed by the
solve pass to soft-constrain LM flight fits against monocular depth
ambiguity.  One sidecar per shot, keyed by that shot's own frame index.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BallFix:
    """One triangulated 3D fix for a single shot frame."""

    frame: int
    xyz: tuple[float, float, float]
    ray_miss_m: float
    parallax_deg: float
    partner_shot: str
    partner_frame: int


@dataclass(frozen=True)
class BallFixSet:
    """Collection of cross-replay fixes for one shot."""

    clip_id: str
    group_id: str
    cross_replay: dict[str, Any]
    fixes: tuple[BallFix, ...]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "clip_id": self.clip_id,
            "group_id": self.group_id,
            "cross_replay": self.cross_replay,
            "fixes": [asdict(fx) for fx in self.fixes],
        }
        with path.open("w") as fh:
            json.dump(data, fh, indent=2)

    @classmethod
    def load(cls, path: Path) -> "BallFixSet":
        with path.open() as fh:
            data = json.load(fh)
        fixes = tuple(
            BallFix(
                frame=int(fx["frame"]),
                xyz=(float(fx["xyz"][0]), float(fx["xyz"][1]), float(fx["xyz"][2])),
                ray_miss_m=float(fx["ray_miss_m"]),
                parallax_deg=float(fx["parallax_deg"]),
                partner_shot=str(fx["partner_shot"]),
                partner_frame=int(fx["partner_frame"]),
            )
            for fx in data.get("fixes", [])
        )
        return cls(
            clip_id=str(data["clip_id"]),
            group_id=str(data["group_id"]),
            cross_replay=dict(data.get("cross_replay", {})),
            fixes=fixes,
        )
