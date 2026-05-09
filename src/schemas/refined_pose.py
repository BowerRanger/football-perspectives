"""Per-player fused SMPL track and per-frame fusion diagnostics.

Output of the ``refined_poses`` stage. RefinedPose mirrors
SmplWorldTrack but is keyed by player_id only and indexed onto the
shared reference timeline (see src/schemas/sync_map.py for the
sign convention). Diagnostics record which shots contributed to each
fused frame and how much they disagreed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class RefinedPose:
    """Per-player SMPL track fused across all shots that saw the player.

    Pitch-world frame matches SmplWorldTrack: z-up, x along nearside
    touchline, y toward far side, root_t in pitch metres.
    """

    player_id: str
    frames: np.ndarray              # (N,) reference-timeline frame indices
    betas: np.ndarray               # (10,) shared across the whole track
    thetas: np.ndarray              # (N, 24, 3) axis-angle
    root_R: np.ndarray              # (N, 3, 3)
    root_t: np.ndarray              # (N, 3) pitch metres
    confidence: np.ndarray          # (N,) fused confidence (sum of contributing weights)
    view_count: np.ndarray          # (N,) int — how many shots contributed at this frame
    contributing_shots: tuple[str, ...]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            player_id=self.player_id,
            frames=self.frames,
            betas=self.betas,
            thetas=self.thetas,
            root_R=self.root_R,
            root_t=self.root_t,
            confidence=self.confidence,
            view_count=self.view_count,
            contributing_shots=np.array(list(self.contributing_shots)),
        )

    @classmethod
    def load(cls, path: Path) -> "RefinedPose":
        z = np.load(path, allow_pickle=False)
        return cls(
            player_id=str(z["player_id"]),
            frames=z["frames"],
            betas=z["betas"],
            thetas=z["thetas"],
            root_R=z["root_R"],
            root_t=z["root_t"],
            confidence=z["confidence"],
            view_count=z["view_count"],
            contributing_shots=tuple(str(s) for s in z["contributing_shots"]),
        )


@dataclass(frozen=True)
class FrameDiagnostic:
    """One reference frame's fusion bookkeeping."""

    frame: int
    contributing_shots: tuple[str, ...]
    dropped_shots: tuple[str, ...]
    pos_disagreement_m: float
    rot_disagreement_rad: float
    low_coverage: bool
    high_disagreement: bool


@dataclass(frozen=True)
class RefinedPoseDiagnostics:
    """Per-player diagnostics; companion to a RefinedPose NPZ."""

    player_id: str
    frames: tuple[FrameDiagnostic, ...]
    contributing_shots: tuple[str, ...]
    summary: dict

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "player_id": self.player_id,
            "contributing_shots": list(self.contributing_shots),
            "frames": [
                {
                    "frame": f.frame,
                    "contributing_shots": list(f.contributing_shots),
                    "dropped_shots": list(f.dropped_shots),
                    "pos_disagreement_m": f.pos_disagreement_m,
                    "rot_disagreement_rad": f.rot_disagreement_rad,
                    "low_coverage": f.low_coverage,
                    "high_disagreement": f.high_disagreement,
                }
                for f in self.frames
            ],
            "summary": self.summary,
        }
        path.write_text(json.dumps(payload, indent=2))

    @classmethod
    def load(cls, path: Path) -> "RefinedPoseDiagnostics":
        data = json.loads(path.read_text())
        return cls(
            player_id=data["player_id"],
            contributing_shots=tuple(data["contributing_shots"]),
            frames=tuple(
                FrameDiagnostic(
                    frame=int(f["frame"]),
                    contributing_shots=tuple(f["contributing_shots"]),
                    dropped_shots=tuple(f["dropped_shots"]),
                    pos_disagreement_m=float(f["pos_disagreement_m"]),
                    rot_disagreement_rad=float(f["rot_disagreement_rad"]),
                    low_coverage=bool(f["low_coverage"]),
                    high_disagreement=bool(f["high_disagreement"]),
                )
                for f in data["frames"]
            ),
            summary=data["summary"],
        )
