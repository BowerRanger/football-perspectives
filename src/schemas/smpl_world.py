from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class SmplWorldTrack:
    """Per-player SMPL parameters expressed in pitch-world coordinates.

    Pitch-world frame: z-up, x along nearside touchline, y toward far side
    (FIFA-standard 105 m x 68 m). Root translation (root_t) is in pitch
    metres. Root rotation (root_R) is the SMPL canonical-to-pose rotation
    expressed in the pitch frame (see src/utils/smpl_pitch_transform.py).
    """

    player_id: str
    frames: np.ndarray          # (N,)   global frame indices
    betas: np.ndarray           # (10,)
    thetas: np.ndarray          # (N, 24, 3)  axis-angle
    root_R: np.ndarray          # (N, 3, 3)
    root_t: np.ndarray          # (N, 3)      pitch metres
    confidence: np.ndarray      # (N,)
    # Multi-shot routing: which shot the player was detected in.
    # Default empty string for backwards-compat with pre-multi-shot
    # NPZ files; downstream stages (export) treat "" as "include in
    # the legacy single-shot scene".
    shot_id: str = ""

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
            shot_id=self.shot_id,
        )

    @classmethod
    def load(cls, path: Path) -> "SmplWorldTrack":
        z = np.load(path, allow_pickle=False)
        return cls(
            player_id=str(z["player_id"]),
            frames=z["frames"],
            betas=z["betas"],
            thetas=z["thetas"],
            root_R=z["root_R"],
            root_t=z["root_t"],
            confidence=z["confidence"],
            shot_id=str(z["shot_id"]) if "shot_id" in z.files else "",
        )
