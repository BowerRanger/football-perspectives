from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class SmplResult:
    """SMPL body model parameters for a single player across frames.

    ``player_name`` and ``team`` are carried through from the triangulation
    stage so the export/viewer can display human-readable identities.
    """

    player_id: str
    betas: np.ndarray  # (10,) float32 — body shape (shared across frames)
    poses: np.ndarray  # (N, 72) float32 — axis-angle rotations for 24 joints
    transl: np.ndarray  # (N, 3) float32 — global translation per frame
    fps: float
    player_name: str = ""
    team: str = ""

    def save(self, path: Path) -> None:
        np.savez_compressed(
            path,
            player_id=np.array(self.player_id),
            player_name=np.array(self.player_name),
            team=np.array(self.team),
            betas=self.betas.astype(np.float32),
            poses=self.poses.astype(np.float32),
            transl=self.transl.astype(np.float32),
            fps=np.array(self.fps, dtype=np.float32),
        )

    @classmethod
    def load(cls, path: Path) -> "SmplResult":
        data = np.load(path, allow_pickle=False)
        return cls(
            player_id=str(data["player_id"]),
            player_name=str(data["player_name"]) if "player_name" in data else "",
            team=str(data["team"]) if "team" in data else "",
            betas=data["betas"],
            poses=data["poses"],
            transl=data["transl"],
            fps=float(data["fps"]),
        )
