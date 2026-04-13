from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class TriangulatedBall:
    """3D ball trajectory across frames.

    Per-frame state covers the ball's world position, velocity, the
    method used to estimate that frame ("multi" / "single_ground" /
    "flight" / "interp"), and a coarse confidence (0 = no data).
    """

    positions: np.ndarray   # (N, 3) float32 — pitch-metres, NaN where unknown
    confidences: np.ndarray  # (N,) float32
    methods: np.ndarray     # (N,) int8 — 0=none, 1=multi, 2=single_ground, 3=flight
    fps: float
    start_frame: int

    def save(self, path: Path) -> None:
        np.savez_compressed(
            path,
            positions=self.positions.astype(np.float32),
            confidences=self.confidences.astype(np.float32),
            methods=self.methods.astype(np.int8),
            fps=np.array(self.fps, dtype=np.float32),
            start_frame=np.array(self.start_frame, dtype=np.int32),
        )

    @classmethod
    def load(cls, path: Path) -> "TriangulatedBall":
        data = np.load(path, allow_pickle=False)
        return cls(
            positions=data["positions"],
            confidences=data["confidences"],
            methods=data["methods"],
            fps=float(data["fps"]),
            start_frame=int(data["start_frame"]),
        )


@dataclass
class TriangulatedPlayer:
    """3D joint positions for a single player across frames.

    Arrays are stored as compressed numpy archives (.npz).  ``player_name``
    and ``team`` are carried through from the tracking stage so downstream
    stages (SMPL fitting, export, viewer) can display human-readable
    identities.
    """

    player_id: str
    positions: np.ndarray  # (N, 17, 3) float32 — pitch-metres
    confidences: np.ndarray  # (N, 17) float32
    reprojection_errors: np.ndarray  # (N, 17) float32
    num_views: np.ndarray  # (N, 17) int8
    fps: float
    start_frame: int  # reference-shot frame index of first frame
    player_name: str = ""  # manually-edited display name; may be empty
    team: str = ""          # "A" | "B" | "referee" | "unknown" | ""

    def save(self, path: Path) -> None:
        np.savez_compressed(
            path,
            player_id=np.array(self.player_id),
            player_name=np.array(self.player_name),
            team=np.array(self.team),
            positions=self.positions.astype(np.float32),
            confidences=self.confidences.astype(np.float32),
            reprojection_errors=self.reprojection_errors.astype(np.float32),
            num_views=self.num_views.astype(np.int8),
            fps=np.array(self.fps, dtype=np.float32),
            start_frame=np.array(self.start_frame, dtype=np.int32),
        )

    @classmethod
    def load(cls, path: Path) -> "TriangulatedPlayer":
        data = np.load(path, allow_pickle=False)
        return cls(
            player_id=str(data["player_id"]),
            player_name=str(data["player_name"]) if "player_name" in data else "",
            team=str(data["team"]) if "team" in data else "",
            positions=data["positions"],
            confidences=data["confidences"],
            reprojection_errors=data["reprojection_errors"],
            num_views=data["num_views"],
            fps=float(data["fps"]),
            start_frame=int(data["start_frame"]),
        )
