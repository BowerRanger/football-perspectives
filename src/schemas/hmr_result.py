"""Per-view monocular HMR output schema.

Stores GVHMR results in the native 'ay' (align-Y) coordinate frame
where Y=gravity-down and X/Z form the horizontal ground plane.
Coordinate transform to pitch space is deferred to the global
optimisation stage.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


# SMPL 22-joint skeleton used by GVHMR (SMPLX body-only, no hands/face).
SMPL22_JOINT_NAMES: list[str] = [
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
]

# Parent indices for the 22-joint skeleton (root has parent -1).
SMPL22_PARENTS: list[int] = [
    -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19,
]

# Bone pairs for skeleton rendering (child, parent).
SMPL22_BONES: list[tuple[int, int]] = [
    (i, SMPL22_PARENTS[i]) for i in range(1, 22)
]


@dataclass
class HmrPlayerTrack:
    """GVHMR results for a single tracked player across frames.

    All arrays have leading dimension N (number of frames).
    SMPL params are in GVHMR's 'ay' coordinate frame (Y=gravity-down).
    """

    track_id: str
    player_id: str
    player_name: str
    team: str
    frame_indices: np.ndarray       # (N,) int32 — original frame numbers
    global_orient: np.ndarray       # (N, 3) float32 — axis-angle root in ay coords
    body_pose: np.ndarray           # (N, 63) float32 — 21 joints * 3 axis-angle
    betas: np.ndarray               # (10,) float32 — shape (shared across frames)
    transl: np.ndarray              # (N, 3) float32 — root translation in ay coords
    joints_3d: np.ndarray           # (N, 22, 3) float32 — joint positions in ay coords
    pred_cam: np.ndarray            # (N, 3) float32 — weak-perspective [s, tx, ty]
    bbx_xys: np.ndarray             # (N, 3) float32 — bbox [center_x, center_y, size]
    confidences: np.ndarray         # (N,) float32 — per-frame detection confidence
    kp2d: np.ndarray                # (N, 17, 3) float32 — ViTPose COCO-17 keypoints in image-pixel coords [x, y, conf]


@dataclass
class HmrResult:
    """Monocular HMR output for all tracked players in a single shot."""

    shot_id: str
    fps: float
    players: list[HmrPlayerTrack]

    def save(self, directory: Path) -> None:
        """Save as one .npz per player inside *directory*."""
        directory.mkdir(parents=True, exist_ok=True)
        for player in self.players:
            path = directory / f"{self.shot_id}_{player.track_id}_hmr.npz"
            np.savez_compressed(
                path,
                shot_id=np.array(self.shot_id),
                fps=np.array(self.fps, dtype=np.float32),
                track_id=np.array(player.track_id),
                player_id=np.array(player.player_id),
                player_name=np.array(player.player_name),
                team=np.array(player.team),
                frame_indices=player.frame_indices.astype(np.int32),
                global_orient=player.global_orient.astype(np.float32),
                body_pose=player.body_pose.astype(np.float32),
                betas=player.betas.astype(np.float32),
                transl=player.transl.astype(np.float32),
                joints_3d=player.joints_3d.astype(np.float32),
                pred_cam=player.pred_cam.astype(np.float32),
                bbx_xys=player.bbx_xys.astype(np.float32),
                confidences=player.confidences.astype(np.float32),
                kp2d=player.kp2d.astype(np.float32),
            )

    @classmethod
    def load(cls, directory: Path, shot_id: str) -> HmrResult:
        """Load all player .npz files for a given shot from *directory*."""
        players: list[HmrPlayerTrack] = []
        fps = 30.0
        for path in sorted(directory.glob(f"{shot_id}_*_hmr.npz")):
            data = np.load(path, allow_pickle=False)
            fps = float(data["fps"])
            n = data["frame_indices"].shape[0]
            # ``kp2d`` was added later — fall back to zeros for older
            # .npz files saved before the upgrade.
            kp2d = (
                data["kp2d"] if "kp2d" in data.files
                else np.zeros((n, 17, 3), dtype=np.float32)
            )
            players.append(
                HmrPlayerTrack(
                    track_id=str(data["track_id"]),
                    player_id=str(data["player_id"]),
                    player_name=str(data["player_name"]),
                    team=str(data["team"]),
                    frame_indices=data["frame_indices"],
                    global_orient=data["global_orient"],
                    body_pose=data["body_pose"],
                    betas=data["betas"],
                    transl=data["transl"],
                    joints_3d=data["joints_3d"],
                    pred_cam=data["pred_cam"],
                    bbx_xys=data["bbx_xys"],
                    confidences=data["confidences"],
                    kp2d=kp2d,
                )
            )
        return cls(shot_id=shot_id, fps=fps, players=players)

    @classmethod
    def load_all(cls, directory: Path) -> list[HmrResult]:
        """Load HMR results for all shots found in *directory*."""
        shot_ids: set[str] = set()
        for path in directory.glob("*_hmr.npz"):
            # filename pattern: {shot_id}_{track_id}_hmr.npz
            parts = path.stem.removesuffix("_hmr").rsplit("_", 1)
            if len(parts) == 2:
                shot_ids.add(parts[0])
        return [cls.load(directory, sid) for sid in sorted(shot_ids)]
