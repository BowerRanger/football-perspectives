"""Per-frame world + pixel positions of player contact joints.

Feeds the ball stage's automatic contact/event detection: for every
camera frame of a shot, the world position (SMPL forward kinematics)
and projected pixel of each contact-relevant bone of each player.

Source precedence per player per frame:
  1. ``refined_poses/{player}_refined.npz`` — cleaned, fused track on the
     reference timeline (translated to shot-local frames via the shot's
     sync offset: ``local = ref + offset``).
  2. ``hmr_world/{shot}__{player}_smpl_world.npz`` — raw per-shot track,
     shot-local frames. Used when no refined frame covers the query.

Bones use the anchor-editor vocabulary (``BONE_TO_SMPL_INDEX``) so the
samples can flow straight into ``player_touch`` auto-anchors that pass
``BallAnchorSet`` validation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.utils.ball_anchor_heights import BONE_TO_SMPL_INDEX
from src.utils.camera_projection import project_world_to_image
from src.utils.smpl_skeleton import compute_all_joint_worlds_batch

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class JointSample:
    """One bone of one player at one frame."""

    player_id: str
    bone: str
    world_xyz: tuple[float, float, float]
    # None when the joint is behind the camera (unprojectable).
    uv: tuple[float, float] | None
    confidence: float


def _sync_offset(output_dir: Path, shot_id: str) -> int:
    """Shot-local-to-reference offset (``local = ref + offset``); 0 when
    no sync map exists or the shot is unaligned."""
    from src.schemas.sync_map import SyncMap

    sync_path = output_dir / "shots" / "sync_map.json"
    if not shot_id or not sync_path.exists():
        return 0
    try:
        return SyncMap.load(sync_path).offset_for_shot(shot_id)
    except Exception as exc:
        logger.warning(
            "player context: sync_map.json failed to load (%s) — "
            "defaulting offset 0 for shot %r", exc, shot_id,
        )
        return 0


def _discover_refined(output_dir: Path) -> dict[str, Path]:
    refined_dir = output_dir / "refined_poses"
    out: dict[str, Path] = {}
    if refined_dir.exists():
        for path in sorted(refined_dir.glob("*_refined.npz")):
            out[path.name[: -len("_refined.npz")]] = path
    return out


def _discover_hmr(output_dir: Path, shot_id: str) -> dict[str, Path]:
    hmr_dir = output_dir / "hmr_world"
    out: dict[str, Path] = {}
    if not hmr_dir.exists():
        return out
    suffix = "_smpl_world.npz"
    if shot_id:
        prefix = f"{shot_id}__"
        for path in sorted(hmr_dir.glob(f"{prefix}*{suffix}")):
            out[path.name[len(prefix): -len(suffix)]] = path
    else:
        for path in sorted(hmr_dir.glob(f"*{suffix}")):
            if "__" not in path.name:
                out[path.name[: -len(suffix)]] = path
    return out


class _LoadedTrack:
    """A pose track plus a shot-local frame index."""

    __slots__ = ("track", "index_by_local_frame")

    def __init__(self, track: object, local_frames: np.ndarray) -> None:
        self.track = track
        self.index_by_local_frame = {
            int(f): i for i, f in enumerate(local_frames)
        }


def _load_refined(path: Path, offset: int) -> _LoadedTrack | None:
    from src.schemas.refined_pose import RefinedPose

    try:
        track = RefinedPose.load(path)
    except Exception as exc:
        logger.warning("player context: failed to load %s: %s", path, exc)
        return None
    return _LoadedTrack(track, np.asarray(track.frames) + offset)


def _load_hmr(path: Path) -> _LoadedTrack | None:
    from src.schemas.smpl_world import SmplWorldTrack

    try:
        track = SmplWorldTrack.load(path)
    except Exception as exc:
        logger.warning("player context: failed to load %s: %s", path, exc)
        return None
    return _LoadedTrack(track, np.asarray(track.frames))


class PlayerContext:
    """Immutable per-shot lookup of contact-joint samples."""

    def __init__(
        self,
        samples_by_frame: dict[int, tuple[JointSample, ...]],
        player_ids: tuple[str, ...],
    ) -> None:
        self._by_frame = dict(samples_by_frame)
        self.player_ids = player_ids
        self._world_index: dict[tuple[int, str, str], tuple[float, float, float]] = {
            (frame, s.player_id, s.bone): s.world_xyz
            for frame, samples in self._by_frame.items()
            for s in samples
        }

    def joints_at(self, frame: int) -> tuple[JointSample, ...]:
        return self._by_frame.get(int(frame), ())

    def joints_near_pixel(
        self, frame: int, uv: tuple[float, float], radius_px: float
    ) -> list[JointSample]:
        """Samples within ``radius_px`` of ``uv``, nearest first."""
        u, v = float(uv[0]), float(uv[1])
        hits: list[tuple[float, JointSample]] = []
        for s in self.joints_at(frame):
            if s.uv is None:
                continue
            d = ((s.uv[0] - u) ** 2 + (s.uv[1] - v) ** 2) ** 0.5
            if d <= radius_px:
                hits.append((d, s))
        hits.sort(key=lambda pair: pair[0])
        return [s for _, s in hits]

    def joint_world(
        self, frame: int, player_id: str, bone: str
    ) -> np.ndarray | None:
        world = self._world_index.get((int(frame), player_id, bone))
        if world is None:
            return None
        return np.asarray(world, dtype=float)

    @classmethod
    def load(
        cls,
        output_dir: Path,
        shot_id: str,
        *,
        per_frame_K: dict[int, np.ndarray],
        per_frame_R: dict[int, np.ndarray],
        per_frame_t: dict[int, np.ndarray],
        distortion: tuple[float, float] = (0.0, 0.0),
        bones: tuple[str, ...] | None = None,
    ) -> "PlayerContext":
        bone_names = tuple(bones) if bones is not None else tuple(BONE_TO_SMPL_INDEX)
        joint_indices = np.array(
            [BONE_TO_SMPL_INDEX[b] for b in bone_names], dtype=int
        )
        offset = _sync_offset(output_dir, shot_id)
        refined_paths = _discover_refined(output_dir)
        hmr_paths = _discover_hmr(output_dir, shot_id)

        loaded: dict[str, tuple[_LoadedTrack | None, _LoadedTrack | None]] = {}
        for pid in sorted(set(refined_paths) | set(hmr_paths)):
            refined = (
                _load_refined(refined_paths[pid], offset)
                if pid in refined_paths else None
            )
            hmr = _load_hmr(hmr_paths[pid]) if pid in hmr_paths else None
            if refined is not None or hmr is not None:
                loaded[pid] = (refined, hmr)

        frame_list = sorted(per_frame_K)

        # Per player: gather every query frame's FK inputs (drawn from
        # whichever of refined/hmr covers that frame, same precedence as
        # before) and run ONE batched FK call over the player's whole
        # valid-frame span, instead of one `compute_all_joint_worlds`
        # call per player per frame. Numerically identical to the
        # per-frame loop (see `compute_all_joint_worlds_batch` docstring).
        player_bones: dict[str, dict[int, np.ndarray]] = {}
        player_conf: dict[str, dict[int, float]] = {}
        for pid, (refined, hmr) in loaded.items():
            valid_frames: list[int] = []
            thetas_rows: list[np.ndarray] = []
            root_R_rows: list[np.ndarray] = []
            root_t_rows: list[np.ndarray] = []
            conf_rows: list[float] = []
            for fi in frame_list:
                chosen: _LoadedTrack | None = None
                idx: int | None = None
                for candidate in (refined, hmr):
                    if candidate is None:
                        continue
                    idx = candidate.index_by_local_frame.get(fi)
                    if idx is not None:
                        chosen = candidate
                        break
                if chosen is None or idx is None:
                    continue
                track = chosen.track
                valid_frames.append(fi)
                thetas_rows.append(track.thetas[idx])      # type: ignore[attr-defined]
                root_R_rows.append(track.root_R[idx])      # type: ignore[attr-defined]
                root_t_rows.append(track.root_t[idx])      # type: ignore[attr-defined]
                conf_rows.append(float(track.confidence[idx]))  # type: ignore[attr-defined]
            if not valid_frames:
                continue
            all_joints = compute_all_joint_worlds_batch(
                np.stack(thetas_rows), np.stack(root_R_rows), np.stack(root_t_rows),
            )
            worlds_by_frame = all_joints[:, joint_indices]  # (F, n_bones, 3)
            player_bones[pid] = {
                fi: worlds_by_frame[j] for j, fi in enumerate(valid_frames)
            }
            player_conf[pid] = dict(zip(valid_frames, conf_rows))

        # Projection: grouped per frame across players, since every
        # player at a given frame shares the same (K, R, t) — one
        # `project_world_to_image` call per frame instead of one per
        # player per frame.
        samples_by_frame: dict[int, tuple[JointSample, ...]] = {}
        for fi in frame_list:
            present_pids = [pid for pid in loaded if fi in player_bones.get(pid, {})]
            if not present_pids:
                continue
            K, R, t = per_frame_K[fi], per_frame_R[fi], per_frame_t[fi]
            worlds_concat = np.concatenate(
                [player_bones[pid][fi] for pid in present_pids], axis=0
            )
            cam_z = (worlds_concat @ R.T + t)[:, 2]
            uvs = project_world_to_image(K, R, t, distortion, worlds_concat)
            frame_samples: list[JointSample] = []
            start = 0
            n_bones = len(bone_names)
            for pid in present_pids:
                worlds = worlds_concat[start:start + n_bones]
                pid_uvs = uvs[start:start + n_bones]
                pid_z = cam_z[start:start + n_bones]
                conf = player_conf[pid][fi]
                for bone, world, uv, z in zip(bone_names, worlds, pid_uvs, pid_z):
                    frame_samples.append(JointSample(
                        player_id=pid,
                        bone=bone,
                        world_xyz=(
                            float(world[0]), float(world[1]), float(world[2])
                        ),
                        uv=(float(uv[0]), float(uv[1])) if z > 0 else None,
                        confidence=conf,
                    ))
                start += n_bones
            if frame_samples:
                samples_by_frame[fi] = tuple(frame_samples)

        return cls(samples_by_frame, tuple(sorted(loaded)))
