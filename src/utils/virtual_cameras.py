"""Synthesised player POV / over-the-shoulder cameras.

Pure math + rig builders. No file I/O — the export stage handles reading
selections and writing CameraTrack JSON. Conventions match the broadcast
camera: ``R`` is world->camera (OpenCV: +Z optical ray into scene, +X
right, +Y down); per-frame ``t`` satisfies camera-centre ``C = -R.T @ t``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from src.schemas.camera_track import CameraFrame, CameraTrack
from src.utils.smpl_skeleton import compute_joint_world_pose

WORLD_UP = np.array([0.0, 0.0, 1.0])
WORLD_UP.flags.writeable = False

_FrameTuple = tuple[int, np.ndarray, np.ndarray, float]  # (frame_idx, R_world2cam, t, confidence)


def _normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(v))
    return v / n if n > eps else v


def intrinsics_from_fov(fov_deg: float, image_size: tuple[int, int]) -> list[list[float]]:
    """3x3 K from a horizontal field of view. Principal point centred."""
    if not (0.0 < fov_deg < 180.0):
        raise ValueError(f"fov_deg must be in (0, 180), got {fov_deg}")
    w, h = int(image_size[0]), int(image_size[1])
    f = (w / 2.0) / math.tan(math.radians(fov_deg) / 2.0)
    return [[f, 0.0, w / 2.0], [0.0, f, h / 2.0], [0.0, 0.0, 1.0]]


def look_at_view(
    center: np.ndarray,
    target: np.ndarray,
    up: np.ndarray = WORLD_UP,
) -> tuple[np.ndarray, np.ndarray]:
    """World->camera (R, t) for a camera at ``center`` looking at ``target``.

    Rows of ``R`` are the camera axes in world coords: right (+X), down
    (+Y), forward (+Z). ``t = -R @ center``.
    """
    center = np.asarray(center, dtype=np.float64).reshape(3)
    target = np.asarray(target, dtype=np.float64).reshape(3)
    if float(np.linalg.norm(target - center)) < 1e-9:
        raise ValueError(
            f"look_at_view: center and target are coincident (center={center}, target={target})"
        )
    z = _normalize(target - center)
    up = np.asarray(up, dtype=np.float64).reshape(3)
    x = np.cross(z, up)
    if float(np.linalg.norm(x)) < 1e-9:
        # Optical axis parallel to up — pick an arbitrary stable basis.
        x = np.cross(z, np.array([0.0, 1.0, 0.0]))
        if float(np.linalg.norm(x)) < 1e-9:
            x = np.cross(z, np.array([1.0, 0.0, 0.0]))
    x = _normalize(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z], axis=0)
    t = -R @ center
    return R, t


HEAD_JOINT_IDX = 15
# SMPL canonical (y-up) facing axis. +Z is "forward" out of the torso for the
# rest pose; sign may need flipping after the first real export — kept as a
# module constant so tuning is a one-line change.
FACE_AXIS_CANONICAL = np.array([0.0, 0.0, 1.0])
FACE_AXIS_CANONICAL.flags.writeable = False


@dataclass(frozen=True)
class RigConfig:
    pov_fov_deg: float = 75.0
    ots_fov_deg: float = 60.0
    ots_back_m: float = 0.4
    ots_up_m: float = 0.3
    ots_right_m: float = 0.0
    ball_target_max_occlusion_frames: int = 10


def _head_pose_world(
    track: "SmplWorldTrack",
    i: int,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Return ``(head_pos, head_R, ok)`` for frame index ``i``.

    ``ok`` is ``True`` on the normal FK path and ``False`` when FK raises
    and we fall back to root-only head pose. Callers should halve frame
    confidence when ``ok`` is ``False``.
    """
    try:
        pos, R = compute_joint_world_pose(
            track.thetas[i], track.root_R[i], track.root_t[i], HEAD_JOINT_IDX
        )
        return pos, R, True
    except (ValueError, IndexError, np.linalg.LinAlgError):
        pos = np.asarray(track.root_t[i], dtype=np.float64) + np.array([0.0, 0.0, 1.6])
        return pos, np.asarray(track.root_R[i], dtype=np.float64), False


def _ball_xyz_by_frame(ball_track: object) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    if ball_track is None:
        return out
    for f in getattr(ball_track, "frames", ()):
        xyz = getattr(f, "world_xyz", None)
        if xyz is not None:
            out[int(f.frame)] = np.asarray(xyz, dtype=np.float64).reshape(3)
    return out


def _make_track(
    clip_id: str,
    image_size: tuple[int, int],
    fps: float,
    K: list[list[float]],
    per_frame: list[_FrameTuple],
) -> CameraTrack:
    frames = tuple(
        CameraFrame(
            frame=int(fr),
            K=[list(map(float, row)) for row in K],
            R=[list(map(float, row)) for row in R],
            confidence=float(conf),
            is_anchor=False,
            t=[float(x) for x in t],
        )
        for (fr, R, t, conf) in per_frame
    )
    if per_frame:
        centres = np.array(
            [-(np.asarray(R)).T @ np.asarray(t) for (_, R, t, _) in per_frame]
        )
        t_world = centres.mean(axis=0).tolist()
    else:
        t_world = [0.0, 0.0, 0.0]
    return CameraTrack(
        clip_id=clip_id,
        fps=float(fps),
        image_size=(int(image_size[0]), int(image_size[1])),
        t_world=t_world,
        frames=frames,
    )


def build_pov_track(
    track: "SmplWorldTrack",
    cfg: RigConfig,
    image_size: tuple[int, int],
    fps: float,
    clip_id: str,
) -> CameraTrack:
    """Build a first-person (POV) CameraTrack from a player's SmplWorldTrack.

    The camera is placed at the player's head joint and aimed in the
    player's facing direction (SMPL canonical forward rotated into world).
    """
    K = intrinsics_from_fov(cfg.pov_fov_deg, image_size)
    per_frame: list[_FrameTuple] = []
    for i, fr in enumerate(np.asarray(track.frames).tolist()):
        head_pos, head_R, ok = _head_pose_world(track, i)
        facing = _normalize(head_R @ FACE_AXIS_CANONICAL)
        R, t = look_at_view(head_pos, head_pos + facing)
        conf = float(track.confidence[i]) * (1.0 if ok else 0.5)
        per_frame.append((int(fr), R, t, conf))
    return _make_track(clip_id, image_size, fps, K, per_frame)


def build_ots_track(
    track: "SmplWorldTrack",
    ball_track: object,
    cfg: RigConfig,
    image_size: tuple[int, int],
    fps: float,
    clip_id: str,
) -> CameraTrack:
    """Build an over-the-shoulder (OTS) CameraTrack from a player's SmplWorldTrack.

    Camera is positioned slightly behind, above, and optionally to the side
    of the player's head. Target is the ball when available (with short
    occlusion bridging), otherwise falls back to a point ahead of the player.
    """
    K = intrinsics_from_fov(cfg.ots_fov_deg, image_size)
    ball_xyz = _ball_xyz_by_frame(ball_track)
    per_frame: list[_FrameTuple] = []
    last_target: np.ndarray | None = None
    frames_since_ball = 0
    for i, fr in enumerate(np.asarray(track.frames).tolist()):
        head_pos, head_R, ok = _head_pose_world(track, i)
        facing = _normalize(head_R @ FACE_AXIS_CANONICAL)
        facing_ground = _normalize(np.array([facing[0], facing[1], 0.0]))
        right_ground = _normalize(np.cross(facing_ground, WORLD_UP))
        center = (
            head_pos
            - cfg.ots_back_m * facing_ground
            + cfg.ots_up_m * WORLD_UP
            + cfg.ots_right_m * right_ground
        )
        target = ball_xyz.get(int(fr))
        if target is not None:
            last_target = target
            frames_since_ball = 0
        elif (
            last_target is not None
            and frames_since_ball < cfg.ball_target_max_occlusion_frames
        ):
            target = last_target
            frames_since_ball += 1
        else:
            target = head_pos + facing * 10.0
        R, t = look_at_view(center, target)
        conf = float(track.confidence[i]) * (1.0 if ok else 0.5)
        per_frame.append((int(fr), R, t, conf))
    return _make_track(clip_id, image_size, fps, K, per_frame)
