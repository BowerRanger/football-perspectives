"""Body-kinematics touch proposer.

Generates candidate ball touches from limb motion, independent of whether a
ball-pixel velocity break exists at the contact. The body is the trigger; the
ball is only a confidence modifier. See
docs/superpowers/specs/2026-06-27-body-kinematics-touch-proposer-design.md.

Pure and torch-free: PlayerContext samples + a ball pixel dict in, BallEvents
out.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from typing import TYPE_CHECKING

import numpy as np

from src.utils.ball_pose_touch import joint_pixel_velocity
from src.utils.camera_projection import point_to_pixel_ray_distance

if TYPE_CHECKING:
    from src.utils.ball_player_context import PlayerContext

FOOT_BONES = ("l_foot", "r_foot")
KNEE_BONES = ("l_knee", "r_knee")
SPEED_GATED_BONES = FOOT_BONES + KNEE_BONES
HEAD_BONES = ("head",)

# pixel speed (px/frame) at which a foot is "clearly kicking" — strength saturates
_KICK_SPEED_PX = 12.0
# head 3-D speed (m/frame) at which a header strength saturates
_HEAD_SPEED_SAT_M = 0.15


@dataclass(frozen=True)
class KinematicTouchCfg:
    """Thresholds + score weights for the kinematic touch proposer.

    Pixel units are px/frame; gaps are 3-D metres; speeds for feet are
    px/frame, for the head metres/frame.

    Score weights: ``w_gap + w_kin + w_confirm + w_fk == 1.0`` are the positive
    contributors (a perfect touch scores 1.0); ``w_interp`` is a separate
    subtractive penalty applied when the ball pixel was interpolated.
    """

    enabled: bool = True
    contact_gap_m: float = 0.30
    touch_relaxed_px: float = 60.0
    max_ball_gap_frames: int = 6
    min_fk_conf: float = 0.3
    kin_window: int = 2
    kin_min_foot_speed: float = 8.0
    kin_min_head_speed_m: float = 0.05
    confirm_window: int = 3
    nms_window: int = 2
    w_gap: float = 0.35
    w_kin: float = 0.30
    w_confirm: float = 0.25
    w_fk: float = 0.10
    w_interp: float = 0.15
    min_emit_score: float = 0.25


def interpolate_ball_uvs(
    ball_uvs: dict[int, np.ndarray], max_gap_frames: int
) -> tuple[dict[int, np.ndarray], frozenset[int]]:
    """Linear-fill ball-pixel gaps of length <= ``max_gap_frames``.

    Returns ``(filled, interpolated_frames)``. Frames present in ``ball_uvs``
    are copied through; gaps longer than the cap are left empty.
    """
    if not ball_uvs:
        return {}, frozenset()
    frames = sorted(ball_uvs)
    filled: dict[int, np.ndarray] = {f: np.asarray(ball_uvs[f], dtype=float) for f in frames}
    interp: set[int] = set()
    for a, b in zip(frames[:-1], frames[1:]):
        span = b - a
        if span <= 1 or span - 1 > max_gap_frames:
            continue
        pa, pb = filled[a], filled[b]
        for f in range(a + 1, b):
            w = (f - a) / span
            filled[f] = pa * (1.0 - w) + pb * w
            interp.add(f)
    return filled, frozenset(interp)


def ray_gap_series(
    player_ctx: "PlayerContext",
    ball_uvs: dict[int, np.ndarray],
    per_frame_K: dict[int, np.ndarray],
    per_frame_R: dict[int, np.ndarray],
    per_frame_t: dict[int, np.ndarray],
    distortion: tuple[float, float],
    min_fk_conf: float,
) -> dict[tuple[str, str], dict[int, tuple[float, float, float]]]:
    """Per-(player, bone) frame -> (gap3d_m, pixgap_px, fk_conf)."""
    out: dict[tuple[str, str], dict[int, tuple[float, float, float]]] = {}
    for frame, ball_uv in ball_uvs.items():
        K = per_frame_K.get(frame)
        R = per_frame_R.get(frame)
        t = per_frame_t.get(frame)
        if K is None or R is None or t is None:
            continue
        for s in player_ctx.joints_at(frame):
            if s.confidence < min_fk_conf or s.uv is None:
                continue
            world = np.asarray(s.world_xyz, dtype=float)
            gap3d = point_to_pixel_ray_distance(world, ball_uv, K, R, t, distortion)
            pixgap = hypot(s.uv[0] - float(ball_uv[0]), s.uv[1] - float(ball_uv[1]))
            out.setdefault((s.player_id, s.bone), {})[frame] = (
                float(gap3d),
                float(pixgap),
                float(s.confidence),
            )
    return out


def local_minima_below(series: dict[int, float], threshold: float) -> list[int]:
    """Frames that are strict-left, non-strict-right local minima over present
    neighbours and <= ``threshold``. Endpoints count if below their one
    neighbour."""
    frames = sorted(series)
    minima: list[int] = []
    for i, f in enumerate(frames):
        v = series[f]
        if v > threshold:
            continue
        left = series[frames[i - 1]] if i > 0 else float("inf")
        right = series[frames[i + 1]] if i < len(frames) - 1 else float("inf")
        if v < left and v <= right:
            minima.append(f)
    return minima


def _peak_foot_speed(player_ctx: "PlayerContext", frame: int, player_id: str, bone: str,
                     window: int) -> float:
    """Max central-difference foot pixel speed within +/- window of frame."""
    peak = 0.0
    for f in range(frame - window, frame + window + 1):
        vel = joint_pixel_velocity(player_ctx, f, player_id, bone)
        if vel is not None:
            peak = max(peak, hypot(vel[0], vel[1]))
    return peak


def _head_speed_m(player_ctx: "PlayerContext", frame: int, player_id: str, bone: str) -> float:
    """Central-difference 3-D speed (m/frame) of a bone, 0 if unavailable."""
    prev = player_ctx.joint_world(frame - 1, player_id, bone)
    nxt = player_ctx.joint_world(frame + 1, player_id, bone)
    if prev is None or nxt is None:
        return 0.0
    return float(np.linalg.norm((nxt - prev) / 2.0))


def kinematic_gate(player_ctx: "PlayerContext", frame: int, player_id: str, bone: str,
                   cfg: KinematicTouchCfg) -> tuple[bool, float]:
    """(passed, strength in [0,1]) for the bone's contact signature."""
    if bone in SPEED_GATED_BONES:
        peak = _peak_foot_speed(player_ctx, frame, player_id, bone, cfg.kin_window)
        return (peak >= cfg.kin_min_foot_speed, min(1.0, peak / _KICK_SPEED_PX))
    if bone in HEAD_BONES:
        speed = _head_speed_m(player_ctx, frame, player_id, bone)
        return (speed >= cfg.kin_min_head_speed_m,
                min(1.0, speed / _HEAD_SPEED_SAT_M))
    # hands / shoulders / chest: a body part on the ball line is a contact;
    # no speed requirement (keeper saves, blocks).
    return (True, 0.5)
