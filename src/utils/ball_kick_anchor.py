"""Layer 3 of the ball-tracking improvement plan: anchor flight-segment
p0 to a player's foot when a kick is detected at the segment's seed
frame.

Inputs are in pixel space (ball uv per frame, ankle uv per frame). When
a kick is detected, the closest ankle pixel is ray-cast onto the ground
plane at z = foot_anchor_z_m via :func:`src.utils.foot_anchor.ankle_ray_to_pitch`,
yielding the world anchor point ``p0``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.utils.foot_anchor import ankle_ray_to_pitch


@dataclass(frozen=True)
class KickAnchorCfg:
    enabled: bool
    max_pixel_distance_px: float
    lookahead_frames: int
    min_pixel_acceleration_px_per_frame: float
    foot_anchor_z_m: float


def _pixel_acceleration(
    ball_uvs: dict[int, tuple[float, float]],
    start: int,
    lookahead: int,
) -> float:
    """Max change in pixel-velocity magnitude over the lookahead window."""
    frames = sorted(f for f in ball_uvs if start <= f <= start + lookahead)
    if len(frames) < 3:
        return 0.0
    speeds = []
    for a, b in zip(frames[:-1], frames[1:]):
        du = ball_uvs[b][0] - ball_uvs[a][0]
        dv = ball_uvs[b][1] - ball_uvs[a][1]
        speeds.append(np.hypot(du, dv) / max(b - a, 1))
    return float(max(speeds) - min(speeds))


def find_kick_anchor(
    *,
    segment_start_frame: int,
    ball_uvs: dict[int, tuple[float, float]],
    foot_uvs_by_frame: dict[int, tuple[float, float]],
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    cfg: KickAnchorCfg,
    distortion: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray | None:
    """Return a world-space (x, y, foot_anchor_z_m) anchor if a kick is
    detected at ``segment_start_frame``; otherwise ``None``.

    A kick is declared when:
      - a foot pixel is available at the seed frame within
        ``max_pixel_distance_px`` of the ball pixel; AND
      - the ball pixel-speed varies by at least
        ``min_pixel_acceleration_px_per_frame`` across the
        ``lookahead_frames``-frame window starting at the seed frame.
    """
    if not cfg.enabled:
        return None
    if segment_start_frame not in ball_uvs:
        return None
    if segment_start_frame not in foot_uvs_by_frame:
        return None

    ball_uv = ball_uvs[segment_start_frame]
    foot_uv = foot_uvs_by_frame[segment_start_frame]
    pixel_distance = float(np.hypot(ball_uv[0] - foot_uv[0], ball_uv[1] - foot_uv[1]))
    if pixel_distance > cfg.max_pixel_distance_px:
        return None

    accel = _pixel_acceleration(ball_uvs, segment_start_frame, cfg.lookahead_frames)
    if accel < cfg.min_pixel_acceleration_px_per_frame:
        return None

    anchor_world = ankle_ray_to_pitch(
        foot_uv,
        K=K,
        R=R,
        t=t,
        plane_z=cfg.foot_anchor_z_m,
        distortion=distortion,
    )
    return np.asarray(anchor_world, dtype=float)
