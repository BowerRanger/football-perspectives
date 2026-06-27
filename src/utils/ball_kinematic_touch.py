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

import numpy as np


@dataclass(frozen=True)
class KinematicTouchCfg:
    """Thresholds + score weights for the kinematic touch proposer.

    Pixel units are px/frame; gaps are 3-D metres; speeds for feet are
    px/frame, for the head metres/frame.
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
