"""Touch bone-attribution refinement (spec §4.4).

Validated on gberch: with the same ±2-frame tolerance, ignoring the bone
claim lifts touch recall from 0.25 to 0.50 — half the touch moments are
found but pinned to the wrong body part. The original attribution happens
at the (noisy) break/proposal moment; this post-pass re-picks each touch
event's (player, bone) as the joint with the smallest 3-D bone↔ball-ray
gap over a small window around the event frame, keeping the original when
the improvement is within an ambiguity margin. It relabels ONLY — never
adds, removes, re-frames, or re-scores events. Pure and torch-free.

Default-off: relabelling trusts the ball pixel, which is exactly what is
unreliable on detector-limited clips — enable per-clip once detection
quality at touch moments is validated.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

import numpy as np

from src.utils.ball_kinematic_touch import point_to_pixel_ray_distance

if TYPE_CHECKING:  # pragma: no cover — typing only
    from src.utils.ball_auto_events import BallEvent


@dataclass(frozen=True)
class TouchAttributionCfg:
    enabled: bool = False
    window: int = 2          # +/- frames considered around the event frame
    max_gap_m: float = 0.45  # candidate joints beyond this never relabel
    margin_m: float = 0.05   # new joint must beat the current one by this
    min_fk_conf: float = 0.3


def _best_gaps_in_window(
    frame: int,
    *,
    player_ctx,
    ball_uvs: dict[int, np.ndarray],
    per_frame_K: dict[int, np.ndarray],
    per_frame_R: dict[int, np.ndarray],
    per_frame_t: dict[int, np.ndarray],
    distortion: tuple[float, float],
    cfg: TouchAttributionCfg,
) -> dict[tuple[str, str], float]:
    """Per-(player, bone) minimal ray gap over the window around ``frame``."""
    best: dict[tuple[str, str], float] = {}
    for f in range(frame - cfg.window, frame + cfg.window + 1):
        ball_uv = ball_uvs.get(f)
        K, R, t = per_frame_K.get(f), per_frame_R.get(f), per_frame_t.get(f)
        if ball_uv is None or K is None or R is None or t is None:
            continue
        for s in player_ctx.joints_at(f):
            if s.confidence < cfg.min_fk_conf or s.world_xyz is None:
                continue
            gap = float(point_to_pixel_ray_distance(
                np.asarray(s.world_xyz, dtype=float), ball_uv,
                K, R, t, distortion,
            ))
            key = (s.player_id, s.bone)
            if gap < best.get(key, float("inf")):
                best[key] = gap
    return best


def refine_touch_attribution(
    events: "Sequence[BallEvent]",
    *,
    player_ctx,
    ball_uvs: dict[int, np.ndarray],
    per_frame_K: dict[int, np.ndarray],
    per_frame_R: dict[int, np.ndarray],
    per_frame_t: dict[int, np.ndarray],
    distortion: tuple[float, float],
    cfg: TouchAttributionCfg,
) -> "tuple[BallEvent, ...]":
    """Relabel touch events to the ray-closest joint; everything else
    passes through untouched (same order, same length)."""
    if not cfg.enabled:
        return tuple(events)
    out: "list[BallEvent]" = []
    for e in events:
        if e.kind != "touch" or not e.player_id or not e.bone:
            out.append(e)
            continue
        gaps = _best_gaps_in_window(
            e.frame, player_ctx=player_ctx, ball_uvs=ball_uvs,
            per_frame_K=per_frame_K, per_frame_R=per_frame_R,
            per_frame_t=per_frame_t, distortion=distortion, cfg=cfg,
        )
        if not gaps:
            out.append(e)
            continue
        (best_pid, best_bone), best_gap = min(
            gaps.items(), key=lambda kv: (kv[1], kv[0]))
        current_gap = gaps.get((e.player_id, e.bone))
        relabel = (
            best_gap <= cfg.max_gap_m
            and (best_pid, best_bone) != (e.player_id, e.bone)
            and (current_gap is None or best_gap + cfg.margin_m < current_gap)
        )
        if relabel:
            out.append(dataclasses.replace(
                e, player_id=best_pid, bone=best_bone))
        else:
            out.append(e)
    return tuple(out)
