"""Pose-anchored touch attribution (Phase C).

The bottleneck for touch recall is not raw ball detection but *attribution*:
on gberch there were 22 direction-change breaks but only 1 became a touch,
because the strict 25px joint-proximity gate rejected the rest. This module
attributes a break to a player body part with:

  * a **relaxed** proximity radius (``touch_relaxed_px``), and
  * a **kinematic bonus** — a body part moving fast (in pixels) at the break
    is the likely toucher (a planted/standing foot is not).

Dense, reliable poses (``PlayerContext``) make this robust. High-recall by
design (the editor prunes false positives). Replaces ``_classify_touch`` when
``cfg.use_pose_touch``. See
docs/superpowers/specs/2026-06-15-ball-detection-direction-changes-design.md §6.
"""

from __future__ import annotations

from math import hypot

from src.utils.ball_auto_events import AutoEventCfg, BallEvent, _Break

# Pixel speed (px/frame) at which a foot is "clearly kicking" — bonus saturates.
_KICK_SPEED_PX = 12.0


def _joint_uv(player_ctx, frame: int, player_id: str, bone: str):
    for s in player_ctx.joints_at(frame):
        if s.player_id == player_id and s.bone == bone and s.uv is not None:
            return s.uv
    return None


def joint_pixel_velocity(
    player_ctx, frame: int, player_id: str, bone: str
) -> tuple[float, float] | None:
    """Central finite-difference pixel velocity (px/frame) of a body part,
    or ``None`` if either neighbour frame lacks the joint."""
    prev = _joint_uv(player_ctx, frame - 1, player_id, bone)
    nxt = _joint_uv(player_ctx, frame + 1, player_id, bone)
    if prev is None or nxt is None:
        return None
    return ((nxt[0] - prev[0]) / 2.0, (nxt[1] - prev[1]) / 2.0)


def classify_touch(
    brk: _Break,
    uv: tuple[float, float],
    player_ctx,
    cfg: AutoEventCfg,
) -> BallEvent | None:
    """Attribute a direction-change break to the most likely body part.

    Searches a relaxed radius around the ball pixel (probing frame +/-1 for
    pose-sampling jitter) and scores each candidate joint by pixel proximity,
    break strength, foot pixel speed (kinematic bonus) and FK confidence.
    Returns the best as a ``touch`` event, or ``None`` if no joint is near.
    """
    radius = float(getattr(cfg, "touch_relaxed_px", 60.0))
    kw = float(getattr(cfg, "kinematic_bonus_weight", 0.3))
    best: tuple[float, str, str] | None = None  # (score, player_id, bone)
    seen: set[tuple[str, str]] = set()
    for probe in (0, -1, 1):
        for s in player_ctx.joints_near_pixel(
            brk.frame + probe, (float(uv[0]), float(uv[1])), radius
        ):
            key = (s.player_id, s.bone)
            if key in seen:
                continue
            seen.add(key)
            d = hypot(s.uv[0] - uv[0], s.uv[1] - uv[1])
            vel = joint_pixel_velocity(player_ctx, brk.frame, s.player_id, s.bone)
            foot_speed = hypot(vel[0], vel[1]) if vel is not None else 0.0
            kin = min(1.0, foot_speed / _KICK_SPEED_PX)
            score = (
                0.4 * (1.0 - d / radius)
                + 0.3 * brk.strength
                + kw * kin
                + 0.1 * float(s.confidence)
            )
            if best is None or score > best[0]:
                best = (score, s.player_id, s.bone)
    if best is None:
        return None
    return BallEvent(
        frame=brk.frame, kind="touch", score=min(1.0, best[0]),
        player_id=best[1], bone=best[2],
    )


__all__ = ["classify_touch", "joint_pixel_velocity"]
