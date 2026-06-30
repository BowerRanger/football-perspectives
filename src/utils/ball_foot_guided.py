"""Foot-guided ball detection (inference-only).

Diagnostics showed WASB fires confidently on the WRONG object at touch
frames (often the frame top, 230-615px from any player), while the true ball
sits ~15px from a contact joint. Since events-mode body-pins the ball to the
joint anyway, we don't need WASB's wrong global hit — we look where the ball
must be: **around the reliable player feet**.

This module is the pure core: (1) ``gated_feet`` picks frames + contact
joints that are moving fast (a kick/contact signature) so the expensive zoom
only runs where a touch is plausible; (2) ``foot_ball_detections`` accepts a
zoom result as a ball when it lands near the foot. The buffer-safe zoom
itself (video crop + WASB) is injected, so this is unit-testable. The stage
wires the real zoom (reusing the second-pass ``_zoom_detect``). See
docs/superpowers/specs/2026-06-15-ball-detection-direction-changes-design.md.
"""

from __future__ import annotations

from math import hypot

from src.utils.ball_pose_touch import joint_pixel_velocity

# Contact joints worth zooming around (feet first; head/chest for aerials).
_DEFAULT_BONES = ("l_foot", "r_foot", "l_knee", "r_knee", "head", "chest")


def gated_feet(
    player_ctx,
    n_frames: int,
    *,
    min_foot_speed_px: float = 8.0,
    bones: tuple[str, ...] = _DEFAULT_BONES,
) -> dict[int, list[tuple[str, str, tuple[float, float]]]]:
    """``{frame: [(player_id, bone, foot_uv), ...]}`` for contact joints whose
    pixel speed at that frame exceeds ``min_foot_speed_px`` — the likely-kick
    moments where a foot-zoom should look for the ball."""
    out: dict[int, list[tuple[str, str, tuple[float, float]]]] = {}
    for f in range(n_frames):
        feet: list[tuple[str, str, tuple[float, float]]] = []
        for s in player_ctx.joints_at(f):
            if s.bone not in bones or s.uv is None:
                continue
            v = joint_pixel_velocity(player_ctx, f, s.player_id, s.bone)
            speed = hypot(v[0], v[1]) if v is not None else 0.0
            if speed >= min_foot_speed_px:
                feet.append((s.player_id, s.bone,
                             (float(s.uv[0]), float(s.uv[1]))))
        if feet:
            out[f] = feet
    return out


def foot_ball_detections(
    gated: dict[int, list[tuple[str, str, tuple[float, float]]]],
    zoom_fn,
    *,
    ball_near_foot_px: float = 40.0,
    min_score: float = 0.2,
) -> list[tuple[int, str, str, tuple[float, float], float]]:
    """For each gated frame, zoom around each foot and accept the strongest
    ball that lands within ``ball_near_foot_px`` of a foot.

    ``zoom_fn(frame, foot_uv) -> ((u, v), score) | None`` is the injected
    buffer-safe zoom detector. Returns one
    ``(frame, player_id, bone, ball_uv, score)`` per frame (the best foot).
    """
    out: list[tuple[int, str, str, tuple[float, float], float]] = []
    for f in sorted(gated):
        best: tuple[int, str, str, tuple[float, float], float] | None = None
        for pid, bone, foot_uv in gated[f]:
            res = zoom_fn(f, foot_uv)
            if res is None:
                continue
            buv, score = res
            if score < min_score:
                continue
            if hypot(buv[0] - foot_uv[0], buv[1] - foot_uv[1]) > ball_near_foot_px:
                continue
            if best is None or score > best[4]:
                best = (f, pid, bone,
                        (float(buv[0]), float(buv[1])), float(score))
        if best is not None:
            out.append(best)
    return out


__all__ = ["gated_feet", "foot_ball_detections"]
