"""Carry / possession span detection.

A dribble is a run of touches by the SAME player, close together in time
and space — the ball travels *with* the player rather than being passed
away. Representing these spans as ``carry`` interpolation segments stops a
dribble from being drawn as straight lines between sparse touches (see the
§6.3 / §10 design notes).

Carry is modelled as a SEGMENT classification between two consecutive
``player_touch`` keyframes, so no new anchor state is required.
"""

from __future__ import annotations

from collections.abc import Sequence

from src.schemas.ball_keyframes import BallKeyframe

_DEFAULT_MAX_GAP_FRAMES = 15
_DEFAULT_MAX_STEP_M = 3.0


def _dist(a, b) -> float:
    return sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)) ** 0.5


def detect_carry_spans(
    keyframes: Sequence[BallKeyframe],
    *,
    max_gap_frames: int = _DEFAULT_MAX_GAP_FRAMES,
    max_step_m: float = _DEFAULT_MAX_STEP_M,
) -> list[tuple[int, int]]:
    """Return ``(start_frame, end_frame)`` spans between consecutive
    same-player touches that look like a dribble.

    A pair qualifies when both keyframes are ``player_touch`` by the same
    ``player_id``, separated by at most ``max_gap_frames`` frames and at
    most ``max_step_m`` metres (so a long pass is excluded).
    """
    touches = sorted(
        (k for k in keyframes
         if k.state == "player_touch" and k.player_id and k.world_xyz),
        key=lambda k: k.frame,
    )
    spans: list[tuple[int, int]] = []
    for a, b in zip(touches, touches[1:]):
        if a.player_id != b.player_id:
            continue
        if b.frame - a.frame <= 0 or b.frame - a.frame > max_gap_frames:
            continue
        if _dist(a.world_xyz, b.world_xyz) > max_step_m:
            continue
        spans.append((a.frame, b.frame))
    return spans


__all__ = ["detect_carry_spans"]
