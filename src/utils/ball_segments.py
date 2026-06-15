"""Derive interpolation ``BallSegment``s from an ordered list of resolved
``BallKeyframe``s.

A segment describes how the ball travels between two consecutive keyframes
(see the §10 contract). Segmentation is derived once here so the Python
reference interpolator and the UE side agree rather than each re-deriving
it. Clip-boundary open ends (before the first / after the last keyframe)
get a constant-position hold (the safe default; airborne back/forward
extrapolation is a later refinement — a detected ``goal_impact`` /
``off_screen_flight`` event usually supplies the real endpoint).
"""

from __future__ import annotations

from collections.abc import Sequence

from src.schemas.ball_keyframes import BallKeyframe, BallSegment

_DEFAULT_GRAVITY = -9.81
_DEFAULT_RESTITUTION = 0.7

# Airborne position buckets.
_AIRBORNE_STATES = frozenset({"airborne_low", "airborne_mid", "airborne_high"})

# States that, at either end of a gap, imply the ball was in flight across
# that gap (so the gap is ballistic rather than a ground roll).
_FLIGHT_IMPLYING = frozenset({
    "airborne_low", "airborne_mid", "airborne_high",
    "kick", "volley", "header", "chest", "catch", "bounce", "goal_impact",
})

# Squared metres: positions closer than this are treated as "the same"
# (a hold / rest rather than a roll).
_REST_EPS_M2 = 0.04 * 0.04


def _implies_flight(kf: BallKeyframe) -> bool:
    if kf.state in _FLIGHT_IMPLYING:
        return True
    if kf.state == "player_touch" and kf.touch_type in ("shot", "volley"):
        return True
    return False


def _dist2(a, b) -> float:
    return sum((float(x) - float(y)) ** 2 for x, y in zip(a, b))


def _between_kind(a: BallKeyframe, b: BallKeyframe) -> str:
    if a.state == "off_screen_flight" or b.state == "off_screen_flight":
        return "free_flight"
    if _implies_flight(a) or _implies_flight(b):
        return "ballistic"
    if a.world_xyz is not None and b.world_xyz is not None:
        if _dist2(a.world_xyz, b.world_xyz) <= _REST_EPS_M2:
            return "rest"
    return "roll"


def _hints_for(a: BallKeyframe, kind: str) -> dict:
    hints: dict = {}
    if kind in ("ballistic", "free_flight"):
        hints["gravity"] = _DEFAULT_GRAVITY
        hints["restitution"] = _DEFAULT_RESTITUTION
        if a.omega_rad_s is not None:
            hints["omega_rad_s"] = [float(x) for x in a.omega_rad_s]
        if a.spin is not None:
            hints["spin"] = a.spin
    if a.player_id is not None:
        hints["player_id"] = a.player_id
    return hints


def derive_segments(
    keyframes: Sequence[BallKeyframe],
    *,
    n_frames: int,
    fps: float,
    carry_spans: Sequence[tuple[int, int]] = (),
) -> tuple[BallSegment, ...]:
    """Return ordered segments covering the clip.

    ``keyframes`` need not be pre-sorted; they are sorted by frame here.
    ``carry_spans`` are ``(start_frame, end_frame)`` pairs (from
    :func:`ball_possession.detect_carry_spans`) that force the segment
    between those two touch keyframes to be a ``carry``.
    """
    if not keyframes:
        return ()
    carry_set = set(carry_spans or ())
    kfs = sorted(keyframes, key=lambda k: k.frame)
    segments: list[BallSegment] = []

    first = kfs[0]
    if first.frame > 0:
        # Open start: hold the first keyframe's position back to frame 0.
        segments.append(BallSegment(
            start_frame=0, end_frame=first.frame, kind="rest",
            hints={"inferred": True},
        ))

    for a, b in zip(kfs, kfs[1:]):
        if b.frame <= a.frame:
            continue
        kind = "carry" if (a.frame, b.frame) in carry_set else _between_kind(a, b)
        segments.append(BallSegment(
            start_frame=a.frame, end_frame=b.frame, kind=kind,
            hints=_hints_for(a, kind),
        ))

    last = kfs[-1]
    if last.frame < n_frames - 1:
        # Open end: hold the last keyframe's position to the final frame.
        segments.append(BallSegment(
            start_frame=last.frame, end_frame=n_frames - 1, kind="rest",
            hints={"open_ended": True},
        ))

    return tuple(segments)


__all__ = ["derive_segments"]
