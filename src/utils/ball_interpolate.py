"""Pure reference interpolator: render a sparse ``BallKeyframeSet`` (plus
its derived ``BallSegment``s) into a dense ``BallTrack``.

This is the §10 interpolation contract implemented in Python. It exists so
the web viewer / glTF export still get a dense, moving ball, and so the UE
side has an executable reference to validate its own interpolation against
(see ``docs/superpowers/specs/2026-06-15-ball-touch-events-design.md`` §11).
It is deterministic and depends only on numpy + the existing orientation
integrator — no torch, no video.

The authoritative product remains the sparse keyframes + segments; this
module just materialises them per-frame.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from src.schemas.ball_keyframes import BallKeyframe, BallKeyframeSet, BallSegment
from src.schemas.ball_track import BallFrame, BallTrack, FlightSegment
from src.utils.ball_orientation import integrate_orientation

_DEFAULT_GRAVITY = -9.81
_BALL_RADIUS_M = 0.11

# Segment kinds whose frames are airborne (dense state "flight").
_FLIGHT_KINDS = frozenset({"ballistic", "free_flight"})


def _xyz(v) -> np.ndarray | None:
    if v is None:
        return None
    return np.asarray(v, dtype=float)


def _n_frames(keyframe_set: BallKeyframeSet, n_frames: int | None) -> int:
    if n_frames is not None:
        return int(n_frames)
    last = 0
    for kf in keyframe_set.keyframes:
        last = max(last, kf.frame, kf.end_frame or kf.frame)
    for seg in keyframe_set.segments:
        last = max(last, seg.end_frame)
    return last + 1


def _ballistic_v0(
    p0: np.ndarray, p1: np.ndarray, total_t: float, gravity: float,
) -> np.ndarray:
    """Launch velocity so a gravity parabola hits ``p0`` at t=0 and ``p1``
    at t=total_t. ``a = (0, 0, gravity)``."""
    a = np.array([0.0, 0.0, gravity])
    if total_t <= 0.0:
        return np.zeros(3)
    return (p1 - p0 - 0.5 * a * total_t * total_t) / total_t


def _eval_ballistic(
    seg: BallSegment,
    p0: np.ndarray,
    p1: np.ndarray,
    fps: float,
    world: dict[int, np.ndarray | None],
) -> FlightSegment | None:
    gravity = float(seg.hints.get("gravity", _DEFAULT_GRAVITY))
    total_t = (seg.end_frame - seg.start_frame) / fps
    v0 = _ballistic_v0(p0, p1, total_t, gravity)
    a = np.array([0.0, 0.0, gravity])
    for f in range(seg.start_frame, seg.end_frame + 1):
        t = (f - seg.start_frame) / fps
        world[f] = p0 + v0 * t + 0.5 * a * t * t
    # Endpoints exact (guard against float drift).
    world[seg.start_frame] = p0
    world[seg.end_frame] = p1
    parabola: dict = {
        "p0": [float(x) for x in p0],
        "v0": [float(x) for x in v0],
        "g": gravity,
    }
    omega = seg.hints.get("omega_rad_s")
    if omega is not None:
        omega_arr = np.asarray(omega, dtype=float)
        mag = float(np.linalg.norm(omega_arr))
        if mag > 0.0:
            parabola["spin_axis_world"] = [float(x) for x in omega_arr / mag]
            parabola["spin_omega_rad_s"] = mag
    return FlightSegment(
        id=seg.start_frame,
        frame_range=(seg.start_frame, seg.end_frame),
        parabola=parabola,
        fit_residual_px=0.0,
    )


def _eval_linear(
    seg: BallSegment,
    p0: np.ndarray,
    p1: np.ndarray,
    world: dict[int, np.ndarray | None],
) -> None:
    span = seg.end_frame - seg.start_frame
    for f in range(seg.start_frame, seg.end_frame + 1):
        s = 0.0 if span == 0 else (f - seg.start_frame) / span
        world[f] = p0 + (p1 - p0) * s


def interpolate_events(
    keyframe_set: BallKeyframeSet,
    *,
    n_frames: int | None = None,
    ball_radius_m: float = _BALL_RADIUS_M,
) -> BallTrack:
    """Materialise ``keyframe_set`` into a dense ``BallTrack``.

    ``n_frames`` defaults to one past the last keyframe / segment frame.
    Frames not covered by any segment fall back to their keyframe world
    position (if any) or ``None`` (state ``"missing"``).
    """
    fps = float(keyframe_set.fps)
    total = _n_frames(keyframe_set, n_frames)
    kf_by_frame: dict[int, BallKeyframe] = {
        kf.frame: kf for kf in keyframe_set.keyframes
    }

    world: dict[int, np.ndarray | None] = {}
    state: dict[int, str] = {}
    flight_segments: list[FlightSegment] = []

    # Seed exact keyframe positions first so isolated keyframes survive.
    for kf in keyframe_set.keyframes:
        world[kf.frame] = _xyz(kf.world_xyz)

    for seg in keyframe_set.segments:
        kf0 = kf_by_frame.get(seg.start_frame)
        kf1 = kf_by_frame.get(seg.end_frame)
        p0 = _xyz(kf0.world_xyz) if kf0 else None
        p1 = _xyz(kf1.world_xyz) if kf1 else None
        flight = seg.kind in _FLIGHT_KINDS
        if seg.kind == "ballistic" and p0 is not None and p1 is not None:
            fs = _eval_ballistic(seg, p0, p1, fps, world)
            if fs is not None:
                flight_segments.append(fs)
        elif p0 is not None and p1 is not None and seg.kind in (
            "roll", "carry", "free_flight",
        ):
            _eval_linear(seg, p0, p1, world)
        elif seg.kind == "rest" and (p0 is not None or p1 is not None):
            # Hold a constant position. Works from whichever endpoint has a
            # keyframe so clip-boundary holds (open start has only the end
            # keyframe; open end has only the start keyframe) both resolve.
            anchor = p0 if p0 is not None else p1
            for f in range(seg.start_frame, seg.end_frame + 1):
                world[f] = anchor
        else:
            # Underdetermined (e.g. free_flight with an unknown endpoint):
            # interior frames have no world position.
            for f in range(seg.start_frame + 1, seg.end_frame):
                world.setdefault(f, None)
            if p0 is not None:
                world[seg.start_frame] = p0
            if p1 is not None:
                world[seg.end_frame] = p1
        for f in range(seg.start_frame, seg.end_frame + 1):
            state[f] = "flight" if flight else "grounded"

    frames: list[BallFrame] = []
    seg_id_by_frame: dict[int, int] = {}
    for fs in flight_segments:
        for f in range(fs.frame_range[0], fs.frame_range[1] + 1):
            seg_id_by_frame[f] = fs.id

    for f in range(total):
        w = world.get(f)
        st = state.get(f)
        if w is not None:
            frames.append(BallFrame(
                frame=f,
                world_xyz=(float(w[0]), float(w[1]), float(w[2])),
                state=st or "grounded",
                confidence=1.0,
                flight_segment_id=seg_id_by_frame.get(f),
            ))
        else:
            frames.append(BallFrame(
                frame=f,
                world_xyz=None,
                state="flight" if st == "flight" else "missing",
                confidence=0.0,
                flight_segment_id=seg_id_by_frame.get(f),
            ))

    quat_by_frame = integrate_orientation(
        frames, tuple(flight_segments), fps, ball_radius_m,
    )
    frames = [
        dataclasses.replace(bf, quat_wxyz=quat_by_frame.get(bf.frame))
        for bf in frames
    ]

    return BallTrack(
        clip_id=keyframe_set.clip_id,
        fps=fps,
        frames=tuple(frames),
        flight_segments=tuple(flight_segments),
    )


def derived_world_by_frame(
    track: BallTrack,
) -> dict[int, tuple[tuple[float, float, float], float]]:
    """Convenience: ``{frame: (world_xyz, confidence)}`` for frames that
    have a resolved position (used to feed the stage's existing emit path)."""
    out: dict[int, tuple[tuple[float, float, float], float]] = {}
    for f in track.frames:
        if f.world_xyz is not None:
            out[f.frame] = (f.world_xyz, f.confidence)
    return out


__all__ = ["interpolate_events", "derived_world_by_frame"]
