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
    a = np.array([0.0, 0.0, gravity])
    fit_p0 = seg.hints.get("fit_p0")
    fit_v0 = seg.hints.get("fit_v0")
    if fit_p0 is not None and fit_v0 is not None:
        # A detection-fitted arc (W5l) overrides the naive two-endpoint
        # parabola: in-span rays constrained it, so soft/auto endpoints
        # no longer dictate the flight shape. The arc may disagree with
        # the endpoint keyframes (that is the point, for auto pins), so
        # blend the residual in over the outer quarters of the span —
        # endpoints stay exact without a teleport.
        base = np.asarray(fit_p0, dtype=float)
        vel = np.asarray(fit_v0, dtype=float)
        span = max(seg.end_frame - seg.start_frame, 1)
        ramp_n = max(1, span // 4)

        def _arc_at(f: int) -> np.ndarray:
            t = (f - seg.start_frame) / fps
            return base + vel * t + 0.5 * a * t * t

        err_start = p0 - _arc_at(seg.start_frame)
        err_end = p1 - _arc_at(seg.end_frame)
        for f in range(seg.start_frame, seg.end_frame + 1):
            w = _arc_at(f)
            ds = f - seg.start_frame
            de = seg.end_frame - f
            if ds < ramp_n:
                w = w + err_start * (1.0 - ds / ramp_n)
            if de < ramp_n:
                w = w + err_end * (1.0 - de / ramp_n)
            world[f] = w
        v0 = vel
    else:
        base = p0
        vel = _ballistic_v0(p0, p1, total_t, gravity)
        v0 = vel
        for f in range(seg.start_frame, seg.end_frame + 1):
            t = (f - seg.start_frame) / fps
            world[f] = base + vel * t + 0.5 * a * t * t
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


# A roll-evidence knot farther than this from the endpoints' chord is a
# false-detection cluster (detector locked on a static object), not a curved
# roll. The budget scales with span length: genuine curvature grows with
# the distance travelled (origi01's 60+-frame rolls carry 2 m of real
# sagitta) while false clusters sit 4–8 m off regardless of span.
_ROLL_KNOT_CHORD_DEV_BASE_M = 1.5
_ROLL_KNOT_CHORD_DEV_PER_FRAME_M = 0.02
_ROLL_KNOT_CHORD_DEV_CAP_M = 3.5


def _evidence_knots(
    seg: BallSegment,
    evidence_worlds: dict[int, np.ndarray],
    stride: int,
    window: int,
    p0: np.ndarray | None = None,
    p1: np.ndarray | None = None,
) -> list[tuple[int, np.ndarray]]:
    """Strided, jitter-filtered evidence knots strictly inside a span.

    One knot per ``stride`` frames; each knot is the symmetric-window mean
    of nearby evidence, and knots deviating more than
    ``_ROLL_KNOT_MAX_CHORD_DEV_M`` from the endpoint chord are rejected as
    false-detection clusters.
    """
    span = seg.end_frame - seg.start_frame
    max_dev = min(
        _ROLL_KNOT_CHORD_DEV_CAP_M,
        _ROLL_KNOT_CHORD_DEV_BASE_M
        + _ROLL_KNOT_CHORD_DEV_PER_FRAME_M * span,
    )

    def _chord_dev(f: int, w: np.ndarray) -> float:
        if p0 is None or p1 is None or span == 0:
            return 0.0
        s = (f - seg.start_frame) / span
        return float(np.linalg.norm(w - (p0 + (p1 - p0) * s)))

    inside = sorted(
        f for f in evidence_worlds
        if seg.start_frame < f < seg.end_frame
        and _chord_dev(f, np.asarray(evidence_worlds[f], dtype=float))
        <= max_dev
    )
    knots: list[tuple[int, np.ndarray]] = []
    last = None
    for f in inside:
        if last is not None and f - last < stride:
            continue
        # Symmetric window only: an unbalanced aggregate drags the knot
        # along the path near run edges; a lone unbalanced point would
        # pass its jitter straight through, so skip it entirely.
        lo = [g for g in inside if f - window <= g < f]
        hi = [g for g in inside if f < g <= f + window]
        k = min(len(lo), len(hi))
        if k == 0 and len(inside) > 1:
            continue
        group = [*lo[len(lo) - k:], f, *hi[:k]]
        near = [np.asarray(evidence_worlds[g], dtype=float) for g in group]
        # Mean (not median): detection jitter is zero-mean and outliers are
        # rejected upstream by the track cleaner.
        knots.append((f, np.mean(np.stack(near), axis=0)))
        last = f
    return knots


# Natural-motion limits mirrored from the validator (ball_eval
# NaturalnessCfg): a roll/carry may never turn or speed up faster than an
# event-free ball can. Rendering enforces them by construction.
_MAX_HEADING_DEG_PER_FRAME = 12.0
_MAX_SPEED_RATIO = 1.15
_MIN_LIMIT_SPEED_M_S = 1.0
_MAX_SMOOTH_ROUNDS = 12


def _span_violates_limits(vals: dict[int, np.ndarray], fps: float) -> bool:
    frames = sorted(vals)
    for f in frames[1:-1]:
        if f - 1 not in vals or f + 1 not in vals:
            continue
        v_in = (vals[f] - vals[f - 1]) * fps
        v_out = (vals[f + 1] - vals[f]) * fps
        sp_in = float(np.linalg.norm(v_in[:2]))
        sp_out = float(np.linalg.norm(v_out[:2]))
        if min(sp_in, sp_out) > 2.0:
            dh = float(np.degrees(abs(np.arctan2(v_out[1], v_out[0])
                                      - np.arctan2(v_in[1], v_in[0]))))
            dh = min(dh, 360.0 - dh)
            if dh > _MAX_HEADING_DEG_PER_FRAME - 1.0:
                return True
        if (min(sp_in, sp_out) > _MIN_LIMIT_SPEED_M_S
                and sp_out > sp_in * (_MAX_SPEED_RATIO - 0.02)):
            return True
    return False


def _smooth_span(vals: dict[int, np.ndarray], p0: np.ndarray,
                 p1: np.ndarray, fps: float, base_iters: int = 3) -> None:
    """Endpoint-preserving smoothing, repeated until the span satisfies
    the natural-motion limits (it converges toward the always-legal
    endpoint chord), then reparameterized to a lightly-smoothed version
    of the ORIGINAL arc schedule so knot timing and deceleration physics
    survive the geometric smoothing."""
    frames = sorted(vals)
    original_steps = (
        [float(np.linalg.norm(vals[b] - vals[a]))
         for a, b in zip(frames, frames[1:])]
        if len(frames) >= 3 else None
    )

    def _round() -> None:
        prev = dict(vals)
        for f in frames[1:-1]:
            vals[f] = 0.25 * prev[f - 1] + 0.5 * prev[f] + 0.25 * prev[f + 1]
        vals[frames[0]] = p0
        vals[frames[-1]] = p1

    for _ in range(base_iters):
        _round()
    rounds = 0
    while _span_violates_limits(vals, fps) and rounds < _MAX_SMOOTH_ROUNDS:
        _round()
        rounds += 1
    # Arc-length reparameterization: neighbour-averaging with pinned
    # endpoints redistributes arc length (slow near knots, catch-up
    # between), which MANUFACTURES roll_speedup violations (measured on
    # origi01: 1 violation raw vs 8 smoothed). Walk the smoothed geometry
    # on the ORIGINAL (knot-honouring) arc schedule with the speed
    # sequence lightly smoothed — constant-speed resampling was measured
    # to decouple evidence timing (+23 dense >20 cm frames), because real
    # rolls decelerate; the original schedule carries that physics.
    if len(frames) >= 3 and original_steps is not None:
        pts = [vals[f] for f in frames]
        seg_len = [float(np.linalg.norm(b - a))
                   for a, b in zip(pts, pts[1:])]
        total = sum(seg_len)
        # Linear-speed (constant-deceleration) schedule: least-squares
        # quadratic through the raw cumulative arc — averages knot jitter
        # out of the timing while keeping the span's true deceleration.
        raw_cum = [0.0]
        for L in original_steps:
            raw_cum.append(raw_cum[-1] + L)
        m = len(raw_cum)
        xs = np.arange(m, dtype=float)
        coef = np.polyfit(xs, np.asarray(raw_cum), 2)
        fitted = np.polyval(coef, xs)
        steps = [max(0.0, float(b - a))
                 for a, b in zip(fitted, fitted[1:])]
        ssum = sum(steps)
        if total > 1e-9 and ssum > 1e-9:
            schedule = [s * total / ssum for s in steps]
            cum = [0.0]
            for L in seg_len:
                cum.append(cum[-1] + L)
            resampled = [pts[0]]
            target = 0.0
            j = 0
            for st in schedule:
                target += st
                while j < len(seg_len) - 1 and cum[j + 1] < target:
                    j += 1
                den = max(cum[j + 1] - cum[j], 1e-12)
                s = (target - cum[j]) / den
                resampled.append(pts[j] + (pts[j + 1] - pts[j]) * s)
            resampled[-1] = pts[-1]
            assert len(resampled) == len(frames)
            for f, w in zip(frames, resampled):
                vals[f] = w
    vals[frames[0]] = p0
    vals[frames[-1]] = p1


def _eval_polyline(
    seg: BallSegment,
    p0: np.ndarray,
    p1: np.ndarray,
    knots: list[tuple[int, np.ndarray]],
    world: dict[int, np.ndarray | None],
    fps: float = 30.0,
) -> None:
    """Piecewise-linear render through endpoint-pinned evidence knots,
    smoothed (adaptively) until the span satisfies the natural-motion
    limits — knot kinks and jitter never reach the emitted track."""
    pts = [(seg.start_frame, p0), *knots, (seg.end_frame, p1)]
    vals: dict[int, np.ndarray] = {}
    for (fa, pa), (fb, pb) in zip(pts, pts[1:]):
        span = fb - fa
        for f in range(fa, fb + 1):
            s = 0.0 if span == 0 else (f - fa) / span
            vals[f] = pa + (pb - pa) * s
    _smooth_span(vals, p0, p1, fps)
    world.update(vals)


def interpolate_events(
    keyframe_set: BallKeyframeSet,
    *,
    n_frames: int | None = None,
    ball_radius_m: float = _BALL_RADIUS_M,
    evidence_worlds: dict[int, tuple[float, float, float]] | None = None,
    evidence_stride: int = 4,
    evidence_window: int = 2,
    carry_worlds: dict[int, tuple[float, float, float]] | None = None,
) -> BallTrack:
    """Materialise ``keyframe_set`` into a dense ``BallTrack``.

    ``n_frames`` defaults to one past the last keyframe / segment frame.
    Frames not covered by any segment fall back to their keyframe world
    position (if any) or ``None`` (state ``"missing"``).

    ``evidence_worlds`` (sub-20cm campaign W4) maps frames to real-evidence
    world positions (e.g. detection ray ∩ ground); ``roll`` spans containing
    such evidence render as an endpoint-pinned polyline through strided
    median knots instead of a straight line, so curved or decelerating
    ground passes follow what the detector actually saw.

    ``carry_worlds`` (W4b) maps interior frames of ``carry`` spans to
    precomputed ball positions that follow the owning player's foot path
    (endpoint-blended by the resolver); used verbatim, so a dribbler's
    curved run no longer renders as a straight chord.
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
        elif seg.kind == "free_flight" and p0 is not None and p1 is not None:
            # Two known endpoints under gravity ARE a ballistic span; the
            # historical linear fallback violated flight physics (W4 fix).
            fs = _eval_ballistic(seg, p0, p1, fps, world)
            if fs is not None:
                flight_segments.append(fs)
        elif p0 is not None and p1 is not None and seg.kind in (
            "roll", "carry",
        ):
            # NOTE: ``carry`` interpolates LINEARLY here as a deliberate P1
            # interim. The §10 design target is to follow the owning
            # player's foot/ground path; that needs the player context
            # threaded into the interpolator (a Phase-3 follow-up). Linear
            # between two nearby same-player touches (gap ≤15 fr, ≤3 m by
            # detect_carry_spans) is a reasonable approximation until then.
            if seg.kind == "carry" and carry_worlds and all(
                f in carry_worlds
                for f in range(seg.start_frame + 1, seg.end_frame)
            ):
                vals: dict[int, np.ndarray] = {
                    f: np.asarray(carry_worlds[f], dtype=float)
                    for f in range(seg.start_frame + 1, seg.end_frame)
                }
                vals[seg.start_frame] = p0
                vals[seg.end_frame] = p1
                # FK foot jitter at distant-zoom depth renders as heading
                # wobble; smooth to naturalness-clean like roll spans.
                _smooth_span(vals, p0, p1, fps)
                world.update(vals)
            else:
                knots = (
                    _evidence_knots(seg, evidence_worlds, evidence_stride,
                                    evidence_window, p0=p0, p1=p1)
                    if evidence_worlds and seg.kind == "roll" else []
                )
                if knots:
                    _eval_polyline(seg, p0, p1, knots, world, fps=fps)
                else:
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
