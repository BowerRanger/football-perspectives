"""Pure functions for the hybrid extractor's broadcast/physics blend.

The hybrid track is ``P_phys(frame) + delta_smoothed(frame)``, where
``delta`` at an evidence frame is ``faithful_point - P_phys`` (how far the
physics-only prediction is from the point on a confident detection's ray
closest to it). This module owns only the *smoothing* of that delta: an
exponentially-decaying weighted average that is exact at evidence (full
weight there), decays to zero within a few halflives of the nearest
evidence, and never smooths across a hard event frame (bounce/touch),
since those are supposed to bend the path sharply.

No numpy typing beyond plain arrays/dicts; no camera or IO dependency —
this is exercised directly by ``tests/test_hybrid.py`` on synthetic
scalar/vector series.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import numpy as np

Vec3 = tuple[float, float, float]


def exponential_kernel(dt: float, halflife_frames: float) -> float:
    """Weight of an evidence point ``dt`` frames away, decaying by half
    every ``halflife_frames``. ``halflife_frames <= 0`` degenerates to a
    delta function (only ``dt == 0`` contributes)."""
    if halflife_frames <= 0:
        return 1.0 if dt == 0 else 0.0
    return 0.5 ** (abs(dt) / halflife_frames)


def segment_frames(frames: Sequence[int], event_frames: Iterable[int]) -> list[list[int]]:
    """Split ``frames`` (sorted) into contiguous segments that never cross
    an event frame: each event frame ends its segment, so evidence before
    an event cannot smear into frames after it (and vice versa)."""
    events = set(int(e) for e in event_frames)
    segments: list[list[int]] = []
    current: list[int] = []
    for f in frames:
        current.append(f)
        if f in events:
            segments.append(current)
            current = []
    if current:
        segments.append(current)
    return segments


def blend_deltas(
    frames: Sequence[int],
    evidence: Mapping[int, tuple[Vec3, float]],
    *,
    halflife_frames: float,
    event_frames: Iterable[int] = (),
    window_halflives: float = 6.0,
) -> dict[int, tuple[Vec3, float]]:
    """Smooth per-frame evidence deltas.

    ``evidence`` maps a frame to ``(delta_xyz, weight)`` (weight is the
    detection confidence, or a large constant for a hard ray anchor).
    Returns, for every frame in ``frames``, ``(smoothed_delta, conf)``
    where ``conf`` in ``[0, 1]`` is how much evidence-backed weight
    landed on this frame (1.0 at/near a confident or hard evidence
    point, decaying to 0.0 a few halflives away or across an event).

    The kernel is *not* renormalised to sum to 1 across evidence: a
    frame far from any evidence has a small total weight and its
    average is scaled down by that same total (clamped to 1), which is
    what makes delta actually decay to zero in a gap rather than just
    interpolating between distant evidence.
    """
    sorted_frames = sorted(int(f) for f in frames)
    segments = segment_frames(sorted_frames, event_frames)

    if halflife_frames > 0:
        window = max(1, int(round(halflife_frames * window_halflives)))
    else:
        window = 0

    out: dict[int, tuple[Vec3, float]] = {}
    for seg in segments:
        seg_set = set(seg)
        ev_frames_list = sorted(f for f in evidence if f in seg_set)
        if not ev_frames_list:
            for f in seg:
                out[f] = ((0.0, 0.0, 0.0), 0.0)
            continue
        ev_frames = np.array(ev_frames_list, dtype=float)
        ev_deltas = np.array([evidence[f][0] for f in ev_frames_list], dtype=float)
        ev_weights = np.array([evidence[f][1] for f in ev_frames_list], dtype=float)

        for f in seg:
            dts = ev_frames - f
            if window > 0:
                mask = np.abs(dts) <= window
            else:
                mask = dts == 0
            if not mask.any():
                out[f] = ((0.0, 0.0, 0.0), 0.0)
                continue
            kernel = np.array([exponential_kernel(float(d), halflife_frames)
                                for d in dts[mask]])
            w = ev_weights[mask] * kernel
            wsum = float(w.sum())
            if wsum <= 1e-9:
                out[f] = ((0.0, 0.0, 0.0), 0.0)
                continue
            avg = (w[:, None] * ev_deltas[mask]).sum(axis=0) / wsum
            conf = min(1.0, wsum)
            smoothed = tuple(float(x) for x in (avg * conf))
            out[f] = (smoothed, conf)
    return out
