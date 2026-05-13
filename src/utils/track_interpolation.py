"""Fill short detector gaps inside a single Track by linear bbox interp.

The tracking stage emits one TrackFrame per detected frame; gaps where
the detector missed a player (brief occlusion, motion blur) leave that
frame absent from the track. Downstream stages then have to paper over
the dropout with their own smoothing fallbacks — and ball player_touch
anchoring, which needs an exact SMPL frame at the contact frame, simply
misses the anchor when the contact frame is in a gap.

This module provides a pure post-pass that, for each consecutive pair
of frames inside a track separated by ``1 < step <= max_gap + 1``,
inserts linearly-interpolated bbox + confidence at every missing frame
in between. Inserted frames are tagged ``interpolated=True`` so quality
reports and the overlay can distinguish them from real detections.

Linear interpolation is sound only when player motion is roughly
constant across the gap. The caller is responsible for choosing a
``max_gap`` small enough that the assumption holds — sprint direction
changes inside the gap will produce wrong crops. The dashboard exposes
this as the ``Interp Missing Frames`` button so the operator chooses
which tracks to apply it to after manual review.
"""

from __future__ import annotations

from src.schemas.tracks import Track, TrackFrame


def interpolate_track_gaps(track: Track, max_gap: int) -> tuple[Track, int]:
    """Insert linearly-interpolated frames for short gaps inside ``track``.

    Args:
        track: A Track with frames sorted by ``frame`` index.
        max_gap: Largest gap (in missing frames) to fill. A gap of N
            means N frames were skipped between two known frames; e.g.
            frames 10 and 14 are a gap of 3 (frames 11, 12, 13 are
            missing). ``max_gap=0`` disables interpolation.

    Returns:
        A tuple ``(track, frames_added)``. The returned Track shares
        identity with the input — frames are mutated in place. ``track.frames``
        is replaced with a new list that includes the interpolated entries
        in order. Interpolated frames carry ``interpolated=True``;
        existing frames are unmodified.
    """
    if max_gap <= 0 or len(track.frames) < 2:
        return track, 0

    new_frames: list[TrackFrame] = [track.frames[0]]
    added = 0
    for prev, nxt in zip(track.frames, track.frames[1:]):
        gap = nxt.frame - prev.frame - 1
        if 0 < gap <= max_gap:
            for k in range(1, gap + 1):
                t = k / (gap + 1)
                new_frames.append(_lerp_frame(prev, nxt, t))
                added += 1
        new_frames.append(nxt)
    track.frames = new_frames
    return track, added


def _lerp_frame(a: TrackFrame, b: TrackFrame, t: float) -> TrackFrame:
    """Linearly interpolate one frame between ``a`` and ``b`` at parameter ``t`` ∈ (0, 1)."""
    bbox = [a.bbox[i] + (b.bbox[i] - a.bbox[i]) * t for i in range(4)]
    return TrackFrame(
        frame=a.frame + round((b.frame - a.frame) * t),
        bbox=bbox,
        confidence=a.confidence + (b.confidence - a.confidence) * t,
        pitch_position=None,
        interpolated=True,
    )
