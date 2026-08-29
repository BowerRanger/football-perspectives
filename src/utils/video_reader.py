"""Sequential-friendly video frame reading.

``cv2.VideoCapture.set(CAP_PROP_POS_FRAMES, i)`` performs a real seek on
most backends/codecs (nearest-keyframe seek, then forward decode within the
GOP), so calling it before every single ``read()`` in a loop that mostly
walks forward is far more expensive than reading sequentially. The camera
stage had a dozen such per-frame-seek loops. These helpers replace that
pattern with a single capture + a tracked decode cursor: nearby frames are
reached with ``grab()`` (decode-and-discard, cheaper than a seek) and only
requests that jump far ahead or backward force an explicit seek.

Two forms are provided:

- :func:`read_frames` — batch API for "give me these frame indices as a
  dict", used where a call site collects frames into a dict before further
  processing (e.g. ``frames_bgr[i] = frame`` loops).
- :class:`VideoFrameReader` — incremental/cached form for call sites that
  already keep a local ``_cache: dict[int, frame | None]`` + ``_read(i)``
  closure and pull frames on demand, sometimes for the same index more than
  once, in a not-strictly-monotonic order (e.g. a down-then-up neighbour
  sweep). Wraps exactly that pattern.

Neither form changes which frames are read, their order as seen by the
caller, or decoded pixel content — only how the decoder gets positioned.

This claim has been probed directly (not just assumed) against a real
broadcast-clip stream: origi01.mp4 (h264, yuv420p, 1920x1080, 30fps,
506 frames, GOP=29, B-frame-heavy — the actual clip that surfaced a real,
deterministic camera-track divergence between the old per-frame-seek
camera.py code and this module). Thousands of probe cases — every forward
gap from 1 to 40 frames from cursor positions spanning the whole clip,
and the exact non-monotonic down-then-up neighbour-sweep access pattern
the camera stage's propagate-coverage loop (``_run_propagation`` in
``src/stages/camera.py``) performs — found **zero** content mismatches
between ``cap.set(CAP_PROP_POS_FRAMES, i); cap.read()`` (the old pattern),
this module's grab()-skip/seek logic, and a continuous no-seek decode.
``tests/test_video_reader.py::TestRealH264Fixture`` pins this down as a
regression test on a real (if small) libx264/B-frame/GOP fixture. In
short: for this codebase's actual clips and access patterns,
``CAP_PROP_POS_FRAMES`` seeking has NOT been shown to be imprecise, and
the invariant above holds as tested — but it was reached by direct
measurement on one real clip's specific codec parameters, not a general
guarantee about all H.264/ffmpeg/OpenCV-backend combinations, so treat it
as "verified for the clips this pipeline actually processes" rather than
a universal claim. The origi01 track divergence that prompted this probe
was RESOLVED (2026-08-25) as unrelated to this module: baseline runs from
a git worktree silently lost the PnLCalib detection bootstrap to a
sys.path shadowing bug (see ``_paths_shadowing_pnlcalib`` in
``src/utils/neural_calibrator.py``); with the environment held equal, the
old per-frame-seek camera stage and this module produce byte-identical
camera tracks on origi01, gberch, and origi02.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

# Frames to skip via grab() before falling back to an explicit seek. Chosen
# well above typical camera-stage access gaps (a few frames to a couple
# dozen) so common patterns never seek, while still bounding the cost of a
# single grab-skip run when a request does jump far ahead.
DEFAULT_SKIP_THRESHOLD = 32


def _is_capture_like(obj) -> bool:
    """True for a cv2.VideoCapture or anything duck-typing its read/set/grab
    surface (e.g. a counting wrapper in tests) — as opposed to a path."""
    return (
        hasattr(obj, "read") and hasattr(obj, "set") and hasattr(obj, "grab")
    )


class VideoFrameReader:
    """Sequential-friendly random-access reader over one video capture.

    Wraps a ``cv2.VideoCapture`` (or opens one from a path) and tracks the
    decode cursor itself. Requesting frame ``i``:

    - if it is already cached (previously read), returns the cached result
      with no capture access at all;
    - else, if ``i`` is within ``skip_threshold`` frames ahead of the
      cursor, walks forward with ``grab()`` then decodes ``i`` with
      ``read()``;
    - otherwise (backward request, or a gap beyond ``skip_threshold``)
      issues an explicit ``set(CAP_PROP_POS_FRAMES, i)`` seek.

    The grab-skip gap is measured against the capture's ACTUAL reported
    position (``get(CAP_PROP_POS_FRAMES)``), not private bookkeeping, so a
    capture shared with other readers or with legacy per-frame ``set``+
    ``read`` call sites stays correct: if someone else moved the decoder
    between two of our reads, we see the real position and reseek instead
    of silently grab-skipping from a stale one. The private cursor is kept
    only as a fallback for capture-like objects without ``get``.

    Not thread-safe: no locking around the shared decode cursor.
    """

    def __init__(
        self,
        path_or_cap,
        skip_threshold: int = DEFAULT_SKIP_THRESHOLD,
    ) -> None:
        self._owns_cap = not _is_capture_like(path_or_cap)
        self._cap = (
            cv2.VideoCapture(str(path_or_cap)) if self._owns_cap
            else path_or_cap
        )
        self._skip_threshold = skip_threshold
        # Index of the last frame successfully decoded; -1 means "decoder
        # position unknown" (unread, or recovering from a failed read) and
        # always forces an explicit seek on the next request.
        self._cursor = -1
        self._cache: dict[int, np.ndarray | None] = {}

    @property
    def cap(self):
        return self._cap

    def read(self, index: int) -> np.ndarray | None:
        """Return frame ``index`` as a BGR array, or None if unreadable
        (e.g. past EOF). Cached, so repeat requests for the same index
        never touch the capture again."""
        if index in self._cache:
            return self._cache[index]
        frame = self._read_uncached(index)
        self._cache[index] = frame
        return frame

    def _read_uncached(self, index: int) -> np.ndarray | None:
        if index < 0:
            return None
        pos = self._next_decode_index()
        gap = index - pos if pos is not None else -1  # frames to skip via grab()
        if gap < 0 or gap > self._skip_threshold:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            gap = 0
        for _ in range(gap):
            if not self._cap.grab():
                # Ran into EOF (or a decode error) while skipping —
                # the target is unreachable and the decoder position is
                # no longer trustworthy; force a reseek next time.
                self._cursor = -1
                return None
        ok, frame = self._cap.read()
        self._cursor = index if ok else -1
        return frame if ok else None

    def _next_decode_index(self) -> int | None:
        """Index of the next frame the capture would decode, or None when
        unknown (which forces an explicit seek). Prefers the capture's own
        reported position — the shared-capture safety property above —
        falling back to the private cursor for capture-like objects
        without ``get``."""
        get = getattr(self._cap, "get", None)
        if get is not None:
            pos = get(cv2.CAP_PROP_POS_FRAMES)
            if pos is not None and pos >= 0:
                return int(pos)
        return self._cursor + 1 if self._cursor >= 0 else None

    def close(self) -> None:
        if self._owns_cap:
            self._cap.release()

    def __enter__(self) -> "VideoFrameReader":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


def read_frames(path_or_cap, indices: Iterable[int]) -> dict[int, np.ndarray]:
    """Read a batch of frame indices from a video efficiently.

    Opens ``path_or_cap`` once (or reuses a supplied capture-like object,
    which is left open and positioned wherever the last read landed — the
    caller keeps ownership), sorts + dedupes ``indices``, and reads them in
    a single forward sweep via :class:`VideoFrameReader` (grab()-skip for
    small gaps, explicit seek for large ones) instead of a
    ``set(CAP_PROP_POS_FRAMES)`` seek per frame.

    Returns ``{index: frame}`` for every index that read successfully;
    indices past EOF or otherwise unreadable are simply absent — the same
    contract as the naive per-frame ``cap.set`` + ``cap.read`` loop it
    replaces.
    """
    ordered = sorted({int(i) for i in indices})
    out: dict[int, np.ndarray] = {}
    if not ordered:
        return out
    with VideoFrameReader(path_or_cap) as reader:
        for idx in ordered:
            frame = reader.read(idx)
            if frame is not None:
                out[idx] = frame
    return out
