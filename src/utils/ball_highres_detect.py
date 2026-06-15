"""High-resolution ball detection: zoom-refine + tiled-relocate over a base
detector.

The base WASB detector letterboxes the whole frame to 512x288, so a ~5-10px
broadcast ball becomes ~2-4px and is barely detectable. Both mechanisms here
feed the detector a *crop* instead, which its letterbox then upscales — the
ball lands at a usable size:

  * ``refine(frame, center)`` — crop a ``zoom_crop_px`` window around a
    predicted ball location and re-detect (the primary per-frame path once
    the ball is located). Mirrors ``ball_second_pass`` zoom semantics.
  * ``relocate(frame)`` — when the ball is lost (cold start / long gap),
    sweep the frame as overlapping full-res tiles and merge candidates.

Pure-Python orchestration: the detector is injected (anything with
``detect_candidates(frame, min_score, top_k)``), so this is torch-free and
unit-testable with a fake. See
docs/superpowers/specs/2026-06-15-ball-detection-direction-changes-design.md §4.
"""

from __future__ import annotations

import numpy as np

Candidate = tuple[float, float, float]  # (u, v, score) in full-frame pixels


def map_crop_candidates(
    candidates: list[Candidate], x0: int, y0: int
) -> list[Candidate]:
    """Offset crop-local candidates back to full-frame coordinates."""
    return [(float(u) + x0, float(v) + y0, float(s)) for u, v, s in candidates]


def tile_windows(
    w: int, h: int, tile: int, overlap: int
) -> list[tuple[int, int, int]]:
    """Overlapping ``(x0, y0, tile)`` windows covering a ``w x h`` frame.

    Stride is ``tile - overlap``; the last window in each axis is clamped to
    the edge so the far border is always covered.
    """
    stride = max(1, tile - overlap)

    def _starts(extent: int) -> list[int]:
        if extent <= tile:
            return [0]
        starts = list(range(0, extent - tile + 1, stride))
        if starts[-1] != extent - tile:
            starts.append(extent - tile)
        return starts

    return [(x0, y0, tile) for y0 in _starts(h) for x0 in _starts(w)]


def _crop(frame: np.ndarray, x0: int, y0: int, size: int) -> np.ndarray:
    return frame[y0:y0 + size, x0:x0 + size]


class HighResDetector:
    """Wrap a base detector with zoom-refine + tiled-relocate."""

    def __init__(
        self,
        base,
        *,
        zoom_crop_px: int = 320,
        tile: int = 416,
        overlap: int = 96,
        top_k: int = 5,
        min_score: float = 0.05,
    ) -> None:
        self._base = base
        self._zoom = int(zoom_crop_px)
        self._tile = int(tile)
        self._overlap = int(overlap)
        self._top_k = int(top_k)
        self._min_score = float(min_score)

    def refine(
        self, frame: np.ndarray, center: tuple[float, float]
    ) -> Candidate | None:
        """Best candidate from a zoom crop centred on ``center``; None if
        nothing clears ``min_score``."""
        h, w = frame.shape[:2]
        half = self._zoom // 2
        x0 = int(np.clip(round(center[0]) - half, 0, max(0, w - self._zoom)))
        y0 = int(np.clip(round(center[1]) - half, 0, max(0, h - self._zoom)))
        crop = _crop(frame, x0, y0, self._zoom)
        if crop.shape[0] == 0 or crop.shape[1] == 0:
            return None
        cands = map_crop_candidates(
            self._base.detect_candidates(crop, self._min_score, self._top_k),
            x0, y0,
        )
        if not cands:
            return None
        return max(cands, key=lambda c: c[2])

    def relocate(self, frame: np.ndarray) -> list[Candidate]:
        """Top-K candidates from a tiled sweep (ball-lost recovery)."""
        h, w = frame.shape[:2]
        merged: list[Candidate] = []
        for x0, y0, t in tile_windows(w, h, self._tile, self._overlap):
            crop = _crop(frame, x0, y0, t)
            if crop.shape[0] == 0 or crop.shape[1] == 0:
                continue
            merged.extend(map_crop_candidates(
                self._base.detect_candidates(crop, self._min_score, self._top_k),
                x0, y0,
            ))
        merged.sort(key=lambda c: c[2], reverse=True)
        return merged[: self._top_k]


__all__ = ["HighResDetector", "map_crop_candidates", "tile_windows", "Candidate"]
