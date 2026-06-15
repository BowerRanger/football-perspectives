"""High-res detection geometry (pure) + zoom/tile wrapper over a base
detector (torch-free via an injected fake)."""

from __future__ import annotations

import numpy as np

from src.utils.ball_highres_detect import (
    HighResDetector,
    map_crop_candidates,
    tile_windows,
)


def test_map_crop_candidates_offsets_to_full_frame():
    assert map_crop_candidates([(10.0, 20.0, 0.9)], x0=100, y0=50) == [
        (110.0, 70.0, 0.9)
    ]


def test_tile_windows_cover_frame_with_overlap():
    wins = tile_windows(w=1000, h=600, tile=400, overlap=100)
    assert all(t == 400 for _, _, t in wins)
    xs = [x for x, _, _ in wins]
    ys = [y for _, y, _ in wins]
    assert min(xs) == 0 and min(ys) == 0
    assert max(xs) + 400 >= 1000  # right edge covered
    assert max(ys) + 400 >= 600   # bottom edge covered


class _FakeDetector:
    """Models the real zoom benefit: the detector letterboxes its input to a
    fixed size, so a *smaller* crop makes the ball bigger and easier to see.
    Score therefore rises as the crop shrinks (inverse of crop extent)."""

    def detect_candidates(self, frame, min_score, top_k=5):
        h, w = frame.shape[:2]
        score = min(0.95, 320.0 / max(w, h))  # full frame -> low; zoom -> high
        return [(w / 2.0, h / 2.0, score)]


def test_zoom_refine_raises_confidence():
    base = _FakeDetector()
    hr = HighResDetector(base, zoom_crop_px=320)
    frame = np.zeros((720, 1280, 3), np.uint8)
    coarse = base.detect_candidates(frame, 0.05)[0]
    refined = hr.refine(frame, center=(640.0, 360.0))
    assert refined is not None
    # refined detection sits near the requested centre and is more confident
    assert refined[2] > coarse[2]
    assert abs(refined[0] - 640.0) <= 320 and abs(refined[1] - 360.0) <= 320


def test_relocate_tiles_and_returns_candidates():
    base = _FakeDetector()
    hr = HighResDetector(base, tile=400, overlap=100, top_k=5)
    frame = np.zeros((600, 1000, 3), np.uint8)
    cands = hr.relocate(frame)
    assert cands  # at least one candidate from the tile sweep
    # candidates are in full-frame coordinates
    assert all(0 <= u <= 1000 and 0 <= v <= 600 for u, v, _ in cands)
