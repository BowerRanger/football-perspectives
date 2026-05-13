"""Unit tests for SAHI tile-iteration, edge-clip filtering, and
cross-tile NMS-merge helpers in ``src.utils.player_detector``."""

from __future__ import annotations

import numpy as np
import pytest

from src.utils.player_detector import (
    Detection,
    _is_edge_clipped,
    _iter_tiles,
    _nms_merge_per_class,
)


# --- tile iteration ---------------------------------------------------------

def _tile_starts(frame_shape, tile, overlap):
    frame = np.zeros(frame_shape, dtype=np.uint8)
    return [(x, y, tile_arr.shape[:2]) for x, y, tile_arr in _iter_tiles(frame, tile, overlap)]


def test_tile_iteration_covers_full_1080p_frame_with_no_gaps():
    """3x2 grid on 1920x1080 with tile=960, overlap=0.25 (step=720).
    Tile starts in x: 0, 720, then anchored-to-edge at 960 (1920-960).
    Tile starts in y: 0, then anchored-to-edge at 120 (1080-960)."""
    starts = _tile_starts((1080, 1920, 3), tile=960, overlap=0.25)
    xs = sorted({x for x, _, _ in starts})
    ys = sorted({y for _, y, _ in starts})
    assert xs == [0, 720, 960]
    assert ys == [0, 120]
    # Every tile is full-sized.
    for _, _, (h, w) in starts:
        assert (h, w) == (960, 960)


def test_tile_iteration_handles_small_frames():
    """If the frame is smaller than the tile, a single tile covers it."""
    starts = _tile_starts((600, 800, 3), tile=960, overlap=0.25)
    assert len(starts) == 1
    (x, y, (h, w)) = starts[0]
    assert (x, y) == (0, 0)
    assert (h, w) == (600, 800)


# --- edge-clip detection ----------------------------------------------------

def test_edge_clipped_drops_box_touching_interior_tile_seam():
    # tile sits at (0,0) of a wider frame; the right edge of the tile
    # is interior (not the frame edge), so a box touching it gets
    # dropped — the neighbouring tile contains the full body.
    assert _is_edge_clipped(
        (50, 100, 960, 200),
        tile_x=0, tile_y=0,
        tile_w=960, tile_h=960,
        frame_w=1920, frame_h=1080,
    )


def test_edge_clipped_keeps_box_touching_frame_edge():
    # tile at (960,0) — its right edge (1920) IS the frame edge. A box
    # touching that edge is genuinely at the frame boundary, not a
    # tile clip; we keep it.
    assert not _is_edge_clipped(
        (1500, 100, 1920, 200),
        tile_x=960, tile_y=0,
        tile_w=960, tile_h=960,
        frame_w=1920, frame_h=1080,
    )


def test_edge_clipped_keeps_interior_box():
    assert not _is_edge_clipped(
        (100, 100, 200, 200),
        tile_x=0, tile_y=0,
        tile_w=960, tile_h=960,
        frame_w=1920, frame_h=1080,
    )


# --- per-class NMS merge ----------------------------------------------------

def test_nms_merge_keeps_higher_confidence_of_duplicate_pair():
    """Two near-identical boxes from neighbouring tiles — the higher-
    confidence one wins and the duplicate is suppressed."""
    dets = [
        Detection(bbox=(100, 100, 200, 200), confidence=0.6, class_name="player"),
        Detection(bbox=(102, 101, 198, 202), confidence=0.9, class_name="player"),
    ]
    merged = _nms_merge_per_class(dets, iou_threshold=0.5)
    assert len(merged) == 1
    assert merged[0].confidence == 0.9


def test_nms_merge_does_not_suppress_distinct_players():
    """Two players at different positions — both survive."""
    dets = [
        Detection(bbox=(100, 100, 150, 200), confidence=0.7, class_name="player"),
        Detection(bbox=(300, 100, 350, 200), confidence=0.8, class_name="player"),
    ]
    merged = _nms_merge_per_class(dets, iou_threshold=0.5)
    assert len(merged) == 2


def test_nms_merge_is_per_class():
    """A player and a ball at the same location aren't duplicates of
    each other — class-aware NMS keeps both."""
    dets = [
        Detection(bbox=(100, 100, 200, 200), confidence=0.7, class_name="player"),
        Detection(bbox=(100, 100, 200, 200), confidence=0.6, class_name="ball"),
    ]
    merged = _nms_merge_per_class(dets, iou_threshold=0.5)
    assert len(merged) == 2
    classes = sorted(d.class_name for d in merged)
    assert classes == ["ball", "player"]


def test_nms_merge_empty_list():
    assert _nms_merge_per_class([], iou_threshold=0.5) == []


def test_nms_merge_preserves_three_close_distinct_players():
    """Three players standing in a row, each with their own bbox —
    classic close-quarters scenario where NMS shouldn't suppress
    legitimate detections. Each pair has IoU well below threshold."""
    dets = [
        Detection(bbox=(100, 100, 150, 250), confidence=0.8, class_name="player"),
        Detection(bbox=(155, 100, 205, 250), confidence=0.7, class_name="player"),
        Detection(bbox=(210, 100, 260, 250), confidence=0.75, class_name="player"),
    ]
    merged = _nms_merge_per_class(dets, iou_threshold=0.5)
    assert len(merged) == 3
