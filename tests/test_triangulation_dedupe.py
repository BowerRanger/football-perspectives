"""Unit tests for src/utils/triangulation_dedupe.py."""

from __future__ import annotations

import numpy as np
import pytest

from src.schemas.triangulated import TriangulatedPlayer
from src.utils.triangulation_dedupe import deduplicate_players

_LEFT_HIP = 11
_RIGHT_HIP = 12


def _make_player(
    player_id: str,
    n_frames: int = 100,
    hip_x: float = 50.0,
    hip_y: float = 30.0,
    nan_frames: range | None = None,
    name: str = "",
) -> TriangulatedPlayer:
    """Build a synthetic TriangulatedPlayer with stationary hips at (hip_x, hip_y)."""
    positions = np.zeros((n_frames, 17, 3), dtype=np.float32)
    positions[:, _LEFT_HIP] = [hip_x - 0.1, hip_y, 0.95]
    positions[:, _RIGHT_HIP] = [hip_x + 0.1, hip_y, 0.95]
    # Fill other joints too so the array isn't all-zero (matches real data shape).
    for j in range(17):
        if j in (_LEFT_HIP, _RIGHT_HIP):
            continue
        positions[:, j] = [hip_x, hip_y, 1.5]
    if nan_frames is not None:
        positions[list(nan_frames), :, :] = np.nan
    confidences = np.full((n_frames, 17), 0.8, dtype=np.float32)
    if nan_frames is not None:
        confidences[list(nan_frames), :] = 0.0
    return TriangulatedPlayer(
        player_id=player_id,
        player_name=name,
        team="A",
        positions=positions,
        confidences=confidences,
        reprojection_errors=np.full((n_frames, 17), 1.0, dtype=np.float32),
        num_views=np.ones((n_frames, 17), dtype=np.int8),
        fps=25.0,
        start_frame=0,
    )


class TestDeduplicatePlayers:
    def test_empty_and_single_player_passthrough(self):
        assert deduplicate_players([]) == []
        single = [_make_player("P01")]
        result = deduplicate_players(single)
        assert len(result) == 1
        assert result[0].player_id == "P01"

    def test_keeps_well_separated_players(self):
        a = _make_player("P01", hip_x=20.0, hip_y=30.0)
        b = _make_player("P02", hip_x=80.0, hip_y=30.0)  # 60 m away
        result = deduplicate_players([a, b], distance_m=1.5)
        assert len(result) == 2
        assert {p.player_id for p in result} == {"P01", "P02"}

    def test_merges_co_located_duplicates(self):
        a = _make_player("P01", hip_x=50.0, hip_y=30.0)
        b = _make_player("P02", hip_x=50.05, hip_y=30.05)  # ~7 cm apart
        result = deduplicate_players([a, b], distance_m=1.5)
        assert len(result) == 1
        # Either id may survive depending on which has more valid frames;
        # they're tied here, so just check the merged record has full frames.
        merged = result[0]
        assert not np.any(np.isnan(merged.positions[:, _LEFT_HIP, 0]))

    def test_does_not_merge_when_overlap_too_small(self):
        # Both players claim hip_x=50, but only overlap on 5 frames
        a = _make_player("P01", n_frames=100, nan_frames=range(5, 100))
        b = _make_player("P02", n_frames=100, nan_frames=range(0, 95))
        result = deduplicate_players([a, b], distance_m=1.5, min_overlap_frames=30)
        assert len(result) == 2

    def test_merge_fills_nan_gaps_from_other_players(self):
        # P01 has frames 0–49, P02 has frames 50–99 (no overlap on hip).
        # Same location; we want fill-only behaviour requires overlap to
        # detect them as duplicates.  Use a small overlap of 30 frames in
        # the middle that's enough to trigger the merge, then verify the
        # gaps are filled.
        a = _make_player("P01", n_frames=100, nan_frames=range(60, 100))
        b = _make_player("P02", n_frames=100, nan_frames=range(0, 40))
        result = deduplicate_players([a, b], distance_m=1.5, min_overlap_frames=10)
        assert len(result) == 1
        merged = result[0]
        # Every frame should now have valid hip data.
        assert not np.any(np.isnan(merged.positions[:, _LEFT_HIP, 0]))

    def test_kept_player_has_more_valid_frames(self):
        a = _make_player("P01", n_frames=100, nan_frames=range(0, 80), name="A")  # 20 valid
        b = _make_player("P02", n_frames=100, nan_frames=range(0, 30), name="B")  # 70 valid
        result = deduplicate_players([a, b], distance_m=1.5, min_overlap_frames=15)
        assert len(result) == 1
        assert result[0].player_id == "P02"
        assert result[0].player_name == "B"

    def test_transitive_merge_chain(self):
        # Three players forming a chain: a≈b, b≈c, a not directly close to c
        # (within threshold), but union-find should merge all three.
        a = _make_player("P01", hip_x=50.0)
        b = _make_player("P02", hip_x=50.5)   # 0.5 m from a
        c = _make_player("P03", hip_x=51.0)   # 0.5 m from b, 1.0 m from a
        result = deduplicate_players([a, b, c], distance_m=0.8)
        assert len(result) == 1
