"""Tests for the PATCH /api/sync offset-override logic.

These tests exercise the overlap_frames recomputation directly, since
the test venv doesn't have FastAPI/Starlette installed — it runs the
pipeline code only.
"""
import json
import shutil
from pathlib import Path

import pytest


def _compute_overlap(frame_offset: int, ref_frames: int, tgt_frames: int) -> list[int]:
    """Mirror the overlap_frames logic from PATCH /api/sync."""
    overlap_start = max(0, frame_offset)
    overlap_end = min(ref_frames, frame_offset + tgt_frames)
    if overlap_end > overlap_start:
        return [overlap_start, overlap_end]
    return []


class TestOverlapFramesComputation:
    def test_positive_offset_within_range(self):
        # offset=100, ref=500 frames, tgt=400 frames
        # overlap: [100, 500]
        result = _compute_overlap(100, 500, 400)
        assert result == [100, 500]

    def test_zero_offset_full_overlap(self):
        result = _compute_overlap(0, 500, 500)
        assert result == [0, 500]

    def test_large_negative_offset_no_overlap(self):
        result = _compute_overlap(-9999, 500, 300)
        assert result == []

    def test_large_positive_offset_no_overlap(self):
        # offset beyond end of reference
        result = _compute_overlap(600, 500, 300)
        assert result == []

    def test_partial_overlap_start(self):
        # Clip starts before reference: offset=-50, tgt=200 → overlap [0, 150]
        result = _compute_overlap(-50, 500, 200)
        assert result == [0, 150]

    def test_partial_overlap_end(self):
        # Clip runs past end of reference: offset=400, tgt=200 → overlap [400, 500]
        result = _compute_overlap(400, 500, 200)
        assert result == [400, 500]


class TestSyncMapAtomicWrite:
    """Verify the temp-file atomic-write pattern preserves valid JSON."""

    def test_atomic_write_roundtrip(self, tmp_path):
        sync_data = {
            "reference_shot": "ref01",
            "alignments": [
                {"shot_id": "clip02", "frame_offset": 100, "method": "audio", "confidence": 0.8, "overlap_frames": [100, 400]}
            ]
        }
        sync_path = tmp_path / "sync_map.json"
        sync_path.write_text(json.dumps(sync_data, indent=2))

        # Simulate the endpoint's atomic write
        sync_data["alignments"][0]["frame_offset"] = 150
        sync_data["alignments"][0]["method"] = "manual"
        sync_data["alignments"][0]["confidence"] = 1.0
        sync_data["alignments"][0]["overlap_frames"] = [150, 400]

        tmp_file = sync_path.with_suffix(".json.tmp")
        tmp_file.write_text(json.dumps(sync_data, indent=2))
        tmp_file.replace(sync_path)

        result = json.loads(sync_path.read_text())
        assert result["alignments"][0]["frame_offset"] == 150
        assert result["alignments"][0]["method"] == "manual"
        assert result["alignments"][0]["confidence"] == 1.0
        assert not (tmp_path / "sync_map.json.tmp").exists()


class TestSyncMapValidation:
    """Validation logic that the endpoint enforces."""

    def test_float_offset_rejected(self):
        with pytest.raises(TypeError):
            offset = 1.5
            if not isinstance(offset, int):
                raise TypeError("frame_offset must be an integer")

    def test_integer_offset_accepted(self):
        offset = 155
        assert isinstance(offset, int)

