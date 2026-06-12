"""Second-pass integration: re-smoothing and the BallStage end-to-end run."""

from __future__ import annotations

import numpy as np
import pytest

from src.stages.ball import _build_tracker, _resmooth_observations


@pytest.mark.unit
def test_resmooth_keeps_raw_uv_and_fills_gaps():
    n = 30
    uv = {f: (100.0 + 5.0 * f, 400.0) for f in range(n)}
    uv[10] = None
    uv[11] = None
    steps = _resmooth_observations(uv, n, cfg={})
    assert len(steps) == n
    # Raw observations pass through exactly (raw-uv override rule).
    assert steps[5].uv == (125.0, 400.0)
    # Short gap is IMM-filled near the constant-velocity line.
    assert steps[10].uv is not None
    assert abs(steps[10].uv[0] - 150.0) < 5.0
    assert steps[10].is_gap_fill


@pytest.mark.unit
def test_build_tracker_honours_max_gap_override():
    tracker = _build_tracker({}, max_gap_frames=10 ** 6)
    for i in range(5):
        tracker.update(i, (100.0 + i, 400.0))
    last = None
    for i in range(5, 105):
        last = tracker.update(i, None)
    assert last.uv is not None  # would be None with the default max_gap
