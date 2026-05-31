"""C3a: flight spans with <2 hard knots are flagged."""
from __future__ import annotations

from src.stages.ball import _underconstrained_spans


def test_flags_zero_and_one_knot_spans_only():
    # spans: list of (fa, fb, n_hard_knots)
    spans = [(10, 25, 2), (40, 55, 1), (70, 85, 0)]
    flagged = _underconstrained_spans(spans, min_hard_knots=2)
    assert (40, 55, 1) in flagged
    assert (70, 85, 0) in flagged
    assert (10, 25, 2) not in flagged
    assert len(flagged) == 2


def test_empty_input():
    assert _underconstrained_spans([], min_hard_knots=2) == []
