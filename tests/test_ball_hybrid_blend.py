"""Tests for src/utils/ball_hybrid_blend.py.

Ported from prototypes/ball_hybrid_poc/tests/test_hybrid.py's blend group
plus a bit-for-bit cross-check against the PoC module it was ported from
(CONTRACT.md's porting requirement).
"""

from __future__ import annotations

import numpy as np
import pytest

from prototypes.ball_hybrid_poc import blend as poc_blend
from src.utils import ball_hybrid_blend as blend


# ---------------------------------------------------------------------------
# Ported behavioural tests
# ---------------------------------------------------------------------------

def test_smooth_kernel_decays_and_halves():
    assert blend.smooth_kernel(0.0, 5.0) == 1.0
    assert blend.smooth_kernel(5.0, 5.0) == pytest.approx(0.5)
    assert blend.smooth_kernel(10.0, 5.0) == pytest.approx(0.0625)
    assert blend.smooth_kernel(-5.0, 5.0) == pytest.approx(0.5)  # symmetric


def test_clamp_delta_rate_bounds_step_and_passes_through_conf():
    frames = list(range(6))
    blended = {
        0: ((0.0, 0.0, 0.0), 0.9),
        1: ((0.0, 0.0, 0.0), 0.9),
        2: ((0.0, 0.0, 0.0), 0.9),
        3: ((1.0, 0.0, 0.0), 0.8),
        4: ((1.0, 0.0, 0.0), 0.8),
        5: ((1.0, 0.0, 0.0), 0.8),
    }
    out = blend.clamp_delta_rate(frames, blended, max_step_m=0.1)
    assert out[3][1] == 0.8
    for f in range(1, 6):
        step = np.linalg.norm(np.array(out[f][0]) - np.array(out[f - 1][0]))
        assert step <= 0.1 + 1e-9
    assert out[5][0][0] == pytest.approx(0.3, abs=1e-6)
    assert out[5][0][0] > out[4][0][0] > out[3][0][0] > 0.0


def test_clamp_delta_rate_respects_event_walls():
    frames = list(range(4))
    blended = {0: ((0.0, 0.0, 0.0), 1.0), 1: ((5.0, 0.0, 0.0), 1.0),
               2: ((0.0, 0.0, 0.0), 1.0), 3: ((0.0, 0.0, 0.0), 1.0)}
    out = blend.clamp_delta_rate(frames, blended, max_step_m=0.5, event_frames=[1])
    assert out[2][0] == (0.0, 0.0, 0.0)


def test_segment_frames_splits_at_events():
    segs = blend.segment_frames(list(range(0, 10)), event_frames=[3, 6])
    assert segs == [[0, 1, 2, 3], [4, 5, 6], [7, 8, 9]]


def test_blend_deltas_exact_at_evidence_and_decays_away():
    frames = list(range(0, 61))
    evidence = {30: ((1.0, 0.0, 0.0), 1.0)}
    out = blend.blend_deltas(frames, evidence, halflife_frames=5.0)
    d30, conf30 = out[30]
    assert conf30 == pytest.approx(1.0)
    assert d30[0] == pytest.approx(1.0, abs=1e-6)
    d_far, conf_far = out[0]
    assert conf_far < 0.02
    assert abs(d_far[0]) < 0.02


def test_blend_deltas_never_crosses_an_event():
    frames = list(range(0, 21))
    evidence = {5: ((1.0, 0.0, 0.0), 1.0)}
    out_with_event = blend.blend_deltas(frames, evidence, halflife_frames=50.0,
                                         event_frames=[10])
    out_no_event = blend.blend_deltas(frames, evidence, halflife_frames=50.0)
    assert out_with_event[15][1] == 0.0
    assert out_no_event[15][1] > 0.0


def test_blend_has_no_jitter_spikes_on_noisy_input():
    rng = np.random.default_rng(3)
    n = 200
    frames = list(range(n))
    noise = rng.normal(scale=0.05, size=n)
    evidence = {f: ((float(noise[f]), 0.0, 0.0), 1.0) for f in frames}
    out = blend.blend_deltas(frames, evidence, halflife_frames=6.0)
    smoothed = np.array([out[f][0][0] for f in frames])

    def third_diff(x):
        return np.diff(x, n=3)

    raw_td = third_diff(noise)
    smoothed_td = third_diff(smoothed)
    assert np.max(np.abs(smoothed_td)) < 0.3 * np.max(np.abs(raw_td))


# ---------------------------------------------------------------------------
# Bit-for-bit parity with the PoC module
# ---------------------------------------------------------------------------

def test_blend_deltas_matches_poc_bit_for_bit():
    rng = np.random.default_rng(11)
    n = 80
    frames = list(range(n))
    evidence = {f: ((float(rng.normal()), float(rng.normal()), float(rng.normal())),
                     float(rng.uniform(0.1, 1.0)))
                for f in range(0, n, 3)}
    events = [20, 55]
    a = blend.blend_deltas(frames, evidence, halflife_frames=5.0, event_frames=events)
    b = poc_blend.blend_deltas(frames, evidence, halflife_frames=5.0, event_frames=events)
    assert a.keys() == b.keys()
    for f in frames:
        assert a[f][0] == pytest.approx(b[f][0], abs=1e-9)
        assert a[f][1] == pytest.approx(b[f][1], abs=1e-9)


def test_clamp_delta_rate_matches_poc_bit_for_bit():
    frames = list(range(10))
    blended = {f: ((float(f) * 0.3, 0.0, 0.0), 0.5) for f in frames}
    a = blend.clamp_delta_rate(frames, blended, max_step_m=0.1, event_frames=[5])
    b = poc_blend.clamp_delta_rate(frames, blended, max_step_m=0.1, event_frames=[5])
    assert a == b
