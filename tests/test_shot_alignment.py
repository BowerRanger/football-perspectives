"""Motion-energy curve alignment (NCC) for highlight groups."""
from pathlib import Path

import numpy as np

from src.utils.shot_alignment import (
    align_curves,
    align_group,
    motion_energy_curve,
)
from tests.fixtures.synthetic_reel import build_reel


def _pulse(n: int, at: int, width: int = 6) -> np.ndarray:
    x = np.zeros(n)
    lo, hi = max(0, at - width), min(n, at + width)
    window = np.hanning(2 * width)
    x[lo:hi] = window[: hi - lo]
    return x


def test_align_curves_recovers_known_lag():
    ref = _pulse(200, 60) + 0.05
    shifted = _pulse(200, 100) + 0.05  # same event, 40 frames later
    r = align_curves(ref, shifted, min_overlap=25)
    # offset = frame_in_shot - frame_in_reference
    assert r.frame_offset == 40
    assert r.confidence > 0.9
    assert r.method == "motion_profile"


def test_align_curves_negative_lag():
    ref = _pulse(200, 120) + 0.05
    shifted = _pulse(200, 80) + 0.05  # event 40 frames EARLIER in shot
    r = align_curves(ref, shifted, min_overlap=25)
    assert r.frame_offset == -40


def test_align_curves_flat_signal_low_confidence():
    r = align_curves(np.ones(100), np.ones(120), min_overlap=25)
    assert r.method == "low_confidence"
    # fallback aligns clip ends
    assert r.frame_offset == 20


def test_motion_energy_curve_has_energy(tmp_path: Path):
    clip = tmp_path / "c.mp4"
    build_reel(clip, [("green", 2.0)])
    curve = motion_energy_curve(clip, width_px=96)
    assert len(curve) >= 45
    assert float(np.max(curve)) > 0


def test_align_group_returns_reference_zero(tmp_path: Path):
    a = tmp_path / "a.mp4"
    b = tmp_path / "b.mp4"
    build_reel(a, [("green", 2.0)])
    build_reel(b, [("green", 2.0)])
    results = align_group({"a": a, "b": b}, reference_id="a",
                          width_px=96, smooth_sigma=2.0,
                          min_overlap_frames=12, min_confidence=0.5)
    assert results["a"].frame_offset == 0
    assert results["a"].confidence == 1.0
    assert "b" in results
