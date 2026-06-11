"""Per-shot features: pitch ratio, fades, motion rate, speed factor."""
from pathlib import Path

import numpy as np
import pytest

from src.utils.shot_features import (
    classify_kind,
    classify_scale,
    compute_span_features,
    estimate_speed_factors,
    is_replay,
    pitch_ratio,
)
from src.utils.shot_split import ShotSpan
from tests.fixtures.synthetic_reel import FPS, build_reel


def _spans_from(info: dict) -> list[ShotSpan]:
    return [
        ShotSpan(s["start_frame"], s["end_frame"],
                 s["start_frame"] / FPS, (s["end_frame"] + 1) / FPS)
        for s in info["spans"]
    ]


def test_pitch_ratio_green_vs_grey():
    green = np.zeros((40, 40, 3), np.uint8)
    green[:, :] = (40, 140, 60)
    grey = np.full((40, 40, 3), 110, np.uint8)
    assert pitch_ratio(green) > 0.8
    assert pitch_ratio(grey) < 0.1


def test_compute_span_features_classifies_reaction(tmp_path: Path):
    reel = tmp_path / "reel.mp4"
    info = build_reel(reel, [("green", 2.0), ("crowd", 2.0)])
    feats = compute_span_features(reel, _spans_from(info),
                                  sample_points=[0.2, 0.5, 0.8])
    assert classify_kind(feats[0]) == "gameplay"
    assert classify_kind(feats[1]) == "reaction"


def test_fade_classified_as_transition(tmp_path: Path):
    reel = tmp_path / "reel.mp4"
    info = build_reel(reel, [("black", 1.5)])
    feats = compute_span_features(reel, _spans_from(info),
                                  sample_points=[0.2, 0.5, 0.8])
    assert classify_kind(feats[0]) == "transition"


def test_scale_bands():
    class F:  # minimal stand-in: classify_scale reads pitch_ratio_median
        def __init__(self, m):
            self.pitch_ratio_median = m

    assert classify_scale(F(0.6)) == "wide"
    assert classify_scale(F(0.3)) == "medium"
    assert classify_scale(F(0.1)) == "tight"


def test_speed_factor_slow_clip_above_one(tmp_path: Path):
    reel = tmp_path / "reel.mp4"
    info = build_reel(reel, [("green", 3.0), ("green_slow", 3.0)])
    feats = compute_span_features(reel, _spans_from(info),
                                  sample_points=[0.2, 0.5, 0.8])
    feats = estimate_speed_factors(feats)
    assert feats[0].speed_factor == pytest.approx(1.0, abs=0.25)
    assert feats[1].speed_factor > 1.4
    assert not is_replay(feats[0])
    assert is_replay(feats[1])


def test_speed_factor_static_scene_defaults_to_one(tmp_path: Path):
    reel = tmp_path / "reel.mp4"
    info = build_reel(reel, [("green", 2.0), ("black", 2.0)])
    feats = compute_span_features(reel, _spans_from(info),
                                  sample_points=[0.2, 0.5, 0.8])
    feats = estimate_speed_factors(feats)
    # black span has ~zero texture and flow -> guard returns 1.0
    assert feats[1].speed_factor == pytest.approx(1.0)
