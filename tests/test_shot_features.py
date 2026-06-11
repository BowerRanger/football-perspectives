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


def test_long_span_with_dark_dip_is_not_a_transition():
    """Broadcast fades last ~a second. A 40 s gameplay span that happens
    to sample one dark frame (shadowed close-up, wipe graphic) must NOT
    be killed by the fade rule — that's how the Liverpool reel lost its
    entire first-goal sequence (s002, 4–48 s)."""
    from src.utils.shot_features import ShotFeatures
    from src.utils.shot_split import ShotSpan

    long_gameplay = ShotFeatures(
        span=ShotSpan(100, 1200, 4.0, 48.0),
        pitch_ratio_median=0.56, pitch_ratio_peak=0.7,
        brightness_min=0.12, brightness_range=0.4,
        motion_rate=0.05,
    )
    assert classify_kind(long_gameplay) == "gameplay"

    short_fade = ShotFeatures(
        span=ShotSpan(100, 130, 4.0, 5.2),
        pitch_ratio_median=0.2, pitch_ratio_peak=0.3,
        brightness_min=0.12, brightness_range=0.4,
        motion_rate=0.05,
    )
    assert classify_kind(short_fade) == "transition"


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


def test_person_dominant_shot_classified_closeup():
    """Celebration/player close-ups: big person, pitch still visible
    behind them — pitch ratio alone keeps them, person dominance must
    not. (Liverpool reel: wide gameplay max-person-height <= 0.17,
    close-ups >= 0.51.)"""
    from src.utils.shot_features import ShotFeatures
    from src.utils.shot_split import ShotSpan

    closeup = ShotFeatures(
        span=ShotSpan(0, 99, 0.0, 4.0),
        pitch_ratio_median=0.6, pitch_ratio_peak=0.8,
        brightness_min=0.4, brightness_range=0.1,
        motion_rate=0.05, max_person_height=0.85,
    )
    assert classify_kind(closeup) == "closeup"

    wide = ShotFeatures(
        span=ShotSpan(0, 99, 0.0, 4.0),
        pitch_ratio_median=0.7, pitch_ratio_peak=0.9,
        brightness_min=0.4, brightness_range=0.1,
        motion_rate=0.05, max_person_height=0.15,
    )
    assert classify_kind(wide) == "gameplay"


def test_reaction_takes_precedence_over_closeup():
    """Crowd close-ups have no pitch at all — keep the more specific
    'reaction' label."""
    from src.utils.shot_features import ShotFeatures
    from src.utils.shot_split import ShotSpan

    crowd = ShotFeatures(
        span=ShotSpan(0, 99, 0.0, 4.0),
        pitch_ratio_median=0.02, pitch_ratio_peak=0.05,
        brightness_min=0.4, brightness_range=0.1,
        motion_rate=0.05, max_person_height=0.9,
    )
    assert classify_kind(crowd) == "reaction"


def test_person_height_fn_feeds_features(tmp_path: Path):
    """compute_span_features aggregates the injected per-frame person
    measurement as a median; kind flips to closeup when dominant."""
    reel = tmp_path / "reel.mp4"
    info = build_reel(reel, [("green", 2.0)])
    feats = compute_span_features(
        reel, _spans_from(info), sample_points=[0.2, 0.5, 0.8],
        person_height_fn=lambda frame: 0.8,
    )
    assert feats[0].max_person_height == pytest.approx(0.8)
    assert feats[0].kind == "closeup"


def test_no_person_fn_means_no_closeup_classification(tmp_path: Path):
    reel = tmp_path / "reel.mp4"
    info = build_reel(reel, [("green", 2.0)])
    feats = compute_span_features(reel, _spans_from(info),
                                  sample_points=[0.2, 0.5, 0.8])
    assert feats[0].max_person_height == 0.0
    assert feats[0].kind == "gameplay"


def test_speed_reference_pool_excludes_closeups(tmp_path: Path):
    """Close-ups have inflated apparent motion; they must not drag the
    real-time reference rate (pool already filters kind == gameplay)."""
    reel = tmp_path / "reel.mp4"
    info = build_reel(reel, [("green", 3.0), ("green_slow", 3.0)])
    feats = compute_span_features(
        reel, _spans_from(info), sample_points=[0.2, 0.5, 0.8],
        # mark the SLOW span as a closeup via a frame-dependent stub:
        person_height_fn=None,
    )
    from dataclasses import replace
    feats = [feats[0], replace(feats[1], kind="closeup")]
    out = estimate_speed_factors(feats)
    # reference pool = the one wide gameplay shot -> its factor ~1.0
    assert out[0].speed_factor == pytest.approx(1.0, abs=0.15)


def test_medium_replay_shots_stay_gameplay():
    """Bournemouth GT: the operator KEEPS medium/close replay shots up
    to person-height ~0.74; only outright celebration close-ups (0.8+)
    drop. The old 0.5 threshold was tuned on unlabelled Liverpool data
    and over-dropped."""
    from src.utils.shot_features import ShotFeatures
    from src.utils.shot_split import ShotSpan

    replay = ShotFeatures(
        span=ShotSpan(0, 99, 0.0, 4.0),
        pitch_ratio_median=0.42, pitch_ratio_peak=0.6,
        brightness_min=0.4, brightness_range=0.1,
        motion_rate=0.05, max_person_height=0.74,
    )
    assert classify_kind(replay) == "gameplay"


def test_pitch_without_players_is_ambient():
    """A wide pitch shot with ZERO person detections (intro stadium
    shots, empty-pitch scenics) is not reconstructable gameplay."""
    from src.utils.shot_features import ShotFeatures
    from src.utils.shot_split import ShotSpan

    scenic = ShotFeatures(
        span=ShotSpan(0, 99, 0.0, 4.0),
        pitch_ratio_median=0.32, pitch_ratio_peak=0.5,
        brightness_min=0.4, brightness_range=0.1,
        motion_rate=0.05, max_person_height=0.0,
    )
    assert classify_kind(scenic, person_checked=True) == "ambient"
    # without the person check the signal is absent, not zero
    assert classify_kind(scenic, person_checked=False) == "gameplay"
