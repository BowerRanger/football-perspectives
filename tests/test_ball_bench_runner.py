"""Fast unit tests for ``src/utils/ball_bench_runner.py``: the
content-hash-keyed ``SyntheticDetector`` and the config-override helpers.
No video decode, no real ball-stage run.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.utils import ball_bench_runner as BR
from src.utils.ball_bench_types import Observation, SynthRun


def _fake_frame(fill: int, shape=(20, 20, 3)) -> np.ndarray:
    return np.full(shape, fill, dtype=np.uint8)


# ---------------------------------------------------------------------------
# SyntheticDetector: content-hash keying
# ---------------------------------------------------------------------------

def test_synthetic_detector_keys_by_content_not_call_order():
    """Three distinct fake frames, indexed once; querying them OUT OF
    ORDER (as a seek-then-read pass would) must still resolve each to its
    own observation — proving resolution is by content hash, not call
    sequence."""
    frames = [_fake_frame(10), _fake_frame(20), _fake_frame(30)]
    run = SynthRun(
        clip_id="fake", scenario="unit",
        observations=(
            Observation(frame=0, uv=(1.0, 1.0), conf=0.9, source="detector"),
            Observation(frame=1, uv=(2.0, 2.0), conf=0.8, source="detector"),
            Observation(frame=2, uv=(3.0, 3.0), conf=0.7, source="detector"),
        ),
        anchors=(), noise_model={},
    )
    det = BR.SyntheticDetector(run, frames=frames)

    assert det.detect(frames[2]) == (3.0, 3.0, 0.7)
    assert det.detect(frames[0]) == (1.0, 1.0, 0.9)
    assert det.detect(frames[1]) == (2.0, 2.0, 0.8)
    assert det.hash_misses == 0
    assert det.crop_calls == 0


def test_synthetic_detector_frame_with_no_observation_returns_none():
    frames = [_fake_frame(10), _fake_frame(20)]
    run = SynthRun(
        clip_id="fake", scenario="unit",
        observations=(
            Observation(frame=0, uv=(1.0, 1.0), conf=0.9, source="detector"),
        ),
        anchors=(), noise_model={},
    )
    det = BR.SyntheticDetector(run, frames=frames)
    assert det.detect(frames[0]) is not None
    assert det.detect(frames[1]) is None
    assert det.hash_misses == 0
    assert det.crop_calls == 0


def test_synthetic_detector_counts_hash_miss_for_unindexed_content():
    frames = [_fake_frame(10), _fake_frame(20)]
    run = SynthRun(clip_id="fake", scenario="unit", observations=(),
                    anchors=(), noise_model={})
    det = BR.SyntheticDetector(run, frames=frames)
    det.detect(frames[0])
    assert det.hash_misses == 0
    det.detect(_fake_frame(99))
    assert det.hash_misses == 1
    assert det.positional_fallbacks == 1


def test_synthetic_detector_counts_crop_calls_not_hash_misses():
    """A smaller crop must be counted as a crop call, not conflated with a
    genuine hash miss on full-frame content."""
    frames = [_fake_frame(10, shape=(20, 20, 3))]
    run = SynthRun(clip_id="fake", scenario="unit", observations=(),
                    anchors=(), noise_model={})
    det = BR.SyntheticDetector(run, frames=frames)
    crop = _fake_frame(10, shape=(8, 8, 3))
    assert det.detect(crop) is None
    assert det.crop_calls == 1
    assert det.hash_misses == 0


def test_synthetic_detector_candidates_merge_obs_fp_and_weak():
    frames = [_fake_frame(10)]
    run = SynthRun(
        clip_id="fake", scenario="unit",
        observations=(
            Observation(frame=0, uv=(1.0, 1.0), conf=0.9, source="detector"),
        ),
        anchors=(),
        noise_model={
            "false_positives": {"0": [[5.0, 5.0, 0.4]]},
            "weak_candidates": {"0": [[6.0, 6.0, 0.1]]},
        },
    )
    det = BR.SyntheticDetector(run, frames=frames)
    cands = det.detect_candidates(frames[0], min_score=0.05, top_k=5)
    scores = sorted(c[2] for c in cands)
    assert scores == [0.1, 0.4, 0.9]
    cands2 = det.detect_candidates(frames[0], min_score=0.2, top_k=5)
    assert sorted(c[2] for c in cands2) == [0.4, 0.9]


def test_synthetic_detector_requires_exactly_one_source():
    run = SynthRun(clip_id="fake", scenario="unit", observations=(),
                    anchors=(), noise_model={})
    with pytest.raises(ValueError):
        BR.SyntheticDetector(run)
    with pytest.raises(ValueError):
        BR.SyntheticDetector(run, video_path="x", frames=[_fake_frame(1)])


# ---------------------------------------------------------------------------
# config override helpers (pure dict mutation, no I/O)
# ---------------------------------------------------------------------------

def test_apply_synth_overrides_sets_exact_keys():
    config = {"ball": {"second_pass": {"zoom_min_ball_px": 40},
                        "foot_guided": {"enabled": True},
                        "appearance_bridge": {"enabled": True}}}
    BR.apply_synth_overrides(config)
    assert config["ball"]["second_pass"]["zoom_min_ball_px"] == 0
    assert config["ball"]["foot_guided"]["enabled"] is False
    assert config["ball"]["appearance_bridge"]["enabled"] is False


def test_apply_trajectory_sets_key_without_touching_rest():
    config = {"ball": {"other": 1}}
    BR.apply_trajectory(config, "hybrid")
    assert config["ball"]["trajectory"] == "hybrid"
    assert config["ball"]["other"] == 1


def test_apply_set_overrides_parses_yaml_scalars():
    config = {"ball": {"physics": {"cd": 0.25}}}
    BR.apply_set_overrides(config, [
        "ball.physics.cd=0.5",
        "ball.physics.fit_cd=false",
        "ball.new_key=hello",
    ])
    assert config["ball"]["physics"]["cd"] == 0.5
    assert config["ball"]["physics"]["fit_cd"] is False
    assert config["ball"]["new_key"] == "hello"


def test_apply_set_overrides_rejects_bad_format():
    with pytest.raises(ValueError):
        BR.apply_set_overrides({}, ["not_a_kv_pair"])


def test_build_config_applies_overrides_in_documented_order():
    config = BR.build_config(trajectory="hybrid", synthetic=True,
                              overrides=["ball.trajectory=override_wins"])
    # --set is applied last, so it must win over apply_trajectory.
    assert config["ball"]["trajectory"] == "override_wins"
    assert config["ball"]["foot_guided"]["enabled"] is False


def test_normalize_frame_keyed_accepts_list_and_dict_forms():
    flat = [{"frame": 3, "uv": [1.0, 2.0], "score": 0.5}]
    out = BR._normalize_frame_keyed(flat)
    assert out == {3: [(1.0, 2.0, 0.5)]}

    keyed = {"3": [[1.0, 2.0, 0.5]]}
    out2 = BR._normalize_frame_keyed(keyed)
    assert out2 == {3: [(1.0, 2.0, 0.5)]}

    assert BR._normalize_frame_keyed(None) == {}
