"""Fast unit tests for ``src/utils/ball_bench_synth.py`` that don't need a
real ``ClipContext``: the miss-probability model and hidden-scenario
derivation (pure functions of a ``TruthTrack``/``SynthRun`` pair)."""

from __future__ import annotations

import pytest

from src.utils import ball_bench_synth as BS
from src.utils.ball_bench_types import (
    Observation,
    SynthRun,
    TruthFrame,
    TruthTrack,
)


# ---------------------------------------------------------------------------
# _MissModel
# ---------------------------------------------------------------------------

def test_miss_model_monotonic_in_speed_and_clamped():
    model = BS._MissModel(base_p=0.1, speed_scale=0.5,
                          occlusion_bonus=0.25, speed_ref_px_frame=10.0)
    p_slow = model.p_miss(0.0, occluded=False)
    p_fast = model.p_miss(10.0, occluded=False)
    assert p_slow < p_fast
    assert p_fast <= 0.95
    p_occluded = model.p_miss(0.0, occluded=True)
    assert p_occluded == pytest.approx(p_slow + 0.25)


def test_calibrate_base_p_hits_target_coverage_roughly():
    speeds = [0.0] * 50 + [30.0] * 50
    occluded = [False] * 100
    base_p = BS._calibrate_base_p(
        target_coverage=0.7, speeds=speeds, occluded=occluded,
        speed_scale=0.35, occlusion_bonus=0.0, speed_ref=25.0)
    model = BS._MissModel(base_p, 0.35, 0.0, 25.0)
    achieved = 1.0 - sum(model.p_miss(s, o) for s, o in zip(speeds, occluded)) / 100
    assert achieved == pytest.approx(0.7, abs=0.02)


# ---------------------------------------------------------------------------
# derive_hidden_scenario
# ---------------------------------------------------------------------------

def _mismatch_truth_and_run():
    truth = TruthTrack(
        clip_id="fake", scenario=BS.SOURCE_SCENARIO, fps=25.0,
        frames=tuple(TruthFrame(i, (float(i), 0.0, 0.11), "ground")
                     for i in range(10)),
    )
    anchors = tuple({
        "frame": i * 2, "image_xy": [float(i), float(i)], "state": "grounded",
        "player_id": None, "bone": None, "goal_element": None,
        "touch_type": None, "spin": None, "confidence": 1.0,
        "end_frame": None,
    } for i in range(6))
    run = SynthRun(
        clip_id="fake", scenario=BS.SOURCE_SCENARIO,
        observations=(Observation(frame=0, uv=(1.0, 1.0), conf=0.9,
                                   source="detector"),),
        anchors=anchors,
        noise_model={"sigma_px": 1.5},
    )
    return truth, run


def test_derive_hidden_scenario_keeps_only_fold0_half():
    truth, run = _mismatch_truth_and_run()
    hidden_truth, hidden_run = BS.derive_hidden_scenario(truth, run)

    assert hidden_truth.scenario == BS.HIDDEN_SCENARIO
    assert hidden_run.scenario == BS.HIDDEN_SCENARIO
    # Truth trajectory (dense frames/events) is untouched.
    assert hidden_truth.frames == truth.frames
    assert hidden_truth.events == truth.events
    # Anchors are a strict, non-empty subset of the original set.
    assert 0 < len(hidden_run.anchors) < len(run.anchors)
    kept_frames = {a["frame"] for a in hidden_run.anchors}
    orig_frames = {a["frame"] for a in run.anchors}
    assert kept_frames <= orig_frames
    # noise_model records provenance.
    assert hidden_run.noise_model["derived_from"] == BS.SOURCE_SCENARIO
    assert hidden_run.noise_model["anchors_kept"] == len(hidden_run.anchors)
    assert (hidden_run.noise_model["anchors_withheld"]
            == len(run.anchors) - len(hidden_run.anchors))
    # Original sigma_px etc. survive alongside the new provenance keys.
    assert hidden_run.noise_model["sigma_px"] == 1.5


def test_derive_hidden_scenario_deterministic():
    truth, run = _mismatch_truth_and_run()
    a = BS.derive_hidden_scenario(truth, run)
    b = BS.derive_hidden_scenario(truth, run)
    assert a[1].anchors == b[1].anchors
