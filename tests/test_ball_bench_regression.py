"""Unit tests for ``src/utils/ball_bench_regression.py``'s gate logic:
``aggregate_runs``, ``compare_to_baseline``, ``discover_clips``."""

from __future__ import annotations

import json

import pytest

from src.utils import ball_bench_regression as BRG


# ---------------------------------------------------------------------------
# aggregate_runs
# ---------------------------------------------------------------------------

def test_aggregate_runs_median_and_spread():
    dicts = [{"p50": 0.10, "p95": 1.0}, {"p50": 0.14, "p95": 1.2},
             {"p50": 0.12, "p95": 1.1}]
    metrics, spread = BRG.aggregate_runs(dicts, ("p50", "p95"))
    assert metrics["p50"] == 0.12
    assert metrics["p95"] == 1.1
    assert spread["p50"] == pytest.approx(0.04)  # max-min = 0.14-0.10
    assert spread["p95"] == pytest.approx(0.2)


def test_aggregate_runs_ignores_none_but_keeps_key():
    dicts = [{"x": None}, {"x": 5.0}, {"x": 7.0}]
    metrics, spread = BRG.aggregate_runs(dicts, ("x",))
    assert metrics["x"] == 6.0
    assert spread["x"] == 2.0


def test_aggregate_runs_all_none_stays_none_zero_spread():
    dicts = [{"x": None}, {"x": None}]
    metrics, spread = BRG.aggregate_runs(dicts, ("x",))
    assert metrics["x"] is None
    assert spread["x"] == 0.0


# ---------------------------------------------------------------------------
# compare_to_baseline
# ---------------------------------------------------------------------------

def _baseline(**overrides):
    base = {
        "synth_metrics": {"pct_le_20cm": 0.70, "p95": 1.0,
                          "ground_float_sink": 0.10,
                          "naturalness_violations_minus_truth": 2},
        "synth_spread": {"pct_le_20cm": 0.02, "p95": 0.1,
                        "ground_float_sink": 0.01,
                        "naturalness_violations_minus_truth": 0},
        "real_metrics": {"p50": 0.15, "p95": 0.5},
        "real_spread": {"p50": 0.01, "p95": 0.05},
        "tolerances": {},
    }
    base.update(overrides)
    return base


def test_compare_to_baseline_no_regression_passes():
    baseline = _baseline()
    synth = {"pct_le_20cm": 0.71, "p95": 0.95, "ground_float_sink": 0.09,
             "naturalness_violations_minus_truth": 2}
    real = {"p50": 0.14, "p95": 0.48}
    assert BRG.compare_to_baseline(synth, real, baseline) == []


def test_compare_to_baseline_flags_pct_le_20cm_drop():
    baseline = _baseline()
    synth = {"pct_le_20cm": 0.10, "p95": 1.0, "ground_float_sink": 0.10,
             "naturalness_violations_minus_truth": 2}
    real = {"p50": 0.15, "p95": 0.5}
    failures = BRG.compare_to_baseline(synth, real, baseline)
    assert any("pct_le_20cm" in f for f in failures)


def test_compare_to_baseline_flags_p95_increase_beyond_slack():
    baseline = _baseline()
    synth = {"pct_le_20cm": 0.70, "p95": 5.0, "ground_float_sink": 0.10,
             "naturalness_violations_minus_truth": 2}
    real = {"p50": 0.15, "p95": 0.5}
    failures = BRG.compare_to_baseline(synth, real, baseline)
    assert any("synth.p95" in f for f in failures)


def test_compare_to_baseline_flags_real_p50_regression():
    baseline = _baseline()
    synth = {"pct_le_20cm": 0.70, "p95": 1.0, "ground_float_sink": 0.10,
             "naturalness_violations_minus_truth": 2}
    real = {"p50": 5.0, "p95": 0.5}
    failures = BRG.compare_to_baseline(synth, real, baseline)
    assert any("real.p50" in f for f in failures)


def test_compare_to_baseline_missing_real_metrics_skips_real_gate():
    baseline = _baseline(real_metrics=None)
    synth = {"pct_le_20cm": 0.70, "p95": 1.0, "ground_float_sink": 0.10,
             "naturalness_violations_minus_truth": 2}
    real = {"p50": None, "p95": None}
    assert BRG.compare_to_baseline(synth, real, baseline) == []


def test_compare_to_baseline_missing_current_value_fails():
    baseline = _baseline()
    synth = {"pct_le_20cm": None, "p95": 1.0, "ground_float_sink": 0.10,
             "naturalness_violations_minus_truth": 2}
    real = {"p50": 0.15, "p95": 0.5}
    failures = BRG.compare_to_baseline(synth, real, baseline)
    assert any("pct_le_20cm" in f and "missing" in f for f in failures)


# ---------------------------------------------------------------------------
# discover_clips
# ---------------------------------------------------------------------------

def test_discover_clips_requires_all_files(tmp_path):
    clip_dir = tmp_path / "gberch"
    clip_dir.mkdir()
    assert BRG.discover_clips(tmp_path) == []

    for name in ("baseline.json", "anchors.json", "shot.json",
                 "truth_mismatch.json", "synth_obs_mismatch.json"):
        (clip_dir / name).write_text(json.dumps({}))
    assert BRG.discover_clips(tmp_path) == ["gberch"]


def test_discover_clips_empty_root():
    from pathlib import Path
    assert BRG.discover_clips(Path("/nonexistent/path/xyz")) == []
