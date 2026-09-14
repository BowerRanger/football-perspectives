"""Unit tests for camera-regression baseline comparison and fixture
discovery (src/utils/camera_regression.py). Pure logic — no media."""

import pytest

from src.utils.camera_regression import compare_to_baseline, discover_clips


def _metrics(med=6.0, p90=14.0, covered=24, total=24, frames=429,
             conf=0.5) -> dict:
    return {"clicks": 463, "med_px": med, "p90_px": p90, "max_px": 38.0,
            "per_anchor": [], "anchor_frames_total": total,
            "anchor_frames_covered": covered, "track_frames": frames,
            "mean_confidence": conf}


def _baseline(**overrides) -> dict:
    base = {"clip_id": "clip", "metrics": _metrics(),
            "spread": {"med_px": 0.4, "p90_px": 0.9,
                       "mean_confidence": 0.02},
            "tolerances": {"med_rel": 0.15, "p90_rel": 0.20,
                           "confidence_abs": 0.05}}
    base.update(overrides)
    return base


@pytest.mark.unit
def test_identical_metrics_pass():
    assert compare_to_baseline(_metrics(), _baseline()) == []


@pytest.mark.unit
def test_regression_within_slack_passes():
    # med slack = max(6.0 * 0.15, 0.4) = 0.9 → 6.8 is inside
    assert compare_to_baseline(_metrics(med=6.8), _baseline()) == []


@pytest.mark.unit
def test_median_regression_beyond_slack_fails():
    failures = compare_to_baseline(_metrics(med=7.1), _baseline())
    assert len(failures) == 1
    assert "med_px" in failures[0]


@pytest.mark.unit
def test_p90_regression_beyond_slack_fails():
    # p90 slack = max(14.0 * 0.20, 0.9) = 2.8 → 17.0 is out
    failures = compare_to_baseline(_metrics(p90=17.0), _baseline())
    assert len(failures) == 1
    assert "p90_px" in failures[0]


@pytest.mark.unit
def test_anchor_coverage_drop_fails():
    failures = compare_to_baseline(_metrics(covered=23), _baseline())
    assert any("anchor_frames_covered" in f for f in failures)


@pytest.mark.unit
def test_track_frame_count_drop_fails():
    failures = compare_to_baseline(_metrics(frames=400), _baseline())
    assert any("track_frames" in f for f in failures)


@pytest.mark.unit
def test_confidence_drop_beyond_slack_fails():
    # conf slack = max(0.05, 0.02) = 0.05 → 0.44 is out, 0.46 is in
    assert compare_to_baseline(_metrics(conf=0.46), _baseline()) == []
    failures = compare_to_baseline(_metrics(conf=0.44), _baseline())
    assert any("mean_confidence" in f for f in failures)


@pytest.mark.unit
def test_unscored_track_fails():
    metrics = _metrics()
    metrics["med_px"] = metrics["p90_px"] = metrics["max_px"] = None
    metrics["clicks"] = 0
    failures = compare_to_baseline(metrics, _baseline())
    assert any("no clicks" in f for f in failures)


@pytest.mark.unit
def test_improvements_pass():
    assert compare_to_baseline(
        _metrics(med=4.0, p90=10.0, conf=0.7), _baseline()) == []


@pytest.mark.unit
def test_discover_clips_finds_only_complete_fixture_dirs(tmp_path):
    for name, files in [("alpha", ["baseline.json", "anchors.json"]),
                        ("beta", ["baseline.json"]),  # incomplete
                        ("gamma", ["baseline.json", "anchors.json"])]:
        d = tmp_path / name
        d.mkdir()
        for f in files:
            (d / f).write_text("{}")
    (tmp_path / "notes.md").write_text("not a clip dir")
    assert discover_clips(tmp_path) == ["alpha", "gamma"]


@pytest.mark.unit
def test_discover_clips_empty_root(tmp_path):
    assert discover_clips(tmp_path / "missing") == []
