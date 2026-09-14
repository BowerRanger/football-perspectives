"""Unit tests for the camera-regression harness plumbing: the scratch
solve-dir builder and multi-run metric aggregation. No media needed."""

import json

import pytest

from src.utils.camera_regression import aggregate_runs, build_solve_dir


def _shot_fixture() -> dict:
    return {"fps": 30.0,
            "shot": {"id": "clipx", "start_frame": 0, "end_frame": 428,
                     "start_time": 0.0, "end_time": 14.3,
                     "clip_file": "shots/clipx.mp4", "speed_factor": 1.0,
                     "kind": "gameplay", "excluded": False,
                     "exclude_reason": "", "group_id": "",
                     "source_start_s": -1.0, "source_end_s": -1.0}}


@pytest.mark.unit
def test_build_solve_dir_lays_out_single_shot_pipeline_inputs(tmp_path):
    clip = tmp_path / "src_clip.mp4"
    clip.write_bytes(b"fake-mp4-bytes")
    anchors = {"clip_id": "clipx", "anchors": []}
    work = tmp_path / "work"

    build_solve_dir(work, _shot_fixture(), anchors, clip)

    manifest = json.loads(
        (work / "shots" / "shots_manifest.json").read_text())
    assert manifest["fps"] == 30.0
    assert [s["id"] for s in manifest["shots"]] == ["clipx"]
    assert manifest["shots"][0]["excluded"] is False
    assert (work / "shots" / "clipx.mp4").read_bytes() == b"fake-mp4-bytes"
    written = json.loads(
        (work / "camera" / "clipx_anchors.json").read_text())
    assert written == anchors


@pytest.mark.unit
def test_aggregate_runs_medians_percentiles_and_floors_counts():
    runs = [
        {"clicks": 463, "med_px": 6.0, "p90_px": 14.0, "max_px": 38.0,
         "per_anchor": [{"frame": 0}], "anchor_frames_total": 24,
         "anchor_frames_covered": 24, "track_frames": 429,
         "mean_confidence": 0.50},
        {"clicks": 463, "med_px": 6.4, "p90_px": 15.0, "max_px": 40.0,
         "per_anchor": [{"frame": 0}], "anchor_frames_total": 24,
         "anchor_frames_covered": 23, "track_frames": 428,
         "mean_confidence": 0.48},
        {"clicks": 463, "med_px": 6.2, "p90_px": 14.5, "max_px": 39.0,
         "per_anchor": [{"frame": 0}], "anchor_frames_total": 24,
         "anchor_frames_covered": 24, "track_frames": 429,
         "mean_confidence": 0.52},
    ]
    metrics, spread = aggregate_runs(runs)
    assert metrics["med_px"] == pytest.approx(6.2)
    assert metrics["p90_px"] == pytest.approx(14.5)
    assert metrics["mean_confidence"] == pytest.approx(0.50)
    # counts take the conservative floor so a once-flaky run can't
    # bake in a coverage bar later runs can't meet
    assert metrics["anchor_frames_covered"] == 23
    assert metrics["track_frames"] == 428
    assert spread["med_px"] == pytest.approx(0.4)
    assert spread["p90_px"] == pytest.approx(1.0)
    assert spread["mean_confidence"] == pytest.approx(0.04)


@pytest.mark.unit
def test_aggregate_runs_single_run_has_zero_spread():
    run = {"clicks": 10, "med_px": 5.0, "p90_px": 9.0, "max_px": 12.0,
           "per_anchor": [], "anchor_frames_total": 3,
           "anchor_frames_covered": 3, "track_frames": 100,
           "mean_confidence": 0.7}
    metrics, spread = aggregate_runs([run])
    assert metrics["med_px"] == pytest.approx(5.0)
    assert spread == {"med_px": 0.0, "p90_px": 0.0, "mean_confidence": 0.0}
