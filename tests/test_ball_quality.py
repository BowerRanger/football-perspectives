"""build_quality_payload: sidecar aggregation + annotate-next ranking."""

from __future__ import annotations

from src.utils.ball_quality import (
    build_quality_payload,
    detection_gaps,
    rank_annotate_next,
)


def _obs_frame(frame: int, conf: float, gap_fill: bool = False) -> dict:
    return {"frame": frame, "uv": [100.0, 200.0], "confidence": conf,
            "p_flight": 0.1, "gap_fill": gap_fill, "source": "detector"}


def test_missing_sidecars_degrade_to_empty_payload():
    payload = build_quality_payload(None, None, None)
    assert payload == {
        "n_frames": 0, "fps": None, "frames": [], "events": [],
        "underconstrained_spans": [], "segments": [],
        "detection_coverage": None, "annotate_next": [],
    }


def test_detection_gap_run_detected():
    frames = [_obs_frame(i, 0.9) for i in range(5)]
    frames += [_obs_frame(5 + i, 0.0) for i in range(15)]      # 15-frame hole
    frames += [_obs_frame(20 + i, 0.9) for i in range(5)]
    assert detection_gaps(frames, min_gap_frames=12) == [(5, 19)]


def test_gap_fill_frames_count_as_missing():
    frames = [_obs_frame(i, 0.8, gap_fill=True) for i in range(12)]
    assert detection_gaps(frames, min_gap_frames=12) == [(0, 11)]


def test_short_gap_ignored():
    frames = [_obs_frame(0, 0.9)] + [_obs_frame(1 + i, 0.0) for i in range(5)] \
        + [_obs_frame(6, 0.9)]
    assert detection_gaps(frames, min_gap_frames=12) == []


def test_underconstrained_span_outranks_gap():
    spans = [{"start": 10, "end": 20, "residual_px": 8.0}]
    gaps = [(40, 60)]
    ranked = rank_annotate_next(spans, gaps)
    assert [it["reason"] for it in ranked] == [
        "underconstrained_flight", "detection_gap"]
    assert ranked[0]["start"] == 10 and ranked[0]["end"] == 20
    assert ranked[0]["severity"] > ranked[1]["severity"]


def test_payload_aggregates_all_three_sidecars():
    observations = {"clip_id": "play", "fps": 30.0,
                    "frames": [_obs_frame(0, 0.9), _obs_frame(1, 0.0)]}
    diag = {
        "underconstrained_spans": [{"start": 0, "end": 1, "residual_px": None}],
        "events": [{"frame": 1, "kind": "touch", "score": 0.8,
                    "player_id": "P1", "bone": "r_foot",
                    "goal_element": None, "end_frame": None}],
        "detection_coverage": {"pass1": 0.5, "second_pass": 0.0,
                               "total": 0.5, "zoom_recoveries": 0},
    }
    keyframes = {"segments": [
        {"start_frame": 0, "end_frame": 1, "kind": "roll", "hints": {}}]}
    payload = build_quality_payload(observations, diag, keyframes)
    assert payload["n_frames"] == 2
    assert payload["fps"] == 30.0
    assert payload["frames"][1] == {
        "frame": 1, "confidence": 0.0, "gap_fill": False, "source": "detector"}
    assert payload["events"][0]["kind"] == "touch"
    assert payload["segments"] == [
        {"start_frame": 0, "end_frame": 1, "kind": "roll"}]
    assert payload["detection_coverage"]["total"] == 0.5
    assert payload["annotate_next"][0]["reason"] == "underconstrained_flight"
