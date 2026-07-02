"""GET /ball-quality/{shot_id}: sidecar aggregation, degradation, legacy names."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from src.web.server import create_app


def _client(tmp_path: Path) -> TestClient:
    return TestClient(create_app(output_dir=tmp_path, config_path=None))


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def test_no_sidecars_degrades_to_empty_payload(tmp_path: Path):
    r = _client(tmp_path).get("/ball-quality/play")
    assert r.status_code == 200
    body = r.json()
    assert body["n_frames"] == 0
    assert body["annotate_next"] == []


def test_aggregates_prefixed_sidecars(tmp_path: Path):
    ball = tmp_path / "ball"
    _write(ball / "play_ball_observations.json", {
        "clip_id": "play", "fps": 30.0,
        "frames": [
            {"frame": 0, "uv": [1.0, 2.0], "confidence": 0.9,
             "p_flight": 0.0, "gap_fill": False, "source": "detector"},
            {"frame": 1, "uv": None, "confidence": 0.0,
             "p_flight": 0.0, "gap_fill": True, "source": "none"},
        ],
    })
    _write(ball / "play_ball_diag.json", {
        "underconstrained_spans": [{"start": 0, "end": 1, "residual_px": 5.0}],
        "events": [{"frame": 0, "kind": "touch", "score": 0.7,
                    "player_id": "P1", "bone": "l_foot",
                    "goal_element": None, "end_frame": None}],
        "detection_coverage": {"pass1": 0.5, "second_pass": 0.0,
                               "total": 0.5, "zoom_recoveries": 0},
    })
    _write(ball / "play_ball_keyframes.json", {
        "segments": [{"start_frame": 0, "end_frame": 1,
                      "kind": "roll", "hints": {}}],
    })
    body = _client(tmp_path).get("/ball-quality/play").json()
    assert body["n_frames"] == 2
    assert body["events"][0]["bone"] == "l_foot"
    assert body["segments"] == [
        {"start_frame": 0, "end_frame": 1, "kind": "roll"}]
    assert body["annotate_next"][0]["reason"] == "underconstrained_flight"


def test_legacy_unprefixed_sidecar_fallback(tmp_path: Path):
    _write(tmp_path / "ball" / "ball_observations.json", {
        "clip_id": "play", "fps": 25.0,
        "frames": [{"frame": 0, "uv": [1.0, 2.0], "confidence": 0.9,
                    "p_flight": 0.0, "gap_fill": False, "source": "detector"}],
    })
    body = _client(tmp_path).get("/ball-quality/play").json()
    assert body["n_frames"] == 1
    assert body["fps"] == 25.0


def test_corrupt_sidecar_degrades_not_500(tmp_path: Path):
    p = tmp_path / "ball" / "play_ball_diag.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("{not json")
    r = _client(tmp_path).get("/ball-quality/play")
    assert r.status_code == 200
    assert r.json()["events"] == []
