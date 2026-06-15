"""Web API: ball-anchor confidence/end_frame round-trip + /joints-near."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from src.web.server import create_app


def _client(tmp_path: Path) -> TestClient:
    return TestClient(create_app(output_dir=tmp_path, config_path=None))


def test_ball_anchor_post_preserves_confidence(tmp_path: Path):
    client = _client(tmp_path)
    payload = {
        "clip_id": "play", "image_size": [1280, 720],
        "anchors": [{
            "frame": 5, "image_xy": [10.0, 20.0], "state": "grounded",
            "confidence": 0.5,
        }],
    }
    r = client.post("/ball-anchors/play", json=payload)
    assert r.status_code == 200, r.text
    got = client.get("/ball-anchors/play").json()
    assert got["anchors"][0]["confidence"] == 0.5


def test_joints_near_graceful_without_player_data(tmp_path: Path):
    client = _client(tmp_path)
    r = client.get(
        "/joints-near",
        params={"shot": "play", "frame": 0, "u": 100, "v": 100, "r": 40},
    )
    assert r.status_code == 200
    assert r.json().get("joints") == []
