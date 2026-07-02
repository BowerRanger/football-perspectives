"""dismissed_auto round-trips through the save endpoint."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from src.web.server import create_app


def _client(tmp_path: Path) -> TestClient:
    return TestClient(create_app(output_dir=tmp_path, config_path=None))


def test_post_persists_dismissed_auto(tmp_path: Path):
    client = _client(tmp_path)
    payload = {
        "clip_id": "play", "image_size": [1280, 720],
        "anchors": [{"frame": 5, "image_xy": [1.0, 2.0], "state": "grounded"}],
        "dismissed_auto": [
            {"frame": 20, "state": "player_touch",
             "player_id": "P003", "bone": "l_foot"},
        ],
    }
    r = client.post("/ball-anchors/play", json=payload)
    assert r.status_code == 200, r.text
    got = client.get("/ball-anchors/play").json()
    assert got["dismissed_auto"] == [
        {"frame": 20, "state": "player_touch",
         "player_id": "P003", "bone": "l_foot"},
    ]


def test_post_without_dismissals_unchanged(tmp_path: Path):
    client = _client(tmp_path)
    payload = {"clip_id": "play", "image_size": [1280, 720],
               "anchors": []}
    assert client.post("/ball-anchors/play", json=payload).status_code == 200
    assert client.get("/ball-anchors/play").json().get(
        "dismissed_auto", []) == []
