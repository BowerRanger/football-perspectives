"""Smoke tests for the /ball-anchors endpoints."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.web.server import create_app


@pytest.fixture
def client(tmp_path: Path):
    (tmp_path / "shots").mkdir()
    app = create_app(output_dir=tmp_path, config_path=None)
    return TestClient(app), tmp_path


def test_get_ball_anchors_returns_empty_when_no_file(client):
    c, _ = client
    r = c.get("/ball-anchors/play")
    assert r.status_code == 200
    body = r.json()
    assert body["anchors"] == []


def test_post_then_get_roundtrips(client):
    c, tmp_path = client
    payload = {
        "clip_id": "play",
        "image_size": [1280, 720],
        "anchors": [
            {"frame": 5, "image_xy": [640.0, 360.0], "state": "grounded"},
            {"frame": 78, "image_xy": [700.0, 200.0], "state": "kick"},
            {"frame": 120, "image_xy": None, "state": "off_screen_flight"},
        ],
    }
    r = c.post("/ball-anchors/play", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["saved"] is True
    assert body["count"] == 3
    # File should exist on disk.
    p = tmp_path / "ball" / "play_ball_anchors.json"
    assert p.exists()
    saved = json.loads(p.read_text())
    assert len(saved["anchors"]) == 3
    # GET should return what we POSTed.
    r2 = c.get("/ball-anchors/play")
    assert r2.status_code == 200
    body2 = r2.json()
    assert len(body2["anchors"]) == 3
    assert body2["anchors"][0]["state"] == "grounded"


def test_post_invalid_state_rejected(client):
    c, _ = client
    payload = {
        "clip_id": "play", "image_size": [1280, 720],
        "anchors": [{"frame": 1, "image_xy": [0, 0], "state": "bogus"}],
    }
    r = c.post("/ball-anchors/play", json=payload)
    assert r.status_code == 400


def test_post_missing_pixel_for_grounded_state_rejected(client):
    c, _ = client
    payload = {
        "clip_id": "play", "image_size": [1280, 720],
        "anchors": [{"frame": 1, "image_xy": None, "state": "grounded"}],
    }
    r = c.post("/ball-anchors/play", json=payload)
    assert r.status_code == 400
