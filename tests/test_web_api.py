"""Integration tests for the Phase 4a web API endpoints.

Uses FastAPI's ``TestClient`` — no running server required.  Each test
boots a fresh app in a temporary output directory, exercises a single
endpoint, and asserts on the response.

The empty-state shapes asserted here mirror what the server actually
returns for an uninitialised pipeline (see ``src/web/server.py``):

    * ``GET /anchors`` — ``{"clip_id": "", "image_size": [0, 0], "anchors": []}``
    * ``GET /camera/track`` — extended empty dict with ``frames=[]``
    * ``GET /ball/preview`` — extended empty dict with ``frames=[]`` and ``flight_segments=[]``
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.web.server import create_app


@pytest.fixture
def client(tmp_path: Path):
    app = create_app(output_dir=tmp_path, config_path=None)
    return TestClient(app), tmp_path


@pytest.mark.integration
def test_get_anchors_empty(client) -> None:
    c, _ = client
    resp = c.get("/anchors")
    assert resp.status_code == 200
    assert resp.json() == {"clip_id": "", "image_size": [0, 0], "anchors": []}


@pytest.mark.integration
def test_post_anchors_round_trips(client) -> None:
    c, tmp = client
    payload = {
        "clip_id": "play_037",
        "image_size": [1920, 1080],
        "anchors": [{"frame": 0, "landmarks": []}],
    }
    resp = c.post("/anchors", json=payload)
    assert resp.status_code == 200
    saved = json.loads((tmp / "camera" / "anchors.json").read_text())
    assert saved["clip_id"] == "play_037"
    assert list(saved["image_size"]) == [1920, 1080]
    assert len(saved["anchors"]) == 1

    # GET round-trips the same data back.
    resp2 = c.get("/anchors")
    assert resp2.status_code == 200
    body = resp2.json()
    assert body["clip_id"] == "play_037"
    assert body["image_size"] == [1920, 1080]
    assert body["anchors"][0]["frame"] == 0


@pytest.mark.integration
def test_get_camera_track_empty(client) -> None:
    c, _ = client
    resp = c.get("/camera/track")
    assert resp.status_code == 200
    body = resp.json()
    assert body["frames"] == []


@pytest.mark.integration
def test_get_ball_preview_empty(client) -> None:
    c, _ = client
    resp = c.get("/ball/preview")
    assert resp.status_code == 200
    body = resp.json()
    assert body["frames"] == []
    assert body["flight_segments"] == []
