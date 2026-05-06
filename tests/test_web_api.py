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


@pytest.mark.integration
def test_get_pitch_lines_returns_catalogue(client) -> None:
    c, _ = client
    resp = c.get("/pitch_lines")
    assert resp.status_code == 200
    body = resp.json()
    names = [ln["name"] for ln in body["lines"]]
    # Spot-check a few canonical lines and that the list is sorted.
    assert "near_touchline" in names
    assert "halfway_line" in names
    assert "left_goal_left_post" in names
    assert names == sorted(names)
    # Each entry is ((x1,y1,z1),(x2,y2,z2))
    halfway = next(ln for ln in body["lines"] if ln["name"] == "halfway_line")
    assert halfway["world_segment"] == [[52.5, 0.0, 0.0], [52.5, 68.0, 0.0]]


@pytest.mark.integration
def test_post_anchors_round_trips_lines(client) -> None:
    """An anchor with both points and lines round-trips through POST/GET."""
    c, tmp = client
    payload = {
        "clip_id": "play_037",
        "image_size": [1920, 1080],
        "anchors": [
            {
                "frame": 50,
                "landmarks": [
                    {
                        "name": "near_left_corner",
                        "image_xy": [400.0, 900.0],
                        "world_xyz": [0.0, 0.0, 0.0],
                    }
                ],
                "lines": [
                    {
                        "name": "near_touchline",
                        "image_segment": [[400.0, 900.0], [1500.0, 880.0]],
                        "world_segment": [[0.0, 0.0, 0.0], [105.0, 0.0, 0.0]],
                    }
                ],
            }
        ],
    }
    resp = c.post("/anchors", json=payload)
    assert resp.status_code == 200

    resp2 = c.get("/anchors")
    body = resp2.json()
    a = body["anchors"][0]
    assert len(a["landmarks"]) == 1
    assert len(a["lines"]) == 1
    line = a["lines"][0]
    assert line["name"] == "near_touchline"
    assert line["image_segment"] == [[400.0, 900.0], [1500.0, 880.0]]
    assert line["world_segment"] == [[0.0, 0.0, 0.0], [105.0, 0.0, 0.0]]
    # Position-known line has world_direction = null on the wire.
    assert line["world_direction"] is None


@pytest.mark.integration
def test_post_anchors_round_trips_vanishing_line(client) -> None:
    """A direction-only (VP) line annotation round-trips correctly."""
    c, tmp = client
    payload = {
        "clip_id": "play_037",
        "image_size": [1920, 1080],
        "anchors": [
            {
                "frame": 169,
                "landmarks": [],
                "lines": [
                    {
                        "name": "vertical_separator",
                        "image_segment": [[800.0, 200.0], [802.0, 380.0]],
                        "world_segment": None,
                        "world_direction": [0.0, 0.0, 1.0],
                    }
                ],
            }
        ],
    }
    resp = c.post("/anchors", json=payload)
    assert resp.status_code == 200

    resp2 = c.get("/anchors")
    body = resp2.json()
    line = body["anchors"][0]["lines"][0]
    assert line["name"] == "vertical_separator"
    assert line["world_segment"] is None
    assert line["world_direction"] == [0.0, 0.0, 1.0]


@pytest.mark.integration
def test_get_pitch_lines_includes_vanishing_lines(client) -> None:
    c, _ = client
    resp = c.get("/pitch_lines")
    body = resp.json()
    names = [ln["name"] for ln in body["lines"]]
    assert "vertical_separator" in names
    vs = next(ln for ln in body["lines"] if ln["name"] == "vertical_separator")
    assert vs["world_direction"] == [0.0, 0.0, 1.0]
    assert "world_segment" not in vs or vs.get("world_segment") is None


@pytest.mark.integration
def test_get_stadiums_returns_registry(client) -> None:
    c, _ = client
    resp = c.get("/stadiums")
    assert resp.status_code == 200
    body = resp.json()
    assert "stadiums" in body
    ids = [s["id"] for s in body["stadiums"]]
    # ``default_premier_league`` ships in config/stadiums.yaml
    assert "default_premier_league" in ids


@pytest.mark.integration
def test_get_pitch_lines_no_stadium_excludes_mow_entries(client) -> None:
    c, _ = client
    resp = c.get("/pitch_lines")
    names = [ln["name"] for ln in resp.json()["lines"]]
    assert not any(n.startswith("mow_y_") for n in names)
    assert not any(n.startswith("mow_x_") for n in names)


@pytest.mark.integration
def test_get_pitch_lines_with_stadium_includes_mow_entries(client) -> None:
    c, _ = client
    resp = c.get("/pitch_lines", params={"stadium": "default_premier_league"})
    body = resp.json()
    names = [ln["name"] for ln in body["lines"]]
    mow = [n for n in names if n.startswith("mow_y_")]
    assert mow, "expected at least one mow_y_* entry from default_premier_league"
    # Default stadium uses width 5.5 from origin 0 → first inner boundary at y=5.5.
    sample = next(ln for ln in body["lines"] if ln["name"] == "mow_y_5.5")
    assert sample["world_segment"] == [[0.0, 5.5, 0.0], [105.0, 5.5, 0.0]]
    assert sample.get("category") == "mowing"


@pytest.mark.integration
def test_get_pitch_lines_with_unknown_stadium_falls_back_to_static(client) -> None:
    c, _ = client
    resp = c.get("/pitch_lines", params={"stadium": "no_such_stadium"})
    names = [ln["name"] for ln in resp.json()["lines"]]
    assert "halfway_line" in names
    assert not any(n.startswith("mow_y_") for n in names)


@pytest.mark.integration
def test_post_anchors_round_trips_stadium(client) -> None:
    c, tmp = client
    payload = {
        "clip_id": "play_037",
        "image_size": [1920, 1080],
        "stadium": "default_premier_league",
        "anchors": [{"frame": 0, "landmarks": []}],
    }
    resp = c.post("/anchors", json=payload)
    assert resp.status_code == 200
    saved = json.loads((tmp / "camera" / "anchors.json").read_text())
    assert saved["stadium"] == "default_premier_league"
    # Round-trip via GET
    got = c.get("/anchors").json()
    assert got["stadium"] == "default_premier_league"
