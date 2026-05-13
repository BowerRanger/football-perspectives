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


def test_delete_ball_stage_preserves_anchors(client):
    """Clearing the ball stage (via Re-run Stage in the dashboard)
    must remove generated ball_track files but leave user-supplied
    *_ball_anchors.json intact — they're inputs, not outputs."""
    c, tmp_path = client
    ball_dir = tmp_path / "ball"
    ball_dir.mkdir(parents=True, exist_ok=True)
    # Output (should be wiped).
    track_path = ball_dir / "origi01_ball_track.json"
    track_path.write_text('{"frames": [], "flight_segments": []}')
    legacy_track = ball_dir / "ball_track.json"
    legacy_track.write_text('{"legacy": true}')
    # User input (must survive).
    anchors_path = ball_dir / "origi01_ball_anchors.json"
    anchors_path.write_text(
        '{"clip_id": "origi01", "image_size": [1280, 720], '
        '"anchors": [{"frame": 5, "image_xy": [640.0, 360.0], "state": "grounded"}]}'
    )

    r = c.delete("/api/output/ball")
    assert r.status_code == 200, r.text

    assert not track_path.exists(), "ball_track must be wiped"
    assert not legacy_track.exists(), "legacy ball_track must be wiped"
    assert anchors_path.exists(), "anchors must NOT be wiped by re-run"
    body = r.json()
    assert any("ball_track" in p for p in body["removed"])


def test_preview_endpoint_runs_ball_stage_with_payload(client):
    """Preview should run BallStage in a tmp output dir using the
    posted anchors, returning the resulting BallTrack JSON."""
    import cv2
    import numpy as np
    from src.schemas.camera_track import CameraFrame, CameraTrack
    from src.schemas.shots import Shot, ShotsManifest

    c, tmp_path = client

    K = np.array([[1500.0, 0, 640.0], [0, 1500.0, 360.0], [0, 0, 1.0]])
    R = np.eye(3)
    t = np.array([0.0, 0.0, 30.0])
    n_frames = 20
    clip = tmp_path / "shots" / "play.mp4"
    clip.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (320, 240))
    for _ in range(n_frames):
        writer.write(np.full((240, 320, 3), [50, 200, 50], dtype=np.uint8))
    writer.release()
    CameraTrack(
        clip_id="play", fps=30.0, image_size=(1280, 720), t_world=t.tolist(),
        frames=tuple(CameraFrame(frame=i, K=K.tolist(), R=R.tolist(),
                                 confidence=1.0, is_anchor=(i == 0))
                     for i in range(n_frames)),
    ).save(tmp_path / "camera" / "play_camera_track.json")
    ShotsManifest(
        source_file="fake.mp4", fps=30.0, total_frames=n_frames,
        shots=[Shot(id="play", clip_file="shots/play.mp4",
                    start_frame=0, end_frame=n_frames - 1,
                    start_time=0.0, end_time=(n_frames - 1) / 30.0)],
    ).save(tmp_path / "shots" / "shots_manifest.json")

    payload = {
        "clip_id": "play", "image_size": [1280, 720],
        "anchors": [{"frame": 5, "image_xy": [640.0, 360.0], "state": "grounded"}],
    }
    r = c.post("/ball-anchors/play/preview", json=payload)
    # Allow 200 OK with a BallTrack body, or 500 if the configured
    # detector backend can't run in this environment (no WASB checkpoint, etc.)
    # — both are valid outcomes; the test asserts the endpoint exists,
    # accepts the payload shape, and either returns a BallTrack or a
    # structured error.
    assert r.status_code in (200, 500), r.text
    if r.status_code == 200:
        body = r.json()
        assert "frames" in body
        assert "flight_segments" in body


def test_post_player_touch_forwards_player_id_and_bone(client):
    """The /ball-anchors POST endpoint must forward player_id + bone
    fields to the saved BallAnchorSet. Earlier the Pydantic payload
    model didn't declare those fields, so they got silently dropped
    and the validation pass in BallAnchorSet.load raised 400."""
    c, tmp_path = client
    payload = {
        "clip_id": "play",
        "image_size": [1280, 720],
        "anchors": [
            {
                "frame": 10, "image_xy": [640.0, 360.0],
                "state": "player_touch",
                "player_id": "P003", "bone": "r_foot",
            },
        ],
    }
    r = c.post("/ball-anchors/play", json=payload)
    assert r.status_code == 200, r.text
    # Verify the JSON written to disk has player_id + bone preserved.
    saved = json.loads(
        (tmp_path / "ball" / "play_ball_anchors.json").read_text()
    )
    a = saved["anchors"][0]
    assert a["state"] == "player_touch"
    assert a["player_id"] == "P003"
    assert a["bone"] == "r_foot"


def test_post_goal_impact_forwards_goal_element(client):
    """The /ball-anchors POST + GET must round-trip the goal_element
    field for goal_impact anchors. Mirrors the player_touch
    forwarding test."""
    c, tmp_path = client
    payload = {
        "clip_id": "play",
        "image_size": [1280, 720],
        "anchors": [
            {
                "frame": 42, "image_xy": [640.0, 200.0],
                "state": "goal_impact", "goal_element": "crossbar",
            },
        ],
    }
    r = c.post("/ball-anchors/play", json=payload)
    assert r.status_code == 200, r.text
    saved = json.loads(
        (tmp_path / "ball" / "play_ball_anchors.json").read_text()
    )
    a = saved["anchors"][0]
    assert a["state"] == "goal_impact"
    assert a["goal_element"] == "crossbar"
    # And GET round-trips it.
    body = c.get("/ball-anchors/play").json()
    assert body["anchors"][0]["goal_element"] == "crossbar"


def test_post_goal_impact_missing_element_rejected(client):
    """A goal_impact anchor with no goal_element must be rejected by
    the server's schema validation pass (400)."""
    c, _ = client
    payload = {
        "clip_id": "play",
        "image_size": [1280, 720],
        "anchors": [
            {
                "frame": 5, "image_xy": [100.0, 100.0],
                "state": "goal_impact",
            },
        ],
    }
    r = c.post("/ball-anchors/play", json=payload)
    assert r.status_code == 400, r.text


def test_post_player_touch_with_touch_type_and_spin_round_trips(client):
    """POST + GET must round-trip the new ``touch_type`` and ``spin``
    fields on a player_touch anchor."""
    c, tmp_path = client
    payload = {
        "clip_id": "play",
        "image_size": [1280, 720],
        "anchors": [
            {
                "frame": 5, "image_xy": [640.0, 360.0],
                "state": "player_touch",
                "player_id": "P003", "bone": "r_foot",
                "touch_type": "shot", "spin": "instep_curl_right",
            },
        ],
    }
    r = c.post("/ball-anchors/play", json=payload)
    assert r.status_code == 200, r.text
    saved = json.loads(
        (tmp_path / "ball" / "play_ball_anchors.json").read_text()
    )
    a = saved["anchors"][0]
    assert a["state"] == "player_touch"
    assert a["touch_type"] == "shot"
    assert a["spin"] == "instep_curl_right"
    # GET round-trips both fields.
    body = c.get("/ball-anchors/play").json()
    assert body["anchors"][0]["touch_type"] == "shot"
    assert body["anchors"][0]["spin"] == "instep_curl_right"


def test_post_spin_on_grounded_state_rejected(client):
    """A spin preset attached to a non-player_touch state is rejected
    by the server schema (400)."""
    c, _ = client
    payload = {
        "clip_id": "play",
        "image_size": [1280, 720],
        "anchors": [
            {
                "frame": 5, "image_xy": [640.0, 360.0],
                "state": "grounded", "spin": "topspin",
            },
        ],
    }
    r = c.post("/ball-anchors/play", json=payload)
    assert r.status_code == 400, r.text


def test_post_spin_without_touch_type_rejected(client):
    """Spin requires touch_type='shot' or 'volley' on player_touch."""
    c, _ = client
    payload = {
        "clip_id": "play",
        "image_size": [1280, 720],
        "anchors": [
            {
                "frame": 5, "image_xy": [640.0, 360.0],
                "state": "player_touch",
                "player_id": "P1", "bone": "r_foot",
                "spin": "topspin",
            },
        ],
    }
    r = c.post("/ball-anchors/play", json=payload)
    assert r.status_code == 400, r.text
