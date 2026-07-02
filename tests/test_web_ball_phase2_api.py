"""Phase-2 web API: suggest endpoints + landmark/shot_chains persistence."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from src.schemas.camera_track import CameraFrame, CameraTrack
from src.web.server import create_app


def _client(tmp_path: Path) -> TestClient:
    return TestClient(create_app(output_dir=tmp_path, config_path=None))


def _camera_pose():
    look = np.array([0.0, 64.0, -30.0])
    look /= np.linalg.norm(look)
    right = np.array([1.0, 0.0, 0.0])
    down = np.cross(look, right)
    R = np.array([right, down, look], dtype=float)
    t = -R @ np.array([52.5, -30.0, 30.0])
    K = np.array([[1500.0, 0, 640.0], [0, 1500.0, 360.0], [0, 0, 1.0]])
    return K, R, t


def _save_camera(tmp_path: Path, shot: str = "play", n: int = 10) -> tuple:
    K, R, t = _camera_pose()
    CameraTrack(
        clip_id=shot, fps=30.0, image_size=(1280, 720),
        t_world=t.tolist(),
        frames=tuple(
            CameraFrame(frame=i, K=K.tolist(), R=R.tolist(),
                        confidence=1.0, is_anchor=(i == 0))
            for i in range(n)),
    ).save(tmp_path / "camera" / f"{shot}_camera_track.json")
    return K, R, t


def _project(p, K, R, t):
    cam = R @ np.asarray(p, dtype=float) + t
    pix = K @ cam
    return float(pix[0] / pix[2]), float(pix[1] / pix[2])


def test_goal_element_suggest_ranks_by_residual(tmp_path: Path):
    K, R, t = _save_camera(tmp_path)
    # Pixel of the near-goal left post at mid height (x=0, y=30.34, z=1.2).
    u, v = _project(np.array([0.0, 30.34, 1.2]), K, R, t)
    r = _client(tmp_path).get(
        "/goal-element-suggest",
        params={"shot": "play", "frame": 0, "u": u, "v": v})
    assert r.status_code == 200
    cands = r.json()["candidates"]
    assert cands, "expected goal element candidates for a post pixel"
    assert cands[0]["element"] == "post"
    assert cands[0]["residual_m"] < 0.2
    assert len(cands[0]["world_xyz"]) == 3


def test_goal_element_suggest_graceful_without_camera(tmp_path: Path):
    r = _client(tmp_path).get(
        "/goal-element-suggest",
        params={"shot": "play", "frame": 0, "u": 10, "v": 10})
    assert r.status_code == 200
    assert r.json() == {"candidates": []}


def test_pitch_fix_suggest_finds_post_base(tmp_path: Path):
    K, R, t = _save_camera(tmp_path)
    u, v = _project(np.array([0.2, 30.3, 0.11]), K, R, t)
    r = _client(tmp_path).get(
        "/pitch-fix-suggest",
        params={"shot": "play", "frame": 0, "u": u, "v": v})
    assert r.status_code == 200
    body = r.json()
    assert body["ground_xy"] is not None
    names = [s["name"] for s in body["suggestions"]]
    assert "left_goal_left_post_base" in names


def test_pitch_fix_suggest_graceful_without_camera(tmp_path: Path):
    r = _client(tmp_path).get(
        "/pitch-fix-suggest",
        params={"shot": "play", "frame": 0, "u": 10, "v": 10})
    assert r.status_code == 200
    assert r.json() == {"ground_xy": None, "suggestions": []}


def test_post_persists_landmark_and_shot_chains(tmp_path: Path):
    client = _client(tmp_path)
    payload = {
        "clip_id": "play", "image_size": [1280, 720],
        "anchors": [
            {"frame": 5, "image_xy": [100.0, 200.0], "state": "grounded",
             "landmark": "left_goal_left_post_base"},
            {"frame": 30, "image_xy": [300.0, 400.0], "state": "player_touch",
             "player_id": "P1", "bone": "r_foot", "touch_type": "shot"},
            {"frame": 55, "image_xy": [500.0, 300.0], "state": "goal_impact",
             "goal_element": "back_net"},
        ],
        "shot_chains": [[30, 55]],
    }
    r = client.post("/ball-anchors/play", json=payload)
    assert r.status_code == 200, r.text
    got = client.get("/ball-anchors/play").json()
    assert got["anchors"][0]["landmark"] == "left_goal_left_post_base"
    assert got["shot_chains"] == [[30, 55]]
