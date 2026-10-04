"""Replay-speed endpoints against a tiny synthetic output dir."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.schemas.shots import HighlightGroup, Shot, ShotsManifest
from src.schemas.sync_map import Alignment, GroupSync, SyncMap
from src.web.server import create_app

N = 60


@pytest.fixture
def env(tmp_path: Path):
    out = tmp_path
    (out / "shots").mkdir()
    (out / "tracks").mkdir()
    for sid in ("a", "b"):
        vw = cv2.VideoWriter(str(out / "shots" / f"{sid}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                             25.0, (32, 24))
        for _ in range(N):
            vw.write(np.zeros((24, 32, 3), np.uint8))
        vw.release()
    (out / "tracks" / "b_tracks.json").write_text(json.dumps({"shot_id": "b", "tracks": [
        {"track_id": "T", "class_name": "player", "frames": [{"frame": i, "bbox": [0, 0, 1, 1]} for i in range(N)]}]}))
    ShotsManifest(source_file="x", fps=25.0, total_frames=N, shots=[
        Shot(s, 0, N - 1, 0, 1, f"shots/{s}.mp4", group_id="g" if s != "z" else "h") for s in ("a", "b", "z")],
        groups=[HighlightGroup("g", "g", ["a", "b"])]).save(out / "shots" / "shots_manifest.json")
    SyncMap(groups=[GroupSync("g", "a", [Alignment("a", 0), Alignment("b", 0)])]).save(
        out / "shots" / "sync_map.json")
    return TestClient(create_app(output_dir=out, config_path=None)), out


def test_get_replay_sync_empty_and_present(env):
    c, out = env
    assert c.get("/api/replay-sync").json() == {"version": 1, "groups": []}
    (out / "shots" / "replay_sync.json").write_text(json.dumps({"version": 1, "groups": [{"group_id": "g"}]}))
    assert c.get("/api/replay-sync").json()["groups"][0]["group_id"] == "g"


def test_moments_save_manual_alignment_with_fit(env):
    c, out = env
    body = {"shot_id": "b", "moments": [{"reference_frame": 137, "shot_frame": 28},
                                        {"reference_frame": 182, "shot_frame": 160}]}
    r = c.post("/api/sync/groups/g/moments", json=body)
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["rate"] == pytest.approx(45 / 132)
    assert j["n_moments"] == 2 and j["ramp"] is False
    assert j["residual_frames"] == pytest.approx(0.0, abs=1e-6)
    assert len(j["interval_rates"]) == 1
    al = j["alignment"]
    assert al["method"] == "manual" and al["playback_rate"] == pytest.approx(45 / 132)
    assert al["frame_offset"] == round(-(137 - 28 * 45 / 132))
    assert not j["retimed"]
    saved = SyncMap.load(out / "shots" / "sync_map.json").group("g")
    assert saved.rate_for("b") == pytest.approx(45 / 132)


def test_moments_retime_rebases_alignment(env):
    c, out = env
    body = {"shot_id": "b", "retime": True,
            "moments": [{"reference_frame": 10, "shot_frame": 0}, {"reference_frame": 30, "shot_frame": 40}]}
    j = c.post("/api/sync/groups/g/moments", json=body).json()
    assert j["retimed"] and j["alignment"]["playback_rate"] == 1.0
    assert j["alignment"]["frame_offset"] == -10
    assert j["retime"]["frames_in"] == N and j["retime"]["frames_out"] == int((N - 1) * 0.5) + 1
    assert "tracks/b_tracks.json" in j["retime"]["files_written"]
    shot = next(s for s in ShotsManifest.load(out / "shots" / "shots_manifest.json").shots if s.id == "b")
    assert shot.retimed


@pytest.mark.parametrize("body", [
    {"shot_id": "b", "moments": [{"reference_frame": 1, "shot_frame": 1}]},                 # one moment
    {"shot_id": "b", "moments": [{"reference_frame": 1, "shot_frame": 5}, {"reference_frame": 9, "shot_frame": 5}]},
    {"shot_id": "b", "moments": [{"reference_frame": 9, "shot_frame": 1}, {"reference_frame": 3, "shot_frame": 8}]},
    {"shot_id": "b", "moments": [{"reference_frame": -1, "shot_frame": 1}, {"reference_frame": 3, "shot_frame": 8}]},
    {"shot_id": "a", "moments": [{"reference_frame": 1, "shot_frame": 1}, {"reference_frame": 3, "shot_frame": 8}]},  # reference
    {"shot_id": "z", "moments": [{"reference_frame": 1, "shot_frame": 1}, {"reference_frame": 3, "shot_frame": 8}]},  # other group
    {"moments": []},
])
def test_moments_validation_422(env, body):
    c, _ = env
    assert c.post("/api/sync/groups/g/moments", json=body).status_code == 422


def test_moments_unknown_group_404(env):
    c, _ = env
    body = {"shot_id": "b", "moments": [{"reference_frame": 1, "shot_frame": 1}, {"reference_frame": 3, "shot_frame": 8}]}
    assert c.post("/api/sync/groups/nope/moments", json=body).status_code == 404


def test_retime_and_restore_round_trip(env):
    c, out = env
    c.post("/api/sync", json={"group_id": "g", "reference_shot": "a", "alignments": [
        {"shot_id": "a", "frame_offset": 0}, {"shot_id": "b", "frame_offset": -7, "playback_rate": 0.5}]})
    r = c.post("/api/shots/b/retime", json={"rate": 0.5})
    assert r.status_code == 200, r.text
    assert r.json()["retime"]["frames_out"] == int((N - 1) * 0.5) + 1
    assert r.json()["alignment"]["playback_rate"] == 1.0
    assert r.json()["alignment"]["frame_offset"] == -7
    r = c.post("/api/shots/b/restore-native")
    assert r.status_code == 200 and r.json()["restored"]
    assert r.json()["alignment"]["playback_rate"] == pytest.approx(0.5)
    assert c.post("/api/shots/b/restore-native").status_code == 409  # no longer retimed
    cap = cv2.VideoCapture(str(out / "shots" / "b.mp4"))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == N


@pytest.mark.parametrize("rate", [0, -1, 1.5, 0.01])
def test_retime_bad_rate_422(env, rate):
    c, _ = env
    assert c.post("/api/shots/b/retime", json={"rate": rate}).status_code == 422


def test_retime_unknown_shot_404(env):
    c, _ = env
    assert c.post("/api/shots/nope/retime", json={"rate": 0.5}).status_code == 404
    assert c.post("/api/shots/nope/restore-native").status_code == 404
