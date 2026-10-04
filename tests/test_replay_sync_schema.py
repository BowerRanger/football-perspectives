"""playback_rate on Alignment / GroupSync mapping / Shot retime fields."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.schemas.shots import Shot, ShotsManifest
from src.schemas.sync_map import Alignment, GroupSync, SyncMap
from src.web.server import create_app


def test_old_sync_file_loads_with_rate_one(tmp_path: Path):
    p = tmp_path / "sync_map.json"
    p.write_text(json.dumps({"version": 2, "groups": [{
        "group_id": "g", "reference_shot": "a",
        "alignments": [{"shot_id": "b", "frame_offset": 5, "method": "manual",
                        "confidence": 1.0}]}]}))
    a = SyncMap.load(p).group("g").alignments[0]
    assert a.playback_rate == 1.0


def test_rate_round_trips_and_unknown_keys_ignored(tmp_path: Path):
    p = tmp_path / "s.json"
    SyncMap(groups=[GroupSync("g", "a", [Alignment("b", -10, "player_formation", 0.8, 0.34)])]).save(p)
    d = json.loads(p.read_text())
    d["groups"][0]["alignments"][0]["future_field"] = 1
    p.write_text(json.dumps(d))
    assert SyncMap.load(p).group("g").alignments[0].playback_rate == 0.34


def test_frame_mapping_inverse():
    g = GroupSync("g", "a", [Alignment("a", 0), Alignment("b", -20, playback_rate=0.5)])
    assert g.ref_frame_of("b", 10) == pytest.approx(25.0)
    assert g.shot_frame_of("b", 25.0) == pytest.approx(10.0)
    assert g.ref_frame_of("a", 7) == 7
    assert g.ref_frame_of("zz", 7) == 7  # unknown shot: identity


def test_shot_retimed_fields_back_compatible(tmp_path: Path):
    p = tmp_path / "m.json"
    ShotsManifest(source_file="x", fps=25, total_frames=1,
                  shots=[Shot("a", 0, 0, 0, 0, "a.mp4", retimed=True, native_frames=9)]).save(p)
    assert ShotsManifest.load(p).shots[0].native_frames == 9
    d = json.loads(p.read_text())
    del d["shots"][0]["retimed"], d["shots"][0]["native_frames"]
    p.write_text(json.dumps(d))
    s = ShotsManifest.load(p).shots[0]
    assert s.retimed is False and s.native_frames == 0


def test_sync_endpoints_never_drop_rate(tmp_path: Path):
    (tmp_path / "shots").mkdir()
    (tmp_path / "shots" / "shots_manifest.json").write_text(json.dumps({
        "source_file": "x", "fps": 25, "total_frames": 0, "groups": [],
        "shots": [{"id": s, "start_frame": 0, "end_frame": 0, "start_time": 0,
                   "end_time": 0, "clip_file": f"shots/{s}.mp4", "group_id": "g"}
                  for s in ("a", "b")]}))
    c = TestClient(create_app(output_dir=tmp_path, config_path=None))
    body = {"group_id": "g", "reference_shot": "a", "alignments": [
        {"shot_id": "a", "frame_offset": 0},
        {"shot_id": "b", "frame_offset": -9, "playback_rate": 0.4}]}
    assert c.post("/api/sync", json=body).status_code == 200
    # an older client omits the rate: the saved one survives
    del body["alignments"][1]["playback_rate"]
    assert c.post("/api/sync", json=body).status_code == 200
    g = next(g for g in c.get("/api/sync").json()["groups"] if g["group_id"] == "g")
    assert next(a for a in g["alignments"] if a["shot_id"] == "b")["playback_rate"] == 0.4
    body["alignments"][1]["playback_rate"] = 0
    assert c.post("/api/sync", json=body).status_code == 422
