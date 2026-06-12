"""Reel upload endpoint: save to output/source/ + spawn split job."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.web.server import create_app
from tests.fixtures.synthetic_reel import build_reel


@pytest.fixture
def client(tmp_path: Path):
    app = create_app(output_dir=tmp_path, config_path=None)
    return TestClient(app), tmp_path


def _tiny_mp4_bytes(tmp_path: Path) -> bytes:
    p = tmp_path / "_upload_src.mp4"
    build_reel(p, [("green", 1.0)])
    return p.read_bytes()


def _wait_for_job(c: TestClient, job_id: str, timeout_s: float = 15.0) -> str:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        status = c.get(f"/api/jobs/{job_id}/status").json()["status"]
        if status not in ("running", "queued"):
            return status
        time.sleep(0.05)
    raise TimeoutError(f"job {job_id} still running after {timeout_s}s")


@pytest.mark.integration
def test_upload_reel_saves_and_spawns_job(client, monkeypatch) -> None:
    c, tmp = client
    calls: dict = {}
    monkeypatch.setattr(
        "src.web.server.run_pipeline", lambda **kw: calls.update(kw),
    )
    res = c.post(
        "/api/shots/upload-reel",
        files={"file": ("My Reel (1).mp4", _tiny_mp4_bytes(tmp), "video/mp4")},
    )
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["job_id"]
    assert body["saved"] == "MyReel1"

    source_dir = tmp / "source"
    assert (source_dir / "MyReel1.mp4").exists()

    assert _wait_for_job(c, body["job_id"]) == "done"
    assert calls["stages"] == "prepare_shots"
    assert Path(calls["video_path"]) == source_dir / "MyReel1.mp4"


@pytest.mark.integration
def test_upload_reel_rejects_non_mp4(client) -> None:
    c, _ = client
    res = c.post(
        "/api/shots/upload-reel",
        files={"file": ("x.avi", b"xx", "video/avi")},
    )
    assert res.status_code == 400


@pytest.mark.integration
def test_upload_reel_rejects_duplicate_source(client, monkeypatch) -> None:
    c, tmp = client
    monkeypatch.setattr("src.web.server.run_pipeline", lambda **kw: None)
    payload = _tiny_mp4_bytes(tmp)
    first = c.post("/api/shots/upload-reel",
                   files={"file": ("reel.mp4", payload, "video/mp4")})
    assert first.status_code == 200
    _wait_for_job(c, first.json()["job_id"])
    second = c.post("/api/shots/upload-reel",
                    files={"file": ("reel.mp4", payload, "video/mp4")})
    assert second.status_code == 409


@pytest.mark.integration
def test_run_rejects_input_path_outside_source(client) -> None:
    c, _ = client
    res = c.post("/api/run", json={
        "stages": "prepare_shots",
        "input_path": "/etc/passwd",
    })
    assert res.status_code == 400


@pytest.mark.integration
def test_resplit_respawns_job_on_recorded_source(client, monkeypatch) -> None:
    """POST /api/shots/resplit re-runs split mode on the source video
    already recorded in the manifest — no re-upload."""
    c, tmp = client
    calls: dict = {}
    monkeypatch.setattr(
        "src.web.server.run_pipeline", lambda **kw: calls.update(kw),
    )
    # seed: a manifest pointing at an on-disk source video
    source = tmp / "source"
    source.mkdir()
    reel = source / "reel.mp4"
    reel.write_bytes(_tiny_mp4_bytes(tmp))
    (tmp / "shots").mkdir()
    (tmp / "shots" / "shots_manifest.json").write_text(json.dumps({
        "source_file": str(reel), "fps": 25.0, "total_frames": 100,
        "shots": [{"id": "s001", "start_frame": 0, "end_frame": 99,
                   "start_time": 0.0, "end_time": 4.0,
                   "clip_file": "shots/s001.mp4"}],
    }))

    res = c.post("/api/shots/resplit")
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["job_id"]
    assert body["source"] == str(reel)
    assert _wait_for_job(c, body["job_id"]) == "done"
    assert Path(calls["video_path"]) == reel
    assert calls["force_resplit"] is True


@pytest.mark.integration
def test_resplit_404_without_source(client) -> None:
    c, _ = client
    assert c.post("/api/shots/resplit").status_code == 404
