"""Job cancel (POST /api/jobs/{id}/cancel) and track-edit undo
(GET/POST /api/tracks/undo)."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.web.server import Job, _jobs, _jobs_lock, create_app


def _wait_for(pred, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(0.02)
    return False


# ---------------------------------------------------------------------------
# Cancel
# ---------------------------------------------------------------------------


@pytest.fixture
def slow_client(tmp_path: Path, monkeypatch):
    """run_pipeline that spins in Python until interrupted (like a stage loop)."""
    started = {"n": 0}

    def fake_run_pipeline(**_kw):
        started["n"] += 1
        for _ in range(2000):  # ~20 s ceiling if cancel never lands
            time.sleep(0.01)

    monkeypatch.setattr("src.web.server.run_pipeline", fake_run_pipeline)
    app = create_app(output_dir=tmp_path, config_path=None)
    return TestClient(app), started


@pytest.mark.integration
def test_cancel_interrupts_running_job(slow_client) -> None:
    c, started = slow_client
    job_id = c.post("/api/run", json={"stages": "camera"}).json()["job_id"]
    assert _wait_for(lambda: started["n"] == 1)

    r = c.post(f"/api/jobs/{job_id}/cancel")

    assert r.status_code == 202, r.text
    assert r.json()["status"] == "cancelling"
    assert _wait_for(lambda: c.get(f"/api/jobs/{job_id}/status").json()["status"] == "cancelled")
    logs = _jobs[job_id].log_lines
    assert any("cancelled" in line.lower() for line in logs)


@pytest.mark.integration
def test_cancelled_job_streams_cancelled_done_event(slow_client) -> None:
    c, started = slow_client
    job_id = c.post("/api/run", json={"stages": "camera"}).json()["job_id"]
    assert _wait_for(lambda: started["n"] == 1)
    c.post(f"/api/jobs/{job_id}/cancel")
    assert _wait_for(lambda: _jobs[job_id].status == "cancelled")

    body = c.get(f"/api/jobs/{job_id}/logs").text

    assert 'event: done\ndata: {"status": "cancelled"}' in body


@pytest.mark.integration
def test_cancel_unknown_and_finished_jobs(tmp_path: Path) -> None:
    c = TestClient(create_app(output_dir=tmp_path, config_path=None))
    assert c.post("/api/jobs/nope/cancel").status_code == 404
    done = Job(job_id="finished1", stages="camera", status="done")
    with _jobs_lock:
        _jobs[done.job_id] = done
    try:
        assert c.post("/api/jobs/finished1/cancel").status_code == 409
    finally:
        with _jobs_lock:
            _jobs.pop(done.job_id, None)


# ---------------------------------------------------------------------------
# Track undo
# ---------------------------------------------------------------------------


def _track(tid: str, pid: str, name: str, frames: range) -> dict:
    return {
        "track_id": tid,
        "class_name": "player",
        "team": "A",
        "player_id": pid,
        "player_name": name,
        "frames": [{"frame": f, "bbox": [0, 0, 10, 20], "confidence": 0.9, "pitch_position": None} for f in frames],
    }


@pytest.fixture
def tracks_client(tmp_path: Path):
    tdir = tmp_path / "tracks"
    tdir.mkdir()
    doc = {"shot_id": "s1", "tracks": [
        _track("T001", "P001", "Salah", range(0, 10)),
        _track("T002", "P002", "Salah", range(12, 20)),
        _track("T003", "P003", "", range(0, 5)),
    ]}
    (tdir / "s1_tracks.json").write_text(json.dumps(doc))
    other = {"shot_id": "s2", "tracks": [_track("T001", "P009", "ignore", range(0, 3))]}
    (tdir / "s2_tracks.json").write_text(json.dumps(other))
    return TestClient(create_app(output_dir=tmp_path, config_path=None)), tdir


def _ids(tdir: Path, shot: str) -> list[str]:
    return sorted(t["track_id"] for t in json.loads((tdir / f"{shot}_tracks.json").read_text())["tracks"])


@pytest.mark.integration
def test_undo_restores_bulk_delete(tracks_client) -> None:
    c, tdir = tracks_client
    before = (tdir / "s1_tracks.json").read_bytes()
    r = c.post("/api/tracks/s1/delete-bulk", json={"track_ids": ["T001", "T003"]})
    assert r.status_code == 200
    assert r.json()["undo_id"]
    assert _ids(tdir, "s1") == ["T002"]

    u = c.post("/api/tracks/undo", json={"undo_id": r.json()["undo_id"]})

    assert u.status_code == 200, u.text
    assert (tdir / "s1_tracks.json").read_bytes() == before
    assert c.get("/api/tracks/undo").json() == []


@pytest.mark.integration
def test_undo_is_a_stack_across_operations(tracks_client) -> None:
    c, tdir = tracks_client
    c.post("/api/tracks/merge", json={"shot_id": "s1", "track_ids": ["T001", "T002"]})
    c.delete("/api/tracks/s1/T003")
    history = c.get("/api/tracks/undo").json()
    assert [h["label"] for h in history][:2] == ["Delete T003", "Merge T001 + T002"]

    c.post("/api/tracks/undo")  # newest first: restores T003
    assert _ids(tdir, "s1") == ["T001", "T003"]
    c.post("/api/tracks/undo")  # then un-merge
    assert _ids(tdir, "s1") == ["T001", "T002", "T003"]


@pytest.mark.integration
def test_undo_restores_multi_file_operations(tracks_client) -> None:
    c, tdir = tracks_client
    before = {p.name: p.read_bytes() for p in tdir.glob("*_tracks.json")}
    c.post("/api/tracks/merge-by-name")
    c.post("/api/tracks/delete-ignored")
    assert _ids(tdir, "s2") == []

    c.post("/api/tracks/undo")
    c.post("/api/tracks/undo")

    assert {p.name: p.read_bytes() for p in tdir.glob("*_tracks.json")} == before


@pytest.mark.integration
def test_undo_rejects_stale_id_and_empty_stack(tracks_client) -> None:
    c, _ = tracks_client
    assert c.post("/api/tracks/undo").status_code == 404
    first = c.delete("/api/tracks/s1/T003").json()["undo_id"]
    c.post("/api/tracks/s1/delete-bulk", json={"track_ids": ["T001"]})

    r = c.post("/api/tracks/undo", json={"undo_id": first})

    assert r.status_code == 409, "undo must only pop the newest operation"
