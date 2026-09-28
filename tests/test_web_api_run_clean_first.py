"""POST /api/run ``clean_first`` and GET /api/jobs.

``clean_first`` backs the dashboard's "Re-run stage": the stage's generated
outputs are cleared only once the run has been accepted, so a rejected run
(409 hmr_world already running) never loses output. ``GET /api/jobs`` lets
a reloaded dashboard reattach to an in-flight run.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.web.server import Job, _jobs, _jobs_lock, create_app


@pytest.fixture
def client(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("src.web.server.run_pipeline", lambda *a, **k: None)
    app = create_app(output_dir=tmp_path, config_path=None)
    return TestClient(app), tmp_path


@pytest.fixture
def running_hmr_job():
    fake = Job(job_id="hmrbusy1", stages="hmr_world")
    fake.status = "running"
    with _jobs_lock:
        _jobs[fake.job_id] = fake
    yield fake
    with _jobs_lock:
        _jobs.pop(fake.job_id, None)


def _seed_hmr_output(root: Path) -> Path:
    out = root / "hmr_world" / "P001_smpl_world.npz"
    out.parent.mkdir(parents=True)
    out.write_bytes(b"npz")
    return out


@pytest.mark.integration
def test_clean_first_clears_outputs_after_acceptance(client) -> None:
    c, root = client
    cam = root / "camera"
    cam.mkdir()
    track = cam / "gberch_camera_track.json"
    track.write_text("{}")
    anchors = cam / "gberch_anchors.json"
    anchors.write_text('{"anchors": []}')

    r = c.post("/api/run", json={"stages": "camera", "clean_first": True})

    assert r.status_code == 202, r.text
    assert not track.exists(), "generated camera track must be cleared"
    assert anchors.exists(), "operator anchors must survive a re-run"


@pytest.mark.integration
def test_clean_first_keeps_outputs_when_run_rejected(client, running_hmr_job) -> None:
    c, root = client
    npz = _seed_hmr_output(root)

    r = c.post("/api/run", json={"stages": "hmr_world", "clean_first": True})

    assert r.status_code == 409
    assert npz.exists(), "a rejected run must not delete the stage's output"


@pytest.mark.integration
def test_clean_first_requires_single_stage(client) -> None:
    c, root = client
    npz = _seed_hmr_output(root)

    r = c.post("/api/run", json={"stages": "all", "clean_first": True})

    assert r.status_code == 400
    assert npz.exists()


@pytest.mark.integration
def test_run_shot_keeps_outputs_when_rejected(client, running_hmr_job) -> None:
    c, root = client
    per_shot = root / "hmr_world" / "gberch_P001_smpl_world.npz"
    per_shot.parent.mkdir(parents=True)
    per_shot.write_bytes(b"npz")
    pair = root / "hmr_world" / "gberch__P001_kp2d.json"
    pair.write_text("{}")

    r1 = c.post("/api/run-shot", json={"stage": "hmr_world", "shot_id": "gberch"})
    r2 = c.post("/api/run-shot-player", json={"shot_id": "gberch", "player_id": "P001"})

    assert r1.status_code == 409 and r2.status_code == 409
    assert per_shot.exists() and pair.exists(), "rejected per-shot runs must not delete output"


@pytest.mark.integration
def test_run_without_clean_first_keeps_outputs(client) -> None:
    c, root = client
    npz = _seed_hmr_output(root)

    r = c.post("/api/run", json={"stages": "hmr_world", "from_stage": "hmr_world"})

    assert r.status_code == 202
    assert npz.exists(), "Continue must never clear output"


@pytest.mark.integration
def test_list_jobs_filters_running_newest_first(client, running_hmr_job) -> None:
    c, _ = client
    done = Job(job_id="olddone1", stages="camera", status="done", started_at=1.0)
    with _jobs_lock:
        _jobs[done.job_id] = done
    try:
        running = c.get("/api/jobs", params={"status": "running"}).json()
        everything = c.get("/api/jobs").json()
    finally:
        with _jobs_lock:
            _jobs.pop(done.job_id, None)

    running_ids = [j["job_id"] for j in running]
    assert "hmrbusy1" in running_ids
    assert "olddone1" not in running_ids
    all_ids = [j["job_id"] for j in everything]
    assert all_ids.index("hmrbusy1") < all_ids.index("olddone1")
    assert {"job_id", "stages", "status", "started_at"} <= set(everything[0])


@pytest.mark.integration
def test_artifacts_dry_run_lists_outputs_but_not_operator_input(client) -> None:
    c, root = client
    cam = root / "camera"
    (cam / "debug").mkdir(parents=True)
    (cam / "gberch_camera_track.json").write_text("{}")
    anchors = cam / "gberch_anchors.json"
    anchors.write_text('{"anchors": []}')

    r = c.get("/api/output/camera/artifacts")

    assert r.status_code == 200
    paths = {p["path"]: p["is_dir"] for p in r.json()["paths"]}
    assert paths == {"camera/gberch_camera_track.json": False, "camera/debug": True}
    assert anchors.exists() and (cam / "debug").exists(), "dry run must not delete"
    assert c.get("/api/output/nope/artifacts").status_code == 404


@pytest.mark.integration
def test_stages_report_partial_output(client) -> None:
    c, root = client
    _seed_hmr_output(root)

    stages = {s["name"]: s for s in c.get("/api/stages").json()}

    assert stages["hmr_world"]["complete"] is False
    assert stages["hmr_world"]["partial"] is True
    assert stages["camera"]["partial"] is False, "no output means not partial"
