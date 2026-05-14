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
def test_delete_camera_stage_wipes_per_shot_tracks_keeps_anchors(client) -> None:
    """Clearing the camera stage (Re-run Stage button) must remove
    every per-shot ``{shot}_camera_track.json`` and the debug dir but
    leave user-placed ``{shot}_anchors.json`` files intact — they're
    inputs to the solver, not outputs."""
    c, tmp_path = client
    cam_dir = tmp_path / "camera"
    cam_dir.mkdir(parents=True, exist_ok=True)
    # Per-shot solver outputs (must be wiped).
    track_a = cam_dir / "origi01_camera_track.json"
    track_a.write_text('{"frames": []}')
    track_b = cam_dir / "gberch_camera_track.json"
    track_b.write_text('{"frames": []}')
    # Legacy single-shot path — also wiped if present.
    legacy_track = cam_dir / "camera_track.json"
    legacy_track.write_text('{"legacy": true}')
    # Debug subdir (must be wiped).
    debug = cam_dir / "debug"
    debug.mkdir()
    (debug / "frame_0.png").write_bytes(b"\x89PNG\r\n")
    # User-placed anchors (must survive).
    anchors_a = cam_dir / "origi01_anchors.json"
    anchors_a.write_text('{"anchors": []}')
    anchors_b = cam_dir / "gberch_anchors.json"
    anchors_b.write_text('{"anchors": []}')

    r = c.delete("/api/output/camera")
    assert r.status_code == 200, r.text

    assert not track_a.exists(), "per-shot camera_track must be wiped"
    assert not track_b.exists(), "per-shot camera_track must be wiped"
    assert not legacy_track.exists(), "legacy camera_track must be wiped"
    assert not debug.exists(), "debug dir must be wiped"
    assert anchors_a.exists(), "anchors must NOT be wiped by re-run"
    assert anchors_b.exists(), "anchors must NOT be wiped by re-run"
    removed = r.json()["removed"]
    assert any("camera_track" in p for p in removed)


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


# ── Track-annotation endpoints: physical-merge regressions ───────────


def _seed_tracks(tmp: Path, *tracks_per_track: tuple[str, list[tuple[int, float]]]) -> None:
    """Write a minimal play_tracks.json into ``tmp``.

    Each input is ``(track_id, [(frame, confidence), ...])``. Bbox + team
    are constants because the merge endpoint doesn't consult them.
    """
    from src.schemas.tracks import Track, TrackFrame, TracksResult

    (tmp / "tracks").mkdir(parents=True, exist_ok=True)
    tracks = [
        Track(
            track_id=tid,
            class_name="player",
            team="A",
            player_id="",
            player_name="",
            frames=[
                TrackFrame(frame=fi, bbox=[0, 0, 10, 10], confidence=conf, pitch_position=None)
                for fi, conf in frames
            ],
        )
        for tid, frames in tracks_per_track
    ]
    TracksResult(shot_id="play", tracks=tracks).save(tmp / "tracks" / "play_tracks.json")


def _load_tracks(tmp: Path):
    from src.schemas.tracks import TracksResult

    return TracksResult.load(tmp / "tracks" / "play_tracks.json")


@pytest.mark.integration
def test_merge_physically_combines_tracks(client) -> None:
    c, tmp = client
    # T001 covers frames 0–4, T007 covers frames 5–9 — disjoint, the
    # ByteTrack-dropped-and-restarted-the-same-player scenario.
    _seed_tracks(
        tmp,
        ("T001", [(i, 0.9) for i in range(5)]),
        ("T007", [(i, 0.8) for i in range(5, 10)]),
    )
    resp = c.post(
        "/api/tracks/merge",
        json={"shot_id": "play", "track_ids": ["T001", "T007"]},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    # Canonical track_id is the lex-smallest of the inputs.
    assert body["merged_into"] == "T001"
    assert body["removed_track_ids"] == ["T007"]
    assert body["frame_collisions"] == 0
    assert body["player_id"].startswith("P")  # freshly minted

    tr = _load_tracks(tmp)
    assert len(tr.tracks) == 1, "originals should have been dropped"
    kept = tr.tracks[0]
    assert kept.track_id == "T001"
    assert [f.frame for f in kept.frames] == list(range(10))
    assert kept.player_id == body["player_id"]


@pytest.mark.integration
def test_merge_resolves_frame_collisions_by_confidence(client) -> None:
    c, tmp = client
    # Both tracks claim frame 5; the higher-confidence observation wins.
    _seed_tracks(
        tmp,
        ("T001", [(5, 0.5), (6, 0.9)]),
        ("T002", [(5, 0.95), (7, 0.9)]),
    )
    resp = c.post(
        "/api/tracks/merge",
        json={"shot_id": "play", "track_ids": ["T001", "T002"]},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["merged_into"] == "T001"
    assert body["frame_collisions"] == 1

    tr = _load_tracks(tmp)
    assert len(tr.tracks) == 1
    kept = tr.tracks[0]
    by_frame = {f.frame: f.confidence for f in kept.frames}
    assert by_frame == {5: pytest.approx(0.95), 6: pytest.approx(0.9), 7: pytest.approx(0.9)}


@pytest.mark.integration
def test_merge_by_name_consolidates_tracks(client) -> None:
    c, tmp = client
    _seed_tracks(
        tmp,
        ("T001", [(0, 0.9), (1, 0.9)]),
        ("T002", [(2, 0.8), (3, 0.8)]),
        ("T003", [(0, 0.7), (1, 0.7)]),
    )
    # Name T001 + T002 "Salah", T003 "Mané".
    for tid in ("T001", "T002"):
        c.patch(f"/api/tracks/play/{tid}", json={"player_name": "Salah"})
    c.patch("/api/tracks/play/T003", json={"player_name": "Mané"})

    resp = c.post("/api/tracks/merge-by-name")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["merged_groups"] == 2
    assert body["tracks_removed"] == 1  # only Salah's two collapse to one

    tr = _load_tracks(tmp)
    assert len(tr.tracks) == 2
    by_name = {t.player_name: t for t in tr.tracks}
    assert set(by_name) == {"Salah", "Mané"}
    salah = by_name["Salah"]
    assert salah.track_id == "T001"
    assert [f.frame for f in salah.frames] == [0, 1, 2, 3]
    # Both Salah tracks share the same canonical player_id.
    assert salah.player_id.startswith("P")


@pytest.mark.integration
def test_interpolate_gaps_fills_short_gaps_and_tags_new_frames(client) -> None:
    c, tmp = client
    # Track has one fillable gap (0→3, 2 missing) and one too-wide gap
    # (5→20). With max_gap=4, only the short gap is filled.
    _seed_tracks(
        tmp,
        ("T001", [(0, 0.9), (3, 0.9), (5, 0.9), (20, 0.9)]),
    )
    resp = c.post(
        "/api/tracks/play/interpolate-gaps",
        json={"track_ids": ["T001"], "max_gap": 4},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["total_frames_added"] == 3  # 2 between 0–3, 1 between 3–5
    assert body["results"][0]["frames_after"] == 7
    tr = _load_tracks(tmp)
    kept = tr.tracks[0]
    frame_idxs = [f.frame for f in kept.frames]
    assert frame_idxs == [0, 1, 2, 3, 4, 5, 20]
    # New frames carry the interpolated flag; originals do not.
    by_frame = {f.frame: f for f in kept.frames}
    assert by_frame[1].interpolated is True
    assert by_frame[2].interpolated is True
    assert by_frame[4].interpolated is True
    assert by_frame[0].interpolated is False
    assert by_frame[20].interpolated is False


@pytest.mark.integration
def test_interpolate_gaps_uses_config_default_when_max_gap_omitted(client) -> None:
    c, tmp = client
    _seed_tracks(
        tmp,
        ("T001", [(0, 0.9), (5, 0.9)]),  # 4-frame gap — fits default (8)
    )
    resp = c.post(
        "/api/tracks/play/interpolate-gaps",
        json={"track_ids": ["T001"]},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["max_gap"] == 8
    assert body["total_frames_added"] == 4


@pytest.mark.integration
def test_interpolate_gaps_rejects_empty_track_ids(client) -> None:
    c, tmp = client
    _seed_tracks(tmp, ("T001", [(0, 0.9), (5, 0.9)]))
    resp = c.post(
        "/api/tracks/play/interpolate-gaps",
        json={"track_ids": []},
    )
    assert resp.status_code == 400


@pytest.mark.integration
def test_interpolate_gaps_reports_missing_track_ids(client) -> None:
    c, tmp = client
    _seed_tracks(tmp, ("T001", [(0, 0.9), (5, 0.9)]))
    resp = c.post(
        "/api/tracks/play/interpolate-gaps",
        json={"track_ids": ["T001", "T999"], "max_gap": 4},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["missing"] == ["T999"]
    # T001 was still processed.
    assert any(r["track_id"] == "T001" for r in body["results"])


@pytest.mark.integration
def test_interpolate_gaps_skips_write_when_nothing_added(client) -> None:
    c, tmp = client
    _seed_tracks(tmp, ("T001", [(0, 0.9), (1, 0.9), (2, 0.9)]))
    track_path = tmp / "tracks" / "play_tracks.json"
    mtime_before = track_path.stat().st_mtime_ns
    resp = c.post(
        "/api/tracks/play/interpolate-gaps",
        json={"track_ids": ["T001"], "max_gap": 4},
    )
    assert resp.status_code == 200
    assert resp.json()["total_frames_added"] == 0
    # File untouched — no spurious resaves on no-op.
    assert track_path.stat().st_mtime_ns == mtime_before


@pytest.mark.unit
def test_get_anchors_per_shot(client):
    c, tmp_path = client
    (tmp_path / "camera").mkdir()
    (tmp_path / "camera" / "alpha_anchors.json").write_text(
        '{"clip_id":"alpha","image_size":[640,360],"anchors":[]}'
    )
    r = c.get("/anchors/alpha")
    assert r.status_code == 200
    assert r.json()["clip_id"] == "alpha"


@pytest.mark.unit
def test_get_anchors_per_shot_returns_empty_stub_for_missing(client):
    c, _ = client
    r = c.get("/anchors/beta")
    assert r.status_code == 200
    j = r.json()
    assert j["clip_id"] == "beta"
    assert j["anchors"] == []


@pytest.mark.unit
def test_post_anchors_per_shot(client):
    c, tmp_path = client
    payload = {
        "clip_id": "alpha",
        "image_size": [640, 360],
        "anchors": [],
    }
    r = c.post("/anchors/alpha", json=payload)
    assert r.status_code == 200
    assert (tmp_path / "camera" / "alpha_anchors.json").exists()


@pytest.mark.unit
def test_run_shot_player_dispatches_filtered_hmr_job(client, monkeypatch):
    """POST /api/run-shot-player wipes the per-(shot, player) hmr_world
    artefacts and dispatches a background hmr_world job filtered to
    just that pair."""
    c, tmp_path = client
    captured: dict = {}

    def fake_run_pipeline(*args, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr("src.web.server.run_pipeline", fake_run_pipeline)

    hmr_dir = tmp_path / "hmr_world"
    hmr_dir.mkdir()
    # The endpoint should remove the targeted pair's artefacts...
    target_npz = hmr_dir / "alpha__P001_smpl_world.npz"
    target_npz.write_bytes(b"to-be-wiped")
    target_kp = hmr_dir / "alpha__P001_kp2d.json"
    target_kp.write_text("{}")
    # ...but leave other shots/players alone.
    keep_npz = hmr_dir / "alpha__P002_smpl_world.npz"
    keep_npz.write_bytes(b"untouched")
    other_shot = hmr_dir / "beta__P001_smpl_world.npz"
    other_shot.write_bytes(b"untouched")

    r = c.post(
        "/api/run-shot-player",
        json={"shot_id": "alpha", "player_id": "P001"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["stage"] == "hmr_world"
    assert body["shot_id"] == "alpha"
    assert body["player_id"] == "P001"

    # Wipe scoping
    assert not target_npz.exists()
    assert not target_kp.exists()
    assert keep_npz.exists()
    assert other_shot.exists()

    # The dispatched job carries both filters.
    import time
    for _ in range(20):
        if "stages" in captured:
            break
        time.sleep(0.05)
    assert captured.get("stages") == "hmr_world"
    assert captured.get("from_stage") == "hmr_world"
    assert captured.get("shot_filter") == "alpha"
    assert captured.get("player_filter") == "P001"


@pytest.mark.unit
def test_run_shot_player_rejects_when_hmr_already_running(client, monkeypatch):
    """Concurrent hmr_world dispatches corrupt GVHMR's process-global
    Tensor.cuda monkey-patch (one job restores it mid-extract for the
    other and the original C-level .cuda() raises). The endpoint must
    block the second dispatch with 409 instead of letting both proceed."""
    c, _ = client
    monkeypatch.setattr("src.web.server.run_pipeline", lambda *a, **k: None)

    # Inject a fake "running" hmr_world job into the registry. The
    # endpoint reads from the same module-level dict, so we don't need
    # the dispatched thread to actually run.
    from src.web.server import Job, _jobs, _jobs_lock
    fake = Job(job_id="abc12345", stages="hmr_world")
    fake.status = "running"
    with _jobs_lock:
        _jobs["abc12345"] = fake

    try:
        r = c.post(
            "/api/run-shot-player",
            json={"shot_id": "alpha", "player_id": "P001"},
        )
        assert r.status_code == 409
        assert "already running" in r.json()["detail"]

        r2 = c.post(
            "/api/run-shot",
            json={"stage": "hmr_world", "shot_id": "alpha"},
        )
        assert r2.status_code == 409

        # Other stages aren't gated by hmr_world.
        r3 = c.post(
            "/api/run-shot",
            json={"stage": "camera", "shot_id": "alpha"},
        )
        assert r3.status_code == 200
    finally:
        with _jobs_lock:
            _jobs.pop("abc12345", None)


@pytest.mark.unit
def test_run_shot_player_rejects_invalid_ids(client, monkeypatch):
    c, _ = client
    monkeypatch.setattr("src.web.server.run_pipeline", lambda *a, **k: None)
    r = c.post(
        "/api/run-shot-player",
        json={"shot_id": "../etc", "player_id": "P001"},
    )
    assert r.status_code == 400
    r = c.post(
        "/api/run-shot-player",
        json={"shot_id": "alpha", "player_id": "../etc"},
    )
    assert r.status_code == 400


@pytest.mark.unit
def test_export_endpoints_are_shot_aware_and_fall_back(client):
    """``/api/export/{scene.glb,metadata}`` should resolve per-shot
    files when ``?shot=`` is supplied, and fall back to the first
    available per-shot file when the legacy singular file is absent
    (covers the multi-shot-only output of the modern export stage).
    ``/api/export/shots`` lists every shot with a baked scene."""
    c, tmp_path = client
    gltf_dir = tmp_path / "export" / "gltf"
    gltf_dir.mkdir(parents=True)
    (gltf_dir / "alpha_scene.glb").write_bytes(b"alpha-glb")
    (gltf_dir / "alpha_scene_metadata.json").write_text('{"shot":"alpha"}')
    (gltf_dir / "beta_scene.glb").write_bytes(b"beta-glb")
    (gltf_dir / "beta_scene_metadata.json").write_text('{"shot":"beta"}')

    # Per-shot resolution
    r = c.get("/api/export/scene.glb?shot=beta")
    assert r.status_code == 200
    assert r.content == b"beta-glb"
    r = c.get("/api/export/metadata?shot=alpha")
    assert r.json() == {"shot": "alpha"}

    # Legacy fallback — no singular files, but per-shot exist
    r = c.get("/api/export/metadata")
    assert r.status_code == 200
    assert r.json()["shot"] in {"alpha", "beta"}

    # Listing
    r = c.get("/api/export/shots")
    assert r.json() == {"shots": ["alpha", "beta"]}

    # Bad shot id rejected
    assert c.get("/api/export/scene.glb?shot=../etc").status_code == 400
    assert c.get("/api/export/metadata?shot=../etc").status_code == 400


@pytest.mark.unit
def test_hmr_players_resolves_name_for_unannotated_track_with_player_name(client):
    """A track whose ``player_name`` was set but whose ``player_id`` is
    blank lands on disk as ``{shot_id}__{shot_id}_T{track_id}_smpl_world.npz``
    (per ``HmrWorldStage._build_player_groups``). The dashboard must
    resolve the name from the operator-set ``player_name`` even though
    the filename's pid doesn't match the bare ``track_id``."""
    c, tmp_path = client

    # Tracks: alpha shot has Matip with player_id="" (named only),
    # beta shot has Matip with player_id="P005" (named + merged).
    tracks_dir = tmp_path / "tracks"
    tracks_dir.mkdir()
    tracks_dir.joinpath("alpha_tracks.json").write_text(
        '{"shot_id":"alpha","tracks":[{"track_id":"T004","player_id":"",'
        '"player_name":"Matip","team":"unknown","class_name":"player",'
        '"frames":[]}]}'
    )
    tracks_dir.joinpath("beta_tracks.json").write_text(
        '{"shot_id":"beta","tracks":[{"track_id":"T004","player_id":"P005",'
        '"player_name":"Matip","team":"unknown","class_name":"player",'
        '"frames":[]}]}'
    )

    # HMR outputs for both shots, named the same way the stage would.
    hmr_dir = tmp_path / "hmr_world"
    hmr_dir.mkdir()
    (hmr_dir / "alpha__alpha_TT004_smpl_world.npz").write_bytes(b"x")
    (hmr_dir / "beta__P005_smpl_world.npz").write_bytes(b"x")

    body_alpha = c.get("/hmr_world/players?shot=alpha").json()
    by_pid = {p["player_id"]: p for p in body_alpha["players"]}
    assert "alpha_TT004" in by_pid
    assert by_pid["alpha_TT004"]["player_name"] == "Matip"

    body_beta = c.get("/hmr_world/players?shot=beta").json()
    by_pid = {p["player_id"]: p for p in body_beta["players"]}
    assert "P005" in by_pid
    assert by_pid["P005"]["player_name"] == "Matip"


@pytest.mark.unit
def test_hmr_players_cross_shot_player_id_fallback(client):
    """If shot A merged Matip to player_id=P005 (filename uses P005)
    but shot B was not merged, B's filename pid would still be P005 if
    the operator manually merged in B. But if B's hmr filename uses
    P005 even though B's tracks file has no player_id=P005 entry, the
    name should still resolve via the cross-shot ("", "P005") fallback."""
    c, tmp_path = client

    tracks_dir = tmp_path / "tracks"
    tracks_dir.mkdir()
    # Only alpha names P005; beta has nothing about P005.
    tracks_dir.joinpath("alpha_tracks.json").write_text(
        '{"shot_id":"alpha","tracks":[{"track_id":"T01","player_id":"P005",'
        '"player_name":"Matip","team":"unknown","class_name":"player",'
        '"frames":[]}]}'
    )

    hmr_dir = tmp_path / "hmr_world"
    hmr_dir.mkdir()
    # Beta's HMR file uses P005 even though beta's tracks don't name it.
    (hmr_dir / "beta__P005_smpl_world.npz").write_bytes(b"x")

    body = c.get("/hmr_world/players?shot=beta").json()
    assert body["players"][0]["player_name"] == "Matip"


@pytest.mark.unit
def test_hmr_players_endpoint_filters_by_shot(client):
    """``/hmr_world/players?shot=xxx`` returns only the rows whose
    filename key starts with ``xxx__``."""
    c, tmp_path = client
    hmr_dir = tmp_path / "hmr_world"
    hmr_dir.mkdir()
    for stem in ("alpha__P001", "alpha__P002", "beta__P001"):
        (hmr_dir / f"{stem}_smpl_world.npz").write_bytes(b"x")

    body_all = c.get("/hmr_world/players").json()
    keys_all = sorted(
        f"{p['shot_id']}__{p['player_id']}" for p in body_all["players"]
    )
    assert keys_all == ["alpha__P001", "alpha__P002", "beta__P001"]

    body_alpha = c.get("/hmr_world/players?shot=alpha").json()
    keys_alpha = sorted(p["player_id"] for p in body_alpha["players"])
    assert keys_alpha == ["P001", "P002"]
    assert all(p["shot_id"] == "alpha" for p in body_alpha["players"])


@pytest.mark.unit
def test_camera_stage_complete_requires_every_shot_track(client):
    """The camera stage shows green only when every shot in the manifest
    has its per-shot ``{shot_id}_camera_track.json`` on disk. A partial
    multi-shot solve must not flip the stage to complete."""
    c, tmp_path = client

    shots_dir = tmp_path / "shots"
    shots_dir.mkdir()
    (shots_dir / "shots_manifest.json").write_text(
        '{"source_file":"x","fps":30,"total_frames":0,"shots":['
        '{"id":"alpha","start_frame":0,"end_frame":0,"start_time":0.0,'
        '"end_time":0.0,"clip_file":"shots/alpha.mp4"},'
        '{"id":"beta","start_frame":0,"end_frame":0,"start_time":0.0,'
        '"end_time":0.0,"clip_file":"shots/beta.mp4"}]}'
    )

    cam_dir = tmp_path / "camera"
    cam_dir.mkdir()

    def stage_complete(stage_name):
        body = c.get("/api/stages").json()
        return next(s for s in body if s["name"] == stage_name)["complete"]

    assert stage_complete("camera") is False

    (cam_dir / "alpha_camera_track.json").write_text("{}")
    assert stage_complete("camera") is False

    (cam_dir / "beta_camera_track.json").write_text("{}")
    assert stage_complete("camera") is True


@pytest.mark.unit
def test_upload_shots_writes_files_and_dispatches_job(client, monkeypatch):
    """POST /api/shots/upload saves uploaded .mp4 files into shots/ and
    fires a prepare_shots job that runs without wiping the directory."""
    c, tmp_path = client
    captured: dict = {}

    def fake_run_pipeline(*args, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr("src.web.server.run_pipeline", fake_run_pipeline)

    files = [
        ("files", ("alpha.mp4", b"fake-mp4-bytes-1", "video/mp4")),
        ("files", ("beta clip.mp4", b"fake-mp4-bytes-2", "video/mp4")),
    ]
    r = c.post("/api/shots/upload", files=files)
    assert r.status_code == 200
    body = r.json()
    assert sorted(body["saved"]) == ["alpha", "betaclip"]
    assert body["skipped"] == []
    assert body["job_id"] is not None

    # Files landed under sanitised names.
    assert (tmp_path / "shots" / "alpha.mp4").read_bytes() == b"fake-mp4-bytes-1"
    assert (tmp_path / "shots" / "betaclip.mp4").read_bytes() == b"fake-mp4-bytes-2"

    # The dispatched job runs prepare_shots with from_stage set, so the
    # runner doesn't short-circuit on its is_complete() cache.
    import time
    for _ in range(20):
        if "stages" in captured:
            break
        time.sleep(0.05)
    assert captured.get("stages") == "prepare_shots"
    assert captured.get("from_stage") == "prepare_shots"


@pytest.mark.unit
def test_upload_shots_skips_existing_shot_id(client, monkeypatch):
    """An upload whose sanitised name collides with an existing shot is
    skipped rather than overwriting the on-disk clip."""
    c, tmp_path = client
    monkeypatch.setattr("src.web.server.run_pipeline", lambda *a, **k: None)

    shots_dir = tmp_path / "shots"
    shots_dir.mkdir()
    (shots_dir / "alpha.mp4").write_bytes(b"already here")

    r = c.post(
        "/api/shots/upload",
        files=[("files", ("alpha.mp4", b"new bytes", "video/mp4"))],
    )
    assert r.status_code == 200
    body = r.json()
    assert body["saved"] == []
    assert len(body["skipped"]) == 1
    assert "already exists" in body["skipped"][0]["reason"]
    # Existing file untouched.
    assert (shots_dir / "alpha.mp4").read_bytes() == b"already here"
    # No job dispatched when nothing was saved.
    assert body["job_id"] is None


@pytest.mark.unit
def test_upload_shots_rejects_non_mp4(client, monkeypatch):
    c, _ = client
    monkeypatch.setattr("src.web.server.run_pipeline", lambda *a, **k: None)
    r = c.post(
        "/api/shots/upload",
        files=[("files", ("notes.txt", b"hello", "text/plain"))],
    )
    assert r.status_code == 200
    body = r.json()
    assert body["saved"] == []
    assert body["skipped"][0]["reason"] == "not an .mp4 file"


@pytest.mark.unit
def test_run_shot_endpoint_dispatches_filtered_job(client, monkeypatch):
    """POST /api/run-shot dispatches a background job with the correct
    stages= and shot_filter= values plumbed through to run_pipeline."""
    c, tmp_path = client
    captured: dict = {}

    def fake_run_pipeline(*args, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr("src.web.server.run_pipeline", fake_run_pipeline)
    r = c.post(
        "/api/run-shot",
        json={"stage": "camera", "shot_id": "alpha"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["stage"] == "camera"
    assert body["shot_id"] == "alpha"
    # The background thread should run quickly with our fake.
    import time
    for _ in range(20):
        if "stages" in captured:
            break
        time.sleep(0.05)
    assert captured.get("stages") == "camera"
    assert captured.get("shot_filter") == "alpha"


def _stage_status(stages_response: list[dict], name: str) -> dict | None:
    for s in stages_response:
        if s["name"] == name:
            return s
    return None


@pytest.mark.unit
def test_refined_poses_appears_in_stages_list(client) -> None:
    """``GET /api/stages`` includes refined_poses between ball and export
    so the dashboard sidebar can render it as a first-class stage."""
    c, _ = client
    r = c.get("/api/stages")
    assert r.status_code == 200
    stages = r.json()
    names = [s["name"] for s in stages]
    assert "refined_poses" in names
    assert names.index("refined_poses") == names.index("ball") + 1
    assert names.index("export") == names.index("refined_poses") + 1


@pytest.mark.integration
def test_refined_poses_players_endpoint_lists_refined_tracks(client) -> None:
    """``GET /refined_poses/players`` returns one row per fused track
    NPZ on disk, including contributing-shot list and view-count
    breakdown derived from the saved arrays.
    """
    import numpy as np

    from src.schemas.refined_pose import RefinedPose

    c, tmp = client
    rp_dir = tmp / "refined_poses"
    rp_dir.mkdir()
    RefinedPose(
        player_id="P001",
        frames=np.arange(5, dtype=np.int64),
        betas=np.zeros(10),
        thetas=np.zeros((5, 24, 3)),
        root_R=np.tile(np.eye(3), (5, 1, 1)),
        root_t=np.zeros((5, 3)),
        confidence=np.full(5, 0.8),
        view_count=np.array([2, 2, 1, 2, 2], dtype=np.int32),
        contributing_shots=("origi01", "origi02"),
    ).save(rp_dir / "P001_refined.npz")

    r = c.get("/refined_poses/players")
    assert r.status_code == 200
    rows = r.json()["players"]
    assert len(rows) == 1
    row = rows[0]
    assert row["player_id"] == "P001"
    assert row["contributing_shots"] == ["origi01", "origi02"]
    assert row["n_frames"] == 5
    assert row["multi_view_frames"] == 4
    assert row["single_view_frames"] == 1
    assert row["mean_confidence"] == pytest.approx(0.8, abs=1e-9)


@pytest.mark.integration
def test_refined_poses_preview_endpoint_returns_track(client) -> None:
    """``GET /refined_poses/preview?player_id=P001`` returns the fused
    track on the reference timeline, with view_count and contributing
    shots so the dashboard can flag low-coverage frames."""
    import numpy as np

    from src.schemas.refined_pose import RefinedPose

    c, tmp = client
    rp_dir = tmp / "refined_poses"
    rp_dir.mkdir()
    RefinedPose(
        player_id="P001",
        frames=np.array([0, 1, 2], dtype=np.int64),
        betas=np.zeros(10),
        thetas=np.zeros((3, 24, 3)),
        root_R=np.tile(np.eye(3), (3, 1, 1)),
        root_t=np.array([[1.0, 2.0, 0.0], [1.5, 2.0, 0.0], [2.0, 2.0, 0.0]]),
        confidence=np.array([0.9, 0.7, 0.5]),
        view_count=np.array([2, 2, 1], dtype=np.int32),
        contributing_shots=("A", "B"),
    ).save(rp_dir / "P001_refined.npz")

    r = c.get("/refined_poses/preview?player_id=P001")
    assert r.status_code == 200
    body = r.json()
    assert body["player_id"] == "P001"
    assert body["frames"] == [0, 1, 2]
    assert body["view_count"] == [2, 2, 1]
    assert body["contributing_shots"] == ["A", "B"]
    assert body["root_t"][0] == [1.0, 2.0, 0.0]


@pytest.mark.integration
def test_refined_poses_preview_returns_404_when_missing(client) -> None:
    c, _ = client
    r = c.get("/refined_poses/preview?player_id=NEVER")
    assert r.status_code == 404


@pytest.mark.integration
def test_refined_poses_summary_endpoint(client) -> None:
    """``GET /refined_poses/summary`` returns the JSON written by the
    stage; an empty dict when the stage hasn't run."""
    c, tmp = client
    r = c.get("/refined_poses/summary")
    assert r.status_code == 200
    assert r.json() == {}

    rp_dir = tmp / "refined_poses"
    rp_dir.mkdir()
    summary = {
        "players_refined": 3,
        "single_shot_players": 1,
        "multi_shot_players": 2,
        "total_fused_frames": 100,
        "single_view_frames": 20,
        "high_disagreement_frames": 4,
        "shots_missing_sync": [],
        "beta_disagreement_warnings": [],
    }
    (rp_dir / "refined_poses_summary.json").write_text(json.dumps(summary))
    r = c.get("/refined_poses/summary")
    assert r.status_code == 200
    assert r.json()["players_refined"] == 3


# ── Sub-pixel click snap endpoint ───────────────────────────────────────────


@pytest.mark.integration
def test_snap_endpoint_validates_inputs(client) -> None:
    """POST /api/anchor/snap rejects malformed payloads with 4xx errors."""
    c, _ = client
    # Missing shot_id
    r = c.post("/api/anchor/snap", json={"frame": 0, "click": [100, 100]})
    assert r.status_code == 400
    # Invalid shot_id (path-traversal characters)
    r = c.post("/api/anchor/snap", json={
        "shot_id": "../etc/passwd", "frame": 0, "click": [100, 100],
    })
    assert r.status_code == 400
    # Missing click
    r = c.post("/api/anchor/snap", json={"shot_id": "x", "frame": 0})
    assert r.status_code == 400
    # Bad click shape
    r = c.post("/api/anchor/snap", json={
        "shot_id": "x", "frame": 0, "click": [100],
    })
    assert r.status_code == 400


@pytest.mark.integration
def test_snap_endpoint_returns_click_unchanged_on_blank_frame(client) -> None:
    """When the clip patch is featureless, snap returns the input
    coords with ``snapped=False``. Validates the no-feature fallback
    behaviour without needing a real broadcast frame."""
    import cv2
    import numpy as np

    c, tmp = client
    shot_dir = tmp / "shots"
    shot_dir.mkdir()
    # Write a 100×100 mid-grey clip (no painted lines)
    clip_path = shot_dir / "blankshot.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(clip_path), fourcc, 30.0, (100, 100))
    grey = np.full((100, 100, 3), 128, dtype=np.uint8)
    for _ in range(5):
        writer.write(grey)
    writer.release()
    assert clip_path.exists()

    r = c.post("/api/anchor/snap", json={
        "shot_id": "blankshot", "frame": 0, "click": [50.0, 50.0], "mode": "auto",
    })
    assert r.status_code == 200
    body = r.json()
    assert body["snapped"] is False
    assert body["xy"] == [50.0, 50.0]
    assert body["confidence"] == 0.0


@pytest.mark.integration
def test_detected_lines_endpoint(client) -> None:
    """``GET /camera/detected-lines`` returns the line_extraction debug
    side-output; an empty ``frames`` map when the pass hasn't run."""
    c, tmp = client
    # No file yet → empty shape
    r = c.get("/camera/detected-lines?shot=gberch")
    assert r.status_code == 200
    assert r.json()["frames"] == {}

    cam_dir = tmp / "camera"
    cam_dir.mkdir()
    payload = {
        "shot_id": "gberch",
        "image_size": [1920, 1080],
        "fps": 30.0,
        "frames": {
            "12": {
                "lines": [
                    {
                        "name": "left_18yd_front",
                        "image_segment": [[860.0, 580.0], [1729.0, 229.0]],
                        "world_segment": [[16.5, 13.84, 0.0], [16.5, 54.16, 0.0]],
                    }
                ]
            }
        },
    }
    (cam_dir / "gberch_detected_lines.json").write_text(json.dumps(payload))
    r = c.get("/camera/detected-lines?shot=gberch")
    assert r.status_code == 200
    body = r.json()
    assert "12" in body["frames"]
    assert body["frames"]["12"]["lines"][0]["name"] == "left_18yd_front"

    # Invalid shot id rejected
    r = c.get("/camera/detected-lines?shot=../etc")
    assert r.status_code == 400
