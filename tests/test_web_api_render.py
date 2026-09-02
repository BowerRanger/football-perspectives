from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.schemas.smpl_world import SmplWorldTrack
from src.web.server import create_app


@pytest.fixture
def client(tmp_path: Path):
    app = create_app(output_dir=tmp_path, config_path=None)
    return TestClient(app), tmp_path


def _write_player(out: Path, pid: str, shot_id: str) -> None:
    (out / "hmr_world").mkdir(parents=True, exist_ok=True)
    n = 2
    SmplWorldTrack(
        player_id=pid, frames=np.arange(n), betas=np.zeros(10),
        thetas=np.zeros((n, 24, 3)), root_R=np.broadcast_to(np.eye(3), (n, 3, 3)).copy(),
        root_t=np.zeros((n, 3)), confidence=np.ones(n), shot_id=shot_id,
    ).save(out / "hmr_world" / f"{pid}_smpl_world.npz")


# ── /api/render/outputs ─────────────────────────────────────────────────

@pytest.mark.integration
def test_render_outputs_empty_when_no_render_dir(client) -> None:
    c, _ = client
    r = c.get("/api/render/outputs")
    assert r.status_code == 200
    assert r.json() == {"shots": {}}


@pytest.mark.integration
def test_render_outputs_lists_cameras_with_vertical_flag_timings_and_aov(client) -> None:
    c, out = client
    shot_dir = out / "render" / "shot01"
    shot_dir.mkdir(parents=True)
    (shot_dir / "broadcast.mp4").write_bytes(b"x" * 100)
    (shot_dir / "drone.mp4").write_bytes(b"x" * 200)
    (shot_dir / "drone_9x16.mp4").write_bytes(b"x" * 50)
    # excluded: nested subdirs must never surface as top-level cameras
    (shot_dir / "cameras").mkdir()
    (shot_dir / "cameras" / "drone_camera_track.json").write_text("{}")
    aov_dir = shot_dir / "aov" / "broadcast"
    aov_dir.mkdir(parents=True)
    (aov_dir / "0001.exr").write_bytes(b"exr")
    (out / "render" / "render_timings.json").write_text(
        json.dumps({"shot01": 12.5})
    )

    r = c.get("/api/render/outputs")
    assert r.status_code == 200
    body = r.json()
    assert set(body["shots"].keys()) == {"shot01"}
    shot = body["shots"]["shot01"]
    assert shot["render_seconds"] == 12.5
    assert shot["aov"] is True

    cams = {(c_["id"], c_["vertical"]): c_ for c_ in shot["cameras"]}
    assert set(cams.keys()) == {
        ("broadcast", False), ("drone", False), ("drone", True),
    }
    assert cams[("broadcast", False)]["file"] == "broadcast.mp4"
    assert cams[("broadcast", False)]["size_bytes"] == 100
    assert cams[("drone", True)]["file"] == "drone_9x16.mp4"
    assert cams[("drone", True)]["size_bytes"] == 50
    for entry in shot["cameras"]:
        assert "mtime" in entry


@pytest.mark.integration
def test_render_outputs_render_seconds_null_and_aov_false_when_absent(client) -> None:
    c, out = client
    shot_dir = out / "render" / "shot01"
    shot_dir.mkdir(parents=True)
    (shot_dir / "broadcast.mp4").write_bytes(b"x")

    r = c.get("/api/render/outputs")
    shot = r.json()["shots"]["shot01"]
    assert shot["render_seconds"] is None
    assert shot["aov"] is False


@pytest.mark.integration
def test_render_outputs_legacy_clip_directory(client) -> None:
    c, out = client
    shot_dir = out / "render" / "clip"
    shot_dir.mkdir(parents=True)
    (shot_dir / "broadcast.mp4").write_bytes(b"x")
    (out / "render" / "render_timings.json").write_text(json.dumps({"clip": 4.0}))

    r = c.get("/api/render/outputs")
    body = r.json()
    assert "clip" in body["shots"]
    assert body["shots"]["clip"]["render_seconds"] == 4.0


# ── /api/render/video/{shot_id}/{camera} ────────────────────────────────

@pytest.mark.integration
def test_render_video_200_full(client) -> None:
    c, out = client
    shot_dir = out / "render" / "shot01"
    shot_dir.mkdir(parents=True)
    (shot_dir / "broadcast.mp4").write_bytes(b"0123456789")
    r = c.get("/api/render/video/shot01/broadcast")
    assert r.status_code == 200
    assert r.content == b"0123456789"


@pytest.mark.integration
def test_render_video_206_with_range(client) -> None:
    c, out = client
    shot_dir = out / "render" / "shot01"
    shot_dir.mkdir(parents=True)
    (shot_dir / "broadcast.mp4").write_bytes(b"0123456789")
    r = c.get("/api/render/video/shot01/broadcast", headers={"Range": "bytes=2-5"})
    assert r.status_code == 206
    assert r.content == b"2345"
    assert r.headers["Content-Range"] == "bytes 2-5/10"


@pytest.mark.integration
def test_render_video_vertical_variant_filename(client) -> None:
    c, out = client
    shot_dir = out / "render" / "shot01"
    shot_dir.mkdir(parents=True)
    (shot_dir / "drone_9x16.mp4").write_bytes(b"abc")
    r = c.get("/api/render/video/shot01/drone_9x16")
    assert r.status_code == 200
    assert r.content == b"abc"


@pytest.mark.integration
def test_render_video_404_missing(client) -> None:
    c, out = client
    (out / "render" / "shot01").mkdir(parents=True)
    r = c.get("/api/render/video/shot01/broadcast")
    assert r.status_code == 404


@pytest.mark.integration
def test_render_video_400_bad_shot_id(client) -> None:
    c, _ = client
    r = c.get("/api/render/video/../etc/broadcast")
    assert r.status_code in (400, 404)  # path traversal never resolves as a route either way


@pytest.mark.integration
def test_render_video_400_bad_shot_id_chars(client) -> None:
    c, _ = client
    r = c.get("/api/render/video/bad%20shot/broadcast")
    assert r.status_code == 400


@pytest.mark.integration
def test_render_video_400_bad_camera_id_chars(client) -> None:
    c, out = client
    (out / "render" / "shot01").mkdir(parents=True)
    r = c.get("/api/render/video/shot01/bad.camera")
    assert r.status_code == 400


# ── /api/render/selection ────────────────────────────────────────────────

@pytest.mark.integration
def test_render_selection_get_default_empty(client) -> None:
    c, _ = client
    r = c.get("/api/render/selection", params={"shot": "shot01"})
    assert r.status_code == 200
    assert r.json() == {"shot_id": "shot01", "cameras": [], "vertical_variant": None}


@pytest.mark.integration
def test_render_selection_get_default_empty_legacy_no_shot(client) -> None:
    c, _ = client
    r = c.get("/api/render/selection")
    assert r.status_code == 200
    assert r.json() == {"shot_id": "", "cameras": [], "vertical_variant": None}


@pytest.mark.integration
def test_render_selection_put_round_trip(client) -> None:
    c, out = client
    _write_player(out, "P003", "shot01")
    body = {"shot_id": "shot01", "cameras": ["broadcast", "pov:P003"], "vertical_variant": True}
    r = c.put("/api/render/selection", params={"shot": "shot01"}, json=body)
    assert r.status_code == 200
    saved = out / "render" / "shot01_render_selection.json"
    assert saved.exists()
    again = c.get("/api/render/selection", params={"shot": "shot01"})
    assert again.json()["cameras"] == ["broadcast", "pov:P003"]
    assert again.json()["vertical_variant"] is True


@pytest.mark.integration
def test_render_selection_put_legacy_shot_writes_clip_filename(client) -> None:
    c, out = client
    body = {"shot_id": "", "cameras": ["broadcast"]}
    r = c.put("/api/render/selection", json=body)
    assert r.status_code == 200
    assert (out / "render" / "clip_render_selection.json").exists()


@pytest.mark.integration
def test_render_selection_put_rejects_unknown_player(client) -> None:
    c, out = client
    _write_player(out, "P003", "shot01")
    body = {"shot_id": "shot01", "cameras": ["pov:P999"]}
    r = c.put("/api/render/selection", params={"shot": "shot01"}, json=body)
    assert r.status_code == 400


@pytest.mark.integration
def test_render_selection_put_rejects_bad_camera_id(client) -> None:
    c, _ = client
    body = {"shot_id": "shot01", "cameras": ["dolly"]}
    r = c.put("/api/render/selection", params={"shot": "shot01"}, json=body)
    assert r.status_code == 400


@pytest.mark.integration
def test_render_selection_get_rejects_bad_shot_id(client) -> None:
    c, _ = client
    r = c.get("/api/render/selection", params={"shot": "../etc"})
    assert r.status_code == 400


@pytest.mark.integration
def test_render_selection_put_rejects_bad_shot_id(client) -> None:
    c, _ = client
    body = {"shot_id": "shot01", "cameras": []}
    r = c.put("/api/render/selection", params={"shot": "bad shot"}, json=body)
    assert r.status_code == 400


@pytest.mark.integration
def test_render_selection_broadcast_drone_do_not_require_player_data(client) -> None:
    c, _ = client
    body = {"shot_id": "shot01", "cameras": ["broadcast", "drone"]}
    r = c.put("/api/render/selection", params={"shot": "shot01"}, json=body)
    assert r.status_code == 200


# ── /api/stages registration ────────────────────────────────────────────

@pytest.mark.unit
def test_render_appears_after_export_in_stages_list(client) -> None:
    c, _ = client
    r = c.get("/api/stages")
    assert r.status_code == 200
    names = [s["name"] for s in r.json()]
    assert "render" in names
    assert names.index("export") < names.index("render")


@pytest.mark.unit
def test_render_stage_complete_mirrors_render_stage_is_complete(client) -> None:
    c, out = client

    def stage_complete():
        body = c.get("/api/stages").json()
        return next(s for s in body if s["name"] == "render")["complete"]

    assert stage_complete() is False
    clip_dir = out / "render" / "clip"
    clip_dir.mkdir(parents=True)
    (clip_dir / "broadcast.mp4").write_bytes(b"x")
    assert stage_complete() is True


@pytest.mark.unit
def test_render_stage_complete_requires_every_active_shot(client) -> None:
    c, out = client
    shots_dir = out / "shots"
    shots_dir.mkdir()
    (shots_dir / "shots_manifest.json").write_text(
        '{"source_file":"x","fps":30,"total_frames":0,"shots":['
        '{"id":"alpha","start_frame":0,"end_frame":0,"start_time":0.0,'
        '"end_time":0.0,"clip_file":"shots/alpha.mp4"},'
        '{"id":"beta","start_frame":0,"end_frame":0,"start_time":0.0,'
        '"end_time":0.0,"clip_file":"shots/beta.mp4"}]}'
    )

    def stage_complete():
        body = c.get("/api/stages").json()
        return next(s for s in body if s["name"] == "render")["complete"]

    assert stage_complete() is False
    alpha_dir = out / "render" / "alpha"
    alpha_dir.mkdir(parents=True)
    (alpha_dir / "broadcast.mp4").write_bytes(b"x")
    assert stage_complete() is False  # beta still missing

    beta_dir = out / "render" / "beta"
    beta_dir.mkdir(parents=True)
    (beta_dir / "broadcast.mp4").write_bytes(b"x")
    assert stage_complete() is True


@pytest.mark.unit
def test_render_stage_complete_respects_sidecar_camera_request(client) -> None:
    """The dashboard's empty-config RenderStage() still resolves the
    RenderSelection sidecar (operator input always wins), so requesting
    an extra camera keeps the stage incomplete until that camera is
    rendered too — camera-granularity completeness, not just "some mp4
    exists"."""
    c, out = client
    from src.schemas.render_selection import RenderSelection
    RenderSelection(shot_id="", cameras=("broadcast", "drone")).save(
        out / "render" / "clip_render_selection.json"
    )

    def stage_complete():
        body = c.get("/api/stages").json()
        return next(s for s in body if s["name"] == "render")["complete"]

    clip_dir = out / "render" / "clip"
    clip_dir.mkdir(parents=True)
    (clip_dir / "broadcast.mp4").write_bytes(b"x")
    assert stage_complete() is False  # drone still requested but missing

    (clip_dir / "drone.mp4").write_bytes(b"x")
    assert stage_complete() is True
