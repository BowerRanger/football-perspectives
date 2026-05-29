from __future__ import annotations

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


@pytest.mark.integration
def test_available_players_lists_shot_players(client) -> None:
    c, out = client
    _write_player(out, "P003", "shot_01")
    _write_player(out, "P012", "shot_01")
    r = c.get("/api/export/available-players", params={"shot": "shot_01"})
    assert r.status_code == 200
    assert {p["player_id"] for p in r.json()["players"]} == {"P003", "P012"}


@pytest.mark.integration
def test_selection_get_default_empty(client) -> None:
    c, _ = client
    r = c.get("/api/export/camera-selection", params={"shot": "shot_01"})
    assert r.status_code == 200
    assert r.json() == {"shot_id": "shot_01", "selections": []}


@pytest.mark.integration
def test_selection_put_round_trip(client) -> None:
    c, out = client
    _write_player(out, "P003", "shot_01")
    body = {"shot_id": "shot_01", "selections": [{"player_id": "P003", "rigs": ["pov", "ots"]}]}
    r = c.put("/api/export/camera-selection", params={"shot": "shot_01"}, json=body)
    assert r.status_code == 200
    saved = (out / "export" / "shot_01_camera_selection.json")
    assert saved.exists()
    again = c.get("/api/export/camera-selection", params={"shot": "shot_01"})
    assert again.json()["selections"][0]["player_id"] == "P003"


@pytest.mark.integration
def test_selection_put_rejects_unknown_player(client) -> None:
    c, out = client
    _write_player(out, "P003", "shot_01")
    body = {"shot_id": "shot_01", "selections": [{"player_id": "P999", "rigs": ["pov"]}]}
    r = c.put("/api/export/camera-selection", params={"shot": "shot_01"}, json=body)
    assert r.status_code == 400


@pytest.mark.integration
def test_selection_put_rejects_unknown_rig(client) -> None:
    c, out = client
    _write_player(out, "P003", "shot_01")
    body = {"shot_id": "shot_01", "selections": [{"player_id": "P003", "rigs": ["dolly"]}]}
    r = c.put("/api/export/camera-selection", params={"shot": "shot_01"}, json=body)
    assert r.status_code == 400
