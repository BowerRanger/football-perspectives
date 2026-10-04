"""appearance + shorts in GET /api/stages: ordering, completion, partial,
and the clear-list protecting operator input."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.web import server
from src.web.server import create_app


@pytest.fixture
def client(tmp_path: Path):
    return TestClient(create_app(output_dir=tmp_path, config_path=None)), tmp_path


def _stages(c: TestClient) -> dict[str, dict]:
    return {s["name"]: s for s in c.get("/api/stages").json()}


@pytest.mark.unit
def test_stage_order_places_appearance_after_ball_and_shorts_last(client) -> None:
    c, _ = client
    names = [s["name"] for s in c.get("/api/stages").json()]
    assert names.index("ball") < names.index("appearance") < names.index("export")
    assert names[-1] == "shorts"
    assert names == server.STAGE_ORDER
    idx = [s["index"] for s in c.get("/api/stages").json()]
    assert idx == list(range(1, len(names) + 1))


@pytest.mark.unit
def test_empty_output_reports_new_stages_pending(client) -> None:
    c, _ = client
    stages = _stages(c)
    for name in ("appearance", "shorts"):
        assert stages[name]["complete"] is False
        assert stages[name]["partial"] is False


@pytest.mark.unit
def test_appearance_complete_when_kits_json_exists(client) -> None:
    c, tmp = client
    (tmp / "appearance").mkdir()
    (tmp / "appearance" / "kits.json").write_text(json.dumps({}))
    assert _stages(c)["appearance"]["complete"] is True


@pytest.mark.unit
def test_appearance_partial_with_suggestions_only(client) -> None:
    c, tmp = client
    (tmp / "appearance").mkdir()
    (tmp / "appearance" / "players_suggested.json").write_text("{}")
    st = _stages(c)["appearance"]
    assert st["complete"] is False and st["partial"] is True


@pytest.mark.unit
def test_shorts_partial_when_an_mp4_exists_without_goal_shot_sidecar(client) -> None:
    c, tmp = client
    (tmp / "shorts").mkdir()
    (tmp / "shorts" / "s1_comic.mp4").write_bytes(b"x")
    st = _stages(c)["shorts"]
    assert st["partial"] is True


@pytest.mark.unit
def test_shorts_completeness_failure_degrades_to_not_complete(client, monkeypatch) -> None:
    c, _ = client

    def boom(*_a, **_k):
        raise RuntimeError("broken sidecar")

    monkeypatch.setattr("src.stages.shorts.ShortsStage.is_complete", boom)
    assert _stages(c)["shorts"]["complete"] is False


@pytest.mark.unit
def test_rerun_clear_list_never_includes_operator_input(client) -> None:
    c, tmp = client
    (tmp / "appearance").mkdir()
    (tmp / "appearance" / "kits.json").write_text("{}")
    (tmp / "appearance" / "kits_operator.json").write_text("{}")
    (tmp / "shorts").mkdir()
    (tmp / "shorts" / "s1_comic.mp4").write_bytes(b"x")
    (tmp / "shorts" / "s1_operator.json").write_text("{}")
    a = {p["path"] for p in c.get("/api/output/appearance/artifacts").json()["paths"]}
    s = {p["path"] for p in c.get("/api/output/shorts/artifacts").json()["paths"]}
    assert a == {"appearance/kits.json"}
    assert s == {"shorts/s1_comic.mp4"}
