"""Integration tests for the output-directory switcher API.

The dashboard can re-point the running server at a different ``output*``
sibling directory at runtime (see
``docs/superpowers/specs/2026-06-06-output-dir-switcher-design.md``).

Each test boots a fresh app served from ``<tmp>/output`` with a couple of
sibling dirs alongside it, then exercises the discovery / switch / create
endpoints. ``GET /api/output/quality-report`` is used as the witness that a
switch actually reaches the *existing* endpoint closures (it reads
``output_dir / "quality_report.json"``).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.web.server import create_app


@pytest.fixture
def client(tmp_path: Path):
    """Serve ``<tmp>/output`` with siblings ``output-a`` and ``output-b``."""
    served = tmp_path / "output"
    served.mkdir()
    (tmp_path / "output-a").mkdir()
    (tmp_path / "output-b").mkdir()
    app = create_app(output_dir=served, config_path=None)
    return TestClient(app), tmp_path


@pytest.mark.integration
def test_list_returns_siblings_and_current(client) -> None:
    c, tmp = client
    resp = c.get("/api/output-dirs")
    assert resp.status_code == 200
    body = resp.json()
    assert body["current"] == "output"
    assert body["dirs"] == ["output", "output-a", "output-b"]
    assert Path(body["parent"]) == tmp.resolve()


@pytest.mark.integration
def test_list_ignores_files_and_non_output_dirs(client) -> None:
    c, tmp = client
    (tmp / "output-note.txt").write_text("not a dir")  # file, must be skipped
    (tmp / "scratch").mkdir()  # dir, but not output*
    resp = c.get("/api/output-dirs")
    assert resp.json()["dirs"] == ["output", "output-a", "output-b"]


@pytest.mark.integration
def test_list_includes_current_even_when_name_not_output(tmp_path: Path) -> None:
    served = tmp_path / "custom"
    served.mkdir()
    (tmp_path / "output-a").mkdir()
    c = TestClient(create_app(output_dir=served, config_path=None))
    body = c.get("/api/output-dirs").json()
    assert body["current"] == "custom"
    assert "custom" in body["dirs"]
    assert "output-a" in body["dirs"]


@pytest.mark.integration
def test_switch_active_reaches_existing_endpoint(client) -> None:
    c, tmp = client
    # Distinctive quality report only in output-a.
    (tmp / "output-a" / "quality_report.json").write_text(json.dumps({"marker": "A"}))

    # Before switching, the served "output" dir has no report.
    assert c.get("/api/output/quality-report").json() == {}

    resp = c.put("/api/output-dirs/active", json={"name": "output-a"})
    assert resp.status_code == 200
    assert resp.json()["current"] == "output-a"

    # The *existing* quality-report endpoint now reads output-a.
    assert c.get("/api/output/quality-report").json() == {"marker": "A"}

    # And switching back restores the original.
    c.put("/api/output-dirs/active", json={"name": "output"})
    assert c.get("/api/output/quality-report").json() == {}


@pytest.mark.integration
def test_switch_rejects_unknown_dir(client) -> None:
    c, _ = client
    resp = c.put("/api/output-dirs/active", json={"name": "output-nope"})
    assert resp.status_code == 400
    # Active dir unchanged.
    assert c.get("/api/output-dirs").json()["current"] == "output"


@pytest.mark.integration
@pytest.mark.parametrize("bad", ["../output-a", "/etc", "output/../output-a", ".."])
def test_switch_rejects_traversal(client, bad: str) -> None:
    c, _ = client
    resp = c.put("/api/output-dirs/active", json={"name": bad})
    assert resp.status_code == 400
    assert c.get("/api/output-dirs").json()["current"] == "output"


@pytest.mark.integration
def test_create_makes_dir_and_switches(client) -> None:
    c, tmp = client
    resp = c.post("/api/output-dirs", json={"name": "fresh"})
    assert resp.status_code == 200
    assert resp.json()["current"] == "output-fresh"
    assert (tmp / "output-fresh").is_dir()
    # Now selectable in the list and active.
    body = c.get("/api/output-dirs").json()
    assert body["current"] == "output-fresh"
    assert "output-fresh" in body["dirs"]


@pytest.mark.integration
def test_create_keeps_existing_output_prefix(client) -> None:
    c, tmp = client
    resp = c.post("/api/output-dirs", json={"name": "output-exp1"})
    assert resp.status_code == 200
    assert resp.json()["current"] == "output-exp1"
    assert (tmp / "output-exp1").is_dir()
    assert not (tmp / "output-output-exp1").exists()


@pytest.mark.integration
@pytest.mark.parametrize("bad", ["", "   ", "../evil", "a/b", "..", "out put"])
def test_create_rejects_invalid_name(client, bad: str) -> None:
    c, tmp = client
    before = {p.name for p in tmp.iterdir()}
    resp = c.post("/api/output-dirs", json={"name": bad})
    assert resp.status_code == 400
    # No directory created, active dir unchanged.
    assert {p.name for p in tmp.iterdir()} == before
    assert c.get("/api/output-dirs").json()["current"] == "output"
