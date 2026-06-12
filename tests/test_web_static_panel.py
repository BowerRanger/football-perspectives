"""The prepare-shots panel module is served and referenced."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.web.server import create_app


@pytest.fixture
def client(tmp_path: Path):
    app = create_app(output_dir=tmp_path, config_path=None)
    return TestClient(app)


@pytest.mark.integration
def test_panel_script_served_and_referenced(client) -> None:
    res = client.get("/static/js/prepare_shots_panel.js")
    assert res.status_code == 200
    assert "renderPrepareShots" in res.text
    assert "/api/shots/bulk" in res.text

    index = client.get("/").text
    assert "/static/js/prepare_shots_panel.js" in index


@pytest.mark.integration
def test_index_no_longer_defines_moved_functions(client) -> None:
    index = client.get("/").text
    assert "async function renderPrepareShots" not in index
    assert "function _buildSyncTimeline" not in index
