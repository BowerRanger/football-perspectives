"""The ball anchor editor ships the merged event list with dismiss/undo,
persisted dismissed_auto, and end-frame span controls."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from src.web.server import create_app


def _client(tmp_path: Path) -> TestClient:
    return TestClient(create_app(output_dir=tmp_path, config_path=None))


def test_editor_served_with_event_list(tmp_path: Path):
    html = _client(tmp_path).get("/ball-anchor-editor").text
    assert "dismissedAuto" in html          # JS state
    assert "dismissed_auto" in html         # payload key round-trip
    assert 'title="Dismiss this suggestion' in html
    assert 'title="Undo dismissal' in html
    assert 'title="Set end frame' in html
    assert 'title="Clear end frame' in html
    # merged chronological list marker
    assert "Events (manual + auto" in html
