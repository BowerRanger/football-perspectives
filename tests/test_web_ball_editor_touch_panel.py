"""The ball anchor editor exposes player-touch authoring (palette entry,
player/bone selectors, click-to-suggest via /joints-near)."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from src.web.server import create_app


def _client(tmp_path: Path) -> TestClient:
    return TestClient(create_app(output_dir=tmp_path, config_path=None))


def test_editor_served_with_touch_authoring(tmp_path: Path):
    html = _client(tmp_path).get("/ball-anchor-editor").text
    # Palette gained a player_touch tag and a touch-authoring section.
    assert 'id: "player_touch"' in html
    assert 'id="touchAuthor"' in html
    assert 'id="touchBone"' in html
    # Click-to-suggest hits the new endpoint.
    assert "/joints-near?shot=" in html
    # Confidence surfaced on auto rows.
    assert "a.confidence" in html
