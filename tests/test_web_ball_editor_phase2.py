"""The ball anchor editor ships goal-impact, pitch-fix, and shot-chain
authoring (palette entries, sub-forms, suggest-endpoint wiring,
shot_chains persistence)."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from src.web.server import create_app


def _client(tmp_path: Path) -> TestClient:
    return TestClient(create_app(output_dir=tmp_path, config_path=None))


def test_editor_served_with_phase2_authoring(tmp_path: Path):
    html = _client(tmp_path).get("/ball-anchor-editor").text
    # Palette gained goal_impact and the pitch-fix mode.
    assert 'id: "goal_impact"' in html
    assert 'id: "pitch_fix"' in html
    # Authoring sub-forms.
    assert 'id="goalAuthor"' in html
    assert 'id="goalElement"' in html
    assert 'id="pitchFixAuthor"' in html
    assert 'id="pitchFixName"' in html
    # Suggest endpoints wired.
    assert "/goal-element-suggest?shot=" in html
    assert "/pitch-fix-suggest?shot=" in html
    # Shot-chain authoring + persistence.
    assert 'id="chainBtn"' in html
    assert 'id="chainList"' in html
    assert "shot_chains" in html
    # Preview warnings surfaced.
    assert "shot_chain_warnings" in html
