"""The ball anchor editor ships the quality timeline strip wired to
/ball-quality (strip canvas, annotate-next list, click-to-seek)."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from src.web.server import create_app


def _client(tmp_path: Path) -> TestClient:
    return TestClient(create_app(output_dir=tmp_path, config_path=None))


def test_editor_served_with_quality_strip(tmp_path: Path):
    html = _client(tmp_path).get("/ball-anchor-editor").text
    assert 'id="qualityStrip"' in html
    assert 'id="qualityCanvas"' in html
    assert 'id="annotateNext"' in html
    # Strip is fed by the new endpoint and seeks on click.
    assert "/ball-quality/" in html
    assert "renderQualityStrip" in html
    assert "annotate_next" in html
