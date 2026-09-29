"""The dashboard pages are served as the committed React SPA shell, and the
prepare-shots / render panels ship in both the source and the build."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.web.server import create_app
from tests.frontend_source import assert_markers, bundle_text, source_text

PAGE_ROUTES = ["/", "/anchor_editor", "/ball-anchor-editor", "/viewer"]


@pytest.fixture
def client(tmp_path: Path):
    app = create_app(output_dir=tmp_path, config_path=None)
    return TestClient(app)


@pytest.mark.integration
@pytest.mark.parametrize("route", PAGE_ROUTES)
def test_page_routes_serve_spa_shell(client, route: str) -> None:
    res = client.get(route)
    assert res.status_code == 200
    assert res.headers["cache-control"] == "no-store"
    assert '<div id="root"></div>' in res.text
    assert "/static/app/assets/" in res.text


@pytest.mark.integration
def test_spa_assets_referenced_by_shell_are_served(client) -> None:
    html = client.get("/").text
    asset_paths = [
        part.split('"')[0]
        for part in html.split('src="')[1:] + html.split('href="')[1:]
        if part.startswith("/static/app/")
    ]
    assert asset_paths, "shell references no built assets"
    for path in asset_paths:
        assert client.get(path).status_code == 200, path


@pytest.mark.integration
def test_legacy_static_pages_are_gone(client) -> None:
    for legacy in ["index.html", "viewer.html", "anchor_editor.html", "ball_anchor_editor.html",
                   "js/prepare_shots_panel.js", "js/render_panel.js"]:
        assert client.get(f"/static/{legacy}").status_code == 404, legacy


def test_prepare_shots_panel_wires_bulk_and_sync() -> None:
    markers = ["/api/shots/bulk", "/api/shots/manifest", "/api/sync", "/api/shots/upload-reel", "/api/match"]
    assert_markers(source_text("features/stages/prepare-shots"), markers, "prepare-shots source")
    assert_markers(bundle_text(), markers, "committed dashboard build")


def test_render_panel_wires_outputs_selection_and_video() -> None:
    markers = ["/api/render/outputs", "/api/render/selection", "/api/render/video/"]
    assert_markers(source_text("features/stages/render"), markers, "render source")
    assert_markers(bundle_text(), markers, "committed dashboard build")


def test_rerun_uses_server_side_clean_first() -> None:
    # Re-run must not DELETE outputs client-side before the run is accepted.
    src = source_text("hooks/use-pipeline.tsx")
    assert "clean_first" in src
    assert "/api/output/" not in src
