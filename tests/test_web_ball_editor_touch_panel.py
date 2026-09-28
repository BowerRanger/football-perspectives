"""The ball anchor editor exposes player-touch authoring (palette entry,
player/bone selectors, click-to-suggest via /joints-near)."""

from __future__ import annotations

from tests.frontend_source import assert_markers, bundle_text, source_text

MARKERS = [
    '"player_touch"',
    # Click-to-suggest hits the joints-near endpoint.
    "/joints-near",
    # Confidence surfaced on auto rows.
    ".confidence",
]


def test_editor_source_has_touch_authoring():
    assert_markers(source_text("pages/ball-anchor-editor"), MARKERS, "ball anchor editor source")


def test_committed_build_has_touch_authoring():
    assert_markers(bundle_text(), ["player_touch", "/joints-near"], "committed dashboard build")
