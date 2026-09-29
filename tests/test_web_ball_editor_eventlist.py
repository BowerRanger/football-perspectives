"""The ball anchor editor ships the merged event list with dismiss/undo,
persisted dismissed_auto, and end-frame span controls."""

from __future__ import annotations

from tests.frontend_source import assert_markers, bundle_text, source_text

MARKERS = [
    "dismissed_auto",  # payload key round-trip
    "Dismiss this suggestion",
    "Undo dismissal",
    "Set end frame",
    "Clear end frame",
    # merged chronological list marker
    "Events (manual + auto",
]


def test_editor_source_has_event_list():
    assert_markers(source_text("pages/ball-anchor-editor"), MARKERS, "ball anchor editor source")


def test_committed_build_has_event_list():
    assert_markers(bundle_text(), MARKERS, "committed dashboard build")
