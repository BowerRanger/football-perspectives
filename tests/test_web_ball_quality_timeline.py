"""The ball anchor editor ships the quality timeline strip wired to
/ball-quality (strip canvas, annotate-next list, click-to-seek)."""

from __future__ import annotations

from tests.frontend_source import assert_markers, bundle_text, source_text

MARKERS = [
    # Strip is fed by the ball-quality endpoint and lists the next weak spans.
    "/ball-quality/",
    "annotate_next",
]


def test_editor_source_has_quality_strip():
    assert_markers(source_text("pages/ball-anchor-editor"), MARKERS, "ball anchor editor source")


def test_committed_build_has_quality_strip():
    assert_markers(bundle_text(), MARKERS, "committed dashboard build")
