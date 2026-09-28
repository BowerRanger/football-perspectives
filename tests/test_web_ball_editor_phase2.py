"""The ball anchor editor ships goal-impact, pitch-fix, and shot-chain
authoring (palette entries, sub-forms, suggest-endpoint wiring,
shot_chains persistence)."""

from __future__ import annotations

from tests.frontend_source import assert_markers, bundle_text, source_text

MARKERS = [
    # Palette gained goal_impact and the pitch-fix mode.
    '"goal_impact"',
    '"pitch_fix"',
    # Suggest endpoints wired.
    "/goal-element-suggest",
    "/pitch-fix-suggest",
    # Shot-chain authoring + persistence, preview warnings surfaced.
    "shot_chains",
    "shot_chain_warnings",
]


def test_editor_source_has_phase2_authoring():
    assert_markers(source_text("pages/ball-anchor-editor"), MARKERS, "ball anchor editor source")


def test_committed_build_has_phase2_authoring():
    assert_markers(bundle_text(), [m.strip('"') for m in MARKERS], "committed dashboard build")
