"""Each dashboard area still wires every backend endpoint it used before the
React rewrite. Endpoint URLs survive minification, so they are checked in
both the source and the committed build."""

from __future__ import annotations

import pytest

from tests.frontend_source import assert_markers, bundle_text, source_text

AREAS: dict[str, tuple[str, list[str]]] = {
    "tracking": (
        "features/stages/tracking",
        ["/tracking/shots", "/tracking/preview", "/tracking/frames", "/api/tracks/split",
         "/api/tracks/merge", "/api/tracks/merge-by-name", "/api/tracks/ignore-unknown/",
         "/delete-bulk", "/interpolate-gaps"],
    ),
    "camera": (
        "features/stages/camera",
        ["/api/camera/metrics", "/camera/track", "/anchors/"],
    ),
    "hmr_world": (
        "features/stages/hmr-world",
        ["/hmr_world/kp2d_players", "/hmr_world/kp2d_preview", "/hmr_world/preview",
         "/api/run-shot-player", "/api/run-shot"],
    ),
    "refined_poses": (
        "features/stages/refined-poses",
        ["/refined_poses/summary", "/refined_poses/diagnostics", "/refined_poses/players"],
    ),
    "export": (
        "features/stages/export",
        ["/api/export/camera-selection", "/api/export/available-players", "/api/export/scene.glb"],
    ),
    "anchor_editor": (
        "pages/anchor-editor",
        ["/api/anchor/snap", "/camera/detected-lines", "/pitch_lines", "/stadiums", "/landmarks"],
    ),
    "viewer": (
        "pages/viewer",
        ["/api/smpl_model", "/api/export/metadata", "/refined_poses/preview", "/ball/preview"],
    ),
    "ball": (
        "features/stages/ball",
        ["<BallAnchorEditor", "import(\"three\")"],
    ),
    "ball_anchor_editor": (
        "pages/ball-anchor-editor",
        ["/ball-anchors/", "/ball/preview", "/api/video/"],
    ),
}


@pytest.mark.parametrize("area", sorted(AREAS))
def test_area_source_wires_its_endpoints(area: str) -> None:
    rel, markers = AREAS[area]
    assert_markers(source_text(rel), markers, f"{area} source ({rel})")


def test_committed_build_wires_every_area() -> None:
    bundle = bundle_text()
    for area, (_, markers) in AREAS.items():
        endpoints = [m for m in markers if m.startswith("/")]
        assert_markers(bundle, endpoints, f"committed build ({area})")


def test_editors_are_components_not_iframes() -> None:
    src = source_text("features", "pages")
    assert "<iframe" not in src, "editors must be embedded as React components, not iframes"
    assert_markers(source_text("features/stages/camera"), ["<AnchorEditor"], "camera stage")
    assert_markers(source_text("features/stages/export"), ["<Viewer"], "export stage")


def test_unsaved_edits_are_guarded() -> None:
    for rel in ("pages/anchor-editor", "pages/ball-anchor-editor", "features/stages/prepare-shots"):
        assert_markers(source_text(rel), ["useUnsavedGuard("], rel)
