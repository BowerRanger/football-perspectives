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


def test_run_log_can_cancel_and_is_virtualised() -> None:
    assert_markers(source_text("hooks/use-pipeline.tsx"), ["/cancel", '"cancelled"'], "pipeline store")
    assert_markers(source_text("components/log-dock.tsx"), ["useVirtualizer", "Cancel run"], "log dock")
    assert_markers(bundle_text(), ["/cancel", "Cancel run"], "committed build")


def test_track_edits_are_undoable() -> None:
    assert_markers(source_text("features/stages/tracking"), ["/api/tracks/undo", "Undo"], "tracking source")
    assert_markers(bundle_text(), ["/api/tracks/undo"], "committed build")


def test_every_transport_uses_the_shared_frame_player() -> None:
    players = [
        "features/stages/hmr-world/kp2d-viewer.tsx",
        "features/stages/hmr-world/trajectory-panel.tsx",
        "features/stages/camera/overlaid-pitch-map.tsx",
        "features/stages/tracking/track-video.tsx",
        "pages/viewer/transport.tsx",
        "pages/anchor-editor/anchor-transport.tsx",
        "pages/ball-anchor-editor/transport.tsx",
    ]
    for rel in players:
        assert_markers(source_text(rel), ["<FramePlayer"], rel)
    # No hand-rolled scrubbers left: the only Sliders outside the shared
    # player are value controls (sync offsets, touch confidence).
    from tests.frontend_source import FRONTEND_SRC

    sliders = sorted(
        str(p.relative_to(FRONTEND_SRC))
        for p in FRONTEND_SRC.rglob("*.tsx")
        if "<Slider" in p.read_text(encoding="utf-8") and "components/" not in str(p.relative_to(FRONTEND_SRC))
    )
    assert sliders == [
        "features/stages/prepare-shots/sync-offsets.tsx",
        "pages/ball-anchor-editor/authoring-panels.tsx",
    ]


def test_failed_reads_are_not_rendered_as_empty() -> None:
    # getJsonOrNull swallows every failure; it is reserved for optional
    # lookups. Main payloads use getJson / getJsonOr404 + useResource.
    from tests.frontend_source import FRONTEND_SRC

    uses = sum(
        p.read_text(encoding="utf-8").count("getJsonOrNull<")
        for p in FRONTEND_SRC.rglob("*.ts*")
        if p.name != "api.ts" or "lib" not in p.parts
    )
    assert uses <= 6, f"{uses} best-effort reads — main payloads must surface errors"


def test_unsaved_edits_are_guarded() -> None:
    for rel in ("pages/anchor-editor", "pages/ball-anchor-editor", "features/stages/prepare-shots"):
        assert_markers(source_text(rel), ["useUnsavedGuard("], rel)
