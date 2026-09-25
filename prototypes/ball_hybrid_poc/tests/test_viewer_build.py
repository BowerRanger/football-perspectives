"""Tiny build-pipeline test for the Ball Truth Lab viewer: mock data -> build
-> the emitted HTML contains the embedded results JSON and the stable
control ids the app JS binds to.

Run: .venv311/bin/python -m pytest prototypes/ball_hybrid_poc/tests/test_viewer_build.py -q
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

VIEWER_DIR = Path(__file__).resolve().parents[1] / "viewer"


def _load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, VIEWER_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


make_mock_results = _load_module("ball_poc_make_mock_results", "make_mock_results.py")
build_viewer = _load_module("ball_poc_build_viewer", "build_viewer.py")

REQUIRED_IDS = [
    "clip-select",
    "scenario-select",
    "summary-strip",
    "theme-toggle",
    "cam-broadcast",
    "cam-side",
    "cam-top",
    "cam-behind",
    "cam-free",
    "method-toggles",
    "scene-canvas-wrap",
    "scene-canvas",
    "scene-legend",
    "metrics-table",
    "metrics-thead",
    "metrics-tbody",
    "metrics-more",
    "metrics-thead-more",
    "metrics-tbody-more",
    "frame-scrubber",
    "play-btn",
    "frame-readout",
    "error-chart",
]


def _write_mock_clip(tmp_path: Path, clip_id: str, seed: int) -> Path:
    results = make_mock_results.build_clip_results(clip_id, fps=30, image_size=(1920, 1080), seed=seed)
    clip_dir = tmp_path / clip_id
    clip_dir.mkdir()
    out_path = clip_dir / "results.json"
    out_path.write_text(json.dumps(results))
    return out_path


def test_build_viewer_embeds_data_and_required_ids(tmp_path):
    _write_mock_clip(tmp_path, "gberch", seed=1)
    _write_mock_clip(tmp_path, "origi01", seed=2)

    out_html = tmp_path / "viewer.html"
    clips = build_viewer.load_clip_results([str(tmp_path)])
    html = build_viewer.build_html(clips)
    out_html.write_text(html)

    # Artifact-page structural contract: starts with the exact title, no
    # doctype/html/head/body wrapper tags.
    assert html.startswith('<title>Ball Truth Lab</title>')
    for banned in ("<!doctype", "<html", "<head>", "<body"):
        assert banned not in html.lower()

    # Embedded data: both clip ids present as JSON keys, not fetched.
    assert '"gberch"' in html
    assert '"origi01"' in html
    assert "fetch(" not in html

    # Stable control ids the app JS binds to are all present.
    for control_id in REQUIRED_IDS:
        assert f'id="{control_id}"' in html, f"missing id={control_id!r}"

    # External scripts only from the two allowed, pinned CDN sources.
    assert "cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js" in html
    assert "cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js" in html


def test_rounding_keeps_precision_reasonable(tmp_path):
    _write_mock_clip(tmp_path, "gberch", seed=3)
    clips = build_viewer.load_clip_results([str(tmp_path)])
    frame0 = clips["gberch"]["scenarios"]["lob_pass"]["truth"]["frames"][0]
    x, y, z = frame0["xyz"]
    for v in (x, y, z):
        # rounded to 1 cm -> at most 2 decimal places
        assert round(v, 2) == v


def test_build_viewer_cli_roundtrip(tmp_path):
    mock_dir = tmp_path / "mock"
    mock_dir.mkdir()
    _write_mock_clip(mock_dir, "gberch", seed=4)

    out_path = tmp_path / "out" / "viewer.html"
    sys.argv = [
        "build_viewer.py",
        "--results", str(mock_dir),
        "--out", str(out_path),
    ]
    build_viewer.main()

    assert out_path.exists()
    html = out_path.read_text()
    assert html.startswith("<title>Ball Truth Lab</title>")
    assert '"gberch"' in html
    assert out_path.stat().st_size < 8 * 1024 * 1024


def test_findings_omitted_when_not_provided(tmp_path):
    _write_mock_clip(tmp_path, "gberch", seed=5)
    clips = build_viewer.load_clip_results([str(tmp_path)])
    html = build_viewer.build_html(clips)
    assert 'id="findings-panel"' not in html


def test_findings_included_when_provided(tmp_path):
    _write_mock_clip(tmp_path, "gberch", seed=6)
    clips = build_viewer.load_clip_results([str(tmp_path)])
    fragment = "<p>The hybrid method wins on depth accuracy.</p>"
    html = build_viewer.build_html(clips, fragment)

    assert 'id="findings-panel"' in html
    assert 'id="findings-details" open' in html
    assert fragment in html
    # inserted directly under the summary strip, before the main 3-D/metrics grid
    assert html.index('id="findings-panel"') > html.index('id="summary-strip"')
    assert html.index('id="findings-panel"') < html.index('class="main-grid"')


def test_findings_cli_flag(tmp_path):
    mock_dir = tmp_path / "mock"
    mock_dir.mkdir()
    _write_mock_clip(mock_dir, "gberch", seed=7)
    findings_file = tmp_path / "findings.html"
    findings_file.write_text("<p>Headline: hybrid wins.</p>")

    out_path = tmp_path / "out" / "viewer.html"
    sys.argv = [
        "build_viewer.py",
        "--results", str(mock_dir),
        "--findings", str(findings_file),
        "--out", str(out_path),
    ]
    build_viewer.main()

    html = out_path.read_text()
    assert "Headline: hybrid wins." in html
    assert 'id="findings-panel"' in html


def test_real_tab_tolerates_null_ground_truth_points(tmp_path):
    """Real anchors_heldout/fixes entries may have a null xyz_gt/xyz (engine
    couldn't resolve ground truth for that held-out point) -- the build must
    still succeed and embed them, since the viewer JS is responsible for
    skipping them rather than the build script filtering them out."""
    results = make_mock_results.build_clip_results("gberch", fps=30, image_size=(1920, 1080), seed=8)
    results["real"]["anchors_heldout"].append({"frame": 0, "xyz_gt": None})
    results["real"]["fixes"].append({"frame": 0, "xyz": None})
    clip_dir = tmp_path / "gberch"
    clip_dir.mkdir()
    (clip_dir / "results.json").write_text(json.dumps(results))

    clips = build_viewer.load_clip_results([str(tmp_path)])
    html = build_viewer.build_html(clips)
    assert '"xyz_gt":null' in html.replace(" ", "")
