"""Camera-stage regression gate against golden clips.

Each dir under ``tests/regression/camera/<clip_id>/`` is a golden clip:
committed anchors (the operator's hand-clicked landmarks are the
pseudo-ground-truth) plus a ``baseline.json`` captured by
``scripts/capture_camera_regression_baseline.py``. The test re-solves
the camera stage from scratch on the real clip and fails if anchor-click
reprojection, coverage, or confidence regress beyond the baseline's
nondeterminism-aware tolerances.

Opt-in and local-media-bound: minutes per clip, so it only runs with
``-m regression`` (see the collection hook in tests/conftest.py) and
skips per-clip when the local media is missing or differs from the clip
the baseline was captured against. Adding a golden clip = run the
capture script and commit the new fixture dir; no code changes.

    .venv311/bin/python -m pytest tests/test_camera_regression.py -m regression -q
"""

import hashlib
import json
from pathlib import Path

import pytest

from src.utils.anchor_click_eval import score_track
from src.utils.camera_regression import (
    build_solve_dir,
    compare_to_baseline,
    discover_clips,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = Path(__file__).parent / "regression" / "camera"

_NO_FIXTURES = "<no-fixtures>"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@pytest.mark.regression
@pytest.mark.parametrize(
    "clip_id", discover_clips(FIXTURE_ROOT) or [_NO_FIXTURES])
def test_camera_solve_has_not_regressed(clip_id: str, tmp_path: Path):
    if clip_id == _NO_FIXTURES:
        pytest.skip("no camera regression fixtures committed yet")

    fixture = FIXTURE_ROOT / clip_id
    baseline = json.loads((fixture / "baseline.json").read_text())
    anchors = json.loads((fixture / "anchors.json").read_text())
    shot_fixture = json.loads((fixture / "shot.json").read_text())

    clip_path = REPO_ROOT / baseline["clip_file"]
    if not clip_path.exists():
        pytest.skip(f"local media missing: {baseline['clip_file']}")
    if _sha256(clip_path) != baseline["clip_sha256"]:
        pytest.skip(
            f"local clip {baseline['clip_file']} differs from the one the "
            "baseline was captured against — re-capture the baseline")

    from src.pipeline.config import load_config
    from src.pipeline.runner import run_pipeline

    build_solve_dir(tmp_path, shot_fixture, anchors, clip_path)
    run_pipeline(output_dir=tmp_path, stages="camera", from_stage=None,
                 config=load_config())

    track_path = tmp_path / "camera" / f"{clip_id}_camera_track.json"
    assert track_path.exists(), "camera stage produced no track"
    metrics = score_track(anchors, json.loads(track_path.read_text()))
    failures = compare_to_baseline(metrics, baseline)

    summary = (
        f"clicks={metrics['clicks']} med={metrics['med_px']}px "
        f"p90={metrics['p90_px']}px max={metrics['max_px']}px "
        f"covered={metrics['anchor_frames_covered']}"
        f"/{metrics['anchor_frames_total']} "
        f"conf={metrics['mean_confidence']}")
    assert not failures, (
        f"camera solve regressed on {clip_id} ({summary}):\n  "
        + "\n  ".join(failures))
