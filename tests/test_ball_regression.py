"""Ball-stage regression gate against golden clips.

Mirrors ``tests/test_camera_regression.py``'s pattern. Each dir under
``tests/regression/ball/<clip_id>/`` is a golden clip: a frozen manual
anchor set + frozen synthetic ``mismatch``-scenario truth/evidence (the
pseudo-ground-truth) plus a ``baseline.json`` captured by
``scripts/capture_ball_regression_baseline.py``. The test re-runs the real
ball stage (the shipped ``ball.trajectory``) against that frozen synthetic
evidence AND the real detector's 2-fold held-out anchor evaluation, and
fails if accuracy regresses beyond the baseline's measured-spread-aware
tolerances.

Opt-in: only runs with ``-m regression`` (see the collection hook in
``tests/conftest.py``). Skips per-clip when the real clip media in
``$M/output*`` is missing or has changed since the baseline was captured
— this gate re-runs the real ball stage on real footage (with a cached
detector), so it needs the clip video + camera track to exist locally.

    .venv311/bin/python -m pytest tests/test_ball_regression.py -m regression -q
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from src.schemas.ball_anchor import BallAnchorSet
from src.utils import ball_bench_metrics as BM
from src.utils import ball_bench_regression as BRG
from src.utils import ball_bench_runner as BR
from src.utils.ball_bench_clip import CLIPS, load_clip
from src.utils.ball_bench_types import SynthRun, TruthTrack

FIXTURE_ROOT = Path(__file__).parent / "regression" / "ball"

_NO_FIXTURES = "<no-fixtures>"
_SYNTH_KEYS = ("pct_le_20cm", "p95", "ground_float_sink",
               "naturalness_violations_minus_truth")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@pytest.mark.regression
@pytest.mark.parametrize(
    "clip_id", BRG.discover_clips(FIXTURE_ROOT) or [_NO_FIXTURES])
def test_ball_stage_has_not_regressed(clip_id: str, tmp_path: Path):
    if clip_id == _NO_FIXTURES:
        pytest.skip("no ball regression fixtures committed yet")

    fixture = FIXTURE_ROOT / clip_id
    baseline = json.loads((fixture / "baseline.json").read_text())
    shot_fixture = json.loads((fixture / "shot.json").read_text())
    frozen_anchors = BallAnchorSet.load(fixture / "anchors.json")
    truth = TruthTrack.from_json(
        json.loads((fixture / "truth_mismatch.json").read_text()))
    synth = SynthRun.from_json(
        json.loads((fixture / "synth_obs_mismatch.json").read_text()))

    if clip_id not in CLIPS:
        pytest.skip(f"clip {clip_id!r} not in the CLIPS registry")
    output_dir = Path(CLIPS[clip_id][0])
    video_path = output_dir / shot_fixture["video_relpath"]
    if not video_path.exists():
        pytest.skip(f"local media missing: {video_path}")
    if _sha256(video_path) != shot_fixture["video_sha256"]:
        pytest.skip(
            f"local clip {video_path} differs from the one the baseline "
            "was captured against — re-capture the baseline")

    trajectory = BR.shipped_trajectory()
    baseline_trajectory = baseline.get("trajectory", "reference")
    if baseline_trajectory != trajectory:
        pytest.fail(
            f"baseline for {clip_id!r} was captured with ball.trajectory="
            f"{baseline_trajectory!r} but the pipeline ships {trajectory!r} — "
            "re-capture it with scripts/capture_ball_regression_baseline.py")

    clip_ctx = load_clip(clip_id)
    clip_ctx = dataclasses.replace(clip_ctx, anchors=frozen_anchors)

    # --- synthetic mismatch scenario, shipped ball.trajectory -----------
    track = BR.run_synthetic(clip_ctx, synth, "mismatch", trajectory)
    side_cam = BM.build_side_camera([f.xyz for f in truth.frames])
    synth_flat, _detail = BM.compute_scenario_metrics(
        clip_ctx, track, truth, side_camera=side_cam)

    # --- real detector, 2-fold held-out anchor error (isolated scratch) -
    det_cache = BR.det_cache_path(clip_id, bench_root=tmp_path)
    combined_errs: list[float] = []
    for fold in range(BR.N_FOLDS):
        real_track = BR.run_real(clip_ctx, trajectory, fold=fold,
                                  det_cache=det_cache, bench_root=tmp_path)
        held = BR.anchor_heldout_error(clip_ctx, real_track, fold,
                                        bench_root=tmp_path)
        combined_errs.extend(held["errs"])
    real_flat = {
        "p50": float(np.percentile(combined_errs, 50)) if combined_errs else None,
        "p95": float(np.percentile(combined_errs, 95)) if combined_errs else None,
        "n": len(combined_errs),
    }

    failures = BRG.compare_to_baseline(synth_flat, real_flat, baseline)

    summary = (
        f"synth: p50={synth_flat['p50']} p95={synth_flat['p95']} "
        f"pct_le_20cm={synth_flat['pct_le_20cm']} "
        f"float_sink={synth_flat['ground_float_sink']} "
        f"nat_delta={synth_flat['naturalness_violations_minus_truth']} | "
        f"real: p50={real_flat['p50']} n={real_flat['n']}")
    assert not failures, (
        f"ball stage regressed on {clip_id} ({summary}):\n  "
        + "\n  ".join(failures))


@pytest.mark.regression
def test_gberch_finish_held_out_line_cross():
    """D6.5 held-out case: the decisive gberch shot must cross the goal line
    within 0.3 m of the operator's frame-394 anchor ray ∩ x=0, with no hand
    splice. Runs the shipped hybrid solver on the frozen fast fixture
    (tests/fixtures/ball/gberch_finish), so no media is needed."""
    from tests.test_ball_goal_constraint import FIXTURE, _run_finish

    if not FIXTURE.exists():
        pytest.skip("gberch_finish fixture missing")
    d, ctx, _frames, diag = _run_finish()
    anchor = next(a for a in d["manual_anchors"] if a["frame"] == 394)
    C, ray = ctx.ray(394, tuple(anchor["image_xy"]))
    operator_pt = C + (0.0 - C[0]) / ray[0] * ray
    gc = diag["goal_check"]
    assert gc["status"] == "ok", gc
    got = np.array(gc["line_cross"]["xyz"])
    assert np.linalg.norm(got - operator_pt) < d["expected"]["tol_m"], (got, operator_pt)
