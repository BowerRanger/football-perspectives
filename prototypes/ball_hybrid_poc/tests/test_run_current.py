"""A2 spike tests for ``run_current.py``.

Two tiers:

* Fast, pure-unit: ``SyntheticDetector``'s frame-CONTENT keying (tiny fake
  frames, no video decode, no ball stage) — always runs.
* Slow, integration: a real (but cheap — synthetic detector, no real
  WASB inference) ``run_synthetic`` smoke run on gberch, used ALSO to
  prove the overlay never mutates the main repo's source output dirs
  (mtime snapshot before/after). Gated behind ``POC_SLOW=1`` because it
  decodes gberch's real clip end to end through the real ball stage
  (~5-10s, see the module docstring's own smoke-test numbers) — cheap in
  absolute terms but not "always run on every collection" cheap.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from ..ctx import M, load_clip
from ..types import Observation, SynthRun
from ..run_current import (
    SyntheticDetector,
    build_standin_synth_run,
    run_synthetic,
)

_SLOW = os.environ.get("POC_SLOW") == "1"
_GBERCH_OUTPUT = Path(M) / "output"


def _fake_frame(fill: int, shape=(20, 20, 3)) -> np.ndarray:
    return np.full(shape, fill, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Fast unit tests: content-hash keying
# ---------------------------------------------------------------------------

def test_synthetic_detector_keys_by_content_not_call_order():
    """Three distinct fake frames, indexed once; querying them OUT OF
    ORDER (as a seek-then-read pass would) must still resolve each to its
    own observation — proving resolution is by content hash, not by call
    sequence."""
    frames = [_fake_frame(10), _fake_frame(20), _fake_frame(30)]
    run = SynthRun(
        clip_id="fake", scenario="unit",
        observations=(
            Observation(frame=0, uv=(1.0, 1.0), conf=0.9, source="synthetic"),
            Observation(frame=1, uv=(2.0, 2.0), conf=0.8, source="synthetic"),
            Observation(frame=2, uv=(3.0, 3.0), conf=0.7, source="synthetic"),
        ),
        anchors=(), noise_model={},
    )
    det = SyntheticDetector(run, frames=frames)

    # Query frame 2's content, then frame 0's, then frame 1's — deliberately
    # out of index order.
    assert det.detect(frames[2]) == (3.0, 3.0, 0.7)
    assert det.detect(frames[0]) == (1.0, 1.0, 0.9)
    assert det.detect(frames[1]) == (2.0, 2.0, 0.8)
    assert det.hash_misses == 0
    assert det.crop_calls == 0


def test_synthetic_detector_frame_with_no_observation_returns_none():
    frames = [_fake_frame(10), _fake_frame(20)]
    run = SynthRun(
        clip_id="fake", scenario="unit",
        observations=(
            Observation(frame=0, uv=(1.0, 1.0), conf=0.9, source="synthetic"),
        ),
        anchors=(), noise_model={},
    )
    det = SyntheticDetector(run, frames=frames)
    assert det.detect(frames[0]) is not None
    assert det.detect(frames[1]) is None
    assert det.hash_misses == 0
    assert det.crop_calls == 0


def test_synthetic_detector_counts_hash_miss_for_unindexed_content():
    frames = [_fake_frame(10), _fake_frame(20)]
    run = SynthRun(clip_id="fake", scenario="unit", observations=(),
                    anchors=(), noise_model={})
    det = SyntheticDetector(run, frames=frames)
    det.detect(frames[0])
    assert det.hash_misses == 0
    # Content never seen during indexing.
    det.detect(_fake_frame(99))
    assert det.hash_misses == 1
    assert det.positional_fallbacks == 1


def test_synthetic_detector_counts_crop_calls_not_hash_misses():
    """A smaller crop must be counted as a crop call, not conflated with a
    genuine hash miss on full-frame content (the two invariants
    run_synthetic checks are independent signals)."""
    frames = [_fake_frame(10, shape=(20, 20, 3))]
    run = SynthRun(clip_id="fake", scenario="unit", observations=(),
                    anchors=(), noise_model={})
    det = SyntheticDetector(run, frames=frames)
    crop = _fake_frame(10, shape=(8, 8, 3))
    assert det.detect(crop) is None
    assert det.crop_calls == 1
    assert det.hash_misses == 0


def test_synthetic_detector_candidates_merge_obs_fp_and_weak(monkeypatch=None):
    frames = [_fake_frame(10)]
    run = SynthRun(
        clip_id="fake", scenario="unit",
        observations=(
            Observation(frame=0, uv=(1.0, 1.0), conf=0.9, source="synthetic"),
        ),
        anchors=(),
        noise_model={
            "false_positives": {"0": [[5.0, 5.0, 0.4]]},
            "weak_candidates": {"0": [[6.0, 6.0, 0.1]]},
        },
    )
    det = SyntheticDetector(run, frames=frames)
    cands = det.detect_candidates(frames[0], min_score=0.05, top_k=5)
    scores = sorted(c[2] for c in cands)
    assert scores == [0.1, 0.4, 0.9]
    # min_score filters the weak candidate out.
    cands2 = det.detect_candidates(frames[0], min_score=0.2, top_k=5)
    assert sorted(c[2] for c in cands2) == [0.4, 0.9]


def test_synth_anchor_set_build_standin_round_trips_real_anchors():
    """build_standin_synth_run + the loose-dict -> BallAnchor conversion
    must reproduce the real anchor set's frames/states exactly (this is
    what makes the stand-in a faithful smoke test)."""
    pytest.importorskip("cv2")
    if not (Path(M) / "output" / "ball" / "gberch_ball_anchors.json").exists():
        pytest.skip("gberch main-repo output not present")
    from ..run_current import synth_anchor_set

    ctx = load_clip("gberch")
    run = build_standin_synth_run(ctx)
    rebuilt = synth_anchor_set(ctx.clip_id, ctx.image_size, run.anchors)
    assert len(rebuilt.anchors) == len(ctx.anchors.anchors)
    real_by_frame = {a.frame: a.state for a in ctx.anchors.anchors}
    rebuilt_by_frame = {a.frame: a.state for a in rebuilt.anchors}
    assert rebuilt_by_frame == real_by_frame
    assert rebuilt.image_size == ctx.image_size


# ---------------------------------------------------------------------------
# Slow integration: real ball-stage run + overlay-isolation proof
# ---------------------------------------------------------------------------

def _snapshot_mtimes(root: Path) -> dict[str, float]:
    return {
        str(p): p.stat().st_mtime
        for p in root.rglob("*") if p.is_file()
    }


@pytest.mark.skipif(not _SLOW, reason="runs the real ball stage end to end "
                     "on gberch; set POC_SLOW=1 to run")
def test_run_synthetic_gberch_smoke_and_overlay_never_touches_source():
    if not _GBERCH_OUTPUT.exists():
        pytest.skip("main-repo output/ dir not present")

    before = _snapshot_mtimes(_GBERCH_OUTPUT / "ball")

    track = run_synthetic("gberch", "standin")

    after = _snapshot_mtimes(_GBERCH_OUTPUT / "ball")
    assert before == after, (
        "run_synthetic must never write into the main repo's output/ball "
        "dir — the overlay symlinks it read-only"
    )

    assert track.method == "current"
    assert len(track.frames) > 0
    modes = {f.mode for f in track.frames}
    assert modes <= {"grounded", "flight", "occluded", "missing"}
