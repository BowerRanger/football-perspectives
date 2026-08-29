"""Tests for src/utils/ball_detection_cache.py — the opt-in persistent
content-hash cache around a BallDetector, promoted from
scripts/eval_ball_accuracy.py's CachingDetector."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.utils.ball_detection_cache import (
    CachingBallDetector,
    build_detector_fingerprint,
    wrap_if_enabled,
)
from src.utils.ball_detector import BallDetector


class _Counting(BallDetector):
    """Records how many times detect/detect_candidates are actually
    invoked, returning a deterministic answer keyed on a counter so a
    cache hit (no inner call) is observably different from a miss."""

    SUPPORTS_REDETECT = True

    def __init__(self, answer: tuple[float, float, float] | None = (10.0, 20.0, 0.9)):
        self.calls = 0
        self.candidate_calls = 0
        self.reset_calls = 0
        self._answer = answer

    def detect(self, frame):
        self.calls += 1
        return self._answer

    def detect_candidates(self, frame, min_score, top_k=5):
        self.candidate_calls += 1
        if self._answer is None or self._answer[2] < min_score:
            return []
        return [self._answer]

    def reset(self):
        self.reset_calls += 1


def _frame(fill: int, shape=(64, 64, 3)) -> np.ndarray:
    return np.full(shape, fill, dtype=np.uint8)


@pytest.mark.unit
def test_cache_hit_returns_stored_result_without_calling_inner(tmp_path: Path):
    inner = _Counting()
    cache = tmp_path / "det.json"
    det = CachingBallDetector(inner, cache)
    f1 = _frame(0)

    assert det.detect(f1) == (10.0, 20.0, 0.9)
    assert inner.calls == 1
    assert det.detect(f1) == (10.0, 20.0, 0.9)  # cache hit
    assert inner.calls == 1  # inner NOT called again


@pytest.mark.unit
def test_miss_calls_through_and_stores(tmp_path: Path):
    inner = _Counting()
    cache = tmp_path / "det.json"
    det = CachingBallDetector(inner, cache)
    f1, f2 = _frame(0), _frame(7)

    assert det.detect(f1) == (10.0, 20.0, 0.9)
    assert det.detect(f2) == (10.0, 20.0, 0.9)
    assert inner.calls == 2  # two distinct frames -> two misses
    det.save()

    # A fresh instance loading the saved file replays without calling in.
    inner2 = _Counting()
    det2 = CachingBallDetector(inner2, cache)
    assert det2.detect(f1) == (10.0, 20.0, 0.9)
    assert det2.detect(f2) == (10.0, 20.0, 0.9)
    assert inner2.calls == 0


@pytest.mark.unit
def test_none_detection_is_cached_too(tmp_path: Path):
    inner = _Counting(answer=None)
    cache = tmp_path / "det.json"
    det = CachingBallDetector(inner, cache)
    f1 = _frame(1)

    assert det.detect(f1) is None
    assert det.detect(f1) is None
    assert inner.calls == 1  # second call served from cache


@pytest.mark.unit
def test_crop_and_full_frame_calls_dont_collide(tmp_path: Path):
    """detect() (full frame) and detect_candidates() (used for both the
    second-pass full-frame corridor gate and crop zooms) cache into
    separate stores, so identical byte content passed to one never
    satisfies a lookup for the other."""
    inner = _Counting()
    cache = tmp_path / "det.json"
    det = CachingBallDetector(inner, cache)
    frame = _frame(3)

    assert det.detect(frame) == (10.0, 20.0, 0.9)
    assert inner.calls == 1
    # Same exact frame content via detect_candidates must still be a
    # miss against the *candidates* store.
    cands = det.detect_candidates(frame, min_score=0.05, top_k=5)
    assert cands == [(10.0, 20.0, 0.9)]
    assert inner.candidate_calls == 1

    # A genuinely different crop (different shape) never collides either.
    crop = _frame(3, shape=(32, 32, 3))
    det.detect_candidates(crop, min_score=0.05, top_k=5)
    assert inner.candidate_calls == 2


@pytest.mark.unit
def test_detect_candidates_key_includes_min_score_and_top_k(tmp_path: Path):
    inner = _Counting()
    cache = tmp_path / "det.json"
    det = CachingBallDetector(inner, cache)
    frame = _frame(9)

    det.detect_candidates(frame, min_score=0.05, top_k=5)
    assert inner.candidate_calls == 1
    # Different min_score/top_k on the identical frame is a distinct key.
    det.detect_candidates(frame, min_score=0.5, top_k=3)
    assert inner.candidate_calls == 2
    # Repeating the first call is now a hit.
    det.detect_candidates(frame, min_score=0.05, top_k=5)
    assert inner.candidate_calls == 2


@pytest.mark.unit
def test_fingerprint_mismatch_invalidates_cache(tmp_path: Path):
    inner = _Counting()
    cache = tmp_path / "det.json"
    fp_a = {"class": "Counting", "checkpoint_sha256": "aaa"}
    det = CachingBallDetector(inner, cache, fingerprint=fp_a)
    f1 = _frame(0)
    det.detect(f1)
    assert inner.calls == 1
    det.save()

    # Same fingerprint: replays.
    inner_same = _Counting()
    det_same = CachingBallDetector(inner_same, cache, fingerprint=fp_a)
    det_same.detect(f1)
    assert inner_same.calls == 0

    # Different fingerprint (simulating a checkpoint/config change):
    # cache is discarded and the detector runs fresh.
    fp_b = {"class": "Counting", "checkpoint_sha256": "bbb"}
    inner_changed = _Counting()
    det_changed = CachingBallDetector(inner_changed, cache, fingerprint=fp_b)
    det_changed.detect(f1)
    assert inner_changed.calls == 1  # cache miss: fingerprint invalidated it


@pytest.mark.unit
def test_no_fingerprint_always_trusts_disk_cache(tmp_path: Path):
    """fingerprint=None (the default, and what eval_ball_accuracy.py still
    uses) must behave exactly like the pre-existing CachingDetector: no
    validation, on-disk cache always replayed regardless of what built
    it. This keeps docs/superpowers/notes/ball-accuracy/det_cache/*.json
    (written before the fingerprint concept existed) working unchanged."""
    inner = _Counting()
    cache = tmp_path / "det.json"
    det = CachingBallDetector(inner, cache)
    f1 = _frame(0)
    det.detect(f1)
    det.save()

    inner2 = _Counting()
    det2 = CachingBallDetector(inner2, cache)  # no fingerprint passed
    assert det2.detect(f1) == (10.0, 20.0, 0.9)
    assert inner2.calls == 0


@pytest.mark.unit
def test_supports_redetect_propagates_through_wrapper(tmp_path: Path):
    class _NoRedetect(_Counting):
        SUPPORTS_REDETECT = False

    det = CachingBallDetector(_NoRedetect(), tmp_path / "det.json")
    assert det.SUPPORTS_REDETECT is False

    det2 = CachingBallDetector(_Counting(), tmp_path / "det2.json")
    assert det2.SUPPORTS_REDETECT is True


@pytest.mark.unit
def test_build_detector_fingerprint_wasb_includes_checkpoint_hash(tmp_path: Path):
    ckpt = tmp_path / "ckpt.pth.tar"
    ckpt.write_bytes(b"weights-v1")
    cfg = {
        "detector": "wasb",
        "wasb": {"checkpoint": str(ckpt), "confidence": 0.3,
                 "input_size": [512, 288]},
    }
    inner = _Counting()
    fp1 = build_detector_fingerprint(cfg, inner)
    assert fp1["class"] == "_Counting"
    assert fp1["backend"] == "wasb"
    assert fp1["checkpoint_path"] == str(ckpt.resolve())
    assert fp1["checkpoint_size"] == len(b"weights-v1")
    assert "checkpoint_sha256" in fp1

    # Changing the checkpoint file content changes the fingerprint.
    ckpt.write_bytes(b"weights-v2-different-length")
    fp2 = build_detector_fingerprint(cfg, inner)
    assert fp2["checkpoint_sha256"] != fp1["checkpoint_sha256"]
    assert fp2 != fp1

    # Changing confidence/input_size config also changes the fingerprint
    # even with the checkpoint held fixed.
    cfg_conf = dict(cfg, wasb=dict(cfg["wasb"], confidence=0.5))
    fp3 = build_detector_fingerprint(cfg_conf, inner)
    assert fp3 != fp2


@pytest.mark.unit
def test_checkpoint_fingerprint_end_to_end_invalidation(tmp_path: Path):
    """A realistic end-to-end: two detector 'generations' built from
    different checkpoint files must not share cached detections."""
    ckpt = tmp_path / "ckpt.pth.tar"
    ckpt.write_bytes(b"gen-1")
    cfg = {"detector": "wasb",
           "wasb": {"checkpoint": str(ckpt), "confidence": 0.3,
                    "input_size": [512, 288]}}
    cache_path = tmp_path / "cache.json"
    f1 = _frame(0)

    inner1 = _Counting(answer=(1.0, 2.0, 0.7))
    fp1 = build_detector_fingerprint(cfg, inner1)
    det1 = CachingBallDetector(inner1, cache_path, fingerprint=fp1)
    assert det1.detect(f1) == (1.0, 2.0, 0.7)
    det1.save()

    # "Regenerate" the checkpoint (finetune produced a new file) and
    # rebuild the detector — a differently-answering inner this time.
    ckpt.write_bytes(b"gen-2-finetuned")
    inner2 = _Counting(answer=(9.0, 9.0, 0.99))
    fp2 = build_detector_fingerprint(cfg, inner2)
    det2 = CachingBallDetector(inner2, cache_path, fingerprint=fp2)
    assert det2.detect(f1) == (9.0, 9.0, 0.99)  # not the stale gen-1 answer
    assert inner2.calls == 1


@pytest.mark.unit
def test_wrap_if_enabled_false_leaves_detector_unwrapped(tmp_path: Path):
    inner = _Counting()
    cfg = {"detection_cache": {"enabled": False}}
    out = wrap_if_enabled(inner, cfg, tmp_path)
    assert out is inner


@pytest.mark.unit
def test_wrap_if_enabled_default_is_false(tmp_path: Path):
    """No detection_cache key at all -> unwrapped (default off, matches
    first-run behaviour before this cache existed)."""
    inner = _Counting()
    out = wrap_if_enabled(inner, {}, tmp_path)
    assert out is inner


@pytest.mark.unit
def test_wrap_if_enabled_true_wraps_and_uses_output_dir_relative_path(tmp_path: Path):
    inner = _Counting()
    cfg = {"detection_cache": {"enabled": True, "path": "ball/detection_cache.json"}}
    out = wrap_if_enabled(inner, cfg, tmp_path)
    assert isinstance(out, CachingBallDetector)
    assert out._path == tmp_path / "ball" / "detection_cache.json"

    f1 = _frame(0)
    out.detect(f1)
    out.save()
    assert (tmp_path / "ball" / "detection_cache.json").exists()


@pytest.mark.unit
def test_wrap_if_enabled_default_path_under_output_dir(tmp_path: Path):
    inner = _Counting()
    cfg = {"detection_cache": {"enabled": True}}
    out = wrap_if_enabled(inner, cfg, tmp_path)
    assert isinstance(out, CachingBallDetector)
    assert out._path == tmp_path / "ball" / "detection_cache.json"


@pytest.mark.unit
def test_wrap_if_enabled_absolute_path_not_rejoined(tmp_path: Path):
    inner = _Counting()
    abs_path = tmp_path / "elsewhere" / "cache.json"
    cfg = {"detection_cache": {"enabled": True, "path": str(abs_path)}}
    out = wrap_if_enabled(inner, cfg, tmp_path / "output")
    assert out._path == abs_path
