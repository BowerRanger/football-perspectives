"""D4/G17: detection cache default-on semantics — path-independent
fingerprint, legacy caches still hit, fake/no-op detectors never cached."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.utils.ball_detection_cache import (
    CachingBallDetector,
    build_detector_fingerprint,
    wrap_if_enabled,
)
from src.utils.ball_detector import BallDetector


class _Counting(BallDetector):
    SUPPORTS_REDETECT = True

    def __init__(self):
        self.calls = 0

    def detect(self, frame):
        self.calls += 1
        return (10.0, 20.0, 0.9)

    def detect_candidates(self, frame, min_score, top_k=5):
        return []


def _frame(fill: int) -> np.ndarray:
    return np.full((64, 64, 3), fill, dtype=np.uint8)


def _wasb_cfg(path: Path) -> dict:
    return {"detector": "wasb", "wasb": {"checkpoint": str(path)}}


@pytest.mark.unit
def test_fingerprint_ignores_checkpoint_path_for_same_weights(tmp_path: Path):
    a = tmp_path / "a.pth.tar"
    a.write_bytes(b"weights")
    b = tmp_path / "sub" / "b.pth.tar"
    b.parent.mkdir()
    b.write_bytes(b"weights")
    inner = _Counting()
    fa = build_detector_fingerprint(_wasb_cfg(a), inner)
    fb = build_detector_fingerprint(_wasb_cfg(b), inner)
    assert "checkpoint_path" not in fa
    assert fa == fb


@pytest.mark.unit
def test_different_weights_still_invalidate(tmp_path: Path):
    a = tmp_path / "a.pth.tar"
    a.write_bytes(b"weights-1")
    b = tmp_path / "b.pth.tar"
    b.write_bytes(b"weights-2")
    inner = _Counting()
    assert (build_detector_fingerprint(_wasb_cfg(a), inner)
            != build_detector_fingerprint(_wasb_cfg(b), inner))


@pytest.mark.unit
def test_legacy_cache_with_checkpoint_path_still_hits(tmp_path: Path):
    ckpt = tmp_path / "w.pth.tar"
    ckpt.write_bytes(b"weights")
    fp_new = build_detector_fingerprint(_wasb_cfg(ckpt), _Counting())
    cache = tmp_path / "det.json"
    det = CachingBallDetector(_Counting(), cache, fingerprint=fp_new)
    f1 = _frame(3)
    det.detect(f1)
    det.save()
    data = json.loads(cache.read_text())
    data["fingerprint"]["checkpoint_path"] = "/somewhere/else/w.pth.tar"  # legacy shape
    cache.write_text(json.dumps(data))

    inner2 = _Counting()
    det2 = CachingBallDetector(inner2, cache, fingerprint=fp_new)
    assert det2.detect(f1) == (10.0, 20.0, 0.9)
    assert inner2.calls == 0


@pytest.mark.unit
def test_wrap_skips_non_real_detector_unless_forced(tmp_path: Path):
    inner = _Counting()
    cfg = {"detection_cache": {"enabled": True}}
    assert wrap_if_enabled(inner, cfg, tmp_path) is inner
    forced = {"detection_cache": {"enabled": True, "force": True}}
    assert isinstance(wrap_if_enabled(inner, forced, tmp_path), CachingBallDetector)


@pytest.mark.unit
def test_noop_detector_never_touches_cache_file(tmp_path: Path):
    inner = _Counting()
    out = wrap_if_enabled(inner, {"detection_cache": {"enabled": True}}, tmp_path)
    out.detect(_frame(1))
    assert not (tmp_path / "ball" / "detection_cache.json").exists()


@pytest.mark.unit
def test_default_yaml_ships_cache_enabled():
    root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load((root / "config" / "default.yaml").read_text())
    assert cfg["ball"]["detection_cache"]["enabled"] is True
