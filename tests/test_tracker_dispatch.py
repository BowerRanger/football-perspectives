"""Tests for the tracker dispatch helper in src/stages/tracking.

Verifies the config switch between ByteTrack (motion-only, default for
the dev test config) and BoT-SORT (BoxMOT with OSNet ReID — used in
production for ID stability through occlusions).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.stages.tracking import (
    _BotSortAdapter,
    _ByteTrackAdapter,
    _build_tracker,
)


def test_bytetrack_dispatch_returns_bytetrack_adapter():
    cfg = {"tracking": {"tracker": "bytetrack"}}
    tracker = _build_tracker(cfg)
    assert isinstance(tracker, _ByteTrackAdapter)
    assert tracker.name == "bytetrack"


def test_default_tracker_is_botsort():
    """When the config omits `tracker`, the dispatch helper defaults to
    botsort — matches config/default.yaml so the production path is the
    one exercised by default."""
    # The actual construction needs boxmot + weights; here we just check
    # the dispatch decision by intercepting before BotSort is built.
    cfg = {"tracking": {"reid_weights": "/nonexistent/path.pt"}}
    with pytest.raises((FileNotFoundError, ImportError)):
        _build_tracker(cfg)


def test_unknown_tracker_raises():
    cfg = {"tracking": {"tracker": "deepsort"}}
    with pytest.raises(ValueError, match="Unknown tracker"):
        _build_tracker(cfg)


def test_botsort_missing_weights_raises_filenotfound():
    """The botsort branch errors clearly when the OSNet checkpoint is
    absent — points the operator at the setup script instead of leaving
    them with a cryptic boxmot import error."""
    boxmot = pytest.importorskip("boxmot")  # noqa: F841
    cfg = {
        "tracking": {
            "tracker": "botsort",
            "reid_weights": "third_party/boxmot/__missing__.pt",
            "tracker_device": "cpu",
        }
    }
    with pytest.raises(FileNotFoundError, match="setup_boxmot.sh"):
        _build_tracker(cfg)


def test_botsort_construction_with_real_weights():
    """End-to-end: when boxmot is installed AND the OSNet checkpoint is
    present, the dispatch returns a usable _BotSortAdapter. Skipped
    otherwise so the test passes in dev environments without GPU/ReID
    weights."""
    pytest.importorskip("boxmot")
    repo_root = Path(__file__).resolve().parents[1]
    weights = repo_root / "third_party" / "boxmot" / "osnet_x0_25_msmt17.pt"
    if not weights.exists():
        pytest.skip(f"ReID weights not present at {weights}")
    cfg = {
        "tracking": {
            "tracker": "botsort",
            "reid_weights": str(weights),
            "tracker_device": "cpu",
        }
    }
    tracker = _build_tracker(cfg)
    assert isinstance(tracker, _BotSortAdapter)
    assert tracker.name == "botsort"
