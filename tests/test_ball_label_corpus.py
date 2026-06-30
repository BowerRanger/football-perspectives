"""Auto-accumulating ball-label corpus: every anchor save appends gold
labels to a growing WASB-format training set."""

from __future__ import annotations

from pathlib import Path

from src.schemas.ball_anchor import BallAnchor
from src.utils.ball_label_corpus import load_manifest, record_labels


def _a(frame, xy, state="grounded", **kw):
    return BallAnchor(frame=frame, image_xy=xy, state=state, **kw)


def test_record_writes_anno_and_manifest(tmp_path: Path):
    anchors = (_a(5, (100.0, 200.0)), _a(8, None, "off_screen_flight"))
    entry = record_labels(tmp_path, "gberch", "shots/gberch.mp4", anchors)
    assert (tmp_path / "annos" / "gberch.xml").exists()
    assert entry["n_labels"] == 1  # only the one with image_xy
    m = load_manifest(tmp_path)
    assert m["clips"]["gberch"]["clip_path"] == "shots/gberch.mp4"
    assert m["clips"]["gberch"]["n_labels"] == 1
    assert m["clips"]["gberch"]["saves"] == 1


def test_resave_increments_saves_and_updates_count(tmp_path: Path):
    record_labels(tmp_path, "g", "shots/g.mp4", (_a(1, (1.0, 2.0)),))
    record_labels(tmp_path, "g", "shots/g.mp4",
                  (_a(1, (1.0, 2.0)), _a(2, (3.0, 4.0))))
    m = load_manifest(tmp_path)
    assert m["clips"]["g"]["saves"] == 2
    assert m["clips"]["g"]["n_labels"] == 2


def test_skips_when_no_labelled_anchors(tmp_path: Path):
    entry = record_labels(tmp_path, "g", "shots/g.mp4",
                          (_a(1, None, "off_screen_flight"),))
    assert entry is None
    assert load_manifest(tmp_path)["clips"] == {}


def test_accumulates_multiple_clips(tmp_path: Path):
    record_labels(tmp_path, "a", "shots/a.mp4", (_a(1, (1.0, 2.0)),))
    record_labels(tmp_path, "b", "shots/b.mp4", (_a(1, (1.0, 2.0)),))
    assert set(load_manifest(tmp_path)["clips"]) == {"a", "b"}


def test_load_manifest_empty_when_absent(tmp_path: Path):
    assert load_manifest(tmp_path) == {"clips": {}}
