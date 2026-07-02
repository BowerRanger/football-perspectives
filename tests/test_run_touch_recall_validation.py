"""Pure helpers of the two-config touch-recall validation runner."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.run_touch_recall_validation import (
    snapshot_auto_anchors,
    with_kinematic_toggle,
)


def test_toggle_sets_flag_without_mutating_input():
    cfg = {"ball": {"kinematic_touch": {"enabled": True, "kin_window": 2}}}
    off = with_kinematic_toggle(cfg, False)
    assert off["ball"]["kinematic_touch"]["enabled"] is False
    assert off["ball"]["kinematic_touch"]["kin_window"] == 2
    # input untouched (immutability)
    assert cfg["ball"]["kinematic_touch"]["enabled"] is True


def test_toggle_creates_missing_subtrees():
    on = with_kinematic_toggle({}, True)
    assert on["ball"]["kinematic_touch"]["enabled"] is True


def test_snapshot_copies_auto_sidecar(tmp_path: Path):
    ball_dir = tmp_path / "ball"
    ball_dir.mkdir()
    payload = {"clip_id": "play", "image_size": [1280, 720], "anchors": []}
    (ball_dir / "play_ball_anchors_auto.json").write_text(json.dumps(payload))
    dst = snapshot_auto_anchors(ball_dir, "play", "break_only")
    assert dst == ball_dir / "play_ball_anchors_auto_break_only.json"
    assert json.loads(dst.read_text()) == payload


def test_snapshot_missing_sidecar_raises(tmp_path: Path):
    ball_dir = tmp_path / "ball"
    ball_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        snapshot_auto_anchors(ball_dir, "play", "union")
