"""Tests for scripts/eval_ball_accuracy.py — overlay runner + CLI."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


@pytest.mark.unit
def test_build_overlay_symlinks_inputs_and_filters_anchors(tmp_path):
    from eval_ball_accuracy import build_overlay

    from src.schemas.ball_anchor import BallAnchor, BallAnchorSet

    src = tmp_path / "out"
    for d in ("camera", "refined_poses", "shots", "tracks", "ball",
              "ball_pre_x", "logs", "renders"):
        (src / d).mkdir(parents=True)
    (src / "camera" / "s1_camera_track.json").write_text("{}")
    full = BallAnchorSet(
        clip_id="s1", image_size=(1920, 1080),
        anchors=(BallAnchor(frame=1, image_xy=(5.0, 5.0), state="grounded"),))
    full.save(src / "ball" / "s1_ball_anchors.json")

    kept = BallAnchorSet(clip_id="s1", image_size=(1920, 1080), anchors=())
    ov = build_overlay(src, tmp_path / "work", "s1", kept)
    assert (ov / "camera").is_symlink()
    assert (ov / "shots").is_symlink()
    assert not (ov / "ball").is_symlink() and (ov / "ball").is_dir()
    assert not (ov / "ball_pre_x").exists()
    assert not (ov / "renders").exists()
    # logs must be a real dir so stage logging never writes into the source.
    assert (ov / "logs").is_dir() and not (ov / "logs").is_symlink()
    saved = BallAnchorSet.load(ov / "ball" / "s1_ball_anchors.json")
    assert len(saved.anchors) == 0


@pytest.mark.unit
def test_build_overlay_copies_original_anchors_when_kept_is_none(tmp_path):
    from eval_ball_accuracy import build_overlay

    from src.schemas.ball_anchor import BallAnchor, BallAnchorSet

    src = tmp_path / "out"
    (src / "ball").mkdir(parents=True)
    (src / "camera").mkdir()
    full = BallAnchorSet(
        clip_id="s1", image_size=(1920, 1080),
        anchors=(BallAnchor(frame=2, image_xy=(9.0, 9.0), state="grounded"),))
    full.save(src / "ball" / "s1_ball_anchors.json")
    ov = build_overlay(src, tmp_path / "work", "s1", None)
    saved = BallAnchorSet.load(ov / "ball" / "s1_ball_anchors.json")
    assert [a.frame for a in saved.anchors] == [2]


@pytest.mark.integration
def test_run_and_evaluate_gberch_noop_holdout():
    import yaml

    from eval_ball_accuracy import run_and_evaluate

    out = ROOT / "output"
    if not (out / "ball" / "gberch_ball_anchors.json").exists():
        pytest.skip("gberch output not present")
    cfg = yaml.safe_load(open(ROOT / "config" / "default.yaml"))
    rep = run_and_evaluate(out, "gberch", detector="noop", holdout=True,
                           n_folds=2, config=cfg)
    assert rep["summary"]["anchors_held_out"]["n"] > 0
    assert rep["clip"] == "gberch" and rep["detector"] == "noop"
    json.dumps(rep)
