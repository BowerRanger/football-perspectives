"""Recall/precision of auto-detected touches vs a manual anchor set
(pseudo-ground-truth). Pure, torch-free."""

from __future__ import annotations

import json
from pathlib import Path

from src.utils.ball_touch_recall import match_touches, touches_from_anchor_set


def test_recall_matches_by_frame_and_bone():
    manual = [(100, "P1", "r_foot"), (200, "P2", "head"), (300, "P1", "l_foot")]
    auto = [(101, "P1", "r_foot"), (260, "P2", "head")]
    r = match_touches(manual, auto, frame_tol=2)
    assert r["n_manual"] == 3 and r["n_auto"] == 2
    assert r["true_positive"] == 1
    assert r["false_positive"] == 1
    assert abs(r["recall"] - 1 / 3) < 1e-9
    assert abs(r["precision"] - 1 / 2) < 1e-9


def test_bone_mismatch_is_not_a_match_when_required():
    manual = [(100, "P1", "r_foot")]
    auto = [(100, "P1", "l_foot")]
    assert match_touches(manual, auto, frame_tol=2, require_bone=True)["true_positive"] == 0
    assert match_touches(manual, auto, frame_tol=2, require_bone=False)["true_positive"] == 1


def test_one_auto_cannot_claim_two_manuals():
    manual = [(100, "P1", "r_foot"), (101, "P1", "r_foot")]
    auto = [(100, "P1", "r_foot")]
    r = match_touches(manual, auto, frame_tol=2)
    assert r["true_positive"] == 1  # greedy 1:1, not 2


def test_empty_sets():
    r = match_touches([], [], frame_tol=2)
    assert r["recall"] == 0.0 and r["precision"] == 0.0 and r["n_manual"] == 0


def test_touches_from_anchor_set(tmp_path: Path):
    p = tmp_path / "a.json"
    p.write_text(json.dumps({
        "clip_id": "c", "image_size": [100, 100],
        "anchors": [
            {"frame": 5, "image_xy": [1, 2], "state": "player_touch",
             "player_id": "P1", "bone": "r_foot"},
            {"frame": 9, "image_xy": [3, 4], "state": "grounded"},
        ],
    }))
    assert touches_from_anchor_set(p) == [(5, "P1", "r_foot")]


def test_fp_breakdown_partitions_dismissed_and_unreviewed():
    from src.utils.ball_touch_recall import fp_breakdown

    manual = [(10, "P1", "r_foot")]
    auto = [(10, "P1", "r_foot"),   # TP
            (30, "P2", "l_foot"),   # FP, dismissed
            (50, "P3", "head")]     # FP, unreviewed
    dismissed = [(30, "P2", "l_foot")]
    out = fp_breakdown(auto, manual, dismissed, frame_tol=2)
    assert out == {"fp_total": 2, "fp_dismissed": 1, "fp_unreviewed": 1}


def test_dismissed_touches_loader(tmp_path):
    import json
    from src.utils.ball_touch_recall import dismissed_touches_from_anchor_set

    p = tmp_path / "a.json"
    p.write_text(json.dumps({
        "clip_id": "x", "image_size": [1280, 720], "anchors": [],
        "dismissed_auto": [
            {"frame": 30, "state": "player_touch",
             "player_id": "P2", "bone": "l_foot"},
            {"frame": 40, "state": "bounce"},
        ],
    }))
    assert dismissed_touches_from_anchor_set(p) == [(30, "P2", "l_foot")]
