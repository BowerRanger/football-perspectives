"""Tests for the T5 cross-replay review-proposal wiring in
``BallStage._triangulate_groups`` (orchestrator note: IC-C's
``ball_replay_review.build_replay_review_proposals``/
``summarize_replay_reviews`` wired into the empty-pair-fixes branch, the
geometry-rejected branch, symmetric B-side recording, and the
``productive`` filter fix so a zero-fix-but-not-rejected partner is no
longer mistaken for a working pairing).

Reuses tests/test_ball_stage_cross_replay.py's synthetic 2-camera/2-shot
fixture builders so the stage-level smoke test runs through the REAL
``BallStage._triangulate_groups``, not a mock.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.schemas.shots import Shot, ShotsManifest
from src.stages.ball import BallStage, _record_nonproductive_b_side
from src.utils.ball_detector import FakeBallDetector

from .test_ball_stage_cross_replay import (
    FPS,
    N_A,
    N_B,
    OFFSET,
    _camera_pose_a,
    _camera_pose_b,
    _make_detections_a,
    _make_detections_b,
    _save_camera_track,
    _write_blank_clip,
    _write_sync_map_v1,
)


# ---------------------------------------------------------------------------
# Unit: _record_nonproductive_b_side
# ---------------------------------------------------------------------------

def test_record_nonproductive_b_side_builds_review_proposal():
    summaries: dict[str, dict] = {}
    meta = {"saved_offset": 5.0, "refined_offset": 5.1, "n_pairs": 3, "n_fixes": 0}
    _record_nonproductive_b_side(summaries, "shotB", "shotA", meta)
    assert "shotB" in summaries
    entry = summaries["shotB"]
    assert entry["n_inlier_fixes"] == 0
    assert entry["partner_shots"] == ["shotA"]
    assert entry["partners"] == {"shotA": meta}
    assert len(entry["review"]) == 1
    assert entry["review"][0]["partner"] == "shotA"
    assert entry["review"][0]["reason"] == "no_inlier_fixes"


def test_record_nonproductive_b_side_never_clobbers_an_existing_entry():
    summaries = {"shotB": {"n_inlier_fixes": 12, "partners": {}, "partner_shots": []}}
    _record_nonproductive_b_side(
        summaries, "shotB", "shotC", {"saved_offset": 0.0, "refined_offset": 0.0})
    assert summaries["shotB"]["n_inlier_fixes"] == 12  # untouched


# ---------------------------------------------------------------------------
# Stage-level smoke: every pairing in the group produces zero fixes ->
# both shots get a review proposal naming the other as the partner.
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_triangulate_groups_review_proposal_when_pairing_yields_no_fixes(
    tmp_path: Path, monkeypatch,
):
    Ka, Ra, ta = _camera_pose_a()
    Kb, Rb, tb = _camera_pose_b()

    _save_camera_track(tmp_path / "camera" / "shotA_camera_track.json",
                        Ka, Ra, ta, N_A, clip_id="shotA")
    _save_camera_track(tmp_path / "camera" / "shotB_camera_track.json",
                        Kb, Rb, tb, N_B, clip_id="shotB")
    _write_blank_clip(tmp_path / "shots" / "shotA.mp4", N_A)
    _write_blank_clip(tmp_path / "shots" / "shotB.mp4", N_B)

    manifest = ShotsManifest(
        source_file="synthetic", fps=FPS, total_frames=N_A + N_B,
        shots=[
            Shot(id="shotA", start_frame=0, end_frame=N_A - 1, start_time=0.0,
                 end_time=(N_A - 1) / FPS, clip_file="shots/shotA.mp4", group_id="g1"),
            Shot(id="shotB", start_frame=0, end_frame=N_B - 1, start_time=0.0,
                 end_time=(N_B - 1) / FPS, clip_file="shots/shotB.mp4", group_id="g1"),
        ],
    )
    manifest.save(tmp_path / "shots" / "shots_manifest.json")
    _write_sync_map_v1(tmp_path / "shots" / "sync_map.json")

    dets_a = _make_detections_a(Ka, Ra, ta)
    dets_b = _make_detections_b(Kb, Rb, tb)

    # Force triangulate_pair to report zero pair fixes for every pairing
    # (geometry/offset refinement still runs normally) -- deterministically
    # exercises the "geometry passed but nothing survived" branch without
    # depending on parallax/physical-fix gating specifics.
    import src.stages.ball as ball_module
    monkeypatch.setattr(ball_module, "triangulate_pair", lambda **kwargs: [])

    stage = BallStage(
        config={"ball": {
            "detector": "fake",
            "appearance_bridge": {"enabled": False},
            "second_pass": {"enabled": False},
        }},
        output_dir=tmp_path,
        ball_detector=FakeBallDetector(dets_a + dets_b),
    )
    stage.run()

    ball_dir = tmp_path / "ball"
    for shot_id, partner in (("shotA", "shotB"), ("shotB", "shotA")):
        diag = json.loads((ball_dir / f"{shot_id}_ball_diag.json").read_text())
        cr = diag.get("cross_replay")
        assert cr is not None, f"{shot_id}: diag cross_replay is null"
        assert cr["n_inlier_fixes"] == 0
        review = cr.get("review")
        assert review, f"{shot_id}: expected a review proposal, got {review!r}"
        assert any(p["partner"] == partner for p in review)
        assert not (ball_dir / f"{shot_id}_ball_fixes.json").exists() or (
            len(json.loads((ball_dir / f"{shot_id}_ball_fixes.json").read_text())
                .get("fixes", [])) == 0
        )
