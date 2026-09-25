"""Tests for src/utils/ball_replay_review.py — review proposals for
cross-replay groups whose partner geometry left a shot with no usable
fixes (the s013 g02 "6/6 partner cameras globally wrong" scenario, W6)."""

from __future__ import annotations

from src.utils.ball_replay_review import (
    build_replay_review_proposals,
    summarize_replay_reviews,
)


def test_geometry_rejected_partner_produces_proposal():
    partners_meta = {
        "s013b": {
            "saved_offset": 3.0, "refined_offset": 3.25,
            "rejected": "implausible_geometry",
            "n_impossible": 41, "n_fixes": 47,
        },
    }
    proposals = build_replay_review_proposals("s013", partners_meta)

    assert len(proposals) == 1
    p = proposals[0]
    assert p == {
        "shot": "s013",
        "partner": "s013b",
        "reason": "implausible_geometry",
        "n_gated": 41,
        "suggestion": "review partner camera anchors in the anchor editor",
    }


def test_silently_empty_partner_produces_no_inlier_fixes_proposal():
    """A pair whose geometry passed pair_geometry_trusted but whose
    fixes were individually filtered to zero (no 'rejected' key) still
    gets a proposal — the group had a replay partner and ended up with
    nothing usable from it."""
    partners_meta = {"partnerB": {"saved_offset": 1.0, "refined_offset": 1.0,
                                   "n_pairs": 12, "n_fixes": 0}}
    proposals = build_replay_review_proposals("shotA", partners_meta)

    assert len(proposals) == 1
    assert proposals[0]["reason"] == "no_inlier_fixes"
    assert proposals[0]["n_gated"] == 0


def test_multi_partner_group_all_gated_out_s013_g02_shape():
    """Mirrors the documented W6 finding: 6 replay partners, every one
    rejected on implausible geometry -> 6 proposals, all reviewable."""
    partners_meta = {
        f"s013_replay{i}": {
            "rejected": "implausible_geometry",
            "n_impossible": 10 + i, "n_fixes": 12 + i,
        }
        for i in range(6)
    }
    proposals = build_replay_review_proposals("s013", partners_meta)

    assert len(proposals) == 6
    assert {p["partner"] for p in proposals} == set(partners_meta)
    assert all(p["reason"] == "implausible_geometry" for p in proposals)
    assert all(p["shot"] == "s013" for p in proposals)


def test_proposals_sorted_by_partner_for_determinism():
    partners_meta = {
        "z_partner": {"rejected": "implausible_geometry", "n_fixes": 1},
        "a_partner": {"rejected": "implausible_geometry", "n_fixes": 1},
    }
    proposals = build_replay_review_proposals("shot", partners_meta)
    assert [p["partner"] for p in proposals] == ["a_partner", "z_partner"]


def test_summarize_replay_reviews_drops_empty_shots_and_counts():
    by_shot = {
        "s013": build_replay_review_proposals("s013", {
            "p1": {"rejected": "implausible_geometry", "n_fixes": 1},
            "p2": {"rejected": "implausible_geometry", "n_fixes": 1},
        }),
        "origi01": [],  # productive group — nothing to review
    }
    summary = summarize_replay_reviews(by_shot)
    assert summary["n_proposals"] == 2
    assert set(summary["shots"]) == {"s013"}
    assert "origi01" not in summary["shots"]


def test_empty_partners_meta_produces_no_proposals():
    assert build_replay_review_proposals("shot", {}) == []
