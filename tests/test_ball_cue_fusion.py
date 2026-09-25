"""Tests for the fusion policies: no single unsupported cue mints an
event alone (any2), the net_blur_combo policy's specific-pairing gate,
the weighted policy's threshold behaviour, and the cfg dispatcher."""

from __future__ import annotations

import pytest

from src.utils.ball_cue_config import CueCfg
from src.utils.ball_cue_fusion import (
    CueReliability,
    fuse_cues,
    fuse_cues_combo,
    fuse_cues_weighted,
    fuse_cues_with_cfg,
)
from src.utils.ball_hybrid_types import CueEvidence


def _ce(frame, kind, cue, conf, uv=None, xyz=None):
    return CueEvidence(frame=frame, kind=kind, cue=cue, conf=conf, xyz=xyz, uv=uv)


def test_two_independent_cues_mint_event():
    evidence = {
        "audio": [_ce(100, "contact", "audio_onset", 0.6)],
        "blur": [_ce(101, "contact", "blur_direction_change", 0.5)],
    }
    fused = fuse_cues(evidence, auto_event_frames=[])
    assert len(fused) == 1
    assert fused[0].support == "cues"
    assert set(fused[0].cues) == {"audio_onset", "blur_direction_change"}
    assert fused[0].frame in (100, 101, 100.5)


def test_single_cue_without_auto_support_is_dropped():
    evidence = {"audio": [_ce(50, "contact", "audio_onset", 0.9)]}
    assert fuse_cues(evidence, auto_event_frames=[]) == []


def test_single_cue_with_nearby_auto_event_mints_event():
    evidence = {"audio": [_ce(50, "contact", "audio_onset", 0.9)]}
    fused = fuse_cues(evidence, auto_event_frames=[51])
    assert len(fused) == 1
    assert fused[0].support == "cue+auto"
    assert fused[0].cues == ("audio_onset",)


def test_single_cue_far_from_auto_event_is_dropped():
    evidence = {"audio": [_ce(50, "contact", "audio_onset", 0.9)]}
    assert fuse_cues(evidence, auto_event_frames=[80]) == []


def test_net_cue_kind_wins_goal_impact_label():
    evidence = {
        "audio": [_ce(200, "contact", "audio_onset", 0.5)],
        "net": [_ce(201, "goal_impact", "net_energy", 0.7,
                     xyz=(0.0, 34.0, 1.0), uv=(500.0, 300.0))],
    }
    fused = fuse_cues(evidence, auto_event_frames=[])
    assert len(fused) == 1
    assert fused[0].kind == "goal_impact"
    assert fused[0].xyz == (0.0, 34.0, 1.0)
    assert fused[0].uv == (500.0, 300.0)


def test_clusters_split_beyond_frame_tol_and_both_dropped():
    evidence = {
        "audio": [_ce(10, "contact", "audio_onset", 0.5)],
        "blur": [_ce(50, "contact", "blur_direction_change", 0.5)],
    }
    assert fuse_cues(evidence, auto_event_frames=[]) == []


def test_fused_conf_rises_with_more_corroborating_cues():
    evidence_one = {
        "audio": [_ce(10, "contact", "audio_onset", 0.5)],
        "blur": [_ce(11, "contact", "blur_direction_change", 0.5)],
    }
    evidence_two = {
        "audio": [_ce(10, "contact", "audio_onset", 0.5)],
        "blur": [_ce(11, "contact", "blur_direction_change", 0.5)],
        "net": [_ce(10, "goal_impact", "net_energy", 0.5)],
    }
    conf_two_cues = fuse_cues(evidence_one, auto_event_frames=[])[0].conf
    conf_three_cues = fuse_cues(evidence_two, auto_event_frames=[])[0].conf
    assert conf_three_cues > conf_two_cues


def test_empty_evidence_returns_empty():
    assert fuse_cues({}, auto_event_frames=[1, 2, 3]) == []


# -- fuse_cues_combo ("net_blur_combo") --

def test_combo_accepts_net_plus_blur():
    evidence = {
        "net": [_ce(10, "goal_impact", "net_energy", 0.6)],
        "blur": [_ce(11, "contact", "blur_direction_change", 0.5)],
    }
    fused = fuse_cues_combo(evidence, auto_event_frames=[])
    assert len(fused) == 1
    assert fused[0].support == "cues"


def test_combo_accepts_audio_plus_blur():
    evidence = {
        "audio": [_ce(10, "contact", "audio_onset", 0.6)],
        "blur": [_ce(11, "contact", "blur_direction_change", 0.5)],
    }
    fused = fuse_cues_combo(evidence, auto_event_frames=[])
    assert len(fused) == 1


def test_combo_accepts_blur_plus_auto():
    evidence = {"blur": [_ce(50, "contact", "blur_direction_change", 0.6)]}
    fused = fuse_cues_combo(evidence, auto_event_frames=[51])
    assert len(fused) == 1
    assert fused[0].support == "cue+auto"


def test_combo_rejects_audio_plus_net_without_blur():
    # 2 distinct cues, but blur isn't one of them -- net_blur_combo
    # requires blur in every accepted cluster, unlike any2.
    evidence = {
        "audio": [_ce(10, "contact", "audio_onset", 0.6)],
        "net": [_ce(11, "goal_impact", "net_energy", 0.6)],
    }
    assert fuse_cues_combo(evidence, auto_event_frames=[]) == []


def test_combo_rejects_net_alone_near_auto():
    evidence = {"net": [_ce(50, "goal_impact", "net_energy", 0.6)]}
    assert fuse_cues_combo(evidence, auto_event_frames=[51]) == []


# -- fuse_cues_weighted --

def test_weighted_mints_when_score_clears_threshold():
    reliability = CueReliability(
        weights={"audio_onset": 0.3, "blur_direction_change": 0.3},
        auto_weight=1.0, threshold=0.5)
    evidence = {
        "audio": [_ce(10, "contact", "audio_onset", 0.9)],
        "blur": [_ce(11, "contact", "blur_direction_change", 0.9)],
    }
    fused = fuse_cues_weighted(evidence, auto_event_frames=[], reliability=reliability)
    assert len(fused) == 1
    assert fused[0].conf == pytest.approx(0.6)


def test_weighted_drops_below_threshold():
    reliability = CueReliability(
        weights={"audio_onset": 0.1}, auto_weight=0.2, threshold=0.5)
    evidence = {"audio": [_ce(10, "contact", "audio_onset", 0.9)]}
    assert fuse_cues_weighted(evidence, auto_event_frames=[], reliability=reliability) == []


def test_weighted_auto_bonus_can_clear_threshold_alone():
    reliability = CueReliability(
        weights={"audio_onset": 0.1}, auto_weight=1.0, threshold=1.0)
    evidence = {"audio": [_ce(50, "contact", "audio_onset", 0.9)]}
    fused = fuse_cues_weighted(evidence, auto_event_frames=[51], reliability=reliability)
    assert len(fused) == 1
    assert fused[0].conf == pytest.approx(1.1)


# -- fuse_cues_with_cfg dispatch --

def test_dispatch_any2():
    cfg = CueCfg(fusion_policy="any2")
    evidence = {
        "audio": [_ce(10, "contact", "audio_onset", 0.5)],
        "net": [_ce(11, "goal_impact", "net_energy", 0.5)],
    }
    fused = fuse_cues_with_cfg(evidence, auto_event_frames=[], cfg=cfg)
    assert len(fused) == 1  # any2 allows audio+net, unlike combo


def test_dispatch_net_blur_combo():
    cfg = CueCfg(fusion_policy="net_blur_combo")
    evidence = {
        "audio": [_ce(10, "contact", "audio_onset", 0.5)],
        "net": [_ce(11, "goal_impact", "net_energy", 0.5)],
    }
    fused = fuse_cues_with_cfg(evidence, auto_event_frames=[], cfg=cfg)
    assert fused == []  # no blur present


def test_dispatch_weighted():
    cfg = CueCfg(
        fusion_policy="weighted",
        fusion_weights={"audio_onset": 0.6, "blur_direction_change": 0.6},
        fusion_auto_weight=1.0, fusion_threshold=1.0,
    )
    evidence = {
        "audio": [_ce(10, "contact", "audio_onset", 0.9)],
        "blur": [_ce(11, "contact", "blur_direction_change", 0.9)],
    }
    fused = fuse_cues_with_cfg(evidence, auto_event_frames=[], cfg=cfg)
    assert len(fused) == 1


def test_dispatch_unknown_policy_raises():
    cfg = CueCfg(fusion_policy="not_a_real_policy")
    with pytest.raises(ValueError):
        fuse_cues_with_cfg({}, auto_event_frames=[], cfg=cfg)
