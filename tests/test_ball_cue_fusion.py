"""Tests for the fusion policy: no single cue mints an event alone,
>= 2 distinct cues within the frame tolerance do, and a single cue near
an existing auto/kinematic anchor does too."""

from __future__ import annotations

from src.utils.ball_cue_fusion import fuse_cues
from src.utils.ball_cue_types import CueEvidence


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
