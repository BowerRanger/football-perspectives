"""Unit test: quality_report aggregator builds the expected JSON shape."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.pipeline.quality_report import write_quality_report
from src.schemas.anchor import Anchor, AnchorSet, LandmarkObservation
from src.schemas.ball_track import BallFrame, BallTrack, FlightSegment
from src.schemas.camera_track import CameraFrame, CameraTrack
from src.schemas.shots import KitColors, MatchInfo, MomentInfo, ShotsManifest
from src.schemas.smpl_world import SmplWorldTrack


@pytest.mark.unit
def test_quality_report_aggregates_three_stages(tmp_path: Path) -> None:
    AnchorSet(
        clip_id="play",
        image_size=(1280, 720),
        anchors=(
            Anchor(
                frame=0,
                landmarks=(
                    LandmarkObservation(name="x", image_xy=(0.0, 0.0), world_xyz=(0.0, 0.0, 0.0)),
                ),
            ),
            Anchor(
                frame=10,
                landmarks=(
                    LandmarkObservation(name="y", image_xy=(0.0, 0.0), world_xyz=(1.0, 0.0, 0.0)),
                ),
            ),
        ),
    ).save(tmp_path / "camera" / "anchors.json")

    eye = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    CameraTrack(
        clip_id="play",
        fps=30.0,
        image_size=(1280, 720),
        t_world=[0.0, 0.0, 0.0],
        frames=tuple(
            CameraFrame(
                frame=i,
                K=eye,
                R=eye,
                confidence=(0.9 if i < 8 else 0.3),
                is_anchor=(i in (0, 10)),
            )
            for i in range(11)
        ),
    ).save(tmp_path / "camera" / "camera_track.json")

    SmplWorldTrack(
        player_id="P001",
        frames=np.arange(11),
        betas=np.zeros(10),
        thetas=np.zeros((11, 24, 3)),
        root_R=np.tile(np.eye(3), (11, 1, 1)),
        root_t=np.zeros((11, 3)),
        confidence=np.full(11, 0.8),
    ).save(tmp_path / "hmr_world" / "P001_smpl_world.npz")

    BallTrack(
        clip_id="play",
        fps=30.0,
        frames=tuple(
            BallFrame(
                frame=i,
                world_xyz=(0.0, 0.0, 0.11),
                state="grounded",
                confidence=0.9,
            )
            for i in range(11)
        ),
        flight_segments=(),
    ).save(tmp_path / "ball" / "ball_track.json")

    write_quality_report(tmp_path)
    report = json.loads((tmp_path / "quality_report.json").read_text())

    # Camera section
    assert report["camera"]["anchor_count"] == 2
    assert report["camera"]["low_confidence_frame_count"] == 3
    assert report["camera"]["low_confidence_frame_ranges"] == [[8, 10]]
    assert "mean_anchor_residual_px" in report["camera"]
    assert isinstance(report["camera"]["mean_anchor_residual_px"], float)
    # body_drift_max_m is reported as None when the track has no
    # camera_centre (moving-camera clip) and as a float otherwise.
    assert "body_drift_max_m" in report["camera"]
    assert report["camera"]["body_drift_max_m"] is None

    # HMR section
    assert report["hmr_world"]["tracked_players"] == 1
    assert report["hmr_world"]["mean_per_player_confidence"] == pytest.approx(0.8)
    assert report["hmr_world"]["low_confidence_players"] == []

    # Ball section
    assert report["ball"]["flight_segments"] == 0
    assert report["ball"]["missing_frames"] == 0
    assert report["ball"]["shots"][0]["grounded_frames"] == 11


@pytest.mark.unit
def test_quality_report_ball_spin_coverage(tmp_path: Path) -> None:
    """spin_coverage_pct = flight frames inside a spin-bearing segment /
    total flight frames."""
    frames: list[BallFrame] = []
    for i in range(20):
        if 5 <= i <= 9:
            seg_id, state = 0, "flight"
        elif 15 <= i <= 19:
            seg_id, state = 1, "flight"
        else:
            seg_id, state = None, "grounded"
        frames.append(
            BallFrame(
                frame=i, world_xyz=(0.0, 0.0, 0.5), state=state,
                confidence=0.9, flight_segment_id=seg_id,
            )
        )
    segs = (
        FlightSegment(
            id=0, frame_range=(5, 9), fit_residual_px=1.0,
            parabola={
                "p0": [0, 0, 0.5], "v0": [1, 0, 5], "g": -9.81,
                "spin_axis_world": None, "spin_omega_rad_s": None,
                "spin_confidence": None,
            },
        ),
        FlightSegment(
            id=1, frame_range=(15, 19), fit_residual_px=1.2,
            parabola={
                "p0": [10, 0, 0.5], "v0": [1, 0, 5], "g": -9.81,
                "spin_axis_world": [0.0, 0.0, 1.0],
                "spin_omega_rad_s": 20.0,
                "spin_confidence": 0.8,
            },
        ),
    )
    BallTrack(
        clip_id="play", fps=30.0, frames=tuple(frames), flight_segments=segs,
    ).save(tmp_path / "ball" / "ball_track.json")

    write_quality_report(tmp_path)
    report = json.loads((tmp_path / "quality_report.json").read_text())
    assert report["ball"]["flight_segments"] == 2
    # 5 of 10 flight frames are inside the spin-bearing segment.
    assert report["ball"]["shots"][0]["spin_coverage_pct"] == pytest.approx(50.0)


@pytest.mark.unit
def test_quality_report_handles_empty_dir(tmp_path: Path) -> None:
    """Missing inputs => sections absent, no exception."""
    write_quality_report(tmp_path)
    report = json.loads((tmp_path / "quality_report.json").read_text())
    assert report == {}


@pytest.mark.unit
def test_quality_report_body_drift_for_static_camera(tmp_path: Path) -> None:
    """When camera_centre is set on the track, body_drift_max_m is the
    worst ||(-R^T @ t) - C|| across frames."""
    AnchorSet(
        clip_id="play",
        image_size=(1280, 720),
        anchors=(
            Anchor(
                frame=0,
                landmarks=(
                    LandmarkObservation(
                        name="x", image_xy=(0.0, 0.0), world_xyz=(0.0, 0.0, 0.0),
                    ),
                ),
            ),
        ),
    ).save(tmp_path / "camera" / "anchors.json")

    eye = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    C = (52.5, -30.0, 30.0)
    # Frame 0: t such that -R^T @ t == C exactly.
    t0 = [-C[0], -C[1], -C[2]]                 # R = I, so t = -C
    # Frame 1: a deliberately drifted t to verify the metric picks it up.
    drift = 0.4
    t1 = [-C[0] + drift, -C[1], -C[2]]         # body shifted by 0.4 m in x
    CameraTrack(
        clip_id="play",
        fps=30.0,
        image_size=(1280, 720),
        t_world=t0,
        frames=(
            CameraFrame(
                frame=0, K=eye, R=eye, confidence=1.0, is_anchor=True, t=t0,
            ),
            CameraFrame(
                frame=1, K=eye, R=eye, confidence=0.7, is_anchor=False, t=t1,
            ),
        ),
        camera_centre=C,
    ).save(tmp_path / "camera" / "camera_track.json")

    write_quality_report(tmp_path)
    report = json.loads((tmp_path / "quality_report.json").read_text())
    assert report["camera"]["body_drift_max_m"] == pytest.approx(drift, abs=1e-6)


@pytest.mark.unit
def test_quality_report_includes_refined_poses_section(tmp_path: Path) -> None:
    refined_dir = tmp_path / "refined_poses"
    refined_dir.mkdir()
    summary = {
        "players_refined": 3,
        "single_shot_players": 1,
        "multi_shot_players": 2,
        "total_fused_frames": 100,
        "single_view_frames": 20,
        "high_disagreement_frames": 4,
        "shots_missing_sync": [],
        "beta_disagreement_warnings": [],
    }
    (refined_dir / "refined_poses_summary.json").write_text(json.dumps(summary))
    write_quality_report(tmp_path)
    report = json.loads((tmp_path / "quality_report.json").read_text())
    assert report["refined_poses"]["players_refined"] == 3
    assert report["refined_poses"]["high_disagreement_frames"] == 4


@pytest.mark.unit
def test_quality_report_lifts_jitter_correction_to_top_level(
    tmp_path: Path,
) -> None:
    """The cross-player jitter pass writes a ``jitter`` block inside
    ``refined_poses_summary.json``. Surface it at top level as
    ``jitter_correction`` so dashboard / CI checks can read it
    alongside camera confidence and HMR coverage without parsing the
    nested refined-poses payload."""
    refined_dir = tmp_path / "refined_poses"
    refined_dir.mkdir()
    summary = {
        "players_refined": 4,
        "single_shot_players": 4,
        "multi_shot_players": 0,
        "total_frames": 120,
        "jitter": {
            "enabled": True,
            "corrected_frames": 7,
            "total_frames_evaluated": 119,
            "max_offset_m": 0.42,
            "mean_offset_m": 0.18,
            "shots": [
                {
                    "shot_id": "play",
                    "corrected_frames": 7,
                    "total_frames_evaluated": 119,
                    "max_offset_m": 0.42,
                    "mean_offset_m": 0.18,
                },
            ],
        },
    }
    (refined_dir / "refined_poses_summary.json").write_text(json.dumps(summary))
    write_quality_report(tmp_path)
    report = json.loads((tmp_path / "quality_report.json").read_text())
    assert report["jitter_correction"]["corrected_frames"] == 7
    assert report["jitter_correction"]["max_offset_m"] == pytest.approx(0.42)
    assert report["jitter_correction"]["fraction_corrected"] == pytest.approx(
        7 / 119, abs=1e-6,
    )


@pytest.mark.unit
def test_quality_report_omits_jitter_correction_when_missing(
    tmp_path: Path,
) -> None:
    """No refined_poses_summary.json → no ``jitter_correction`` block.
    Older outputs from before the jitter pass landed must still
    aggregate cleanly."""
    write_quality_report(tmp_path)
    report = json.loads((tmp_path / "quality_report.json").read_text())
    assert "jitter_correction" not in report


@pytest.mark.unit
def test_quality_report_includes_match_when_set(tmp_path: Path) -> None:
    """When ``shots_manifest.json`` has a ``match`` block, the quality
    report mirrors it under a top-level ``match`` key so downstream
    tooling can answer 'which match was this?' without re-reading the
    manifest."""
    manifest = ShotsManifest(
        source_file="src.mp4",
        fps=30.0,
        total_frames=0,
        shots=[],
        match=MatchInfo(
            home_team="Liverpool",
            away_team="Real Madrid",
            home_score=0,
            away_score=1,
            venue="Stade de France",
            date="2022-05-28",
            moment=MomentInfo(minute=59, description="Vinicius"),
            kits=KitColors(home_primary="#c8102e", away_primary="#ffffff"),
        ),
    )
    (tmp_path / "shots").mkdir()
    manifest.save(tmp_path / "shots" / "shots_manifest.json")

    write_quality_report(tmp_path)
    report = json.loads((tmp_path / "quality_report.json").read_text())
    assert report["match"]["home_team"] == "Liverpool"
    assert report["match"]["moment"]["minute"] == 59
    assert report["match"]["kits"]["home_primary"] == "#c8102e"


@pytest.mark.unit
def test_quality_report_omits_match_when_unset(tmp_path: Path) -> None:
    """No manifest → no ``match`` key in the report."""
    write_quality_report(tmp_path)
    report = json.loads((tmp_path / "quality_report.json").read_text())
    assert "match" not in report


@pytest.mark.unit
def test_quality_report_prepare_shots_section(tmp_path: Path) -> None:
    """Manifest with groups/exclusions + sync map → ingestion summary."""
    from src.schemas.shots import HighlightGroup, Shot
    from src.schemas.sync_map import Alignment, GroupSync, SyncMap

    def _shot(sid, **kw):
        base = dict(id=sid, start_frame=0, end_frame=49, start_time=0.0,
                    end_time=2.0, clip_file=f"shots/{sid}.mp4")
        base.update(kw)
        return Shot(**base)

    manifest = ShotsManifest(
        source_file="reel.mp4", fps=25.0, total_frames=250,
        shots=[
            _shot("s001", group_id="g01"),
            _shot("s002", kind="reaction", excluded=True,
                  exclude_reason="reaction"),
            _shot("s003", group_id="g01", speed_factor=1.8),
            _shot("s004", kind="transition", excluded=True,
                  exclude_reason="transition"),
            _shot("s005", group_id="g02"),
        ],
        groups=[
            HighlightGroup(id="g01", label="Highlight 1",
                           shot_ids=["s001", "s003"],
                           boundary_rule="start", boundary_confidence=1.0),
            HighlightGroup(id="g02", label="Highlight 2",
                           shot_ids=["s005"],
                           boundary_rule="transition",
                           boundary_confidence=0.9),
        ],
    )
    (tmp_path / "shots").mkdir()
    manifest.save(tmp_path / "shots" / "shots_manifest.json")
    SyncMap(groups=[GroupSync(
        group_id="g01", reference_shot="s001",
        alignments=[
            Alignment("s001", 0, "motion_profile", 1.0),
            Alignment("s003", 12, "low_confidence", 0.3),
        ],
    )]).save(tmp_path / "shots" / "sync_map.json")

    write_quality_report(tmp_path)
    report = json.loads((tmp_path / "quality_report.json").read_text())
    sec = report["prepare_shots"]
    assert sec["total_shots"] == 5
    assert sec["active_shots"] == 3
    assert sec["excluded"] == {"reaction": 1, "transition": 1}
    assert sec["group_count"] == 2
    g01 = next(g for g in sec["groups"] if g["id"] == "g01")
    assert g01["shots"] == 2
    assert g01["alignment"]["min_confidence"] == 0.3
    assert "low_confidence" in g01["alignment"]["methods"]
    assert sec["low_confidence_groups"] == ["g01"]


@pytest.mark.unit
def test_quality_report_no_prepare_section_without_groups(
    tmp_path: Path,
) -> None:
    """Flat copy-mode manifests (no groups, nothing excluded) skip the
    ingestion section — there is nothing to review."""
    manifest = ShotsManifest(source_file="x", fps=25.0, total_frames=0,
                             shots=[])
    (tmp_path / "shots").mkdir()
    manifest.save(tmp_path / "shots" / "shots_manifest.json")
    write_quality_report(tmp_path)
    report = json.loads((tmp_path / "quality_report.json").read_text())
    assert "prepare_shots" not in report


@pytest.mark.unit
def test_quality_report_omits_timings_when_missing(tmp_path: Path) -> None:
    """No timings.json (e.g. quality_report run standalone, or an older
    output dir predating the runner's timing instrumentation) → no
    'timings' key. Mirrors the 'match'/'jitter_correction' omit pattern."""
    write_quality_report(tmp_path)
    report = json.loads((tmp_path / "quality_report.json").read_text())
    assert "timings" not in report


@pytest.mark.unit
def test_quality_report_mirrors_timings_additively(tmp_path: Path) -> None:
    """timings.json (written by src.pipeline.runner.run_pipeline) is
    mirrored verbatim under report['timings']. Purely additive: every
    other section built from an existing fixture in this file must be
    completely unaffected by timings.json being present."""
    AnchorSet(
        clip_id="play",
        image_size=(1280, 720),
        anchors=(
            Anchor(
                frame=0,
                landmarks=(
                    LandmarkObservation(name="x", image_xy=(0.0, 0.0), world_xyz=(0.0, 0.0, 0.0)),
                ),
            ),
        ),
    ).save(tmp_path / "camera" / "anchors.json")
    eye = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    CameraTrack(
        clip_id="play", fps=30.0, image_size=(1280, 720), t_world=[0.0, 0.0, 0.0],
        frames=(CameraFrame(frame=0, K=eye, R=eye, confidence=0.9, is_anchor=True),),
    ).save(tmp_path / "camera" / "camera_track.json")

    timings_payload = {
        "stages": {
            "camera": {"seconds": 1.5, "per_shot": {}},
            "hmr_world": {"seconds": 42.0, "per_shot": {"play": 42.0}},
        },
        "total_seconds": 43.5,
    }
    (tmp_path / "timings.json").write_text(json.dumps(timings_payload))

    write_quality_report(tmp_path)
    report = json.loads((tmp_path / "quality_report.json").read_text())

    # New section mirrors the file verbatim.
    assert report["timings"] == timings_payload
    # Pre-existing section (built from the fixtures above, same as
    # test_quality_report_aggregates_three_stages) is untouched — same
    # keys, same values as when timings.json is absent.
    assert report["camera"]["anchor_count"] == 1
    assert "timings" in report
    # Additive-only: the key set gained exactly 'timings', nothing else
    # changed shape (camera keys unchanged from the no-timings case).
    assert set(report["camera"]) == {
        "anchor_count", "low_confidence_frame_count",
        "low_confidence_frame_ranges", "mean_anchor_residual_px",
        "body_drift_max_m", "distortion",
    }

@pytest.mark.unit
def test_quality_report_ball_per_shot_with_diag(tmp_path: Path) -> None:
    """Per-shot ball entries pull anchoring/solver diagnostics from the
    ball_diag sidecar: anchor counts, event tallies, flagged bounces,
    underconstrained spans, contact gaps."""
    from src.schemas.shots import Shot, ShotsManifest

    (tmp_path / "shots").mkdir(parents=True, exist_ok=True)
    (tmp_path / "ball").mkdir(parents=True, exist_ok=True)
    ShotsManifest(
        source_file="match.mp4",
        fps=30.0,
        total_frames=20,
        shots=[Shot(id="s1", clip_file="shots/s1.mp4", start_frame=0,
                    end_frame=19, start_time=0.0, end_time=19 / 30.0)],
    ).save(tmp_path / "shots" / "shots_manifest.json")

    BallTrack(
        clip_id="s1",
        fps=30.0,
        frames=tuple(
            BallFrame(frame=i, world_xyz=(0.0, 0.0, 0.11),
                      state="grounded", confidence=0.9)
            for i in range(20)
        ),
        flight_segments=(),
    ).save(tmp_path / "ball" / "s1_ball_track.json")
    (tmp_path / "ball" / "s1_ball_diag.json").write_text(json.dumps({
        "underconstrained_spans": [{"start": 3, "end": 9, "residual_px": 7.2}],
        "segments": [],
        "bounces": [
            {"frame": 9, "restitution": 1.4, "flagged": True},
            {"frame": 15, "restitution": 0.7, "flagged": False},
        ],
        "splits": 1,
        "contact_gaps": [
            {"frame": 4, "player_id": "P001", "bone": "l_foot",
             "gap_m": 0.31, "manual": False},
        ],
        "events": [
            {"frame": 4, "kind": "touch", "score": 0.8,
             "player_id": "P001", "bone": "l_foot",
             "goal_element": None, "end_frame": None},
            {"frame": 9, "kind": "bounce", "score": 0.6, "player_id": None,
             "bone": None, "goal_element": None, "end_frame": None},
        ],
        "anchors": {"manual": 1, "auto_generated": 4, "merged": 5, "nodes": 5},
        "detection_coverage": {"pass1": 0.8, "second_pass": 0.1, "total": 0.9},
        "cross_replay": {"partner_shots": ["origi02"], "refined_offset": -144.0,
                         "n_inlier_fixes": 11, "offset_disagreement_frames": 2.0},
        "mode_search": {"hypotheses_explored": 42, "beam_width": 8,
                        "winning_cost": 12.5, "runner_up_cost": 18.0,
                        "fit_calls": 30, "budget_hit": False},
        "out_of_view_spans": [{"start": 11, "end": 17}],
    }))

    write_quality_report(tmp_path)
    report = json.loads((tmp_path / "quality_report.json").read_text())

    ball = report["ball"]
    assert ball["underconstrained_span_count"] == 1
    assert ball["flagged_bounce_count"] == 1
    shot = ball["shots"][0]
    assert shot["shot_id"] == "s1"
    assert shot["anchors"]["auto_generated"] == 4
    assert shot["events"] == {"touch": 1, "bounce": 1}
    assert shot["flagged_bounces"][0]["frame"] == 9
    assert shot["max_contact_gap_m"] == pytest.approx(0.31)
    assert shot["splits"] == 1
    assert shot["detection_coverage"] == {
        "pass1": 0.8, "second_pass": 0.1, "total": 0.9,
    }
    assert shot["cross_replay"]["n_inlier_fixes"] == 11
    # Phase-2 mode-search + out-of-view diagnostics surface (passthrough).
    assert shot["mode_search"]["hypotheses_explored"] == 42
    assert shot["mode_search"]["winning_cost"] == 12.5
    assert shot["mode_search"]["budget_hit"] is False
    assert shot["out_of_view_spans"] == [{"start": 11, "end": 17}]
