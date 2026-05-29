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
    assert report["ball"]["grounded_frames"] == 11
    assert report["ball"]["flight_segments"] == 0
    assert report["ball"]["missing_frames"] == 0


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
    assert report["ball"]["spin_coverage_pct"] == pytest.approx(50.0)


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
