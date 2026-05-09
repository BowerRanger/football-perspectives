"""Unit test: quality_report aggregator builds the expected JSON shape."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.pipeline.quality_report import write_quality_report
from src.schemas.anchor import Anchor, AnchorSet, LandmarkObservation
from src.schemas.ball_track import BallFrame, BallTrack
from src.schemas.camera_track import CameraFrame, CameraTrack
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
