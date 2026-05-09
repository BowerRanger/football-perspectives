"""Tests for src.stages.refined_poses.RefinedPosesStage."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.schemas.refined_pose import RefinedPose, RefinedPoseDiagnostics
from src.schemas.smpl_world import SmplWorldTrack
from src.schemas.sync_map import Alignment, SyncMap
from src.stages.refined_poses import RefinedPosesStage


def _default_config() -> dict:
    return {
        "refined_poses": {
            "outlier_k_sigma": 3.0,
            "min_contributing_views": 1,
            "high_disagreement_pos_m": 0.5,
            "high_disagreement_rot_rad": 0.5,
            "savgol_window": 1,        # tests run with smoothing disabled
            "savgol_poly": 2,
            "smooth_rotations": False,
            "beta_aggregation": "weighted_mean",
            "beta_disagreement_warn": 0.3,
        }
    }


def _make_smpl_track(
    *,
    player_id: str,
    shot_id: str,
    n_frames: int,
    root_t_x_per_frame: float = 0.5,
    confidence: float = 1.0,
) -> SmplWorldTrack:
    frames = np.arange(n_frames, dtype=np.int64)
    return SmplWorldTrack(
        player_id=player_id,
        frames=frames,
        betas=np.zeros(10),
        thetas=np.zeros((n_frames, 24, 3)),
        root_R=np.tile(np.eye(3), (n_frames, 1, 1)),
        root_t=np.column_stack(
            [frames * root_t_x_per_frame, np.zeros(n_frames), np.zeros(n_frames)]
        ),
        confidence=np.full(n_frames, confidence),
        shot_id=shot_id,
    )


def _write_sync_map(output_dir: Path, *, ref: str, offsets: dict[str, int]) -> None:
    alignments = [
        Alignment(shot_id=sid, frame_offset=off, method="manual", confidence=1.0)
        for sid, off in offsets.items()
    ]
    sm = SyncMap(reference_shot=ref, alignments=alignments)
    (output_dir / "shots").mkdir(parents=True, exist_ok=True)
    sm.save(output_dir / "shots" / "sync_map.json")


@pytest.mark.integration
def test_refined_poses_single_shot_passthrough(tmp_path: Path) -> None:
    output_dir = tmp_path
    (output_dir / "hmr_world").mkdir()
    _write_sync_map(output_dir, ref="origi01", offsets={"origi01": 0})
    track = _make_smpl_track(player_id="P001", shot_id="origi01", n_frames=10)
    track.save(output_dir / "hmr_world" / "origi01__P001_smpl_world.npz")

    stage = RefinedPosesStage(config=_default_config(), output_dir=output_dir)
    assert stage.is_complete() is False
    stage.run()
    assert stage.is_complete() is True

    refined = RefinedPose.load(output_dir / "refined_poses" / "P001_refined.npz")
    assert refined.player_id == "P001"
    assert refined.contributing_shots == ("origi01",)
    np.testing.assert_array_equal(refined.frames, track.frames)
    np.testing.assert_allclose(refined.root_t, track.root_t)
    assert refined.view_count.tolist() == [1] * 10

    diag = RefinedPoseDiagnostics.load(
        output_dir / "refined_poses" / "P001_diagnostics.json"
    )
    assert diag.contributing_shots == ("origi01",)
    assert all(f.low_coverage for f in diag.frames)
    assert all(not f.high_disagreement for f in diag.frames)

    summary = json.loads(
        (output_dir / "refined_poses" / "refined_poses_summary.json").read_text()
    )
    assert summary["players_refined"] == 1
    assert summary["single_shot_players"] == 1
    assert summary["multi_shot_players"] == 0


@pytest.mark.integration
def test_refined_poses_is_complete_after_run(tmp_path: Path) -> None:
    output_dir = tmp_path
    (output_dir / "hmr_world").mkdir()
    _write_sync_map(output_dir, ref="origi01", offsets={"origi01": 0})
    _make_smpl_track(player_id="P001", shot_id="origi01", n_frames=5).save(
        output_dir / "hmr_world" / "origi01__P001_smpl_world.npz"
    )

    stage = RefinedPosesStage(config=_default_config(), output_dir=output_dir)
    stage.run()
    assert stage.is_complete() is True

    (output_dir / "refined_poses" / "P001_refined.npz").unlink()
    assert stage.is_complete() is False
