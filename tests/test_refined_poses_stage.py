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


@pytest.mark.integration
def test_refined_poses_two_shots_fuses_root_t(tmp_path: Path) -> None:
    """Two shots, sync offset, both see player at the same wall-clock instant.

    Each shot's local frame f corresponds to reference frame f - offset.
    Shot A: offset 0,  local frames 0..9, root_t = [f, 0, 0]
    Shot B: offset 5,  local frames 5..14, root_t = [(f-5)+0.2, 0, 0]
                       i.e. on the reference timeline B sees ref_f = local_f - 5
                       at root_t_x = ref_f + 0.2.
    """
    output_dir = tmp_path
    (output_dir / "hmr_world").mkdir()
    _write_sync_map(output_dir, ref="A", offsets={"A": 0, "B": 5})

    a = _make_smpl_track(
        player_id="P001", shot_id="A", n_frames=10, root_t_x_per_frame=1.0
    )
    a.save(output_dir / "hmr_world" / "A__P001_smpl_world.npz")

    n_b = 10
    b_local_frames = np.arange(5, 5 + n_b, dtype=np.int64)
    b_root_t = np.column_stack(
        [
            (b_local_frames - 5) + 0.2,
            np.zeros(n_b),
            np.zeros(n_b),
        ]
    )
    b = SmplWorldTrack(
        player_id="P001",
        frames=b_local_frames,
        betas=np.zeros(10),
        thetas=np.zeros((n_b, 24, 3)),
        root_R=np.tile(np.eye(3), (n_b, 1, 1)),
        root_t=b_root_t,
        confidence=np.ones(n_b),
        shot_id="B",
    )
    b.save(output_dir / "hmr_world" / "B__P001_smpl_world.npz")

    stage = RefinedPosesStage(config=_default_config(), output_dir=output_dir)
    stage.run()
    refined = RefinedPose.load(output_dir / "refined_poses" / "P001_refined.npz")

    # Reference timeline: A at ref 0..9, B at ref 0..9 (local 5..14 - offset 5).
    np.testing.assert_array_equal(refined.frames, np.arange(0, 10))
    # All ref frames have both A and B contributing.
    # Mean of ref_f and (ref_f + 0.2) = ref_f + 0.1.
    np.testing.assert_allclose(
        refined.root_t[:, 0], np.arange(10) + 0.1, atol=1e-9
    )
    assert refined.view_count.tolist() == [2] * 10
    assert set(refined.contributing_shots) == {"A", "B"}
