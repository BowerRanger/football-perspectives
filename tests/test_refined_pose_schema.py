"""Round-trip tests for RefinedPose and RefinedPoseDiagnostics."""

from pathlib import Path

import numpy as np
import pytest

from src.schemas.refined_pose import (
    FrameDiagnostic,
    RefinedPose,
    RefinedPoseDiagnostics,
)


@pytest.mark.unit
def test_refined_pose_round_trip(tmp_path: Path) -> None:
    pose = RefinedPose(
        player_id="P002",
        frames=np.array([0, 1, 2, 3], dtype=np.int64),
        betas=np.zeros(10),
        thetas=np.zeros((4, 24, 3)),
        root_R=np.tile(np.eye(3), (4, 1, 1)),
        root_t=np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]
        ),
        confidence=np.ones(4),
        view_count=np.array([1, 2, 2, 1], dtype=np.int32),
        contributing_shots=("origi01", "origi02"),
    )
    p = tmp_path / "P002_refined.npz"
    pose.save(p)
    loaded = RefinedPose.load(p)
    assert loaded.player_id == "P002"
    np.testing.assert_array_equal(loaded.frames, pose.frames)
    np.testing.assert_array_equal(loaded.view_count, pose.view_count)
    np.testing.assert_allclose(loaded.root_t, pose.root_t)
    assert loaded.contributing_shots == ("origi01", "origi02")


@pytest.mark.unit
def test_refined_pose_diagnostics_round_trip(tmp_path: Path) -> None:
    diag = RefinedPoseDiagnostics(
        player_id="P002",
        frames=(
            FrameDiagnostic(
                frame=0,
                contributing_shots=("origi01",),
                dropped_shots=(),
                pos_disagreement_m=0.0,
                rot_disagreement_rad=0.0,
                low_coverage=True,
                high_disagreement=False,
            ),
            FrameDiagnostic(
                frame=1,
                contributing_shots=("origi01", "origi02"),
                dropped_shots=("origi03",),
                pos_disagreement_m=0.05,
                rot_disagreement_rad=0.02,
                low_coverage=False,
                high_disagreement=False,
            ),
        ),
        contributing_shots=("origi01", "origi02", "origi03"),
        summary={
            "total_frames": 2,
            "single_view_frames": 1,
            "high_disagreement_frames": 0,
        },
    )
    p = tmp_path / "P002_diagnostics.json"
    diag.save(p)
    loaded = RefinedPoseDiagnostics.load(p)
    assert loaded.player_id == "P002"
    assert len(loaded.frames) == 2
    assert loaded.frames[0].low_coverage is True
    assert loaded.frames[1].contributing_shots == ("origi01", "origi02")
    assert loaded.frames[1].dropped_shots == ("origi03",)
    assert loaded.summary["total_frames"] == 2
