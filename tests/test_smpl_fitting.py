import pytest

pytest.skip(
    "awaiting later phase: imports a module deleted in Phase 0 of the "
    "broadcast-mono pipeline rewrite",
    allow_module_level=True,
)


import numpy as np
import pytest
from pathlib import Path

from src.schemas.smpl_result import SmplResult
from src.schemas.triangulated import TriangulatedPlayer
from src.stages.smpl_fitting import SmplFittingStage
from src.utils.smpl_fitting import fit_smpl_sequence, _fit_fallback
from src.pipeline.config import load_config


def _synthetic_triangulated(
    tmp_path,
    player_id="P001",
    n_frames=10,
    player_name="",
    team="",
):
    """Create a synthetic triangulated player and save to tmp_path."""
    positions = np.zeros((n_frames, 17, 3), dtype=np.float32)
    for f in range(n_frames):
        for j in range(17):
            positions[f, j] = [50.0 + f * 0.1, 30.0, j * 0.1]
    # Set hips at a known height
    for f in range(n_frames):
        positions[f, 11] = [50.0 + f * 0.1 - 0.15, 30.0, 0.9]
        positions[f, 12] = [50.0 + f * 0.1 + 0.15, 30.0, 0.9]

    confidences = np.ones((n_frames, 17), dtype=np.float32) * 0.8
    reproj = np.ones((n_frames, 17), dtype=np.float32) * 2.0
    n_views = np.ones((n_frames, 17), dtype=np.int8) * 2

    tri = TriangulatedPlayer(
        player_id=player_id,
        player_name=player_name,
        team=team,
        positions=positions,
        confidences=confidences,
        reprojection_errors=reproj,
        num_views=n_views,
        fps=25.0,
        start_frame=0,
    )
    tri_dir = tmp_path / "triangulated"
    tri_dir.mkdir(parents=True, exist_ok=True)
    tri.save(tri_dir / f"{player_id}_3d_joints.npz")
    return tri


def test_fallback_fitting_produces_correct_shapes():
    positions = np.random.randn(20, 17, 3).astype(np.float32)
    confidences = np.ones((20, 17), dtype=np.float32)
    betas, poses, transl = _fit_fallback(positions, confidences)

    assert betas.shape == (10,)
    assert poses.shape == (20, 72)
    assert transl.shape == (20, 3)


def test_fallback_translation_from_hip_midpoint():
    positions = np.zeros((5, 17, 3), dtype=np.float32)
    for f in range(5):
        positions[f, 11] = [10.0 + f, 20.0, 0.9]  # left hip
        positions[f, 12] = [10.0 + f + 0.3, 20.0, 0.9]  # right hip

    confidences = np.ones((5, 17), dtype=np.float32)
    betas, poses, transl = _fit_fallback(positions, confidences)

    for f in range(5):
        expected_x = 10.0 + f + 0.15
        assert abs(transl[f, 0] - expected_x) < 0.01
        assert abs(transl[f, 1] - 20.0) < 0.01


def test_fit_smpl_sequence_uses_fallback_without_smplx():
    """Without smplx installed, fit_smpl_sequence should use fallback."""
    positions = np.random.randn(10, 17, 3).astype(np.float32)
    confidences = np.ones((10, 17), dtype=np.float32)
    betas, poses, transl = fit_smpl_sequence(
        positions, confidences,
        model_path="nonexistent/path.pkl",
    )
    assert betas.shape == (10,)
    assert poses.shape == (10, 72)
    assert transl.shape == (10, 3)


def test_smpl_stage_produces_output(tmp_path):
    _synthetic_triangulated(tmp_path, "P001")
    cfg = load_config()
    stage = SmplFittingStage(config=cfg, output_dir=tmp_path)

    assert not stage.is_complete()
    stage.run()
    assert stage.is_complete()

    npz_files = list((tmp_path / "smpl").glob("*.npz"))
    assert len(npz_files) == 1


def test_smpl_stage_carries_player_name_and_team(tmp_path):
    _synthetic_triangulated(tmp_path, "P001", player_name="Salah", team="A")
    cfg = load_config()
    SmplFittingStage(config=cfg, output_dir=tmp_path).run()

    result = SmplResult.load(tmp_path / "smpl" / "P001_smpl.npz")
    assert result.player_name == "Salah"
    assert result.team == "A"


def test_smpl_result_round_trip(tmp_path):
    original = SmplResult(
        player_id="P007",
        player_name="Henderson",
        team="A",
        betas=np.random.randn(10).astype(np.float32),
        poses=np.random.randn(15, 72).astype(np.float32),
        transl=np.random.randn(15, 3).astype(np.float32),
        fps=30.0,
    )
    path = tmp_path / "test_smpl.npz"
    original.save(path)
    loaded = SmplResult.load(path)

    assert loaded.player_id == "P007"
    assert loaded.player_name == "Henderson"
    assert loaded.team == "A"
    assert loaded.fps == pytest.approx(30.0)
    np.testing.assert_allclose(loaded.betas, original.betas, atol=1e-5)
    np.testing.assert_allclose(loaded.poses, original.poses, atol=1e-5)
    np.testing.assert_allclose(loaded.transl, original.transl, atol=1e-5)
