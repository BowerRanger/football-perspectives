"""Tests for src.pipeline.runner.resolve_stages — single-mode stage resolution."""

from pathlib import Path

import numpy as np
import pytest

from src.pipeline.runner import resolve_stages


@pytest.mark.unit
def test_resolve_all():
    assert resolve_stages("all", None) == [
        "prepare_shots",
        "tracking",
        "camera",
        "hmr_world",
        "refined_poses",
        "ball",
        "export",
    ]


@pytest.mark.unit
def test_resolve_all_runs_refined_poses_before_ball() -> None:
    """Ball reads cleaned bone positions from refined_poses for
    player_touch anchors, so refined_poses must run first."""
    stages = resolve_stages("all", None)
    assert "refined_poses" in stages
    assert stages.index("refined_poses") < stages.index("ball")
    assert stages.index("ball") < stages.index("export")


@pytest.mark.unit
def test_resolve_from_refined_poses_includes_ball_and_export() -> None:
    assert resolve_stages("all", "refined_poses") == [
        "refined_poses", "ball", "export",
    ]


@pytest.mark.integration
def test_pipeline_refined_poses_end_to_end(tmp_path: Path) -> None:
    """Two shots, one shared player. Stage HMR outputs directly, then run
    refined_poses + export through run_pipeline. Asserts one fused NPZ per
    player and that quality_report picks up the summary.
    """
    import json

    import numpy as np

    from src.pipeline.runner import run_pipeline
    from src.schemas.refined_pose import RefinedPose
    from src.schemas.shots import Shot, ShotsManifest
    from src.schemas.smpl_world import SmplWorldTrack
    from src.schemas.sync_map import Alignment, SyncMap

    out = tmp_path
    (out / "shots").mkdir()
    (out / "hmr_world").mkdir()

    ShotsManifest(
        source_file="",
        fps=30.0,
        total_frames=20,
        shots=[
            Shot(id="A", start_frame=0, end_frame=9, start_time=0.0,
                 end_time=0.3, clip_file="shots/A.mp4", speed_factor=1.0),
            Shot(id="B", start_frame=0, end_frame=9, start_time=0.0,
                 end_time=0.3, clip_file="shots/B.mp4", speed_factor=1.0),
        ],
    ).save(out / "shots" / "shots_manifest.json")
    SyncMap(
        reference_shot="A",
        alignments=[
            Alignment(shot_id="A", frame_offset=0),
            Alignment(shot_id="B", frame_offset=0),
        ],
    ).save(out / "shots" / "sync_map.json")

    n = 10
    frames = np.arange(n, dtype=np.int64)
    base = np.column_stack([frames * 1.0, np.zeros(n), np.zeros(n)])
    for sid in ("A", "B"):
        SmplWorldTrack(
            player_id="P001", frames=frames, betas=np.zeros(10),
            thetas=np.zeros((n, 24, 3)),
            root_R=np.tile(np.eye(3), (n, 1, 1)),
            root_t=base.copy(),
            confidence=np.ones(n), shot_id=sid,
        ).save(out / "hmr_world" / f"{sid}__P001_smpl_world.npz")

    config = {
        "pitch": {"length_m": 105.0, "width_m": 68.0},
        "ball": {},
        "export": {"gltf_enabled": False, "fbx_enabled": False},
        "refined_poses": {
            "outlier_k_sigma": 3.0,
            "min_contributing_views": 1,
            "high_disagreement_pos_m": 0.5,
            "high_disagreement_rot_rad": 0.5,
            "savgol_window": 1,
            "savgol_poly": 2,
            "smooth_rotations": False,
            "beta_aggregation": "weighted_mean",
            "beta_disagreement_warn": 0.3,
            # Disable ground-snap — the test rigs identity root_R which
            # the snap interprets unrealistically. Snap's stage-level
            # wiring is exercised in test_refined_poses_stage.py.
            "ground_snap_max_distance": 0.0,
        },
    }
    run_pipeline(
        output_dir=out,
        stages="refined_poses,export",
        from_stage=None,
        config=config,
    )
    assert (out / "refined_poses" / "P001_refined.npz").exists()
    refined = RefinedPose.load(out / "refined_poses" / "P001_refined.npz")
    assert refined.contributing_shots == ("A", "B")
    np.testing.assert_array_equal(refined.frames, frames)
    # Savgol smoothing in refined_poses introduces sub-µm float32
    # quantization on already-linear inputs; tolerance accommodates it.
    np.testing.assert_allclose(refined.root_t, base, atol=1e-5)
    summary = json.loads(
        (out / "refined_poses" / "refined_poses_summary.json").read_text()
    )
    assert summary["players_refined"] == 1
    assert summary["multi_shot_players"] == 1

    report = json.loads((out / "quality_report.json").read_text())
    assert report["refined_poses"]["players_refined"] == 1


@pytest.mark.unit
def test_resolve_subset():
    assert resolve_stages("camera,hmr_world", None) == ["camera", "hmr_world"]


@pytest.mark.unit
def test_resolve_unknown_raises():
    with pytest.raises(ValueError):
        resolve_stages("calibration", None)


@pytest.mark.unit
def test_resolve_with_from_stage_skips_earlier():
    result = resolve_stages("all", "hmr_world")
    assert result == ["hmr_world", "refined_poses", "ball", "export"]


@pytest.mark.integration
def test_pipeline_two_shots_end_to_end(tmp_path: Path) -> None:
    """Stage two synthetic clips with anchors and hand-rolled SMPL fixtures,
    then run camera + export through ``run_pipeline``. Asserts that each
    shot produces its own ``{shot_id}_camera_track.json`` and
    ``{shot_id}_scene.glb`` and that the artefacts do not collide.

    Tracking and hmr_world are skipped — both call out to YOLO/GVHMR which
    are too heavy for CI. The plumbing is what's under test, not the ML
    weights, so we stage the SmplWorldTrack outputs directly.
    """
    from src.pipeline.runner import run_pipeline
    from src.schemas.shots import Shot, ShotsManifest
    from src.schemas.smpl_world import SmplWorldTrack
    from tests.fixtures.synthetic_clip import render_synthetic_clip
    from tests.test_camera_stage import (
        _LANDMARK_WORLD, _build_anchor_set, _write_clip_mp4,
    )

    clip_a = render_synthetic_clip(n_frames=20)
    clip_b = render_synthetic_clip(n_frames=20)
    shots = tmp_path / "shots"
    shots.mkdir()
    _write_clip_mp4(clip_a, shots / "alpha.mp4")
    _write_clip_mp4(clip_b, shots / "beta.mp4")
    fps = clip_a.fps if clip_a.fps > 0 else 25.0
    ShotsManifest(
        source_file=str(shots),
        fps=fps,
        total_frames=40,
        shots=[
            Shot(id="alpha", start_frame=0, end_frame=19, start_time=0.0,
                 end_time=20 / fps, clip_file="shots/alpha.mp4"),
            Shot(id="beta", start_frame=0, end_frame=19, start_time=0.0,
                 end_time=20 / fps, clip_file="shots/beta.mp4"),
        ],
    ).save(shots / "shots_manifest.json")

    n = len(clip_a.frames)
    for sid, clip in (("alpha", clip_a), ("beta", clip_b)):
        anchor_set = _build_anchor_set(clip, [0, n // 2, n - 1], _LANDMARK_WORLD)
        anchor_set.save(tmp_path / "camera" / f"{sid}_anchors.json")

    # Stage hand-rolled hmr_world output so export has something to package.
    hmr_dir = tmp_path / "hmr_world"
    hmr_dir.mkdir()
    for sid in ("alpha", "beta"):
        SmplWorldTrack(
            player_id=f"{sid}_T1",
            frames=np.array([0]),
            betas=np.zeros(10),
            thetas=np.zeros((1, 24, 3)),
            root_R=np.tile(np.eye(3), (1, 1, 1)),
            root_t=np.zeros((1, 3)),
            confidence=np.full(1, 0.9),
            shot_id=sid,
        ).save(hmr_dir / f"{sid}_T1_smpl_world.npz")

    run_pipeline(
        output_dir=tmp_path,
        stages="camera,export",
        from_stage=None,
        config={
            "camera": {"static_camera": False},
            "export": {"gltf_enabled": True, "fbx_enabled": False},
        },
    )

    # Per-shot camera tracks must exist and be distinct files.
    assert (tmp_path / "camera" / "alpha_camera_track.json").exists()
    assert (tmp_path / "camera" / "beta_camera_track.json").exists()
    # Per-shot GLBs.
    assert (tmp_path / "export" / "gltf" / "alpha_scene.glb").exists()
    assert (tmp_path / "export" / "gltf" / "beta_scene.glb").exists()
    # Legacy unprefixed file must NOT be written when manifest is present.
    assert not (tmp_path / "export" / "gltf" / "scene.glb").exists()


@pytest.mark.unit
def test_run_pipeline_shot_filter_propagates_to_stage(
    tmp_path: Path, monkeypatch,
) -> None:
    """run_pipeline(stages='camera', shot_filter='alpha') sets
    stage.shot_filter='alpha' on the constructed CameraStage."""
    from src.pipeline.runner import run_pipeline
    from src.pipeline.base import BaseStage

    captured: dict = {}

    class FakeCameraStage(BaseStage):
        name = "camera"

        def is_complete(self) -> bool:
            return False

        def run(self) -> None:
            captured["shot_filter"] = self.shot_filter

    def fake_stage_class(name: str):
        if name == "camera":
            return FakeCameraStage
        return None

    monkeypatch.setattr("src.pipeline.runner._stage_class", fake_stage_class)
    run_pipeline(
        output_dir=tmp_path,
        stages="camera",
        from_stage=None,
        config={},
        shot_filter="alpha",
    )
    assert captured["shot_filter"] == "alpha"
