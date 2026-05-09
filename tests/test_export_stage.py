"""Multi-shot export — one GLB per shot."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.schemas.camera_track import CameraFrame, CameraTrack
from src.schemas.shots import Shot, ShotsManifest
from src.schemas.smpl_world import SmplWorldTrack
from src.stages.export import ExportStage


@pytest.mark.unit
def test_export_emits_one_glb_per_shot(tmp_path: Path) -> None:
    eye = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    (tmp_path / "shots").mkdir()
    (tmp_path / "camera").mkdir()
    (tmp_path / "hmr_world").mkdir()
    (tmp_path / "ball").mkdir()
    ShotsManifest(
        source_file="x", fps=25.0, total_frames=2,
        shots=[
            Shot(id="alpha", start_frame=0, end_frame=0, start_time=0,
                 end_time=0.04, clip_file="shots/alpha.mp4"),
            Shot(id="beta", start_frame=0, end_frame=0, start_time=0,
                 end_time=0.04, clip_file="shots/beta.mp4"),
        ],
    ).save(tmp_path / "shots" / "shots_manifest.json")
    for sid in ("alpha", "beta"):
        CameraTrack(
            clip_id=sid, fps=25.0, image_size=(640, 360),
            t_world=[0.0, 0.0, 0.0],
            frames=(CameraFrame(
                frame=0, K=eye, R=eye, confidence=1.0, is_anchor=True,
            ),),
        ).save(tmp_path / "camera" / f"{sid}_camera_track.json")

    SmplWorldTrack(
        player_id="alpha_T3",
        frames=np.array([0]),
        betas=np.zeros(10),
        thetas=np.zeros((1, 24, 3)),
        root_R=np.tile(np.eye(3), (1, 1, 1)),
        root_t=np.zeros((1, 3)),
        confidence=np.full(1, 0.9),
        shot_id="alpha",
    ).save(tmp_path / "hmr_world" / "alpha_T3_smpl_world.npz")
    SmplWorldTrack(
        player_id="beta_T1",
        frames=np.array([0]),
        betas=np.zeros(10),
        thetas=np.zeros((1, 24, 3)),
        root_R=np.tile(np.eye(3), (1, 1, 1)),
        root_t=np.zeros((1, 3)),
        confidence=np.full(1, 0.9),
        shot_id="beta",
    ).save(tmp_path / "hmr_world" / "beta_T1_smpl_world.npz")

    stage = ExportStage(
        config={"export": {"gltf_enabled": True, "fbx_enabled": False}},
        output_dir=tmp_path,
    )
    stage.run()

    assert (tmp_path / "export" / "gltf" / "alpha_scene.glb").exists()
    assert (tmp_path / "export" / "gltf" / "beta_scene.glb").exists()
    # Legacy unprefixed file should NOT exist when manifest is present.
    assert not (tmp_path / "export" / "gltf" / "scene.glb").exists()


@pytest.mark.integration
def test_export_consumes_refined_poses_when_present(tmp_path: Path) -> None:
    """Export's per-shot player loader pulls from refined_poses/*.npz when that
    directory is non-empty, projecting frames back to the shot's local timeline.
    """
    from src.schemas.refined_pose import RefinedPose
    from src.schemas.sync_map import Alignment, SyncMap
    from src.stages.export import _per_shot_smpl_tracks

    output_dir = tmp_path
    (output_dir / "shots").mkdir()
    SyncMap(
        reference_shot="A",
        alignments=[
            Alignment(shot_id="A", frame_offset=0),
            Alignment(shot_id="B", frame_offset=5),
        ],
    ).save(output_dir / "shots" / "sync_map.json")

    refined = RefinedPose(
        player_id="P001",
        frames=np.arange(10, dtype=np.int64),
        betas=np.zeros(10),
        thetas=np.zeros((10, 24, 3)),
        root_R=np.tile(np.eye(3), (10, 1, 1)),
        root_t=np.column_stack(
            [np.arange(10, dtype=np.float64), np.zeros(10), np.zeros(10)]
        ),
        confidence=np.ones(10),
        view_count=np.full(10, 2, dtype=np.int32),
        contributing_shots=("A", "B"),
    )
    (output_dir / "refined_poses").mkdir()
    refined.save(output_dir / "refined_poses" / "P001_refined.npz")

    a_tracks = _per_shot_smpl_tracks(output_dir, shot_id="A")
    assert len(a_tracks) == 1
    assert a_tracks[0].player_id == "P001"
    assert a_tracks[0].shot_id == "A"
    np.testing.assert_array_equal(a_tracks[0].frames, np.arange(10))

    b_tracks = _per_shot_smpl_tracks(output_dir, shot_id="B")
    assert len(b_tracks) == 1
    # Reference frames 0..9 → B-local frames 5..14 via f_local = f_ref + offset.
    np.testing.assert_array_equal(b_tracks[0].frames, np.arange(5, 15))
    np.testing.assert_allclose(
        b_tracks[0].root_t[:, 0], np.arange(10, dtype=np.float64)
    )


@pytest.mark.integration
def test_export_falls_back_to_smpl_world_when_refined_dir_empty(tmp_path: Path) -> None:
    """When output/refined_poses/ is empty, export reads SmplWorldTracks
    from output/hmr_world/ as before.
    """
    from src.stages.export import _per_shot_smpl_tracks

    output_dir = tmp_path
    (output_dir / "hmr_world").mkdir()
    SmplWorldTrack(
        player_id="P001",
        frames=np.arange(5, dtype=np.int64),
        betas=np.zeros(10),
        thetas=np.zeros((5, 24, 3)),
        root_R=np.tile(np.eye(3), (5, 1, 1)),
        root_t=np.zeros((5, 3)),
        confidence=np.ones(5),
        shot_id="A",
    ).save(output_dir / "hmr_world" / "A__P001_smpl_world.npz")

    tracks = _per_shot_smpl_tracks(output_dir, shot_id="A")
    assert len(tracks) == 1
    assert tracks[0].player_id == "P001"
    assert tracks[0].shot_id == "A"
