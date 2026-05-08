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
