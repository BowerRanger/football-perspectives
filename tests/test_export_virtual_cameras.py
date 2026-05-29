from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.schemas.camera_selection import CameraSelection, RigSelection
from src.schemas.camera_track import CameraFrame, CameraTrack
from src.schemas.smpl_world import SmplWorldTrack
from src.stages.export import ExportStage


def _write_broadcast_camera(path: Path) -> None:
    frames = tuple(
        CameraFrame(frame=i, K=[[1000, 0, 960], [0, 1000, 540], [0, 0, 1]],
                    R=[[1, 0, 0], [0, 1, 0], [0, 0, 1]], confidence=1.0,
                    is_anchor=False, t=[0.0, 0.0, 0.0])
        for i in range(3)
    )
    CameraTrack(clip_id="shot_01", fps=30.0, image_size=(1920, 1080),
                t_world=[0, 0, 30], frames=frames).save(path)


def _write_player(path: Path) -> None:
    n = 3
    # Upright standing player in pitch coords: root_R maps canonical y-up to
    # pitch z-up (rotation about x by +90deg), pelvis ~0.94 m above ground.
    Rx = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=float)
    SmplWorldTrack(
        player_id="P003", frames=np.arange(n), betas=np.zeros(10),
        thetas=np.zeros((n, 24, 3)), root_R=np.broadcast_to(Rx, (n, 3, 3)).copy(),
        root_t=np.tile([10.0, 20.0, 0.939], (n, 1)), confidence=np.ones(n), shot_id="shot_01",
    ).save(path)


@pytest.mark.unit
def test_generate_virtual_cameras_writes_rig_tracks(tmp_path: Path) -> None:
    out = tmp_path
    (out / "camera").mkdir(parents=True)
    (out / "hmr_world").mkdir(parents=True)
    _write_broadcast_camera(out / "camera" / "shot_01_camera_track.json")
    _write_player(out / "hmr_world" / "P003_smpl_world.npz")
    CameraSelection(shot_id="shot_01",
                    selections=(RigSelection("P003", ("pov", "ots")),)).save(
        out / "export" / "shot_01_camera_selection.json")

    stage = ExportStage(output_dir=out, config={"export": {"virtual_cameras": {}}})
    named = stage._generate_virtual_cameras(shot_id="shot_01")

    pov = out / "camera" / "shot_01_P003_pov_camera_track.json"
    ots = out / "camera" / "shot_01_P003_ots_camera_track.json"
    assert pov.exists() and ots.exists()
    assert {c.name for c in named} == {"P003_pov", "P003_ots"}
    cam = CameraTrack.load(pov)
    R = np.array(cam.frames[0].R); t = np.array(cam.frames[0].t)
    assert 1.4 < (-R.T @ t)[2] < 2.0


@pytest.mark.unit
def test_generate_virtual_cameras_no_selection_returns_empty(tmp_path: Path) -> None:
    out = tmp_path
    (out / "camera").mkdir(parents=True)
    _write_broadcast_camera(out / "camera" / "shot_01_camera_track.json")
    stage = ExportStage(output_dir=out, config={"export": {"virtual_cameras": {}}})
    assert stage._generate_virtual_cameras(shot_id="shot_01") == []
