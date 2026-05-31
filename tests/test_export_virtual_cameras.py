from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.schemas.camera_selection import CameraSelection, RigSelection
from src.schemas.camera_track import CameraFrame, CameraTrack
from src.schemas.smpl_world import SmplWorldTrack
from src.schemas.ue_manifest import UeManifest
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
def test_virtual_cameras_named_after_player_display_name(tmp_path: Path) -> None:
    # When players.json maps the id to a name, the camera entries + track
    # files use the (sanitised) display name, mirroring the player FBX
    # naming convention rather than the raw track id.
    out = tmp_path
    (out / "camera").mkdir(parents=True)
    (out / "hmr_world").mkdir(parents=True)
    _write_broadcast_camera(out / "camera" / "shot_01_camera_track.json")
    _write_player(out / "hmr_world" / "P003_smpl_world.npz")
    (out / "players.json").write_text(json.dumps({"P003": "Alisson Becker"}))
    CameraSelection(shot_id="shot_01",
                    selections=(RigSelection("P003", ("pov", "ots")),)).save(
        out / "export" / "shot_01_camera_selection.json")

    stage = ExportStage(output_dir=out, config={"export": {"virtual_cameras": {}}})
    named = stage._generate_virtual_cameras(shot_id="shot_01")

    assert {c.name for c in named} == {"Alisson_Becker_pov", "Alisson_Becker_ots"}
    assert (out / "camera" / "shot_01_Alisson_Becker_pov_camera_track.json").exists()
    pov = next(c for c in named if c.name == "Alisson_Becker_pov")
    assert pov.track_json == "camera/shot_01_Alisson_Becker_pov_camera_track.json"


@pytest.mark.unit
def test_generate_virtual_cameras_no_selection_returns_empty(tmp_path: Path) -> None:
    out = tmp_path
    (out / "camera").mkdir(parents=True)
    _write_broadcast_camera(out / "camera" / "shot_01_camera_track.json")
    stage = ExportStage(output_dir=out, config={"export": {"virtual_cameras": {}}})
    assert stage._generate_virtual_cameras(shot_id="shot_01") == []


@pytest.mark.unit
def test_generate_virtual_cameras_skips_unknown_player(tmp_path: Path) -> None:
    out = tmp_path
    (out / "camera").mkdir(parents=True)
    (out / "hmr_world").mkdir(parents=True)
    _write_broadcast_camera(out / "camera" / "shot_01_camera_track.json")
    _write_player(out / "hmr_world" / "P003_smpl_world.npz")
    CameraSelection(shot_id="shot_01",
                    selections=(RigSelection("P999", ("pov", "ots")),)).save(
        out / "export" / "shot_01_camera_selection.json")
    stage = ExportStage(output_dir=out, config={"export": {"virtual_cameras": {}}})
    assert stage._generate_virtual_cameras(shot_id="shot_01") == []


@pytest.mark.unit
def test_write_ue_manifest_includes_virtual_cameras(tmp_path: Path) -> None:
    out = tmp_path
    (out / "camera").mkdir(parents=True)
    (out / "hmr_world").mkdir(parents=True)
    (out / "export" / "fbx").mkdir(parents=True)
    _write_broadcast_camera(out / "camera" / "shot_01_camera_track.json")
    _write_player(out / "hmr_world" / "P003_smpl_world.npz")
    (out / "export" / "fbx" / "shot_01__P003.fbx").write_bytes(b"x")  # player fbx so manifest writes
    CameraSelection(shot_id="shot_01",
                    selections=(RigSelection("P003", ("pov",)),)).save(
        out / "export" / "shot_01_camera_selection.json")

    stage = ExportStage(output_dir=out, config={"export": {"virtual_cameras": {}},
                                                 "pitch": {"length_m": 105.0, "width_m": 68.0}})
    stage._generate_virtual_cameras(shot_id="shot_01")
    (out / "shots").mkdir()
    (out / "shots" / "shots_manifest.json").write_text(json.dumps(
        {"source_file": "shot_01.mp4", "fps": 30.0, "total_frames": 3,
         "shots": [{"id": "shot_01", "clip_file": "shot_01.mp4",
                    "start_frame": 0, "end_frame": 2,
                    "start_time": 0.0, "end_time": 0.1}],
         "match": None}))
    stage.write_ue_manifest("shot_01")

    m = UeManifest.load(out / "export" / "ue_manifest.json")
    names = {c.name for c in m.cameras}
    assert "broadcast" in names
    assert "P003_pov" in names
    pov = next(c for c in m.cameras if c.name == "P003_pov")
    assert pov.track_json == "camera/shot_01_P003_pov_camera_track.json"
    assert pov.frame_range == (0, 2)


@pytest.mark.unit
def test_write_ue_manifest_backfills_virtual_camera_fbx(tmp_path: Path) -> None:
    """FBX file for a virtual camera is reflected in the manifest entry when
    the file exists on disk by the time write_ue_manifest runs."""
    out = tmp_path
    (out / "camera").mkdir(parents=True)
    (out / "hmr_world").mkdir(parents=True)
    (out / "export" / "fbx").mkdir(parents=True)
    _write_broadcast_camera(out / "camera" / "shot_01_camera_track.json")
    _write_player(out / "hmr_world" / "P003_smpl_world.npz")
    (out / "export" / "fbx" / "shot_01__P003.fbx").write_bytes(b"x")  # player FBX so manifest writes
    # Create the virtual camera FBX that Blender would have written.
    (out / "export" / "fbx" / "shot_01_P003_pov.fbx").write_bytes(b"vcam")
    CameraSelection(shot_id="shot_01",
                    selections=(RigSelection("P003", ("pov",)),)).save(
        out / "export" / "shot_01_camera_selection.json")

    stage = ExportStage(output_dir=out, config={"export": {"virtual_cameras": {}},
                                                 "pitch": {"length_m": 105.0, "width_m": 68.0}})
    stage._generate_virtual_cameras(shot_id="shot_01")
    (out / "shots").mkdir()
    (out / "shots" / "shots_manifest.json").write_text(json.dumps(
        {"source_file": "shot_01.mp4", "fps": 30.0, "total_frames": 3,
         "shots": [{"id": "shot_01", "clip_file": "shot_01.mp4",
                    "start_frame": 0, "end_frame": 2,
                    "start_time": 0.0, "end_time": 0.1}],
         "match": None}))
    stage.write_ue_manifest("shot_01")

    m = UeManifest.load(out / "export" / "ue_manifest.json")
    pov = next(c for c in m.cameras if c.name == "P003_pov")
    assert pov.fbx == "fbx/shot_01_P003_pov.fbx"


@pytest.mark.unit
def test_write_ue_manifest_virtual_camera_fbx_empty_when_missing(tmp_path: Path) -> None:
    """When the virtual camera FBX does not exist yet, fbx stays empty string."""
    out = tmp_path
    (out / "camera").mkdir(parents=True)
    (out / "hmr_world").mkdir(parents=True)
    (out / "export" / "fbx").mkdir(parents=True)
    _write_broadcast_camera(out / "camera" / "shot_01_camera_track.json")
    _write_player(out / "hmr_world" / "P003_smpl_world.npz")
    (out / "export" / "fbx" / "shot_01__P003.fbx").write_bytes(b"x")
    # NOTE: no shot_01_P003_pov.fbx written — Blender didn't run.
    CameraSelection(shot_id="shot_01",
                    selections=(RigSelection("P003", ("pov",)),)).save(
        out / "export" / "shot_01_camera_selection.json")

    stage = ExportStage(output_dir=out, config={"export": {"virtual_cameras": {}},
                                                 "pitch": {"length_m": 105.0, "width_m": 68.0}})
    stage._generate_virtual_cameras(shot_id="shot_01")
    (out / "shots").mkdir()
    (out / "shots" / "shots_manifest.json").write_text(json.dumps(
        {"source_file": "shot_01.mp4", "fps": 30.0, "total_frames": 3,
         "shots": [{"id": "shot_01", "clip_file": "shot_01.mp4",
                    "start_frame": 0, "end_frame": 2,
                    "start_time": 0.0, "end_time": 0.1}],
         "match": None}))
    stage.write_ue_manifest("shot_01")

    m = UeManifest.load(out / "export" / "ue_manifest.json")
    pov = next(c for c in m.cameras if c.name == "P003_pov")
    assert pov.fbx == ""


@pytest.mark.unit
def test_virtual_cameras_present_when_gltf_disabled(tmp_path: Path) -> None:
    out = tmp_path
    (out / "camera").mkdir(parents=True)
    (out / "hmr_world").mkdir(parents=True)
    (out / "export" / "fbx").mkdir(parents=True)
    (out / "shots").mkdir()
    _write_broadcast_camera(out / "camera" / "shot_01_camera_track.json")
    _write_player(out / "hmr_world" / "P003_smpl_world.npz")
    (out / "export" / "fbx" / "shot_01__P003.fbx").write_bytes(b"x")
    CameraSelection(shot_id="shot_01",
                    selections=(RigSelection("P003", ("pov",)),)).save(
        out / "export" / "shot_01_camera_selection.json")
    (out / "shots" / "shots_manifest.json").write_text(json.dumps(
        {"shots": [{"id": "shot_01", "clip_file": "shot_01.mp4",
                    "start_frame": 0, "end_frame": 2, "start_time": 0.0, "end_time": 0.1}],
         "match": None, "source_file": "x.mp4", "fps": 30.0, "total_frames": 3}))

    stage = ExportStage(output_dir=out, config={
        "export": {"gltf_enabled": False, "fbx_enabled": False, "virtual_cameras": {}},
        "pitch": {"length_m": 105.0, "width_m": 68.0}})
    stage.run()

    m = UeManifest.load(out / "export" / "ue_manifest.json")
    assert "P003_pov" in {c.name for c in m.cameras}
