"""Smoke test: ExportStage writes ue_manifest.json from in-memory inputs.

Bypasses Blender (no FBX subprocess) by writing fake FBX placeholders
and asserting only that the manifest is produced and validates.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.schemas.smpl_world import SmplWorldTrack
from src.schemas.ue_manifest import UeManifest
from src.stages.export import ExportStage


def _write_min_inputs(output_dir: Path) -> None:
    (output_dir / "hmr_world").mkdir(parents=True)
    SmplWorldTrack(
        player_id="P001",
        frames=np.arange(5, dtype=np.int64),
        betas=np.zeros(10),
        thetas=np.zeros((5, 24, 3)),
        root_R=np.tile(np.eye(3), (5, 1, 1)),
        root_t=np.zeros((5, 3)),
        confidence=np.ones(5),
    ).save(output_dir / "hmr_world" / "P001_smpl_world.npz")
    cam_dir = output_dir / "camera"
    cam_dir.mkdir(parents=True)
    (cam_dir / "camera_track.json").write_text(
        json.dumps(
            {
                "fps": 30.0,
                "frames": [],
                "image_size": [1920, 1080],
            }
        )
    )


def test_manifest_written_with_one_player(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    _write_min_inputs(output_dir)
    fbx_dir = output_dir / "export" / "fbx"
    fbx_dir.mkdir(parents=True)
    (fbx_dir / "P001.fbx").write_bytes(b"\x00")

    cfg = {
        "export": {"gltf_enabled": False, "fbx_enabled": False},
        "pitch": {"length_m": 105.0, "width_m": 68.0},
        "ball": {"ball_radius_m": 0.11},
    }
    stage = ExportStage(output_dir=output_dir, config=cfg)
    stage.write_ue_manifest(clip_name="clip_demo")

    manifest_path = output_dir / "export" / "ue_manifest.json"
    assert manifest_path.exists()
    m = UeManifest.load(manifest_path)
    assert m.clip_name == "clip_demo"
    assert len(m.players) == 1
    assert m.players[0].player_id == "P001"
    assert m.players[0].fbx == "fbx/P001.fbx"


def test_manifest_uses_player_name_mapping(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    _write_min_inputs(output_dir)
    fbx_dir = output_dir / "export" / "fbx"
    fbx_dir.mkdir(parents=True)
    # Mapping makes the FBX file appear under the mapped name.
    (fbx_dir / "Bellingham.fbx").write_bytes(b"\x00")
    (output_dir / "players.json").write_text(
        json.dumps({"P001": "Bellingham"})
    )

    cfg = {
        "export": {"gltf_enabled": False, "fbx_enabled": False},
        "pitch": {"length_m": 105.0, "width_m": 68.0},
        "ball": {"ball_radius_m": 0.11},
    }
    stage = ExportStage(output_dir=output_dir, config=cfg)
    stage.write_ue_manifest(clip_name="clip_demo")

    m = UeManifest.load(output_dir / "export" / "ue_manifest.json")
    assert m.players[0].player_id == "P001"
    assert m.players[0].display_name == "Bellingham"
    assert m.players[0].fbx == "fbx/Bellingham.fbx"


def test_manifest_falls_back_to_legacy_id_named_fbx(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    _write_min_inputs(output_dir)
    fbx_dir = output_dir / "export" / "fbx"
    fbx_dir.mkdir(parents=True)
    # Mapping says Bellingham, but the FBX on disk is still legacy P001.fbx.
    (fbx_dir / "P001.fbx").write_bytes(b"\x00")
    (output_dir / "players.json").write_text(
        json.dumps({"P001": "Bellingham"})
    )

    cfg = {
        "export": {"gltf_enabled": False, "fbx_enabled": False},
        "pitch": {"length_m": 105.0, "width_m": 68.0},
        "ball": {"ball_radius_m": 0.11},
    }
    stage = ExportStage(output_dir=output_dir, config=cfg)
    stage.write_ue_manifest(clip_name="clip_demo")

    m = UeManifest.load(output_dir / "export" / "ue_manifest.json")
    assert m.players[0].display_name == "Bellingham"
    assert m.players[0].fbx == "fbx/P001.fbx"


def test_manifest_uses_shot_prefixed_fbx_when_shots_manifest_present(
    tmp_path: Path,
) -> None:
    """Multi-shot output names FBXs ``{shot_id}__{display_name}.fbx``.

    When the shots manifest contains the clip_name as a shot id, the
    manifest writer must look up FBXs by the prefixed name, not the
    bare display name.
    """
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    _write_min_inputs(output_dir)

    shots_dir = output_dir / "shots"
    shots_dir.mkdir(exist_ok=True)
    (shots_dir / "shots_manifest.json").write_text(
        json.dumps(
            {
                "source_file": "",
                "fps": 30.0,
                "total_frames": 5,
                "shots": [
                    {
                        "id": "origi01",
                        "start_frame": 0,
                        "end_frame": 4,
                        "start_time": 0.0,
                        "end_time": 0.166,
                        "clip_file": "shots/origi01.mp4",
                        "speed_factor": 1.0,
                    }
                ],
            }
        )
    )

    # The hmr_world track from _write_min_inputs has empty shot_id;
    # tag it as origi01 by re-saving so the per-shot filter accepts it.
    SmplWorldTrack(
        player_id="P001",
        frames=np.arange(5, dtype=np.int64),
        betas=np.zeros(10),
        thetas=np.zeros((5, 24, 3)),
        root_R=np.tile(np.eye(3), (5, 1, 1)),
        root_t=np.zeros((5, 3)),
        confidence=np.ones(5),
        shot_id="origi01",
    ).save(output_dir / "hmr_world" / "P001_smpl_world.npz")

    fbx_dir = output_dir / "export" / "fbx"
    fbx_dir.mkdir(parents=True)
    (fbx_dir / "origi01__P001.fbx").write_bytes(b"\x00")

    cfg = {
        "export": {"gltf_enabled": False, "fbx_enabled": False},
        "pitch": {"length_m": 105.0, "width_m": 68.0},
        "ball": {"ball_radius_m": 0.11},
    }
    stage = ExportStage(output_dir=output_dir, config=cfg)
    stage.write_ue_manifest(clip_name="origi01")

    m = UeManifest.load(output_dir / "export" / "ue_manifest.json")
    assert m.clip_name == "origi01"
    assert len(m.players) == 1
    assert m.players[0].player_id == "P001"
    assert m.players[0].fbx == "fbx/origi01__P001.fbx"


def test_manifest_skipped_when_no_player_fbx(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    _write_min_inputs(output_dir)
    fbx_dir = output_dir / "export" / "fbx"
    fbx_dir.mkdir(parents=True)
    # Note: no P001.fbx written.

    cfg = {
        "export": {"gltf_enabled": False, "fbx_enabled": False},
        "pitch": {"length_m": 105.0, "width_m": 68.0},
        "ball": {"ball_radius_m": 0.11},
    }
    stage = ExportStage(output_dir=output_dir, config=cfg)
    stage.write_ue_manifest(clip_name="clip_demo")
    manifest_path = output_dir / "export" / "ue_manifest.json"
    assert not manifest_path.exists()


def test_manifest_picks_up_per_shot_ball_and_camera_fbx(tmp_path: Path) -> None:
    """When a shots_manifest names the clip's primary shot, the UE
    manifest must reference per-shot FBX/JSON for ball and camera
    ({shot_id}_ball.fbx, ball/{shot_id}_ball_track.json, etc.) — not
    the legacy unprefixed paths."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    _write_min_inputs(output_dir)

    # Re-save the hmr_world track with shot_id so per-shot filter accepts it.
    SmplWorldTrack(
        player_id="P001",
        frames=np.arange(5, dtype=np.int64),
        betas=np.zeros(10),
        thetas=np.zeros((5, 24, 3)),
        root_R=np.tile(np.eye(3), (5, 1, 1)),
        root_t=np.zeros((5, 3)),
        confidence=np.ones(5),
        shot_id="origi01",
    ).save(output_dir / "hmr_world" / "P001_smpl_world.npz")

    shots_dir = output_dir / "shots"
    shots_dir.mkdir(exist_ok=True)
    (shots_dir / "shots_manifest.json").write_text(
        json.dumps(
            {
                "source_file": "",
                "fps": 30.0,
                "total_frames": 5,
                "shots": [
                    {
                        "id": "origi01",
                        "start_frame": 0,
                        "end_frame": 4,
                        "start_time": 0.0,
                        "end_time": 0.166,
                        "clip_file": "shots/origi01.mp4",
                        "speed_factor": 1.0,
                    }
                ],
            }
        )
    )

    # Per-shot camera_track + ball_track.
    cam_dir = output_dir / "camera"
    (cam_dir / "origi01_camera_track.json").write_text(
        json.dumps(
            {
                "fps": 30.0,
                "frames": [
                    {"frame": 0, "K": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "R": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]},
                    {"frame": 4, "K": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "R": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]},
                ],
                "image_size": [1920, 1080],
            }
        )
    )
    ball_dir = output_dir / "ball"
    ball_dir.mkdir()
    (ball_dir / "origi01_ball_track.json").write_text(
        json.dumps({
            "frames": [
                {"frame": 0, "world_xyz": [10.0, 20.0, 0.11], "state": "grounded"},
                {"frame": 3, "world_xyz": [12.0, 21.0, 0.11], "state": "grounded"},
            ],
        })
    )

    fbx_dir = output_dir / "export" / "fbx"
    fbx_dir.mkdir(parents=True)
    (fbx_dir / "origi01__P001.fbx").write_bytes(b"\x00")
    (fbx_dir / "origi01_ball.fbx").write_bytes(b"\x00")
    (fbx_dir / "origi01_camera.fbx").write_bytes(b"\x00")

    cfg = {
        "export": {"gltf_enabled": False, "fbx_enabled": False},
        "pitch": {"length_m": 105.0, "width_m": 68.0},
        "ball": {"ball_radius_m": 0.11},
    }
    stage = ExportStage(output_dir=output_dir, config=cfg)
    stage.write_ue_manifest(clip_name="origi01")

    m = UeManifest.load(output_dir / "export" / "ue_manifest.json")
    assert m.ball is not None, "manifest should reference per-shot ball FBX"
    assert m.ball.fbx == "fbx/origi01_ball.fbx"
    assert m.ball.frame_range == (0, 3)
    assert m.camera is not None, "manifest should reference per-shot camera FBX"
    assert m.camera.fbx == "fbx/origi01_camera.fbx"
