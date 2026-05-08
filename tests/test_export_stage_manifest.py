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
