import json
import numpy as np
import pytest
from pathlib import Path

from src.schemas.smpl_result import SmplResult
from src.schemas.player_matches import PlayerMatches, MatchedPlayer, PlayerView
from src.schemas.export_result import ExportResult
from src.stages.export import ExportStage
from src.utils.gltf_builder import build_minimal_glb, build_scene_metadata
from src.pipeline.config import load_config


def _setup_export_workspace(tmp_path, n_players=2, n_frames=10):
    """Create synthetic SMPL results and matching data."""
    smpl_dir = tmp_path / "smpl"
    smpl_dir.mkdir()

    for i in range(n_players):
        pid = f"P{i+1:03d}"
        SmplResult(
            player_id=pid,
            betas=np.random.randn(10).astype(np.float32),
            poses=np.random.randn(n_frames, 72).astype(np.float32),
            transl=np.random.randn(n_frames, 3).astype(np.float32) * 10 + [50, 30, 0],
            fps=25.0,
        ).save(smpl_dir / f"{pid}_smpl.npz")

    match_dir = tmp_path / "matching"
    match_dir.mkdir()
    PlayerMatches(matched_players=[
        MatchedPlayer(player_id=f"P{i+1:03d}", team="A" if i < n_players // 2 else "B",
                      views=[PlayerView(shot_id="cam_a", track_id=f"T{i+1:03d}")])
        for i in range(n_players)
    ]).save(match_dir / "player_matches.json")


def test_build_scene_metadata():
    results = [
        SmplResult("P001", np.zeros(10), np.zeros((20, 72)), np.zeros((20, 3)), 25.0),
        SmplResult("P002", np.zeros(10), np.zeros((15, 72)), np.zeros((15, 3)), 25.0),
    ]
    teams = {"P001": "A", "P002": "B"}
    meta = build_scene_metadata(results, teams, 25.0)

    assert meta["fps"] == 25.0
    assert meta["total_frames"] == 20
    assert meta["num_players"] == 2
    assert len(meta["players"]) == 2
    assert meta["players"][0]["team"] == "A"
    assert meta["pitch"]["length"] == 105.0


def test_build_minimal_glb_produces_valid_binary():
    results = [
        SmplResult("P001", np.zeros(10), np.zeros((5, 72)),
                    np.array([[50, 30, 0], [51, 30, 0], [52, 30, 0], [53, 30, 0], [54, 30, 0]], dtype=np.float32),
                    25.0),
    ]
    glb = build_minimal_glb(results, {"P001": "A"})

    # GLB magic number
    assert glb[:4] == b'glTF'
    # Version 2
    assert int.from_bytes(glb[4:8], 'little') == 2
    # Total length matches
    total_len = int.from_bytes(glb[8:12], 'little')
    assert total_len == len(glb)


def test_export_stage_produces_glb(tmp_path):
    _setup_export_workspace(tmp_path)
    cfg = load_config()
    stage = ExportStage(config=cfg, output_dir=tmp_path)

    assert not stage.is_complete()
    stage.run()
    assert stage.is_complete()

    assert (tmp_path / "export" / "gltf" / "scene.glb").exists()
    assert (tmp_path / "export" / "gltf" / "scene_metadata.json").exists()
    assert (tmp_path / "export" / "export_result.json").exists()

    # Verify metadata content
    meta = json.loads((tmp_path / "export" / "gltf" / "scene_metadata.json").read_text())
    assert meta["num_players"] == 2
    assert meta["fps"] == 25.0


def test_export_result_round_trip(tmp_path):
    original = ExportResult(
        players=["P001", "P002"],
        gltf_file="export/gltf/scene.glb",
        metadata_file="export/gltf/scene_metadata.json",
    )
    path = tmp_path / "result.json"
    original.save(path)
    loaded = ExportResult.load(path)

    assert loaded.players == ["P001", "P002"]
    assert loaded.gltf_file == "export/gltf/scene.glb"
