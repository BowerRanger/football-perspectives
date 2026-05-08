"""Tests for prepare_shots directory ingestion + legacy migration."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from src.schemas.shots import ShotsManifest
from src.stages.prepare_shots import PrepareShotsStage


def _write_dummy_mp4(path: Path, n_frames: int = 5, w: int = 320, h: int = 240) -> None:
    """Write a minimal black-frame .mp4 so cv2 can read frame count."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(path), fourcc, 25.0, (w, h))
    for _ in range(n_frames):
        vw.write(np.zeros((h, w, 3), dtype=np.uint8))
    vw.release()


@pytest.mark.unit
def test_directory_ingestion_one_shot_per_mp4(tmp_path: Path) -> None:
    in_dir = tmp_path / "clips"
    in_dir.mkdir()
    for name in ("alpha", "beta", "gamma"):
        _write_dummy_mp4(in_dir / f"{name}.mp4")

    output_dir = tmp_path / "out"
    stage = PrepareShotsStage(config={}, output_dir=output_dir, video_path=in_dir)
    stage.run()

    manifest = ShotsManifest.load(output_dir / "shots" / "shots_manifest.json")
    assert [s.id for s in manifest.shots] == ["alpha", "beta", "gamma"]
    for shot in manifest.shots:
        assert (output_dir / "shots" / f"{shot.id}.mp4").exists()


@pytest.mark.unit
def test_directory_glob_is_depth_one_only(tmp_path: Path) -> None:
    """Nested .mp4s in subdirectories are NOT picked up. Only depth-1."""
    in_dir = tmp_path / "clips"
    sub = in_dir / "sub"
    sub.mkdir(parents=True)
    _write_dummy_mp4(in_dir / "play.mp4")
    _write_dummy_mp4(sub / "deep.mp4")
    output_dir = tmp_path / "out"
    stage = PrepareShotsStage(config={}, output_dir=output_dir, video_path=in_dir)
    stage.run()
    manifest = ShotsManifest.load(output_dir / "shots" / "shots_manifest.json")
    assert [s.id for s in manifest.shots] == ["play"]


@pytest.mark.unit
def test_duplicate_stems_after_sanitisation_raises(tmp_path: Path) -> None:
    in_dir = tmp_path / "clips"
    in_dir.mkdir()
    _write_dummy_mp4(in_dir / "my clip.mp4")     # → "myclip"
    _write_dummy_mp4(in_dir / "my.clip.mp4")     # → "myclip" (collision)
    output_dir = tmp_path / "out"
    stage = PrepareShotsStage(config={}, output_dir=output_dir, video_path=in_dir)
    with pytest.raises(ValueError, match="duplicate shot_id"):
        stage.run()


@pytest.mark.unit
def test_legacy_single_shot_migration(tmp_path: Path) -> None:
    """Pre-existing 'output/camera/anchors.json' (no shot prefix) must be
    renamed to '{shot_id}_anchors.json' on first run with the new code.
    Idempotent — running prepare_shots twice is a no-op the second time.
    """
    output_dir = tmp_path / "out"
    (output_dir / "camera").mkdir(parents=True)
    (output_dir / "camera" / "anchors.json").write_text(
        '{"clip_id":"old","image_size":[1920,1080],"anchors":[]}'
    )
    (output_dir / "camera" / "camera_track.json").write_text('{"clip_id":"old"}')
    (output_dir / "ball").mkdir()
    (output_dir / "ball" / "ball_track.json").write_text('{"clip_id":"old"}')
    (output_dir / "export" / "gltf").mkdir(parents=True)
    (output_dir / "export" / "gltf" / "scene.glb").write_bytes(b"fake glb")

    in_dir = tmp_path / "clips"
    in_dir.mkdir()
    _write_dummy_mp4(in_dir / "play.mp4")

    stage = PrepareShotsStage(
        config={}, output_dir=output_dir, video_path=in_dir / "play.mp4",
    )
    stage.run()

    assert (output_dir / "camera" / "play_anchors.json").exists()
    assert not (output_dir / "camera" / "anchors.json").exists()
    assert (output_dir / "camera" / "play_camera_track.json").exists()
    assert (output_dir / "ball" / "play_ball_track.json").exists()
    assert (output_dir / "export" / "gltf" / "play_scene.glb").exists()

    # Idempotent: a second run is a no-op (no legacy files left to migrate).
    stage.run()
    assert (output_dir / "camera" / "play_anchors.json").exists()
