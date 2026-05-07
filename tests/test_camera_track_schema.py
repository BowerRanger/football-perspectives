"""Round-trip tests for the CameraTrack schema fields added in Phase 1
(``camera_centre``) and Phase 2 (``distortion``)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.schemas.camera_track import CameraTrack


@pytest.mark.unit
def test_camera_track_carries_camera_centre(tmp_path: Path):
    """A CameraTrack saved with ``camera_centre`` round-trips through JSON."""
    track = CameraTrack(
        clip_id="test",
        fps=25.0,
        image_size=(1920, 1080),
        t_world=[0.0, 0.0, 30.0],
        frames=tuple(),
        camera_centre=(52.5, -30.0, 30.0),
    )
    out = tmp_path / "track.json"
    track.save(out)
    loaded = CameraTrack.load(out)
    assert loaded.camera_centre == (52.5, -30.0, 30.0)


@pytest.mark.unit
def test_camera_track_legacy_load_without_camera_centre(tmp_path: Path):
    """Older saved tracks (no ``camera_centre`` key) load with a None default."""
    out = tmp_path / "legacy.json"
    out.write_text(
        json.dumps(
            {
                "clip_id": "legacy",
                "fps": 25.0,
                "image_size": [1920, 1080],
                "t_world": [0.0, 0.0, 30.0],
                "frames": [],
            }
        )
    )
    loaded = CameraTrack.load(out)
    assert loaded.camera_centre is None


@pytest.mark.unit
def test_camera_track_carries_distortion(tmp_path: Path):
    track = CameraTrack(
        clip_id="t",
        fps=25.0,
        image_size=(1920, 1080),
        t_world=[0.0, 0.0, 30.0],
        frames=tuple(),
        distortion=(0.12, -0.04),
    )
    out = tmp_path / "track.json"
    track.save(out)
    loaded = CameraTrack.load(out)
    assert loaded.distortion == (0.12, -0.04)


@pytest.mark.unit
def test_camera_track_legacy_load_distortion_default_zero(tmp_path: Path):
    """Older saved tracks (no ``distortion`` key) load with (0, 0) default."""
    out = tmp_path / "legacy.json"
    out.write_text(
        json.dumps(
            {
                "clip_id": "legacy",
                "fps": 25.0,
                "image_size": [1920, 1080],
                "t_world": [0.0, 0.0, 30.0],
                "frames": [],
            }
        )
    )
    loaded = CameraTrack.load(out)
    assert loaded.distortion == (0.0, 0.0)
