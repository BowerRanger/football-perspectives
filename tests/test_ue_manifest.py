"""Tests for the UE manifest schema."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.schemas.ue_manifest import (
    BallEntry,
    CameraEntry,
    PitchInfo,
    PlayerEntry,
    UeManifest,
    UeManifestError,
    WorldBBox,
)


def _good() -> UeManifest:
    return UeManifest(
        schema_version=1,
        clip_name="clip_demo",
        fps=30.0,
        frame_range=(0, 149),
        pitch=PitchInfo(length_m=105.0, width_m=68.0),
        players=[
            PlayerEntry(
                player_id="P001",
                fbx="fbx/P001.fbx",
                frame_range=(0, 149),
                world_bbox=WorldBBox(
                    min=(-12.3, -8.1, 0.0),
                    max=(10.8, 6.5, 1.95),
                ),
            ),
        ],
        ball=BallEntry(fbx="fbx/ball.fbx", frame_range=(12, 78)),
        camera=CameraEntry(
            fbx="fbx/camera.fbx",
            image_size=(1920, 1080),
            frame_range=(0, 149),
        ),
    )


def test_round_trip_to_disk(tmp_path: Path) -> None:
    m = _good()
    p = tmp_path / "ue_manifest.json"
    m.save(p)
    loaded = UeManifest.load(p)
    assert loaded == m


def test_optional_ball_and_camera_omitted(tmp_path: Path) -> None:
    m = _good()
    m.ball = None
    m.camera = None
    p = tmp_path / "ue_manifest.json"
    m.save(p)
    raw = json.loads(p.read_text())
    assert "ball" not in raw
    assert "camera" not in raw
    loaded = UeManifest.load(p)
    assert loaded.ball is None
    assert loaded.camera is None


def test_rejects_unknown_schema_version(tmp_path: Path) -> None:
    p = tmp_path / "ue_manifest.json"
    p.write_text(json.dumps({"schema_version": 99, "clip_name": "x"}))
    with pytest.raises(UeManifestError, match="schema_version"):
        UeManifest.load(p)


def test_rejects_empty_players() -> None:
    with pytest.raises(UeManifestError, match="players"):
        UeManifest(
            schema_version=1,
            clip_name="x",
            fps=30.0,
            frame_range=(0, 1),
            pitch=PitchInfo(length_m=105.0, width_m=68.0),
            players=[],
        ).validate()


def test_rejects_non_finite_fps() -> None:
    bad = _good()
    bad.fps = float("nan")
    with pytest.raises(UeManifestError, match="fps"):
        bad.validate()


def test_rejects_inverted_frame_range() -> None:
    bad = _good()
    bad.frame_range = (10, 5)
    with pytest.raises(UeManifestError, match="frame_range"):
        bad.validate()


def test_display_name_defaults_to_player_id() -> None:
    p = PlayerEntry(
        player_id="P001",
        fbx="fbx/P001.fbx",
        frame_range=(0, 1),
        world_bbox=WorldBBox(min=(0.0, 0.0, 0.0), max=(1.0, 1.0, 1.0)),
    )
    assert p.display_name == "P001"


def test_display_name_round_trips(tmp_path: Path) -> None:
    m = _good()
    m.players[0].display_name = "Bellingham"
    p = tmp_path / "ue_manifest.json"
    m.save(p)
    raw = json.loads(p.read_text())
    assert raw["players"][0]["display_name"] == "Bellingham"
    loaded = UeManifest.load(p)
    assert loaded.players[0].display_name == "Bellingham"
