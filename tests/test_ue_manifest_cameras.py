from __future__ import annotations

from src.schemas.ue_manifest import (
    CameraEntry,
    NamedCameraEntry,
    PitchInfo,
    PlayerEntry,
    UeManifest,
    UeManifestError,
    WorldBBox,
    SCHEMA_VERSION,
)


def _manifest_with_cameras() -> UeManifest:
    return UeManifest(
        schema_version=SCHEMA_VERSION,
        clip_name="clip",
        fps=30.0,
        frame_range=(0, 10),
        pitch=PitchInfo(length_m=105.0, width_m=68.0),
        players=[
            PlayerEntry(
                player_id="P001",
                fbx="fbx/P001.fbx",
                frame_range=(0, 10),
                world_bbox=WorldBBox(min=(0, 0, 0), max=(1, 1, 1)),
            )
        ],
        camera=CameraEntry(
            fbx="fbx/camera.fbx", image_size=(1920, 1080), frame_range=(0, 10),
            track_json="camera/camera_track.json",
        ),
        cameras=[
            NamedCameraEntry(
                name="broadcast", fbx="fbx/camera.fbx", image_size=(1920, 1080),
                frame_range=(0, 10), track_json="camera/camera_track.json",
            ),
            NamedCameraEntry(
                name="P001_pov", fbx="", image_size=(1920, 1080),
                frame_range=(0, 10), track_json="camera/shot_01_P001_pov_camera_track.json",
            ),
        ],
    )


def test_cameras_round_trip(tmp_path) -> None:
    m = _manifest_with_cameras()
    path = tmp_path / "ue_manifest.json"
    m.save(path)
    loaded = UeManifest.load(path)
    assert [c.name for c in loaded.cameras] == ["broadcast", "P001_pov"]
    assert loaded.cameras[1].track_json.endswith("P001_pov_camera_track.json")


def test_cameras_optional_for_backwards_compat(tmp_path) -> None:
    m = _manifest_with_cameras()
    m.cameras = []  # legacy manifest without the new field
    path = tmp_path / "ue_manifest.json"
    m.save(path)
    loaded = UeManifest.load(path)
    assert loaded.cameras == []


def test_validate_rejects_camera_with_inverted_frame_range(tmp_path) -> None:
    import pytest
    from src.schemas.ue_manifest import NamedCameraEntry, UeManifestError
    m = _manifest_with_cameras()
    m.cameras[1] = NamedCameraEntry(
        name="bad", fbx="", image_size=(1920, 1080), frame_range=(10, 0),
        track_json="camera/bad.json",
    )
    with pytest.raises(UeManifestError):
        m.save(tmp_path / "m.json")  # save() calls validate()


def test_named_camera_omits_empty_track_json(tmp_path) -> None:
    import json
    from src.schemas.ue_manifest import NamedCameraEntry
    m = _manifest_with_cameras()
    m.cameras = [NamedCameraEntry(
        name="P002_pov", fbx="fbx/x.fbx", image_size=(1920, 1080),
        frame_range=(0, 10), track_json="",
    )]
    path = tmp_path / "m.json"
    m.save(path)
    raw = json.loads(path.read_text())
    assert "track_json" not in raw["cameras"][0]
