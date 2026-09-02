from __future__ import annotations

import pytest

from src.schemas.render_selection import RenderSelection, RenderSelectionError


@pytest.mark.unit
def test_round_trip(tmp_path) -> None:
    sel = RenderSelection(
        shot_id="shot_01",
        cameras=("broadcast", "drone", "pov:P003", "ots:P012"),
        vertical_variant=True,
    )
    path = tmp_path / "shot_01_render_selection.json"
    sel.save(path)
    loaded = RenderSelection.load(path)
    assert loaded == sel


@pytest.mark.unit
def test_round_trip_with_none_vertical_variant(tmp_path) -> None:
    sel = RenderSelection(shot_id="shot_01", cameras=("broadcast",), vertical_variant=None)
    path = tmp_path / "shot_01_render_selection.json"
    sel.save(path)
    loaded = RenderSelection.load(path)
    assert loaded.vertical_variant is None


@pytest.mark.unit
def test_from_dict_accepts_broadcast_drone_pov_ots() -> None:
    sel = RenderSelection.from_dict(
        {"shot_id": "s", "cameras": ["broadcast", "drone", "pov:P001", "ots:P002"]}
    )
    assert sel.cameras == ("broadcast", "drone", "pov:P001", "ots:P002")


@pytest.mark.unit
def test_from_dict_rejects_unknown_camera_id() -> None:
    with pytest.raises(RenderSelectionError):
        RenderSelection.from_dict({"shot_id": "s", "cameras": ["dolly"]})


@pytest.mark.unit
def test_from_dict_rejects_malformed_pov_id() -> None:
    with pytest.raises(RenderSelectionError):
        RenderSelection.from_dict({"shot_id": "s", "cameras": ["pov:"]})


@pytest.mark.unit
def test_from_dict_rejects_non_list_cameras() -> None:
    with pytest.raises(RenderSelectionError):
        RenderSelection.from_dict({"shot_id": "s", "cameras": "broadcast"})


@pytest.mark.unit
def test_from_dict_rejects_non_bool_vertical_variant() -> None:
    with pytest.raises(RenderSelectionError):
        RenderSelection.from_dict(
            {"shot_id": "s", "cameras": [], "vertical_variant": "yes"}
        )


@pytest.mark.unit
def test_from_dict_defaults() -> None:
    sel = RenderSelection.from_dict({"shot_id": "s"})
    assert sel.cameras == ()
    assert sel.vertical_variant is None


@pytest.mark.unit
def test_empty_factory_returns_defaults() -> None:
    sel = RenderSelection.empty("shot_01")
    assert sel.shot_id == "shot_01"
    assert sel.cameras == ()
    assert sel.vertical_variant is None


@pytest.mark.unit
def test_empty_shot_id_allowed_for_legacy_no_manifest_layout() -> None:
    # Mirrors RenderStage._active_shot_ids' "" legacy sentinel (no manifest
    # on disk) — must NOT raise, unlike CameraSelection which requires a
    # non-empty shot_id.
    sel = RenderSelection.from_dict({"shot_id": "", "cameras": ["broadcast"]})
    assert sel.shot_id == ""


@pytest.mark.unit
def test_load_missing_path_raises(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        RenderSelection.load(tmp_path / "does_not_exist.json")
