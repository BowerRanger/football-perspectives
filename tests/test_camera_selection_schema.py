from __future__ import annotations

import pytest

from src.schemas.camera_selection import (
    CameraSelection,
    CameraSelectionError,
    RigSelection,
)


@pytest.mark.unit
def test_round_trip(tmp_path) -> None:
    sel = CameraSelection(
        shot_id="shot_01",
        selections=(
            RigSelection(player_id="P003", rigs=("pov", "ots")),
            RigSelection(player_id="P012", rigs=("pov",)),
        ),
    )
    path = tmp_path / "shot_01_camera_selection.json"
    sel.save(path)
    loaded = CameraSelection.load(path)
    assert loaded == sel


@pytest.mark.unit
def test_from_dict_rejects_unknown_rig() -> None:
    with pytest.raises(CameraSelectionError):
        CameraSelection.from_dict(
            {"shot_id": "shot_01", "selections": [{"player_id": "P003", "rigs": ["dolly"]}]}
        )


@pytest.mark.unit
def test_from_dict_dedupes_and_orders_rigs() -> None:
    sel = CameraSelection.from_dict(
        {"shot_id": "s", "selections": [{"player_id": "P1", "rigs": ["ots", "pov", "ots"]}]}
    )
    assert sel.selections[0].rigs == ("pov", "ots")


@pytest.mark.unit
def test_empty_factory_returns_defaults() -> None:
    sel = CameraSelection.empty("shot_01")
    assert sel.shot_id == "shot_01"
    assert sel.selections == ()


@pytest.mark.unit
def test_load_missing_path_raises(tmp_path) -> None:
    import pytest
    with pytest.raises(FileNotFoundError):
        CameraSelection.load(tmp_path / "does_not_exist.json")
