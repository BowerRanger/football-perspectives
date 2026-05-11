"""JSON round-trip tests for BallAnchor / BallAnchorSet."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.schemas.ball_anchor import BallAnchor, BallAnchorSet


def test_roundtrip_with_pixel_anchors(tmp_path: Path):
    aset = BallAnchorSet(
        clip_id="origi01",
        image_size=(1280, 720),
        anchors=(
            BallAnchor(frame=43, image_xy=(640.0, 360.0), state="grounded"),
            BallAnchor(frame=78, image_xy=(700.0, 200.0), state="airborne_mid"),
            BallAnchor(frame=84, image_xy=(720.0, 340.0), state="kick"),
        ),
    )
    p = tmp_path / "anchors.json"
    aset.save(p)
    loaded = BallAnchorSet.load(p)
    assert loaded.clip_id == "origi01"
    assert loaded.image_size == (1280, 720)
    assert len(loaded.anchors) == 3
    assert loaded.anchors[0].frame == 43
    assert loaded.anchors[0].image_xy == (640.0, 360.0)
    assert loaded.anchors[0].state == "grounded"
    assert loaded.anchors[2].state == "kick"


def test_off_screen_flight_anchor_allows_none_image_xy(tmp_path: Path):
    aset = BallAnchorSet(
        clip_id="origi01",
        image_size=(1280, 720),
        anchors=(
            BallAnchor(frame=120, image_xy=None, state="off_screen_flight"),
        ),
    )
    p = tmp_path / "anchors.json"
    aset.save(p)
    loaded = BallAnchorSet.load(p)
    assert loaded.anchors[0].image_xy is None
    assert loaded.anchors[0].state == "off_screen_flight"


def test_load_missing_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        BallAnchorSet.load(tmp_path / "does_not_exist.json")


def test_invalid_state_rejected(tmp_path: Path):
    # Hand-craft a JSON with an invalid state string.
    p = tmp_path / "anchors.json"
    p.write_text(json.dumps({
        "clip_id": "x", "image_size": [1, 1],
        "anchors": [{"frame": 1, "image_xy": [0, 0], "state": "bogus"}],
    }))
    with pytest.raises(ValueError):
        BallAnchorSet.load(p)


def test_pixel_required_for_non_off_screen_states(tmp_path: Path):
    p = tmp_path / "anchors.json"
    p.write_text(json.dumps({
        "clip_id": "x", "image_size": [1, 1],
        "anchors": [{"frame": 1, "image_xy": None, "state": "grounded"}],
    }))
    with pytest.raises(ValueError):
        BallAnchorSet.load(p)


def test_empty_anchor_set_roundtrips(tmp_path: Path):
    aset = BallAnchorSet(
        clip_id="x", image_size=(640, 480), anchors=(),
    )
    p = tmp_path / "anchors.json"
    aset.save(p)
    loaded = BallAnchorSet.load(p)
    assert loaded.anchors == ()
