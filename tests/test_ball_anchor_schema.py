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


def test_goal_impact_round_trip_preserves_goal_element(tmp_path: Path):
    aset = BallAnchorSet(
        clip_id="play",
        image_size=(1280, 720),
        anchors=(
            BallAnchor(
                frame=42, image_xy=(640.0, 200.0),
                state="goal_impact", goal_element="crossbar",
            ),
            BallAnchor(
                frame=58, image_xy=(91.0, 393.0),
                state="goal_impact", goal_element="post",
            ),
        ),
    )
    p = tmp_path / "anchors.json"
    aset.save(p)
    loaded = BallAnchorSet.load(p)
    assert len(loaded.anchors) == 2
    assert loaded.anchors[0].state == "goal_impact"
    assert loaded.anchors[0].goal_element == "crossbar"
    assert loaded.anchors[1].goal_element == "post"


def test_goal_impact_requires_goal_element(tmp_path: Path):
    p = tmp_path / "anchors.json"
    p.write_text(json.dumps({
        "clip_id": "x", "image_size": [1280, 720],
        "anchors": [
            {"frame": 1, "image_xy": [10, 20], "state": "goal_impact"},
        ],
    }))
    with pytest.raises(ValueError, match="goal_element is required"):
        BallAnchorSet.load(p)


def test_goal_impact_rejects_unknown_goal_element(tmp_path: Path):
    p = tmp_path / "anchors.json"
    p.write_text(json.dumps({
        "clip_id": "x", "image_size": [1280, 720],
        "anchors": [
            {"frame": 1, "image_xy": [10, 20],
             "state": "goal_impact", "goal_element": "bar"},
        ],
    }))
    with pytest.raises(ValueError, match="unknown goal_element"):
        BallAnchorSet.load(p)


def test_goal_impact_requires_image_xy(tmp_path: Path):
    p = tmp_path / "anchors.json"
    p.write_text(json.dumps({
        "clip_id": "x", "image_size": [1280, 720],
        "anchors": [
            {"frame": 1, "image_xy": None,
             "state": "goal_impact", "goal_element": "post"},
        ],
    }))
    with pytest.raises(ValueError, match="image_xy is required"):
        BallAnchorSet.load(p)


def test_touch_type_round_trip_on_player_touch(tmp_path: Path):
    aset = BallAnchorSet(
        clip_id="play",
        image_size=(1280, 720),
        anchors=(
            BallAnchor(
                frame=5, image_xy=(640.0, 360.0),
                state="player_touch", player_id="P003", bone="r_foot",
                touch_type="shot", spin="instep_curl_right",
            ),
            BallAnchor(
                frame=20, image_xy=(700.0, 320.0),
                state="player_touch", player_id="P003", bone="r_foot",
                touch_type="volley", spin="topspin",
            ),
        ),
    )
    p = tmp_path / "anchors.json"
    aset.save(p)
    loaded = BallAnchorSet.load(p)
    assert loaded.anchors[0].touch_type == "shot"
    assert loaded.anchors[0].spin == "instep_curl_right"
    assert loaded.anchors[1].touch_type == "volley"
    assert loaded.anchors[1].spin == "topspin"


def test_touch_type_rejected_on_non_player_touch_state(tmp_path: Path):
    p = tmp_path / "anchors.json"
    p.write_text(json.dumps({
        "clip_id": "x", "image_size": [1280, 720],
        "anchors": [
            {"frame": 1, "image_xy": [10, 20],
             "state": "grounded", "touch_type": "shot"},
        ],
    }))
    with pytest.raises(ValueError, match="touch_type is only valid"):
        BallAnchorSet.load(p)


def test_touch_type_rejects_unknown_value(tmp_path: Path):
    p = tmp_path / "anchors.json"
    p.write_text(json.dumps({
        "clip_id": "x", "image_size": [1280, 720],
        "anchors": [
            {"frame": 1, "image_xy": [10, 20], "state": "player_touch",
             "player_id": "P1", "bone": "r_foot", "touch_type": "lob"},
        ],
    }))
    with pytest.raises(ValueError, match="unknown touch_type"):
        BallAnchorSet.load(p)


def test_spin_requires_player_touch_state(tmp_path: Path):
    p = tmp_path / "anchors.json"
    p.write_text(json.dumps({
        "clip_id": "x", "image_size": [1280, 720],
        "anchors": [
            {"frame": 1, "image_xy": [10, 20],
             "state": "grounded", "spin": "topspin"},
        ],
    }))
    with pytest.raises(ValueError, match="spin is only valid on states"):
        BallAnchorSet.load(p)


def test_spin_requires_shot_or_volley_touch_type(tmp_path: Path):
    """Spin on a plain player_touch (no touch_type) is rejected — the
    user has to explicitly mark it as a shot or volley to opt in."""
    p = tmp_path / "anchors.json"
    p.write_text(json.dumps({
        "clip_id": "x", "image_size": [1280, 720],
        "anchors": [
            {"frame": 1, "image_xy": [10, 20], "state": "player_touch",
             "player_id": "P1", "bone": "r_foot", "spin": "topspin"},
        ],
    }))
    with pytest.raises(ValueError, match="spin requires touch_type"):
        BallAnchorSet.load(p)


def test_spin_rejects_unknown_preset(tmp_path: Path):
    p = tmp_path / "anchors.json"
    p.write_text(json.dumps({
        "clip_id": "x", "image_size": [1280, 720],
        "anchors": [
            {"frame": 1, "image_xy": [10, 20], "state": "player_touch",
             "player_id": "P1", "bone": "r_foot",
             "touch_type": "shot", "spin": "flutter"},
        ],
    }))
    with pytest.raises(ValueError, match="unknown spin preset"):
        BallAnchorSet.load(p)


def test_player_touch_without_touch_type_or_spin_is_valid(tmp_path: Path):
    """A bare player_touch (no touch_type, no spin) remains valid —
    most contacts are just a pass / control with no shot semantics."""
    aset = BallAnchorSet(
        clip_id="play", image_size=(1280, 720),
        anchors=(
            BallAnchor(
                frame=5, image_xy=(640.0, 360.0),
                state="player_touch", player_id="P1", bone="r_foot",
            ),
        ),
    )
    p = tmp_path / "anchors.json"
    aset.save(p)
    loaded = BallAnchorSet.load(p)
    assert loaded.anchors[0].touch_type is None
    assert loaded.anchors[0].spin is None


def test_anchor_confidence_and_end_frame_roundtrip(tmp_path: Path):
    """Auto events carry a confidence score and (for span events) an
    end_frame; both must survive a save/load round-trip."""
    aset = BallAnchorSet(
        clip_id="clip", image_size=(1920, 1080),
        anchors=(
            BallAnchor(
                frame=5, image_xy=(10.0, 20.0), state="player_touch",
                player_id="P001", bone="r_foot", confidence=0.42,
            ),
        ),
    )
    p = tmp_path / "a.json"
    aset.save(p)
    back = BallAnchorSet.load(p)
    assert back.anchors[0].confidence == 0.42
    assert back.anchors[0].end_frame is None


def test_anchor_confidence_defaults_to_one_for_legacy(tmp_path: Path):
    """A legacy anchors file with no confidence field loads as 1.0 so
    existing on-disk anchors keep working."""
    p = tmp_path / "legacy.json"
    p.write_text(json.dumps({
        "clip_id": "c", "image_size": [100, 100],
        "anchors": [{"frame": 0, "image_xy": [1, 2], "state": "grounded"}],
    }))
    back = BallAnchorSet.load(p)
    assert back.anchors[0].confidence == 1.0
    assert back.anchors[0].end_frame is None


def test_anchor_confidence_clamped_to_unit_range(tmp_path: Path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({
        "clip_id": "c", "image_size": [100, 100],
        "anchors": [{"frame": 0, "image_xy": [1, 2], "state": "grounded",
                     "confidence": 1.7}],
    }))
    back = BallAnchorSet.load(p)
    assert back.anchors[0].confidence == 1.0
