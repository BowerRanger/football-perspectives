"""dismissed_auto: schema round-trip, merge suppression, stage plumbing."""

from __future__ import annotations

import json
from pathlib import Path

from src.schemas.ball_anchor import BallAnchor, BallAnchorSet, DismissedAuto
from src.utils.ball_auto_anchor import merge_anchors


def test_schema_roundtrip(tmp_path: Path):
    payload = {
        "clip_id": "play", "image_size": [1280, 720],
        "anchors": [{"frame": 5, "image_xy": [1.0, 2.0], "state": "grounded"}],
        "dismissed_auto": [
            {"frame": 20, "state": "player_touch",
             "player_id": "P003", "bone": "l_foot"},
            {"frame": 33, "state": "bounce"},
        ],
    }
    p = tmp_path / "a.json"
    p.write_text(json.dumps(payload))
    aset = BallAnchorSet.load(p)
    assert aset.dismissed_auto == (
        DismissedAuto(frame=20, state="player_touch",
                      player_id="P003", bone="l_foot"),
        DismissedAuto(frame=33, state="bounce"),
    )
    out = tmp_path / "b.json"
    aset.save(out)
    assert json.loads(out.read_text())["dismissed_auto"][0]["frame"] == 20


def test_legacy_payload_defaults_empty(tmp_path: Path):
    p = tmp_path / "a.json"
    p.write_text(json.dumps({
        "clip_id": "play", "image_size": [1280, 720], "anchors": []}))
    assert BallAnchorSet.load(p).dismissed_auto == ()


def _auto(frame: int, state: str = "player_touch",
          player_id: str | None = "P003",
          bone: str | None = "l_foot") -> BallAnchor:
    return BallAnchor(frame=frame, image_xy=(10.0, 10.0), state=state,
                      player_id=player_id, bone=bone, confidence=0.5)


def test_merge_drops_exactly_matching_dismissal():
    auto = {20: _auto(20), 30: _auto(30)}
    merged = merge_anchors(
        {}, auto, 3,
        dismissed=(DismissedAuto(frame=20, state="player_touch",
                                 player_id="P003", bone="l_foot"),),
    )
    assert set(merged) == {30}


def test_merge_partial_match_is_inert():
    auto = {20: _auto(20)}
    merged = merge_anchors(
        {}, auto, 3,
        dismissed=(
            DismissedAuto(frame=20, state="player_touch",
                          player_id="P003", bone="r_foot"),  # wrong bone
            DismissedAuto(frame=21, state="player_touch",
                          player_id="P003", bone="l_foot"),  # wrong frame
        ),
    )
    assert set(merged) == {20}


def test_merge_default_no_dismissals_unchanged():
    manual = {10: _auto(10, state="grounded", player_id=None, bone=None)}
    auto = {11: _auto(11), 30: _auto(30)}
    merged = merge_anchors(manual, auto, 3)
    # 11 suppressed by radius as before, 30 kept.
    assert set(merged) == {10, 30}
