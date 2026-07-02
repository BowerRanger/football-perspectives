"""Phase-2 schema additions: BallAnchor.landmark + BallAnchorSet.shot_chains."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.schemas.ball_anchor import BallAnchor, BallAnchorSet


def _write(tmp_path: Path, payload: dict) -> Path:
    p = tmp_path / "anchors.json"
    p.write_text(json.dumps(payload))
    return p


def _base(**extra) -> dict:
    payload = {
        "clip_id": "play", "image_size": [1280, 720],
        "anchors": [{"frame": 5, "image_xy": [10.0, 20.0], "state": "grounded"}],
    }
    payload.update(extra)
    return payload


def test_landmark_roundtrip_on_grounded(tmp_path: Path):
    payload = _base()
    payload["anchors"][0]["landmark"] = "left_goal_left_post_base"
    aset = BallAnchorSet.load(_write(tmp_path, payload))
    assert aset.anchors[0].landmark == "left_goal_left_post_base"
    out = tmp_path / "roundtrip.json"
    aset.save(out)
    assert json.loads(out.read_text())["anchors"][0]["landmark"] == \
        "left_goal_left_post_base"


def test_landmark_line_prefix_accepted(tmp_path: Path):
    from src.utils.pitch_lines_catalogue import LINE_CATALOGUE
    line_name = sorted(LINE_CATALOGUE)[0]
    payload = _base()
    payload["anchors"][0]["landmark"] = f"line:{line_name}"
    aset = BallAnchorSet.load(_write(tmp_path, payload))
    assert aset.anchors[0].landmark == f"line:{line_name}"


def test_landmark_rejected_on_non_grounded(tmp_path: Path):
    payload = _base()
    payload["anchors"][0]["state"] = "airborne_low"
    payload["anchors"][0]["landmark"] = "left_goal_left_post_base"
    with pytest.raises(ValueError, match="landmark"):
        BallAnchorSet.load(_write(tmp_path, payload))


def test_unknown_landmark_rejected(tmp_path: Path):
    payload = _base()
    payload["anchors"][0]["landmark"] = "no_such_feature"
    with pytest.raises(ValueError, match="landmark"):
        BallAnchorSet.load(_write(tmp_path, payload))


def test_shot_chains_roundtrip(tmp_path: Path):
    payload = _base(shot_chains=[[10, 34], [50, 61, 70]])
    aset = BallAnchorSet.load(_write(tmp_path, payload))
    assert aset.shot_chains == ((10, 34), (50, 61, 70))
    out = tmp_path / "roundtrip.json"
    aset.save(out)
    assert json.loads(out.read_text())["shot_chains"] == [[10, 34], [50, 61, 70]]


def test_shot_chain_must_be_ascending_and_len2(tmp_path: Path):
    with pytest.raises(ValueError, match="shot_chain"):
        BallAnchorSet.load(_write(tmp_path, _base(shot_chains=[[34, 10]])))
    with pytest.raises(ValueError, match="shot_chain"):
        BallAnchorSet.load(_write(tmp_path, _base(shot_chains=[[10]])))


def test_legacy_payload_without_new_fields_loads(tmp_path: Path):
    aset = BallAnchorSet.load(_write(tmp_path, _base()))
    assert aset.anchors[0].landmark is None
    assert aset.shot_chains == ()


def test_default_construction_unchanged():
    a = BallAnchor(frame=1, image_xy=(1.0, 2.0), state="grounded")
    assert a.landmark is None
    s = BallAnchorSet(clip_id="c", image_size=(1280, 720), anchors=(a,))
    assert s.shot_chains == ()
