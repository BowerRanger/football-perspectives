"""Ball Studio truth schema: validation, defaults, persistence."""

from __future__ import annotations

import copy
import json

import pytest

from src.schemas import ball_truth as bt


def base() -> dict:
    d = bt.empty_truth("g", "a", 30.0, [("a", 0), ("b", -10)])
    d["keys"] = [
        {"id": "k0", "frame": 5, "xyz": [1, 2, 0.11], "source": "triangulated",
         "observations": [
             {"shot_id": "a", "shot_frame": 5, "uv": [10, 20]},
             {"shot_id": "b", "shot_frame": -5, "uv": [30, 40]}]},
        {"id": "k1", "frame": 25, "xyz": [5, 2, 0.11], "source": "ray_ground",
         "observations": [{"shot_id": "a", "shot_frame": 25, "uv": [11, 21]}]},
    ]
    return d


def test_empty_skeleton_is_valid():
    norm, errs = bt.validate_truth(bt.empty_truth("g", "a", 30, [("a", 0)]))
    assert errs == [] and norm["meta"]["status"] == "draft"


def test_valid_document_normalised_with_defaults():
    norm, errs = bt.validate_truth(base(), group_id="g", member_shots=["a", "b"])
    assert errs == []
    k = norm["keys"][0]
    assert k["constraint"]["height_m"] is None and k["note"] == ""
    assert norm["keys"][1]["source"] == "ray_ground"


def test_collects_all_errors():
    d = base()
    d["version"] = 2
    d["outcome"] = "maybe"
    d["keys"][1]["frame"] = 5
    d["bogus"] = 1
    _, errs = bt.validate_truth(d)
    paths = {e["path"] for e in errs}
    assert {"version", "outcome", "keys[1].frame", "bogus"} <= paths


def test_group_mismatch_and_non_member_shot():
    _, errs = bt.validate_truth(base(), group_id="other", member_shots=["a"])
    paths = {e["path"] for e in errs}
    assert "group_id" in paths and "shots[1].shot_id" in paths


def test_triangulated_needs_two_shots():
    d = base()
    d["keys"][0]["observations"] = d["keys"][0]["observations"][:1]
    _, errs = bt.validate_truth(d)
    assert any("triangulated" in e["message"] for e in errs)


def test_observation_must_match_reference_instant():
    d = base()
    d["keys"][0]["observations"][1]["shot_frame"] = 0  # maps to ref 10, key is 5
    _, errs = bt.validate_truth(d)
    assert any("reference frame" in e["message"] for e in errs)


@pytest.mark.parametrize("src,constraint", [
    ("ray_height", {}), ("ray_plane", {}), ("ray_depth", {}), ("player", {}),
])
def test_constraint_requirements(src, constraint):
    d = base()
    d["keys"][1]["source"] = src
    d["keys"][1]["constraint"] = constraint
    _, errs = bt.validate_truth(d)
    assert errs


def test_player_key_without_observation_is_valid():
    d = base()
    d["keys"][1] = {"id": "k1", "frame": 25, "xyz": [0, 0, 0], "source": "player",
                    "constraint": {"player_id": "P1", "bone": "r_foot"}}
    assert bt.validate_truth(d)[1] == []


def test_segment_rules():
    d = base()
    d["segments"] = [{"from": "k1", "to": "k0", "kind": "roll"}]
    assert bt.validate_truth(d)[1]
    d["segments"] = [{"from": "k0", "to": "k1", "kind": "teleport"}]
    assert bt.validate_truth(d)[1]
    d["segments"] = [{"from": "k0", "to": "k1", "kind": "flight",
                      "params": {"magnus": "off", "cd": 0.3}}]
    norm, errs = bt.validate_truth(d)
    assert errs == [] and norm["segments"][0]["params"]["drag"] is True


def test_events_and_meta_status():
    d = base()
    d["events"] = [{"frame": 5, "kind": "touch", "player_id": "P1", "bone": "r_foot"}]
    d["meta"] = {"status": "reviewed"}
    norm, errs = bt.validate_truth(d)
    assert errs == [] and norm["meta"]["status"] == "reviewed"
    d["events"][0]["kind"] = "hug"
    d["meta"]["status"] = "final"
    paths = {e["path"] for e in bt.validate_truth(d)[1]}
    assert {"events[0].kind", "meta.status"} <= paths


def test_non_finite_and_bad_types_rejected():
    d = base()
    d["keys"][0]["xyz"] = [float("nan"), 0, 0]
    d["fps"] = -1
    paths = {e["path"] for e in bt.validate_truth(d)[1]}
    assert {"keys[0].xyz", "fps"} <= paths
    assert bt.validate_truth([])[1]


def test_require_valid_raises():
    d = copy.deepcopy(base())
    d["version"] = 9
    with pytest.raises(bt.BallTruthError) as e:
        bt.require_valid(d)
    assert e.value.errors


def test_save_atomic_with_history(tmp_path):
    doc = bt.require_valid(base())
    assert bt.save_truth(tmp_path, "g", doc) is None
    doc["outcome"] = "goal"
    hist = bt.save_truth(tmp_path, "g", doc)
    assert hist and (tmp_path / "ball_truth" / hist).exists()
    assert bt.load_truth(tmp_path, "g")["outcome"] == "goal"
    assert json.loads((tmp_path / "ball_truth" / hist).read_text())["outcome"] == "unknown"
    assert not list((tmp_path / "ball_truth").glob("*.tmp"))
    bt.save_dense(tmp_path, "g", {"x": 1})
    assert bt.load_dense(tmp_path, "g") == {"x": 1}
