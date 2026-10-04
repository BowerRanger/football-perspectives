"""shorts sidecar schema: validation, operator-block preservation, pins."""
from __future__ import annotations

import json

import pytest

from src.schemas import shorts as sc
from src.utils import shorts_templates as st

MOMENTS = {"strike": 371, "line_cross": 394, "impact": 402, "keeper_dive": 397, "buildup_start": 296,
           "scorer_pid": "P006", "keeper_pid": "P005", "goal_end": "left",
           "sources": {"strike": "operator_shot_anchor"}}


def _generated():
    resolved = st.resolve_template(st.load_template("keeper"), MOMENTS)
    data = sc.empty_sidecar("gberch")
    data["moments"] = MOMENTS
    data["templates"]["keeper"] = sc.template_entry(resolved, {"mp4": "shorts/gberch_keeper.mp4"})
    return data


def test_roundtrip_and_validation(tmp_path):
    p = sc.sidecar_path(tmp_path, "gberch")
    assert p == tmp_path / "shorts" / "gberch_shorts.json"
    sc.save_sidecar(p, _generated())
    got = sc.load_sidecar(p)
    assert got["templates"]["keeper"]["ok"] is True
    assert got["templates"]["keeper"]["outputs"]["mp4"].endswith("keeper.mp4")
    assert set(got["templates"]["keeper"]["framing"]) == set()      # no framing_check supplied
    assert got["operator"] == sc.empty_operator()


def test_load_absent_is_none_and_corrupt_raises(tmp_path):
    p = tmp_path / "x.json"
    assert sc.load_sidecar(p) is None
    p.write_text("{nope")
    with pytest.raises(ValueError, match="not valid JSON"):
        sc.load_sidecar(p)


def test_operator_block_survives_regeneration():
    existing = _generated()
    existing["operator"] = {"moments": {"strike": 372},
                            "captions": {"keeper": [{"text": "SAVE IT?", "style": "title", "start": 0}]},
                            "templates": ["keeper"], "candidates": {"keeper": {"keeper_eyes": 0}}}
    fresh = _generated()
    fresh["templates"]["keeper"]["ok"] = False
    merged = sc.merge_regenerated(existing, fresh)
    assert merged["operator"] == sc.validate_operator(existing["operator"])
    assert merged["operator"]["captions"]["keeper"][0]["text"] == "SAVE IT?"
    assert merged["templates"]["keeper"]["ok"] is False          # generated part is replaced


def test_moment_pins_win_and_are_recorded():
    eff = sc.effective_moments(MOMENTS, {"moments": {"strike": 372}})
    assert eff["strike"] == 372 and eff["sources"]["strike"] == "operator_pin"
    assert eff["impact"] == 402 and MOMENTS["strike"] == 371          # input not mutated


@pytest.mark.parametrize("bad", [
    {"moments": {"goal": 3}}, {"moments": {"strike": "371"}}, {"moments": {"strike": True}},
    {"captions": {"k": [{"text": ""}]}}, {"captions": {"k": [{"text": "x", "style": "huge"}]}},
    {"templates": "keeper"}, {"candidates": {"k": {"s": -1}}}, {"junk": 1}])
def test_bad_operator_blocks_are_rejected(bad):
    with pytest.raises(ValueError):
        sc.validate_operator(bad)


def test_save_validates_before_writing(tmp_path):
    p = tmp_path / "s.json"
    with pytest.raises(ValueError):
        sc.save_sidecar(p, {"version": 1, "shot": "g", "operator": {"moments": {"x": 1}}})
    assert not p.exists()


def test_operator_pinned_captions_replace_template_captions():
    cap = [{"text": "MINE", "style": "title", "start": 0}]
    r = st.resolve_template(st.load_template("keeper"), MOMENTS, captions_override=cap)
    assert r["edl"]["captions"] == cap
    json.dumps(sc.template_entry(r))     # JSON-serialisable
