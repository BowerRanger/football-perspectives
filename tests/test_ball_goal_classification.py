"""Goal classification: only an actual goal (operator back_net / mouth
goal_impact, or an explicit ``ball.goal.outcome: goal``) runs the line-cross /
spin / net-containment / goal_check path; woodwork, side-net, auto-only and
``no_goal`` shots do not."""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from src.utils.ball_goal_constraint import (
    classify_goal, find_goal_event, infer_line_cross_knots, normalize_outcome,
    outcome_for_shot)
from src.utils.ball_hybrid_gating import gating_cfg
from src.utils.ball_hybrid_trajectory import run_trajectory
from tests.test_ball_goal_constraint import (
    FIXTURE, _anchor, _load_finish, _look_at_ctx, _pixel_of)


@pytest.fixture()
def ctx():
    return _look_at_ctx((25.0, -25.0, 20.0), (0.0, 34.0, 1.0))


PT = {"back_net": (-1.5, 35.8, 1.8), "mouth": (0.0, 35.0, 1.0),
      "post": (0.0, 30.34, 1.2), "crossbar": (0.0, 34.0, 2.44),
      "side_net": (-0.8, 30.3, 1.0)}


def _anchors(ctx, element):
    return [_anchor(394, "airborne_low", _pixel_of(ctx, (0.0, 36.0, 1.5))),
            _anchor(402, "goal_impact", _pixel_of(ctx, PT[element]), element)]


@pytest.mark.parametrize("element", ["back_net", "mouth"])
def test_net_and_mouth_are_goals(ctx, element):
    assert classify_goal(ctx, _anchors(ctx, element)) is not None


@pytest.mark.parametrize("element", ["post", "crossbar", "side_net"])
def test_woodwork_and_side_net_are_not_goals(ctx, element):
    anchors = _anchors(ctx, element)
    assert find_goal_event(ctx, anchors) is None
    knots, event = infer_line_cross_knots(ctx, anchors)
    assert knots == [] and event is None


@pytest.mark.parametrize("element", ["post", "side_net"])
def test_outcome_goal_forces_goal_on_a_non_net_hit(ctx, element):
    event = find_goal_event(ctx, _anchors(ctx, element), "goal")
    assert event is not None and event.frame == 402


def test_outcome_no_goal_disables_even_with_back_net(ctx):
    anchors = _anchors(ctx, "back_net")
    assert find_goal_event(ctx, anchors, "no_goal") is None
    assert infer_line_cross_knots(ctx, anchors, outcome="no_goal") == ([], None)


def test_outcome_validation_and_lookup():
    assert normalize_outcome(None) is None and normalize_outcome("GOAL") == "goal"
    with pytest.raises(ValueError):
        normalize_outcome("maybe")
    cfg = {"goal": {"outcome": {"s1": "goal", "s2": "no_goal"}}}
    assert outcome_for_shot(cfg, "s1") == "goal"
    assert outcome_for_shot(cfg, "s2") == "no_goal"
    assert outcome_for_shot(cfg, "s3") is None
    assert outcome_for_shot({}, "s1") is None


# ---------------------------------------------------------------- hybrid layer

def _run(anchors_mut=None, outcome=None):
    d, ctx, obs, players = _load_finish()
    anchors = [dict(a) for a in d["manual_anchors"]]
    if anchors_mut:
        anchors_mut(anchors)
    cfg = {"spin": {"enabled": False, "shot_spans": True, "shot_max_accel_m_s2": 10.0}}
    return run_trajectory(
        ctx, obs, anchors, auto_anchors=d["auto_anchors"], cfg=cfg,
        gating_cfg=gating_cfg({}), player_context=players, goal_outcome=outcome)


def _set_element(element):
    def mut(anchors):
        for a in anchors:
            if a["state"] == "goal_impact":
                a["goal_element"] = element
    return mut


@pytest.mark.skipif(not FIXTURE.exists(), reason="gberch_finish fixture missing")
def test_gberch_finish_baseline_is_a_goal():
    _, diag = _run()
    assert diag["goal_check"]["status"] == "ok"
    assert diag["goal_check"]["knot_source"].startswith("operator_airborne_ray")


@pytest.mark.skipif(not FIXTURE.exists(), reason="gberch_finish fixture missing")
@pytest.mark.parametrize("element", ["post", "side_net"])
def test_non_goal_hit_has_no_goal_path(element):
    _, diag = _run(_set_element(element))
    assert "goal_check" not in diag  # no goal_check => no quality-report failure
    assert not any(s.get("rad_s") for s in diag["spans"])  # no shot-span spin


@pytest.mark.skipif(not FIXTURE.exists(), reason="gberch_finish fixture missing")
def test_outcome_goal_on_post_hit_runs_goal_path():
    _, diag = _run(_set_element("post"), outcome="goal")
    assert "goal_check" in diag


@pytest.mark.skipif(not FIXTURE.exists(), reason="gberch_finish fixture missing")
def test_outcome_no_goal_on_back_net_disables_goal_path():
    _, diag = _run(outcome="no_goal")
    assert "goal_check" not in diag
    assert not any(s.get("rad_s") for s in diag["spans"])


@pytest.mark.skipif(not FIXTURE.exists(), reason="gberch_finish fixture missing")
def test_auto_goal_impact_alone_is_not_a_goal():
    """No operator goal_impact at all: an accepted auto goal_impact knot must
    not create a goal (it did before the classification)."""
    def drop_goal_impact(anchors):
        anchors[:] = [a for a in anchors if a["state"] != "goal_impact"]
    _, diag = _run(drop_goal_impact)
    assert "goal_check" not in diag


# ------------------------------------------------------------------ stage seam

def test_stage_goal_check_helper_respects_classification(ctx):
    from src.schemas.ball_anchor import BallAnchor
    from src.stages.ball import _goal_check_for_shot

    art = SimpleNamespace(
        camera_clip_id="t", camera_fps=30.0, camera_image_size=(1280, 720),
        per_frame_K=ctx.per_frame_K, per_frame_R=ctx.per_frame_R,
        per_frame_t=ctx.per_frame_t, distortion=ctx.distortion)
    world = {f: (np.array([float(402 - f) * 0.5 - 1.5, 36.0, 1.5]), 1.0)
             for f in range(380, 410)}
    state = {f: "flight" for f in world}

    def anc(element):
        return {402: BallAnchor(frame=402, image_xy=_pixel_of(ctx, PT[element]),
                                state="goal_impact", goal_element=element)}
    assert _goal_check_for_shot(art, anc("back_net"), world, state, None) is not None
    assert _goal_check_for_shot(art, anc("post"), world, state, None) is None
    assert _goal_check_for_shot(art, anc("side_net"), world, state, None) is None
    assert _goal_check_for_shot(art, anc("post"), world, state, None,
                                outcome="goal") is not None
    assert _goal_check_for_shot(art, anc("back_net"), world, state, None,
                                outcome="no_goal") is None
    # a stale hybrid prior never resurrects a no_goal shot
    assert _goal_check_for_shot(
        art, {}, world, state,
        {"goal_check": {"goal_frame": 402, "goal_end_x": 0.0}},
        outcome="no_goal") is None


# --------------------------------------------------------------------- shorts

def test_diag_is_goal():
    from src.utils.shorts_moments import diag_is_goal
    assert diag_is_goal({"goal_check": {"status": "ok"}})
    assert diag_is_goal({"goal": {"is_goal": True}})
    assert not diag_is_goal({"goal": {"is_goal": False}})
    assert not diag_is_goal({}) and not diag_is_goal(None)


def test_shorts_auto_selection_skips_non_goal_shot(tmp_path, monkeypatch, caplog):
    from src.stages import shorts as stage_mod
    from src.stages.shorts import ShortsStage

    ball = tmp_path / "ball"
    ball.mkdir()
    for sid, diag in (("goal1", {"goal_check": {"status": "ok"}}),
                      ("post1", {"goal": {"is_goal": False}})):
        (ball / f"{sid}_ball_track.json").write_text("{}")
        (ball / f"{sid}_ball_diag.json").write_text(json.dumps(diag))
    monkeypatch.setattr(stage_mod, "derive_moments",
                        lambda out, shot: {"impact": 402, "strike": 371})
    with caplog.at_level("INFO"):
        stage = ShortsStage({"shorts": {"shot": "auto"}}, tmp_path)
        assert stage._target_shots() == ["goal1"]
    assert any("skip post1" in r.message for r in caplog.records)
    # explicit operator pin still works
    stage = ShortsStage({"shorts": {"shot": "post1"}}, tmp_path)
    assert stage._target_shots() == ["post1"]
