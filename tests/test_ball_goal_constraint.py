"""D6.1 goal-mouth constraint: line-cross knot inference, goal_check, net
containment, and the fast gberch-finish fixture (no WASB, no video)."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.utils.ball_goal_constraint import (
    STATUS_MISSES_MOUTH,
    STATUS_NO_LINE_CROSS,
    STATUS_OK,
    STATUS_OVER_CROSSBAR,
    GoalEvent,
    contain_in_net,
    find_goal_event,
    goal_check,
    infer_line_cross_knots,
)
from src.utils.ball_hybrid_gating import gating_cfg
from src.utils.ball_hybrid_trajectory import run_trajectory
from src.utils.ball_hybrid_types import HybridShotCtx

FIXTURE = (Path(__file__).parent / "fixtures" / "ball" / "gberch_finish" / "finish.json")


def _look_at_ctx(cam_pos, target, frames=range(0, 500)) -> HybridShotCtx:
    C = np.asarray(cam_pos, float)
    fwd = np.asarray(target, float) - C
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    R = np.array([right, down, fwd])
    t = -R @ C
    K = np.array([[1500.0, 0, 640.0], [0, 1500.0, 360.0], [0, 0, 1.0]])
    return HybridShotCtx(
        "t", 30.0, (1280, 720),
        {f: K for f in frames}, {f: R for f in frames}, {f: t for f in frames},
        (0.0, 0.0))


def _pixel_of(ctx, xyz, frame=0):
    return tuple(float(v) for v in ctx.project(frame, np.asarray(xyz, float)))


def _anchor(frame, state, xy, element=None):
    return {"frame": frame, "state": state, "image_xy": list(xy),
            "goal_element": element, "player_id": None, "bone": None}


@pytest.fixture()
def ctx():
    # broadcast-style camera behind the near touchline, looking at goal x=0
    return _look_at_ctx((25.0, -25.0, 20.0), (0.0, 34.0, 1.0))


def test_mouth_is_a_valid_goal_element():
    from src.utils.ball_anchor_heights import VALID_GOAL_ELEMENTS
    assert "mouth" in VALID_GOAL_ELEMENTS


def test_infers_knot_from_latest_airborne_anchor_in_mouth(ctx):
    target = (0.0, 36.0, 1.5)
    anchors = [
        _anchor(380, "airborne_low", _pixel_of(ctx, (5.0, 35.0, 1.4))),
        _anchor(394, "airborne_low", _pixel_of(ctx, target)),
        _anchor(402, "goal_impact", _pixel_of(ctx, (-1.5, 35.8, 1.8)), "back_net"),
    ]
    before = copy.deepcopy(anchors)
    knots, event = infer_line_cross_knots(ctx, anchors)
    assert [k.frame for k in knots] == [394]
    k = knots[0]
    assert k.kind == "line_cross" and k.depth_hard and k.source == "auto"
    assert np.allclose(k.xyz, target, atol=0.02)
    assert event.knot_source == "operator_airborne_ray"
    assert event.goal_end_x == 0.0 and event.frame == 402
    assert anchors == before  # operator data never mutated


def test_airborne_anchor_outside_mouth_is_not_promoted(ctx):
    anchors = [
        _anchor(394, "airborne_low", _pixel_of(ctx, (0.0, 44.0, 1.0))),  # wide
        _anchor(402, "goal_impact", _pixel_of(ctx, (-1.5, 35.8, 1.8)), "back_net"),
    ]
    knots, event = infer_line_cross_knots(ctx, anchors)
    assert knots == [] and event.knot_source is None


def test_ray_just_outside_post_snaps_inside_the_mouth(ctx):
    """kroupi01: footage shows the ball inside the near post but the anchor
    ray meets the goal line 0.5 m wide (calibration error) — inside the
    margin, so the knot is snapped in and flagged, not rejected."""
    anchors = [
        _anchor(394, "airborne_low", _pixel_of(ctx, (0.0, 38.17, 1.72))),
        _anchor(402, "goal_impact", _pixel_of(ctx, (-1.5, 35.8, 1.8)), "back_net"),
    ]
    knots, event = infer_line_cross_knots(ctx, anchors)
    (k,) = knots
    assert k.xyz[1] == pytest.approx(37.66 - 0.11, abs=0.02)
    assert k.xyz[2] == pytest.approx(1.72, abs=0.02)
    assert event.knot_source == "operator_airborne_ray_snapped"


def test_anchor_outside_window_is_not_promoted(ctx):
    anchors = [
        _anchor(300, "airborne_low", _pixel_of(ctx, (0.0, 36.0, 1.5))),
        _anchor(402, "goal_impact", _pixel_of(ctx, (-1.5, 35.8, 1.8)), "back_net"),
    ]
    knots, _ = infer_line_cross_knots(ctx, anchors)
    assert knots == []


def test_falls_back_to_earlier_anchor_when_latest_misses(ctx):
    anchors = [
        _anchor(390, "airborne_low", _pixel_of(ctx, (0.0, 35.0, 1.2))),
        _anchor(396, "airborne_low", _pixel_of(ctx, (0.0, 45.0, 1.0))),  # wide
        _anchor(402, "goal_impact", _pixel_of(ctx, (-1.5, 35.8, 1.8)), "back_net"),
    ]
    knots, _ = infer_line_cross_knots(ctx, anchors)
    assert [k.frame for k in knots] == [390]


@pytest.mark.parametrize("element,source", [
    ("post", "goal_impact_on_line"),
    ("crossbar", "goal_impact_on_line"),
    ("mouth", "operator_mouth"),
])
def test_on_line_elements_need_no_inferred_knot(ctx, element, source):
    pt = {"post": (0.0, 30.34, 1.2), "crossbar": (0.0, 34.0, 2.44),
          "mouth": (0.0, 35.0, 1.0)}[element]
    anchors = [
        _anchor(394, "airborne_low", _pixel_of(ctx, (0.0, 36.0, 1.5))),
        _anchor(396, "goal_impact", _pixel_of(ctx, pt), element),
    ]
    knots, event = infer_line_cross_knots(ctx, anchors)
    assert knots == []
    assert event.knot_source == source


def test_no_goal_impact_means_no_event(ctx):
    anchors = [_anchor(394, "airborne_low", _pixel_of(ctx, (0.0, 36.0, 1.5)))]
    knots, event = infer_line_cross_knots(ctx, anchors)
    assert knots == [] and event is None
    assert find_goal_event(ctx, anchors) is None


def test_disabled_flag_skips_inference(ctx):
    anchors = [
        _anchor(394, "airborne_low", _pixel_of(ctx, (0.0, 36.0, 1.5))),
        _anchor(402, "goal_impact", _pixel_of(ctx, (-1.5, 35.8, 1.8)), "back_net"),
    ]
    knots, event = infer_line_cross_knots(ctx, anchors, {"enabled": False})
    assert knots == [] and event is not None


# ---------------------------------------------------------------- goal_check

def _track(points: dict[int, tuple[float, float, float]]) -> dict:
    return {f: {"xyz": p, "state": "flight"} for f, p in points.items()}


def _line_track(y, z, gx=0.0, sign=-1.0):
    """1 m/frame run toward (sign=-1: x falling to the line) the goal line,
    crossing it at frame 15."""
    return {f: (gx + sign * (i - 5) * 1.0, y, z)
            for i, f in enumerate(range(10, 21))}


@pytest.mark.parametrize("y,z,status", [
    (34.5, 1.0, STATUS_OK),
    (38.5, 1.0, STATUS_MISSES_MOUTH),
    (34.0, 3.8, STATUS_OVER_CROSSBAR),
    (38.5, 3.8, STATUS_MISSES_MOUTH),
])
def test_goal_check_statuses_near_goal(y, z, status):
    frames = _track(_line_track(y, z))
    gc = goal_check(frames, GoalEvent(20, 0.0, "back_net"))
    assert gc["status"] == status
    assert gc["goal_end_x"] == 0.0 and gc["goal_frame"] == 20
    assert gc["line_cross"]["frame"] == pytest.approx(15.0, abs=1e-6)
    assert gc["line_cross"]["xyz"][0] == pytest.approx(0.0, abs=1e-6)


def test_goal_check_far_goal():
    pts = {f: (105.0 - (5 - (f - 10)) * 1.0, 34.0, 1.0) for f in range(10, 21)}
    gc = goal_check(_track(pts), GoalEvent(20, 105.0, "back_net"))
    assert gc["status"] == STATUS_OK
    assert gc["line_cross"]["xyz"][0] == pytest.approx(105.0, abs=1e-6)


def test_goal_check_no_line_cross():
    pts = {f: (8.0 - 0.1 * (f - 10), 34.0, 1.0) for f in range(10, 21)}
    gc = goal_check(_track(pts), GoalEvent(20, 0.0, "back_net"))
    assert gc["status"] == STATUS_NO_LINE_CROSS and gc["line_cross"] is None


def test_goal_check_uses_first_crossing_not_net_wobble():
    # crosses at 15 inside the mouth, then wobbles back out and in again
    # (post-line fit garbage) -- the first inward crossing is the line-cross.
    pts = dict(_line_track(34.5, 1.0))
    pts[17] = (1.0, 34.5, 1.0)    # back out ...
    pts[18] = (-0.5, 34.5, 3.9)   # ... and in again, high
    gc = goal_check(_track(pts), GoalEvent(20, 0.0, "back_net"))
    assert gc["status"] == STATUS_OK
    assert gc["line_cross"]["frame"] == pytest.approx(15.0, abs=1e-6)


def test_goal_check_none_without_event():
    assert goal_check(_track({1: (1.0, 2.0, 3.0)}), None) is None


# --------------------------------------------------------- net containment

def test_contain_in_net_clamps_inside_box_and_spares_protected():
    pts = {f: (-3.0, 40.0, 3.0) for f in range(10, 31)}  # far outside the box
    pts[15] = (0.0, 36.0, 1.8)   # line-cross
    pts[24] = (-1.5, 35.8, 1.9)  # impact
    frames = _track(pts)
    ev = GoalEvent(24, 0.0, "back_net")
    gc = {"status": STATUS_OK, "line_cross": {"frame": 15.0, "xyz": [0.0, 36.0, 1.8]}}
    out, n = contain_in_net(frames, ev, gc, protect_frames=[18])
    assert n > 0
    for f in range(16, 24):
        x, y, z = out[f]["xyz"]
        if f == 18:
            assert (x, y, z) == (-3.0, 40.0, 3.0)  # protected anchor untouched
            continue
        assert -1.5 <= x <= 0.0 and 30.34 <= y <= 37.66 and 0.11 <= z <= 2.44
        assert out[f]["state"] == "flight"
    assert out[15]["xyz"] == (0.0, 36.0, 1.8) and out[24]["xyz"] == (-1.5, 35.8, 1.9)
    assert out[29]["xyz"] == (-3.0, 40.0, 3.0)  # outside the window
    assert frames[16]["xyz"] == (-3.0, 40.0, 3.0)  # input not mutated


def test_contain_in_net_noop_unless_ok_or_short_window():
    frames = _track({f: (-3.0, 40.0, 3.0) for f in range(10, 40)})
    ev = GoalEvent(35, 0.0, "back_net")
    bad = {"status": STATUS_MISSES_MOUTH, "line_cross": {"frame": 12.0, "xyz": [0, 40, 3]}}
    assert contain_in_net(frames, ev, bad)[1] == 0
    long_window = {"status": STATUS_OK, "line_cross": {"frame": 12.0, "xyz": [0, 36, 1]}}
    out, n = contain_in_net(frames, ev, long_window)  # 23-frame window: not a net entry
    assert n == 0 and out[20]["xyz"] == (-3.0, 40.0, 3.0)


# ------------------------------------------------ gberch finish (fast fixture)

def _load_finish():
    d = json.loads(FIXTURE.read_text())
    K = {c["frame"]: np.array(c["K"]) for c in d["camera"]}
    R = {c["frame"]: np.array(c["R"]) for c in d["camera"]}
    t = {c["frame"]: np.array(c["t"]) for c in d["camera"]}
    ctx = HybridShotCtx("gberch", d["fps"], tuple(d["image_size"]), K, R, t,
                        tuple(d["distortion"]))
    obs = [SimpleNamespace(frame=o["frame"], uv=tuple(o["uv"]), conf=o["conf"],
                           source=o["source"]) for o in d["observations"]]
    joints = {k: np.array(v) for k, v in d["joints"].items()}

    class _Players:
        def joint_world(self, frame, player_id, bone):
            return joints.get(f"{frame}|{player_id}|{bone}")

    return d, ctx, obs, _Players()


def _run_finish(extra_cfg=None):
    d, ctx, obs, players = _load_finish()
    cfg = {"spin": {"enabled": False, "shot_spans": True, "shot_max_accel_m_s2": 10.0}}
    cfg.update(extra_cfg or {})
    frames, diag = run_trajectory(
        ctx, obs, d["manual_anchors"], auto_anchors=d["auto_anchors"], cfg=cfg,
        gating_cfg=gating_cfg({}), player_context=players)
    return d, ctx, frames, diag


@pytest.mark.skipif(not FIXTURE.exists(), reason="gberch_finish fixture missing")
def test_gberch_finish_line_cross_within_30cm_of_operator_ray():
    d, ctx, frames, diag = _run_finish()
    anchor394 = next(a for a in d["manual_anchors"] if a["frame"] == 394)
    C, ray = ctx.ray(394, tuple(anchor394["image_xy"]))
    operator_pt = C + (0.0 - C[0]) / ray[0] * ray  # operator ray ∩ x=0
    assert operator_pt[1] == pytest.approx(37.09, abs=0.05)
    gc = diag["goal_check"]
    assert gc["status"] == STATUS_OK
    assert gc["knot_source"] == "operator_airborne_ray"
    got = np.array(gc["line_cross"]["xyz"])
    assert np.linalg.norm(got - operator_pt) < 0.30
    assert abs(got[0]) < 1e-3


@pytest.mark.skipif(not FIXTURE.exists(), reason="gberch_finish fixture missing")
def test_gberch_finish_constraint_off_reports_no_goal_check():
    """Control: with the D6 constraint switched off the stage derives no
    goal event from the operator anchors (the pre-D6 behaviour)."""
    d, ctx, frames, diag = _run_finish({"goal": {"enabled": False}})
    assert "goal_check" not in diag


@pytest.mark.skipif(not FIXTURE.exists(), reason="gberch_finish fixture missing")
def test_gberch_finish_shot_span_gets_bounded_curl():
    from src.utils.ball_hybrid_physics import DEFAULT_MAGNUS_COEFF

    d, ctx, frames, diag = _run_finish()
    shot = next(s for s in diag["spans"] if s["span"][0] == 371 and s["model"] == "flight")
    assert shot["rad_s"] > 1.0  # spin accepted on the shot span
    omega = np.array(shot["omega_world"])
    v0 = np.array(shot["v0"])
    accel = DEFAULT_MAGNUS_COEFF * np.linalg.norm(np.cross(omega, v0))
    assert accel <= 10.5  # bounded ~10 m/s^2 curl
    # a plain (non-shot) config leaves the span spin-free
    _, _, _, diag_off = _run_finish({"spin": {"enabled": False, "shot_spans": False}})
    span_off = next(s for s in diag_off["spans"] if s["span"][0] == 371)
    assert "rad_s" not in span_off


@pytest.mark.skipif(not FIXTURE.exists(), reason="gberch_finish fixture missing")
def test_gberch_finish_ball_stays_in_goal_box_after_line_cross():
    d, ctx, frames, diag = _run_finish()
    for f in range(395, 402):
        x, y, z = frames[f]["xyz"]
        assert -1.55 <= x <= 0.05 and 30.3 <= y <= 37.7 and z <= 2.45


# ------------------------------------------------------ stage helper wiring

def test_stage_goal_check_helper_reads_final_track(ctx):
    from src.schemas.ball_anchor import BallAnchor
    from src.stages.ball import _goal_check_for_shot

    art = SimpleNamespace(
        camera_clip_id="t", camera_fps=30.0, camera_image_size=(1280, 720),
        per_frame_K=ctx.per_frame_K, per_frame_R=ctx.per_frame_R,
        per_frame_t=ctx.per_frame_t, distortion=ctx.distortion)
    gi = BallAnchor(frame=402, image_xy=_pixel_of(ctx, (-1.5, 35.8, 1.8)),
                    state="goal_impact", goal_element="back_net")
    world = {f: (np.array([float(402 - f) * 0.5 - 1.5, 36.0, 1.5]), 1.0)
             for f in range(380, 410)}
    state = {f: "flight" for f in world}
    gc = _goal_check_for_shot(art, {402: gi}, world, state,
                              {"goal_check": {"knot_source": "operator_airborne_ray"}})
    assert gc["status"] == STATUS_OK
    assert gc["knot_source"] == "operator_airborne_ray"
    assert _goal_check_for_shot(art, {}, world, state, None) is None
