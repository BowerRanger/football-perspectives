"""Tests for src/utils/ball_replay_knots.py — fixes -> hybrid Knots, with
the operator-wins and physical-volume drop gates."""

from __future__ import annotations

import numpy as np

from src.schemas.ball_anchor import BallAnchor
from src.schemas.ball_fixes import BallFix
from src.utils.ball_replay_knots import fixes_to_knots
from src.utils.camera_projection import project_world_to_image

# Simple pinhole camera sitting at the world origin, looking down the
# world +Y axis with world +Z (pitch height) as image "up": camera-space
# (x_cam, y_cam, z_cam) = (x_world, -z_world, y_world), so depth is the
# world Y coordinate. Keeps test fixtures in ordinary pitch coordinates
# (x/y horizontal in metres, z a small height) instead of needing world Z
# to double as camera depth, which would fight the court-volume gate's
# z <= 30 height cap.
_K = np.array([[1000.0, 0.0, 960.0],
               [0.0, 1000.0, 540.0],
               [0.0, 0.0, 1.0]])
_R = np.array([[1.0, 0.0, 0.0],
               [0.0, 0.0, -1.0],
               [0.0, 1.0, 0.0]])
_T = np.zeros(3)
_CAM = (_K, _R, _T)


def _uv_for(xyz) -> tuple[float, float]:
    out = project_world_to_image(_K, _R, _T, (0.0, 0.0), np.array([xyz]))
    return (float(out[0, 0]), float(out[0, 1]))


def _anchor(frame: int, xyz, state: str = "airborne_mid") -> BallAnchor:
    return BallAnchor(frame=frame, image_xy=_uv_for(xyz), state=state)


def _fix(frame: int, xyz, partner_frame: int = 900) -> BallFix:
    return BallFix(
        frame=frame, xyz=tuple(float(v) for v in xyz),
        ray_miss_m=0.05, parallax_deg=20.0,
        partner_shot="partnerX", partner_frame=partner_frame,
    )


def test_agreeing_fix_becomes_depth_hard_knot():
    true_xyz = (5.0, 50.0, 1.0)
    anchors = [_anchor(10, true_xyz, state="player_touch")]
    fixes = [_fix(10, true_xyz)]
    cams = {10: _CAM}

    knots, dropped = fixes_to_knots(fixes, anchors, cams, tol_px=5.0)

    assert dropped == []
    assert len(knots) == 1
    k = knots[0]
    assert k.frame == 10
    assert k.source == "fix"
    assert k.kind == "fix"
    assert k.depth_hard is True
    assert np.allclose(k.xyz, true_xyz)


def test_fix_outside_court_volume_dropped_regardless_of_anchor_agreement():
    # Deep underground — the W6 failure signature (globally wrong partner
    # camera). No manual anchor at all: must still be dropped on physical
    # grounds alone.
    bad_xyz = (30.0, 20.0, -8.0)
    fixes = [_fix(20, bad_xyz)]
    cams = {20: _CAM}

    knots, dropped = fixes_to_knots(fixes, [], cams, tol_px=5.0)

    assert knots == []
    assert len(dropped) == 1
    assert dropped[0]["reason"] == "physically_impossible"
    assert dropped[0]["frame"] == 20


def test_fix_contradicting_manual_anchor_dropped_operator_wins():
    # Physically plausible (inside court volume) but laterally miles from
    # where the operator actually clicked at the same frame.
    clicked_xyz = (5.0, 50.0, 1.0)
    fix_xyz = (25.0, 50.0, 1.0)
    anchors = [_anchor(10, clicked_xyz, state="player_touch")]
    fixes = [_fix(10, fix_xyz)]
    cams = {10: _CAM}

    knots, dropped = fixes_to_knots(fixes, anchors, cams, tol_px=5.0)

    assert knots == []
    assert len(dropped) == 1
    d = dropped[0]
    assert d["reason"] == "manual_anchor_conflict"
    assert d["frame"] == 10
    assert d["anchor_frame"] == 10
    assert d["px_dist"] > d["tol_px"]


def test_adjacent_frame_anchor_still_gates_a_fix():
    clicked_xyz = (5.0, 50.0, 1.0)
    fix_xyz = (25.0, 50.0, 1.0)
    # Anchor at frame 11, fix at frame 10 — within adjacent_frames=1.
    anchors = [_anchor(11, clicked_xyz, state="player_touch")]
    fixes = [_fix(10, fix_xyz)]
    cams = {10: _CAM, 11: _CAM}

    knots, dropped = fixes_to_knots(
        fixes, anchors, cams, tol_px=5.0, adjacent_frames=1)
    assert knots == []
    assert dropped[0]["reason"] == "manual_anchor_conflict"
    assert dropped[0]["anchor_frame"] == 11

    # With adjacent_frames=0 the anchor at frame 11 no longer gates a
    # fix at frame 10 — nothing to disagree with, so it's kept.
    knots2, dropped2 = fixes_to_knots(
        fixes, anchors, cams, tol_px=5.0, adjacent_frames=0)
    assert dropped2 == []
    assert len(knots2) == 1


def test_closest_anchor_wins_when_several_in_window():
    true_xyz = (5.0, 50.0, 1.0)
    far_xyz = (25.0, 50.0, 1.0)
    # Frame 10's fix agrees with the frame-10 anchor but would disagree
    # with a frame-9 anchor if that were checked instead.
    anchors = [
        _anchor(9, far_xyz, state="player_touch"),
        _anchor(10, true_xyz, state="player_touch"),
    ]
    fixes = [_fix(10, true_xyz)]
    cams = {9: _CAM, 10: _CAM}

    knots, dropped = fixes_to_knots(
        fixes, anchors, cams, tol_px=5.0, adjacent_frames=1)
    assert dropped == []
    assert len(knots) == 1


def test_off_screen_flight_anchor_never_gates():
    fix_xyz = (5.0, 50.0, 1.0)
    anchors = [BallAnchor(frame=10, image_xy=None, state="off_screen_flight")]
    fixes = [_fix(10, fix_xyz)]
    cams = {10: _CAM}

    knots, dropped = fixes_to_knots(fixes, anchors, cams, tol_px=5.0)
    assert dropped == []
    assert len(knots) == 1


def test_multiple_fixes_mixed_outcomes():
    good_xyz = (5.0, 50.0, 1.0)
    bad_xyz = (0.0, 20.0, -10.0)
    conflict_xyz = (25.0, 50.0, 1.0)
    anchors = [_anchor(10, good_xyz), _anchor(30, good_xyz)]
    fixes = [
        _fix(10, good_xyz),
        _fix(20, bad_xyz),
        _fix(30, conflict_xyz),
    ]
    cams = {10: _CAM, 20: _CAM, 30: _CAM}

    knots, dropped = fixes_to_knots(fixes, anchors, cams, tol_px=5.0)

    assert {k.frame for k in knots} == {10}
    reasons = {d["frame"]: d["reason"] for d in dropped}
    assert reasons[20] == "physically_impossible"
    assert reasons[30] == "manual_anchor_conflict"
