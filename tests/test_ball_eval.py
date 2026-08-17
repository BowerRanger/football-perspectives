"""Unit tests for src/utils/ball_eval.py — the sub-20cm campaign harness.

All tests use a synthetic pinhole camera; no clip data required.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.schemas.ball_anchor import BallAnchor

pytestmark = pytest.mark.unit


def _cam():
    """Simple pinhole: camera at (0,-20,10), looking at the pitch origin."""
    K = np.array([[1500.0, 0, 960], [0, 1500.0, 540], [0, 0, 1.0]])
    fwd = np.array([0.0, 20.0, -10.0])
    fwd /= np.linalg.norm(fwd)
    up = np.array([0.0, 0.0, 1.0])
    right = np.cross(fwd, up)
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    R = np.stack([right, down, fwd])  # world→cam rows
    C = np.array([0.0, -20.0, 10.0])
    t = -R @ C
    return K, R, t


def test_pixel_ray_roundtrip():
    from src.utils.ball_eval import pixel_ray, point_ray_distance
    from src.utils.camera_projection import project_world_to_image

    K, R, t = _cam()
    P = np.array([2.0, 5.0, 0.11])
    uv = project_world_to_image(K, R, t, (0.0, 0.0), P.reshape(1, 3))[0]
    C, d = pixel_ray(uv, K, R, t)
    perp, along = point_ray_distance(P, C, d)
    assert perp < 1e-6
    assert along > 0


def test_ray_plane_z_recovers_ground_point():
    from src.utils.ball_eval import pixel_ray, ray_plane_z
    from src.utils.camera_projection import project_world_to_image

    K, R, t = _cam()
    P = np.array([-3.0, 12.0, 0.11])
    uv = project_world_to_image(K, R, t, (0.0, 0.0), P.reshape(1, 3))[0]
    C, d = pixel_ray(uv, K, R, t)
    X = ray_plane_z(C, d, 0.11)
    assert X is not None
    assert np.allclose(X, P, atol=1e-6)


def test_ray_plane_z_none_when_parallel():
    from src.utils.ball_eval import ray_plane_z

    X = ray_plane_z(np.array([0.0, 0.0, 5.0]), np.array([0.0, 1.0, 0.0]), 0.11)
    assert X is None


def test_anchor_gt_world_ground_exact():
    from src.utils.ball_eval import anchor_gt_world
    from src.utils.camera_projection import project_world_to_image

    K, R, t = _cam()
    P = np.array([4.0, 8.0, 0.11])
    uv = project_world_to_image(K, R, t, (0.0, 0.0), P.reshape(1, 3))[0]
    anc = BallAnchor(frame=5, state="grounded",
                     image_xy=(float(uv[0]), float(uv[1])))
    gt, kind = anchor_gt_world(anc, K, R, t, (0.0, 0.0), ball_radius=0.11)
    assert kind == "ground_exact"
    assert np.allclose(gt, P, atol=1e-6)


def test_anchor_gt_world_joint_depth_projects_joint_onto_ray():
    from src.utils.ball_eval import (anchor_gt_world, pixel_ray,
                                     point_ray_distance)
    from src.utils.camera_projection import project_world_to_image

    K, R, t = _cam()
    true_ball = np.array([1.0, 6.0, 0.3])
    uv = project_world_to_image(K, R, t, (0.0, 0.0),
                                true_ball.reshape(1, 3))[0]
    joint = true_ball + np.array([0.05, 0.4, 0.05])  # FK drift off-ray
    anc = BallAnchor(frame=5, state="player_touch",
                     image_xy=(float(uv[0]), float(uv[1])),
                     player_id="P001", bone="r_foot")
    gt, kind = anchor_gt_world(anc, K, R, t, (0.0, 0.0), ball_radius=0.11,
                               joint_world=tuple(joint))
    assert kind == "joint_depth"
    C, d = pixel_ray(uv, K, R, t)
    perp, _ = point_ray_distance(np.asarray(gt), C, d)
    assert perp < 1e-9  # GT lies on the clicked ray
    assert np.linalg.norm(gt - true_ball) < 0.45  # depth ≈ joint depth


def test_anchor_gt_world_airborne_is_ray_only_and_no_pixel_is_none():
    from src.utils.ball_eval import anchor_gt_world

    K, R, t = _cam()
    anc = BallAnchor(frame=5, state="airborne_low", image_xy=(900.0, 400.0))
    gt, kind = anchor_gt_world(anc, K, R, t, (0.0, 0.0), ball_radius=0.11)
    assert (gt, kind) == (None, "ray_only")
    anc2 = BallAnchor(frame=6, state="off_screen_flight", image_xy=None)
    assert anchor_gt_world(anc2, K, R, t, (0.0, 0.0),
                           ball_radius=0.11) == (None, "none")


def test_eval_rows_at_anchors_grades_holdout_and_kinds():
    from src.utils.ball_eval import eval_rows_at_anchors
    from src.utils.camera_projection import project_world_to_image

    K, R, t = _cam()
    P_true = np.array([4.0, 8.0, 0.11])
    uv = project_world_to_image(K, R, t, (0.0, 0.0), P_true.reshape(1, 3))[0]
    anchors = [BallAnchor(frame=3, state="grounded",
                          image_xy=(float(uv[0]), float(uv[1])))]
    world = {3: tuple(P_true + np.array([0.05, 0.0, 0.0]))}  # 5cm off
    rows = eval_rows_at_anchors(world, anchors, {3: (K, R, t)},
                                ball_radius=0.11, distortion=(0.0, 0.0),
                                held_out_frames=frozenset({3}))
    (row,) = rows
    assert row.held_out and row.kind == "ground_exact"
    assert 0.03 < row.err_3d_m < 0.07
    assert row.lateral_m <= row.err_3d_m + 1e-9
    assert row.reproj_px > 0


def test_eval_rows_at_anchors_missing_track_frame_gives_none_errors():
    from src.utils.ball_eval import eval_rows_at_anchors

    K, R, t = _cam()
    anchors = [BallAnchor(frame=9, state="grounded", image_xy=(900.0, 700.0))]
    (row,) = eval_rows_at_anchors({}, anchors, {9: (K, R, t)},
                                  ball_radius=0.11, distortion=(0.0, 0.0))
    assert row.err_3d_m is None and row.lateral_m is None


def test_eval_rows_at_anchors_uses_joint_world_fn_for_touches():
    from src.utils.ball_eval import eval_rows_at_anchors
    from src.utils.camera_projection import project_world_to_image

    K, R, t = _cam()
    true_ball = np.array([1.0, 6.0, 0.3])
    uv = project_world_to_image(K, R, t, (0.0, 0.0),
                                true_ball.reshape(1, 3))[0]
    anchors = [BallAnchor(frame=7, state="player_touch",
                          image_xy=(float(uv[0]), float(uv[1])),
                          player_id="P001", bone="r_foot")]
    calls = []

    def joint_fn(frame, pid, bone):
        calls.append((frame, pid, bone))
        return tuple(true_ball + np.array([0.0, 0.3, 0.0]))

    world = {7: tuple(true_ball)}
    (row,) = eval_rows_at_anchors(world, anchors, {7: (K, R, t)},
                                  ball_radius=0.11, distortion=(0.0, 0.0),
                                  joint_world_fn=joint_fn)
    assert calls == [(7, "P001", "r_foot")]
    assert row.kind == "joint_depth"
    assert row.err_3d_m < 0.35  # depth error bounded by joint offset


def test_eval_rows_at_fixes():
    from src.utils.ball_eval import eval_rows_at_fixes

    world = {10: (1.0, 2.0, 3.0)}
    rows = eval_rows_at_fixes(world, [(10, (1.0, 2.0, 3.5), 0.2),
                                      (99, (0.0, 0.0, 0.0), 0.1)])
    assert abs(rows[0].err_3d_m - 0.5) < 1e-9 and rows[0].ray_miss_m == 0.2
    assert rows[1].err_3d_m is None


def test_dense_lateral_rows_filters_low_confidence():
    from src.utils.ball_eval import dense_lateral_rows
    from src.utils.camera_projection import project_world_to_image

    K, R, t = _cam()
    P = np.array([0.0, 10.0, 0.11])
    uv = project_world_to_image(K, R, t, (0.0, 0.0), P.reshape(1, 3))[0]
    obs = [(4, (float(uv[0]), float(uv[1])), 0.9, "detector"),
           (5, (float(uv[0]), float(uv[1])), 0.1, "detector")]
    world = {4: tuple(P + np.array([0.1, 0.0, 0.0])), 5: tuple(P)}
    rows = dense_lateral_rows(world, obs, {4: (K, R, t), 5: (K, R, t)},
                              distortion=(0.0, 0.0), min_confidence=0.5)
    assert len(rows) == 1 and rows[0].frame == 4
    assert 0.05 < rows[0].lateral_m < 0.15


def _mk_frames(worlds, states=None):
    from src.schemas.ball_track import BallFrame

    out = []
    for i, w in enumerate(worlds):
        out.append(BallFrame(frame=i, world_xyz=tuple(w),
                             state=(states[i] if states else "grounded"),
                             confidence=1.0))
    return out


def test_naturalness_flags_heading_break_away_from_events():
    from src.utils.ball_eval import naturalness_violations

    fps = 30.0
    pts = [(0.2 * i, 0.0, 0.11) for i in range(6)]
    pts += [(1.0, 0.2 * (i - 5), 0.11) for i in range(6, 11)]
    v = naturalness_violations(_mk_frames(pts), event_frames=set(), fps=fps)
    assert any(x.kind == "heading_break" and abs(x.frame - 5) <= 1 for x in v)


def test_naturalness_allows_break_at_event():
    from src.utils.ball_eval import naturalness_violations

    fps = 30.0
    pts = [(0.2 * i, 0.0, 0.11) for i in range(6)]
    pts += [(1.0, 0.2 * (i - 5), 0.11) for i in range(6, 11)]
    v = naturalness_violations(_mk_frames(pts), event_frames={5}, fps=fps)
    assert not [x for x in v if x.kind == "heading_break"]


def test_naturalness_flags_linear_flight_as_gravity_violation():
    from src.utils.ball_eval import naturalness_violations

    pts = [(0.3 * i, 0.0, 1.0 + 0.05 * i) for i in range(12)]
    v = naturalness_violations(_mk_frames(pts, states=["flight"] * 12),
                               event_frames=set(), fps=30.0)
    assert any(x.kind == "flight_gravity" for x in v)


def test_naturalness_accepts_true_parabola_and_steady_roll():
    from src.utils.ball_eval import naturalness_violations

    fps, g = 30.0, -9.81
    v0 = np.array([6.0, 0.0, 5.0])
    pts = [tuple(np.array([0, 0, 0.11]) + v0 * (i / fps)
                 + 0.5 * np.array([0, 0, g]) * (i / fps) ** 2)
           for i in range(15)]
    v = naturalness_violations(_mk_frames(pts, states=["flight"] * 15),
                               event_frames=set(), fps=fps)
    assert not [x for x in v if x.kind == "flight_gravity"]
    roll = [(0.2 * i, 0.05 * i, 0.11) for i in range(12)]
    v2 = naturalness_violations(_mk_frames(roll), event_frames=set(), fps=fps)
    assert not v2


def test_naturalness_flags_roll_speedup_without_event():
    from src.utils.ball_eval import naturalness_violations

    # Roll at 3 m/s that jumps to 6 m/s at frame 6 with no event.
    fps = 30.0
    pts = [(0.1 * i, 0.0, 0.11) for i in range(7)]
    pts += [(0.6 + 0.2 * (i - 6), 0.0, 0.11) for i in range(7, 13)]
    v = naturalness_violations(_mk_frames(pts), event_frames=set(), fps=fps)
    assert any(x.kind == "roll_speedup" for x in v)
    # Same profile WITH an event at the speed change: no violation.
    v2 = naturalness_violations(_mk_frames(pts), event_frames={6}, fps=fps)
    assert not [x for x in v2 if x.kind == "roll_speedup"]
