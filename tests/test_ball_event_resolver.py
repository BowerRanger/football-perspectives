"""Tests for the EventResolver: body-pinned touch resolution + keyframes
+ segments + dense interpolation. Uses a fake player context and a simple
pinhole camera so it needs no torch / video / npz."""

from __future__ import annotations

import numpy as np

from src.schemas.ball_anchor import BallAnchor
from src.utils.ball_event_resolver import resolve_events
from src.utils.camera_projection import project_world_to_image

_K = np.array([[1000.0, 0.0, 640.0], [0.0, 1000.0, 360.0], [0.0, 0.0, 1.0]])
_R = np.eye(3)
_T = np.zeros(3)
_DIST = (0.0, 0.0)
_RADIUS = 0.11


class _FakeCtx:
    """Returns fixed joint world positions; mimics PlayerContext."""

    def __init__(self, joints: dict):
        self._j = joints

    def joint_world(self, frame, player_id, bone):
        return self._j.get((frame, player_id, bone))


def _uv_of(world) -> tuple[float, float]:
    uv = project_world_to_image(_K, _R, _T, _DIST, np.asarray([world]))[0]
    return (float(uv[0]), float(uv[1]))


def _resolve_one(anchor_by_frame, ctx, n_frames=1):
    return resolve_events(
        anchor_by_frame=anchor_by_frame,
        player_ctx=ctx,
        per_frame_K={f: _K for f in range(n_frames)},
        per_frame_R={f: _R for f in range(n_frames)},
        per_frame_t={f: _T for f in range(n_frames)},
        distortion=_DIST,
        ball_radius=_RADIUS,
        goal_geometry=None,
        n_frames=n_frames,
        fps=25.0,
        clip_id="c",
        image_size=(1280, 720),
    )


def test_touch_with_ball_pixel_resolves_near_joint():
    joint = (0.5, 0.2, 0.4)
    ctx = _FakeCtx({(0, "P1", "r_foot"): joint})
    anc = BallAnchor(frame=0, image_xy=_uv_of(joint), state="player_touch",
                     player_id="P1", bone="r_foot")
    res = _resolve_one({0: anc}, ctx, n_frames=1)
    world, _conf = res.world_by_frame[0]
    d = np.linalg.norm(np.asarray(world) - np.asarray(joint))
    assert d <= _RADIUS + 1e-6
    assert d > 0.0  # offset toward the camera by ~ball radius


def test_occluded_touch_still_resolves_to_joint():
    joint = (1.0, -0.5, 0.3)
    ctx = _FakeCtx({(0, "P2", "head"): joint})
    anc = BallAnchor(frame=0, image_xy=None, state="player_touch",
                     player_id="P2", bone="head")
    # image_xy None is normally rejected by the schema for player_touch, but
    # the resolver must still cope with an occluded contact frame.
    res = _resolve_one({0: anc}, ctx, n_frames=1)
    world, _conf = res.world_by_frame[0]
    assert world is not None
    d = np.linalg.norm(np.asarray(world) - np.asarray(joint))
    # Occluded: resolve straight to the joint position, no camera offset
    # (no ball pixel to define a lateral ray). Spec §7.
    assert d < 1e-6


def test_touch_keyframe_depth_source_is_player_bone():
    joint = (0.5, 0.2, 0.4)
    ctx = _FakeCtx({(0, "P1", "r_foot"): joint})
    anc = BallAnchor(frame=0, image_xy=_uv_of(joint), state="player_touch",
                     player_id="P1", bone="r_foot")
    res = _resolve_one({0: anc}, ctx, n_frames=1)
    kf = res.keyframe_set.keyframes[0]
    assert kf.depth_source == "player_bone"
    assert kf.player_id == "P1" and kf.bone == "r_foot"


def test_dense_world_spans_between_two_touches():
    j0, j1 = (0.0, 0.0, 0.3), (4.0, 0.0, 0.3)
    ctx = _FakeCtx({(0, "P1", "r_foot"): j0, (10, "P1", "l_foot"): j1})
    a0 = BallAnchor(frame=0, image_xy=_uv_of(j0), state="player_touch",
                    player_id="P1", bone="r_foot")
    a1 = BallAnchor(frame=10, image_xy=_uv_of(j1), state="player_touch",
                    player_id="P1", bone="l_foot")
    res = _resolve_one({0: a0, 10: a1}, ctx, n_frames=11)
    # interior frame filled by the roll interpolation between the touches
    assert 5 in res.world_by_frame
    assert len(res.world_by_frame) == 11


# ---------------------------------------------------------------------------
# W2a — ground clamp (sub-20cm campaign): a resolved touch can never sit
# below the pitch. Uses a z-up camera above the ground plane.

_K2 = np.array([[1500.0, 0.0, 960.0], [0.0, 1500.0, 540.0], [0.0, 0.0, 1.0]])


def _cam_zup():
    fwd = np.array([0.0, 20.0, -10.0])
    fwd /= np.linalg.norm(fwd)
    up = np.array([0.0, 0.0, 1.0])
    right = np.cross(fwd, up)
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    R = np.stack([right, down, fwd])
    C = np.array([0.0, -20.0, 10.0])
    return R, -R @ C


def _uv_zup(world, R, t):
    uv = project_world_to_image(_K2, R, t, _DIST, np.asarray([world]))[0]
    return (float(uv[0]), float(uv[1]))


def _resolve_zup(anchor_by_frame, ctx, R, t, n_frames=1):
    return resolve_events(
        anchor_by_frame=anchor_by_frame,
        player_ctx=ctx,
        per_frame_K={f: _K2 for f in range(n_frames)},
        per_frame_R={f: R for f in range(n_frames)},
        per_frame_t={f: t for f in range(n_frames)},
        distortion=_DIST,
        ball_radius=_RADIUS,
        goal_geometry=None,
        n_frames=n_frames,
        fps=25.0,
        clip_id="c",
        image_size=(1920, 1080),
    )


def test_touch_below_ground_with_pixel_clamps_onto_ray_at_ball_radius():
    R, t = _cam_zup()
    true_ball = np.array([2.0, 6.0, _RADIUS])   # ball on the ground
    joint = (2.0, 6.05, -0.08)                  # FK foot below the pitch
    ctx = _FakeCtx({(0, "P1", "r_foot"): joint})
    anc = BallAnchor(frame=0, image_xy=_uv_zup(true_ball, R, t),
                     state="player_touch", player_id="P1", bone="r_foot")
    res = _resolve_zup({0: anc}, ctx, R, t)
    world, _ = res.world_by_frame[0]
    assert world[2] >= _RADIUS - 1e-6
    # Stays on the clicked ray: reprojection matches the clicked pixel.
    uvp = project_world_to_image(_K2, R, t, _DIST,
                                 np.asarray([world]))[0]
    assert np.linalg.norm(uvp - np.asarray(anc.image_xy)) < 0.5
    # And lands at the true ground point (ray ∩ z=r).
    assert np.linalg.norm(np.asarray(world) - true_ball) < 0.05
    assert res.diagnostics.get("touch_ground_clamped", 0) == 1


def test_touch_below_ground_without_pixel_lifts_vertically():
    R, t = _cam_zup()
    joint = (3.0, 8.0, -0.05)
    ctx = _FakeCtx({(0, "P1", "l_foot"): joint})
    anc = BallAnchor(frame=0, image_xy=None, state="player_touch",
                     player_id="P1", bone="l_foot")
    res = _resolve_zup({0: anc}, ctx, R, t)
    world, _ = res.world_by_frame[0]
    assert abs(world[0] - 3.0) < 1e-6 and abs(world[1] - 8.0) < 1e-6
    assert abs(world[2] - _RADIUS) < 1e-6
    assert res.diagnostics.get("touch_ground_clamped", 0) == 1


def test_touch_above_ground_is_not_clamped():
    R, t = _cam_zup()
    true_ball = np.array([1.0, 5.0, 0.6])
    joint = (1.0, 5.1, 0.6)
    ctx = _FakeCtx({(0, "P1", "r_foot"): joint})
    anc = BallAnchor(frame=0, image_xy=_uv_zup(true_ball, R, t),
                     state="player_touch", player_id="P1", bone="r_foot")
    res = _resolve_zup({0: anc}, ctx, R, t)
    world, _ = res.world_by_frame[0]
    assert world[2] > _RADIUS
    assert res.diagnostics.get("touch_ground_clamped", 0) == 0


def test_touch_clamp_prefers_vertical_lift_when_ground_point_is_far():
    """Shallow-ray regression (origi01 f140): sliding along the ray to
    reach z=r can move the ball metres in depth. When ray ∩ z=r is out of
    contact range of the joint, clamp vertically at the joint instead."""
    # Very shallow camera: 2m high, 60m behind — grazing view of the pitch.
    fwd = np.array([0.0, 60.0, -1.9])
    fwd /= np.linalg.norm(fwd)
    up = np.array([0.0, 0.0, 1.0])
    right = np.cross(fwd, up)
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    R = np.stack([right, down, fwd])
    C = np.array([0.0, -60.0, 2.0])
    t = -R @ C
    joint = np.array([0.0, 0.0, -0.15])   # FK foot dipped below the pitch
    uv = project_world_to_image(_K2, R, t, _DIST, joint.reshape(1, 3))[0]
    ctx = _FakeCtx({(0, "P1", "r_foot"): tuple(joint)})
    anc = BallAnchor(frame=0, image_xy=(float(uv[0]), float(uv[1])),
                     state="player_touch", player_id="P1", bone="r_foot")
    res = _resolve_zup({0: anc}, ctx, R, t)
    world, _ = res.world_by_frame[0]
    assert world[2] >= _RADIUS - 1e-6
    # The clamp must stay near the joint, never slide metres along the ray.
    assert np.linalg.norm(np.asarray(world)[:2] - joint[:2]) < 0.5
    assert res.diagnostics.get("touch_ground_clamped", 0) == 1
