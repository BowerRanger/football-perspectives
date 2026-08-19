"""Touch bone-attribution refinement: relabel to the ray-closest joint,
keep originals on ambiguity, never add/remove/re-frame events."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from src.utils.ball_auto_events import BallEvent
from src.utils.ball_touch_attribution import (
    TouchAttributionCfg,
    refine_touch_attribution,
)

CFG = TouchAttributionCfg(enabled=True)


def _camera():
    look = np.array([0.0, 64.0, -30.0])
    look /= np.linalg.norm(look)
    right = np.array([1.0, 0.0, 0.0])
    down = np.cross(look, right)
    R = np.array([right, down, look], dtype=float)
    t = -R @ np.array([52.5, -30.0, 30.0])
    K = np.array([[1500.0, 0, 640.0], [0, 1500.0, 360.0], [0, 0, 1.0]])
    return K, R, t


def _project(p, K, R, t):
    cam = R @ np.asarray(p, dtype=float) + t
    pix = K @ cam
    return float(pix[0] / pix[2]), float(pix[1] / pix[2])


@dataclass(frozen=True)
class _Joint:
    player_id: str
    bone: str
    world_xyz: tuple[float, float, float]
    uv: tuple[float, float] | None
    confidence: float


class _Ctx:
    """PlayerContext stub: fixed joints at every frame."""

    def __init__(self, joints):
        self._joints = joints

    def joints_at(self, frame):
        return list(self._joints)


def _setup(ball_world=(40.0, 34.0, 0.11)):
    K, R, t = _camera()
    ball_uv = _project(np.array(ball_world), K, R, t)
    joints = [
        # l_foot right AT the ball; r_foot 0.8 m away.
        _Joint("P001", "l_foot", (40.0, 34.0, 0.11),
               _project(np.array([40.0, 34.0, 0.11]), K, R, t), 0.9),
        _Joint("P001", "r_foot", (40.8, 34.0, 0.11),
               _project(np.array([40.8, 34.0, 0.11]), K, R, t), 0.9),
    ]
    frames = range(8, 13)
    return (
        _Ctx(joints),
        {f: np.asarray(ball_uv) for f in frames},
        {f: K for f in frames}, {f: R for f in frames}, {f: t for f in frames},
    )


def _refine(events, ctx, uvs, Ks, Rs, ts, cfg=CFG):
    return refine_touch_attribution(
        events, player_ctx=ctx, ball_uvs=uvs,
        per_frame_K=Ks, per_frame_R=Rs, per_frame_t=ts,
        distortion=(0.0, 0.0), cfg=cfg,
    )


def test_wrong_bone_relabelled_to_ray_closest_joint():
    ctx, uvs, Ks, Rs, ts = _setup()
    events = (BallEvent(frame=10, kind="touch", score=0.7,
                        player_id="P001", bone="r_foot"),)
    out = _refine(events, ctx, uvs, Ks, Rs, ts)
    assert len(out) == 1
    assert out[0].bone == "l_foot"
    assert out[0].player_id == "P001"
    assert out[0].frame == 10 and out[0].kind == "touch"
    assert out[0].score == pytest.approx(0.7)


def test_ambiguous_margin_keeps_original():
    # Both feet equidistant-ish: margin gate keeps the original label.
    K, R, t = _camera()
    ball_uv = _project(np.array([40.0, 34.0, 0.11]), K, R, t)
    joints = [
        _Joint("P001", "l_foot", (40.02, 34.0, 0.11),
               _project(np.array([40.02, 34.0, 0.11]), K, R, t), 0.9),
        _Joint("P001", "r_foot", (40.05, 34.0, 0.11),
               _project(np.array([40.05, 34.0, 0.11]), K, R, t), 0.9),
    ]
    ctx = _Ctx(joints)
    uvs = {10: np.asarray(ball_uv)}
    events = (BallEvent(frame=10, kind="touch", score=0.7,
                        player_id="P001", bone="r_foot"),)
    out = refine_touch_attribution(
        events, player_ctx=ctx, ball_uvs=uvs,
        per_frame_K={10: K}, per_frame_R={10: R}, per_frame_t={10: t},
        distortion=(0.0, 0.0), cfg=CFG,
    )
    assert out[0].bone == "r_foot"


def test_far_ball_no_candidate_keeps_original():
    ctx, uvs, Ks, Rs, ts = _setup(ball_world=(60.0, 10.0, 0.11))
    events = (BallEvent(frame=10, kind="touch", score=0.7,
                        player_id="P001", bone="r_foot"),)
    out = _refine(events, ctx, uvs, Ks, Rs, ts)
    assert out[0].bone == "r_foot"  # best gap exceeds max_gap_m -> unchanged


def test_non_touch_events_and_order_preserved():
    ctx, uvs, Ks, Rs, ts = _setup()
    events = (
        BallEvent(frame=5, kind="bounce", score=0.6),
        BallEvent(frame=10, kind="touch", score=0.7,
                  player_id="P001", bone="r_foot"),
        BallEvent(frame=20, kind="goal_impact", score=0.9,
                  goal_element="post"),
    )
    out = _refine(events, ctx, uvs, Ks, Rs, ts)
    assert [e.kind for e in out] == ["bounce", "touch", "goal_impact"]
    assert out[0] == events[0] and out[2] == events[2]


def test_no_ball_uv_in_window_keeps_original():
    ctx, _uvs, Ks, Rs, ts = _setup()
    events = (BallEvent(frame=10, kind="touch", score=0.7,
                        player_id="P001", bone="r_foot"),)
    out = _refine(events, ctx, {}, Ks, Rs, ts)
    assert out[0].bone == "r_foot"


def test_disabled_is_identity():
    ctx, uvs, Ks, Rs, ts = _setup()
    events = (BallEvent(frame=10, kind="touch", score=0.7,
                        player_id="P001", bone="r_foot"),)
    out = _refine(events, ctx, uvs, Ks, Rs, ts,
                  cfg=TouchAttributionCfg(enabled=False))
    assert out == events


def test_config_block_keys():
    import yaml
    from pathlib import Path
    cfg = yaml.safe_load(
        Path("config/default.yaml").read_text())["ball"]["touch_attribution"]
    assert cfg["enabled"] is True
    assert cfg["window"] == 2
    assert cfg["max_gap_m"] == 0.45
    assert cfg["margin_m"] == 0.05


def test_depth_consistency_breaks_ray_gap_tie(  # W5d: depth-blind gate fix
):
    """The kicker's foot can sit nearer the camera-ball RAY than the true
    toucher's knee while being metres off in DEPTH along it. With an
    expected ball world supplied, the depth-consistent joint wins."""
    K, R, t = _camera()
    true_ball = np.array([52.5, 30.0, 0.6])       # ball at the knee
    uv = _project(true_ball, K, R, t)
    C = -R.T @ t
    ray = true_ball - C
    ray /= np.linalg.norm(ray)
    # Kicker's foot: exactly ON the ray, 5m closer to the camera (gap ≈ 0).
    kicker_foot = tuple(true_ball - 5.0 * ray)
    knee = tuple(true_ball + np.array([0.12, 0.05, 0.0]))
    ctx = _Ctx([
        _Joint("P014", "r_foot", kicker_foot, None, 0.9),
        _Joint("P008", "r_knee", knee, None, 0.9),
    ])
    events = (BallEvent(frame=10, kind="touch", score=0.8,
                        player_id="P014", bone="r_foot"),)
    common = dict(
        player_ctx=ctx, ball_uvs={10: np.asarray(uv)},
        per_frame_K={10: K}, per_frame_R={10: R}, per_frame_t={10: t},
        distortion=(0.0, 0.0),
    )
    # Without depth info the near-ray kicker foot keeps the label.
    out_blind = refine_touch_attribution(events, cfg=CFG, **common)
    assert (out_blind[0].player_id, out_blind[0].bone) == ("P014", "r_foot")
    # With the expected ball world, the depth-consistent knee wins.
    out_depth = refine_touch_attribution(
        events, cfg=CFG,
        expected_world_by_frame={10: tuple(true_ball)}, **common)
    assert (out_depth[0].player_id, out_depth[0].bone) == ("P008", "r_knee")
    assert out_depth[0].frame == 10 and len(out_depth) == 1


def test_expected_worlds_interpolate_between_ground_anchors():
    from src.schemas.ball_anchor import BallAnchor
    from src.utils.ball_touch_attribution import expected_ball_worlds

    K, R, t = _camera()
    a = np.array([40.0, 30.0, 0.11])
    b = np.array([46.0, 33.0, 0.11])
    anchors = {
        10: BallAnchor(frame=10, image_xy=_project(a, K, R, t),
                       state="grounded"),
        20: BallAnchor(frame=20, image_xy=_project(b, K, R, t),
                       state="grounded"),
        15: BallAnchor(frame=15, image_xy=None, state="off_screen_flight"),
    }
    worlds = expected_ball_worlds(
        anchors, per_frame_K={f: K for f in range(30)},
        per_frame_R={f: R for f in range(30)},
        per_frame_t={f: t for f in range(30)},
        distortion=(0.0, 0.0), ball_radius=0.11)
    assert np.allclose(worlds[10], a, atol=1e-6)
    assert np.allclose(worlds[20], b, atol=1e-6)
    mid = np.asarray(worlds[15])
    assert np.allclose(mid, (a + b) / 2, atol=0.05)
    assert 25 not in worlds        # no extrapolation past the last anchor


def test_context_expected_worlds_bridge_over_touch_windows():
    from src.utils.ball_touch_attribution import context_expected_worlds

    # Track dragged to a wrong pin at f10 (spike); context bridges over it.
    world = {f: (float(f), 0.0, 0.11) for f in range(21)}
    world[10] = (10.0, 8.0, 0.11)     # dragged toward the wrong joint
    world[9] = (9.0, 4.0, 0.11)
    world[11] = (11.0, 4.0, 0.11)
    exp = context_expected_worlds(world, touch_frames={10}, window=2)
    # Frames inside the ±window around the touch are re-interpolated from
    # the clean context (f7 → f13): the spike is bridged away.
    for f in range(8, 13):
        assert abs(exp[f][1]) < 0.3, f
        assert abs(exp[f][0] - f) < 0.3, f
    # Far frames keep the track's own value.
    assert exp[3] == world[3]


def test_swinging_foot_wins_left_right_disambiguation():
    """f192/f343 class: both feet near the ball at contact — the foot
    SWINGING toward the ball is the toucher (kinematic term)."""
    K, R, t = _camera()
    ball = np.array([40.0, 34.0, 0.11])
    uv = _project(ball, K, R, t)
    # Planted foot slightly NEARER the ray; swinging foot approaches fast.
    planted = np.array([40.10, 34.0, 0.11])
    swing_pos = {9: np.array([39.5, 33.6, 0.11]),
                 10: np.array([40.18, 33.95, 0.11]),
                 11: np.array([40.6, 34.25, 0.11])}

    class _KinCtx:
        def joints_at(self, frame):
            out = [_Joint("P003", "l_foot", tuple(planted), None, 0.9)]
            if frame in swing_pos:
                out.append(_Joint("P003", "r_foot",
                                  tuple(swing_pos[frame]), None, 0.9))
            return out

    events = (BallEvent(frame=10, kind="touch", score=0.8,
                        player_id="P003", bone="l_foot"),)
    out = refine_touch_attribution(
        events, player_ctx=_KinCtx(),
        ball_uvs={f: np.asarray(uv) for f in (9, 10, 11)},
        per_frame_K={f: K for f in (9, 10, 11)},
        per_frame_R={f: R for f in (9, 10, 11)},
        per_frame_t={f: t for f in (9, 10, 11)},
        distortion=(0.0, 0.0), cfg=CFG,
    )
    assert (out[0].player_id, out[0].bone) == ("P003", "r_foot")
