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
    assert cfg["enabled"] is False
    assert cfg["window"] == 2
    assert cfg["max_gap_m"] == 0.45
    assert cfg["margin_m"] == 0.05
