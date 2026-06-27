import numpy as np
import pytest

from src.utils.ball_auto_events import BallEvent
from src.utils.ball_kinematic_touch import (
    KinematicTouchCfg,
    ball_confirm,
    interpolate_ball_uvs,
    kinematic_gate,
    local_minima_below,
    nms_touches,
    propose_touches,
    ray_gap_series,
    touch_score,
)
from src.utils.ball_player_context import JointSample, PlayerContext
from src.utils.camera_projection import project_world_to_image


def test_cfg_defaults_are_high_recall():
    cfg = KinematicTouchCfg()
    assert cfg.enabled is True
    assert cfg.contact_gap_m == pytest.approx(0.30)
    assert cfg.min_emit_score == pytest.approx(0.25)
    assert cfg.max_ball_gap_frames == 6


def test_interpolate_fills_short_gap_and_flags_it():
    uvs = {0: np.array([0.0, 0.0]), 3: np.array([3.0, 6.0])}
    filled, interp = interpolate_ball_uvs(uvs, max_gap_frames=6)
    assert set(filled) == {0, 1, 2, 3}
    assert filled[1] == pytest.approx(np.array([1.0, 2.0]))
    assert filled[2] == pytest.approx(np.array([2.0, 4.0]))
    assert interp == frozenset({1, 2})


def test_interpolate_leaves_long_gap_empty():
    uvs = {0: np.array([0.0, 0.0]), 10: np.array([10.0, 0.0])}
    filled, interp = interpolate_ball_uvs(uvs, max_gap_frames=6)
    assert set(filled) == {0, 10}
    assert interp == frozenset()


def test_interpolate_exact_boundary():
    # Gap of exactly max_gap_frames (6) interior frames -> fill
    uvs = {0: np.array([0.0, 0.0]), 7: np.array([7.0, 0.0])}
    _filled, interp = interpolate_ball_uvs(uvs, max_gap_frames=6)
    assert len(interp) == 6
    # Gap of max_gap_frames + 1 (7) interior frames -> leave empty
    uvs2 = {0: np.array([0.0, 0.0]), 8: np.array([8.0, 0.0])}
    _filled2, interp2 = interpolate_ball_uvs(uvs2, max_gap_frames=6)
    assert interp2 == frozenset()


def _cam(frames):
    K = np.array([[1000.0, 0, 960.0], [0, 1000.0, 540.0], [0, 0, 1.0]])
    R = np.eye(3)
    t = np.zeros(3)
    return ({f: K for f in frames}, {f: R for f in frames}, {f: t for f in frames})


def _ball_uv(world):
    K = np.array([[1000.0, 0, 960.0], [0, 1000.0, 540.0], [0, 0, 1.0]])
    return np.asarray(
        project_world_to_image(K, np.eye(3), np.zeros(3), (0.0, 0.0), np.asarray([world]))[0],
        dtype=float,
    )


def test_local_minima_below_picks_the_dip():
    series = {0: 0.5, 1: 0.3, 2: 0.1, 3: 0.25, 4: 0.4}
    assert local_minima_below(series, 0.3) == [2]
    assert local_minima_below(series, 0.05) == []


def test_ray_gap_zero_when_bone_on_ball_ray():
    frames = range(5)
    K, R, t = _cam(frames)
    # ball fixed at world (0,0,10) -> uv (960,540); ray is the +z axis.
    ball_world = (0.0, 0.0, 10.0)
    ball_uvs = {f: _ball_uv(ball_world) for f in frames}
    # foot x sweeps toward the z-axis (gap == |x|), nearest at frame 3.
    xs = {0: 0.5, 1: 0.3, 2: 0.12, 3: 0.04, 4: 0.2}
    samples = {
        f: (
            JointSample("P1", "r_foot", (xs[f], 0.0, 9.0), (960.0, 540.0), 0.9),
        )
        for f in frames
    }
    ctx = PlayerContext(samples, ("P1",))
    series = ray_gap_series(ctx, ball_uvs, K, R, t, (0.0, 0.0), min_fk_conf=0.3)
    gaps = {f: g for f, (g, _px, _c) in series[("P1", "r_foot")].items()}
    assert gaps[3] == pytest.approx(0.04, abs=1e-6)
    assert local_minima_below(gaps, 0.30) == [3]


def test_ray_gap_skips_low_fk_conf():
    frames = range(3)
    K, R, t = _cam(frames)
    ball_uvs = {f: _ball_uv((0.0, 0.0, 10.0)) for f in frames}
    samples = {
        f: (
            JointSample("P1", "r_foot", (0.0, 0.0, 9.0), (960.0, 540.0), 0.1),
        )
        for f in frames
    }
    ctx = PlayerContext(samples, ("P1",))
    series = ray_gap_series(ctx, ball_uvs, K, R, t, (0.0, 0.0), min_fk_conf=0.3)
    assert ("P1", "r_foot") not in series


def _foot_ctx(positions_uv, world=(0.0, 0.0, 9.0)):
    # positions_uv: dict frame -> (u, v) for the foot bone of P1.
    samples = {
        f: (JointSample("P1", "r_foot", world, uv, 0.9),)
        for f, uv in positions_uv.items()
    }
    return PlayerContext(samples, ("P1",))


def test_kicking_foot_passes_gate():
    # foot u moves 20 px/frame -> central-diff speed 20 at frame 1.
    ctx = _foot_ctx({0: (900.0, 540.0), 1: (920.0, 540.0), 2: (940.0, 540.0)})
    passed, strength = kinematic_gate(ctx, 1, "P1", "r_foot", KinematicTouchCfg())
    assert passed is True
    assert strength == pytest.approx(1.0)  # 20 px/frame > _KICK_SPEED_PX (12)


def test_planted_foot_fails_gate():
    ctx = _foot_ctx({0: (900.0, 540.0), 1: (900.2, 540.0), 2: (900.4, 540.0)})
    passed, _ = kinematic_gate(ctx, 1, "P1", "r_foot", KinematicTouchCfg())
    assert passed is False


def test_keeper_hand_always_passes():
    samples = {1: (JointSample("P1", "l_hand", (0.0, 0.0, 1.5), (500.0, 300.0), 0.9),)}
    ctx = PlayerContext(samples, ("P1",))
    passed, strength = kinematic_gate(ctx, 1, "P1", "l_hand", KinematicTouchCfg())
    assert passed is True
    assert strength == pytest.approx(0.5)


def test_foot_strength_proportional_below_saturation():
    # central diff at frame 1 = (915 - 895) / 2 = 10 px/frame
    ctx = _foot_ctx({0: (895.0, 540.0), 1: (905.0, 540.0), 2: (915.0, 540.0)})
    passed, strength = kinematic_gate(ctx, 1, "P1", "r_foot", KinematicTouchCfg())
    assert passed is True            # 10 >= kin_min_foot_speed (8)
    assert strength == pytest.approx(10.0 / 12.0)


def test_missing_foot_data_fails_gate():
    ctx = PlayerContext({}, ())
    passed, strength = kinematic_gate(ctx, 1, "P1", "r_foot", KinematicTouchCfg())
    assert passed is False
    assert strength == pytest.approx(0.0)


def test_confirm_boost_when_break_nearby():
    cfg = KinematicTouchCfg()
    assert ball_confirm(10, cfg, confirm_frames=frozenset({11}),
                        interp_frames=frozenset(),
                        detected_frames=frozenset(range(0, 30))) == 1.0


def test_confirm_no_penalty_when_occluded():
    cfg = KinematicTouchCfg()
    # frame 10 + neighbours not in detected_frames -> occluded -> 0.0
    assert ball_confirm(10, cfg, confirm_frames=frozenset(),
                        interp_frames=frozenset({9, 10, 11}),
                        detected_frames=frozenset()) == 0.0


def test_confirm_downweight_when_visible_unchanged():
    cfg = KinematicTouchCfg()
    assert ball_confirm(10, cfg, confirm_frames=frozenset(),
                        interp_frames=frozenset(),
                        detected_frames=frozenset(range(0, 30))) == -1.0


def test_score_monotonic_and_clipped():
    cfg = KinematicTouchCfg()
    good = touch_score(0.02, 1.0, 1.0, 0.9, False, cfg)
    poor = touch_score(0.29, 0.0, -1.0, 0.3, True, cfg)
    assert 0.0 <= poor < good <= 1.0
    assert good == pytest.approx(
        min(1.0, cfg.w_gap * (1 - 0.02 / cfg.contact_gap_m)
            + cfg.w_kin * 1.0 + cfg.w_confirm * 1.0 + cfg.w_fk * 0.9), abs=1e-9)


def _kick_scene(drop_contact_frame=None):
    """Foot sweeps fast through a fixed ball at frame 3; optionally drop the
    ball detection at the contact frame (the headline occlusion case)."""
    frames = list(range(7))
    K, R, t = _cam(frames)
    ball_world = (0.0, 0.0, 10.0)
    ball_uvs = {f: _ball_uv(ball_world) for f in frames}
    if drop_contact_frame is not None:
        del ball_uvs[drop_contact_frame]
    # foot x sweeps 0.6 -> -0.6 (fast), nearest the z-axis ray at frame 3,
    # and its projected uv moves fast (kick signature).
    xs = {0: 0.6, 1: 0.4, 2: 0.2, 3: 0.03, 4: -0.2, 5: -0.4, 6: -0.6}
    samples = {}
    for f in frames:
        uv = _ball_uv((xs[f], 0.0, 9.0))
        samples[f] = (JointSample("P1", "r_foot", (xs[f], 0.0, 9.0),
                                  (float(uv[0]), float(uv[1])), 0.9),)
    ctx = PlayerContext(samples, ("P1",))
    return ctx, ball_uvs, K, R, t


def test_propose_detects_kick_when_ball_present():
    ctx, ball_uvs, K, R, t = _kick_scene()
    cfg = KinematicTouchCfg()
    # ball visible + no break -> downweight; raise recall floor for the test.
    touches = propose_touches(
        player_ctx=ctx, ball_uvs=ball_uvs, per_frame_K=K, per_frame_R=R,
        per_frame_t=t, confirm_frames=frozenset({3}), cfg=cfg)
    assert any(e.player_id == "P1" and e.bone == "r_foot" and abs(e.frame - 3) <= 1
               for e in touches)


def test_propose_rescues_touch_when_ball_occluded_at_contact():
    ctx, ball_uvs, K, R, t = _kick_scene(drop_contact_frame=3)
    filled = set(ball_uvs)  # detections that survived
    cfg = KinematicTouchCfg()
    touches = propose_touches(
        player_ctx=ctx, ball_uvs=ball_uvs, per_frame_K=K, per_frame_R=R,
        per_frame_t=t, confirm_frames=frozenset(),
        detected_frames=frozenset(filled), cfg=cfg)
    # One-frame drop: the ball is visible either side, so ball_confirm DOWNWEIGHTS
    # (visible-unchanged), yet the body signal (gap + kinematic gate) still carries
    # the touch over min_emit_score. The fully-occluded no-penalty path is unit-
    # tested separately in test_confirm_no_penalty_when_occluded.
    assert any(e.bone == "r_foot" and abs(e.frame - 3) <= 1 for e in touches)


def test_propose_rejects_planted_foot_ball_grazing():
    frames = list(range(7))
    K, R, t = _cam(frames)
    # ball moves across; foot planted at small fixed offset (gap dips < 0.3)
    # but foot pixel speed ~0 -> kinematic gate fails.
    foot_world = (0.08, 0.0, 9.0)
    foot_uv = _ball_uv(foot_world)
    ball_xs = {0: 0.6, 1: 0.4, 2: 0.2, 3: 0.08, 4: -0.1, 5: -0.3, 6: -0.5}
    ball_uvs = {f: _ball_uv((ball_xs[f], 0.0, 10.0)) for f in frames}
    samples = {f: (JointSample("P1", "r_foot", foot_world,
                               (float(foot_uv[0]), float(foot_uv[1])), 0.9),)
               for f in frames}
    ctx = PlayerContext(samples, ("P1",))
    touches = propose_touches(
        player_ctx=ctx, ball_uvs=ball_uvs, per_frame_K=K, per_frame_R=R,
        per_frame_t=t, cfg=KinematicTouchCfg())
    assert touches == []


def test_nms_keeps_highest_score_per_bone_in_window():
    evs = [
        BallEvent(frame=10, kind="touch", score=0.4, player_id="P1", bone="r_foot"),
        BallEvent(frame=11, kind="touch", score=0.8, player_id="P1", bone="r_foot"),
        BallEvent(frame=40, kind="touch", score=0.5, player_id="P1", bone="r_foot"),
        BallEvent(frame=11, kind="touch", score=0.9, player_id="P2", bone="l_foot"),
    ]
    kept = nms_touches(evs, window=2)
    assert (11, "P1", "r_foot") in {(e.frame, e.player_id, e.bone) for e in kept}
    assert (10, "P1", "r_foot") not in {(e.frame, e.player_id, e.bone) for e in kept}
    assert len(kept) == 3  # P1@11, P1@40, P2@11
    assert [e.frame for e in kept] == sorted(e.frame for e in kept)
