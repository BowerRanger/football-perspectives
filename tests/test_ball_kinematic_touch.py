import numpy as np
import pytest

from src.utils.ball_kinematic_touch import (
    KinematicTouchCfg,
    interpolate_ball_uvs,
    kinematic_gate,
    local_minima_below,
    ray_gap_series,
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
    assert strength > 0.0


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
