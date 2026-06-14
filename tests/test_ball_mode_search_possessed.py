"""Task 4 TDD: POSSESSED mode tethered to player FK.

Tests:
1. Ball rides player P1's foot → low residual, worlds near foot track, kind="possessed",
   boundary_vel non-None.
2. Wrong player_id P2 (far away) → high residual.
3. Missing FK frames for P1 → worlds held (not crashed), segment produced.
4. nearest_players(ball_uv, frame, ctx, k=2) → 2 nearest by projected foot pixel, deduped.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.utils.ball_mode_search import (
    Mode,
    SegmentFit,
    _SegmentSolver,
    nearest_players,
)
from src.utils.ball_player_context import JointSample, PlayerContext
from src.utils.ball_piecewise_solver import SolverCfg
from tests.fixtures.ball_synthetic import (
    FPS,
    broadcast_camera,
    per_frame_cams,
    project_track,
    steps_from_pixels,
)


# ---------------------------------------------------------------------------
# Helpers: build a synthetic PlayerContext with foot worlds we control
# ---------------------------------------------------------------------------

def _make_joint_sample(
    player_id: str,
    bone: str,
    world_xyz: tuple[float, float, float],
    uv: tuple[float, float] | None,
    confidence: float = 0.9,
) -> JointSample:
    return JointSample(
        player_id=player_id,
        bone=bone,
        world_xyz=world_xyz,
        uv=uv,
        confidence=confidence,
    )


def _build_player_ctx(
    foot_worlds_by_frame: dict[int, dict[str, np.ndarray]],
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
) -> PlayerContext:
    """Build a real PlayerContext from explicit foot worlds per frame.

    foot_worlds_by_frame: {frame: {player_id: world_xyz (3,)}}
    Each player gets both l_foot and r_foot set to the same world for
    simplicity (tests only query l_foot for contact).
    """
    from src.utils.camera_projection import project_world_to_image

    samples_by_frame: dict[int, tuple[JointSample, ...]] = {}
    all_player_ids: set[str] = set()

    for frame, pid_worlds in foot_worlds_by_frame.items():
        frame_samples: list[JointSample] = []
        for pid, world_xyz in pid_worlds.items():
            all_player_ids.add(pid)
            w = np.asarray(world_xyz, dtype=float).reshape(1, 3)
            cam_z = float((w @ R.T + t)[0, 2])
            uv_arr = project_world_to_image(K, R, t, (0.0, 0.0), w)
            uv = (float(uv_arr[0, 0]), float(uv_arr[0, 1])) if cam_z > 0 else None
            for bone in ("l_foot", "r_foot"):
                frame_samples.append(_make_joint_sample(
                    player_id=pid,
                    bone=bone,
                    world_xyz=(float(world_xyz[0]), float(world_xyz[1]), float(world_xyz[2])),
                    uv=uv,
                ))
        if frame_samples:
            samples_by_frame[frame] = tuple(frame_samples)

    return PlayerContext(
        samples_by_frame=samples_by_frame,
        player_ids=tuple(sorted(all_player_ids)),
    )


def _solver_kwargs_for_possessed(
    foot_worlds: dict[int, np.ndarray],
    n_frames: int,
    *,
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    cfg: SolverCfg | None = None,
) -> dict:
    """Build _Solver kwargs with ball pixels matching a player's foot track."""
    from src.utils.camera_projection import project_world_to_image

    pixels: dict[int, tuple[float, float]] = {}
    for fi, world in foot_worlds.items():
        w = np.asarray(world, dtype=float).reshape(1, 3)
        uv = project_world_to_image(K, R, t, (0.0, 0.0), w)[0]
        pixels[fi] = (float(uv[0]), float(uv[1]))

    steps = steps_from_pixels(pixels, n_frames)
    per_K = {i: K for i in range(n_frames)}
    per_R = {i: R for i in range(n_frames)}
    per_t = {i: t for i in range(n_frames)}
    return dict(
        nodes=(),
        steps=steps,
        confidences={i: 1.0 for i in range(n_frames)},
        per_frame_K=per_K,
        per_frame_R=per_R,
        per_frame_t=per_t,
        distortion=(0.0, 0.0),
        fps=FPS,
        n_frames=n_frames,
        pitch_length_m=105.0,
        pitch_width_m=68.0,
        split_hints=(),
        z_hints=None,
        manual_obs_frames=None,
        world_fixes=None,
        cfg=cfg or SolverCfg(),
    )


# ---------------------------------------------------------------------------
# Test 1: Ball rides P1's foot — low residual, correct worlds, kind, vel
# ---------------------------------------------------------------------------

class TestPossessedCorrectPlayer:
    def test_low_residual_when_ball_matches_foot(self):
        K, R, t = broadcast_camera()
        n = 30

        # P1 moves steadily across the pitch at z=0.11 (ball on foot)
        foot_worlds_p1 = {
            i: np.array([30.0 + 0.3 * i, 34.0, 0.11])
            for i in range(n)
        }
        foot_worlds_by_frame = {i: {"P1": foot_worlds_p1[i]} for i in range(n)}
        player_ctx = _build_player_ctx(foot_worlds_by_frame, K, R, t)

        kwargs = _solver_kwargs_for_possessed(foot_worlds_p1, n, K=K, R=R, t=t)
        seg = _SegmentSolver(**kwargs, player_ctx=player_ctx)

        fa, fb = 2, 25
        fit = seg.fit_segment(fa, fb, Mode.POSSESSED, player_id="P1")

        assert isinstance(fit, SegmentFit)
        assert fit.kind == "possessed"
        assert fit.residual_px < 5.0, f"residual {fit.residual_px:.2f} too high for correct player"
        assert not fit.underconstrained

        # Worlds should follow P1's foot track
        for f in (5, 12, 20):
            if f in fit.worlds:
                dist = np.linalg.norm(fit.worlds[f] - foot_worlds_p1[f])
                assert dist < 0.5, f"frame {f}: world off by {dist:.3f}m from foot"

    def test_boundary_vel_non_none(self):
        K, R, t = broadcast_camera()
        n = 30
        foot_worlds_p1 = {i: np.array([30.0 + 0.3 * i, 34.0, 0.11]) for i in range(n)}
        foot_worlds_by_frame = {i: {"P1": foot_worlds_p1[i]} for i in range(n)}
        player_ctx = _build_player_ctx(foot_worlds_by_frame, K, R, t)
        kwargs = _solver_kwargs_for_possessed(foot_worlds_p1, n, K=K, R=R, t=t)
        seg = _SegmentSolver(**kwargs, player_ctx=player_ctx)

        fit = seg.fit_segment(2, 25, Mode.POSSESSED, player_id="P1")
        v_a, v_b = fit.boundary_vel
        assert v_a is not None, "start boundary velocity must be non-None for POSSESSED"
        assert v_b is not None, "end boundary velocity must be non-None for POSSESSED"
        # Velocity should be finite and non-zero (player is moving)
        assert np.all(np.isfinite(v_a))
        assert np.all(np.isfinite(v_b))

    def test_kind_is_possessed_not_flight(self):
        """F13: POSSESSED always emits kind='possessed', state maps to 'grounded'."""
        K, R, t = broadcast_camera()
        n = 20
        foot_worlds_p1 = {i: np.array([50.0 + 0.2 * i, 34.0, 0.11]) for i in range(n)}
        foot_worlds_by_frame = {i: {"P1": foot_worlds_p1[i]} for i in range(n)}
        player_ctx = _build_player_ctx(foot_worlds_by_frame, K, R, t)
        kwargs = _solver_kwargs_for_possessed(foot_worlds_p1, n, K=K, R=R, t=t)
        seg = _SegmentSolver(**kwargs, player_ctx=player_ctx)

        fit = seg.fit_segment(2, 15, Mode.POSSESSED, player_id="P1")
        assert fit.kind == "possessed", f"kind must be 'possessed', got {fit.kind!r}"
        assert fit.kind != "flight"
        assert fit.kind != "ballistic"


# ---------------------------------------------------------------------------
# Test 2: Wrong player (P2) → high residual
# ---------------------------------------------------------------------------

class TestPossessedWrongPlayer:
    def test_high_residual_for_wrong_player(self):
        K, R, t = broadcast_camera()
        n = 30

        # Ball follows P1 (near x=30)
        foot_worlds_p1 = {i: np.array([30.0 + 0.3 * i, 34.0, 0.11]) for i in range(n)}
        # P2 is on the far side of the pitch (x=80)
        foot_worlds_p2 = {i: np.array([80.0 + 0.1 * i, 20.0, 0.11]) for i in range(n)}

        foot_worlds_by_frame = {
            i: {"P1": foot_worlds_p1[i], "P2": foot_worlds_p2[i]}
            for i in range(n)
        }
        player_ctx = _build_player_ctx(foot_worlds_by_frame, K, R, t)
        kwargs = _solver_kwargs_for_possessed(foot_worlds_p1, n, K=K, R=R, t=t)

        seg = _SegmentSolver(**kwargs, player_ctx=player_ctx)
        fit_correct = seg.fit_segment(2, 25, Mode.POSSESSED, player_id="P1")
        fit_wrong = seg.fit_segment(2, 25, Mode.POSSESSED, player_id="P2")

        assert fit_wrong.residual_px > fit_correct.residual_px * 3, (
            f"Wrong player residual ({fit_wrong.residual_px:.1f}px) should be "
            f"much higher than correct player ({fit_correct.residual_px:.1f}px)"
        )


# ---------------------------------------------------------------------------
# Test 3: Missing FK frames — held, not crashed
# ---------------------------------------------------------------------------

class TestPossessedMissingFrames:
    def test_missing_fk_frames_held_not_crashed(self):
        """Some frames missing FK for P1 → worlds still produced, no crash."""
        K, R, t = broadcast_camera()
        n = 30

        # P1 has FK for all frames except 10-13 (4 consecutive missing)
        foot_worlds_p1_all = {i: np.array([30.0 + 0.3 * i, 34.0, 0.11]) for i in range(n)}
        missing_frames = {10, 11, 12, 13}
        foot_worlds_p1_sparse = {
            i: foot_worlds_p1_all[i] for i in range(n) if i not in missing_frames
        }

        # Player context only has FK for the sparse frames
        foot_worlds_by_frame = {
            i: {"P1": foot_worlds_p1_sparse[i]}
            for i in foot_worlds_p1_sparse
        }
        player_ctx = _build_player_ctx(foot_worlds_by_frame, K, R, t)

        # Ball pixels follow the complete foot track (so missing FK frames
        # still have ball obs, forcing the fitter to handle them)
        kwargs = _solver_kwargs_for_possessed(foot_worlds_p1_all, n, K=K, R=R, t=t)
        seg = _SegmentSolver(**kwargs, player_ctx=player_ctx)

        # Must NOT raise
        fit = seg.fit_segment(2, 25, Mode.POSSESSED, player_id="P1")

        assert isinstance(fit, SegmentFit)
        assert fit.kind == "possessed"
        # Worlds must be produced for all frames in [fa, fb]
        fa, fb = 2, 25
        for f in range(fa, fb + 1):
            assert f in fit.worlds, f"frame {f} missing from worlds dict"
            assert np.all(np.isfinite(fit.worlds[f])), f"frame {f} has non-finite world"

    def test_missing_fk_frames_hold_last_known(self):
        """Frames with missing FK should use a held (not zero) world position."""
        K, R, t = broadcast_camera()
        n = 20

        # P1 present frames 0-7 and 12-19, absent frames 8-11
        foot_worlds_p1_all = {i: np.array([40.0 + 0.3 * i, 34.0, 0.11]) for i in range(n)}
        missing_frames = {8, 9, 10, 11}
        foot_worlds_by_frame = {
            i: {"P1": foot_worlds_p1_all[i]}
            for i in range(n) if i not in missing_frames
        }
        player_ctx = _build_player_ctx(foot_worlds_by_frame, K, R, t)
        kwargs = _solver_kwargs_for_possessed(foot_worlds_p1_all, n, K=K, R=R, t=t)
        seg = _SegmentSolver(**kwargs, player_ctx=player_ctx)

        fit = seg.fit_segment(2, 16, Mode.POSSESSED, player_id="P1")

        # For missing frames 8-11, held world should be near last known (frame 7)
        last_known = foot_worlds_p1_all[7]
        for f in missing_frames:
            if 2 <= f <= 16:
                held = fit.worlds[f]
                dist = np.linalg.norm(held - last_known)
                # Should be held at or near last known, not wildly different
                assert dist < 2.0, (
                    f"frame {f} world ({held}) is far from last known ({last_known}); "
                    "expected hold behaviour"
                )


# ---------------------------------------------------------------------------
# Test 4: Raises if no player_ctx
# ---------------------------------------------------------------------------

def test_possessed_raises_without_player_ctx():
    """POSSESSED must raise a clear error if player_ctx was not passed."""
    K, R, t = broadcast_camera()
    n = 20
    foot_worlds = {i: np.array([30.0 + 0.3 * i, 34.0, 0.11]) for i in range(n)}
    from src.utils.camera_projection import project_world_to_image
    pixels = {}
    for fi, world in foot_worlds.items():
        w = np.asarray(world, dtype=float).reshape(1, 3)
        uv = project_world_to_image(K, R, t, (0.0, 0.0), w)[0]
        pixels[fi] = (float(uv[0]), float(uv[1]))
    steps = steps_from_pixels(pixels, n)
    per_K = {i: K for i in range(n)}
    per_R = {i: R for i in range(n)}
    per_t = {i: t for i in range(n)}
    kwargs = dict(
        nodes=(), steps=steps, confidences={i: 1.0 for i in range(n)},
        per_frame_K=per_K, per_frame_R=per_R, per_frame_t=per_t,
        distortion=(0.0, 0.0), fps=FPS, n_frames=n,
        pitch_length_m=105.0, pitch_width_m=68.0,
        split_hints=(), z_hints=None, manual_obs_frames=None,
        world_fixes=None, cfg=SolverCfg(),
    )
    seg = _SegmentSolver(**kwargs)  # no player_ctx
    with pytest.raises((ValueError, RuntimeError)):
        seg.fit_segment(2, 15, Mode.POSSESSED, player_id="P1")


# ---------------------------------------------------------------------------
# Test 5: nearest_players returns k nearest by projected foot pixel
# ---------------------------------------------------------------------------

class TestNearestPlayers:
    def test_returns_k_nearest_deduped(self):
        """nearest_players(ball_uv, frame, ctx, k=2) → the 2 nearest player_ids."""
        K, R, t = broadcast_camera()
        n = 10

        # P1 near ball (x=30), P2 slightly further (x=35), P3 far away (x=80)
        foot_worlds_by_frame = {
            i: {
                "P1": np.array([30.0, 34.0, 0.11]),
                "P2": np.array([35.0, 34.0, 0.11]),
                "P3": np.array([80.0, 20.0, 0.11]),
            }
            for i in range(n)
        }
        player_ctx = _build_player_ctx(foot_worlds_by_frame, K, R, t)

        # Ball pixel at frame 0 is at P1's foot pixel
        from src.utils.camera_projection import project_world_to_image
        p1_uv = project_world_to_image(K, R, t, (0.0, 0.0),
                                        np.array([30.0, 34.0, 0.11]).reshape(1, 3))[0]
        ball_uv = (float(p1_uv[0]), float(p1_uv[1]))

        result = nearest_players(ball_uv, 0, player_ctx, k=2)

        assert len(result) == 2
        # P1 should be first (nearest), P2 second
        assert result[0] == "P1"
        assert result[1] == "P2"
        # P3 should not appear (it's far away)
        assert "P3" not in result

    def test_deduplicates_by_player_id(self):
        """Each player_id appears at most once even though they have multiple bones."""
        K, R, t = broadcast_camera()
        n = 5

        foot_worlds_by_frame = {
            i: {"P1": np.array([30.0, 34.0, 0.11]), "P2": np.array([40.0, 34.0, 0.11])}
            for i in range(n)
        }
        player_ctx = _build_player_ctx(foot_worlds_by_frame, K, R, t)

        from src.utils.camera_projection import project_world_to_image
        p1_uv = project_world_to_image(K, R, t, (0.0, 0.0),
                                        np.array([30.0, 34.0, 0.11]).reshape(1, 3))[0]
        ball_uv = (float(p1_uv[0]), float(p1_uv[1]))

        result = nearest_players(ball_uv, 0, player_ctx, k=2)
        # Must be deduped: no player appears twice
        assert len(result) == len(set(result))

    def test_returns_fewer_than_k_when_not_enough_players(self):
        """When fewer players exist than k, return all available."""
        K, R, t = broadcast_camera()
        n = 5
        foot_worlds_by_frame = {i: {"P1": np.array([30.0, 34.0, 0.11])} for i in range(n)}
        player_ctx = _build_player_ctx(foot_worlds_by_frame, K, R, t)

        ball_uv = (960.0, 540.0)  # centre pixel
        result = nearest_players(ball_uv, 0, player_ctx, k=2)
        assert len(result) <= 2
        assert "P1" in result

    def test_empty_frame_returns_empty(self):
        """Frame with no player data → empty list."""
        K, R, t = broadcast_camera()
        n = 5
        foot_worlds_by_frame = {0: {"P1": np.array([30.0, 34.0, 0.11])}}
        player_ctx = _build_player_ctx(foot_worlds_by_frame, K, R, t)

        # Frame 3 has no player data
        result = nearest_players((960.0, 540.0), 3, player_ctx, k=2)
        assert result == []


# ---------------------------------------------------------------------------
# Test 6: Cache soundness — POSSESSED caches by player_id
# ---------------------------------------------------------------------------

def test_possessed_cache_keyed_by_player_id():
    """Different player_ids produce different cache entries."""
    K, R, t = broadcast_camera()
    n = 30

    foot_worlds_p1 = {i: np.array([30.0 + 0.3 * i, 34.0, 0.11]) for i in range(n)}
    foot_worlds_p2 = {i: np.array([80.0 + 0.1 * i, 20.0, 0.11]) for i in range(n)}
    foot_worlds_by_frame = {
        i: {"P1": foot_worlds_p1[i], "P2": foot_worlds_p2[i]}
        for i in range(n)
    }
    player_ctx = _build_player_ctx(foot_worlds_by_frame, K, R, t)
    kwargs = _solver_kwargs_for_possessed(foot_worlds_p1, n, K=K, R=R, t=t)
    seg = _SegmentSolver(**kwargs, player_ctx=player_ctx)

    fit_p1 = seg.fit_segment(2, 25, Mode.POSSESSED, player_id="P1")
    fit_p2 = seg.fit_segment(2, 25, Mode.POSSESSED, player_id="P2")
    fit_p1_again = seg.fit_segment(2, 25, Mode.POSSESSED, player_id="P1")

    assert seg._fit_calls == 2  # P1 + P2; second P1 was cached
    assert fit_p1 is fit_p1_again  # same object from cache
    assert fit_p1 is not fit_p2  # different players → different fits
