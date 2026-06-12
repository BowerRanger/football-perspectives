"""Cross-replay triangulation: pairing, offset refinement, gated fixes.

Two synthetic broadcast cameras 35 m apart observe the same analytic
trajectory; detections in the second view are offset by a known sync
delta. Pins: triangulation recovers 3D within tolerance, gates reject
poor parallax / inconsistent rays, and ray-miss refinement recovers a
deliberately-wrong saved offset.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.utils.ball_cross_replay import (
    CrossReplayCfg,
    interp_uv,
    refine_pair_offset,
    triangulate_pair,
    triangulate_rays,
)
from tests.fixtures.ball_synthetic import broadcast_camera, project_track

FPS = 25.0


def _two_cameras():
    KA, RA, tA = broadcast_camera(cam_centre=(52.5, -20.0, 15.0))
    KB, RB, tB = broadcast_camera(cam_centre=(20.0, -25.0, 18.0))
    return (KA, RA, tA), (KB, RB, tB)


def _arc_worlds(n: int) -> dict[int, np.ndarray]:
    # 2-second arc: kick at (40, 30), apex ~5 m.
    out = {}
    v0 = np.array([8.0, 4.0, 9.81])
    p0 = np.array([40.0, 30.0, 0.11])
    for f in range(n):
        t = f / FPS
        out[f] = p0 + v0 * t + 0.5 * np.array([0.0, 0.0, -9.81]) * t * t
    return out


def _obs(pixels: dict[int, tuple[float, float]], conf: float = 0.9):
    return {f: (uv, conf) for f, uv in pixels.items()}


@pytest.mark.unit
def test_triangulate_rays_recovers_point():
    (KA, RA, tA), (KB, RB, tB) = _two_cameras()
    world = np.array([45.0, 32.0, 3.0])
    pixA = project_track({0: world}, KA, RA, tA)[0]
    pixB = project_track({0: world}, KB, RB, tB)[0]
    point, miss, parallax = triangulate_rays(pixA, KA, RA, tA, pixB, KB, RB, tB)
    assert np.linalg.norm(point - world) < 0.05
    assert miss < 0.01
    assert parallax > 8.0


@pytest.mark.unit
def test_interp_uv_linear_between_neighbours():
    obs = _obs({10: (100.0, 200.0), 12: (110.0, 220.0)})
    uv = interp_uv(obs, 11.0, min_conf=0.3, max_span=3)
    assert uv == pytest.approx((105.0, 210.0))
    uv_frac = interp_uv(obs, 10.5, min_conf=0.3, max_span=3)
    assert uv_frac == pytest.approx((102.5, 205.0))
    assert interp_uv(obs, 20.0, min_conf=0.3, max_span=3) is None


@pytest.mark.unit
def test_triangulate_pair_inliers_and_gates():
    (KA, RA, tA), (KB, RB, tB) = _two_cameras()
    n = 40
    worlds = _arc_worlds(n)
    pixA = project_track(worlds, KA, RA, tA)
    pixB = project_track(worlds, KB, RB, tB)
    # B is offset by +5: B frame f shows the event A saw at frame f-5.
    obsA = _obs(pixA)
    obsB = _obs({f + 5: uv for f, uv in pixB.items()})
    camsA = {f: (KA, RA, tA) for f in range(n)}
    camsB = {f: (KB, RB, tB) for f in range(n + 5)}
    cfg = CrossReplayCfg()
    fixes = triangulate_pair(
        obs_a=obsA, cams_a=camsA, obs_b=obsB, cams_b=camsB,
        offset_b_minus_a=5.0, cfg=cfg,
    )
    assert len(fixes) >= 30
    for fx in fixes:
        assert np.linalg.norm(np.asarray(fx.xyz) - worlds[fx.frame_a]) < 0.10
        assert fx.ray_miss_m <= cfg.max_ray_miss_m
        assert fx.parallax_deg >= cfg.min_parallax_deg
    # A decoy detection in B far off the epipolar geometry is rejected.
    obsB_bad = dict(obsB)
    obsB_bad[10 + 5] = ((pixB[10][0] + 300.0, pixB[10][1]), 0.9)
    fixes_bad = triangulate_pair(
        obs_a=obsA, cams_a=camsA, obs_b=obsB_bad, cams_b=camsB,
        offset_b_minus_a=5.0, cfg=cfg,
    )
    assert all(fx.frame_a != 10 for fx in fixes_bad)


@pytest.mark.unit
def test_refine_pair_offset_recovers_true_delta():
    (KA, RA, tA), (KB, RB, tB) = _two_cameras()
    n = 40
    worlds = _arc_worlds(n)
    obsA = _obs(project_track(worlds, KA, RA, tA))
    obsB = _obs({f + 5: uv for f, uv in
                 project_track(worlds, KB, RB, tB).items()})
    camsA = {f: (KA, RA, tA) for f in range(n)}
    camsB = {f: (KB, RB, tB) for f in range(n + 5)}
    cfg = CrossReplayCfg()
    # Saved offset is wrong by 2 frames; refinement must find ~5.0.
    refined, median_miss, n_pairs = refine_pair_offset(
        obs_a=obsA, cams_a=camsA, obs_b=obsB, cams_b=camsB,
        saved_offset=7.0, cfg=cfg,
    )
    assert refined == pytest.approx(5.0, abs=cfg.offset_search_step)
    assert n_pairs >= cfg.min_pairs_for_refine
    assert median_miss < 0.05


@pytest.mark.unit
def test_refine_pair_offset_keeps_saved_when_too_few_pairs():
    (KA, RA, tA), (KB, RB, tB) = _two_cameras()
    worlds = _arc_worlds(4)
    obsA = _obs(project_track(worlds, KA, RA, tA))
    obsB = _obs({f + 5: uv for f, uv in
                 project_track(worlds, KB, RB, tB).items()})
    camsA = {f: (KA, RA, tA) for f in range(4)}
    camsB = {f: (KB, RB, tB) for f in range(9)}
    refined, _, n_pairs = refine_pair_offset(
        obs_a=obsA, cams_a=camsA, obs_b=obsB, cams_b=camsB,
        saved_offset=7.0, cfg=CrossReplayCfg(),
    )
    assert n_pairs < CrossReplayCfg().min_pairs_for_refine
    assert refined == 7.0
