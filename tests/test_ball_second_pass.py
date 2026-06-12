"""Second-pass detection: corridor prediction, gating, gap runs."""

from __future__ import annotations

import numpy as np
import pytest

from src.utils.ball_second_pass import (
    SecondPassCfg,
    corridor_predictions,
    fuse_gaussians,
)
from src.utils.ball_tracker import BallTracker


def _tracker_factory() -> BallTracker:
    # Huge max_gap so predictions persist through long gaps.
    return BallTracker(max_gap_frames=10 ** 6)


@pytest.mark.unit
def test_fuse_gaussians_tightens_and_weights_by_precision():
    m1, c1 = np.array([0.0, 0.0]), np.eye(2) * 1.0
    m2, c2 = np.array([10.0, 0.0]), np.eye(2) * 9.0
    m, c = fuse_gaussians(m1, c1, m2, c2)
    # Precision-weighted: 9x tighter first estimate dominates.
    assert m[0] == pytest.approx(1.0)
    assert c[0, 0] == pytest.approx(0.9)
    assert c[0, 0] < min(c1[0, 0], c2[0, 0])


@pytest.mark.unit
def test_corridor_bridges_gap_near_interpolation():
    """Constant-velocity roll with a hole at frames 20-29: the fused
    forward/backward corridor must stay near the true line, far closer
    than either causal pass alone could drift."""
    n = 50
    truth = {f: (100.0 + 8.0 * f, 400.0) for f in range(n)}
    obs: dict[int, tuple[float, float] | None] = {
        f: (truth[f] if not (20 <= f < 30) else None) for f in range(n)
    }
    corridors = corridor_predictions(obs, n, _tracker_factory)
    for f in range(20, 30):
        mean, cov = corridors[f]
        du = abs(mean[0] - truth[f][0])
        dv = abs(mean[1] - truth[f][1])
        assert du < 12.0 and dv < 6.0, (f, du, dv)
        assert cov.shape == (2, 2)
        # Mid-gap uncertainty exceeds observed-frame uncertainty.
        assert cov[0, 0] > corridors[5][1][0, 0]


@pytest.mark.unit
def test_corridor_empty_when_no_observations():
    assert corridor_predictions({0: None, 1: None}, 2, _tracker_factory) == {}


from src.utils.ball_second_pass import (  # noqa: E402
    apparent_ball_px,
    best_gated_candidate,
    find_gap_runs,
    map_crop_candidates,
)


@pytest.mark.unit
def test_find_gap_runs_groups_missing_and_outlier_frames():
    sources = {0: "detector", 1: "detector", 5: "bridge", 6: "anchor"}
    runs = find_gap_runs(sources, outlier_frames={5}, n_frames=8)
    assert runs == [(2, 5), (7, 7)]


@pytest.mark.unit
def test_gate_rejects_decoy_outside_corridor_accepts_inside():
    cfg = SecondPassCfg()
    mean, cov = np.array([500.0, 300.0]), np.eye(2) * 25.0
    decoy = (900.0, 300.0, 0.95)          # high score, far outside
    true_cand = (505.0, 302.0, 0.6)       # modest score, inside
    best = best_gated_candidate([decoy, true_cand], mean, cov, cfg)
    assert best is not None
    (u, v), combined = best
    assert (u, v) == (505.0, 302.0)
    assert 0.0 < combined <= 0.6


@pytest.mark.unit
def test_gate_enforces_accept_min():
    cfg = SecondPassCfg(accept_min=0.5)
    mean, cov = np.array([500.0, 300.0]), np.eye(2) * 25.0
    assert best_gated_candidate([(505.0, 302.0, 0.3)], mean, cov, cfg) is None


@pytest.mark.unit
def test_gate_is_deterministic():
    cfg = SecondPassCfg()
    mean, cov = np.array([500.0, 300.0]), np.eye(2) * 25.0
    cands = [(505.0, 302.0, 0.6), (498.0, 297.0, 0.6)]
    assert best_gated_candidate(cands, mean, cov, cfg) == best_gated_candidate(
        cands, mean, cov, cfg
    )


@pytest.mark.unit
def test_apparent_ball_px_scales_inverse_with_depth():
    # Camera 20 m above pitch looking straight down: depth ~ 20 m.
    K = np.array([[2000.0, 0, 640.0], [0, 2000.0, 360.0], [0, 0, 1.0]])
    R = np.array([[1.0, 0, 0], [0, -1.0, 0], [0, 0, -1.0]])  # z_cam = -z_world
    t = -R @ np.array([0.0, 0.0, 20.0])
    size = apparent_ball_px(K, R, t, (640.0, 360.0), ball_radius_m=0.11)
    # f * d / depth = 2000 * 0.22 / (20 - 0.11) ≈ 22.1 px
    assert size == pytest.approx(22.1, abs=0.5)


@pytest.mark.unit
def test_apparent_ball_px_none_when_ray_misses_pitch():
    K = np.array([[2000.0, 0, 640.0], [0, 2000.0, 360.0], [0, 0, 1.0]])
    R = np.eye(3)  # looking along +z_world (up): never reaches the pitch
    t = -R @ np.array([0.0, 0.0, 20.0])
    assert apparent_ball_px(K, R, t, (640.0, 360.0), ball_radius_m=0.11) is None


@pytest.mark.unit
def test_map_crop_candidates_offsets_back_to_full_frame():
    assert map_crop_candidates([(10.0, 20.0, 0.7)], x0=300, y0=100) == [
        (310.0, 120.0, 0.7)
    ]


@pytest.mark.unit
def test_corridor_single_sided_before_first_and_after_last_obs():
    """Frames before the first observation get only the backward pass;
    frames after the last get only the forward pass — both single-sided
    branches must still emit (mean, cov)."""
    n = 30
    obs: dict[int, tuple[float, float] | None] = {f: None for f in range(n)}
    for f in range(10, 16):
        obs[f] = (100.0 + 8.0 * f, 400.0)
    corridors = corridor_predictions(obs, n, _tracker_factory)
    assert 5 in corridors and 25 in corridors
    for f in (5, 25):
        mean, cov = corridors[f]
        assert cov.shape == (2, 2)
        assert np.isfinite(mean).all()
    # Extrapolated frames are more uncertain than interpolated ones.
    assert corridors[5][1][0, 0] > corridors[12][1][0, 0]
