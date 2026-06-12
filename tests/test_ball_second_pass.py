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
