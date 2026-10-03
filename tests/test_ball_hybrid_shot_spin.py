"""D6.3 shot-span Magnus: start-frame tagging, curl bound, robust RSS."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from src.utils.ball_hybrid_physics import DEFAULT_MAGNUS_COEFF
from src.utils.ball_hybrid_spin import DEFAULT_BOUNDS, _rss
from src.utils.ball_hybrid_trajectory import _shot_start_frames, _shot_spin_bounds


def test_shot_start_frames_tags_shot_volley_and_spin_presets():
    anchors = [
        {"frame": 10, "state": "player_touch", "touch_type": "shot"},
        {"frame": 20, "state": "player_touch", "touch_type": "volley"},
        {"frame": 30, "state": "player_touch", "touch_type": "pass", "spin": "instep_curl_right"},
        {"frame": 40, "state": "player_touch", "touch_type": "pass"},
        {"frame": 50, "state": "grounded", "touch_type": "shot"},
        SimpleNamespace(frame=60, state="kick", touch_type="shot", spin=None),
    ]
    assert _shot_start_frames(anchors) == {10, 20, 30, 60}


def test_shot_spin_bounds_cap_peak_magnus_accel():
    a, b = np.array([18.0, 24.4, 0.11]), np.array([0.0, 37.1, 1.8])
    lo, hi = _shot_spin_bounds(DEFAULT_BOUNDS, a, b, 23 / 30.0, 0.25,
                               DEFAULT_MAGNUS_COEFF, 10.0)
    assert lo == -hi and 0 < hi <= DEFAULT_BOUNDS[1]
    # at the cap, a sidespin about +z on a horizontal velocity of that speed
    # produces <= 10 m/s^2
    from src.utils.ball_hybrid_physics import shoot_arc
    v0 = shoot_arc(a, 0.0, b, 23 / 30.0, cd=0.25)
    assert DEFAULT_MAGNUS_COEFF * hi * np.linalg.norm(v0) <= 10.0 + 1e-6
    # a looser cap never exceeds the global box
    assert _shot_spin_bounds(DEFAULT_BOUNDS, a, b, 23 / 30.0, 0.25,
                             DEFAULT_MAGNUS_COEFF, 1e6)[1] == pytest.approx(DEFAULT_BOUNDS[1])


def test_robust_rss_downweights_outliers_but_matches_ls_for_small_residuals():
    small = np.array([0.5, -0.4, 0.3])
    assert _rss(small, None) == pytest.approx(float(np.sum(small ** 2)))
    assert _rss(small, 15.0) == pytest.approx(float(np.sum(small ** 2)), rel=0.01)
    big = np.array([0.5, 200.0])
    assert _rss(big, 15.0) < 0.2 * _rss(big, None)  # soft-L1: linear, not quadratic
