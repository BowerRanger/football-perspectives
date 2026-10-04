"""D6.2 direction-consistency gate (src/utils/ball_direction_gate.py)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from src.utils.ball_direction_gate import (
    direction_gate_cfg,
    filter_reversed_observations,
)


def _obs(frame, uv):
    return SimpleNamespace(frame=frame, uv=tuple(uv), conf=0.7, source="detector")


def _fit(frame):
    """Fitted ball moving left 10 px/frame along y=300."""
    return np.array([900.0 - 10.0 * (frame - 380), 300.0])


def _true_obs(frames):
    return [_obs(f, _fit(f) + [1.0, -1.0]) for f in frames]


def test_drops_run_of_reversed_detections_like_the_gberch_glove():
    good = _true_obs(range(380, 387))
    # false object drifting the OPPOSITE way (rightwards) and off the fit
    fake = [_obs(387 + i, (700.0 + 12.0 * i, 330.0)) for i in range(5)]
    kept, dropped = filter_reversed_observations(good + fake, _fit)
    assert set(dropped) >= {388, 389, 390, 391}
    assert all(o.frame in range(380, 388) for o in kept)


def test_single_reversed_frame_is_kept():
    obs = _true_obs(range(380, 390))
    obs[5] = _obs(385, (900.0 - 50.0 + 25.0, 330.0))  # one blip, then back on the fit
    kept, dropped = filter_reversed_observations(obs, _fit)
    assert dropped == []
    assert len(kept) == len(obs)


def test_reversal_on_the_fit_is_kept():
    """Velocity disagrees but the detection is still within residual_px of
    the fit (jitter) -> not dropped."""
    obs = _true_obs(range(380, 388))
    obs[4] = _obs(384, _fit(384) + [6.0, 0.0])
    obs[5] = _obs(385, _fit(385) - [6.0, 0.0])
    kept, dropped = filter_reversed_observations(obs, _fit)
    assert dropped == []


def test_protected_frames_are_never_dropped():
    good = _true_obs(range(380, 387))
    fake = [_obs(387 + i, (700.0 + 12.0 * i, 330.0)) for i in range(5)]
    kept, dropped = filter_reversed_observations(good + fake, _fit, protect_frames={389})
    assert 389 not in dropped
    assert any(o.frame == 389 for o in kept)


def test_frames_without_a_fit_are_not_judged():
    good = _true_obs(range(380, 387))
    fake = [_obs(387 + i, (700.0 + 12.0 * i, 330.0)) for i in range(5)]
    kept, dropped = filter_reversed_observations(
        good + fake, lambda f: None if f >= 387 else _fit(f))
    assert dropped == []


def test_disabled_and_short_inputs_are_passthrough():
    fake = [_obs(387 + i, (700.0 + 12.0 * i, 330.0)) for i in range(5)]
    good = _true_obs(range(380, 387))
    kept, dropped = filter_reversed_observations(
        good + fake, _fit, cfg={"enabled": False})
    assert dropped == [] and len(kept) == 12
    assert filter_reversed_observations([_obs(1, (0, 0))], _fit) == ([_obs(1, (0, 0))], [])


def test_min_run_is_configurable():
    # 387 and 388 each move against the fit (a 2-long reversed run)
    obs = _true_obs(range(380, 386)) + [
        _obs(386, (700.0, 340.0)), _obs(387, (712.0, 340.0)), _obs(388, (724.0, 340.0))]
    _, dropped2 = filter_reversed_observations(obs, _fit, cfg={"min_run": 2})
    _, dropped3 = filter_reversed_observations(obs, _fit, cfg={"min_run": 3})
    assert dropped3 == [] and dropped2 != []


def test_gap_between_detections_blocks_comparison():
    obs = _true_obs(range(380, 384)) + [_obs(392, (700.0, 340.0)), _obs(393, (712.0, 340.0))]
    _, dropped = filter_reversed_observations(obs, _fit)
    assert 392 not in dropped  # the 392 pair has no predecessor within max_gap


def test_default_cfg_is_enabled_and_overridable():
    assert direction_gate_cfg()["enabled"] is True
    assert direction_gate_cfg({"min_px": 9.0})["min_px"] == 9.0


def test_does_not_mutate_input_and_preserves_order():
    obs = _true_obs(range(380, 390))
    snapshot = list(obs)
    kept, _ = filter_reversed_observations(obs, _fit)
    assert obs == snapshot
    assert [o.frame for o in kept] == sorted(o.frame for o in kept)
