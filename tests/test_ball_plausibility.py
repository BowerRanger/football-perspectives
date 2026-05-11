"""Unit tests for ball trajectory plausibility checks (Layer 1)."""

from __future__ import annotations

import numpy as np
import pytest

from src.utils.ball_plausibility import (
    PitchDims,
    PlausibilityCfg,
    is_plausible_trajectory,
)


def _cfg(**over) -> PlausibilityCfg:
    base = dict(
        z_max_m=50.0,
        horizontal_speed_max_m_s=40.0,
        pitch_margin_m=5.0,
    )
    base.update(over)
    return PlausibilityCfg(**base)


def _pitch() -> PitchDims:
    return PitchDims(length_m=105.0, width_m=68.0)


def test_plausible_parabola_apex_10m_25mps():
    # Apex ~10 m above pitch centre, horizontal speed ~25 m/s along x.
    p0 = np.array([-30.0, 0.0, 0.11])
    v0 = np.array([25.0, 0.0, 14.0])  # 14 m/s upward → apex ≈ 10 m
    duration = 2.5
    assert is_plausible_trajectory(
        p0, v0, omega=None, duration_s=duration, fps=30.0,
        cfg=_cfg(), pitch=_pitch(),
    )


def test_rejects_off_pitch_p0():
    p0 = np.array([-200.0, 0.0, 0.11])
    v0 = np.array([0.0, 0.0, 5.0])
    assert not is_plausible_trajectory(
        p0, v0, omega=None, duration_s=1.0, fps=30.0,
        cfg=_cfg(), pitch=_pitch(),
    )


def test_rejects_excessive_speed():
    p0 = np.array([0.0, 0.0, 0.11])
    v0 = np.array([250.0, 0.0, 5.0])
    assert not is_plausible_trajectory(
        p0, v0, omega=None, duration_s=0.5, fps=30.0,
        cfg=_cfg(), pitch=_pitch(),
    )


def test_rejects_z_above_max():
    p0 = np.array([0.0, 0.0, 0.11])
    v0 = np.array([0.0, 0.0, 60.0])  # ~180 m apex
    assert not is_plausible_trajectory(
        p0, v0, omega=None, duration_s=2.0, fps=30.0,
        cfg=_cfg(), pitch=_pitch(),
    )


def test_rejects_z_far_below_ground():
    p0 = np.array([0.0, 0.0, 0.11])
    v0 = np.array([0.0, 0.0, -50.0])  # plunges through ground
    assert not is_plausible_trajectory(
        p0, v0, omega=None, duration_s=1.0, fps=30.0,
        cfg=_cfg(), pitch=_pitch(),
    )


def test_accepts_p0_just_inside_margin():
    # Half-length 52.5; +5 m margin → boundary at 57.5. 56 should pass.
    p0 = np.array([56.0, 0.0, 0.11])
    v0 = np.array([0.0, 0.0, 5.0])
    assert is_plausible_trajectory(
        p0, v0, omega=None, duration_s=0.5, fps=30.0,
        cfg=_cfg(), pitch=_pitch(),
    )


def test_rejects_p0_just_outside_margin():
    p0 = np.array([59.0, 0.0, 0.11])
    v0 = np.array([0.0, 0.0, 5.0])
    assert not is_plausible_trajectory(
        p0, v0, omega=None, duration_s=0.5, fps=30.0,
        cfg=_cfg(), pitch=_pitch(),
    )


def test_zero_duration_returns_false():
    p0 = np.array([0.0, 0.0, 0.11])
    v0 = np.array([10.0, 0.0, 5.0])
    assert not is_plausible_trajectory(
        p0, v0, omega=None, duration_s=0.0, fps=30.0,
        cfg=_cfg(), pitch=_pitch(),
    )


def test_billion_metre_p0_rejected():
    # Reproduces seg-3 garbage from origi01.
    p0 = np.array([-5_690_504.0, 9_399_056.0, -2_218_511.0])
    v0 = np.array([3_745_003.0, 3_366_928.0, -698_927.0])
    assert not is_plausible_trajectory(
        p0, v0, omega=None, duration_s=0.2, fps=30.0,
        cfg=_cfg(), pitch=_pitch(),
    )
