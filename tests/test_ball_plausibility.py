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


def test_rejects_nan_inputs():
    p0 = np.array([np.nan, 0.0, 0.11])
    v0 = np.array([10.0, 0.0, 5.0])
    assert not is_plausible_trajectory(
        p0, v0, omega=None, duration_s=1.0, fps=30.0,
        cfg=_cfg(), pitch=_pitch(),
    )


def test_rejects_inf_inputs():
    p0 = np.array([0.0, 0.0, 0.11])
    v0 = np.array([np.inf, 0.0, 5.0])
    assert not is_plausible_trajectory(
        p0, v0, omega=None, duration_s=1.0, fps=30.0,
        cfg=_cfg(), pitch=_pitch(),
    )


# ---------------------------------------------------------------------------
# Layer 2 — find_implausible_grounded_runs
# ---------------------------------------------------------------------------

from src.utils.ball_plausibility import (
    GroundedRun,
    GroundPromotionCfg,
    find_implausible_grounded_runs,
)


def _promote_cfg(**over) -> GroundPromotionCfg:
    base = dict(
        enabled=True,
        min_run_frames=6,
        off_pitch_margin_m=5.0,
        max_ground_speed_m_s=35.0,
    )
    base.update(over)
    return GroundPromotionCfg(**base)


def test_no_runs_when_ground_motion_is_credible():
    # Rolling along the pitch at 5 m/s — well within bounds.
    xyzs = {
        i: (np.array([10.0 + 5.0 * i / 30.0, 0.0, 0.11]), 0.5)
        for i in range(20)
    }
    states = {i: "grounded" for i in range(20)}
    runs = find_implausible_grounded_runs(
        per_frame_xyz=xyzs,
        per_frame_state=states,
        fps=30.0,
        cfg=_promote_cfg(),
        pitch=_pitch(),
    )
    assert runs == []


def test_flags_off_pitch_run():
    # Ground-projection at y=40 (well past 34 + 5 margin = 39).
    xyzs = {
        i: (np.array([0.0, 40.5, 0.11]), 0.5)
        for i in range(10)
    }
    states = {i: "grounded" for i in range(10)}
    runs = find_implausible_grounded_runs(
        per_frame_xyz=xyzs,
        per_frame_state=states,
        fps=30.0,
        cfg=_promote_cfg(),
        pitch=_pitch(),
    )
    assert len(runs) == 1
    assert runs[0].start == 0 and runs[0].end == 9


def test_flags_speed_exceeding_run():
    # 40 m/s ground speed (above 35).
    xyzs = {
        i: (np.array([40.0 * i / 30.0, 0.0, 0.11]), 0.5)
        for i in range(10)
    }
    states = {i: "grounded" for i in range(10)}
    runs = find_implausible_grounded_runs(
        per_frame_xyz=xyzs,
        per_frame_state=states,
        fps=30.0,
        cfg=_promote_cfg(),
        pitch=_pitch(),
    )
    assert len(runs) == 1


def test_ignores_runs_shorter_than_min_run_frames():
    xyzs = {
        i: (np.array([0.0, 40.5, 0.11]), 0.5)
        for i in range(4)
    }
    states = {i: "grounded" for i in range(4)}
    runs = find_implausible_grounded_runs(
        per_frame_xyz=xyzs,
        per_frame_state=states,
        fps=30.0,
        cfg=_promote_cfg(min_run_frames=6),
        pitch=_pitch(),
    )
    assert runs == []


def test_run_terminates_at_non_grounded_state():
    xyzs = {
        i: (np.array([0.0, 40.5, 0.11]), 0.5)
        for i in range(20)
    }
    states = {i: "grounded" for i in range(20)}
    states[8] = "missing"
    states[9] = "missing"
    runs = find_implausible_grounded_runs(
        per_frame_xyz=xyzs,
        per_frame_state=states,
        fps=30.0,
        cfg=_promote_cfg(),
        pitch=_pitch(),
    )
    # Two qualifying runs: 0..7 (length 8) and 10..19 (length 10).
    assert len(runs) == 2
    assert (runs[0].start, runs[0].end) == (0, 7)
    assert (runs[1].start, runs[1].end) == (10, 19)
