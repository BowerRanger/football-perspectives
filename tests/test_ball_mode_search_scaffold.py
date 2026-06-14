"""Task 2 TDD: scaffold tests for ball_mode_search.py.

Asserts:
- Mode enum integer ordering
- Construction of each frozen dataclass
- _mode_search_cfg({}) gives defaults
- _mode_search_cfg({"mode_search": {...}}) overrides specified keys, defaults the rest
"""

from __future__ import annotations

import numpy as np
import pytest

from src.utils.ball_mode_search import (
    Breakpoint,
    Hypothesis,
    Mode,
    ModeSearchCfg,
    Segment,
    _mode_search_cfg,
)


# ---------------------------------------------------------------------------
# Mode enum ordering
# ---------------------------------------------------------------------------

def test_mode_enum_integer_values():
    assert Mode.ROLLING.value == 0
    assert Mode.FLIGHT.value == 1
    assert Mode.POSSESSED.value == 2
    assert Mode.STATIONARY.value == 3
    assert Mode.OUT_OF_VIEW.value == 4


def test_mode_enum_ordering_as_tiebreak():
    """Lower integer value = earlier in deterministic ordering."""
    ordered = sorted(Mode, key=lambda m: m.value)
    assert ordered == [
        Mode.ROLLING,
        Mode.FLIGHT,
        Mode.POSSESSED,
        Mode.STATIONARY,
        Mode.OUT_OF_VIEW,
    ]


# ---------------------------------------------------------------------------
# Breakpoint dataclass
# ---------------------------------------------------------------------------

def test_breakpoint_construction_minimal():
    bp = Breakpoint(frame=10, kind="touch", event_score=0.85, is_manual=False)
    assert bp.frame == 10
    assert bp.kind == "touch"
    assert bp.event_score == 0.85
    assert bp.is_manual is False
    assert bp.player_branch is None


def test_breakpoint_construction_with_player():
    bp = Breakpoint(frame=42, kind="touch", event_score=0.9, is_manual=True, player_branch="P001")
    assert bp.player_branch == "P001"
    assert bp.is_manual is True


def test_breakpoint_synthetic_boundary():
    """Clip-boundary synthetic breakpoints carry event_score=0.0."""
    bp = Breakpoint(frame=0, kind="boundary", event_score=0.0, is_manual=False)
    assert bp.event_score == 0.0
    assert bp.kind == "boundary"


def test_breakpoint_is_frozen():
    bp = Breakpoint(frame=5, kind="bounce", event_score=0.7, is_manual=False)
    with pytest.raises((AttributeError, TypeError)):
        bp.frame = 99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Segment dataclass
# ---------------------------------------------------------------------------

def test_segment_construction():
    worlds = {10: np.array([1.0, 2.0, 0.0]), 11: np.array([1.5, 2.5, 0.0])}
    seg = Segment(
        fa=10,
        fb=11,
        mode=Mode.ROLLING,
        worlds=worlds,
        residual_px=2.3,
        underconstrained=False,
        kind="rolling",
        player_id=None,
        boundary_vel=(None, None),
    )
    assert seg.fa == 10
    assert seg.fb == 11
    assert seg.mode is Mode.ROLLING
    assert seg.worlds is worlds
    assert seg.residual_px == 2.3
    assert seg.underconstrained is False
    assert seg.kind == "rolling"
    assert seg.player_id is None
    assert seg.boundary_vel == (None, None)


def test_segment_flight():
    v0 = np.array([10.0, 0.0, 5.0])
    v1 = np.array([10.0, 0.0, -2.0])
    seg = Segment(
        fa=50,
        fb=75,
        mode=Mode.FLIGHT,
        worlds={50: np.array([0.0, 0.0, 1.0])},
        residual_px=4.1,
        underconstrained=False,
        kind="ballistic",
        player_id=None,
        boundary_vel=(v0, v1),
    )
    assert seg.mode is Mode.FLIGHT
    assert seg.kind == "ballistic"
    np.testing.assert_array_equal(seg.boundary_vel[0], v0)
    np.testing.assert_array_equal(seg.boundary_vel[1], v1)


def test_segment_possessed():
    seg = Segment(
        fa=100,
        fb=120,
        mode=Mode.POSSESSED,
        worlds={},
        residual_px=1.5,
        underconstrained=False,
        kind="possessed",
        player_id="P002",
        boundary_vel=(None, None),
    )
    assert seg.mode is Mode.POSSESSED
    assert seg.player_id == "P002"


def test_segment_out_of_view():
    seg = Segment(
        fa=200,
        fb=250,
        mode=Mode.OUT_OF_VIEW,
        worlds={},
        residual_px=0.0,
        underconstrained=True,
        kind="out_of_view",
        player_id=None,
        boundary_vel=(None, None),
    )
    assert seg.mode is Mode.OUT_OF_VIEW
    assert seg.underconstrained is True


# ---------------------------------------------------------------------------
# Hypothesis dataclass
# ---------------------------------------------------------------------------

def test_hypothesis_construction_empty():
    hyp = Hypothesis(
        last_bp_idx=-1,
        cur_mode=None,
        segments=(),
        cost=0.0,
        end_velocity=None,
        player_id=None,
    )
    assert hyp.last_bp_idx == -1
    assert hyp.cur_mode is None
    assert hyp.segments == ()
    assert hyp.cost == 0.0
    assert hyp.end_velocity is None
    assert hyp.player_id is None


def test_hypothesis_with_segments():
    seg = Segment(
        fa=0, fb=10, mode=Mode.ROLLING, worlds={},
        residual_px=1.0, underconstrained=False,
        kind="rolling", player_id=None, boundary_vel=(None, None),
    )
    vel = np.array([2.0, 0.0, 0.0])
    hyp = Hypothesis(
        last_bp_idx=0,
        cur_mode=Mode.ROLLING,
        segments=(seg,),
        cost=7.5,
        end_velocity=vel,
        player_id=None,
    )
    assert len(hyp.segments) == 1
    assert hyp.segments[0] is seg
    assert hyp.cost == 7.5
    np.testing.assert_array_equal(hyp.end_velocity, vel)


def test_hypothesis_is_frozen():
    hyp = Hypothesis(
        last_bp_idx=0, cur_mode=Mode.FLIGHT, segments=(),
        cost=5.0, end_velocity=None, player_id=None,
    )
    with pytest.raises((AttributeError, TypeError)):
        hyp.cost = 99.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# ModeSearchCfg defaults
# ---------------------------------------------------------------------------

def test_mode_search_cfg_defaults():
    cfg = ModeSearchCfg()
    assert cfg.beam_width == 8
    assert cfg.segment_cost_constant == 6.0
    assert cfg.unexplained_break_penalty == 10.0
    assert cfg.ignored_event_penalty == 8.0
    assert cfg.out_of_view_frame_penalty == 1.5
    assert cfg.possessed_tether_px == 40.0
    assert cfg.velocity_discontinuity_weight == 2.0
    assert cfg.max_segment_fit_calls == 20000


def test_mode_search_cfg_is_frozen():
    cfg = ModeSearchCfg()
    with pytest.raises((AttributeError, TypeError)):
        cfg.beam_width = 99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# _mode_search_cfg mapper
# ---------------------------------------------------------------------------

def test_mode_search_cfg_mapper_empty():
    """Empty outer cfg dict returns all defaults."""
    result = _mode_search_cfg({})
    defaults = ModeSearchCfg()
    assert result.beam_width == defaults.beam_width
    assert result.segment_cost_constant == defaults.segment_cost_constant
    assert result.unexplained_break_penalty == defaults.unexplained_break_penalty
    assert result.ignored_event_penalty == defaults.ignored_event_penalty
    assert result.out_of_view_frame_penalty == defaults.out_of_view_frame_penalty
    assert result.possessed_tether_px == defaults.possessed_tether_px
    assert result.velocity_discontinuity_weight == defaults.velocity_discontinuity_weight
    assert result.max_segment_fit_calls == defaults.max_segment_fit_calls


def test_mode_search_cfg_mapper_partial_overrides():
    """Only overridden fields change; rest remain defaults."""
    result = _mode_search_cfg({"mode_search": {"beam_width": 16, "possessed_tether_px": 20.0}})
    defaults = ModeSearchCfg()
    assert result.beam_width == 16
    assert result.possessed_tether_px == 20.0
    # Everything else is default
    assert result.segment_cost_constant == defaults.segment_cost_constant
    assert result.unexplained_break_penalty == defaults.unexplained_break_penalty
    assert result.ignored_event_penalty == defaults.ignored_event_penalty
    assert result.out_of_view_frame_penalty == defaults.out_of_view_frame_penalty
    assert result.velocity_discontinuity_weight == defaults.velocity_discontinuity_weight
    assert result.max_segment_fit_calls == defaults.max_segment_fit_calls


def test_mode_search_cfg_mapper_all_fields():
    """All fields can be overridden."""
    overrides = {
        "beam_width": 4,
        "segment_cost_constant": 3.0,
        "unexplained_break_penalty": 5.0,
        "ignored_event_penalty": 4.0,
        "out_of_view_frame_penalty": 0.5,
        "possessed_tether_px": 60.0,
        "velocity_discontinuity_weight": 1.0,
        "max_segment_fit_calls": 5000,
    }
    result = _mode_search_cfg({"mode_search": overrides})
    assert result.beam_width == 4
    assert result.segment_cost_constant == 3.0
    assert result.unexplained_break_penalty == 5.0
    assert result.ignored_event_penalty == 4.0
    assert result.out_of_view_frame_penalty == 0.5
    assert result.possessed_tether_px == 60.0
    assert result.velocity_discontinuity_weight == 1.0
    assert result.max_segment_fit_calls == 5000


def test_mode_search_cfg_mapper_ignores_unrelated_keys():
    """Unrelated top-level cfg keys don't affect the mapper."""
    result = _mode_search_cfg({"camera": {"foo": 1}, "ball": {"second_pass": {}}})
    defaults = ModeSearchCfg()
    assert result.beam_width == defaults.beam_width
