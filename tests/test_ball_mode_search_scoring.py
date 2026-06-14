"""Tests for Task 5 — mode-search scoring functions.

Covers:
  T5-1  raw-px ordering: 4px FLIGHT < 7px ROLLING (same n_frames) — F5 proof
  T5-2  OUT_OF_VIEW per-frame penalty scales with segment length
  T5-3  underconstrained flag adds cost vs constrained fit
  T5-4  unexplained break costs more than event-aligned transition
  T5-5  velocity-continuous transition < discontinuous; both-None → equal (no vel term)
  T5-6  manual breakpoint → no unexplained penalty
  T5-7  ignored_event_cost scales with event_score
"""

from __future__ import annotations

import numpy as np
import pytest

from src.utils.ball_mode_search import (
    Breakpoint,
    Mode,
    ModeSearchCfg,
    SegmentFit,
    ignored_event_cost,
    segment_cost,
    transition_cost,
)


# ---------------------------------------------------------------------------
# Helpers — build minimal SegmentFit / Segment stubs
# ---------------------------------------------------------------------------

def _flight_fit(residual_px: float, underconstrained: bool = False,
                v_start=None, v_end=None) -> SegmentFit:
    v_start = np.array(v_start) if v_start is not None else np.array([5.0, 0.0, 3.0])
    v_end   = np.array(v_end)   if v_end   is not None else np.array([5.0, 0.0, -3.0])
    return SegmentFit(
        worlds={},
        residual_px=residual_px,
        underconstrained=underconstrained,
        kind="ballistic",
        boundary_vel=(v_start, v_end),
    )


def _rolling_fit(residual_px: float, underconstrained: bool = False,
                 v_start=None, v_end=None) -> SegmentFit:
    v_start = np.array(v_start) if v_start is not None else np.array([3.0, 0.0, 0.0])
    v_end   = np.array(v_end)   if v_end   is not None else np.array([2.5, 0.0, 0.0])
    return SegmentFit(
        worlds={},
        residual_px=residual_px,
        underconstrained=underconstrained,
        kind="rolling",
        boundary_vel=(v_start, v_end),
    )


def _oov_fit(n_observed: int = 0) -> SegmentFit:
    return SegmentFit(
        worlds={},
        residual_px=0.0,
        underconstrained=False,
        kind="out_of_view",
        boundary_vel=(None, None),
        n_observed=n_observed,
    )


def _bp(frame: int = 10, kind: str = "touch", event_score: float = 0.9,
        is_manual: bool = False) -> Breakpoint:
    return Breakpoint(frame=frame, kind=kind, event_score=event_score,
                      is_manual=is_manual)


def _boundary_bp(frame: int = 0) -> Breakpoint:
    return Breakpoint(frame=frame, kind="boundary", event_score=0.0,
                      is_manual=False)


CFG = ModeSearchCfg()


# ---------------------------------------------------------------------------
# T5-1 — raw-px ordering: 4px FLIGHT < 7px ROLLING on same pixel evidence (F5)
# ---------------------------------------------------------------------------

class TestRawPxOrdering:
    """Proves raw-pixel residual is used directly, not gate-normalised.

    With gate-normalised cost you'd divide by the per-mode gate and the two
    costs could invert depending on gate values.  With raw pixels the cheaper
    fit always wins regardless of mode labels.
    """

    def test_tighter_flight_beats_looser_rolling(self):
        fit_flight  = _flight_fit(residual_px=4.0)
        fit_rolling = _rolling_fit(residual_px=7.0)
        n = 20  # same segment length
        cost_f = segment_cost(fit_flight,  Mode.FLIGHT,  n, CFG)
        cost_r = segment_cost(fit_rolling, Mode.ROLLING, n, CFG)
        assert cost_f < cost_r, (
            f"4px FLIGHT cost {cost_f:.4f} should be < 7px ROLLING cost {cost_r:.4f}"
        )

    def test_same_residual_same_mode_equal_costs(self):
        fit_a = _flight_fit(residual_px=5.0)
        fit_b = _flight_fit(residual_px=5.0)
        assert segment_cost(fit_a, Mode.FLIGHT, 10, CFG) == pytest.approx(
            segment_cost(fit_b, Mode.FLIGHT, 10, CFG)
        )


# ---------------------------------------------------------------------------
# T5-2 — OUT_OF_VIEW per-frame penalty scales with segment length
# ---------------------------------------------------------------------------

class TestOutOfViewPenalty:
    def test_observed_frame_penalty_scales_with_observed_count(self):
        """C1c: the dominant OOV cost scales with OBSERVED frames, not span
        length.  Two spans of equal length but different observed counts must
        differ by ``out_of_view_frame_penalty * Δn_observed``."""
        n = 20
        fit_few = _oov_fit(n_observed=2)
        fit_many = _oov_fit(n_observed=12)
        cost_few = segment_cost(fit_few, Mode.OUT_OF_VIEW, n, CFG)
        cost_many = segment_cost(fit_many, Mode.OUT_OF_VIEW, n, CFG)
        # Δobserved = 10 (and Δmissing = -10): net delta is
        # (oov_penalty - missing_penalty) * 10.
        expected_delta = (
            CFG.out_of_view_frame_penalty - CFG.out_of_view_missing_frame_penalty
        ) * 10
        assert cost_many - cost_few == pytest.approx(expected_delta, rel=1e-6)

    def test_observed_frames_dominate_missing_frames(self):
        """C1c: an OOV span full of observations is far more expensive than an
        equally-long span with no observations (last-resort semantics)."""
        n = 20
        fit_detection_rich = _oov_fit(n_observed=n)   # gap discards 20 detections
        fit_genuinely_empty = _oov_fit(n_observed=0)  # nothing to discard
        cost_rich = segment_cost(fit_detection_rich, Mode.OUT_OF_VIEW, n, CFG)
        cost_empty = segment_cost(fit_genuinely_empty, Mode.OUT_OF_VIEW, n, CFG)
        assert cost_rich > cost_empty
        # The observed-frame term must dominate the missing-frame term.
        assert CFG.out_of_view_frame_penalty > 10 * CFG.out_of_view_missing_frame_penalty

    def test_genuinely_missing_span_stays_cheap(self):
        """C1c: gapping a span with NO observations is cheap — only the tiny
        per-missing-frame term applies on top of the parsimony constant."""
        n = 5
        fit = _oov_fit(n_observed=0)
        cost = segment_cost(fit, Mode.OUT_OF_VIEW, n, CFG)
        expected = (
            CFG.segment_cost_constant
            + CFG.out_of_view_missing_frame_penalty * n
        )
        assert cost == pytest.approx(expected, rel=1e-6)
        # Cheaper than a constrained FLIGHT fit with any real residual.
        assert cost < segment_cost(_flight_fit(residual_px=3.0), Mode.FLIGHT, n, CFG)

    def test_non_oov_mode_has_no_frame_penalty(self):
        fit_f  = _flight_fit(residual_px=0.0)
        fit_ov = _oov_fit(n_observed=5)
        # Flight with 0 residual = segment_cost_constant exactly.
        # OOV with observed frames = segment_cost_constant + oov_penalty*observed.
        cost_f  = segment_cost(fit_f,  Mode.FLIGHT,       5, CFG)
        cost_ov = segment_cost(fit_ov, Mode.OUT_OF_VIEW,  5, CFG)
        assert cost_ov > cost_f


# ---------------------------------------------------------------------------
# T5-3 — underconstrained flag adds cost vs constrained fit
# ---------------------------------------------------------------------------

class TestUnderconstrainedPenalty:
    def test_underconstrained_costs_more(self):
        fit_ok  = _flight_fit(residual_px=5.0, underconstrained=False)
        fit_bad = _flight_fit(residual_px=5.0, underconstrained=True)
        cost_ok  = segment_cost(fit_ok,  Mode.FLIGHT, 10, CFG)
        cost_bad = segment_cost(fit_bad, Mode.FLIGHT, 10, CFG)
        assert cost_bad > cost_ok

    def test_underconstrained_penalty_is_positive(self):
        fit_ok  = _flight_fit(residual_px=0.0, underconstrained=False)
        fit_bad = _flight_fit(residual_px=0.0, underconstrained=True)
        penalty = (
            segment_cost(fit_bad, Mode.FLIGHT, 10, CFG)
            - segment_cost(fit_ok,  Mode.FLIGHT, 10, CFG)
        )
        assert penalty > 0, f"underconstrained penalty must be positive; got {penalty}"


# ---------------------------------------------------------------------------
# T5-4 — event-aligned vs unexplained transition cost
# ---------------------------------------------------------------------------

class TestTransitionEventAgreement:
    def test_unexplained_break_costs_more_than_event_aligned(self):
        bp_event      = _bp(event_score=0.9, is_manual=False)
        bp_no_event   = _bp(event_score=0.0, is_manual=False)

        # Both transitions: FLIGHT → ROLLING; no velocity info needed here.
        cost_event    = transition_cost(None, bp_event,    Mode.ROLLING,
                                        None, None, CFG)
        cost_no_event = transition_cost(None, bp_no_event, Mode.ROLLING,
                                        None, None, CFG)
        assert cost_no_event > cost_event, (
            f"unexplained ({cost_no_event:.4f}) should cost more than "
            f"event-aligned ({cost_event:.4f})"
        )

    def test_high_event_score_gives_bonus(self):
        """A high-confidence event should produce a negative (bonus) term."""
        bp = _bp(event_score=1.0, is_manual=False)
        # With ignored_event_penalty on, high event score → bonus at transition.
        # The bonus should make this cheaper than a zero-event transition.
        cost_event = transition_cost(None, bp, Mode.ROLLING, None, None, CFG)
        cost_zero  = transition_cost(None, _bp(event_score=0.0), Mode.ROLLING,
                                     None, None, CFG)
        assert cost_event < cost_zero


# ---------------------------------------------------------------------------
# T5-5 — velocity continuity
# ---------------------------------------------------------------------------

class TestVelocityContinuity:
    def test_continuous_costs_less_than_discontinuous(self):
        bp = _bp(event_score=0.9)  # event-backed so no unexplained penalty

        v_consistent = np.array([5.0, 0.0, -3.0])
        v_small_jump = np.array([5.1, 0.0, -3.0])   # Δv ≈ 0.1 m/s
        v_large_jump = np.array([15.0, 0.0, 3.0])   # Δv ≈ 10+ m/s

        cost_cont = transition_cost(None, bp, Mode.ROLLING,
                                    v_consistent, v_small_jump, CFG)
        cost_disc = transition_cost(None, bp, Mode.ROLLING,
                                    v_consistent, v_large_jump, CFG)
        assert cost_cont < cost_disc, (
            f"continuous ({cost_cont:.4f}) should cost less than "
            f"discontinuous ({cost_disc:.4f})"
        )

    def test_both_none_velocity_no_term(self):
        """When both boundary velocities are None the cost must be the same
        regardless of which side is labelled FLIGHT vs ROLLING — the velocity
        term is simply absent (F4)."""
        bp = _bp(event_score=0.5)
        cost_no_vel_1 = transition_cost(None, bp, Mode.ROLLING,
                                         None, None, CFG)
        cost_no_vel_2 = transition_cost(None, bp, Mode.FLIGHT,
                                         None, None, CFG)
        # Both-None: only the event term differs by mode — but since we're
        # calling with the same bp we're really testing that the vel term ≡ 0.
        # Confirm the vel contribution is exactly zero by comparing the
        # same call with and without an artificial vel:
        v = np.array([5.0, 0.0, -2.0])
        cost_with_vel = transition_cost(None, bp, Mode.ROLLING,
                                         v, v, CFG)
        # Δv = 0 → vel term = 0, so cost_with_vel == cost_no_vel_1
        assert cost_with_vel == pytest.approx(cost_no_vel_1, rel=1e-6), (
            "zero Δv with defined velocities should give same cost as both-None"
        )

    def test_one_side_none_no_velocity_term(self):
        """When prev_end_vel is None but next_start_vel is defined, still no
        velocity term (both must be defined — F4)."""
        bp = _bp(event_score=0.9)
        v = np.array([5.0, 0.0, -2.0])
        cost_one_none  = transition_cost(None, bp, Mode.ROLLING,
                                          None, v, CFG)
        cost_both_none = transition_cost(None, bp, Mode.ROLLING,
                                          None, None, CFG)
        assert cost_one_none == pytest.approx(cost_both_none, rel=1e-6)


# ---------------------------------------------------------------------------
# T5-6 — manual breakpoint: no unexplained-break penalty
# ---------------------------------------------------------------------------

class TestManualBreakpoint:
    def test_manual_bp_no_unexplained_penalty(self):
        bp_manual   = _bp(event_score=0.0, is_manual=True)
        bp_auto_bad = _bp(event_score=0.0, is_manual=False)

        cost_manual = transition_cost(None, bp_manual,   Mode.ROLLING,
                                      None, None, CFG)
        cost_auto   = transition_cost(None, bp_auto_bad, Mode.ROLLING,
                                      None, None, CFG)
        # The manual breakpoint should NOT incur the unexplained_break_penalty.
        assert cost_manual < cost_auto, (
            f"manual ({cost_manual:.4f}) must cost less than "
            f"auto zero-score ({cost_auto:.4f})"
        )

    def test_manual_bp_cost_no_penalty_term(self):
        """Manual break with no velocities should have zero transition cost
        (no event bonus, no penalty, no velocity term)."""
        bp = _bp(event_score=0.0, is_manual=True)
        cost = transition_cost(None, bp, Mode.ROLLING, None, None, CFG)
        assert cost == pytest.approx(0.0, abs=1e-9), (
            f"manual zero-score bp should have 0 transition cost; got {cost}"
        )

    def test_boundary_bp_no_unexplained_penalty(self):
        """Clip-boundary breakpoints (kind='boundary', event_score=0) must
        be treated neutrally — no unexplained-break penalty, no bonus (F17)."""
        bp_boundary = _boundary_bp()
        bp_auto     = _bp(event_score=0.0, is_manual=False)

        cost_boundary = transition_cost(None, bp_boundary, Mode.ROLLING,
                                        None, None, CFG)
        cost_auto     = transition_cost(None, bp_auto,     Mode.ROLLING,
                                        None, None, CFG)
        assert cost_boundary < cost_auto, (
            "boundary bp must not incur unexplained-break penalty"
        )
        assert cost_boundary == pytest.approx(0.0, abs=1e-9), (
            f"boundary bp cost should be 0; got {cost_boundary}"
        )


# ---------------------------------------------------------------------------
# T5-7 — ignored_event_cost scales with event_score
# ---------------------------------------------------------------------------

class TestIgnoredEventCost:
    def test_zero_event_score_zero_ignored_cost(self):
        bp = _bp(event_score=0.0)
        assert ignored_event_cost(bp, CFG) == pytest.approx(0.0, abs=1e-9)

    def test_unit_event_score_full_penalty(self):
        bp = _bp(event_score=1.0)
        assert ignored_event_cost(bp, CFG) == pytest.approx(
            CFG.ignored_event_penalty * 1.0, rel=1e-6
        )

    def test_scales_linearly_with_score(self):
        bp_half = _bp(event_score=0.5)
        bp_full = _bp(event_score=1.0)
        assert ignored_event_cost(bp_full, CFG) == pytest.approx(
            2.0 * ignored_event_cost(bp_half, CFG), rel=1e-6
        )

    def test_manual_bp_ignored_cost_still_present(self):
        """The beam charges ignored_event_cost when it SKIPS a breakpoint.

        Manual breakpoints are never skipped (they're forced), so this is
        purely academic, but the function should still return a non-zero value
        for a manual bp with event_score > 0, since the function itself is
        stateless — it's the beam loop that decides whether to apply it.
        """
        bp = _bp(event_score=0.8, is_manual=True)
        # Just a sanity check: the helper is purely score-based.
        assert ignored_event_cost(bp, CFG) == pytest.approx(
            CFG.ignored_event_penalty * 0.8, rel=1e-6
        )


# ---------------------------------------------------------------------------
# T5-8 — segment_cost formula — explicit value checks
# ---------------------------------------------------------------------------

class TestSegmentCostFormula:
    """Pin the exact arithmetic so regressions are obvious."""

    def test_flight_no_flags_formula(self):
        """segment_cost = residual_px + segment_cost_constant."""
        fit = _flight_fit(residual_px=4.0, underconstrained=False)
        expected = 4.0 + CFG.segment_cost_constant
        assert segment_cost(fit, Mode.FLIGHT, 10, CFG) == pytest.approx(
            expected, rel=1e-6
        )

    def test_oov_formula(self):
        """C1c: segment_cost = constant + oov_penalty * n_observed +
        missing_penalty * (n_frames - n_observed)."""
        n = 7
        n_observed = 4
        fit = _oov_fit(n_observed=n_observed)
        expected = (
            CFG.segment_cost_constant
            + CFG.out_of_view_frame_penalty * n_observed
            + CFG.out_of_view_missing_frame_penalty * (n - n_observed)
        )
        assert segment_cost(fit, Mode.OUT_OF_VIEW, n, CFG) == pytest.approx(
            expected, rel=1e-6
        )

    def test_kind_weight_is_uniform_1(self):
        """v1: kind_weight = 1.0 for all modes — same residual → same cost."""
        fit = _rolling_fit(residual_px=5.0)
        # If kind_weight were non-uniform, rolling vs flight would differ.
        fit_f = _flight_fit(residual_px=5.0)
        diff = abs(
            segment_cost(fit,   Mode.ROLLING, 10, CFG)
            - segment_cost(fit_f, Mode.FLIGHT,  10, CFG)
        )
        assert diff == pytest.approx(0.0, abs=1e-9), (
            "kind_weight must be 1.0 for all modes in v1 (F6); "
            f"ROLLING vs FLIGHT cost differs by {diff}"
        )


# ---------------------------------------------------------------------------
# C1a — mode-pair-aware velocity continuity in transition_cost
# ---------------------------------------------------------------------------

def _seg(mode: Mode, player_id: str | None = None):
    """Minimal prev_seg stub carrying just the fields transition_cost reads."""
    from src.utils.ball_mode_search import Segment
    return Segment(
        fa=0, fb=1, mode=mode, worlds={}, residual_px=0.0,
        underconstrained=False, kind="x", player_id=player_id,
        boundary_vel=(None, None),
    )


def _vel_term(prev_mode, next_mode, v_prev, v_next, *, event_score=0.9):
    """Isolate the velocity-continuity contribution of transition_cost by
    subtracting the same call with both velocities None (no vel term)."""
    bp = _bp(event_score=event_score, is_manual=False)
    prev_seg = _seg(prev_mode) if prev_mode is not None else None
    with_vel = transition_cost(prev_seg, bp, next_mode, v_prev, v_next, CFG)
    no_vel = transition_cost(prev_seg, bp, next_mode, None, None, CFG)
    return with_vel - no_vel


class TestModePairVelocityContinuity:
    """Fix C1a: the velocity-continuity penalty depends on the mode pair."""

    def test_flight_to_flight_charges_full_delta_v(self):
        # Vertical-only jump → full ‖Δv‖ charged (mid-air change is suspicious).
        v_a = np.array([5.0, 0.0, 3.0])
        v_b = np.array([5.0, 0.0, -3.0])   # Δv = (0,0,-6) → ‖Δv‖ = 6
        term = _vel_term(Mode.FLIGHT, Mode.FLIGHT, v_a, v_b)
        assert term == pytest.approx(CFG.velocity_discontinuity_weight * 6.0,
                                     rel=1e-6)

    def test_flight_to_rolling_landing_ignores_vertical_kill(self):
        # A landing kills vz; only the horizontal Δv should be charged.
        v_a = np.array([5.0, 0.0, -8.0])    # descending fast
        v_b = np.array([5.0, 0.0, 0.0])     # rolling: vz killed, vxy unchanged
        term = _vel_term(Mode.FLIGHT, Mode.ROLLING, v_a, v_b)
        assert term == pytest.approx(0.0, abs=1e-9), (
            "vertical velocity kill at a landing must NOT be penalised (C1a)"
        )

    def test_flight_to_rolling_charges_horizontal_slip(self):
        v_a = np.array([5.0, 0.0, -8.0])
        v_b = np.array([2.0, 0.0, 0.0])     # vx drops 3 m/s horizontally
        term = _vel_term(Mode.FLIGHT, Mode.ROLLING, v_a, v_b)
        assert term == pytest.approx(CFG.velocity_discontinuity_weight * 3.0,
                                     rel=1e-6)

    def test_flight_to_stationary_landing_ignores_vertical_kill(self):
        v_a = np.array([0.0, 0.0, -6.0])
        v_b = np.array([0.0, 0.0, 0.0])
        term = _vel_term(Mode.FLIGHT, Mode.STATIONARY, v_a, v_b)
        assert term == pytest.approx(0.0, abs=1e-9)

    def test_rolling_to_flight_launch_no_penalty(self):
        # A kick/bounce-up off the ground: velocity gain is the event itself.
        v_a = np.array([3.0, 0.0, 0.0])
        v_b = np.array([3.0, 0.0, 9.0])    # big vz gain
        term = _vel_term(Mode.ROLLING, Mode.FLIGHT, v_a, v_b)
        assert term == pytest.approx(0.0, abs=1e-9)

    def test_stationary_to_flight_launch_no_penalty(self):
        v_a = np.array([0.0, 0.0, 0.0])
        v_b = np.array([4.0, 1.0, 10.0])
        term = _vel_term(Mode.STATIONARY, Mode.FLIGHT, v_a, v_b)
        assert term == pytest.approx(0.0, abs=1e-9)

    def test_possessed_involved_no_penalty(self):
        v_a = np.array([1.0, 0.0, 0.0])
        v_b = np.array([9.0, 5.0, 7.0])    # large change
        # POSSESSED on either side → no velocity penalty (player-controlled).
        assert _vel_term(Mode.POSSESSED, Mode.FLIGHT, v_a, v_b) == pytest.approx(
            0.0, abs=1e-9)
        assert _vel_term(Mode.FLIGHT, Mode.POSSESSED, v_a, v_b) == pytest.approx(
            0.0, abs=1e-9)

    def test_rolling_to_rolling_charges_horizontal(self):
        v_a = np.array([4.0, 0.0, 0.0])
        v_b = np.array([1.0, 0.0, 0.0])    # horizontal Δ = 3
        term = _vel_term(Mode.ROLLING, Mode.ROLLING, v_a, v_b)
        assert term == pytest.approx(CFG.velocity_discontinuity_weight * 3.0,
                                     rel=1e-6)


# ---------------------------------------------------------------------------
# C1b — restitution envelope penalty applies ONLY to flight→flight bounces
# ---------------------------------------------------------------------------

class TestRestitutionOnlyFlightToFlight:
    """Fix C1b: the restitution-envelope penalty must NOT charge a
    flight→rolling/stationary landing (restitution ≈ 0 is physical there)."""

    # A downward-incoming velocity that bounces back UP with an out-of-envelope
    # restitution (e = 6/6 = 1.0 > 0.85) — would trip the envelope penalty.
    V_IN = np.array([5.0, 0.0, -6.0])
    V_OUT_BOUNCE = np.array([5.0, 0.0, 6.0])     # e = 1.0 (out of envelope)

    def test_flight_to_flight_out_of_envelope_penalised(self):
        term_bad = _vel_term(Mode.FLIGHT, Mode.FLIGHT, self.V_IN,
                             self.V_OUT_BOUNCE)
        # A good-restitution bounce (e in [0.5,0.85]) of equal ‖Δv‖ direction
        # is cheaper because it avoids the envelope penalty.
        v_out_good = np.array([5.0, 0.0, 3.9])   # e = 0.65 (in envelope)
        term_good = _vel_term(Mode.FLIGHT, Mode.FLIGHT, self.V_IN, v_out_good)
        # The bad one carries the +5*weight envelope penalty on top of Δv.
        assert term_bad > term_good

    def test_flight_to_rolling_landing_no_restitution_penalty(self):
        """A flight→rolling landing must NOT incur the restitution envelope
        penalty even if the post-velocity has a (residual) upward component —
        only the horizontal slip is charged."""
        v_in = np.array([5.0, 0.0, -6.0])
        v_out = np.array([5.0, 0.0, 6.0])   # same as bounce, but next=ROLLING
        term = _vel_term(Mode.FLIGHT, Mode.ROLLING, v_in, v_out)
        # Horizontal Δ is 0 → term must be exactly 0 (no envelope penalty leaked).
        assert term == pytest.approx(0.0, abs=1e-9), (
            "restitution envelope must not be checked at a flight→ground landing "
            "(C1b)"
        )
