"""Replay-sync decisions honour the estimate's rate uncertainty."""

from __future__ import annotations

from src.utils import replay_sync_group as g
from src.utils.replay_speed import SpeedEstimate


def _est(rate, unc, window=100.0):
    return SpeedEstimate(rate=rate, offset=100.0, cost_m=1.3, coverage=1.0, margin=0.4,
                         confidence=0.8, rate_first=rate, rate_second=rate, ramp=False,
                         n_replay_frames=200, live_window_frames=window, rate_uncertainty=unc)


def _decided(est):
    m = g.MemberResult(shot_id="r", against="L", estimate=est, decision="",
                       placement=(est.rate, est.offset))
    g._decide(m, {}, dict(g.DEFAULTS))
    return m


def test_clearly_slow_but_imprecise_is_retimed_and_flagged_approximate():
    m = _decided(_est(0.34, 0.08))
    assert m.decision == "applied_retimed"
    assert m.approximate
    assert "marked moments" in m.reason


def test_not_clearly_slow_within_uncertainty_is_not_retimed():
    # 0.88 looks slow, but +-2 sigma (0.10) reaches real time
    m = _decided(_est(0.88, 0.05))
    assert m.decision == "applied"
    assert m.retime_rate is None


def test_precise_slow_rate_is_not_approximate():
    m = _decided(_est(0.5, 0.01, window=600))
    assert m.decision == "applied_retimed"
    assert not m.approximate
