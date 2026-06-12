"""TrackerStep.pos_cov: blended position covariance for corridor gating.

The second-pass corridor (ball_second_pass.py) fuses forward and
backward IMM passes; it needs each step's positional uncertainty. Pins:
covariance present once initialised, grows during a detection gap, and
shrinks again when detections resume.
"""

from __future__ import annotations

import pytest

from src.utils.ball_tracker import BallTracker, TrackerStep


@pytest.mark.unit
def test_pos_cov_none_before_first_detection():
    tracker = BallTracker()
    step = tracker.update(0, None)
    assert step.pos_cov is None


@pytest.mark.unit
def test_pos_cov_emitted_and_grows_during_gap():
    tracker = BallTracker(max_gap_frames=100)
    covs = []
    for i in range(10):
        step = tracker.update(i, (100.0 + 5.0 * i, 400.0))
        covs.append(step.pos_cov)
    assert all(c is not None for c in covs)

    gap_covs = []
    for i in range(10, 20):
        step = tracker.update(i, None)
        gap_covs.append(step.pos_cov)
    # Uncertainty grows monotonically while predicting blind.
    assert gap_covs[-1][0] > gap_covs[0][0] > covs[-1][0]
    assert gap_covs[-1][1] > gap_covs[0][1] > covs[-1][1]

    resumed = tracker.update(20, (200.0, 400.0))
    assert resumed.pos_cov[0] < gap_covs[-1][0]


@pytest.mark.unit
def test_trackerstep_pos_cov_defaults_none():
    step = TrackerStep(frame=0, uv=None, p_flight=0.1,
                       is_outlier=False, is_gap_fill=True)
    assert step.pos_cov is None
