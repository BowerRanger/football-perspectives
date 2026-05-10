"""IMM Kalman ball-tracker over 2D pixel observations.

Pins three primary behaviours BallStage relies on:
- mode posterior crosses 0.5 within a few frames of a sharp velocity
  change (the "kick" that triggers a flight segment);
- pixel position is gap-filled within ~1 px during a short detection
  gap on smooth rolling motion;
- a wildly out-of-range measurement does not catastrophically pull the
  blended state estimate.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.utils.ball_tracker import BallTracker


def _rolling_path(n: int, u0: float, v0: float, vel: tuple[float, float]) -> np.ndarray:
    return np.array([[u0 + vel[0] * i, v0 + vel[1] * i] for i in range(n)])


def _parabolic_path(n: int, u0: float, v0: float, vu: float, vv: float, av: float) -> np.ndarray:
    return np.array(
        [[u0 + vu * i, v0 + vv * i + 0.5 * av * i * i] for i in range(n)]
    )


@pytest.mark.unit
def test_mode_posterior_crosses_on_kick():
    """A sharp acceleration at frame 20 should drive p_flight ≥ 0.5 within ±3 frames."""
    pre = _rolling_path(20, u0=100.0, v0=400.0, vel=(5.0, 0.0))
    # At frame 20 the ball is kicked upward — pixel-v changes sign on v-axis
    # and the trajectory becomes parabolic.
    flight = _parabolic_path(20, u0=pre[-1, 0], v0=pre[-1, 1], vu=6.0, vv=-25.0, av=2.0)
    truth = np.vstack([pre, flight])

    tracker = BallTracker(
        process_noise_grounded_px=2.0,
        process_noise_flight_px=20.0,
        measurement_noise_px=1.0,
        gating_sigma=4.0,
        max_gap_frames=4,
    )
    crossings = []
    for i, p in enumerate(truth):
        step = tracker.update(i, (float(p[0]), float(p[1])))
        if step.p_flight >= 0.5:
            crossings.append(i)
    first_crossing = crossings[0] if crossings else None
    assert first_crossing is not None, "p_flight never crossed 0.5"
    # Allow up to 3 frames of lag — IMM smooths transitions.
    assert 19 <= first_crossing <= 24, f"crossing at frame {first_crossing}"


@pytest.mark.unit
def test_gap_fill_recovers_within_one_pixel_on_smooth_motion():
    """Drop the detector for 3 frames mid-roll; prediction should stay close."""
    truth = _rolling_path(30, u0=100.0, v0=400.0, vel=(4.0, 0.0))
    tracker = BallTracker(
        process_noise_grounded_px=1.0,
        process_noise_flight_px=10.0,
        measurement_noise_px=0.5,
        max_gap_frames=8,
    )
    outputs = []
    for i, p in enumerate(truth):
        # Frames 12, 13, 14 — no detection.
        if 12 <= i <= 14:
            step = tracker.update(i, None)
        else:
            step = tracker.update(i, (float(p[0]), float(p[1])))
        outputs.append(step)

    for i in (12, 13, 14):
        assert outputs[i].uv is not None, f"frame {i} returned None despite gap < max_gap"
        assert outputs[i].is_gap_fill is True
        err = np.linalg.norm(np.array(outputs[i].uv) - truth[i])
        assert err < 1.0, f"frame {i}: gap-fill error {err:.2f} px"


@pytest.mark.unit
def test_max_gap_exceeds_emits_none():
    truth = _rolling_path(30, u0=100.0, v0=400.0, vel=(4.0, 0.0))
    tracker = BallTracker(max_gap_frames=3)
    outputs = []
    for i, p in enumerate(truth):
        if 10 <= i <= 20:
            step = tracker.update(i, None)
        else:
            step = tracker.update(i, (float(p[0]), float(p[1])))
        outputs.append(step)

    # First 3 missed frames: gap-fill emitted.
    for i in (10, 11, 12):
        assert outputs[i].uv is not None
    # After max_gap, output should drop to None.
    for i in (15, 18, 20):
        assert outputs[i].uv is None


@pytest.mark.unit
def test_wild_outlier_does_not_break_state():
    """A single (5000, 5000) observation should not derail subsequent estimates."""
    truth = _rolling_path(30, u0=100.0, v0=400.0, vel=(4.0, 0.0))
    tracker = BallTracker(
        process_noise_grounded_px=1.0,
        process_noise_flight_px=10.0,
        measurement_noise_px=0.5,
        gating_sigma=3.0,
    )
    outputs = []
    for i, p in enumerate(truth):
        if i == 15:
            step = tracker.update(i, (5000.0, 5000.0))
        else:
            step = tracker.update(i, (float(p[0]), float(p[1])))
        outputs.append(step)

    # After the outlier and one normal observation, the tracker should
    # be back near the true trajectory (within 5 px).
    err = np.linalg.norm(np.array(outputs[17].uv) - truth[17])
    assert err < 5.0, f"state pulled too far by outlier: {err:.1f} px"
