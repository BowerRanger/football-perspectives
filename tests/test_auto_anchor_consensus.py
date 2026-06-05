import numpy as np

from src.utils.auto_anchor import (
    compute_keyframes,
    robust_median_position,
    is_plausible_position,
)


def test_compute_keyframes_caps_count():
    got = compute_keyframes(total_frames=1000, keyframe_interval=10, max_keyframes=5)
    assert len(got) <= 6  # <= max_keyframes interior + the appended endpoint
    assert got[0] == 0
    assert got[-1] == 999  # last frame always covered


def test_compute_keyframes_uses_interval_for_short_shots():
    # interior samples at the interval, plus the last frame for tail coverage
    assert compute_keyframes(total_frames=100, keyframe_interval=30, max_keyframes=10) == [0, 30, 60, 90, 99]


def test_compute_keyframes_always_includes_last_frame():
    kf = compute_keyframes(total_frames=429, keyframe_interval=18, max_keyframes=24)
    assert kf[-1] == 428  # was 414 before the tail-coverage fix


def test_robust_median_drops_outliers():
    positions = [
        np.array([52.0, -30.0, 15.0]),
        np.array([52.5, -30.2, 15.1]),
        np.array([52.3, -29.8, 14.9]),
        np.array([200.0, 500.0, 90.0]),   # gross outlier
    ]
    med = robust_median_position(positions)
    assert med is not None and abs(med[0] - 52.3) < 1.0  # outlier excluded


def test_is_plausible_rejects_underground_camera():
    bounds = {"x": (-30, 135), "y": (-60, 130), "z": (3, 80)}
    assert is_plausible_position(np.array([52.5, -30.0, 15.0]), bounds)
    assert not is_plausible_position(np.array([52.5, -30.0, -5.0]), bounds)
    assert not is_plausible_position(np.array([300.0, -30.0, 15.0]), bounds)
