"""Tests for ball_cue_audio on synthetic signals: a known click train
(recovering onset times), a known latency offset (recovering it via
calibration), and the disable gates (too few anchors, too-scattered
matches, retimed shots)."""

from __future__ import annotations

import subprocess

import numpy as np
import pytest

from src.utils.ball_cue_audio import (
    AudioCalibration,
    OnsetCandidate,
    calibrate_latency,
    compute_audio_cues,
    decode_audio_mono,
    spectral_flux_onsets,
)


def _click_train(sr: int, times_s: list[float], n_samples: int, amp: float = 0.9) -> np.ndarray:
    x = np.zeros(n_samples, dtype=np.float32)
    for t in times_s:
        i = int(round(t * sr))
        if 0 <= i < n_samples - 8:
            rng = np.random.default_rng(i)  # broadband burst, not pure DC
            x[i:i + 8] += (amp * rng.standard_normal(8)).astype(np.float32)
    return x


def test_spectral_flux_onsets_recovers_click_times():
    sr = 22050
    true_times = [0.5, 1.2, 2.0, 3.3]
    x = _click_train(sr, true_times, int(4.0 * sr))
    onsets = spectral_flux_onsets(x, sr)
    onset_times = [o.time_s for o in onsets]
    for t in true_times:
        assert any(abs(t - ot) < 0.03 for ot in onset_times), (t, onset_times)


def test_spectral_flux_onsets_silence_has_no_onsets():
    sr = 22050
    x = np.zeros(int(2.0 * sr), dtype=np.float32)
    assert spectral_flux_onsets(x, sr) == []


def test_spectral_flux_onsets_too_short_returns_empty():
    assert spectral_flux_onsets(np.zeros(10, dtype=np.float32), 22050) == []


def test_calibrate_latency_recovers_known_shift():
    fps = 30.0
    contact_frames = [10, 40, 70, 100, 130]
    true_latency_s = 0.035  # audio lags video by 35ms
    onsets = [OnsetCandidate(time_s=f / fps + true_latency_s, strength=5.0)
              for f in contact_frames]
    calib = calibrate_latency(onsets, contact_frames, fps)
    assert calib.enabled
    assert calib.latency_ms == pytest.approx(35.0, abs=5.0)
    assert calib.n_matched == 5


def test_calibrate_latency_disabled_with_too_few_anchors():
    calib = calibrate_latency([], [10, 20], 30.0)
    assert not calib.enabled
    assert "calibration anchors" in calib.reason
    assert calib.latency_ms is None


def test_calibrate_latency_disabled_with_high_spread():
    fps = 30.0
    contact_frames = [10, 40, 70, 100]
    onsets = [
        OnsetCandidate(time_s=10 / fps + 0.02, strength=3),
        OnsetCandidate(time_s=40 / fps - 0.12, strength=3),
        OnsetCandidate(time_s=70 / fps + 0.13, strength=3),
        OnsetCandidate(time_s=100 / fps - 0.01, strength=3),
    ]
    calib = calibrate_latency(onsets, contact_frames, fps)
    assert not calib.enabled
    assert "spread" in calib.reason


def test_calibrate_latency_disabled_when_onsets_miss_window():
    fps = 30.0
    contact_frames = [10, 40, 70, 100]
    onsets = [OnsetCandidate(time_s=5.0 + i, strength=3) for i in range(4)]
    calib = calibrate_latency(onsets, contact_frames, fps, search_window_ms=150.0)
    assert not calib.enabled
    assert "matched" in calib.reason


def test_decode_audio_mono_matches_ffmpeg_tone(tmp_path):
    video = tmp_path / "tone.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-v", "error",
        "-f", "lavfi", "-i", "color=c=black:s=64x64:d=1",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
        "-c:v", "libx264", "-c:a", "aac", str(video),
    ], check=True)
    samples = decode_audio_mono(video, sr=22050)
    assert samples.dtype == np.float32
    assert len(samples) > 20000
    assert np.abs(samples).max() > 0.01


def test_compute_audio_cues_disabled_for_retimed_shot(tmp_path):
    calib, events = compute_audio_cues(
        tmp_path / "does_not_need_to_exist.mp4", fps=30.0,
        contact_frames=[1, 2, 3, 4, 5], speed_factor=4.0)
    assert isinstance(calib, AudioCalibration)
    assert not calib.enabled
    assert "retimed" in calib.reason
    assert events == []
