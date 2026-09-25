"""Tests for ball_cue_audio on synthetic signals: a known click train
(recovering onset times), a known latency offset (recovering it via
calibration), the disable gates (too few anchors, too-scattered matches,
retimed shots), and the 2026-09-25 tuning round's noise-suppressed onset
path (band-limiting + crowd-floor normalization + rise-time sharpness)."""

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
    suppressed_onsets,
)
from src.utils.ball_cue_config import CueCfg


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


def _tone_swell(sr: int, freq: float, total_ms: float, amp: float = 0.6) -> np.ndarray:
    """A smooth (Hann-windowed) up/down amplitude swell on a pure tone --
    a crowd-noise proxy: real in-band energy, continuous-derivative
    envelope, but climbing far too gradually over ``total_ms`` to be a
    percussive impact. (A *linear* ramp has a discontinuous-derivative
    kink at its corners that itself reads as a sharp transient to a
    flux-based detector -- the Hann window avoids that trap.)"""
    n = int(round(total_ms / 1000.0 * sr))
    t = np.arange(n) / sr
    env = np.hanning(n)
    return (amp * env * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_suppressed_onsets_rejects_slow_inband_swell_but_keeps_sharp_click():
    sr = 22050
    n = int(3.0 * sr)
    x = np.zeros(n, dtype=np.float32)
    swell = _tone_swell(sr, freq=3000.0, total_ms=400.0)  # in-band, slow rise
    start = int(1.0 * sr)
    x[start:start + len(swell)] += swell
    click_i = int(2.0 * sr)
    rng = np.random.default_rng(1)
    x[click_i:click_i + 8] += (0.9 * rng.standard_normal(8)).astype(np.float32)

    onsets = suppressed_onsets(x, sr, CueCfg())
    onset_times = [o.time_s for o in onsets]
    assert any(abs(t - 2.0) < 0.05 for t in onset_times), onset_times
    assert not any(0.8 < t < 1.6 for t in onset_times), onset_times


def test_suppressed_onsets_ignores_out_of_band_rumble():
    sr = 22050
    n = int(2.0 * sr)
    t = np.arange(n) / sr
    x = (0.4 * np.sin(2 * np.pi * 150.0 * t)).astype(np.float32)  # well below freq_lo_hz
    assert suppressed_onsets(x, sr, CueCfg()) == []


def test_suppressed_onsets_recovers_click_times_like_baseline():
    sr = 22050
    true_times = [0.5, 1.2, 2.0]
    x = _click_train(sr, true_times, int(3.0 * sr))
    onsets = suppressed_onsets(x, sr, CueCfg())
    onset_times = [o.time_s for o in onsets]
    for t in true_times:
        assert any(abs(t - ot) < 0.05 for ot in onset_times), (t, onset_times)


def test_suppressed_onsets_too_short_returns_empty():
    assert suppressed_onsets(np.zeros(10, dtype=np.float32), 22050, CueCfg()) == []
