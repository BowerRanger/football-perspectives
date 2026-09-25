"""Audio-transient event cue: decode a shot's audio track and pick out
percussive onsets (kicks, bounces, goal-frame/net impacts) as candidate
``CueEvidence``.

No ``librosa``/``soundfile`` dependency -- audio is decoded via a raw
``ffmpeg`` subprocess to float32 PCM, and onset detection is a plain
numpy spectral-flux function with an adaptive median+MAD threshold and
greedy peak picking.

Per-clip latency calibration: cross-correlate the onset candidates
against frames of *clearly visible* manual contact anchors (the caller
passes the frames), searching +-``search_window_ms``. If fewer than
``min_matches`` anchors match within the window, or the matched offsets'
spread is too wide to trust (indicating an unreliable calibration -- lots
of commentary/crowd noise swamping real contact transients), the cue
reports itself disabled with a reason rather than emitting mis-timed
events.

Two known failure modes the ORIGINAL (full-band) onset function did not
attempt to fix, only avoid silently mis-scoring:
  - Highlight clips carry commentary/crowd noise on top of pitch audio;
    the onset function had no voice/crowd suppression, so its raw
    candidate list was noisy.
  - Slow-mo replays are retimed to real time at extraction, so their
    audio track no longer corresponds to the footage at 1x -- callers
    must pass the shot's ``speed_factor`` (see
    ``ball_cue_common.probe_speed_factor``); anything != 1.0 disables
    the cue outright (still true).

2026-09-25 tuning round added ``suppressed_onsets`` (used by
``compute_audio_cues`` by default, via ``CueCfg.audio_band_limit_enabled``)
to attack the first failure mode directly instead of leaning entirely on
the calibration gate:
  - band-limits the flux computation to the impact-transient band
    (``freq_lo_hz``..``freq_hi_hz``, default 1.5-6kHz) -- crowd
    roar/commentary is predominantly lower-frequency energy;
  - normalizes flux by a rolling-median "crowd floor" of in-band energy
    (``floor_window_s``), so a clip with sustained crowd noise doesn't
    just raise the absolute threshold uniformly -- the floor tracks and
    cancels the slow-moving noise level, leaving the fast local
    transients comparable across a clip;
  - requires onset "sharpness": the flux must rise by at least
    ``min_rise_ratio`` of its own peak height over a short lookback
    window, rejecting slow crowd swells that still eventually cross the
    threshold but climb too gradually to be a percussive impact.
The original ``spectral_flux_onsets`` (no band-limit/floor/sharpness) is
kept and still tested standalone -- ``CueCfg.audio_band_limit_enabled``
can select it as the "baseline" path for ablation/tuning comparisons.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.utils.ball_cue_config import CueCfg, DEFAULT_CUE_CFG
from src.utils.ball_hybrid_types import CueEvidence

_SR_DEFAULT = 22050  # plenty of bandwidth for percussive onset content


def decode_audio_mono(video_path: str | Path, sr: int = _SR_DEFAULT) -> np.ndarray:
    """Decode ``video_path``'s audio track to mono float32 PCM in
    ``[-1, 1]`` at ``sr`` Hz via ffmpeg."""
    cmd = [
        "ffmpeg", "-v", "error", "-i", str(video_path),
        "-f", "f32le", "-ac", "1", "-ar", str(sr), "-",
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           check=True)
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()


@dataclass(frozen=True)
class OnsetCandidate:
    time_s: float
    strength: float  # normalized flux magnitude above threshold


def spectral_flux_onsets(
    samples: np.ndarray,
    sr: int,
    *,
    frame_size: int = 1024,
    hop_size: int = 256,
    min_separation_s: float = 0.04,
    k_mad: float = 3.0,
) -> list[OnsetCandidate]:
    """High-frequency-weighted spectral-flux onset function.

    Computes ``|STFT|`` per hop, takes the frame-to-frame positive
    magnitude increase (spectral flux) weighted toward high frequency
    bins (a plain high-frequency-content emphasis -- percussive
    transients like a foot/ball impact have more high-frequency energy
    than sustained tones or low-frequency crowd rumble), then picks local
    maxima above an adaptive ``median + k_mad * MAD`` threshold with a
    minimum separation so one transient doesn't emit multiple onsets.
    """
    n = len(samples)
    if n < frame_size * 2:
        return []
    window = np.hanning(frame_size)
    n_hops = 1 + (n - frame_size) // hop_size
    n_bins = frame_size // 2 + 1
    mags = np.empty((n_hops, n_bins), dtype=np.float64)
    for i in range(n_hops):
        start = i * hop_size
        seg = samples[start:start + frame_size].astype(np.float64) * window
        mags[i] = np.abs(np.fft.rfft(seg))
    freq_weight = np.linspace(0.2, 1.0, n_bins)
    diff = np.diff(mags, axis=0, prepend=mags[:1])
    flux = np.sum(np.clip(diff, 0.0, None) * freq_weight, axis=1)
    flux[0] = 0.0
    med = float(np.median(flux))
    mad = float(np.median(np.abs(flux - med))) + 1e-9
    thresh = med + k_mad * mad
    min_sep_hops = max(1, int(round(min_separation_s * sr / hop_size)))

    candidates: list[OnsetCandidate] = []
    i = 1
    n_flux = len(flux)
    while i < n_flux - 1:
        if flux[i] > thresh and flux[i] >= flux[i - 1] and flux[i] >= flux[i + 1]:
            t = i * hop_size / sr
            strength = float((flux[i] - thresh) / thresh)
            candidates.append(OnsetCandidate(time_s=t, strength=strength))
            i += min_sep_hops
        else:
            i += 1
    return candidates


def _band_flux_and_energy(
    samples: np.ndarray, sr: int, *,
    freq_lo_hz: float, freq_hi_hz: float, frame_size: int, hop_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """``(flux, band_energy)`` restricted to the ``[freq_lo_hz, freq_hi_hz]``
    STFT bins. ``flux`` is the frame-to-frame positive magnitude increase
    (unweighted -- the band restriction itself is the frequency
    weighting); ``band_energy`` is each hop's raw in-band magnitude sum,
    used as the crowd-floor input."""
    n = len(samples)
    if n < frame_size * 2:
        return np.array([]), np.array([])
    window = np.hanning(frame_size)
    n_hops = 1 + (n - frame_size) // hop_size
    freqs = np.fft.rfftfreq(frame_size, d=1.0 / sr)
    band_mask = (freqs >= freq_lo_hz) & (freqs <= freq_hi_hz)
    if not band_mask.any():
        return np.array([]), np.array([])
    mags = np.empty((n_hops, int(band_mask.sum())), dtype=np.float64)
    for i in range(n_hops):
        start = i * hop_size
        seg = samples[start:start + frame_size].astype(np.float64) * window
        mags[i] = np.abs(np.fft.rfft(seg))[band_mask]
    band_energy = mags.sum(axis=1)
    diff = np.diff(mags, axis=0, prepend=mags[:1])
    flux = np.sum(np.clip(diff, 0.0, None), axis=1)
    flux[0] = 0.0
    return flux, band_energy


def _rolling_median(x: np.ndarray, window: int) -> np.ndarray:
    """Centered rolling median (odd or even ``window``, clamped at the
    array edges). O(n * window) -- fine for the few-thousand-hop arrays
    a 5-20s clip produces."""
    n = len(x)
    half = max(1, window // 2)
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        out[i] = float(np.median(x[lo:hi]))
    return out


def suppressed_onsets(
    samples: np.ndarray, sr: int, cfg: CueCfg = DEFAULT_CUE_CFG,
    *, hop_size: int = 256, frame_size: int = 1024,
) -> list[OnsetCandidate]:
    """Band-limited, crowd-floor-normalized, sharpness-gated onset
    function (see the module docstring). Falls back to an empty list
    when the clip is too short for even one floor-window's worth of
    hops."""
    flux, band_energy = _band_flux_and_energy(
        samples, sr, freq_lo_hz=cfg.audio_freq_lo_hz, freq_hi_hz=cfg.audio_freq_hi_hz,
        frame_size=frame_size, hop_size=hop_size)
    floor_window_hops = max(3, int(round(cfg.audio_floor_window_s * sr / hop_size)))
    if len(flux) < floor_window_hops * 2:
        return []
    floor = _rolling_median(band_energy, floor_window_hops)
    norm_flux = flux / (floor + 1e-6)
    med = float(np.median(norm_flux))
    mad = float(np.median(np.abs(norm_flux - med))) + 1e-9
    thresh = med + cfg.audio_k_mad * mad
    min_sep_hops = max(1, int(round(0.04 * sr / hop_size)))
    rise_hops = max(1, int(round(0.02 * sr / hop_size)))  # ~20ms lookback

    candidates: list[OnsetCandidate] = []
    i = rise_hops + 1
    n = len(norm_flux)
    while i < n - 1:
        is_peak = (norm_flux[i] > thresh and norm_flux[i] >= norm_flux[i - 1]
                   and norm_flux[i] >= norm_flux[i + 1])
        if is_peak:
            rise = norm_flux[i] - norm_flux[i - rise_hops]
            peak_height = norm_flux[i] - med
            sharp_enough = (peak_height > 0
                             and rise / peak_height >= cfg.audio_min_rise_ratio)
            if sharp_enough:
                t = i * hop_size / sr
                strength = float((norm_flux[i] - thresh) / thresh)
                candidates.append(OnsetCandidate(time_s=t, strength=strength))
                i += min_sep_hops
                continue
        i += 1
    return candidates


@dataclass(frozen=True)
class AudioCalibration:
    enabled: bool
    reason: str | None
    latency_ms: float | None
    n_matched: int
    match_std_ms: float | None


def calibrate_latency(
    onsets: list[OnsetCandidate],
    contact_frames: list[int],
    fps: float,
    *,
    search_window_ms: float = 150.0,
    min_matches: int = 4,
    max_std_ms: float = 60.0,
) -> AudioCalibration:
    """Median(onset_time - anchor_time) over anchors matched within
    ``+-search_window_ms`` is the per-clip audio latency (positive =
    audio lags the video). Disables (with a reason) when there aren't
    enough calibration anchors, too few onsets land in-window, or the
    matched offsets are too spread out to trust."""
    if len(contact_frames) < min_matches:
        return AudioCalibration(
            False,
            f"< {min_matches} calibration anchors (have {len(contact_frames)})",
            None, 0, None)
    window_s = search_window_ms / 1000.0
    onset_times = sorted(o.time_s for o in onsets)
    deltas_ms: list[float] = []
    for f in contact_frames:
        anchor_t = f / fps
        best_d = window_s
        best = None
        for t in onset_times:
            d = t - anchor_t
            if abs(d) <= best_d:
                best_d, best = abs(d), d
        if best is not None:
            deltas_ms.append(best * 1000.0)
    if len(deltas_ms) < min_matches:
        return AudioCalibration(
            False,
            f"< {min_matches} onsets matched within +-{search_window_ms:.0f}ms "
            f"of a calibration anchor (matched {len(deltas_ms)})",
            None, len(deltas_ms), None)
    latency = float(np.median(deltas_ms))
    std = float(np.std(deltas_ms))
    if std > max_std_ms:
        return AudioCalibration(
            False,
            f"matched-offset spread {std:.1f}ms > {max_std_ms:.0f}ms "
            "(low calibration confidence -- likely commentary/crowd noise "
            "swamping real contact transients)",
            latency, len(deltas_ms), std)
    return AudioCalibration(True, None, latency, len(deltas_ms), std)


def compute_audio_cues(
    video_path: str | Path,
    fps: float,
    contact_frames: list[int],
    *,
    cfg: CueCfg = DEFAULT_CUE_CFG,
    sr: int = _SR_DEFAULT,
    speed_factor: float = 1.0,
) -> tuple[AudioCalibration, list[CueEvidence]]:
    """Full audio-cue pipeline for one shot: decode, detect onsets,
    calibrate latency against ``contact_frames``, and (only if
    calibration succeeds) emit latency-corrected ``CueEvidence``.

    ``speed_factor`` (see ``ball_cue_common.probe_speed_factor``) != 1.0
    disables the cue immediately without decoding -- a retimed slow-mo
    shot's audio does not correspond to its (retimed) video frames.
    ``cfg.audio_band_limit_enabled`` (default ``True``) selects
    ``suppressed_onsets`` (band-limited + crowd-floor + sharpness gated)
    over the original full-band ``spectral_flux_onsets``.
    """
    if abs(speed_factor - 1.0) > 1e-6:
        return AudioCalibration(
            False,
            f"retimed shot (speed_factor={speed_factor}); its audio track "
            "is not real-time and cannot be latency-calibrated",
            None, 0, None), []
    samples = decode_audio_mono(video_path, sr=sr)
    if cfg.audio_band_limit_enabled:
        onsets = suppressed_onsets(samples, sr, cfg)
    else:
        onsets = spectral_flux_onsets(samples, sr)
    calib = calibrate_latency(
        onsets, contact_frames, fps,
        search_window_ms=cfg.audio_search_window_ms, min_matches=cfg.audio_min_matches,
        max_std_ms=cfg.audio_max_std_ms)
    if not calib.enabled:
        return calib, []
    events: list[CueEvidence] = []
    for o in onsets:
        corrected_t = o.time_s - (calib.latency_ms or 0.0) / 1000.0
        frame = int(round(corrected_t * fps))
        conf = float(np.clip(0.3 + 0.5 * min(1.0, o.strength / 3.0), 0.0, 0.95))
        events.append(CueEvidence(
            frame=frame, kind="contact", cue="audio_onset", conf=conf,
            xyz=None, uv=None))
    return calib, events


__all__ = [
    "decode_audio_mono", "spectral_flux_onsets", "suppressed_onsets",
    "OnsetCandidate", "calibrate_latency", "AudioCalibration",
    "compute_audio_cues",
]
