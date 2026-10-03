"""Soundtrack for the vertical Shorts: a commentary-suppressed crowd bed built
from the real shot audio, a procedurally synthesised strike thump (no licensed
assets), and a goal roar swell cut from the REAL post-goal crowd roar in the
source -- all loudness-normalised.

``build_audio(src_video, src_windows, events, duration_s, cfg) -> Path`` is the
single entry point (output is a 48 kHz stereo wav; ``encode_aac`` makes the AAC
the compositor muxes).  Event times are on the OUTPUT timeline; the thump /
swell are rendered sample-exactly at ``t_out_s`` so they land within one video
frame by construction.

Commentary suppression (``suppress_commentary``): crowd noise is stationary and
broadband, speech is non-stationary with harmonic peaks.  Per STFT bin we
estimate a rolling low-percentile "crowd floor" over ``floor_window_s`` and
clamp the speech-band (200-3400 Hz) magnitude to ``clamp_ratio`` x floor, which
flattens the speech peaks to crowd level while the slowly-varying crowd (and a
slow goal roar build) passes through.  A mild speech-band shelf is applied on
top.
"""
from __future__ import annotations

import logging
import subprocess
import tempfile
import wave
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy import ndimage, signal

from src.utils.ball_cue_audio import decode_audio_mono

logger = logging.getLogger(__name__)

SR = 44100  # processing rate (shot audio is 44.1 kHz); output wav is resampled
OUT_SR = 48000

DEFAULTS: dict = {
    "speech_lo_hz": 200.0,
    "speech_hi_hz": 3400.0,
    "floor_window_s": 1.5,
    "floor_percentile": 25,
    "clamp_ratio": 1.2,
    "speech_shelf_db": -3.0,
    "strike_gain": 0.9,       # absolute peak of the synthesised thump (pre-loudnorm)
    "impact_gain": 0.7,
    "bed_rms": 0.06,          # target bed RMS before the events are laid on
    "roar_len_s": 2.5,
    "roar_gain": 2.0,         # roar relative to bed level at the impact
    "roar_attack_s": 0.15,
    "loudnorm_i": -14.0,
    "loudnorm_tp": -1.5,
    "loudnorm_lra": 9.0,
}


def _cfg(cfg: dict | None) -> dict:
    out = dict(DEFAULTS)
    out.update((cfg or {}).get("audio") or cfg or {})
    return out


# --- commentary suppression ---------------------------------------------

_NPERSEG = 2048
_HOP = 512


def _stft(x: np.ndarray, sr: int):
    return signal.stft(x, fs=sr, nperseg=_NPERSEG, noverlap=_NPERSEG - _HOP,
                       boundary="zeros", padded=True)


def _crowd_floor(mag: np.ndarray, sr: int, window_s: float, pct: int) -> np.ndarray:
    w = max(3, int(round(window_s * sr / _HOP)) | 1)
    # percentile over time per bin, on a coarsened frame grid for speed
    step = 8
    coarse = mag[:, ::step]
    f = ndimage.percentile_filter(coarse, pct, size=(1, max(3, w // step)),
                                  mode="nearest")
    xs = np.arange(mag.shape[1])
    xc = np.arange(coarse.shape[1]) * step
    out = np.empty_like(mag)
    for i in range(mag.shape[0]):
        out[i] = np.interp(xs, xc, f[i])
    return out


def suppress_commentary(x: np.ndarray, sr: int = SR, cfg: dict | None = None) -> np.ndarray:
    """Return ``x`` with speech-band peaks clamped to the rolling crowd floor."""
    c = _cfg(cfg)
    if x.size < _NPERSEG:
        return x.copy()
    freqs, _, Z = _stft(x.astype(np.float64), sr)
    mag = np.abs(Z)
    band = (freqs >= c["speech_lo_hz"]) & (freqs <= c["speech_hi_hz"])
    floor = _crowd_floor(mag[band], sr, c["floor_window_s"], c["floor_percentile"])
    gain = np.ones_like(mag)
    limit = c["clamp_ratio"] * floor
    bmag = mag[band]
    bgain = np.where(bmag > limit, limit / np.maximum(bmag, 1e-12), 1.0)
    gain[band, :] = bgain
    gain[band, :] *= 10 ** (c["speech_shelf_db"] / 20.0)
    _, y = signal.istft(Z * gain, fs=sr, nperseg=_NPERSEG,
                        noverlap=_NPERSEG - _HOP, boundary=True)
    y = y[: x.size]
    if y.size < x.size:
        y = np.pad(y, (0, x.size - y.size))
    return y.astype(np.float32)


def commentary_excess_db(x: np.ndarray, sr: int = SR, cfg: dict | None = None) -> float:
    """Energy (dB) of the speech-band content standing ABOVE the rolling
    crowd floor -- the harmonic/syllabic part that distinguishes commentary
    from stationary crowd.  Used to state the suppression margin."""
    c = _cfg(cfg)
    freqs, _, Z = _stft(x.astype(np.float64), sr)
    mag = np.abs(Z)
    band = (freqs >= c["speech_lo_hz"]) & (freqs <= c["speech_hi_hz"])
    floor = _crowd_floor(mag[band], sr, c["floor_window_s"], c["floor_percentile"])
    excess = np.maximum(mag[band] - 1.2 * floor, 0.0)
    return float(10 * np.log10(np.sum(excess ** 2) + 1e-12))


# --- synthesis ------------------------------------------------------------

def synth_strike_thump(sr: int = SR, gain: float = 0.9, seed: int = 7) -> np.ndarray:
    """Boot-on-ball thump: pitched body sweeping ~140->55 Hz with a fast
    decay, plus a short lowpassed noise click for the leather slap."""
    n = int(0.28 * sr)
    t = np.arange(n) / sr
    f = 55.0 + 85.0 * np.exp(-t / 0.035)
    phase = 2 * np.pi * np.cumsum(f) / sr
    body = np.sin(phase) * np.exp(-t / 0.07)
    rng = np.random.default_rng(seed)
    click = rng.standard_normal(n) * np.exp(-t / 0.006)
    sos = signal.butter(2, 2500.0, btype="low", fs=sr, output="sos")
    click = signal.sosfilt(sos, click)
    y = body + 0.45 * click
    y = y / (np.max(np.abs(y)) + 1e-9)
    y[: int(0.0015 * sr)] *= np.linspace(0, 1, int(0.0015 * sr))  # de-click
    return (y * gain).astype(np.float32)


def synth_net_hit(sr: int = SR, gain: float = 0.7, seed: int = 11) -> np.ndarray:
    """Ball into the net: soft broadband rustle + low thud."""
    n = int(0.45 * sr)
    t = np.arange(n) / sr
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(n)
    sos = signal.butter(2, [300.0, 4500.0], btype="band", fs=sr, output="sos")
    rustle = signal.sosfilt(sos, noise) * np.exp(-t / 0.11)
    thud = np.sin(2 * np.pi * 70.0 * t) * np.exp(-t / 0.05)
    y = 0.6 * rustle + 0.8 * thud
    y = y / (np.max(np.abs(y)) + 1e-9)
    y[: int(0.0015 * sr)] *= np.linspace(0, 1, int(0.0015 * sr))
    return (y * gain).astype(np.float32)


# --- bed construction -----------------------------------------------------

def _extend_loop(seg: np.ndarray, n: int, sr: int, xfade_s: float = 0.15) -> np.ndarray:
    """Stretch ``seg`` to ``n`` samples WITHOUT pitch change by looping it with
    equal-power crossfades (crowd is noise-like so loops are inaudible)."""
    if seg.size >= n:
        return seg[:n]
    if seg.size < 2:
        return np.zeros(n, dtype=np.float32)
    xf = min(int(xfade_s * sr), seg.size // 2)
    fade_in = np.sin(np.linspace(0, np.pi / 2, xf))
    fade_out = np.cos(np.linspace(0, np.pi / 2, xf))
    out = seg.astype(np.float64).copy()
    while out.size < n:
        nxt = seg.astype(np.float64)
        out[-xf:] = out[-xf:] * fade_out + nxt[:xf] * fade_in
        out = np.concatenate([out, nxt[xf:]])
    return out[:n].astype(np.float32)


def _assemble_bed(src: np.ndarray, windows: Sequence[Sequence[float]],
                  duration_s: float, sr: int) -> np.ndarray:
    n_total = int(round(duration_s * sr))
    pieces = []
    for w in windows:
        t0, t1 = float(w[0]), float(w[1])
        stretch = float(w[2]) if len(w) > 2 else 1.0
        seg = src[int(max(t0, 0) * sr): int(max(t1, 0) * sr)]
        pieces.append(_extend_loop(seg, int(round((t1 - t0) * stretch * sr)), sr)
                      if stretch > 1.0 else seg)
    bed = np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)
    if bed.size >= n_total:
        return bed[:n_total]
    return _extend_loop(bed, n_total, sr) if bed.size > 1 else np.zeros(n_total, np.float32)


def find_roar(crowd: np.ndarray, sr: int, length_s: float) -> tuple[int, int]:
    """Sample span of the loudest ``length_s`` window of the (commentary-
    suppressed) source crowd -- the post-goal roar."""
    n = int(length_s * sr)
    if crowd.size <= n:
        return 0, crowd.size
    cs = np.concatenate([[0.0], np.cumsum(crowd.astype(np.float64) ** 2)])
    e = (cs[n:] - cs[:-n]) / n
    start = int(np.argmax(e))
    return start, start + n


def _roar_swell(roar: np.ndarray, sr: int, attack_s: float) -> np.ndarray:
    n = roar.size
    env = np.ones(n)
    a = max(1, int(attack_s * sr))
    env[:a] = np.linspace(0, 1, a) ** 2
    d = n // 2
    env[n - d:] = np.cos(np.linspace(0, np.pi / 2, d)) ** 2
    return roar * env


def _mix_in(dst: np.ndarray, clip: np.ndarray, at: int) -> None:
    lo = max(at, 0)
    hi = min(at + clip.size, dst.size)
    if hi > lo:
        dst[lo:hi] += clip[lo - at: hi - at]


# --- entry point ----------------------------------------------------------

def build_audio(src_video: str | Path, src_windows: Sequence[Sequence[float]],
                events: Sequence[tuple[str, float]], duration_s: float,
                cfg: dict | None = None, out_path: str | Path | None = None) -> Path:
    """Build the Short's soundtrack wav (48 kHz stereo, loudnormed).

    ``src_windows``: source-video seconds ``(t0, t1[, stretch])`` concatenated
    in order to form the output timeline (``stretch`` > 1 for slow-mo
    segments: the crowd is looped, not pitch-shifted).  ``events``:
    ``(kind, t_out_s)`` with kind ``"strike"`` | ``"impact"`` (net hit + roar).
    """
    c = _cfg(cfg)
    if duration_s <= 0:
        raise ValueError("duration_s must be > 0")
    src = decode_audio_mono(src_video, sr=SR)
    if src.size == 0:
        raise ValueError(f"no audio decoded from {src_video}")
    crowd_src = suppress_commentary(src, SR, c)
    bed = _assemble_bed(crowd_src, src_windows, duration_s, SR)
    rms = float(np.sqrt(np.mean(bed.astype(np.float64) ** 2))) + 1e-9
    bed = bed * (c["bed_rms"] / rms)

    r0, r1 = find_roar(crowd_src, SR, c["roar_len_s"])
    roar = crowd_src[r0:r1] * (c["bed_rms"] / rms)
    swell = _roar_swell(roar, SR, c["roar_attack_s"]) * c["roar_gain"]
    mix = bed.astype(np.float32).copy()
    for kind, t in events:
        at = int(round(float(t) * SR))
        if kind == "strike":
            _mix_in(mix, synth_strike_thump(SR, c["strike_gain"]), at)
        elif kind == "impact":
            _mix_in(mix, synth_net_hit(SR, c["impact_gain"]), at)
            _mix_in(mix, swell, at)
        else:
            raise ValueError(f"unknown audio event kind {kind!r}")
    peak = float(np.max(np.abs(mix))) + 1e-9
    if peak > 0.98:
        mix = mix * (0.98 / peak)

    out_path = Path(out_path) if out_path else Path(
        tempfile.mkstemp(suffix=".wav", prefix="shorts_audio_")[1])
    _write_loudnormed(mix, out_path, c)
    return out_path


def _write_loudnormed(mix: np.ndarray, out_path: Path, c: dict) -> None:
    pcm = (np.clip(mix, -1, 1) * 32767).astype("<i2").tobytes()
    flt = (f"loudnorm=I={c['loudnorm_i']}:TP={c['loudnorm_tp']}:LRA={c['loudnorm_lra']},"
           "aformat=channel_layouts=stereo")
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "s16le", "-ar", str(SR), "-ac", "1",
           "-i", "-", "-af", flt, "-ar", str(OUT_SR), "-c:a", "pcm_s16le", str(out_path)]
    proc = subprocess.run(cmd, input=pcm, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg loudnorm failed: {proc.stderr[-500:].decode(errors='replace')}")


def encode_aac(wav: str | Path, dst: str | Path, bitrate: str = "192k") -> Path:
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(wav), "-c:a", "aac",
                    "-b:a", bitrate, str(dst)], check=True, capture_output=True)
    return Path(dst)


def read_wav_mono(path: str | Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as w:
        sr, ch = w.getframerate(), w.getnchannels()
        data = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768
    return data.reshape(-1, ch).mean(axis=1), sr
