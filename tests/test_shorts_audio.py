"""shorts_audio: commentary suppression margin, event timing, AAC, loudness."""
from __future__ import annotations

import subprocess
import wave

import numpy as np
import pytest
from scipy import signal

from src.utils import shorts_audio as sa

SR = sa.SR
FPS = 30.0
MIN_SUPPRESSION_DB = 10.0  # stated margin: commentary-band excess energy


def _crowd(n, rng, rms=0.05):
    x = rng.standard_normal(n)
    sos = signal.butter(2, [100, 6000], btype="band", fs=SR, output="sos")
    x = signal.sosfilt(sos, x)
    return x / np.sqrt(np.mean(x ** 2)) * rms


def _speech(n):
    t = np.arange(n) / SR
    f0 = 130 + 15 * np.sin(2 * np.pi * 0.7 * t)
    ph = 2 * np.pi * np.cumsum(f0) / SR
    x = sum(np.sin(k * ph) / (1 + 0.3 * k) for k in range(1, 24))
    am = np.clip(np.sin(2 * np.pi * 3.5 * t), 0, None) ** 0.7
    return x * am * 0.05


def _make_source(path, dur=8.0):
    rng = np.random.default_rng(1)
    n = int(dur * SR)
    crowd = _crowd(n, rng)
    t = np.arange(n) / SR
    # post-goal roar: 5.0-7.5 s, 4x louder crowd with slow rise
    env = np.ones(n)
    m = (t >= 5.0) & (t < 7.5)
    env[m] = 1 + 3 * np.sin(np.pi * (t[m] - 5.0) / 2.5)
    x = crowd * env + _speech(n)
    pcm = (np.clip(x, -1, 1) * 32767).astype("<i2").tobytes()
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
         f"color=c=green:s=64x64:r=30:d={dur}", "-f", "s16le", "-ar", str(SR),
         "-ac", "1", "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-shortest", str(path)],
        input=pcm, check=True, capture_output=True)
    return x


@pytest.fixture(scope="module")
def src_video(tmp_path_factory):
    p = tmp_path_factory.mktemp("aud") / "src.mp4"
    _make_source(p)
    return p


def test_commentary_band_reduced_by_margin():
    rng = np.random.default_rng(2)
    n = 6 * SR
    clean = _crowd(n, rng)
    mixed = (clean + _speech(n)).astype(np.float32)
    before = sa.commentary_excess_db(mixed)
    after = sa.commentary_excess_db(sa.suppress_commentary(mixed))
    assert before - after >= MIN_SUPPRESSION_DB, (before, after)


def test_suppression_keeps_crowd_level():
    rng = np.random.default_rng(3)
    clean = _crowd(6 * SR, rng).astype(np.float32)
    out = sa.suppress_commentary(clean)
    r_in = np.sqrt(np.mean(clean ** 2))
    r_out = np.sqrt(np.mean(out ** 2))
    assert 0.5 < r_out / r_in < 1.1


def test_strike_thump_is_synthesised_and_peaks_early():
    y = sa.synth_strike_thump(SR)
    assert y.size > 0.2 * SR and np.max(np.abs(y)) > 0.5
    assert np.argmax(np.abs(y)) < 0.02 * SR


def test_build_audio_events_land_within_one_frame(src_video, tmp_path):
    wav = sa.build_audio(
        src_video, [(0.0, 6.0)], [("strike", 2.0), ("impact", 4.0)], 6.0,
        {"roar_gain": 0.0, "bed_rms": 1e-4}, out_path=tmp_path / "o.wav")
    x, sr = sa.read_wav_mono(wav)
    assert sr == sa.OUT_SR
    assert abs(x.size / sr - 6.0) < 0.1
    # onset = first sample above 40% of the local peak in a +-250 ms window
    def onset(t):
        a, b = int((t - 0.25) * sr), int((t + 0.25) * sr)
        seg = np.abs(x[a:b])
        thr = 0.4 * seg.max()
        return (a + int(np.argmax(seg > thr))) / sr
    for t in (2.0, 4.0):
        assert abs(onset(t) - t) <= 1.0 / FPS, (t, onset(t))


def test_roar_swell_taken_from_real_source_and_louder_after_impact(src_video, tmp_path):
    base = sa.build_audio(src_video, [(0.0, 6.0)], [], 6.0, out_path=tmp_path / "b.wav")
    roar = sa.build_audio(src_video, [(0.0, 6.0)], [("impact", 2.0)], 6.0,
                          out_path=tmp_path / "r.wav")
    xb, sr = sa.read_wav_mono(base)
    xr, _ = sa.read_wav_mono(roar)
    seg = slice(int(2.5 * sr), int(4.0 * sr))
    pre = slice(int(0.5 * sr), int(1.5 * sr))
    rms = lambda v: float(np.sqrt(np.mean(v ** 2)))  # noqa: E731
    # loudnorm rescales the whole file, so compare the swell to the same
    # file's own pre-impact level
    assert rms(xr[seg]) / rms(xr[pre]) > 1.4 * rms(xb[seg]) / rms(xb[pre])
    # roar window located in the real source's 5.0-7.5 s
    crowd = sa.suppress_commentary(sa.decode_audio_mono(src_video, sr=SR), SR)
    r0, r1 = sa.find_roar(crowd, SR, 2.5)
    assert 4.0 * SR <= r0 <= 6.0 * SR


def test_slowmo_window_stretches_without_changing_duration_contract(src_video, tmp_path):
    wav = sa.build_audio(src_video, [(0.0, 2.0, 3.0)], [], 6.0, out_path=tmp_path / "s.wav")
    x, sr = sa.read_wav_mono(wav)
    assert abs(x.size / sr - 6.0) < 0.1 and np.max(np.abs(x)) > 0.05


def test_aac_nonsilent_and_loudness(src_video, tmp_path):
    wav = sa.build_audio(src_video, [(0.0, 6.0)], [("strike", 2.0), ("impact", 4.0)],
                         6.0, out_path=tmp_path / "o.wav")
    aac = sa.encode_aac(wav, tmp_path / "o.m4a")
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=codec_name",
         "-of", "csv=p=0", str(aac)], capture_output=True, text=True, check=True)
    assert probe.stdout.strip() == "aac"
    x = sa.decode_audio_mono(aac, sr=SR)
    assert np.sqrt(np.mean(x ** 2)) > 0.02
    ln = subprocess.run(
        ["ffmpeg", "-hide_banner", "-i", str(wav), "-af", "ebur128", "-f", "null", "-"],
        capture_output=True, text=True).stderr
    integrated = float(ln.rsplit("I:", 1)[1].split("LUFS")[0])
    assert -17.0 < integrated < -11.0


def test_unknown_event_kind_rejected(src_video, tmp_path):
    with pytest.raises(ValueError):
        sa.build_audio(src_video, [(0.0, 2.0)], [("whistle", 1.0)], 2.0,
                       out_path=tmp_path / "x.wav")


def test_bad_duration_rejected(src_video):
    with pytest.raises(ValueError):
        sa.build_audio(src_video, [(0.0, 2.0)], [], 0.0)
