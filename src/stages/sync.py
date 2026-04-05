from pathlib import Path
from dataclasses import dataclass
import json
import logging
import subprocess

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.signal import correlate, fftconvolve
from scipy.stats import pearsonr

from src.pipeline.base import BaseStage
from src.schemas.calibration import CameraFrame, CalibrationResult
from src.schemas.shots import ShotsManifest
from src.schemas.sync_map import Alignment, SyncMap
from src.schemas.tracks import TracksResult
from src.utils.ball_detector import BallDetector
from src.utils.camera import project_to_pitch
from src.stages.matching import hungarian_match_players


def _fill_nans(sig: np.ndarray) -> np.ndarray:
    """Replace NaN values via linear interpolation; returns zeros when no valid data."""
    valid = ~np.isnan(sig)
    if not np.any(valid):
        return np.zeros_like(sig)
    indices = np.arange(len(sig))
    return np.interp(indices, indices[valid], sig[valid])


@dataclass(frozen=True)
class AlignmentEstimate:
    offset: int
    confidence: float
    method: str
    valid: bool


def project_ball_to_pitch(
    pixel: np.ndarray, cam_frame: CameraFrame
) -> np.ndarray | None:
    """Project a ball pixel position onto the pitch ground plane using calibration."""
    K = np.array(cam_frame.intrinsic_matrix, dtype=np.float32)
    rvec = np.array(cam_frame.rotation_vector, dtype=np.float32)
    tvec = np.array(cam_frame.translation_vector, dtype=np.float32)
    return project_to_pitch(pixel, K, rvec, tvec)


def cross_correlate_trajectories(
    traj_a: np.ndarray, traj_b: np.ndarray
) -> tuple[int, float]:
    """
    Find the integer frame offset of traj_b relative to traj_a via cross-correlation.

    Returns (offset, confidence) where:
      offset > 0  -> traj_b lags traj_a (traj_b event occurs later)
      offset < 0  -> traj_b leads traj_a (traj_b event occurs earlier)
    Convention: traj_b frame + offset = corresponding traj_a frame.

    Confidence is the Pearson r of the two trajectories at the best lag,
    clamped to [0, 1].  Values near 1.0 indicate a strong, clean match;
    values below ~0.4 suggest the signals are largely uncorrelated.
    """
    return _cross_correlate_signals(
        signal_a=traj_a,
        signal_b=traj_b,
        max_lag=None,
        min_overlap_frames=2,
    )


def _compute_overlap_frames(
    reference_len: int,
    shot_len: int,
    offset: int,
) -> tuple[int, int]:
    start = max(0, offset)
    end = min(reference_len, offset + shot_len)
    return start, end


def _cross_correlate_signals(
    signal_a: np.ndarray,
    signal_b: np.ndarray,
    max_lag: int | None,
    min_overlap_frames: int,
) -> tuple[int, float]:
    if len(signal_a) == 0 or len(signal_b) == 0:
        return 0, 0.0

    norm_a = np.linalg.norm(signal_a)
    norm_b = np.linalg.norm(signal_b)
    if norm_a < 1e-8 or norm_b < 1e-8:
        return 0, 0.0

    a = signal_a / norm_a
    b = signal_b / norm_b
    corr = correlate(b, a, mode="full")
    lags = np.arange(-(len(a) - 1), len(b))

    if max_lag is not None:
        lag_mask = np.abs(lags) <= max_lag
        if not np.any(lag_mask):
            return 0, 0.0
        masked_corr = corr[lag_mask]
        masked_lags = lags[lag_mask]
        peak_idx = int(np.argmax(masked_corr))
        offset = int(masked_lags[peak_idx])
    else:
        peak_idx = int(np.argmax(corr))
        offset = int(lags[peak_idx])

    overlap_start, overlap_end = _compute_overlap_frames(len(a), len(b), offset)
    overlap = max(0, overlap_end - overlap_start)
    if overlap < min_overlap_frames:
        return offset, 0.0

    if offset >= 0:
        aligned_a = a[: len(a) - offset] if offset < len(a) else a[:0]
        aligned_b = b[offset: offset + len(aligned_a)]
    else:
        aligned_b = b[: len(b) + offset] if -offset < len(b) else b[:0]
        aligned_a = a[-offset: -offset + len(aligned_b)]

    aligned_overlap = min(len(aligned_a), len(aligned_b), overlap)
    aligned_a = aligned_a[:aligned_overlap]
    aligned_b = aligned_b[:aligned_overlap]

    if (
        aligned_overlap < min_overlap_frames
        or np.std(aligned_a) < 1e-8
        or np.std(aligned_b) < 1e-8
    ):
        return offset, 0.0

    r, _ = pearsonr(aligned_a, aligned_b)
    confidence = float(max(0.0, r))
    return offset, min(1.0, confidence)


def _extract_player_motion_signal(
    tracks: TracksResult,
    total_frames: int,
) -> np.ndarray:
    speeds_by_frame: dict[int, list[float]] = {}

    for track in tracks.tracks:
        if track.class_name == "ball":
            continue

        frame_positions = sorted(
            [
                (tf.frame, np.array(tf.pitch_position, dtype=float))
                for tf in track.frames
                if tf.pitch_position is not None
            ],
            key=lambda item: item[0],
        )

        for (prev_frame, prev_pos), (curr_frame, curr_pos) in zip(
            frame_positions,
            frame_positions[1:],
        ):
            delta = curr_frame - prev_frame
            if delta <= 0:
                continue
            speed = float(np.linalg.norm(curr_pos - prev_pos) / delta)
            frame_speeds = speeds_by_frame.get(curr_frame, [])
            speeds_by_frame[curr_frame] = [*frame_speeds, speed]

    signal = np.full(total_frames, np.nan, dtype=float)
    for frame_idx, speeds in speeds_by_frame.items():
        if 0 <= frame_idx < total_frames and speeds:
            signal[frame_idx] = float(np.median(np.array(speeds, dtype=float)))
    return signal


def _extract_pitch_axis_trajectory(
    tracks_result: TracksResult,
    track_id: str,
    total_frames: int,
    axis: int,
) -> np.ndarray:
    """
    Build a dense 1D time series of one pitch-coordinate axis for a single track.

    ``axis=0`` extracts the x coordinate; ``axis=1`` extracts y.
    Frames where the track has no detection carry ``np.nan``.

    Returns
    -------
    ndarray of shape ``(total_frames,)``
    """
    signal = np.full(total_frames, np.nan, dtype=np.float64)
    for track in tracks_result.tracks:
        if track.track_id != track_id:
            continue
        for tf in track.frames:
            if tf.pitch_position is not None and 0 <= tf.frame < total_frames:
                signal[tf.frame] = float(tf.pitch_position[axis])
    return signal


def _fuse_alignment_estimates(
    ball_estimate: AlignmentEstimate,
    player_estimate: AlignmentEstimate,
    agreement_tolerance_frames: int,
) -> AlignmentEstimate:
    if ball_estimate.valid and player_estimate.valid:
        if abs(ball_estimate.offset - player_estimate.offset) <= agreement_tolerance_frames:
            total_weight = ball_estimate.confidence + player_estimate.confidence
            if total_weight <= 0:
                return AlignmentEstimate(offset=0, confidence=0.0, method="low_confidence", valid=False)
            fused_offset = int(round(
                (
                    ball_estimate.offset * ball_estimate.confidence
                    + player_estimate.offset * player_estimate.confidence
                ) / total_weight
            ))
            fused_confidence = min(1.0, total_weight / 2.0)
            return AlignmentEstimate(
                offset=fused_offset,
                confidence=fused_confidence,
                method="hybrid",
                valid=fused_confidence > 0.0,
            )

        if ball_estimate.confidence >= player_estimate.confidence:
            return ball_estimate
        return player_estimate

    if ball_estimate.valid:
        return ball_estimate
    if player_estimate.valid:
        return player_estimate
    return AlignmentEstimate(offset=0, confidence=0.0, method="low_confidence", valid=False)


# ---------------------------------------------------------------------------
# Visual frame similarity — new primary sync signal
# ---------------------------------------------------------------------------

def _extract_frame_descriptors(
    clip_path: Path,
    fps: float,
    sample_fps: float = 5.0,
    thumb_size: tuple[int, int] = (64, 36),
) -> tuple[np.ndarray, int]:
    """
    Extract L2-normalised grayscale frame descriptors from a clip.

    Frames are sampled at ``sample_fps`` Hz (relative to the clip's native
    ``fps``).  Each descriptor is a 64×36 = 2304-dim float32 vector.
    Zero-luminance frames are represented as zero vectors (black frames
    produce zero similarity rather than undefined similarity).

    Returns
    -------
    descriptors : ndarray, shape (N, D)
    frame_step  : int  — source frames between consecutive samples
    """
    frame_step = max(1, round(fps / sample_fps))
    w, h = thumb_size
    dim = w * h

    cap = cv2.VideoCapture(str(clip_path))
    descriptors: list[np.ndarray] = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_step == 0:
            small = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            vec = gray.flatten().astype(np.float32)
            norm = float(np.linalg.norm(vec))
            if norm > 1e-6:
                vec /= norm
            else:
                vec = np.zeros(dim, dtype=np.float32)
            descriptors.append(vec)
        frame_idx += 1

    cap.release()

    if descriptors:
        return np.array(descriptors, dtype=np.float32), frame_step
    return np.zeros((0, dim), dtype=np.float32), frame_step


def _visual_similarity_profile(
    ref_desc: np.ndarray,
    shot_desc: np.ndarray,
    min_overlap: int,
    max_lag: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Mean cosine similarity for every temporal offset between two clips.

    Offset convention matches the rest of the pipeline:
    ``ref_desc[j + d]`` aligns with ``shot_desc[j]`` at offset ``d``.
    (Equivalently: frame_in_reference = frame_in_shot + d * frame_step.)

    Parameters
    ----------
    ref_desc : np.ndarray
        Reference descriptors, shape (N_ref, D)
    shot_desc : np.ndarray
        Shot descriptors, shape (N_shot, D)
    min_overlap : int
        Minimum overlap in descriptor samples
    max_lag : int | None
        Maximum offset to search (descriptor samples). If None, search full range.

    Returns
    -------
    offsets : int ndarray, shape (M,)   — descriptor-space offsets
    scores  : float ndarray, shape (M,) — mean cosine similarity per offset
    """
    N_ref, N_shot = len(ref_desc), len(shot_desc)
    if N_ref == 0 or N_shot == 0:
        return np.array([0], dtype=np.int64), np.array([0.0])

    # M[i, j] = dot(ref[i], shot[j]) = cosine similarity (L2-normalised)
    M = ref_desc @ shot_desc.T  # shape (N_ref, N_shot)

    offsets = np.arange(-(N_shot - 1), N_ref, dtype=np.int64)
    if max_lag is not None:
        offsets = offsets[(offsets >= -max_lag) & (offsets <= max_lag)]
    scores = np.zeros(len(offsets), dtype=np.float64)

    for k, d in enumerate(offsets):
        # np.diagonal(M, offset=-d) gives M[j+d, j] for all valid j
        diag = np.diagonal(M, offset=-int(d))
        if len(diag) >= min_overlap:
            scores[k] = float(np.mean(diag))

    return offsets, scores


def _align_visual(
    ref_clip: Path,
    shot_clip: Path,
    fps: float,
    min_overlap_frames: int,
    sample_fps: float,
    max_lag_frames: int | None = None,
) -> AlignmentEstimate:
    """
    Align two clips by maximising visual frame similarity over all offsets.

    Confidence is the z-score of the peak similarity relative to the
    median background similarity, normalised to [0, 1].

    Parameters
    ----------
    max_lag_frames : int | None
        Maximum lag in frames. If provided, limits the search window.
    """
    ref_desc, frame_step = _extract_frame_descriptors(ref_clip, fps, sample_fps)
    shot_desc, _ = _extract_frame_descriptors(shot_clip, fps, sample_fps)

    min_overlap_samples = max(1, int(np.ceil(min_overlap_frames / frame_step)))
    max_lag_samples = None if max_lag_frames is None else max(1, max_lag_frames // frame_step)
    offsets, scores = _visual_similarity_profile(ref_desc, shot_desc, min_overlap_samples, max_lag=max_lag_samples)

    if len(scores) == 0 or scores.max() <= 0.0:
        return AlignmentEstimate(offset=0, confidence=0.0, method="visual", valid=False)

    best_idx = int(np.argmax(scores))
    best_d = int(offsets[best_idx])
    best_score = float(scores[best_idx])

    # z-score confidence: how much better is the peak than the median?
    median_score = float(np.median(scores))
    std_score = float(np.std(scores)) + 1e-8
    z = (best_score - median_score) / std_score
    # Scale so z=3 → 0.5, z=6 → 1.0; anything below 1 is near-zero.
    confidence = float(min(1.0, max(0.0, z / 6.0)))

    offset_frames = int(round(best_d * frame_step))

    return AlignmentEstimate(
        offset=offset_frames,
        confidence=confidence,
        method="visual",
        valid=confidence > 0.0,
    )


def _align_formation(
    ref_signal: np.ndarray,
    shot_signal: np.ndarray,
    min_overlap_frames: int,
) -> AlignmentEstimate:
    """
    Align using the per-frame player motion signal via cross-correlation.

    The search is unconstrained — all offsets up to full clip length are
    considered.  This catches replays that are temporally far from the
    reference segment.
    """
    offset, confidence = _cross_correlate_signals(
        signal_a=ref_signal,
        signal_b=shot_signal,
        max_lag=None,
        min_overlap_frames=min_overlap_frames,
    )
    return AlignmentEstimate(
        offset=offset,
        confidence=confidence,
        method="player_formation",
        valid=confidence > 0.0,
    )


# ---------------------------------------------------------------------------
# Spatial formation histograms — replaces motion-speed as the formation signal
# ---------------------------------------------------------------------------

def _compute_pitch_bounds(
    tracks_by_shot: dict[str, TracksResult],
) -> tuple[tuple[float, float], tuple[float, float]]:
    """
    Derive (x_min, x_max), (y_min, y_max) from all valid pitch positions across
    every shot.  Uses 5th–95th percentile bounds padded by 10 % on each side
    to be robust against stray projections far off the pitch.

    Falls back to ``((0, 12000), (0, 8000))`` when no position data is available.
    """
    xs: list[float] = []
    ys: list[float] = []
    for result in tracks_by_shot.values():
        for track in result.tracks:
            if track.class_name == "ball":
                continue
            for tf in track.frames:
                if tf.pitch_position is not None:
                    xs.append(float(tf.pitch_position[0]))
                    ys.append(float(tf.pitch_position[1]))

    if not xs:
        return (0.0, 12000.0), (0.0, 8000.0)

    x_arr = np.array(xs, dtype=np.float64)
    y_arr = np.array(ys, dtype=np.float64)

    x_lo, x_hi = float(np.percentile(x_arr, 5)), float(np.percentile(x_arr, 95))
    y_lo, y_hi = float(np.percentile(y_arr, 5)), float(np.percentile(y_arr, 95))

    # Add 10 % padding on each side; ensure non-degenerate range.
    x_pad = max(1.0, (x_hi - x_lo) * 0.10)
    y_pad = max(1.0, (y_hi - y_lo) * 0.10)

    return (x_lo - x_pad, x_hi + x_pad), (y_lo - y_pad, y_hi + y_pad)


def _extract_formation_descriptors(
    tracks: TracksResult,
    total_frames: int,
    fps: float,
    sample_fps: float,
    pitch_bounds: tuple[tuple[float, float], tuple[float, float]],
    grid_shape: tuple[int, int] = (8, 5),
) -> tuple[np.ndarray, int]:
    """
    Spatial player-position histograms sampled at ``sample_fps``.

    For each sampled frame, all non-ball player ``pitch_position`` values are
    binned into a ``grid_w × grid_h`` grid over the pitch and the resulting
    count vector is L2-normalised.  Frames with no detected players produce a
    zero vector (no contribution to cosine similarity).

    Returns
    -------
    descriptors : ndarray, shape (N_samples, grid_w * grid_h)
    frame_step  : int — source frames between consecutive samples
    """
    frame_step = max(1, round(fps / sample_fps))
    grid_w, grid_h = grid_shape
    dim = grid_w * grid_h
    (x_min, x_max), (y_min, y_max) = pitch_bounds

    x_range = max(x_max - x_min, 1e-6)
    y_range = max(y_max - y_min, 1e-6)

    # Build per-frame position lists from all non-ball tracks.
    positions_by_frame: dict[int, list[tuple[float, float]]] = {}
    for track in tracks.tracks:
        if track.class_name == "ball":
            continue
        for tf in track.frames:
            if tf.pitch_position is None:
                continue
            frame = tf.frame
            entry = positions_by_frame.get(frame)
            if entry is None:
                entry = []
                positions_by_frame[frame] = entry
            entry.append((float(tf.pitch_position[0]), float(tf.pitch_position[1])))

    n_samples = max(1, (total_frames + frame_step - 1) // frame_step)
    descriptors = np.zeros((n_samples, dim), dtype=np.float32)

    for sample_idx in range(n_samples):
        frame_idx = sample_idx * frame_step
        positions = positions_by_frame.get(frame_idx)
        if not positions:
            continue
        hist = np.zeros(dim, dtype=np.float32)
        for x, y in positions:
            col = int(min(grid_w - 1, max(0, (x - x_min) / x_range * grid_w)))
            row = int(min(grid_h - 1, max(0, (y - y_min) / y_range * grid_h)))
            hist[row * grid_w + col] += 1.0
        norm = float(np.linalg.norm(hist))
        if norm > 1e-6:
            descriptors[sample_idx] = hist / norm

    return descriptors, frame_step


def _align_formation_spatial(
    ref_desc: np.ndarray,
    shot_desc: np.ndarray,
    frame_step: int,
    min_overlap_frames: int,
    max_lag_frames: int | None = None,
) -> AlignmentEstimate:
    """
    Align two clips by maximising spatial formation histogram similarity.

    Uses the same similarity-matrix approach as ``_align_visual``.
    Returns ``method='player_formation'``.

    Parameters
    ----------
    max_lag_frames : int | None
        Maximum lag in frames. If provided, limits the search window.
    """
    min_overlap_samples = max(1, int(np.ceil(min_overlap_frames / frame_step)))
    max_lag_samples = None if max_lag_frames is None else max(1, max_lag_frames // frame_step)
    offsets, scores = _visual_similarity_profile(ref_desc, shot_desc, min_overlap_samples, max_lag=max_lag_samples)

    if len(scores) == 0 or scores.max() <= 0.0:
        return AlignmentEstimate(offset=0, confidence=0.0, method="player_formation", valid=False)

    best_idx = int(np.argmax(scores))
    best_d = int(offsets[best_idx])
    best_score = float(scores[best_idx])

    median_score = float(np.median(scores))
    std_score = float(np.std(scores)) + 1e-8
    z = (best_score - median_score) / std_score
    confidence = float(min(1.0, max(0.0, z / 6.0)))

    offset_frames = int(round(best_d * frame_step))

    return AlignmentEstimate(
        offset=offset_frames,
        confidence=confidence,
        method="player_formation",
        valid=confidence > 0.0,
    )


# ---------------------------------------------------------------------------
# Signal 1: Audio cross-correlation
# ---------------------------------------------------------------------------

def _load_audio_mono(clip_path: Path, sample_rate: int = 16000) -> np.ndarray:
    """Extract mono audio from a video file via FFmpeg, returned as float32."""
    cmd = [
        "ffmpeg", "-i", str(clip_path),
        "-f", "f32le", "-acodec", "pcm_f32le",
        "-ac", "1", "-ar", str(sample_rate),
        "-v", "quiet", "-",
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0 or len(result.stdout) == 0:
        return np.array([], dtype=np.float32)
    return np.frombuffer(result.stdout, dtype=np.float32)


def _audio_energy_envelope(
    audio: np.ndarray,
    sample_rate: int = 16000,
    window_s: float = 0.01,
    hop_s: float = 0.005,
) -> np.ndarray:
    """Short-time energy envelope, zero-mean and unit-variance normalised."""
    win = max(1, int(sample_rate * window_s))
    hop = max(1, int(sample_rate * hop_s))
    n_frames = max(0, (len(audio) - win) // hop + 1)
    if n_frames == 0:
        return np.array([], dtype=np.float64)
    envelope = np.array(
        [np.mean(audio[i * hop : i * hop + win] ** 2) for i in range(n_frames)],
        dtype=np.float64,
    )
    std = float(np.std(envelope))
    if std > 1e-10:
        envelope = (envelope - np.mean(envelope)) / std
    return envelope


def _align_audio(
    ref_clip: Path,
    shot_clip: Path,
    fps: float,
    sample_rate: int = 16000,
    min_zscore: float = 4.0,
    min_pearson_r: float = 0.4,
    max_lag_frames: int | None = None,
) -> AlignmentEstimate:
    """
    Align two clips by cross-correlating their raw audio waveforms.

    Uses raw waveform cross-correlation (not energy envelope) for high
    discriminative power.  Only accepts the alignment when the Pearson r of
    the aligned waveform segments exceeds ``min_pearson_r``, which filters
    out false peaks from broadcast commentary that has similar energy
    profiles but different waveform content.
    """
    ref_audio = _load_audio_mono(ref_clip, sample_rate)
    shot_audio = _load_audio_mono(shot_clip, sample_rate)

    if len(ref_audio) < sample_rate // 2 or len(shot_audio) < sample_rate // 2:
        return AlignmentEstimate(offset=0, confidence=0.0, method="audio", valid=False)

    # Raw waveform cross-correlation
    corr = fftconvolve(ref_audio, shot_audio[::-1], mode="full")
    lags = np.arange(-(len(shot_audio) - 1), len(ref_audio))

    if max_lag_frames is not None:
        max_lag_samples = int(round(max_lag_frames / fps * sample_rate))
        mask = np.abs(lags) <= max_lag_samples
        if np.any(mask):
            corr = corr[mask]
            lags = lags[mask]

    best_idx = int(np.argmax(np.abs(corr)))
    best_lag = int(lags[best_idx])

    # z-score of peak vs background
    peak_val = float(np.abs(corr[best_idx]))
    median_val = float(np.median(np.abs(corr)))
    std_val = float(np.std(np.abs(corr))) + 1e-10
    z = (peak_val - median_val) / std_val

    # Convert lag from audio samples to video frames
    offset_seconds = best_lag / sample_rate
    offset_frames = int(round(offset_seconds * fps))

    # Validate: compute Pearson r of aligned waveform segments
    pearson_r = 0.0
    if best_lag >= 0:
        overlap_len = min(len(ref_audio) - best_lag, len(shot_audio))
        if overlap_len > sample_rate // 4:
            seg_ref = ref_audio[best_lag : best_lag + overlap_len]
            seg_shot = shot_audio[:overlap_len]
            if np.std(seg_ref) > 1e-8 and np.std(seg_shot) > 1e-8:
                pearson_r = float(np.corrcoef(seg_ref, seg_shot)[0, 1])
    else:
        overlap_len = min(len(shot_audio) + best_lag, len(ref_audio))
        if overlap_len > sample_rate // 4:
            seg_ref = ref_audio[:overlap_len]
            seg_shot = shot_audio[-best_lag : -best_lag + overlap_len]
            if np.std(seg_ref) > 1e-8 and np.std(seg_shot) > 1e-8:
                pearson_r = float(np.corrcoef(seg_ref, seg_shot)[0, 1])

    valid = z >= min_zscore and pearson_r >= min_pearson_r
    confidence = float(min(1.0, max(0.0, pearson_r))) if valid else 0.0

    logging.info(
        "  [sync/audio] offset=%+d frames (%.3fs), z=%.1f, pearson_r=%.3f, valid=%s",
        offset_frames, offset_seconds, z, pearson_r, valid,
    )

    return AlignmentEstimate(
        offset=offset_frames,
        confidence=confidence,
        method="audio",
        valid=valid,
    )


# ---------------------------------------------------------------------------
# Signal 2: Appearance-based player re-ID + per-player velocity correlation
# ---------------------------------------------------------------------------

def _extract_player_appearances(
    clip_path: Path,
    tracks: TracksResult,
    min_track_frames: int = 20,
    max_samples: int = 5,
) -> dict[str, np.ndarray]:
    """
    Extract a 52-dim HSV colour descriptor (head + torso) for each long-lived
    player track.  Returns {track_id: descriptor}.
    """
    long_tracks = [
        t for t in tracks.tracks
        if t.class_name != "ball" and len(t.frames) >= min_track_frames
    ]
    if not long_tracks:
        return {}

    # Build frame → [(track_id, bbox)] requests
    frame_requests: dict[int, list[tuple[str, list[float]]]] = {}
    for t in long_tracks:
        indices = np.linspace(0, len(t.frames) - 1, min(max_samples, len(t.frames)), dtype=int)
        for idx in indices:
            f = t.frames[int(idx)]
            entry = frame_requests.get(f.frame)
            if entry is None:
                entry = []
                frame_requests[f.frame] = entry
            entry.append((t.track_id, f.bbox))

    # Read frames and extract crops
    crops_by_track: dict[str, list[np.ndarray]] = {}
    sorted_frames = sorted(frame_requests.keys())

    cap = cv2.VideoCapture(str(clip_path))
    frame_iter = iter(sorted_frames)
    next_frame = next(frame_iter, None)
    fidx = 0

    while next_frame is not None:
        ret, frame = cap.read()
        if not ret:
            break
        if fidx == next_frame:
            img_h, img_w = frame.shape[:2]
            for track_id, bbox in frame_requests[fidx]:
                x1 = max(0, int(bbox[0]))
                y1 = max(0, int(bbox[1]))
                x2 = min(img_w, int(bbox[2]))
                y2 = min(img_h, int(bbox[3]))
                if x2 <= x1 or y2 <= y1:
                    continue
                crop = frame[y1:y2, x1:x2]
                crop_resized = cv2.resize(crop, (32, 64))
                entry = crops_by_track.get(track_id)
                if entry is None:
                    entry = []
                    crops_by_track[track_id] = entry
                entry.append(crop_resized)
            next_frame = next(frame_iter, None)
        fidx += 1

    cap.release()

    # Compute descriptors
    descriptors: dict[str, np.ndarray] = {}
    for track_id, crop_list in crops_by_track.items():
        if not crop_list:
            continue
        hists: list[np.ndarray] = []
        for crop in crop_list:
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            h, w = hsv.shape[:2]

            # Head region: top 20%
            head = hsv[0 : int(h * 0.20), int(w * 0.1) : int(w * 0.9)]
            # Torso region: middle 40-70%
            torso = hsv[int(h * 0.25) : int(h * 0.70), int(w * 0.15) : int(w * 0.85)]

            def _region_hist(region: np.ndarray) -> np.ndarray:
                if region.size == 0:
                    return np.zeros(26, dtype=np.float32)
                hue = cv2.calcHist([region], [0], None, [18], [0, 180]).flatten()
                sat = cv2.calcHist([region], [1], None, [8], [0, 256]).flatten()
                return np.concatenate([hue, sat]).astype(np.float32)

            head_hist = _region_hist(head)
            torso_hist = _region_hist(torso)
            combined = np.concatenate([head_hist, torso_hist])
            norm = float(np.linalg.norm(combined))
            if norm > 1e-6:
                combined /= norm
            hists.append(combined)

        descriptors[track_id] = np.mean(hists, axis=0).astype(np.float32)

    return descriptors


def _match_players_across_views(
    ref_descs: dict[str, np.ndarray],
    shot_descs: dict[str, np.ndarray],
    min_similarity: float = 0.6,
) -> list[tuple[str, str, float]]:
    """
    Match players across two views using Hungarian assignment on appearance
    similarity.  Returns [(ref_track_id, shot_track_id, similarity), ...].
    """
    if not ref_descs or not shot_descs:
        return []

    ref_ids = list(ref_descs.keys())
    shot_ids = list(shot_descs.keys())

    ref_mat = np.array([ref_descs[tid] for tid in ref_ids])
    shot_mat = np.array([shot_descs[tid] for tid in shot_ids])

    # Cosine similarity matrix (descriptors are already L2-normalised)
    sim_matrix = ref_mat @ shot_mat.T  # (n_ref, n_shot)

    # Hungarian assignment minimises cost, so use negative similarity
    cost = -sim_matrix
    row_ind, col_ind = linear_sum_assignment(cost)

    matches: list[tuple[str, str, float]] = []
    for r, c in zip(row_ind, col_ind):
        sim = float(sim_matrix[r, c])
        if sim >= min_similarity:
            matches.append((ref_ids[r], shot_ids[c], sim))

    matches.sort(key=lambda m: -m[2])
    return matches


def _extract_bbox_velocity(
    track_frames: list,
    total_frames: int,
    frame_diagonal: float,
) -> np.ndarray:
    """
    Per-frame bbox centre velocity (normalised by frame diagonal) for a single
    track.  Frames without data are NaN.
    """
    signal = np.full(total_frames, np.nan, dtype=np.float64)
    sorted_frames = sorted(track_frames, key=lambda f: f.frame)

    for prev_f, curr_f in zip(sorted_frames, sorted_frames[1:]):
        delta = curr_f.frame - prev_f.frame
        if delta <= 0:
            continue
        prev_cx = (prev_f.bbox[0] + prev_f.bbox[2]) / 2
        prev_cy = (prev_f.bbox[1] + prev_f.bbox[3]) / 2
        curr_cx = (curr_f.bbox[0] + curr_f.bbox[2]) / 2
        curr_cy = (curr_f.bbox[1] + curr_f.bbox[3]) / 2
        dist = np.sqrt((curr_cx - prev_cx) ** 2 + (curr_cy - prev_cy) ** 2)
        speed = dist / (delta * frame_diagonal)
        if 0 <= curr_f.frame < total_frames:
            signal[curr_f.frame] = speed

    return signal


def _correlate_with_speed_sweep(
    ref_signal: np.ndarray,
    shot_signal: np.ndarray,
    speed_factors: list[float],
    min_overlap: int = 10,
) -> tuple[int, float, float]:
    """
    Cross-correlate two 1D signals, trying multiple speed factors for the
    shot signal.  Returns (best_offset, best_pearson_r, best_speed_factor).
    """
    ref_filled = _fill_nans(ref_signal)
    shot_filled = _fill_nans(shot_signal)

    if len(ref_filled) < 2 or len(shot_filled) < 2:
        return 0, 0.0, 1.0

    best_offset = 0
    best_r = 0.0
    best_speed = 1.0

    for speed in speed_factors:
        # Resample shot signal to compensate for playback speed
        if abs(speed - 1.0) < 0.01:
            resampled = shot_filled
        else:
            n_new = max(2, int(round(len(shot_filled) / speed)))
            x_old = np.linspace(0, 1, len(shot_filled))
            x_new = np.linspace(0, 1, n_new)
            resampled = np.interp(x_new, x_old, shot_filled)

        offset, conf = _cross_correlate_signals(
            signal_a=ref_filled,
            signal_b=resampled,
            max_lag=None,
            min_overlap_frames=min_overlap,
        )

        if conf > best_r:
            best_r = conf
            best_offset = offset
            best_speed = speed

    return best_offset, best_r, best_speed


def _align_player_reid(
    ref_clip: Path,
    shot_clip: Path,
    ref_tracks: TracksResult,
    shot_tracks: TracksResult,
    ref_n_frames: int,
    shot_n_frames: int,
    fps: float,
    min_track_frames: int = 20,
    min_similarity: float = 0.6,
    speed_factors: list[float] | None = None,
    agreement_tolerance: int = 5,
) -> AlignmentEstimate:
    """
    Align two clips by matching individual players across views via appearance
    descriptors, then correlating their per-player bbox velocities.

    Uses consensus voting across matched player pairs.
    """
    if speed_factors is None:
        speed_factors = [1.0]

    # Step 1: Extract appearance descriptors
    ref_descs = _extract_player_appearances(ref_clip, ref_tracks, min_track_frames)
    shot_descs = _extract_player_appearances(shot_clip, shot_tracks, min_track_frames)

    logging.info(
        "  [sync/reid] ref=%d descriptors, shot=%d descriptors",
        len(ref_descs), len(shot_descs),
    )

    if not ref_descs or not shot_descs:
        return AlignmentEstimate(offset=0, confidence=0.0, method="player_reid", valid=False)

    # Step 2: Match players across views
    matches = _match_players_across_views(ref_descs, shot_descs, min_similarity)

    logging.info("  [sync/reid] %d player matches (min_sim=%.2f)", len(matches), min_similarity)

    if not matches:
        return AlignmentEstimate(offset=0, confidence=0.0, method="player_reid", valid=False)

    # Build track lookup
    ref_track_map = {t.track_id: t for t in ref_tracks.tracks}
    shot_track_map = {t.track_id: t for t in shot_tracks.tracks}

    # Estimate frame diagonal for velocity normalisation
    cap = cv2.VideoCapture(str(ref_clip))
    img_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1920
    img_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1080
    cap.release()
    frame_diag = np.sqrt(img_w ** 2 + img_h ** 2)

    cap = cv2.VideoCapture(str(shot_clip))
    shot_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1920
    shot_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1080
    cap.release()
    shot_diag = np.sqrt(shot_w ** 2 + shot_h ** 2)

    # Step 3: Per-player velocity correlation with speed sweep
    candidate_offsets: list[tuple[int, float]] = []  # (offset, pearson_r)

    for ref_tid, shot_tid, sim in matches:
        ref_track = ref_track_map.get(ref_tid)
        shot_track = shot_track_map.get(shot_tid)
        if ref_track is None or shot_track is None:
            continue

        ref_vel = _extract_bbox_velocity(ref_track.frames, ref_n_frames, frame_diag)
        shot_vel = _extract_bbox_velocity(shot_track.frames, shot_n_frames, shot_diag)

        offset, r, speed = _correlate_with_speed_sweep(
            ref_vel, shot_vel, speed_factors, min_overlap=20,
        )

        if r > 0.2:
            candidate_offsets.append((offset, r))
            logging.info(
                "  [sync/reid]   %s↔%s sim=%.2f → offset=%+d, r=%.3f, speed=%.2f",
                ref_tid, shot_tid, sim, offset, r, speed,
            )

    if not candidate_offsets:
        return AlignmentEstimate(offset=0, confidence=0.0, method="player_reid", valid=False)

    # Step 4: Consensus voting — cluster offsets within ±tolerance
    offsets_arr = np.array([o for o, _ in candidate_offsets])
    rs_arr = np.array([r for _, r in candidate_offsets])

    best_cluster_size = 0
    best_cluster_offset = 0
    best_cluster_r = 0.0

    for i, (anchor_offset, _) in enumerate(candidate_offsets):
        within = np.abs(offsets_arr - anchor_offset) <= agreement_tolerance
        cluster_size = int(np.sum(within))
        cluster_r = float(np.max(rs_arr[within]))
        if cluster_size > best_cluster_size or (
            cluster_size == best_cluster_size and cluster_r > best_cluster_r
        ):
            best_cluster_size = cluster_size
            best_cluster_offset = int(np.median(offsets_arr[within]))
            best_cluster_r = cluster_r

    confidence = (best_cluster_size / len(candidate_offsets)) * best_cluster_r
    confidence = float(min(1.0, max(0.0, confidence)))

    logging.info(
        "  [sync/reid] consensus: offset=%+d, cluster=%d/%d, r=%.3f, conf=%.3f",
        best_cluster_offset, best_cluster_size, len(candidate_offsets),
        best_cluster_r, confidence,
    )

    return AlignmentEstimate(
        offset=best_cluster_offset,
        confidence=confidence,
        method="player_reid",
        valid=confidence > 0.0,
    )


# ---------------------------------------------------------------------------
# Matched pitch-trajectory refinement
# ---------------------------------------------------------------------------

def _align_by_matched_pitch_trajectories(
    ref_tracks: TracksResult,
    shot_tracks: TracksResult,
    ref_n_frames: int,
    shot_n_frames: int,
    coarse_offset: int,
    n_reference_frames: int = 50,
    coarse_match_distance_m: float = 20.0,
    min_overlap: int = 10,
    agreement_tolerance: int = 5,
) -> AlignmentEstimate:
    """
    Refine a coarse temporal offset using per-player pitch-coordinate trajectories.

    Algorithm
    ---------
    1. Compute the overlap window implied by ``coarse_offset``.
    2. Run a relaxed Hungarian match (``coarse_match_distance_m``) to identify
       corresponding player tracks across the two shots.
    3. For each matched pair, build dense x- and y-axis pitch-position time
       series, then cross-correlate each axis (max lag = |coarse_offset| + 30).
    4. Consensus-vote across all candidate offsets: pick the cluster with the
       most members within ``agreement_tolerance`` frames; confidence is
       ``(cluster_size / total_candidates) * mean_pearson_r``.

    Returns an ``AlignmentEstimate`` with ``method='matched_trajectory'``.
    When no usable data is available the estimate is ``valid=False``.
    """
    overlap_start, overlap_end = _compute_overlap_frames(ref_n_frames, shot_n_frames, coarse_offset)
    if overlap_end <= overlap_start:
        return AlignmentEstimate(offset=coarse_offset, confidence=0.0, method="matched_trajectory", valid=False)

    # Sample reference frames evenly across the overlap for Hungarian input.
    overlap_len = overlap_end - overlap_start
    step = max(1, overlap_len // n_reference_frames)
    ref_frames = list(range(overlap_start, overlap_end, step))[:n_reference_frames]

    matched_pairs = hungarian_match_players(
        ref_tracks,
        shot_tracks,
        sync_offset=coarse_offset,
        reference_frames=ref_frames,
        max_distance_m=coarse_match_distance_m,
    )

    if not matched_pairs:
        logging.info("  [sync/traj] no Hungarian matches at coarse distance %.1f m", coarse_match_distance_m)
        return AlignmentEstimate(offset=coarse_offset, confidence=0.0, method="matched_trajectory", valid=False)

    logging.info("  [sync/traj] %d matched player pairs for trajectory refinement", len(matched_pairs))

    max_lag = abs(coarse_offset) + 30
    candidates: list[tuple[int, float]] = []

    for ref_tid, shot_tid in matched_pairs:
        for axis in (0, 1):
            ref_traj = _extract_pitch_axis_trajectory(ref_tracks, ref_tid, ref_n_frames, axis)
            shot_traj = _extract_pitch_axis_trajectory(shot_tracks, shot_tid, shot_n_frames, axis)

            ref_filled = _fill_nans(ref_traj)
            shot_filled = _fill_nans(shot_traj)

            offset, r = _cross_correlate_signals(
                signal_a=shot_filled,
                signal_b=ref_filled,
                max_lag=max_lag,
                min_overlap_frames=min_overlap,
            )
            if r > 0.2:
                candidates.append((offset, r))

        # Keep the best axis result per pair (avoid double-counting same evidence)
        # — already done above by appending both; consensus absorbs redundancy.

    if not candidates:
        return AlignmentEstimate(offset=coarse_offset, confidence=0.0, method="matched_trajectory", valid=False)

    offsets_arr = np.array([o for o, _ in candidates])
    rs_arr = np.array([r for _, r in candidates])

    best_cluster_size = 0
    best_cluster_offset = coarse_offset
    best_cluster_mean_r = 0.0

    for anchor_offset, _ in candidates:
        within = np.abs(offsets_arr - anchor_offset) <= agreement_tolerance
        cluster_size = int(np.sum(within))
        cluster_mean_r = float(np.mean(rs_arr[within]))
        if cluster_size > best_cluster_size or (
            cluster_size == best_cluster_size and cluster_mean_r > best_cluster_mean_r
        ):
            best_cluster_size = cluster_size
            best_cluster_offset = int(np.median(offsets_arr[within]))
            best_cluster_mean_r = cluster_mean_r

    confidence = float(min(1.0, (best_cluster_size / len(candidates)) * best_cluster_mean_r))

    logging.info(
        "  [sync/traj] refined offset=%+d, cluster=%d/%d, mean_r=%.3f, conf=%.3f",
        best_cluster_offset, best_cluster_size, len(candidates), best_cluster_mean_r, confidence,
    )

    return AlignmentEstimate(
        offset=best_cluster_offset,
        confidence=confidence,
        method="matched_trajectory",
        valid=confidence > 0.0,
    )


# ---------------------------------------------------------------------------
# Signal 3: Celebration event detection
# ---------------------------------------------------------------------------

def _load_pose_data(
    poses_dir: Path,
    shot_id: str,
) -> dict[int, list[dict[str, float]]]:
    """
    Load pose keypoints from the poses JSON file.
    Returns {frame_idx: [list of player keypoint dicts]}.
    Each player's keypoints is a dict mapping joint name → (x, y, conf).
    """
    pose_file = poses_dir / f"{shot_id}_poses.json"
    if not pose_file.exists():
        return {}

    data = json.loads(pose_file.read_text())
    frames_data: dict[int, list[dict[str, tuple[float, float, float]]]] = {}

    for player in data.get("players", []):
        for frame_entry in player.get("frames", []):
            fidx = frame_entry["frame"]
            kp_dict: dict[str, tuple[float, float, float]] = {}
            for kp in frame_entry.get("keypoints", []):
                kp_dict[kp["name"]] = (kp["x"], kp["y"], kp["conf"])
            entry = frames_data.get(fidx)
            if entry is None:
                entry = []
                frames_data[fidx] = entry
            entry.append(kp_dict)

    return frames_data


def _compute_arm_raise_angle(
    kp: dict[str, tuple[float, float, float]],
    side: str,
    min_conf: float = 0.3,
) -> float | None:
    """
    Compute the angle between shoulder→wrist vector and the downward vertical.
    Returns degrees (0 = arms down, 180 = arms straight up), or None if
    keypoints are missing or low confidence.
    """
    shoulder_name = f"{side}_shoulder"
    wrist_name = f"{side}_wrist"

    shoulder = kp.get(shoulder_name)
    wrist = kp.get(wrist_name)
    if shoulder is None or wrist is None:
        return None
    if shoulder[2] < min_conf or wrist[2] < min_conf:
        return None

    dx = wrist[0] - shoulder[0]
    dy = wrist[1] - shoulder[1]  # image coords: y increases downward

    # Downward vertical is (0, 1) in image coords
    dot = dy  # dot product with (0, 1)
    mag = np.sqrt(dx ** 2 + dy ** 2)
    if mag < 1e-6:
        return None

    cos_angle = dot / mag
    cos_angle = float(np.clip(cos_angle, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_angle)))


def _compute_celebration_signal(
    poses_dir: Path,
    shot_id: str,
    total_frames: int,
    angle_threshold: float = 60.0,
    smooth_window: int = 5,
    min_conf: float = 0.3,
) -> np.ndarray:
    """
    Compute a per-frame celebration intensity signal.

    For each frame, counts the number of players (not fraction) whose arm
    raise angle exceeds ``angle_threshold``.  Using raw count rather than
    fraction avoids bias from different player populations across views —
    the absolute number of celebrating players at a given physical moment
    should be similar regardless of camera framing.

    Returns a smoothed signal of shape ``(total_frames,)``.
    """
    frames_data = _load_pose_data(poses_dir, shot_id)
    celebration_count = np.zeros(total_frames, dtype=np.float64)

    for fidx in range(total_frames):
        players = frames_data.get(fidx, [])
        for kp in players:
            left_angle = _compute_arm_raise_angle(kp, "left", min_conf)
            right_angle = _compute_arm_raise_angle(kp, "right", min_conf)
            angles = [a for a in [left_angle, right_angle] if a is not None]
            if angles and max(angles) >= angle_threshold:
                celebration_count[fidx] += 1

    if smooth_window > 1:
        kernel = np.ones(smooth_window) / smooth_window
        celebration_count = np.convolve(celebration_count, kernel, mode="same")

    return celebration_count


def _detect_celebration_onset(
    poses_dir: Path,
    shot_id: str,
    total_frames: int,
    angle_threshold: float = 60.0,
    fraction_threshold: float = 0.30,
    smooth_window: int = 5,
    min_conf: float = 0.3,
) -> int | None:
    """
    Detect the first frame where a significant fraction of visible players
    have arm raise angle exceeding ``angle_threshold``.

    Returns the frame index of celebration onset, or None if not detected.
    """
    frames_data = _load_pose_data(poses_dir, shot_id)
    if not frames_data:
        return None

    celebration_frac = np.zeros(total_frames, dtype=np.float64)

    for fidx in range(total_frames):
        players = frames_data.get(fidx, [])
        if not players:
            continue
        celebrating = 0
        total_valid = 0
        for kp in players:
            left_angle = _compute_arm_raise_angle(kp, "left", min_conf)
            right_angle = _compute_arm_raise_angle(kp, "right", min_conf)
            angles = [a for a in [left_angle, right_angle] if a is not None]
            if not angles:
                continue
            total_valid += 1
            if max(angles) >= angle_threshold:
                celebrating += 1
        if total_valid > 0:
            celebration_frac[fidx] = celebrating / total_valid

    if smooth_window > 1:
        kernel = np.ones(smooth_window) / smooth_window
        celebration_frac = np.convolve(celebration_frac, kernel, mode="same")

    above = np.where(celebration_frac >= fraction_threshold)[0]
    if len(above) == 0:
        return None

    onset = int(above[0])
    logging.info(
        "  [sync/celebration] %s: onset at frame %d (frac=%.2f)",
        shot_id, onset, celebration_frac[onset],
    )
    return onset


def _align_celebration_signal(
    ref_signal: np.ndarray,
    shot_signal: np.ndarray,
    speed_factors: list[float] | None = None,
) -> AlignmentEstimate:
    """
    Align two clips by cross-correlating their celebration count signals.

    Tries multiple speed factors to handle slow-motion replays.
    The celebration count (absolute number of celebrating players) is more
    view-invariant than the fraction, because the same physical players
    celebrate at the same moment regardless of camera framing.
    """
    if speed_factors is None:
        speed_factors = [1.0]

    if np.max(ref_signal) < 0.5 or np.max(shot_signal) < 0.5:
        return AlignmentEstimate(offset=0, confidence=0.0, method="celebration", valid=False)

    best_offset, best_r, best_speed = _correlate_with_speed_sweep(
        ref_signal, shot_signal, speed_factors, min_overlap=5,
    )

    confidence = float(min(1.0, max(0.0, best_r)))

    logging.info(
        "  [sync/celebration_corr] offset=%+d, r=%.3f, speed=%.2f",
        best_offset, best_r, best_speed,
    )

    return AlignmentEstimate(
        offset=best_offset,
        confidence=confidence,
        method="celebration",
        valid=best_r > 0.2,
    )


def _solve_offset_graph(
    n_clips: int,
    edges: list[tuple[int, int, float, float]],
) -> np.ndarray:
    """
    Weighted least-squares solve for clip offsets.

    Clips are indexed 0..n_clips-1. Clip 0 is the reference (offset = 0, pinned).
    Non-reference clips are indexed 1..n_clips-1.

    Edge convention: o_j - o_i = offset_estimate.
    Each edge is (i, j, offset_estimate, weight).

    Returns float array of length n_clips (index 0 = 0.0).
    """
    n_vars = n_clips - 1  # reference is pinned at 0
    result = np.zeros(n_clips, dtype=np.float64)
    if n_vars <= 0:
        return result

    valid_edges = [(i, j, e, w) for i, j, e, w in edges if w > 0]
    if not valid_edges:
        return result

    n_edges = len(valid_edges)
    A = np.zeros((n_edges, n_vars), dtype=np.float64)
    b = np.zeros(n_edges, dtype=np.float64)
    W = np.zeros(n_edges, dtype=np.float64)

    for k, (i, j, e, w) in enumerate(valid_edges):
        # Constraint: o_j - o_i = e  (reference has o_0 = 0, not a variable)
        if i > 0:
            A[k, i - 1] = -1.0
        if j > 0:
            A[k, j - 1] = +1.0
        b[k] = float(e)
        W[k] = float(w)

    sqrt_W = np.sqrt(W)
    x, _, _, _ = np.linalg.lstsq(
        sqrt_W[:, None] * A,
        sqrt_W * b,
        rcond=None,
    )
    result[1:] = x
    return result


class TemporalSyncStage(BaseStage):
    name = "sync"

    def __init__(
        self,
        config: dict,
        output_dir: Path,
        ball_detector: BallDetector | None = None,
        **_,
    ) -> None:
        super().__init__(config, output_dir)
        # ball_detector kept for interface compatibility; not used in this stage.
        _ = ball_detector

    def is_complete(self) -> bool:
        return (self.output_dir / "sync" / "sync_map.json").exists()

    def run(self) -> None:
        sync_dir = self.output_dir / "sync"
        sync_dir.mkdir(parents=True, exist_ok=True)

        cfg = self.config.get("sync", {})
        min_conf = float(cfg.get("min_confidence", 0.3))
        min_overlap_frames = int(cfg.get("min_overlap_frames", 25))
        sample_fps = float(cfg.get("sample_fps", 5.0))
        agreement_tolerance_frames = int(cfg.get("agreement_tolerance_frames", 8))
        grid_cfg = cfg.get("formation_grid", [8, 5])
        grid_shape = (int(grid_cfg[0]), int(grid_cfg[1]))

        # Re-ID config
        reid_min_track_frames = int(cfg.get("reid_min_track_frames", 20))
        reid_min_similarity = float(cfg.get("reid_min_similarity", 0.6))
        speed_factors_cfg = cfg.get("speed_factors", [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5])
        speed_factors = [float(s) for s in speed_factors_cfg]
        audio_min_zscore = float(cfg.get("audio_min_zscore", 4.0))

        # Trajectory-refinement config
        traj_enabled = bool(cfg.get("trajectory_refinement_enabled", True))
        traj_n_reference_frames = int(cfg.get("trajectory_refinement_n_reference_frames", 50))
        traj_coarse_match_distance_m = float(cfg.get("trajectory_refinement_coarse_match_distance_m", 20.0))

        shots_dir = self.output_dir / "shots"
        if not (shots_dir / "shots_manifest.json").exists():
            logging.info("  [sync] inferring shots_manifest.json from clips")
        manifest = ShotsManifest.load_or_infer(shots_dir, persist=True)

        if len(manifest.shots) < 2:
            ref = manifest.shots[0].id if manifest.shots else ""
            SyncMap(reference_shot=ref).save(sync_dir / "sync_map.json")
            logging.info("  [sync] only one shot; no sync needed")
            return

        tracks_dir = self.output_dir / "tracks"
        poses_dir = self.output_dir / "poses"
        tracks_by_shot: dict[str, TracksResult] = {}
        for shot in manifest.shots:
            tracks_file = tracks_dir / f"{shot.id}_tracks.json"
            if tracks_file.exists():
                tracks_by_shot[shot.id] = TracksResult.load(tracks_file)

        fps = manifest.fps if manifest.fps > 0 else 25.0

        # Measure each clip's frame count.
        n_frames_by_shot: dict[str, int] = {}
        for shot in manifest.shots:
            clip_path = self.output_dir / shot.clip_file
            cap = cv2.VideoCapture(str(clip_path))
            n_frames = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
            cap.release()
            n_frames_by_shot[shot.id] = n_frames
            n_tracks = len(tracks_by_shot[shot.id].tracks) if shot.id in tracks_by_shot else 0
            logging.info("  [sync] %s: %d frames, %d tracks", shot.id, n_frames, n_tracks)

        reference = manifest.shots[0].id
        ref_clip = self.output_dir / manifest.shots[0].clip_file
        ref_n_frames = n_frames_by_shot.get(reference, 1)

        alignments: list[Alignment] = []

        for shot in manifest.shots[1:]:
            shot_clip = self.output_dir / shot.clip_file
            shot_n_frames = n_frames_by_shot.get(shot.id, 1)

            logging.info("  [sync] aligning %s → %s", reference, shot.id)

            # --- Signal 1: Audio cross-correlation ---
            audio_estimate = _align_audio(
                ref_clip=ref_clip,
                shot_clip=shot_clip,
                fps=fps,
                min_zscore=audio_min_zscore,
            )

            # --- Signal 2: Appearance-based player re-ID ---
            ref_tracks = tracks_by_shot.get(reference)
            shot_tracks = tracks_by_shot.get(shot.id)

            reid_estimate = AlignmentEstimate(
                offset=0, confidence=0.0, method="player_reid", valid=False,
            )
            if ref_tracks is not None and shot_tracks is not None:
                reid_estimate = _align_player_reid(
                    ref_clip=ref_clip,
                    shot_clip=shot_clip,
                    ref_tracks=ref_tracks,
                    shot_tracks=shot_tracks,
                    ref_n_frames=ref_n_frames,
                    shot_n_frames=shot_n_frames,
                    fps=fps,
                    min_track_frames=reid_min_track_frames,
                    min_similarity=reid_min_similarity,
                    speed_factors=speed_factors,
                    agreement_tolerance=agreement_tolerance_frames,
                )

            # --- Signal 3: Celebration count cross-correlation ---
            # Cross-correlate the absolute count of celebrating players.
            # Try speed factors for slow-motion replay handling.
            ref_celeb_signal = _compute_celebration_signal(
                poses_dir, reference, ref_n_frames,
            )
            shot_celeb_signal = _compute_celebration_signal(
                poses_dir, shot.id, shot_n_frames,
            )
            celebration_corr_estimate = _align_celebration_signal(
                ref_celeb_signal, shot_celeb_signal,
                speed_factors=speed_factors,
            )

            # Peak-based alignment: frame of maximum celebration count.
            # The peak celebration moment (most players celebrating
            # simultaneously) is more view-invariant than onset, which is
            # biased by different player-population fractions across views.
            ref_peak = int(np.argmax(ref_celeb_signal)) if np.max(ref_celeb_signal) > 0.5 else None
            shot_peak = int(np.argmax(shot_celeb_signal)) if np.max(shot_celeb_signal) > 0.5 else None

            celebration_offset: int | None = None
            if ref_peak is not None and shot_peak is not None:
                celebration_offset = ref_peak - shot_peak
                logging.info(
                    "  [sync/celebration] %s → %s: peak ref=%d (count=%.1f), shot=%d (count=%.1f) → offset=%+d",
                    reference, shot.id,
                    ref_peak, ref_celeb_signal[ref_peak],
                    shot_peak, shot_celeb_signal[shot_peak],
                    celebration_offset,
                )

            # --- Fusion: audio → celebration_corr → onset → reid → legacy ---
            #
            # Priority:
            # 1. Audio (high confidence) — frame-accurate for shared feeds
            # 2. Celebration signal cross-correlation — handles speed differences
            # 3. Celebration onset alignment — simpler, good for normal-speed
            # 4. Reid velocity consensus — individual player matching
            # 5. Legacy visual/formation fallback
            if audio_estimate.valid and audio_estimate.confidence >= 0.5:
                best = audio_estimate
            elif celebration_corr_estimate.valid and celebration_corr_estimate.confidence > 0.3:
                best = celebration_corr_estimate
            elif celebration_offset is not None:
                celebration_conf = 0.40

                # If reid agrees within ±30 frames, use reid's finer offset
                if reid_estimate.valid and abs(reid_estimate.offset - celebration_offset) <= 30:
                    best = AlignmentEstimate(
                        offset=reid_estimate.offset,
                        confidence=min(1.0, 0.40 + reid_estimate.confidence),
                        method="celebration+reid",
                        valid=True,
                    )
                else:
                    best = AlignmentEstimate(
                        offset=celebration_offset,
                        confidence=celebration_conf,
                        method="celebration",
                        valid=True,
                    )
            elif reid_estimate.valid:
                best = reid_estimate
            elif audio_estimate.valid:
                best = audio_estimate
            else:
                # Fall back to legacy visual + formation signals
                visual_estimate = _align_visual(
                    ref_clip=ref_clip,
                    shot_clip=shot_clip,
                    fps=fps,
                    min_overlap_frames=min_overlap_frames,
                    sample_fps=sample_fps,
                )

                # Compute formation descriptors on demand for fallback
                ref_tracks_fb = tracks_by_shot.get(reference)
                shot_tracks_fb = tracks_by_shot.get(shot.id)
                formation_estimate = AlignmentEstimate(
                    offset=0, confidence=0.0, method="player_formation", valid=False,
                )
                if ref_tracks_fb is not None and shot_tracks_fb is not None:
                    pitch_bounds = _compute_pitch_bounds(tracks_by_shot)
                    ref_fd, ref_fs = _extract_formation_descriptors(
                        ref_tracks_fb, ref_n_frames, fps, sample_fps,
                        pitch_bounds, grid_shape,
                    )
                    shot_fd, _ = _extract_formation_descriptors(
                        shot_tracks_fb, shot_n_frames, fps, sample_fps,
                        pitch_bounds, grid_shape,
                    )
                    formation_estimate = _align_formation_spatial(
                        ref_fd, shot_fd, ref_fs, min_overlap_frames,
                    )

                best = _fuse_alignment_estimates(
                    ball_estimate=visual_estimate,
                    player_estimate=formation_estimate,
                    agreement_tolerance_frames=agreement_tolerance_frames,
                )

            # --- Trajectory refinement: use preliminary pitch-position matches ---
            # Runs after the coarse signal cascade and replaces `best` when it
            # produces a higher-confidence result.
            if traj_enabled and best.valid and ref_tracks is not None and shot_tracks is not None:
                refined = _align_by_matched_pitch_trajectories(
                    ref_tracks=ref_tracks,
                    shot_tracks=shot_tracks,
                    ref_n_frames=ref_n_frames,
                    shot_n_frames=shot_n_frames,
                    coarse_offset=best.offset,
                    n_reference_frames=traj_n_reference_frames,
                    coarse_match_distance_m=traj_coarse_match_distance_m,
                    agreement_tolerance=agreement_tolerance_frames,
                )
                if refined.valid and refined.confidence > best.confidence:
                    best = AlignmentEstimate(
                        offset=refined.offset,
                        confidence=min(1.0, best.confidence * 0.3 + refined.confidence * 0.7),
                        method=f"{best.method}+matched_trajectory",
                        valid=True,
                    )
                    logging.info(
                        "  [sync/traj] adopted refined offset=%+d conf=%.2f",
                        best.offset, best.confidence,
                    )

            offset = best.offset
            confidence = best.confidence

            start, end = _compute_overlap_frames(ref_n_frames, shot_n_frames, offset)
            overlap = max(0, end - start)

            method = best.method
            if confidence < min_conf or overlap < min_overlap_frames:
                method = "low_confidence"

            alignments.append(Alignment(
                shot_id=shot.id,
                frame_offset=offset,
                confidence=confidence,
                method=method,
                overlap_frames=[start, end],
            ))

            flag = "" if confidence >= min_conf else " [WARNING] low confidence"
            logging.info(
                "  [sync] %s → %s  offset=%+d  conf=%.2f  method=%s%s",
                reference, shot.id, offset, confidence, method, flag,
            )
            print(
                f"  -> {shot.id} offset={offset:+d} frames, "
                f"confidence={confidence:.2f} ({method}){flag}"
            )

        SyncMap(reference_shot=reference, alignments=alignments).save(
            sync_dir / "sync_map.json"
        )
