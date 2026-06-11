"""Per-shot visual features for highlights ingestion.

Sampled-frame statistics that drive three decisions in prepare_shots'
split mode:

- ``kind``  (gameplay | reaction | transition): pitch-green fraction
  separates on-pitch action from crowd/bench close-ups; brightness
  separates fades.
- ``scale`` (wide | medium | tight): pitch-ratio bands — a proxy for
  how much of the frame is playing surface, which tracks camera scale.
- ``speed_factor``: zoom-invariant motion rate (LK optical-flow
  magnitude normalised by Sobel gradient density, ported from the
  pre-broadcast-mono prepare_shots at 262d08a~1). Slow-motion replays
  have a *lower* temporal motion rate than real-time shots of the same
  zoom, so ``speed_factor = reference_rate / shot_rate > 1`` flags them.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import numpy as np

from src.utils.shot_split import ShotSpan

# Below this mean Sobel gradient the frame is near-uniform (black cut,
# heavy defocus) and flow/gradient normalisation would divide by noise;
# we fall back to the raw flow magnitude instead.
_MIN_GRADIENT: float = 0.1

# Motion rates below this are "static scene" — speed estimation is
# meaningless there, so the speed factor defaults to 1.0.
_MIN_MOTION_RATE: float = 1e-4


@dataclass(frozen=True)
class ShotFeatures:
    """Sampled-frame statistics for one detected span."""

    span: ShotSpan
    pitch_ratio_median: float
    pitch_ratio_peak: float
    brightness_min: float
    brightness_range: float
    motion_rate: float
    kind: str = "gameplay"
    scale: str = "medium"
    speed_factor: float = 1.0


def pitch_ratio(frame_bgr: np.ndarray) -> float:
    """Fraction of the frame inside the broadcast-turf green HSV band."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (35, 40, 40), (95, 255, 255))
    return float(mask.mean() / 255.0)


def _brightness(frame_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return float(gray.mean() / 255.0)


def _read_frame(cap: cv2.VideoCapture, frame_idx: int) -> np.ndarray | None:
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_idx))
    ok, frame = cap.read()
    return frame if ok else None


def _pair_motion_rate(gray_a: np.ndarray, gray_b: np.ndarray) -> float | None:
    """Zoom-invariant motion rate for one consecutive-frame pair.

    Returns ``None`` when no corners could be tracked (caller skips the
    sample rather than treating it as zero motion).
    """
    grad_x = cv2.Sobel(gray_a, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray_a, cv2.CV_64F, 0, 1, ksize=3)
    gradient = float(np.mean(np.sqrt(grad_x ** 2 + grad_y ** 2)))

    corners = cv2.goodFeaturesToTrack(
        gray_a, maxCorners=250, qualityLevel=0.01, minDistance=7, blockSize=7,
    )
    if corners is None or len(corners) == 0:
        return None
    next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
        gray_a, gray_b, corners, None,
    )
    if next_pts is None or status is None:
        return None
    tracked = status.reshape(-1) == 1
    if not tracked.any():
        return None
    flow = float(np.mean(np.linalg.norm(
        (next_pts - corners).reshape(-1, 2)[tracked], axis=1,
    )))
    if gradient < _MIN_GRADIENT:
        return flow
    return flow / gradient


def compute_span_features(
    video_path: Path,
    spans: list[ShotSpan],
    *,
    sample_points: list[float],
    motion_samples: int = 3,
    **classify_kwargs,
) -> list[ShotFeatures]:
    """Sample each span at ``sample_points`` fractions and build features.

    ``kind``/``scale`` are filled with :func:`classify_kind` /
    :func:`classify_scale` (threshold overrides via ``classify_kwargs``,
    split by function signature); ``speed_factor`` stays 1.0 until
    :func:`estimate_speed_factors` runs over the whole feature list.
    """
    kind_kwargs = {k: v for k, v in classify_kwargs.items()
                   if k in _KIND_THRESHOLDS}
    scale_kwargs = {k: v for k, v in classify_kwargs.items()
                    if k in _SCALE_THRESHOLDS}
    unknown = set(classify_kwargs) - set(kind_kwargs) - set(scale_kwargs)
    if unknown:
        raise TypeError(f"unknown classify thresholds: {sorted(unknown)}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {video_path}")
    try:
        features: list[ShotFeatures] = []
        for span in spans:
            n_frames = max(1, span.end_frame - span.start_frame + 1)
            sample_idxs = sorted({
                span.start_frame + int(round(min(1.0, max(0.0, p)) * (n_frames - 1)))
                for p in sample_points
            })
            ratios: list[float] = []
            brightness: list[float] = []
            for idx in sample_idxs:
                frame = _read_frame(cap, idx)
                if frame is None:
                    continue
                ratios.append(pitch_ratio(frame))
                brightness.append(_brightness(frame))

            rates: list[float] = []
            stride = max(1, len(sample_idxs) // max(1, motion_samples))
            for idx in sample_idxs[::stride][:motion_samples]:
                pair_idx = min(idx, span.end_frame - 1)
                frame_a = _read_frame(cap, pair_idx)
                ok, frame_b = cap.read()
                if frame_a is None or not ok:
                    continue
                rate = _pair_motion_rate(
                    cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY),
                    cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY),
                )
                if rate is not None:
                    rates.append(rate)

            ratios = ratios or [0.0]
            brightness = brightness or [0.0]
            f = ShotFeatures(
                span=span,
                pitch_ratio_median=float(np.median(ratios)),
                pitch_ratio_peak=float(max(ratios)),
                brightness_min=float(min(brightness)),
                brightness_range=float(max(brightness) - min(brightness)),
                motion_rate=float(np.median(rates)) if rates else 0.0,
            )
            f = replace(
                f,
                kind=classify_kind(f, **kind_kwargs),
                scale=classify_scale(f, **scale_kwargs),
            )
            features.append(f)
        return features
    finally:
        cap.release()


_KIND_THRESHOLDS = {
    "reaction_max_median_pitch_ratio",
    "reaction_max_peak_pitch_ratio",
    "fade_black_frame_threshold",
    "fade_min_brightness_range",
    "transition_max_duration_s",
}
_SCALE_THRESHOLDS = {"wide_min_pitch_ratio", "tight_max_pitch_ratio"}


def classify_kind(
    f: ShotFeatures,
    *,
    reaction_max_median_pitch_ratio: float = 0.12,
    reaction_max_peak_pitch_ratio: float = 0.20,
    fade_black_frame_threshold: float = 0.18,
    fade_min_brightness_range: float = 0.25,
    transition_max_duration_s: float = 2.0,
) -> str:
    """gameplay | reaction | transition for one span's features."""
    # Broadcast fades/wipes last around a second — a long span that
    # merely samples one dark frame (shadowed close-up, replay graphic)
    # is gameplay, not a transition.
    short_enough = f.span.duration_s <= transition_max_duration_s
    dips_to_black = f.brightness_min <= fade_black_frame_threshold
    fades = (short_enough and dips_to_black
             and f.brightness_range >= fade_min_brightness_range)
    hard_black = f.brightness_min + f.brightness_range < 0.06
    if fades or hard_black:
        return "transition"
    if (f.pitch_ratio_median < reaction_max_median_pitch_ratio
            and f.pitch_ratio_peak < reaction_max_peak_pitch_ratio):
        return "reaction"
    return "gameplay"


def classify_scale(
    f: ShotFeatures,
    *,
    wide_min_pitch_ratio: float = 0.40,
    tight_max_pitch_ratio: float = 0.22,
) -> str:
    """wide | medium | tight from the pitch-green fraction."""
    if f.pitch_ratio_median >= wide_min_pitch_ratio:
        return "wide"
    if f.pitch_ratio_median < tight_max_pitch_ratio:
        return "tight"
    return "medium"


def estimate_speed_factors(
    features: list[ShotFeatures],
    *,
    replay_min_speed_factor: float = 1.25,
) -> list[ShotFeatures]:
    """Fill ``speed_factor`` on every feature (returns a new list).

    The real-time reference rate is the 90th percentile of motion rates
    over wide gameplay shots — live broadcast shots cluster at the true
    rate while slow-mo replays sit below it, so a high percentile lands
    inside the live cluster without letting a single outlier dominate.
    """
    pool = [f.motion_rate for f in features
            if f.kind == "gameplay" and f.scale == "wide"
            and f.motion_rate > _MIN_MOTION_RATE]
    if not pool:
        pool = [f.motion_rate for f in features
                if f.kind == "gameplay" and f.motion_rate > _MIN_MOTION_RATE]
    if not pool:
        pool = [f.motion_rate for f in features
                if f.motion_rate > _MIN_MOTION_RATE]
    if not pool:
        return [replace(f, speed_factor=1.0) for f in features]

    reference_rate = float(np.percentile(pool, 90))
    out: list[ShotFeatures] = []
    for f in features:
        if f.motion_rate <= _MIN_MOTION_RATE or reference_rate <= 0:
            out.append(replace(f, speed_factor=1.0))
            continue
        sf = float(np.clip(reference_rate / f.motion_rate, 0.3, 4.0))
        out.append(replace(f, speed_factor=sf))
    return out


def is_replay(
    f: ShotFeatures,
    *,
    replay_min_speed_factor: float = 1.25,
) -> bool:
    return f.speed_factor >= replay_min_speed_factor
