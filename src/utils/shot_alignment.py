"""Auto-alignment of shots within a highlight group.

Different camera angles of the same event share almost no pixels, but
their 1-D *motion-energy profiles* (how much the image changes frame to
frame) are correlated: quiet build-up, the strike/collision, the
eruption after. We cross-correlate each member's profile against the
group reference's and read the offset off the best lag.

This runs on the extracted (already speed-normalised) clips, so slow-mo
replays correlate at real-time scale. Results are approximate (±a few
frames) and honestly labelled: ``motion_profile`` above the confidence
threshold, ``low_confidence`` with an align-ends fallback below it —
replays typically end just after the key moment, so end-alignment is
the best blind prior. The dashboard's group sync editor is the
correction surface.

Sign convention (matches ``src/schemas/sync_map.py``):

    frame_offset = matched_frame_in_shot - matched_frame_in_reference

The NCC search places the shot's frame 0 at ref-axis position ``L``;
the event at shot frame ``S`` then sits at ref frame ``L + S``, so
``frame_offset = S - (L + S) = -L``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d

logger = logging.getLogger(__name__)

_MIN_STD = 1e-9


@dataclass(frozen=True)
class AlignmentResult:
    frame_offset: int
    confidence: float
    method: str  # "motion_profile" | "low_confidence"


def motion_energy_curve(
    clip_path: Path,
    *,
    width_px: int = 192,
    smooth_sigma: float = 2.0,
) -> np.ndarray:
    """Per-frame mean |gray frame-diff| at reduced resolution.

    Index ``i`` holds the energy between frames ``i`` and ``i+1``
    (length = frame_count - 1). Empty/unreadable clips return an empty
    array.
    """
    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        return np.zeros(0)
    try:
        energies: list[float] = []
        prev: np.ndarray | None = None
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            h, w = frame.shape[:2]
            scale = width_px / max(1, w)
            small = cv2.resize(
                frame, (width_px, max(1, int(round(h * scale)))),
                interpolation=cv2.INTER_AREA,
            )
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
            if prev is not None:
                energies.append(float(np.mean(np.abs(gray - prev))))
            prev = gray
    finally:
        cap.release()
    curve = np.asarray(energies, dtype=np.float64)
    if curve.size and smooth_sigma > 0:
        curve = gaussian_filter1d(curve, sigma=smooth_sigma)
    return curve


def align_curves(
    ref: np.ndarray,
    other: np.ndarray,
    *,
    min_overlap: int,
    min_confidence: float = 0.5,
) -> AlignmentResult:
    """Best NCC lag of ``other`` against ``ref`` → AlignmentResult."""
    n, m = len(ref), len(other)
    fallback_offset = m - n  # align clip ends (see module docstring)
    if n < min_overlap or m < min_overlap:
        return AlignmentResult(fallback_offset, 0.0, "low_confidence")

    best_ncc = -np.inf
    best_lag = 0
    for lag in range(-(m - min_overlap), n - min_overlap + 1):
        a = max(0, lag)
        b = min(n, lag + m)
        if b - a < min_overlap:
            continue
        r_seg = ref[a:b]
        s_seg = other[a - lag:b - lag]
        r_std = r_seg.std()
        s_std = s_seg.std()
        if r_std < _MIN_STD or s_std < _MIN_STD:
            continue
        ncc = float(np.mean(
            (r_seg - r_seg.mean()) / r_std * (s_seg - s_seg.mean()) / s_std
        ))
        if ncc > best_ncc:
            best_ncc = ncc
            best_lag = lag

    if not np.isfinite(best_ncc):
        return AlignmentResult(fallback_offset, 0.0, "low_confidence")
    confidence = float(max(0.0, min(1.0, best_ncc)))
    if confidence < min_confidence:
        return AlignmentResult(fallback_offset, confidence, "low_confidence")
    return AlignmentResult(-best_lag, confidence, "motion_profile")


def align_group(
    clip_paths: dict[str, Path],
    reference_id: str,
    *,
    width_px: int = 192,
    smooth_sigma: float = 2.0,
    min_overlap_frames: int = 25,
    min_confidence: float = 0.5,
) -> dict[str, AlignmentResult]:
    """Align every clip in a group against the reference clip.

    The reference maps to offset 0 / confidence 1. Unreadable members
    fall back to ``low_confidence`` at offset 0 rather than failing the
    group.
    """
    ref_path = clip_paths.get(reference_id)
    ref_curve = (
        motion_energy_curve(
            ref_path, width_px=width_px, smooth_sigma=smooth_sigma,
        )
        if ref_path is not None else np.zeros(0)
    )
    results: dict[str, AlignmentResult] = {
        reference_id: AlignmentResult(0, 1.0, "motion_profile"),
    }
    for shot_id, path in clip_paths.items():
        if shot_id == reference_id:
            continue
        curve = motion_energy_curve(
            path, width_px=width_px, smooth_sigma=smooth_sigma,
        )
        if not len(curve) or not len(ref_curve):
            logger.warning(
                "align_group: no usable motion curve for %s; defaulting "
                "to offset 0 (low_confidence)", shot_id,
            )
            results[shot_id] = AlignmentResult(0, 0.0, "low_confidence")
            continue
        results[shot_id] = align_curves(
            ref_curve, curve,
            min_overlap=min_overlap_frames,
            min_confidence=min_confidence,
        )
    return results
