"""Second-pass ball detection: corridor prediction + candidate gating.

Pass 1 (the streaming detect loop) misses frames; this module predicts
where the ball should be on those frames (a forward/backward IMM fusion
over pass-1 observations ONLY — second-pass output never steers its own
corridor) and gates low-threshold detector candidates against that
corridor. Pure logic: no video access, no torch (the stage owns I/O).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np

from src.utils.ball_tracker import BallTracker
from src.utils.camera_projection import pixel_ray

# Covariance floor (px²) so a corridor next to a confident observation
# still admits detector-sized localisation error.
_COV_FLOOR_PX2 = 4.0 ** 2


@dataclass(frozen=True)
class SecondPassCfg:
    enabled: bool = True
    candidate_min_score: float = 0.05
    top_k: int = 5
    corridor_sigma: float = 3.0
    accept_min: float = 0.25
    zoom_min_ball_px: float = 8.0
    zoom_crop_px: int = 320
    # Phase A: also revisit weak (low-confidence) pass-1 frames, not only
    # gaps, with the buffer-safe zoom — and only replace a pass-1 detection
    # when the zoom is strictly more confident.
    redetect_low_conf: bool = True
    redetect_max_conf: float = 0.5


@dataclass(frozen=True)
class SecondPassDetection:
    frame: int
    uv: tuple[float, float]
    combined_score: float
    used_zoom: bool


def fuse_gaussians(
    m1: np.ndarray, c1: np.ndarray, m2: np.ndarray, c2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Product of two 2D Gaussians (precision-weighted mean, fused cov)."""
    i1, i2 = np.linalg.inv(c1), np.linalg.inv(c2)
    cov = np.linalg.inv(i1 + i2)
    return cov @ (i1 @ m1 + i2 @ m2), cov


def _cov_matrix(pos_cov: tuple[float, float, float]) -> np.ndarray:
    suu, svv, suv = pos_cov
    return np.array([[suu, suv], [suv, svv]], dtype=float)


def _run_pass(
    per_frame_uv: dict[int, tuple[float, float] | None],
    order: range,
    tracker_factory: Callable[[], BallTracker],
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    tracker = tracker_factory()
    out: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for i, f in enumerate(order):
        step = tracker.update(i, per_frame_uv.get(f))
        if step.uv is not None and step.pos_cov is not None:
            out[f] = (np.array(step.uv, dtype=float), _cov_matrix(step.pos_cov))
    return out


def corridor_predictions(
    per_frame_uv: dict[int, tuple[float, float] | None],
    n_frames: int,
    tracker_factory: Callable[[], BallTracker],
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Per-frame (mean, cov) search corridor from pass-1 observations.

    Forward and backward IMM passes (the constant-velocity model is
    time-symmetric), fused where both are initialised.

    Statistical invariant: the independent-product fusion (fuse_gaussians)
    is valid because corridors are only *consumed* at gap frames, where
    the forward pass has seen only pre-gap observations and the backward
    pass only post-gap observations — making the two estimates genuinely
    independent. At observed frames the fused covariance is optimistic,
    but those frames are never queried by the second pass.
    """
    fwd = _run_pass(per_frame_uv, range(n_frames), tracker_factory)
    bwd = _run_pass(per_frame_uv, range(n_frames - 1, -1, -1), tracker_factory)
    out: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for f in range(n_frames):
        a, b = fwd.get(f), bwd.get(f)
        if a is not None and b is not None:
            out[f] = fuse_gaussians(a[0], a[1], b[0], b[1])
        elif a is not None or b is not None:
            out[f] = a if a is not None else b
    return out


# ---------------------------------------------------------------------------
# Sources that mean "pass 1 accepted evidence on this frame".
# ---------------------------------------------------------------------------
PASS1_SOURCES = ("detector", "anchor", "bridge")


def find_gap_runs(
    sources: dict[int, str],
    outlier_frames: set[int],
    n_frames: int,
) -> list[tuple[int, int]]:
    """Consecutive runs of frames with no accepted pass-1 detection."""
    gap = [
        f for f in range(n_frames)
        if sources.get(f) not in PASS1_SOURCES or f in outlier_frames
    ]
    runs: list[tuple[int, int]] = []
    for f in gap:
        if runs and f == runs[-1][1] + 1:
            runs[-1] = (runs[-1][0], f)
        else:
            runs.append((f, f))
    return runs


def _group_runs(frames: list[int]) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    for f in frames:
        if runs and f == runs[-1][1] + 1:
            runs[-1] = (runs[-1][0], f)
        else:
            runs.append((f, f))
    return runs


def find_revisit_runs(
    sources: dict[int, str],
    outlier_frames: set[int],
    confidences: dict[int, float],
    n_frames: int,
    max_conf: float,
) -> list[tuple[int, int]]:
    """Runs of frames worth re-detecting at high resolution: detection gaps
    PLUS frames with an accepted pass-1 detection whose confidence is below
    ``max_conf`` (the ball was found but weakly — a zoom often sharpens it).

    Phase A of the direction-change rework: promotes the buffer-safe
    second-pass zoom from gaps-only to all weak frames.
    """
    need = [
        f for f in range(n_frames)
        if (sources.get(f) not in PASS1_SOURCES or f in outlier_frames)
        or (
            sources.get(f) in PASS1_SOURCES
            and float(confidences.get(f, 0.0)) < max_conf
        )
    ]
    return _group_runs(need)


def best_gated_candidate(
    candidates: list[tuple[float, float, float]],
    mean: np.ndarray,
    cov: np.ndarray,
    cfg: SecondPassCfg,
) -> tuple[tuple[float, float], float] | None:
    """Best corridor-gated candidate as ``((u, v), combined_score)``.

    Gate: Mahalanobis² <= corridor_sigma². Score:
    ``candidate_score * exp(-0.5 * d² / corridor_sigma²)``, accepted when
    it clears ``accept_min``.
    """
    cov_f = cov + _COV_FLOOR_PX2 * np.eye(2)
    inv = np.linalg.inv(cov_f)
    best: tuple[tuple[float, float], float] | None = None
    for u, v, score in candidates:
        d = np.array([u, v], dtype=float) - mean
        d2 = float(d @ inv @ d)
        if d2 > cfg.corridor_sigma ** 2:
            continue
        combined = float(score) * math.exp(-0.5 * d2 / cfg.corridor_sigma ** 2)
        if best is None or combined > best[1]:
            best = ((float(u), float(v)), combined)
    if best is None or best[1] < cfg.accept_min:
        return None
    return best


def apparent_ball_px(
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    uv: tuple[float, float],
    ball_radius_m: float,
    distortion: tuple[float, float] = (0.0, 0.0),
) -> float | None:
    """Predicted apparent ball diameter (px) at a pixel's ground depth.

    Ray-casts the pixel to the ball-centre plane ``z = ball_radius_m``;
    None when the ray never reaches it (above-horizon prediction). A
    ground-depth approximation — airborne balls are nearer the camera
    and look bigger, so this under-zooms, never over-zooms.
    """
    C, d_hat = pixel_ray(uv, K, R, t, distortion)
    dz = float(d_hat[2])
    if abs(dz) < 1e-9:
        return None
    s = (ball_radius_m - float(C[2])) / dz
    if s <= 0:
        return None
    return float(K[0][0]) * (2.0 * ball_radius_m) / s


def map_crop_candidates(
    candidates: list[tuple[float, float, float]],
    x0: int,
    y0: int,
) -> list[tuple[float, float, float]]:
    """Translate crop-space candidates back into full-frame pixels."""
    return [(u + x0, v + y0, s) for u, v, s in candidates]


def filter_in_bounds(
    candidates: list[tuple[float, float, float]],
    width: int,
    height: int,
) -> list[tuple[float, float, float]]:
    """Drop candidates outside the image — letterbox-padding blobs map
    through the inverse affine to impossible coordinates."""
    return [
        (u, v, s) for u, v, s in candidates
        if 0.0 <= u < width and 0.0 <= v < height
    ]
