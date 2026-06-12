"""Second-pass ball detection: corridor prediction + candidate gating.

Pass 1 (the streaming detect loop) misses frames; this module predicts
where the ball should be on those frames (a forward/backward IMM fusion
over pass-1 observations ONLY — second-pass output never steers its own
corridor) and gates low-threshold detector candidates against that
corridor. Pure logic: no video access, no torch (the stage owns I/O).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from src.utils.ball_tracker import BallTracker

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
