"""Plausibility checks for ball trajectories and grounded runs.

Layer 1 of the ball-tracking improvement plan: every fitted parabola or
Magnus segment is sampled at several time points and rejected unless all
samples stay within physical bounds (pitch envelope + speed + height).

Layer 2 lives here too because it shares the pitch geometry helpers:
:func:`find_implausible_grounded_runs` flags contiguous ``grounded``
runs whose ground-projected world positions imply impossible rolling
motion (off-pitch or > 35 m/s).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class PlausibilityCfg:
    """Physical bounds for an accepted trajectory."""

    z_max_m: float
    horizontal_speed_max_m_s: float
    pitch_margin_m: float


@dataclass(frozen=True)
class PitchDims:
    length_m: float
    width_m: float


def _sample_positions(
    p0: np.ndarray,
    v0: np.ndarray,
    omega: np.ndarray | None,
    duration_s: float,
    fps: float,
    g: float = -9.81,
    drag_k_over_m: float = 0.005,
) -> np.ndarray:
    """Return (N, 3) sampled positions along the trajectory.

    Without omega: closed-form parabola.
    With omega: forward Euler integration matching
    :func:`src.utils.bundle_adjust._integrate_magnus_positions`.
    """
    n = max(8, int(duration_s * fps))
    times = np.linspace(0.0, duration_s, n)
    g_vec = np.array([0.0, 0.0, g])

    if omega is None:
        positions = p0[None, :] + np.outer(times, v0) + 0.5 * np.outer(times**2, g_vec)
        return positions

    # Forward Euler matching the existing Magnus integrator.
    from src.utils.bundle_adjust import _integrate_magnus_positions

    return _integrate_magnus_positions(p0, v0, omega, g_vec, drag_k_over_m, times)


def is_plausible_trajectory(
    p0: np.ndarray,
    v0: np.ndarray,
    *,
    omega: np.ndarray | None,
    duration_s: float,
    fps: float,
    cfg: PlausibilityCfg,
    pitch: PitchDims,
) -> bool:
    """True when the trajectory stays within the physical envelope.

    Samples the trajectory at >= 8 points and checks:
      - |x| <= pitch_length / 2 + margin
      - |y| <= pitch_width / 2 + margin
      - z in [-1.0, z_max_m]
      - per-sample speed <= horizontal_speed_max_m_s + 5.0
    """
    if duration_s <= 0.0:
        return False

    if not (np.all(np.isfinite(p0)) and np.all(np.isfinite(v0))):
        return False
    if omega is not None and not np.all(np.isfinite(omega)):
        return False

    half_len = pitch.length_m / 2.0 + cfg.pitch_margin_m
    half_wid = pitch.width_m / 2.0 + cfg.pitch_margin_m
    speed_cap = cfg.horizontal_speed_max_m_s + 5.0

    positions = _sample_positions(p0, v0, omega, duration_s, fps)

    if np.any(np.abs(positions[:, 0]) > half_len):
        return False
    if np.any(np.abs(positions[:, 1]) > half_wid):
        return False
    if np.any(positions[:, 2] < -1.0) or np.any(positions[:, 2] > cfg.z_max_m):
        return False

    # Per-sample speed (forward difference plus initial v0).
    if positions.shape[0] >= 2:
        dt = duration_s / (positions.shape[0] - 1)
        diffs = np.diff(positions, axis=0) / max(dt, 1e-9)
        speeds = np.linalg.norm(diffs, axis=1)
        if np.any(speeds > speed_cap):
            return False
    if float(np.linalg.norm(v0)) > speed_cap:
        return False

    return True


# ---------------------------------------------------------------------------
# Layer 2 — grounded-run implausibility detection
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GroundPromotionCfg:
    enabled: bool
    min_run_frames: int
    off_pitch_margin_m: float
    max_ground_speed_m_s: float


@dataclass(frozen=True)
class GroundedRun:
    """A contiguous run of grounded frames flagged for refit-as-flight."""

    start: int
    end: int  # inclusive


def _off_pitch_distance(x: float, y: float, pitch: PitchDims) -> float:
    """Point-to-rectangle distance (0 inside, positive outside) for the
    axis-aligned pitch centred at the origin."""
    dx = max(0.0, abs(x) - pitch.length_m / 2.0)
    dy = max(0.0, abs(y) - pitch.width_m / 2.0)
    return float(np.hypot(dx, dy))


def find_implausible_grounded_runs(
    *,
    per_frame_xyz: dict[int, tuple[np.ndarray, float]],
    per_frame_state: dict[int, str],
    fps: float,
    cfg: GroundPromotionCfg,
    pitch: PitchDims,
) -> list[GroundedRun]:
    """Return runs of ``state="grounded"`` frames whose world positions
    imply impossible rolling motion (off-pitch or > max_ground_speed).
    Empty list when disabled or nothing qualifies."""
    if not cfg.enabled:
        return []
    if not per_frame_state:
        return []

    frames = sorted(per_frame_state)
    out: list[GroundedRun] = []
    run_start: int | None = None
    run_max_off_pitch = 0.0
    run_max_speed = 0.0

    def _close_run(end_frame: int) -> None:
        nonlocal run_start, run_max_off_pitch, run_max_speed
        if run_start is None:
            return
        length = end_frame - run_start + 1
        if length >= cfg.min_run_frames and (
            run_max_off_pitch > cfg.off_pitch_margin_m
            or run_max_speed > cfg.max_ground_speed_m_s
        ):
            out.append(GroundedRun(start=run_start, end=end_frame))
        run_start = None
        run_max_off_pitch = 0.0
        run_max_speed = 0.0

    prev_xy: np.ndarray | None = None
    prev_frame: int | None = None
    for fi in frames:
        state = per_frame_state[fi]
        if state != "grounded" or fi not in per_frame_xyz:
            if run_start is not None:
                _close_run(prev_frame if prev_frame is not None else fi - 1)
            prev_xy = None
            prev_frame = None
            continue
        xyz, _conf = per_frame_xyz[fi]
        xy = np.array([xyz[0], xyz[1]])
        if run_start is None:
            run_start = fi
            run_max_off_pitch = _off_pitch_distance(xy[0], xy[1], pitch)
            run_max_speed = 0.0
        else:
            off = _off_pitch_distance(xy[0], xy[1], pitch)
            run_max_off_pitch = max(run_max_off_pitch, off)
            if prev_xy is not None and prev_frame is not None and fi > prev_frame:
                dt = (fi - prev_frame) / fps
                speed = float(np.linalg.norm(xy - prev_xy) / max(dt, 1e-9))
                run_max_speed = max(run_max_speed, speed)
        prev_xy = xy
        prev_frame = fi

    if run_start is not None and prev_frame is not None:
        _close_run(prev_frame)

    return out
