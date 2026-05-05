"""Savitzky-Golay + SLERP smoothing helpers shared across stages."""

from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation, Slerp


def savgol_axis(x: np.ndarray, *, window: int, order: int, axis: int = 0) -> np.ndarray:
    """Apply SavGol along an axis. Window is auto-clamped to len if larger."""
    n = x.shape[axis]
    w = min(window, n - (1 - n % 2))  # nearest odd <= n
    if w < order + 2:
        return x
    return savgol_filter(x, window_length=w, polyorder=order, axis=axis)


def slerp_window(Rs: np.ndarray, *, window: int) -> np.ndarray:
    """SLERP-smooth a sequence of rotations using a sliding centred window."""
    n = Rs.shape[0]
    if n < 3 or window < 3:
        return Rs
    half = window // 2
    out = np.empty_like(Rs)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        if hi - lo < 2:
            out[i] = Rs[i]
            continue
        rots = Rotation.from_matrix(Rs[lo:hi])
        ts = np.linspace(0, 1, hi - lo)
        slerp = Slerp(ts, rots)
        out[i] = slerp([(i - lo) / max(hi - lo - 1, 1)]).as_matrix()[0]
    return out


def ground_snap_z(
    z: np.ndarray, *, velocity_threshold: float = 0.1
) -> np.ndarray:
    """Snap z toward 0 wherever the per-frame velocity is below threshold."""
    out = z.copy()
    if len(z) < 2:
        return out
    v = np.diff(z, prepend=z[0])
    out[np.abs(v) < velocity_threshold] *= 0.5  # half-life snap toward 0
    return out
