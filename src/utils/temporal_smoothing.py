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


def quat_savgol(Rs: np.ndarray, *, window: int, order: int = 2) -> np.ndarray:
    """Savitzky-Golay-smooth a sequence of rotations via continuous quaternions.

    Unlike :func:`slerp_window` (which re-interpolates through the same
    keyframes and is a no-op at the data points), this actually low-pass
    filters the rotation track: convert to quaternions, flip each into the
    previous one's hemisphere (q and -q are the same rotation, but savgol
    needs a continuous signal), SavGol the four components, renormalise, and
    convert back. Smooths single-frame rotation spikes (e.g. solve seams)
    while preserving the slow broadcast pan.
    """
    n = Rs.shape[0]
    if n < 3 or window < 3:
        return Rs
    q = Rotation.from_matrix(Rs).as_quat()  # (n, 4) x, y, z, w
    for i in range(1, n):
        if float(np.dot(q[i], q[i - 1])) < 0.0:
            q[i] = -q[i]
    qs = savgol_axis(q, window=window, order=order, axis=0)
    norms = np.linalg.norm(qs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    qs = qs / norms
    return Rotation.from_quat(qs).as_matrix()


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
