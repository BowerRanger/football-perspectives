"""Rotation-aware temporal operations for human poses, independent of camera filters."""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from src.utils.temporal_smoothing import quat_savgol


def frame_runs(frames: np.ndarray) -> list[tuple[int, int]]:
    frames = np.asarray(frames)
    if not len(frames):
        return []
    cuts = np.r_[0, np.flatnonzero(np.diff(frames) != 1) + 1, len(frames)]
    return [(int(a), int(b)) for a, b in zip(cuts[:-1], cuts[1:])]


def smooth_rotations(matrices, *, window=7, order=2, robust=False,
                     fps=30.0, max_speed_deg_s=None):
    """Filter continuous quaternions; optionally reject isolated orientation flips.

    Robust rejection uses the local rotation medoid and requires a majority of
    mutually nearby samples. A sustained turn is retained. Speed projection is
    a configurable reconstruction guard, not a claimed anatomical limit.
    Call separately for each contiguous run.
    """
    out = np.asarray(matrices, dtype=float).copy()
    n = len(out)
    if n < 2 or window <= 1:
        return out
    if robust:
        snapshot = Rotation.from_matrix(out).as_quat()
        radius = max(2, window // 2)
        for i in range(n):
            q = snapshot[max(0, i-radius):min(n, i+radius+1)]
            distances = 2*np.arccos(np.clip(np.abs(q @ q.T), 0, 1))
            medoid = int(np.argmin(distances.sum(axis=1)))
            close = distances[medoid] < np.deg2rad(30)
            dist = 2*np.arccos(np.clip(abs(q[medoid] @ snapshot[i]), 0, 1))
            if close.sum() > len(q)/2 and dist > np.deg2rad(60):
                out[i] = Rotation.from_quat(q[medoid]).as_matrix()
    out = quat_savgol(out, window=window, order=order)
    if max_speed_deg_s is not None:
        step = np.deg2rad(float(max_speed_deg_s)) / float(fps)
        # Project both directions to avoid always placing the correction after
        # a seam. A final forward pass guarantees the consecutive-frame bound.
        for reverse in (False, True, False):
            ids = range(n-2, -1, -1) if reverse else range(1, n)
            for i in ids:
                prev = i+1 if reverse else i-1
                delta = Rotation.from_matrix(out[prev].T @ out[i]).as_rotvec()
                norm = np.linalg.norm(delta)
                if norm > step:
                    out[i] = out[prev] @ Rotation.from_rotvec(delta*(step/norm)).as_matrix()
    return out


def smooth_pose(thetas, *, window=9, order=2):
    out = np.asarray(thetas, dtype=float).copy()
    if not len(out) or window <= 1:
        return out
    for j in range(out.shape[1]):
        r = Rotation.from_rotvec(out[:, j]).as_matrix()
        out[:, j] = Rotation.from_matrix(smooth_rotations(r, window=window, order=order)).as_rotvec()
    return out


def interpolate_pose(times, thetas, target_times):
    """Shortest-path joint interpolation; never average absolute rotvecs."""
    out = np.empty((len(target_times),) + thetas.shape[1:], dtype=float)
    for j in range(thetas.shape[1]):
        if len(times) == 1:
            out[:, j] = thetas[0, j]
        else:
            out[:, j] = Slerp(times, Rotation.from_rotvec(thetas[:, j]))(target_times).as_rotvec()
    return out
