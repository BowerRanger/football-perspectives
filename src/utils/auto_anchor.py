"""Generate camera-stage anchors automatically from PnLCalib keypoints.

The learned model replaces the human clicker: per keyframe, PnLCalib's
keypoint detections become point LandmarkObservations (world coords in our
frame), an Anchor per keyframe, and the existing camera solver does the rest.
"""

from __future__ import annotations

import numpy as np

from src.schemas.anchor import Anchor, LandmarkObservation
from src.utils.pnlcalib_pitch_map import keypoint_world_xyz_ours


def keypoints_to_anchor(
    pixels: dict[int, tuple[float, float]],
    frame: int,
    *,
    min_points: int,
) -> Anchor | None:
    """Build a point-only Anchor from PnLCalib keypoint pixels.

    Names are synthesised (``pnl_kp_<id>``); the solver consumes world_xyz,
    not the name. Returns None if fewer than ``min_points`` known keypoints.
    """
    landmarks = []
    for kp_id, image_xy in pixels.items():
        world = keypoint_world_xyz_ours(kp_id)
        if world is None:
            continue
        landmarks.append(
            LandmarkObservation(
                name=f"pnl_kp_{kp_id}",
                image_xy=(float(image_xy[0]), float(image_xy[1])),
                world_xyz=world,
            )
        )
    if len(landmarks) < min_points:
        return None
    return Anchor(frame=frame, landmarks=tuple(landmarks), lines=())


def compute_keyframes(
    total_frames: int, keyframe_interval: int, max_keyframes: int
) -> list[int]:
    """Sample keyframes from [0, total_frames). Interval is at least
    ``keyframe_interval`` but never yields more than ``max_keyframes``.
    Recovered from the shelved calibration stage."""
    if total_frames <= 0:
        return []
    effective = max(keyframe_interval, 1)
    if max_keyframes > 0:
        interval_from_cap = -(-total_frames // max_keyframes)  # ceil div
        effective = max(effective, interval_from_cap)
    return list(range(0, total_frames, effective))


def _median_absolute_deviation(arr: np.ndarray) -> np.ndarray:
    med = np.median(arr, axis=0)
    return np.median(np.abs(arr - med), axis=0)


def robust_median_position(positions: list[np.ndarray]) -> np.ndarray | None:
    """Median camera world position across keyframes, dropping samples >3 MAD
    from the median on any axis. Recovered from the shelved calibration stage."""
    if not positions:
        return None
    arr = np.asarray(positions, dtype=np.float64)
    if len(arr) <= 2:
        return np.median(arr, axis=0)
    med = np.median(arr, axis=0)
    mad = _median_absolute_deviation(arr)
    mad_clipped = np.where(mad > 1e-6, mad, 1e-6)
    deviation = np.abs(arr - med) / mad_clipped
    mask = np.all(deviation < 3.0, axis=1)
    if not np.any(mask):
        return med
    return np.median(arr[mask], axis=0)


def is_plausible_position(
    position: np.ndarray, bounds: dict[str, tuple[float, float]]
) -> bool:
    """Bounds check on a camera world position (our frame). Adapted from the
    recovered _is_plausible (position component only)."""
    x_lo, x_hi = bounds["x"]
    y_lo, y_hi = bounds["y"]
    z_lo, z_hi = bounds["z"]
    return (
        x_lo <= position[0] <= x_hi
        and y_lo <= position[1] <= y_hi
        and z_lo <= position[2] <= z_hi
    )
