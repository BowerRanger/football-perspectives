"""Synthesised player POV / over-the-shoulder cameras.

Pure math + rig builders. No file I/O — the export stage handles reading
selections and writing CameraTrack JSON. Conventions match the broadcast
camera: ``R`` is world->camera (OpenCV: +Z optical ray into scene, +X
right, +Y down); per-frame ``t`` satisfies camera-centre ``C = -R.T @ t``.
"""

from __future__ import annotations

import math

import numpy as np

WORLD_UP = np.array([0.0, 0.0, 1.0])
WORLD_UP.flags.writeable = False


def _normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(v))
    return v / n if n > eps else v


def intrinsics_from_fov(fov_deg: float, image_size: tuple[int, int]) -> list[list[float]]:
    """3x3 K from a horizontal field of view. Principal point centred."""
    if not (0.0 < fov_deg < 180.0):
        raise ValueError(f"fov_deg must be in (0, 180), got {fov_deg}")
    w, h = int(image_size[0]), int(image_size[1])
    f = (w / 2.0) / math.tan(math.radians(fov_deg) / 2.0)
    return [[f, 0.0, w / 2.0], [0.0, f, h / 2.0], [0.0, 0.0, 1.0]]


def look_at_view(
    center: np.ndarray,
    target: np.ndarray,
    up: np.ndarray = WORLD_UP,
) -> tuple[np.ndarray, np.ndarray]:
    """World->camera (R, t) for a camera at ``center`` looking at ``target``.

    Rows of ``R`` are the camera axes in world coords: right (+X), down
    (+Y), forward (+Z). ``t = -R @ center``.
    """
    center = np.asarray(center, dtype=np.float64).reshape(3)
    target = np.asarray(target, dtype=np.float64).reshape(3)
    if float(np.linalg.norm(target - center)) < 1e-9:
        raise ValueError(
            f"look_at_view: center and target are coincident (center={center}, target={target})"
        )
    z = _normalize(target - center)
    up = np.asarray(up, dtype=np.float64).reshape(3)
    x = np.cross(z, up)
    if float(np.linalg.norm(x)) < 1e-9:
        # Optical axis parallel to up — pick an arbitrary stable basis.
        x = np.cross(z, np.array([0.0, 1.0, 0.0]))
        if float(np.linalg.norm(x)) < 1e-9:
            x = np.cross(z, np.array([1.0, 0.0, 0.0]))
    x = _normalize(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z], axis=0)
    t = -R @ center
    return R, t
